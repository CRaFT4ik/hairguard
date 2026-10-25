"""hairguard.analyzer — GestureAnalyzer v2.1 (data-driven) and v3 (model).

Detection pipeline based on empirical analysis of twisting vs combing data.
Uses 3 features with sliding windows + persistence filter:
  1. Smoothed pinch ratio (20-frame rolling mean) — primary signal
  2. Index-to-pinky spread — finger clustering signal
  3. Hand rotation rate — static vs sweeping motion

State machine: IDLE -> TRIGGERED -> ALERTING -> COOLDOWN -> IDLE

How the trained model composes with that
----------------------------------------
Given a ``twist_model``, the model's probability for the last 20 in-zone
frames *replaces the geometric rule as the per-frame verdict* and nothing
else changes. The zone gate still decides when a hand counts as being at the
head, the state machine still demands a sustained verdict before it alerts
and still holds the alert through the dismissal delay, and episodes are still
accounted the same way.

That specific shape was chosen over the two obvious alternatives. Replacing
the state machine with the model throws away hysteresis, which is the
cheapest precision there is: raising the model's threshold costs about ten
points of recall per five points of false-positive rate, while requiring the
verdict to persist costs almost nothing and kills exactly the isolated blip
a single window can produce. Running both detectors and alerting on either
would raise recall by reintroducing the rule's false positives, at an
operating point nobody has measured — the cross-validated numbers describe
the model alone.

What does change is *when* the model gets asked, and this is the part that
had to be measured rather than reasoned about. The app interrupts a
behaviour, so how long it waits is most of its value. Asking only once a full
twenty-frame window exists means waiting about 2.8 s at the 6.75 fps this
loop achieves — and the first version of this integration did exactly that,
which replayed through the state machine on held-out days scored **0.0%
recall on a three-second episode**: the window completes on the final frame,
the trigger delay never elapses, and nothing ever appears. A threefold cut in
missed episodes bought with a detector that cannot react inside a short one
is not a trade worth making.

``hairlab.truncation`` measured the alternative on real clips, cross-
validated by capture day. Truncating a clip to its first *k* frames and
repeat-padding the rest is precisely what the live window holds *k* frames
into a visit, and the degradation is gentle: recall 78.0% ± 8.2 at three
frames against 86.7% ± 6.8 at twenty, with the false-alarm rate on a resting
hand slightly *better* at three (3.2% ± 1.9 against 3.6% ± 3.1). Zero-padding,
by contrast, destroys the model outright — see ``hairguard.live_features``,
which is where the padding lives and why.

So ``detection.model.min_frames`` (3) decides when the model is first asked,
and ``detection.model.trigger_delay_sec`` (0.3 s) supplies the hysteresis on
top. Measured end to end through this state machine on the held-out days: the
rule alerts on 37.7% of twisting clips after a median 1.20 s with a 10.3%
false-alarm rate on resting hands; the model alerts on 80.7% after a median
0.72 s with a 6.9% rate. More than twice the episodes caught, sooner, and
with fewer false alarms — rather than the earlier 0.0%.

Asking early is also strictly safe in one specific sense worth knowing: the
window keeps sliding and still becomes full, so an early verdict only ever
*adds* an opportunity to fire. It cannot delay one.
"""
from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from hairguard.config import Config
from hairguard.detector import PoseResult
from hairguard.face_detector import FaceResult
from hairguard.hand_detector import HandResult, INDEX_TIP, PINKY_TIP
from hairguard.head_zone import HairZone, HairZoneSmoother

# Consecutive scoring failures tolerated before the model is put aside for
# the rest of the session. A detector that scores nothing reports nothing,
# which is strictly worse than the rule it was meant to replace.
_MODEL_FAILURE_LIMIT = 10

# How long one window may take. Measured cost is well under a millisecond
# against a 100 ms tick, so this is thirty times the real figure — it is not
# a performance target but a tripwire. The detection tick also drives the
# tkinter event loop, the overlay and the alert sound, and the model is a
# passenger on it: if it ever becomes capable of holding that up, it stops
# being used rather than being allowed to stutter the alert.
_MODEL_BUDGET_SEC = 0.030
_MODEL_SLOW_LIMIT = 10

# Used only if the configuration cannot answer. Short, because the window
# behind each verdict is already two seconds long.
_DEFAULT_MODEL_TRIGGER_DELAY_SEC = 0.3


class _State(Enum):
    IDLE = auto()
    TRIGGERED = auto()
    ALERTING = auto()
    COOLDOWN = auto()


@dataclass
class AnalysisResult:
    should_alert: bool
    hand_near_head: bool
    state: str
    episode_started: bool
    episode_ended: bool
    episode_duration: float
    # v2.1 debug info
    pinch_ratio_raw: Optional[float] = None
    pinch_ratio_smooth: Optional[float] = None
    spread_index_pinky: Optional[float] = None
    hand_rotation: Optional[float] = None
    combined_score: Optional[float] = None
    fingertips_in_zone: bool = False
    consecutive_frames: int = 0
    # v3: the trained model's probability for the window ending at this
    # frame, or None when there is no model or no full window yet. Kept
    # separate from the rule's fields so the debug view can show both and a
    # human can see where the two detectors disagree.
    model_score: Optional[float] = None
    # Which detector produced ``hand_near_head``: "model" or "rule". Reported
    # rather than inferred, because the two differ by thirty points of recall
    # and the app can demote itself mid-session after repeated failures.
    detector: str = "rule"
    # Index into the hand_results list of the hand found inside the hair zone.
    # Dataset clips are cropped around it, so "a hand is in the zone" alone is
    # not enough information for the collector.
    in_zone_hand_index: Optional[int] = None


class GestureAnalyzer:
    """Data-driven hair-twisting detector.

    Features (per frame, O(1) with ring buffers):
      1. pinch_mean_20f: 20-frame rolling mean of pinch_ratio
      2. spread_8_20: distance(index_tip, pinky_tip) — finger clustering
      3. hand_rotation_20f: rolling mean of hand angle delta — motion detector

    Detection rule:
      frame_positive = (pinch_mean_20f <= pinch_threshold)
                   AND (spread_8_20 <= spread_threshold)
                   AND (hand_rotation_20f <= rotation_threshold)
                   AND (fingertips in hair zone)

    Persistence: require N consecutive positive frames before triggering.
    """

    _WINDOW = 20  # rolling window size (frames)

    def __init__(
        self,
        config: Config,
        *,
        twist_model: Optional[Any] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        det = "detection"
        self._clock = clock
        self._rule_trigger_delay_sec: float = config.get(det, "trigger_delay_sec")
        self._trigger_delay_sec: float = self._rule_trigger_delay_sec
        self._dismiss_delay_sec: float = config.get(det, "dismiss_delay_sec")
        self._min_presence: float = config.get(det, "min_presence_score")
        self._hair_zone_offset: float = config.get(det, "hair_zone_offset")

        # Data-driven thresholds (from empirical analysis)
        self._combined_threshold: float = config.get(det, "combined_threshold")
        self._rotation_threshold: float = config.get(det, "rotation_threshold")
        self._persist_frames: int = int(config.get(det, "persist_frames"))

        # Ring buffers for rolling features
        self._pinch_buf: collections.deque = collections.deque(maxlen=self._WINDOW)
        self._rotation_buf: collections.deque = collections.deque(maxlen=self._WINDOW)
        self._last_hand_angle: Optional[float] = None
        self._consecutive_positive: int = 0

        # --- The trained model, when there is one ---
        # Model settings are read only in this branch, so a caller that has
        # never heard of the model (and a config stub that answers only the
        # rule's keys) keeps working unchanged.
        self._twist = twist_model
        self._feature_window = None
        self._model_failures = 0
        self._model_failure_logged = False
        self._model_slow = 0
        self._model_slow_logged = False
        self._model_worst_sec = 0.0
        if twist_model is not None:
            from hairguard.live_features import FeatureWindow

            self._feature_window = FeatureWindow(
                seq_len=int(twist_model.seq_len),
                min_frames=self._model_min_frames(config, int(twist_model.seq_len)),
            )
            self._trigger_delay_sec = self._model_trigger_delay(config)

        # Temporal smoothing for hair zone
        self._zone_smoother = HairZoneSmoother(alpha=0.3)

        # State machine
        self._state = _State.IDLE
        self._state_entered_at: float = 0.0
        self._episode_started_at: float = 0.0

    @staticmethod
    def _model_min_frames(config: Config, seq_len: int) -> int:
        """In-zone frames before the model is worth asking.

        Falls back to the whole window when the setting cannot be read — the
        conservative direction, because a full window is the operating point
        that was measured first and most.
        """
        try:
            value = int(config.get("detection", "model", "min_frames"))
        except Exception:
            return seq_len
        return max(1, min(value, seq_len))

    @staticmethod
    def _model_trigger_delay(config: Config) -> float:
        try:
            value = float(config.get("detection", "model", "trigger_delay_sec"))
        except Exception:
            return _DEFAULT_MODEL_TRIGGER_DELAY_SEC
        return value if value >= 0.0 else _DEFAULT_MODEL_TRIGGER_DELAY_SEC

    @property
    def detector_name(self) -> str:
        """"model" or "rule" — which one is deciding right now."""
        return "model" if self._twist is not None else "rule"

    @property
    def model_worst_sec(self) -> float:
        """Slowest window the model has scored, in seconds.

        Reported rather than merely enforced: ``_MODEL_BUDGET_SEC`` is a
        tripwire, and knowing how much headroom is actually left is the only
        way to tell a safe margin from a lucky one.
        """
        return self._model_worst_sec

    def _drop_model(self, reason: str) -> None:
        """Hand detection back to the rule for the rest of the session.

        Loudly, always: a detector that changed underneath the user must not
        do so in silence, or the next thing they notice is the app behaving
        differently for no visible reason.
        """
        print(f"[analyzer] {reason} — переключаюсь на старое правило до перезапуска.")
        self._twist = None
        self._feature_window = None
        self._trigger_delay_sec = self._rule_trigger_delay_sec

    def analyze(
        self,
        pose_result: PoseResult,
        hand_results: list[HandResult],
        face_result: FaceResult,
        frame_height: int,
        frame_width: int,
    ) -> AnalysisResult:
        now = self._clock()
        # Captured before detection runs: a model that demotes itself to the
        # rule on this very tick must still report that this verdict came
        # from the model path.
        detector = self.detector_name
        detection = self._detect(pose_result, hand_results, face_result, now)

        hand_near_head = detection["detected"]

        episode_started = False
        episode_ended = False
        episode_duration = 0.0

        # State machine (unchanged)
        if self._state is _State.IDLE:
            if hand_near_head:
                self._transition(_State.TRIGGERED, now)

        elif self._state is _State.TRIGGERED:
            if not hand_near_head:
                self._transition(_State.IDLE, now)
            elif now - self._state_entered_at >= self._trigger_delay_sec:
                self._transition(_State.ALERTING, now)
                self._episode_started_at = now
                episode_started = True

        elif self._state is _State.ALERTING:
            if not hand_near_head:
                self._transition(_State.COOLDOWN, now)

        elif self._state is _State.COOLDOWN:
            if hand_near_head:
                self._transition(_State.ALERTING, now)
            elif now - self._state_entered_at >= self._dismiss_delay_sec:
                episode_duration = now - self._episode_started_at
                episode_ended = True
                self._transition(_State.IDLE, now)

        should_alert = self._state in (_State.ALERTING, _State.COOLDOWN)

        return AnalysisResult(
            should_alert=should_alert,
            hand_near_head=hand_near_head,
            state=self._state.name,
            episode_started=episode_started,
            episode_ended=episode_ended,
            episode_duration=episode_duration,
            pinch_ratio_raw=detection.get("pinch_raw"),
            pinch_ratio_smooth=detection.get("pinch_smooth"),
            spread_index_pinky=detection.get("spread"),
            hand_rotation=detection.get("rotation"),
            combined_score=detection.get("combined_score"),
            fingertips_in_zone=detection.get("in_zone", False),
            consecutive_frames=self._consecutive_positive,
            in_zone_hand_index=detection.get("hand_index"),
            model_score=detection.get("model_score"),
            detector=detector,
        )

    def _transition(self, new_state: _State, now: float) -> None:
        self._state = new_state
        self._state_entered_at = now

    def _reset_rolling(self) -> None:
        """Clear rolling feature buffers.

        Called when the hand is absent from the hair zone so stale low-pinch
        history from a prior visit does not bleed into the next one and cause
        a fast false positive. FPS-independent: both debug (30 fps) and
        prod (10 fps) pipelines behave the same regardless of how many
        frames of history the window would otherwise hold.
        """
        self._pinch_buf.clear()
        self._rotation_buf.clear()
        self._last_hand_angle = None
        self._consecutive_positive = 0
        # The model's window goes with them. Stored clips are each one
        # unbroken visit to the head, so a window stitched across a gap is
        # not a sample of what the model was fitted on.
        if self._feature_window is not None:
            self._feature_window.reset()

    def _detect(
        self,
        pose_result: PoseResult,
        hand_results: list[HandResult],
        face_result: FaceResult,
        now: Optional[float] = None,
    ) -> dict:
        if now is None:
            now = self._clock()
        result: dict = {
            "detected": False,
            "pinch_raw": None,
            "pinch_smooth": None,
            "spread": None,
            "rotation": None,
            "in_zone": False,
            "hand_index": None,
            "model_score": None,
        }

        # Layer 1: Hand present?
        if not hand_results:
            self._reset_rolling()
            return result

        # Layer 2: Head located?
        if not pose_result.detected:
            self._reset_rolling()
            return result

        # Layer 3: Compute hair zone
        zone = HairZone(face_result, pose_result, self._hair_zone_offset)
        zone = self._zone_smoother.smooth(zone)

        # Check each hand
        for hand_index, hand in enumerate(hand_results):
            thumb = hand.thumb_tip()
            index = hand.index_tip()

            # Presence guard
            thumb_p = thumb.presence if thumb.presence is not None else 0.0
            index_p = index.presence if index.presence is not None else 0.0
            if thumb_p < self._min_presence or index_p < self._min_presence:
                continue

            # Layer 4: Fingertips in hair zone?
            if not (zone.contains(thumb.x, thumb.y) or zone.contains(index.x, index.y)):
                continue
            result["in_zone"] = True
            result["hand_index"] = hand_index

            # --- Compute features ---

            # Feature 1: Pinch ratio (smoothed)
            pinch_raw = hand.pinch_ratio()
            result["pinch_raw"] = pinch_raw
            self._pinch_buf.append(pinch_raw)
            pinch_smooth = sum(self._pinch_buf) / len(self._pinch_buf)
            result["pinch_smooth"] = pinch_smooth

            # Feature 2: Index-to-pinky spread
            pinky = hand.landmarks.get(PINKY_TIP)
            idx_lm = hand.landmarks.get(INDEX_TIP)
            if pinky and idx_lm:
                spread = math.sqrt((idx_lm.x - pinky.x) ** 2 + (idx_lm.y - pinky.y) ** 2)
            else:
                spread = 1.0  # no data = assume spread
            result["spread"] = spread

            # Feature 3: Hand rotation rate
            wrist = hand.landmarks.get(0)
            mid_mcp = hand.landmarks.get(9)
            if wrist and mid_mcp:
                angle = math.atan2(mid_mcp.y - wrist.y, mid_mcp.x - wrist.x)
                if self._last_hand_angle is not None:
                    delta = abs(angle - self._last_hand_angle)
                    if delta > math.pi:
                        delta = 2 * math.pi - delta
                    self._rotation_buf.append(delta)
                self._last_hand_angle = angle
            rotation = sum(self._rotation_buf) / len(self._rotation_buf) if self._rotation_buf else 0.0
            result["rotation"] = rotation

            # --- Decision rule: combined score ---
            # Weighted combination of pinch and spread separates twisting from combing
            # better than independent thresholds (distributions overlap on each alone).
            # Score = pinch_smooth * 2 + spread. Lower = more likely twisting.
            # Typical: twisting ~0.3, combing ~0.8, threshold 0.55
            combined_score = pinch_smooth * 2 + spread
            result["combined_score"] = combined_score

            # --- The verdict: the model when there is one, the rule otherwise ---
            # The geometric features above are computed either way. They cost
            # microseconds and they are what the debug view shows, so a human
            # comparing the two detectors can see where they disagree.
            model = self._twist
            if model is not None:
                # Held locally: a scoring failure may put the model aside for
                # good, and this tick still has to finish on the model it
                # started with.
                score = self._score_window(hand, pose_result, now)
                result["model_score"] = score
                frame_positive = score is not None and score >= model.threshold
                # One window is already twenty frames of evidence, so no
                # frame-count filter on top of it; the state machine's
                # trigger delay supplies the hysteresis.
                persist_required = 1
            else:
                frame_positive = (
                    combined_score <= self._combined_threshold
                    and rotation <= self._rotation_threshold
                )
                persist_required = self._persist_frames

            if frame_positive:
                self._consecutive_positive += 1
            else:
                self._consecutive_positive = 0

            # Persistence filter
            if self._consecutive_positive >= persist_required:
                result["detected"] = True
            return result  # return after first hand with features computed

        # No hand matched the zone this frame — clear rolling history so
        # a later re-entry starts from a clean slate.
        self._reset_rolling()
        return result

    # ------------------------------------------------------------------
    # The model path
    # ------------------------------------------------------------------

    def _score_window(
        self, hand: HandResult, pose_result: PoseResult, now: float
    ) -> Optional[float]:
        """The model's probability for the window ending at this frame.

        None means "no verdict": either the window is not full yet, or
        scoring it failed. A failure is a skipped window and never an
        exception out of here — the detection tick also owns the tkinter
        event loop, the overlay and the alert sound, and none of those may be
        taken down by the passenger.

        Scoring that keeps failing is a different problem from scoring that
        failed once. A model which raises on every window reports nothing at
        all, which is worse than the rule it replaced, so after
        ``_MODEL_FAILURE_LIMIT`` consecutive failures it is put aside for the
        rest of the session and the rule takes over — loudly, because a
        detector that changed underneath the user must not do so in silence.
        """
        from hairguard.live_features import frame_record

        started = time.perf_counter()
        try:
            window = self._feature_window.push(frame_record(hand, pose_result, now))
            if window is None:
                return None
            score = float(self._twist.score(window))
        except Exception as exc:
            self._model_failures += 1
            if not self._model_failure_logged:
                # Once per session, not once per tick: at 10 fps the second
                # option buries everything the user could act on.
                print(f"[analyzer] Окно пропущено, модель не дала оценку: {exc}")
                self._model_failure_logged = True
            if self._model_failures >= _MODEL_FAILURE_LIMIT:
                self._drop_model(
                    f"Модель не смогла оценить {self._model_failures} окон подряд"
                )
            return None

        elapsed = time.perf_counter() - started
        self._model_worst_sec = max(self._model_worst_sec, elapsed)
        if elapsed > _MODEL_BUDGET_SEC:
            self._model_slow += 1
            if not self._model_slow_logged:
                print(
                    f"[analyzer] Оценка окна заняла {1000 * elapsed:.0f} мс — "
                    "это уже заметно для цикла обнаружения."
                )
                self._model_slow_logged = True
            if self._model_slow >= _MODEL_SLOW_LIMIT:
                self._drop_model(
                    f"Модель не укладывается в {1000 * _MODEL_BUDGET_SEC:.0f} мс "
                    f"{self._model_slow} окон подряд и задерживает оповещение"
                )
        else:
            self._model_slow = 0

        self._model_failures = 0
        return score
