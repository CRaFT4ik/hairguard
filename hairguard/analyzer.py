from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from hairguard.config import Config
from hairguard.detector import Landmark, PoseResult


class _State(Enum):
    IDLE = auto()       # no hand near head
    TRIGGERED = auto()  # hand detected, waiting for trigger_delay_sec
    ALERTING = auto()   # alert active, hand still at head
    COOLDOWN = auto()   # hand left head, waiting dismiss_delay_sec


@dataclass
class AnalysisResult:
    should_alert: bool        # True while in ALERTING or COOLDOWN (show overlay)
    hand_near_head: bool      # raw detection result this frame
    state: str                # current state name
    episode_started: bool     # one-shot: True on TRIGGERED -> ALERTING transition
    episode_ended: bool       # one-shot: True on COOLDOWN -> IDLE transition
    episode_duration: float   # seconds; set when episode_ended is True


def _euclidean(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class GestureAnalyzer:
    """
    Detects hand-near-head gestures and drives the debounce state machine.

    All timing logic lives here.  The app layer only needs to map
    ``AnalysisResult.should_alert`` to overlay show/hide with no
    additional delays.
    """

    def __init__(self, config: Config) -> None:
        self._distance_threshold: float = config.get("detection", "distance_threshold")
        self._fallback_frame_percent: float = config.get("detection", "fallback_frame_percent")
        self._trigger_delay_sec: float = config.get("detection", "trigger_delay_sec")
        self._dismiss_delay_sec: float = config.get("detection", "dismiss_delay_sec")
        self._min_confidence: float = config.get("detection", "min_confidence")

        self._state = _State.IDLE
        self._state_entered_at: float = 0.0   # monotonic time we entered current state
        self._episode_started_at: float = 0.0  # when ALERTING began (for duration calc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, pose_result: PoseResult, frame_height: int, frame_width: int = 0) -> AnalysisResult:
        """
        Process one frame's PoseResult and advance the state machine.

        Parameters
        ----------
        pose_result:
            Output of ``PoseDetector.process()``.
        frame_height:
            Height of the source frame in pixels (used for fallback threshold).

        Returns
        -------
        AnalysisResult
            Snapshot of the current analysis state.
        """
        now = time.monotonic()
        hand_near_head = self._detect(pose_result, frame_height, frame_width or frame_height)

        episode_started = False
        episode_ended = False
        episode_duration = 0.0

        # ---- state machine transitions ----

        if self._state is _State.IDLE:
            if hand_near_head:
                self._transition(_State.TRIGGERED, now)

        elif self._state is _State.TRIGGERED:
            if not hand_near_head:
                # Hand left before trigger delay elapsed — go back to idle
                self._transition(_State.IDLE, now)
            elif now - self._state_entered_at >= self._trigger_delay_sec:
                # Hand held long enough — start alerting
                self._transition(_State.ALERTING, now)
                self._episode_started_at = now
                episode_started = True

        elif self._state is _State.ALERTING:
            if not hand_near_head:
                self._transition(_State.COOLDOWN, now)

        elif self._state is _State.COOLDOWN:
            if hand_near_head:
                # Hand returned during cooldown — resume alerting
                self._transition(_State.ALERTING, now)
                # Do NOT reset _episode_started_at; episode continues
            elif now - self._state_entered_at >= self._dismiss_delay_sec:
                # Dismiss delay elapsed — episode over
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
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: _State, now: float) -> None:
        self._state = new_state
        self._state_entered_at = now

    def _detect(self, pose_result: PoseResult, frame_height: int, frame_width: int) -> bool:
        """
        Return True if any wrist is close enough to any ear.

        Uses shoulder-normalized distance by default; falls back to absolute
        frame-height ratio when either shoulder confidence is below threshold.
        """
        if not pose_result.detected:
            return False

        lm = pose_result.landmarks

        wrists = [lm.get("left_wrist"), lm.get("right_wrist")]
        # Head reference points: ears + nose (nose catches top/back of head better)
        head_points = [lm.get("left_ear"), lm.get("right_ear"), lm.get("nose")]
        left_shoulder = lm.get("left_shoulder")
        right_shoulder = lm.get("right_shoulder")

        # Filter by confidence
        wrists = [w for w in wrists if w is not None and w.confidence >= self._min_confidence]
        head_points = [h for h in head_points if h is not None and h.confidence >= self._min_confidence]

        if not wrists or not head_points:
            return False

        # Decide which distance metric to use
        use_fallback = (
            left_shoulder is None
            or right_shoulder is None
            or left_shoulder.confidence < self._min_confidence
            or right_shoulder.confidence < self._min_confidence
        )

        if use_fallback:
            return self._detect_fallback(wrists, head_points, frame_height, frame_width)
        else:
            return self._detect_normalized(wrists, head_points, left_shoulder, right_shoulder)

    def _detect_normalized(
        self,
        wrists: list[Landmark],
        ears: list[Landmark],
        left_shoulder: Landmark,
        right_shoulder: Landmark,
    ) -> bool:
        """Shoulder-width-normalized wrist-to-ear distance check."""
        shoulder_dist = _euclidean(left_shoulder, right_shoulder)
        if shoulder_dist < 1e-6:
            # Degenerate — shoulders collapsed; fall back silently
            return False

        for wrist in wrists:
            for ear in ears:
                ratio = _euclidean(wrist, ear) / shoulder_dist
                if ratio < self._distance_threshold:
                    return True
        return False

    def _detect_fallback(
        self,
        wrists: list[Landmark],
        ears: list[Landmark],
        frame_height: int,
        frame_width: int,
    ) -> bool:
        """Absolute distance as fraction of frame height (shoulder occlusion fallback)."""
        scale_y = float(frame_height) if frame_height > 0 else 1.0
        scale_x = float(frame_width) if frame_width > 0 else scale_y
        threshold_px = self._fallback_frame_percent * scale_y

        for wrist in wrists:
            for ear in ears:
                dx = (wrist.x - ear.x) * scale_x
                dy = (wrist.y - ear.y) * scale_y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < threshold_px:
                    return True
        return False
