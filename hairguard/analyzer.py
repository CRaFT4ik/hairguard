"""hairguard.analyzer — GestureAnalyzer v2.

7-layer detection pipeline with pinch/grip/zone gating.

State machine: IDLE -> TRIGGERED -> ALERTING -> COOLDOWN -> IDLE
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from hairguard.config import Config
from hairguard.detector import PoseResult
from hairguard.face_detector import FaceResult
from hairguard.hand_detector import HandResult
from hairguard.head_zone import HairZone


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
    # v2 debug info
    pinch_distance: Optional[float] = None
    fingertips_in_zone: bool = False
    is_gripping: bool = False


class GestureAnalyzer:
    """Detects hair-touching gestures via a 7-layer pipeline and state machine.

    All timing logic lives here. The app layer only needs to map
    ``AnalysisResult.should_alert`` to overlay show/hide.
    """

    def __init__(self, config: Config) -> None:
        det = "detection"
        self._trigger_delay_sec: float = config.get(det, "trigger_delay_sec")
        self._dismiss_delay_sec: float = config.get(det, "dismiss_delay_sec")
        self._min_confidence: float = config.get(det, "min_confidence")
        self._pinch_threshold: float = config.get(det, "pinch_threshold")
        self._min_presence: float = config.get(det, "min_presence_score")
        self._hair_zone_offset: float = config.get(det, "hair_zone_offset")
        self._grip_curl_threshold: int = int(config.get(det, "grip_curl_threshold"))

        self._state = _State.IDLE
        self._state_entered_at: float = 0.0
        self._episode_started_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        pose_result: PoseResult,
        hand_results: list[HandResult],
        face_result: FaceResult,
        frame_height: int,
        frame_width: int,
    ) -> AnalysisResult:
        """Process one frame and advance the state machine.

        Parameters
        ----------
        pose_result:
            Output of PoseDetector.process().
        hand_results:
            List of HandResult from HandDetector.detect(); may be empty.
        face_result:
            Output of FaceDetector.detect().
        frame_height:
            Frame height in pixels (kept for API symmetry; not used in v2).
        frame_width:
            Frame width in pixels (kept for API symmetry; not used in v2).

        Returns
        -------
        AnalysisResult
            Snapshot of the current analysis state with v2 debug fields.
        """
        now = time.monotonic()
        detect_info = self._detect(pose_result, hand_results, face_result)

        hand_near_head: bool = detect_info["detected"]
        episode_started = False
        episode_ended = False
        episode_duration = 0.0

        # ---- state machine transitions ----

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
                # Do NOT reset _episode_started_at; episode continues
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
            pinch_distance=detect_info["pinch_distance"],
            fingertips_in_zone=detect_info["fingertips_in_zone"],
            is_gripping=detect_info["is_gripping"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: _State, now: float) -> None:
        self._state = new_state
        self._state_entered_at = now

    def _detect(
        self,
        pose_result: PoseResult,
        hand_results: list[HandResult],
        face_result: FaceResult,
    ) -> dict:
        """7-layer detection pipeline.

        Returns a dict with keys:
            detected          bool
            pinch_distance    float | None
            fingertips_in_zone bool
            is_gripping       bool
        """
        result: dict = {
            "detected": False,
            "pinch_distance": None,
            "fingertips_in_zone": False,
            "is_gripping": False,
        }

        # Layer 1: Hand present?
        if not hand_results:
            return result

        # Layer 2: Head located?
        if not pose_result.detected:
            return result

        # Layer 3: Compute hair zone
        zone = HairZone(face_result, pose_result, self._hair_zone_offset)

        # For each hand:
        for hand in hand_results:
            thumb = hand.thumb_tip()
            index = hand.index_tip()

            # Filter by presence (Layer 3 guard)
            if (
                thumb.presence < self._min_presence
                or index.presence < self._min_presence
            ):
                continue

            # Layer 4: Fingertips in hair zone?
            if not (zone.contains(thumb.x, thumb.y) or zone.contains(index.x, index.y)):
                continue
            result["fingertips_in_zone"] = True

            # Layer 5: Pinch detected? (normalized by hand size)
            pinch_r = hand.pinch_ratio()
            result["pinch_distance"] = pinch_r
            if pinch_r >= self._pinch_threshold:
                continue

            # Layer 6: Not gripping?
            gripping = hand.is_gripping(self._grip_curl_threshold)
            result["is_gripping"] = gripping
            if gripping:
                continue

            # Layer 7: All checks passed — detected
            result["detected"] = True
            return result

        return result
