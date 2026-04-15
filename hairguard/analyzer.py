"""hairguard.analyzer — GestureAnalyzer v2.1 (data-driven).

Detection pipeline based on empirical analysis of twisting vs combing data.
Uses 3 features with sliding windows + persistence filter:
  1. Smoothed pinch ratio (20-frame rolling mean) — primary signal
  2. Index-to-pinky spread — finger clustering signal
  3. Hand rotation rate — static vs sweeping motion

State machine: IDLE -> TRIGGERED -> ALERTING -> COOLDOWN -> IDLE
"""
from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hairguard.config import Config
from hairguard.detector import PoseResult
from hairguard.hair_segmenter import HairMask
from hairguard.hand_detector import HandResult, INDEX_TIP, PINKY_TIP


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
    fingertips_in_zone: bool = False
    consecutive_frames: int = 0


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

    def __init__(self, config: Config) -> None:
        det = "detection"
        self._trigger_delay_sec: float = config.get(det, "trigger_delay_sec")
        self._dismiss_delay_sec: float = config.get(det, "dismiss_delay_sec")
        self._min_presence: float = config.get(det, "min_presence_score")

        # Data-driven thresholds — frontal (default)
        self._pinch_threshold: float = config.get(det, "pinch_threshold")
        self._spread_threshold: float = config.get(det, "spread_threshold")
        self._rotation_threshold: float = config.get(det, "rotation_threshold")
        # Relaxed thresholds for side camera (yaw_ratio < 0.3)
        self._side_spread_threshold: float = 0.10
        self._side_rotation_threshold: float = 0.10
        self._persist_frames: int = int(config.get(det, "persist_frames"))
        self._hair_dilation_px: int = int(config.get(det, "hair_dilation_px"))
        self._shoulder_distance: float = 0.0  # updated per frame

        # Ring buffers for rolling features
        self._pinch_buf: collections.deque = collections.deque(maxlen=self._WINDOW)
        self._rotation_buf: collections.deque = collections.deque(maxlen=self._WINDOW)
        self._last_hand_angle: Optional[float] = None
        self._consecutive_positive: int = 0

        # State machine
        self._state = _State.IDLE
        self._state_entered_at: float = 0.0
        self._episode_started_at: float = 0.0

    def analyze(
        self,
        pose_result: PoseResult,
        hand_results: list[HandResult],
        hair_mask: HairMask,
        frame_height: int,
        frame_width: int,
    ) -> AnalysisResult:
        now = time.monotonic()
        detection = self._detect(pose_result, hand_results, hair_mask)

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
            fingertips_in_zone=detection.get("in_zone", False),
            consecutive_frames=self._consecutive_positive,
        )

    def _transition(self, new_state: _State, now: float) -> None:
        self._state = new_state
        self._state_entered_at = now

    def _detect(
        self,
        pose_result: PoseResult,
        hand_results: list[HandResult],
        hair_mask: HairMask,
    ) -> dict:
        result: dict = {
            "detected": False,
            "pinch_raw": None,
            "pinch_smooth": None,
            "spread": None,
            "rotation": None,
            "in_zone": False,
        }

        # Layer 1: Hand present?
        if not hand_results:
            self._consecutive_positive = 0
            return result

        # Layer 2: Head located?
        if not pose_result.detected:
            self._consecutive_positive = 0
            return result

        # Compute camera angle: yaw_ratio = ear_distance / shoulder_distance
        # < 0.3 = side view, >= 0.3 = frontal
        lm = pose_result.landmarks
        left_ear = lm.get("left_ear")
        right_ear = lm.get("right_ear")
        left_sh = lm.get("left_shoulder")
        right_sh = lm.get("right_shoulder")
        ear_dist = 0.0
        sh_dist = 0.0
        if left_ear and right_ear and left_ear.confidence >= 0.5 and right_ear.confidence >= 0.5:
            ear_dist = math.sqrt((right_ear.x - left_ear.x)**2 + (right_ear.y - left_ear.y)**2)
        if left_sh and right_sh and left_sh.confidence >= 0.5 and right_sh.confidence >= 0.5:
            sh_dist = math.sqrt((right_sh.x - left_sh.x)**2 + (right_sh.y - left_sh.y)**2)
        yaw_ratio = ear_dist / sh_dist if sh_dist > 0.05 else 0.45

        # Adaptive thresholds based on camera angle
        is_side_view = yaw_ratio < 0.3
        spread_thresh = self._side_spread_threshold if is_side_view else self._spread_threshold
        rotation_thresh = self._side_rotation_threshold if is_side_view else self._rotation_threshold

        # Expand hair mask edges for better coverage (side-camera views need more)
        refined = hair_mask.dilate(self._hair_dilation_px)

        # Check each hand
        for hand in hand_results:
            thumb = hand.thumb_tip()
            index = hand.index_tip()

            # Presence guard
            thumb_p = thumb.presence if thumb.presence is not None else 0.0
            index_p = index.presence if index.presence is not None else 0.0
            if thumb_p < self._min_presence or index_p < self._min_presence:
                continue

            # Layer 3: Fingertips in refined hair mask (dilated + face removed)?
            if not (refined.contains(thumb.x, thumb.y) or refined.contains(index.x, index.y)):
                continue
            result["in_zone"] = True

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

            # --- Decision rule (adaptive thresholds by camera angle) ---
            # Frontal: pinch<=0.25, spread<=0.075, rotation<=0.075
            # Side:    pinch<=0.25, spread<=0.10,  rotation<=0.10
            frame_positive = (
                pinch_smooth <= self._pinch_threshold
                and spread <= spread_thresh
                and rotation <= rotation_thresh
            )

            if frame_positive:
                self._consecutive_positive += 1
            else:
                self._consecutive_positive = 0

            # Persistence filter
            if self._consecutive_positive >= self._persist_frames:
                result["detected"] = True
            return result  # return after first hand with features computed

        # No hand matched
        self._consecutive_positive = 0
        return result
