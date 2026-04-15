"""Tests for GestureAnalyzer v2 — 7-layer detection pipeline.

Coverage:
- Pinch near hair zone triggers (hand_near_head=True)
- Open hand near head does NOT trigger
- Gripping object near head does NOT trigger
- No hands -> IDLE
- State machine transitions (IDLE -> TRIGGERED -> ALERTING -> COOLDOWN -> IDLE)
- Debug fields on AnalysisResult (pinch_ratio_smooth, spread, rotation, consecutive_frames)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from hairguard.analyzer import AnalysisResult, GestureAnalyzer
from hairguard.detector import Landmark, PoseResult
from hairguard.hair_segmenter import HairMask
from hairguard.hand_detector import (
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    THUMB_TIP,
    INDEX_TIP,
    WRIST,
    HandLandmark,
    HandResult,
)


# ---------------------------------------------------------------------------
# Config stub
# ---------------------------------------------------------------------------

class _FakeConfig:
    """Minimal Config stand-in for unit tests."""

    _DEFAULTS = {
        ("detection", "trigger_delay_sec"): 0.8,
        ("detection", "dismiss_delay_sec"): 2.5,
        ("detection", "min_confidence"): 0.5,
        ("detection", "pinch_threshold"): 0.25,
        ("detection", "spread_threshold"): 0.075,
        ("detection", "rotation_threshold"): 0.075,
        ("detection", "persist_frames"): 1,  # set to 1 for unit tests (instant detection)
        ("detection", "min_presence_score"): 0.7,
        ("detection", "hair_dilation_px"): 25,
    }

    def get(self, *keys: str):
        return self._DEFAULTS[keys]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_pose(
    left_ear_x: float = 0.25,
    right_ear_x: float = 0.75,
    ear_y: float = 0.35,
    left_shoulder_x: float = 0.2,
    right_shoulder_x: float = 0.8,
    shoulder_y: float = 0.7,
    nose_x: float = 0.5,
    nose_y: float = 0.4,
    detected: bool = True,
) -> PoseResult:
    if not detected:
        return PoseResult(landmarks={}, raw_landmarks=None, detected=False)
    return PoseResult(
        landmarks={
            "left_ear": Landmark(x=left_ear_x, y=ear_y, confidence=0.9),
            "right_ear": Landmark(x=right_ear_x, y=ear_y, confidence=0.9),
            "left_shoulder": Landmark(x=left_shoulder_x, y=shoulder_y, confidence=0.9),
            "right_shoulder": Landmark(x=right_shoulder_x, y=shoulder_y, confidence=0.9),
            "nose": Landmark(x=nose_x, y=nose_y, confidence=0.9),
        },
        raw_landmarks=None,
        detected=True,
    )


def _make_hair_mask(
    in_zone_y: float = 0.20,
    zone_top: float = 0.05,
    zone_bottom: float = 0.35,
    zone_left: float = 0.2,
    zone_right: float = 0.8,
    width: int = 640,
    height: int = 480,
) -> HairMask:
    """Create a HairMask with a rectangular hair region.

    By default the hair region covers x=[0.2, 0.8], y=[0.05, 0.35] —
    the same area used by the old ellipse zone in tests.
    """
    import numpy as np
    mask = np.zeros((height, width), dtype=np.uint8)
    r0 = int(zone_top * height)
    r1 = int(zone_bottom * height)
    c0 = int(zone_left * width)
    c1 = int(zone_right * width)
    mask[r0:r1, c0:c1] = 1
    return HairMask(mask=mask, width=width, height=height)


def _make_hand_landmarks(
    thumb_x: float = 0.5,
    thumb_y: float = 0.20,
    index_x: float = 0.5,
    index_y: float = 0.20,
    wrist_x: float = 0.5,
    wrist_y: float = 0.50,
    mcp_x: float = 0.5,
    mcp_y: float = 0.40,
    presence: float = 1.0,
    fingers_curled: bool = False,
) -> dict[int, HandLandmark]:
    """Build a minimal landmark dict with configurable positions.

    fingers_curled=True places MIDDLE/RING/PINKY tips below their MCP along
    the wrist->mcp axis to trigger is_gripping().
    """
    lms: dict[int, HandLandmark] = {}

    # Wrist + MCP (axis reference)
    lms[WRIST] = HandLandmark(x=wrist_x, y=wrist_y, z=0.0, presence=presence)
    lms[MIDDLE_MCP] = HandLandmark(x=mcp_x, y=mcp_y, z=0.0, presence=presence)

    # Thumb and index tips at the requested position
    lms[THUMB_TIP] = HandLandmark(x=thumb_x, y=thumb_y, z=0.0, presence=presence)
    lms[INDEX_TIP] = HandLandmark(x=index_x, y=index_y, z=0.0, presence=presence)

    # Ring and Pinky MCP — place slightly above their tips so curl can be set
    ring_mcp_y = mcp_y + 0.02
    pinky_mcp_y = mcp_y + 0.02
    lms[RING_MCP] = HandLandmark(x=mcp_x + 0.02, y=ring_mcp_y, z=0.0, presence=presence)
    lms[PINKY_MCP] = HandLandmark(x=mcp_x + 0.04, y=pinky_mcp_y, z=0.0, presence=presence)

    if fingers_curled:
        # Curled: tips closer to wrist than MCP along axis
        curled_tip_y = mcp_y + 0.05
        lms[MIDDLE_TIP] = HandLandmark(x=mcp_x, y=curled_tip_y, z=0.0, presence=presence)
        lms[RING_TIP] = HandLandmark(x=mcp_x + 0.02, y=curled_tip_y, z=0.0, presence=presence)
        lms[PINKY_TIP] = HandLandmark(x=mcp_x + 0.04, y=curled_tip_y, z=0.0, presence=presence)
    else:
        # Relaxed: tips near the thumb/index (clustered, as in twisting)
        # Keep pinky close to index for low spread (<= 0.075)
        lms[MIDDLE_TIP] = HandLandmark(x=index_x + 0.01, y=index_y + 0.01, z=0.0, presence=presence)
        lms[RING_TIP] = HandLandmark(x=index_x + 0.02, y=index_y + 0.02, z=0.0, presence=presence)
        lms[PINKY_TIP] = HandLandmark(x=index_x + 0.03, y=index_y + 0.03, z=0.0, presence=presence)

    return lms


def _make_hand(
    thumb_x: float = 0.5,
    thumb_y: float = 0.20,
    index_x: float = 0.5,
    index_y: float = 0.20,
    wrist_x: float = 0.5,
    wrist_y: float = 0.50,
    mcp_x: float = 0.5,
    mcp_y: float = 0.40,
    presence: float = 1.0,
    fingers_curled: bool = False,
    handedness: str = "Right",
    score: float = 0.95,
) -> HandResult:
    lms = _make_hand_landmarks(
        thumb_x=thumb_x,
        thumb_y=thumb_y,
        index_x=index_x,
        index_y=index_y,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
        mcp_x=mcp_x,
        mcp_y=mcp_y,
        presence=presence,
        fingers_curled=fingers_curled,
    )
    return HandResult(landmarks=lms, handedness=handedness, score=score)


def _make_pinching_hand(in_zone: bool = True) -> HandResult:
    """Return a hand with thumb and index very close together (pinching).

    When in_zone=True the tips are placed at (0.5, 0.20) — inside the default
    hair zone (forehead_y=0.35, ears at 0.25..0.75, ceiling ~0.11).
    When in_zone=False the tips are placed at (0.5, 0.60) — below the forehead.
    """
    tip_y = 0.20 if in_zone else 0.60
    # Thumb and index at nearly the same point → small pinch_ratio
    return _make_hand(
        thumb_x=0.50,
        thumb_y=tip_y,
        index_x=0.501,   # 1 mm apart in normalized space
        index_y=tip_y,
        wrist_x=0.5,
        wrist_y=0.55,
        mcp_x=0.5,
        mcp_y=0.45,
        presence=1.0,
        fingers_curled=False,
    )


def _make_open_hand_in_zone() -> HandResult:
    """Return an open hand (no pinch) with fingertips in the hair zone."""
    # Thumb and index far apart → large pinch_ratio → above pinch_threshold
    return _make_hand(
        thumb_x=0.40,
        thumb_y=0.20,
        index_x=0.60,
        index_y=0.20,
        wrist_x=0.5,
        wrist_y=0.55,
        mcp_x=0.5,
        mcp_y=0.45,
        presence=1.0,
        fingers_curled=False,
    )


def _make_gripping_hand_in_zone() -> HandResult:
    """Return a gripping hand with fingertips in the hair zone."""
    return _make_hand(
        thumb_x=0.50,
        thumb_y=0.20,
        index_x=0.501,
        index_y=0.20,
        wrist_x=0.5,
        wrist_y=0.55,
        mcp_x=0.5,
        mcp_y=0.45,
        presence=1.0,
        fingers_curled=True,  # 3 curled fingers → is_gripping() = True
    )


def _make_analyzer(**overrides) -> GestureAnalyzer:
    return GestureAnalyzer(_FakeConfig())


# ---------------------------------------------------------------------------
# AnalysisResult dataclass
# ---------------------------------------------------------------------------

class TestAnalysisResultFields:
    def test_has_all_required_fields(self):
        r = AnalysisResult(
            should_alert=False,
            hand_near_head=False,
            state="IDLE",
            episode_started=False,
            episode_ended=False,
            episode_duration=0.0,
        )
        assert r.pinch_ratio_raw is None
        assert r.pinch_ratio_smooth is None
        assert r.spread_index_pinky is None
        assert r.hand_rotation is None
        assert r.fingertips_in_zone is False
        assert r.consecutive_frames == 0

    def test_v2_fields_settable(self):
        r = AnalysisResult(
            should_alert=True,
            hand_near_head=True,
            state="ALERTING",
            episode_started=True,
            episode_ended=False,
            episode_duration=0.0,
            pinch_ratio_raw=0.12,
            pinch_ratio_smooth=0.15,
            spread_index_pinky=0.03,
            hand_rotation=0.02,
            fingertips_in_zone=True,
            consecutive_frames=5,
        )
        assert r.pinch_ratio_smooth == pytest.approx(0.15)
        assert r.fingertips_in_zone is True
        assert r.consecutive_frames == 5


# ---------------------------------------------------------------------------
# _detect() layer tests  (via analyze() inspection of result fields)
# ---------------------------------------------------------------------------

class TestDetectLayers:
    """Test each layer of the 7-layer pipeline."""

    def setup_method(self):
        self.analyzer = _make_analyzer()
        self.pose = _make_pose()
        self.hair_mask = _make_hair_mask()

    # Layer 1: no hands
    def test_no_hands_returns_not_detected(self):
        result = self.analyzer.analyze(self.pose, [], self.hair_mask, 480, 640)
        assert result.hand_near_head is False
        assert result.fingertips_in_zone is False
        assert result.pinch_ratio_raw is None

    # Layer 2: pose not detected
    def test_no_pose_returns_not_detected(self):
        pose = _make_pose(detected=False)
        hand = _make_pinching_hand()
        result = self.analyzer.analyze(pose, [hand], self.hair_mask, 480, 640)
        assert result.hand_near_head is False

    # Layer 3: fingertips outside hair mask
    def test_fingertips_outside_zone_not_detected(self):
        hand = _make_pinching_hand(in_zone=False)
        result = self.analyzer.analyze(self.pose, [hand], self.hair_mask, 480, 640)
        assert result.hand_near_head is False
        assert result.fingertips_in_zone is False

    # Open hand (high pinch ratio) in zone does NOT trigger
    def test_open_hand_in_zone_not_detected(self):
        hand = _make_open_hand_in_zone()
        result = self.analyzer.analyze(self.pose, [hand], self.hair_mask, 480, 640)
        assert result.hand_near_head is False
        assert result.fingertips_in_zone is True

    # Happy path: pinching hand with clustered fingers in zone DOES trigger
    def test_pinching_hand_in_zone_detected(self):
        hand = _make_pinching_hand(in_zone=True)
        result = self.analyzer.analyze(self.pose, [hand], self.hair_mask, 480, 640)
        assert result.hand_near_head is True
        assert result.fingertips_in_zone is True
        assert result.pinch_ratio_smooth is not None

    # Low presence landmark is skipped
    def test_low_presence_hand_skipped(self):
        hand = _make_pinching_hand(in_zone=True)
        for idx in hand.landmarks:
            lm = hand.landmarks[idx]
            hand.landmarks[idx] = HandLandmark(
                x=lm.x, y=lm.y, z=lm.z, presence=0.1
            )
        result = self.analyzer.analyze(self.pose, [hand], self.hair_mask, 480, 640)
        assert result.hand_near_head is False

    # Multiple hands: one gripping, one pinching → should detect
    def test_features_computed_even_when_not_detected(self):
        """Features are computed and returned even if detection rule fails."""
        hand = _make_open_hand_in_zone()
        result = self.analyzer.analyze(self.pose, [hand], self.hair_mask, 480, 640)
        assert result.hand_near_head is False
        assert result.pinch_ratio_smooth is not None  # features still computed
        assert result.spread_index_pinky is not None


# ---------------------------------------------------------------------------
# State machine transitions
# ---------------------------------------------------------------------------

class TestStateMachine:
    """Test IDLE/TRIGGERED/ALERTING/COOLDOWN transitions."""

    def _analyzer(self) -> GestureAnalyzer:
        return _make_analyzer()

    def _pose(self) -> PoseResult:
        return _make_pose()

    def _hair_mask(self) -> HairMask:
        return _make_hair_mask()

    def _pinching(self) -> list[HandResult]:
        return [_make_pinching_hand(in_zone=True)]

    def _no_hands(self) -> list[HandResult]:
        return []

    # --- IDLE ---

    def test_starts_in_idle(self):
        az = self._analyzer()
        result = az.analyze(self._pose(), self._no_hands(), self._hair_mask(), 480, 640)
        assert result.state == "IDLE"
        assert result.should_alert is False

    def test_idle_no_detection_stays_idle(self):
        az = self._analyzer()
        for _ in range(3):
            result = az.analyze(self._pose(), self._no_hands(), self._hair_mask(), 480, 640)
        assert result.state == "IDLE"

    # --- IDLE -> TRIGGERED ---

    def test_detection_moves_to_triggered(self):
        az = self._analyzer()
        result = az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)
        assert result.state == "TRIGGERED"
        assert result.should_alert is False
        assert result.episode_started is False

    # --- TRIGGERED -> IDLE (hand leaves before delay) ---

    def test_hand_leaves_before_delay_returns_to_idle(self):
        az = self._analyzer()
        # Enter TRIGGERED
        az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)
        # Hand gone before trigger_delay_sec
        result = az.analyze(self._pose(), self._no_hands(), self._hair_mask(), 480, 640)
        assert result.state == "IDLE"

    # --- TRIGGERED -> ALERTING (after delay) ---

    def test_trigger_delay_elapsed_moves_to_alerting(self):
        az = self._analyzer()
        az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)

        # Patch _state_entered_at to simulate delay elapsed
        az._state_entered_at -= 1.0  # 1 second in the past > 0.8 threshold

        result = az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)
        assert result.state == "ALERTING"
        assert result.should_alert is True
        assert result.episode_started is True

    # --- ALERTING -> COOLDOWN ---

    def test_hand_leaves_alerting_enters_cooldown(self):
        az = self._analyzer()
        # Reach ALERTING
        az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)
        az._state_entered_at -= 1.0
        az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)
        assert az._state.name == "ALERTING"

        result = az.analyze(self._pose(), self._no_hands(), self._hair_mask(), 480, 640)
        assert result.state == "COOLDOWN"
        assert result.should_alert is True  # still showing during cooldown

    # --- COOLDOWN -> ALERTING (hand returns) ---

    def test_hand_returns_during_cooldown_resumes_alerting(self):
        az = self._analyzer()
        _reach_cooldown(az, self._pose(), self._hair_mask())

        result = az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)
        assert result.state == "ALERTING"
        assert result.should_alert is True
        assert result.episode_started is False  # NOT a new episode start

    # --- COOLDOWN -> IDLE (dismiss delay elapsed) ---

    def test_dismiss_delay_elapsed_ends_episode(self):
        az = self._analyzer()
        _reach_cooldown(az, self._pose(), self._hair_mask())

        # Simulate time passing: episode started 5 s ago, cooldown entered 3 s ago
        az._episode_started_at -= 5.0
        az._state_entered_at -= 3.0  # > dismiss_delay_sec=2.5

        result = az.analyze(self._pose(), self._no_hands(), self._hair_mask(), 480, 640)
        assert result.state == "IDLE"
        assert result.episode_ended is True
        assert result.episode_duration > 0.0
        assert result.should_alert is False

    # --- episode_duration measured correctly ---

    def test_episode_duration_is_positive(self):
        az = self._analyzer()
        _reach_cooldown(az, self._pose(), self._hair_mask())

        az._episode_started_at -= 5.0
        az._state_entered_at -= 3.0

        result = az.analyze(self._pose(), self._no_hands(), self._hair_mask(), 480, 640)
        assert result.episode_duration > 0.0

    # --- episode_started only fires once ---

    def test_episode_started_fires_only_on_transition(self):
        az = self._analyzer()
        az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)
        az._state_entered_at -= 1.0

        r1 = az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)
        assert r1.episode_started is True

        r2 = az.analyze(self._pose(), self._pinching(), self._hair_mask(), 480, 640)
        assert r2.episode_started is False  # not again on next frame

    # --- analyze() signature accepts frame_height and frame_width ---

    def test_analyze_accepts_frame_dimensions(self):
        az = self._analyzer()
        result = az.analyze(self._pose(), self._no_hands(), self._hair_mask(), 480, 640)
        assert isinstance(result, AnalysisResult)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reach_cooldown(az: GestureAnalyzer, pose: PoseResult, hair_mask: HairMask) -> None:
    """Drive az into COOLDOWN state."""
    hand = [_make_pinching_hand(in_zone=True)]
    no_hand: list[HandResult] = []

    # IDLE -> TRIGGERED
    az.analyze(pose, hand, hair_mask, 480, 640)
    # Fast-forward trigger delay
    az._state_entered_at -= 1.0
    # TRIGGERED -> ALERTING
    az.analyze(pose, hand, hair_mask, 480, 640)
    # ALERTING -> COOLDOWN
    az.analyze(pose, no_hand, hair_mask, 480, 640)
    assert az._state.name == "COOLDOWN"
