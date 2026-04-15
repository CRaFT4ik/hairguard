"""Tests for hairguard.hand_detector — HandLandmark, HandResult logic.

These tests exercise the pure-Python geometry methods only; they never
instantiate HandDetector (which would download a model and require a camera).
"""
from __future__ import annotations

import math
import pytest

from hairguard.hand_detector import (
    MIDDLE_MCP,
    HandLandmark,
    HandResult,
    INDEX_TIP,
    THUMB_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
    RING_MCP,
    PINKY_MCP,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lm(x: float, y: float, z: float = 0.0, presence: float = 1.0) -> HandLandmark:
    return HandLandmark(x=x, y=y, z=z, presence=presence)


def _minimal_landmarks() -> dict[int, HandLandmark]:
    """Return a landmark dict that satisfies hand_scale() and is_pinching/gripping checks.

    Layout (all z=0):
      wrist [0]       at (0, 0)
      middle_mcp [9]  at (0, 1)   → hand_scale = 1.0
      thumb_tip [4]   at (0, 0.5) (mid-point — for neutral)
      index_tip [8]   at (0, 0.5) (same as thumb_tip — neutral)
      middle_tip [12] at (0, 2)   (far from wrist, extended)
      ring_mcp [13]   at (0, 0.9)
      ring_tip [16]   at (0, 2)   (extended)
      pinky_mcp [17]  at (0, 0.8)
      pinky_tip [20]  at (0, 2)   (extended)
    """
    lms: dict[int, HandLandmark] = {}
    lms[0] = _lm(0.0, 0.0)          # wrist
    lms[9] = _lm(0.0, 1.0)          # middle_mcp  → hand_scale axis
    lms[4] = _lm(0.0, 0.5)          # thumb_tip
    lms[8] = _lm(0.0, 0.5)          # index_tip
    lms[12] = _lm(0.0, 2.0)         # middle_tip (extended)
    lms[13] = _lm(0.0, 0.9)         # ring_mcp
    lms[16] = _lm(0.0, 2.0)         # ring_tip (extended)
    lms[17] = _lm(0.0, 0.8)         # pinky_mcp
    lms[20] = _lm(0.0, 2.0)         # pinky_tip (extended)
    return lms


def _result(lms: dict[int, HandLandmark] | None = None) -> HandResult:
    return HandResult(
        landmarks=lms if lms is not None else _minimal_landmarks(),
        handedness="Right",
        score=0.95,
    )


# ---------------------------------------------------------------------------
# HandResult creation
# ---------------------------------------------------------------------------

class TestHandResultCreation:
    def test_fields_stored(self):
        lms = _minimal_landmarks()
        r = HandResult(landmarks=lms, handedness="Left", score=0.8)
        assert r.handedness == "Left"
        assert r.score == pytest.approx(0.8)
        assert r.landmarks is lms

    def test_thumb_tip_accessor(self):
        r = _result()
        assert r.thumb_tip() is r.landmarks[THUMB_TIP]

    def test_index_tip_accessor(self):
        r = _result()
        assert r.index_tip() is r.landmarks[INDEX_TIP]


# ---------------------------------------------------------------------------
# hand_scale
# ---------------------------------------------------------------------------

class TestHandScale:
    def test_unit_hand(self):
        """wrist at origin, middle_mcp at (0,1) → scale == 1.0"""
        r = _result()
        assert r.hand_scale() == pytest.approx(1.0)

    def test_scaled_hand(self):
        lms = _minimal_landmarks()
        # Move middle_mcp to (0, 2) → scale == 2.0
        lms[9] = _lm(0.0, 2.0)
        r = _result(lms)
        assert r.hand_scale() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# pinch_ratio
# ---------------------------------------------------------------------------

class TestPinchRatio:
    def test_zero_distance_gives_zero_ratio(self):
        """thumb_tip == index_tip → pinch_ratio == 0"""
        lms = _minimal_landmarks()
        lms[4] = _lm(0.0, 0.5)
        lms[8] = _lm(0.0, 0.5)
        r = _result(lms)
        assert r.pinch_ratio() == pytest.approx(0.0)

    def test_half_hand_scale(self):
        """dist(thumb, index) == 0.5 * hand_scale == 1.0 → ratio == 0.5"""
        lms = _minimal_landmarks()
        lms[4] = _lm(0.0, 0.75)
        lms[8] = _lm(0.0, 0.25)
        r = _result(lms)
        # hand_scale = 1.0, dist = 0.5
        assert r.pinch_ratio() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# is_pinching
# ---------------------------------------------------------------------------

class TestIsPinching:
    def test_pinching_true_when_close(self):
        """thumb and index almost touching → ratio ≈ 0 < 0.4"""
        lms = _minimal_landmarks()
        lms[4] = _lm(0.0, 0.501)
        lms[8] = _lm(0.0, 0.500)
        r = _result(lms)
        assert r.is_pinching() is True

    def test_pinching_false_when_far(self):
        """thumb and index far apart → ratio > 0.4"""
        lms = _minimal_landmarks()
        lms[4] = _lm(0.0, 0.0)   # near wrist
        lms[8] = _lm(0.0, 1.0)   # at middle_mcp height
        r = _result(lms)
        # dist == 1.0, hand_scale == 1.0 → ratio == 1.0 > 0.4
        assert r.is_pinching() is False

    def test_pinching_false_when_low_presence(self):
        """Low presence landmarks should not register as pinching."""
        lms = _minimal_landmarks()
        lms[4] = _lm(0.0, 0.501, presence=0.1)
        lms[8] = _lm(0.0, 0.500, presence=0.1)
        r = _result(lms)
        assert r.is_pinching() is False

    def test_custom_threshold(self):
        """is_pinching respects a custom threshold."""
        lms = _minimal_landmarks()
        # ratio will be 0.3 (dist=0.3, scale=1.0)
        lms[4] = _lm(0.0, 0.65)
        lms[8] = _lm(0.0, 0.35)
        r = _result(lms)
        assert r.is_pinching(threshold=0.5) is True
        assert r.is_pinching(threshold=0.2) is False


# ---------------------------------------------------------------------------
# is_gripping
# ---------------------------------------------------------------------------

class TestIsGripping:
    def _curled_lms(self) -> dict[int, HandLandmark]:
        """Fingers curled: tip projection < MCP projection along wrist→middle_mcp axis.

        Wrist at (0,0), middle_mcp at (0,1). Axis direction = (0,1).
        A point's projection onto this axis = its y-coordinate.

        For curled: tip.y < mcp.y  (tip is closer to wrist than MCP)
        """
        lms = _minimal_landmarks()
        # middle finger curled: tip at y=0.5 < mcp at y=1.0
        lms[9]  = _lm(0.0, 1.0)   # middle_mcp
        lms[12] = _lm(0.0, 0.5)   # middle_tip curled (y < mcp.y)
        # ring finger curled
        lms[13] = _lm(0.0, 0.9)   # ring_mcp
        lms[16] = _lm(0.0, 0.4)   # ring_tip curled
        # pinky curled
        lms[17] = _lm(0.0, 0.8)   # pinky_mcp
        lms[20] = _lm(0.0, 0.3)   # pinky_tip curled
        return lms

    def _extended_lms(self) -> dict[int, HandLandmark]:
        """Fingers extended: tip projection > MCP projection."""
        lms = _minimal_landmarks()
        # tips are further from wrist than MCPs → extended
        lms[12] = _lm(0.0, 2.0)   # middle_tip extended
        lms[16] = _lm(0.0, 2.0)   # ring_tip extended
        lms[20] = _lm(0.0, 2.0)   # pinky_tip extended
        return lms

    def test_gripping_true_all_curled(self):
        lms = self._curled_lms()
        r = _result(lms)
        assert r.is_gripping() is True

    def test_gripping_false_all_extended(self):
        lms = self._extended_lms()
        r = _result(lms)
        assert r.is_gripping() is False

    def test_gripping_threshold_two_fingers(self):
        """With curl_threshold=2, two curled fingers should still grip."""
        lms = _minimal_landmarks()
        lms[9]  = _lm(0.0, 1.0)
        lms[12] = _lm(0.0, 0.5)   # middle curled
        lms[13] = _lm(0.0, 0.9)
        lms[16] = _lm(0.0, 0.4)   # ring curled
        lms[17] = _lm(0.0, 0.8)
        lms[20] = _lm(0.0, 2.0)   # pinky extended
        r = _result(lms)
        assert r.is_gripping(curl_threshold=2) is True
        assert r.is_gripping(curl_threshold=3) is False

    def test_gripping_false_one_curled(self):
        """Default threshold=2: only one curled finger should not grip."""
        lms = _minimal_landmarks()
        lms[9]  = _lm(0.0, 1.0)
        lms[12] = _lm(0.0, 0.5)   # middle curled
        lms[13] = _lm(0.0, 0.9)
        lms[16] = _lm(0.0, 2.0)   # ring extended
        lms[17] = _lm(0.0, 0.8)
        lms[20] = _lm(0.0, 2.0)   # pinky extended
        r = _result(lms)
        assert r.is_gripping() is False
