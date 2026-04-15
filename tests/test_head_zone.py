"""Tests for hairguard.head_zone — TDD suite."""
from __future__ import annotations

import pytest

from hairguard.face_detector import FaceLandmark, FaceResult, FOREHEAD_INDICES
from hairguard.detector import Landmark, PoseResult
from hairguard.head_zone import HairZone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face(forehead_y: float = 0.35, detected: bool = True) -> FaceResult:
    """Build a FaceResult with a flat forehead line at forehead_y spanning x=0.2..0.8."""
    if not detected:
        return FaceResult(landmarks={}, detected=False)
    landmarks = {
        FOREHEAD_INDICES[0]: FaceLandmark(x=0.2, y=forehead_y),
        FOREHEAD_INDICES[1]: FaceLandmark(x=0.8, y=forehead_y),
    }
    return FaceResult(landmarks=landmarks, detected=True)


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
    """Build a PoseResult with configurable landmark positions."""
    if not detected:
        return PoseResult(landmarks={}, raw_landmarks=None, detected=False)
    landmarks = {
        "left_ear": Landmark(x=left_ear_x, y=ear_y, confidence=0.9),
        "right_ear": Landmark(x=right_ear_x, y=ear_y, confidence=0.9),
        "left_shoulder": Landmark(x=left_shoulder_x, y=shoulder_y, confidence=0.9),
        "right_shoulder": Landmark(x=right_shoulder_x, y=shoulder_y, confidence=0.9),
        "nose": Landmark(x=nose_x, y=nose_y, confidence=0.9),
    }
    return PoseResult(landmarks=landmarks, raw_landmarks=None, detected=True)


def _default_zone(**kwargs) -> HairZone:
    """Return a HairZone with sensible defaults; override with kwargs."""
    face = kwargs.pop("face", _make_face())
    pose = kwargs.pop("pose", _make_pose())
    return HairZone(face=face, pose=pose, **kwargs)


# ---------------------------------------------------------------------------
# Construction & bounds
# ---------------------------------------------------------------------------

class TestHairZoneBounds:
    def test_x_bounds_with_margin(self):
        """x_min / x_max include 40% ear margin (hair extends beyond ears)."""
        pose = _make_pose(left_ear_x=0.25, right_ear_x=0.75)
        zone = HairZone(face=_make_face(), pose=pose)
        ear_width = abs(0.75 - 0.25)
        margin = ear_width * 0.40
        x_min, x_max, _, _ = zone.bounds
        assert x_min == pytest.approx(0.25 - margin)
        assert x_max == pytest.approx(0.75 + margin)

    def test_y_forehead_avg(self):
        """bounds y_forehead_avg accounts for hairline shift above eyebrow line.

        With forehead_y=0.35, nose_y=0.40:
        shifted = 0.35 - (0.40 - 0.35) * 0.4 = 0.33
        """
        zone = _default_zone(face=_make_face(forehead_y=0.35))
        _, _, _, y_forehead_avg = zone.bounds
        assert y_forehead_avg == pytest.approx(0.33, abs=0.01)

    def test_ceiling_not_negative(self):
        """y_ceiling is always >= 0.0 even with a large offset."""
        zone = _default_zone(hair_zone_offset=999.0)
        _, _, y_ceiling, _ = zone.bounds
        assert y_ceiling == pytest.approx(0.0)

    def test_ceiling_default_offset(self):
        """Ceiling uses shifted forehead_y_mean with 1.5x multiplier for head tilt."""
        face = _make_face(forehead_y=0.35)
        pose = _make_pose(left_shoulder_x=0.2, right_shoulder_x=0.8)
        zone = HairZone(face=face, pose=pose, hair_zone_offset=0.4)
        shoulder_width = abs(0.8 - 0.2)
        shifted_forehead = 0.33  # 0.35 - (0.40 - 0.35)*0.55
        expected_ceiling = max(0.0, shifted_forehead - shoulder_width * 0.4 * 1.5)
        _, _, y_ceiling, _ = zone.bounds
        assert y_ceiling == pytest.approx(expected_ceiling, abs=0.02)


# ---------------------------------------------------------------------------
# contains — happy path
# ---------------------------------------------------------------------------

class TestContainsInZone:
    def test_point_in_hair_zone_returns_true(self):
        """A point clearly above forehead and between ears is in the zone."""
        # forehead at y=0.35; ears at x=0.25..0.75; ceiling near 0.11
        # point at (0.5, 0.20) should be above forehead (y < 0.35) and
        # below ceiling, between ears.
        face = _make_face(forehead_y=0.35)
        pose = _make_pose(left_ear_x=0.25, right_ear_x=0.75,
                          left_shoulder_x=0.2, right_shoulder_x=0.8)
        zone = HairZone(face=face, pose=pose, hair_zone_offset=0.4)
        # ceiling = max(0, 0.35 - 0.6*0.4) = max(0, 0.35 - 0.24) = 0.11
        # point (0.5, 0.20): above forehead (0.20 < 0.35), below ceiling (0.20 > 0.11)
        assert zone.contains(0.5, 0.20) is True

    def test_point_below_forehead_returns_false(self):
        """A point below the forehead line is not in the zone."""
        zone = _default_zone(face=_make_face(forehead_y=0.35))
        # y=0.50 > forehead_y=0.35 → below forehead
        assert zone.contains(0.5, 0.50) is False

    def test_point_outside_left_ear_returns_false(self):
        """A point far to the left of the ear bounds is not in the zone."""
        pose = _make_pose(left_ear_x=0.35, right_ear_x=0.65)
        zone = HairZone(face=_make_face(), pose=pose)
        # ear_width=0.30, margin=0.30*0.40=0.12, x_min=0.35-0.12=0.23
        assert zone.contains(0.05, 0.20) is False

    def test_point_outside_right_ear_returns_false(self):
        """A point far to the right of the ear bounds is not in the zone."""
        pose = _make_pose(left_ear_x=0.35, right_ear_x=0.65)
        zone = HairZone(face=_make_face(), pose=pose)
        # ear_width=0.30, margin=0.30*0.40=0.12, x_max=0.65+0.12=0.77
        assert zone.contains(0.95, 0.20) is False

    def test_point_above_ceiling_returns_false(self):
        """A point above the ceiling (too high in image) is not in the zone.

        With wider ceiling (1.5x multiplier), use a small shoulder width
        so ceiling doesn't hit 0.0.
        """
        face = _make_face(forehead_y=0.40)
        pose = _make_pose(left_shoulder_x=0.35, right_shoulder_x=0.65)
        zone = HairZone(face=face, pose=pose, hair_zone_offset=0.3)
        # Ceiling should be > 0 but point at y=-0.1 (impossible) would be above
        _, _, y_ceiling, _ = zone.bounds
        assert y_ceiling > 0.0  # ceiling exists
        assert zone.contains(0.5, 0.0) is False  # y=0.0 is above ceiling

    def test_on_forehead_line_not_in_zone(self):
        """A point exactly on the forehead line is NOT in the zone (not above)."""
        zone = _default_zone(face=_make_face(forehead_y=0.35))
        assert zone.contains(0.5, 0.35) is False


# ---------------------------------------------------------------------------
# Tilted head — forehead adapts
# ---------------------------------------------------------------------------

class TestTiltedHead:
    def test_tilted_head_zone_adapts(self):
        """With a higher forehead (y=0.40), the zone should still work correctly."""
        face = _make_face(forehead_y=0.40)
        pose = _make_pose(left_ear_x=0.25, right_ear_x=0.75,
                          left_shoulder_x=0.2, right_shoulder_x=0.8)
        zone = HairZone(face=face, pose=pose, hair_zone_offset=0.4)
        # ceiling = max(0, 0.40 - 0.6*0.4) = max(0, 0.16) = 0.16
        # point (0.5, 0.25): above forehead (0.25 < 0.40), below ceiling (0.25 > 0.16)
        assert zone.contains(0.5, 0.25) is True
        # point (0.5, 0.45): below forehead → not in zone
        assert zone.contains(0.5, 0.45) is False

    def test_tilted_head_y_forehead_avg_updates(self):
        """bounds reflects the updated forehead_y when head is tilted."""
        zone = _default_zone(face=_make_face(forehead_y=0.40))
        _, _, _, y_forehead_avg = zone.bounds
        assert y_forehead_avg == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Fallback — no face detected
# ---------------------------------------------------------------------------

class TestNoFaceFallback:
    def test_fallback_to_nose_when_no_face(self):
        """When face is not detected, nose.y is used as forehead proxy."""
        face = _make_face(detected=False)
        # nose at y=0.40; point at (0.5, 0.25) should be above nose → in zone
        pose = _make_pose(nose_y=0.40, left_ear_x=0.25, right_ear_x=0.75,
                          left_shoulder_x=0.2, right_shoulder_x=0.8)
        zone = HairZone(face=face, pose=pose, hair_zone_offset=0.4)
        # ceiling = max(0, 0.40 - 0.6*0.4) = 0.16
        # point (0.5, 0.25): above nose-proxy (0.25 < 0.40) and below ceiling
        assert zone.contains(0.5, 0.25) is True

    def test_fallback_below_nose_returns_false(self):
        """Point below nose proxy is not in zone."""
        face = _make_face(detected=False)
        pose = _make_pose(nose_y=0.40)
        zone = HairZone(face=face, pose=pose)
        assert zone.contains(0.5, 0.55) is False

    def test_fallback_y_forehead_avg_is_nose_y(self):
        """bounds y_forehead_avg is nose.y when no face detected."""
        face = _make_face(detected=False)
        pose = _make_pose(nose_y=0.42)
        zone = HairZone(face=face, pose=pose)
        _, _, _, y_forehead_avg = zone.bounds
        assert y_forehead_avg == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# forehead_line property
# ---------------------------------------------------------------------------

class TestForeheadLineProperty:
    def test_returns_list_of_tuples(self):
        """forehead_line property returns a list of (x, y) tuples."""
        zone = _default_zone()
        line = zone.forehead_line
        assert isinstance(line, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in line)

    def test_empty_when_no_face(self):
        """forehead_line is empty when no face detected."""
        zone = _default_zone(face=_make_face(detected=False))
        assert zone.forehead_line == []

    def test_has_forehead_points(self):
        """forehead_line has at least one point when face is detected."""
        zone = _default_zone()
        assert len(zone.forehead_line) >= 1


# ---------------------------------------------------------------------------
# bounds property shape
# ---------------------------------------------------------------------------

class TestBoundsProperty:
    def test_bounds_is_4_tuple(self):
        """bounds returns a 4-tuple of floats."""
        zone = _default_zone()
        b = zone.bounds
        assert isinstance(b, tuple)
        assert len(b) == 4

    def test_x_min_less_than_x_max(self):
        """x_min must be strictly less than x_max."""
        zone = _default_zone()
        x_min, x_max, _, _ = zone.bounds
        assert x_min < x_max

    def test_ceiling_less_than_or_equal_forehead(self):
        """y_ceiling must be <= y_forehead_avg (ceiling is higher in image)."""
        zone = _default_zone()
        _, _, y_ceiling, y_forehead_avg = zone.bounds
        assert y_ceiling <= y_forehead_avg
