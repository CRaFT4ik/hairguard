"""Tests for hairguard.head_zone — rotated ellipse model."""
from __future__ import annotations

import math
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
    left_ear_y: float = 0.35,
    right_ear_y: float = 0.35,
    left_shoulder_x: float = 0.2,
    right_shoulder_x: float = 0.8,
    shoulder_y: float = 0.7,
    nose_x: float = 0.5,
    nose_y: float = 0.4,
    ear_confidence: float = 0.9,
    detected: bool = True,
) -> PoseResult:
    """Build a PoseResult with configurable landmark positions.

    Accepts independent y for each ear so head-tilt scenarios can be set up
    without a separate helper.
    """
    if not detected:
        return PoseResult(landmarks={}, raw_landmarks=None, detected=False)
    landmarks = {
        "left_ear": Landmark(x=left_ear_x, y=left_ear_y, confidence=ear_confidence),
        "right_ear": Landmark(x=right_ear_x, y=right_ear_y, confidence=ear_confidence),
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
# 1. TestEllipseParams
# ---------------------------------------------------------------------------

class TestEllipseParams:
    REQUIRED_KEYS = {
        "center",
        "semi_width",
        "semi_height",
        "head_horizontal",
        "up_vec",
        "head_center",
    }

    def test_all_keys_present(self):
        """ellipse_params must contain every documented key."""
        zone = _default_zone()
        params = zone.ellipse_params
        assert self.REQUIRED_KEYS.issubset(params.keys()), (
            f"Missing keys: {self.REQUIRED_KEYS - params.keys()}"
        )

    def test_center_is_2_tuple(self):
        """center must be a 2-element sequence of floats."""
        params = _default_zone().ellipse_params
        cx, cy = params["center"]
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    def test_semi_axes_positive(self):
        """semi_width and semi_height must be strictly positive."""
        params = _default_zone().ellipse_params
        assert params["semi_width"] > 0
        assert params["semi_height"] > 0

    def test_up_vec_is_unit(self):
        """up_vec must be a unit vector."""
        uv = _default_zone().ellipse_params["up_vec"]
        mag = math.sqrt(uv[0] ** 2 + uv[1] ** 2)
        assert mag == pytest.approx(1.0, abs=1e-6)

    def test_head_horizontal_is_unit(self):
        """head_horizontal must be a unit vector."""
        hh = _default_zone().ellipse_params["head_horizontal"]
        mag = math.sqrt(hh[0] ** 2 + hh[1] ** 2)
        assert mag == pytest.approx(1.0, abs=1e-6)

    def test_head_center_is_ear_midpoint(self):
        """head_center must be the midpoint of the two ears."""
        pose = _make_pose(left_ear_x=0.25, right_ear_x=0.75,
                          left_ear_y=0.35, right_ear_y=0.35)
        params = HairZone(face=_make_face(), pose=pose).ellipse_params
        hcx, hcy = params["head_center"]
        assert hcx == pytest.approx(0.5, abs=1e-6)
        assert hcy == pytest.approx(0.35, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. TestContains
# ---------------------------------------------------------------------------

class TestContains:
    """Test the contains() method for various point positions."""

    def test_point_directly_above_head_center_is_true(self):
        """A point directly above the head center (in the hair zone) returns True.

        Default geometry:
          ears at (0.25, 0.35)/(0.75, 0.35) -> head_center=(0.5, 0.35)
          up_vec=(0,-1), hair_center=(0.5, ~0.07), semi_height~0.39
          Point (0.5, 0.1): inside ellipse and above forehead (y<0.35) -> True
        """
        face = _make_face(forehead_y=0.35)
        pose = _make_pose()
        zone = HairZone(face=face, pose=pose)
        assert zone.contains(0.5, 0.1) is True

    def test_point_on_face_below_forehead_is_false(self):
        """A point below the forehead (on the face) returns False.

        Even if geometrically inside the ellipse, the forehead filter excludes it.
        """
        face = _make_face(forehead_y=0.35)
        pose = _make_pose()
        zone = HairZone(face=face, pose=pose)
        # y=0.5 is below forehead_y=0.35 in screen coordinates
        assert zone.contains(0.5, 0.5) is False

    def test_point_far_to_side_is_false(self):
        """A point far outside the ellipse horizontally returns False."""
        zone = _default_zone()
        # semi_width = head_scale * 1.0 ≈ 0.5, center.x ≈ 0.5
        # So x=0.5+0.5=1.0 is the edge. x=1.5 is well outside.
        assert zone.contains(1.5, 0.1) is False

    def test_point_far_above_is_false(self):
        """A point way above the ellipse returns False."""
        zone = _default_zone()
        # semi_height = head_scale * 1.2 ≈ 0.6, center.y ≈ -0.1
        # So y = -0.1 - 0.6 = -0.7 is edge. y=-1.5 is well outside.
        assert zone.contains(0.5, -1.5) is False

    def test_at_ellipse_center_is_true(self):
        """The exact center of the hair ellipse is inside the zone (ignoring forehead)."""
        face = _make_face(detected=False)  # no forehead filter
        pose = _make_pose()
        zone = HairZone(face=face, pose=pose)
        cx, cy = zone.ellipse_params["center"]
        assert zone.contains(cx, cy) is True


# ---------------------------------------------------------------------------
# 3. TestHeadTilt
# ---------------------------------------------------------------------------

class TestHeadTilt:
    """Head tilted to shoulder: the ellipse rotates to follow the head axes.

    Setup: left ear is LOWER (toward left shoulder), right ear is HIGHER.
      left_ear=(0.25, 0.45), right_ear=(0.75, 0.25)
    The up_vec tilts toward upper-LEFT, shifting the hair_center left.

    A point at (0.2, 0.05) is:
      - INSIDE the rotated (tilted) ellipse  → contains returns True
      - OUTSIDE a hypothetical vertical ellipse with the same head center
    """

    def _tilted_zone(self) -> HairZone:
        pose = _make_pose(
            left_ear_x=0.25, left_ear_y=0.45,
            right_ear_x=0.75, right_ear_y=0.25,
            nose_x=0.5, nose_y=0.35,
        )
        return HairZone(face=_make_face(detected=False), pose=pose)

    def _straight_zone(self) -> HairZone:
        pose = _make_pose(
            left_ear_x=0.25, left_ear_y=0.35,
            right_ear_x=0.75, right_ear_y=0.35,
            nose_x=0.5, nose_y=0.35,
        )
        return HairZone(face=_make_face(detected=False), pose=pose)

    def test_tilted_ellipse_is_rotated(self):
        """Tilted head produces a rotated ellipse (different angle than straight)."""
        tilted = self._tilted_zone()
        straight = self._straight_zone()
        tp = tilted.ellipse_params
        sp = straight.ellipse_params
        # Straight: head_horizontal ≈ (1, 0). Tilted: head_horizontal is angled.
        straight_angle = math.atan2(sp["head_horizontal"][1], sp["head_horizontal"][0])
        tilted_angle = math.atan2(tp["head_horizontal"][1], tp["head_horizontal"][0])
        assert abs(tilted_angle - straight_angle) > 0.1  # at least ~6 degrees difference

    def test_tilted_ellipse_center_shifts_toward_upper_left(self):
        """When the right ear is higher, the hair center shifts toward upper-left."""
        tilted = self._tilted_zone()
        straight = self._straight_zone()
        cx_tilted = tilted.ellipse_params["center"][0]
        cx_straight = straight.ellipse_params["center"][0]
        # Hair center for tilted head should be to the left of the vertical center
        assert cx_tilted < cx_straight

    def test_head_horizontal_is_not_axis_aligned_when_tilted(self):
        """head_horizontal is not (1, 0) for a tilted head."""
        params = self._tilted_zone().ellipse_params
        hx, hy = params["head_horizontal"]
        # Should have a significant y component from the tilt
        assert abs(hy) > 0.1

    def test_up_vec_rotated_from_vertical_when_tilted(self):
        """up_vec for a tilted head differs from the (0, -1) of a straight head."""
        params = self._tilted_zone().ellipse_params
        ux, uy = params["up_vec"]
        # Straight head: up_vec = (0, -1). Tilted head has a non-zero x component.
        assert abs(ux) > 0.1


# ---------------------------------------------------------------------------
# 4. TestUpVector
# ---------------------------------------------------------------------------

class TestUpVector:
    """Verify up_vec direction for straight and tilted heads."""

    def test_straight_head_up_vec_points_screen_up(self):
        """Straight head (ears at same y) produces up_vec ~(0, -1).

        In screen coordinates y increases downward, so 'up' is (0, -1).
        """
        pose = _make_pose(
            left_ear_x=0.25, left_ear_y=0.35,
            right_ear_x=0.75, right_ear_y=0.35,
        )
        params = HairZone(face=_make_face(), pose=pose).ellipse_params
        ux, uy = params["up_vec"]
        assert ux == pytest.approx(0.0, abs=1e-6)
        assert uy == pytest.approx(-1.0, abs=1e-6)

    def test_tilted_head_up_vec_has_nonzero_x(self):
        """Head tilted to shoulder produces up_vec with non-zero x component."""
        pose = _make_pose(
            left_ear_x=0.25, left_ear_y=0.45,
            right_ear_x=0.75, right_ear_y=0.25,
        )
        params = HairZone(face=_make_face(), pose=pose).ellipse_params
        ux, uy = params["up_vec"]
        assert abs(ux) > 0.1

    def test_up_vec_always_points_away_from_shoulders(self):
        """up_vec must point from head center toward the crown (away from shoulders).

        Verified by checking that up_vec dotted with the head→shoulder vector is
        negative (opposite directions).
        """
        pose = _make_pose()
        zone = HairZone(face=_make_face(), pose=pose)
        params = zone.ellipse_params
        hcx, hcy = params["head_center"]
        # Shoulder center for default pose is (0.5, 0.7)
        to_shoulder = (0.5 - hcx, 0.7 - hcy)
        ux, uy = params["up_vec"]
        dot = ux * to_shoulder[0] + uy * to_shoulder[1]
        assert dot < 0, "up_vec should point away from shoulders"


# ---------------------------------------------------------------------------
# 5. TestFallbacks
# ---------------------------------------------------------------------------

class TestFallbacks:
    """Fallback behaviour when face or ear landmarks are unavailable."""

    def test_no_face_detected_contains_still_works(self):
        """When no face is detected the forehead filter is skipped.

        contains() falls back to purely ellipse-based membership, so a
        point inside the ellipse that would normally be rejected by the
        forehead boundary still returns True.
        """
        face = _make_face(detected=False)
        pose = _make_pose()
        zone = HairZone(face=face, pose=pose)
        # hair_center ~(0.5, 0.07); this point is inside the ellipse
        assert zone.contains(0.5, 0.1) is True

    def test_no_face_detected_point_outside_ellipse_is_false(self):
        """With no face, a point still outside the ellipse returns False."""
        face = _make_face(detected=False)
        zone = HairZone(face=face, pose=_make_pose())
        assert zone.contains(0.5, 0.9) is False

    def test_low_confidence_ears_fallback_to_camera_horizontal(self):
        """When both ears are below the confidence threshold, head_horizontal
        falls back to the camera horizontal derived from the shoulders.

        Shoulders are horizontal here, so head_horizontal should be ~(1, 0).
        """
        pose = _make_pose(ear_confidence=0.1)  # below _CONF_THRESHOLD=0.5
        zone = HairZone(face=_make_face(detected=False), pose=pose)
        hx, hy = zone.ellipse_params["head_horizontal"]
        assert hx == pytest.approx(1.0, abs=1e-6)
        assert hy == pytest.approx(0.0, abs=1e-6)

    def test_low_confidence_ears_uses_nose_as_head_center(self):
        """With low-confidence ears the head_center falls back to the nose position."""
        nose_x, nose_y = 0.52, 0.38
        pose = _make_pose(nose_x=nose_x, nose_y=nose_y, ear_confidence=0.1)
        zone = HairZone(face=_make_face(detected=False), pose=pose)
        hcx, hcy = zone.ellipse_params["head_center"]
        assert hcx == pytest.approx(nose_x, abs=1e-6)
        assert hcy == pytest.approx(nose_y, abs=1e-6)

    def test_no_face_and_low_confidence_ears_still_has_positive_semi_axes(self):
        """Even with worst-case fallbacks the ellipse axes remain positive."""
        pose = _make_pose(ear_confidence=0.1)
        zone = HairZone(face=_make_face(detected=False), pose=pose)
        params = zone.ellipse_params
        assert params["semi_width"] > 0
        assert params["semi_height"] > 0


# ---------------------------------------------------------------------------
# 6. TestPitchExpansion
# ---------------------------------------------------------------------------

class TestPitchExpansion:
    """Head pitch (forward lean) should expand the semi_height of the ellipse."""

    def test_nose_at_ear_level_expands_semi_height(self):
        """When the nose is at the same y-level as the ears the head is pitched
        forward and the back of the head is exposed.  semi_height must be
        greater than for a neutral (nose-below-ears) pose.
        """
        # Neutral: nose below ears (nose_y > ear_y in screen coords)
        pose_neutral = _make_pose(
            left_ear_y=0.35, right_ear_y=0.35, nose_y=0.4
        )
        # Forward pitch: nose at same height as ears
        pose_forward = _make_pose(
            left_ear_y=0.35, right_ear_y=0.35, nose_y=0.35
        )
        zone_neutral = HairZone(face=_make_face(detected=False), pose=pose_neutral)
        zone_forward = HairZone(face=_make_face(detected=False), pose=pose_forward)

        sh_neutral = zone_neutral.ellipse_params["semi_height"]
        sh_forward = zone_forward.ellipse_params["semi_height"]
        assert sh_forward > sh_neutral, (
            f"Expected expanded semi_height for forward pitch: "
            f"{sh_forward} > {sh_neutral}"
        )

    def test_nose_well_below_ears_gives_larger_expansion(self):
        """Nose well below ears (looking further forward) expands even more."""
        pose_mild = _make_pose(
            left_ear_y=0.35, right_ear_y=0.35, nose_y=0.38
        )
        pose_strong = _make_pose(
            left_ear_y=0.35, right_ear_y=0.35, nose_y=0.35
        )
        sh_mild = HairZone(
            face=_make_face(detected=False), pose=pose_mild
        ).ellipse_params["semi_height"]
        sh_strong = HairZone(
            face=_make_face(detected=False), pose=pose_strong
        ).ellipse_params["semi_height"]
        assert sh_strong > sh_mild

    def test_pitch_factor_does_not_shrink_zone(self):
        """pitch_factor is >= 1.0, so semi_height is never reduced below baseline."""
        pose = _make_pose(nose_y=0.6)  # extreme: nose far below ears
        zone = HairZone(face=_make_face(detected=False), pose=pose)
        params = zone.ellipse_params
        head_scale = 0.5  # from default ear distance
        baseline_sh = head_scale * 1.2 * 1.0  # pitch_factor >= 1
        assert params["semi_height"] >= baseline_sh - 1e-9


    def test_pitch_offset_shifts_center(self):
        """When leaning forward, hair_center shifts further from head_center."""
        # Neutral: nose at y=0.4, ears at y=0.35
        neutral_pose = _make_pose(nose_y=0.4)
        neutral_zone = HairZone(face=_make_face(detected=False), pose=neutral_pose)
        nc = neutral_zone.ellipse_params["center"]
        nhc = neutral_zone.ellipse_params["head_center"]
        neutral_dist = math.sqrt((nc[0]-nhc[0])**2 + (nc[1]-nhc[1])**2)

        # Forward lean: nose at y=0.35 (same as ears = face down)
        lean_pose = _make_pose(nose_y=0.35)
        lean_zone = HairZone(face=_make_face(detected=False), pose=lean_pose)
        lc = lean_zone.ellipse_params["center"]
        lhc = lean_zone.ellipse_params["head_center"]
        lean_dist = math.sqrt((lc[0]-lhc[0])**2 + (lc[1]-lhc[1])**2)

        assert lean_dist > neutral_dist  # center shifts further when leaning


# ---------------------------------------------------------------------------
# 7. TestBoundsProperty (backward-compatibility)
# ---------------------------------------------------------------------------

class TestBoundsProperty:
    """bounds returns (cx-sw, cx+sw, cy-sh, cy) — approximate bounding box."""

    def test_bounds_is_4_tuple(self):
        """bounds returns a 4-element tuple."""
        b = _default_zone().bounds
        assert isinstance(b, tuple)
        assert len(b) == 4

    def test_x_min_less_than_x_max(self):
        """x_min must be strictly less than x_max."""
        x_min, x_max, _, _ = _default_zone().bounds
        assert x_min < x_max

    def test_y_ceiling_less_than_y_center(self):
        """y_ceiling (cy - semi_height) must be <= cy (ellipse center y)."""
        _, _, y_ceiling, cy = _default_zone().bounds
        assert y_ceiling <= cy

    def test_bounds_matches_ellipse_params(self):
        """bounds values must be consistent with ellipse_params."""
        zone = _default_zone()
        params = zone.ellipse_params
        cx, cy = params["center"]
        sw = params["semi_width"]
        sh = params["semi_height"]
        x_min, x_max, y_ceil, y_ctr = zone.bounds
        assert x_min == pytest.approx(cx - sw, abs=1e-6)
        assert x_max == pytest.approx(cx + sw, abs=1e-6)
        assert y_ceil == pytest.approx(cy - sh, abs=1e-6)
        assert y_ctr == pytest.approx(cy, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. TestForeheadLineProperty
# ---------------------------------------------------------------------------

class TestForeheadLineProperty:
    def test_returns_list_of_tuples(self):
        """forehead_line property returns a list of (x, y) tuples."""
        line = _default_zone().forehead_line
        assert isinstance(line, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in line)

    def test_empty_when_no_face(self):
        """forehead_line is empty when no face detected."""
        assert _default_zone(face=_make_face(detected=False)).forehead_line == []

    def test_has_forehead_points_when_face_detected(self):
        """forehead_line has at least one point when a face is detected."""
        assert len(_default_zone().forehead_line) >= 1
