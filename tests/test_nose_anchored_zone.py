"""Tests for NoseAnchoredHairZone (pure unit tests — no model required).

Coverage:
- Zone not learned initially (fallback returns False)
- After feeding enough clean mask frames, zone is learned
- Point inside learned zone returns True when near_hair=True
- Point outside learned zone returns False
- Zone updates via EMA (values shift toward new measurements)
- When near_hair=True (fingers close to hair), zone is NOT updated
- When near_hair=False (fingers away from hair), zone IS updated
"""
from __future__ import annotations

import numpy as np
import pytest

from hairguard.hair_segmenter import HairMask, NoseAnchoredHairZone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask_with_hair(
    h: int = 480,
    w: int = 640,
    hair_top: int = 50,
    hair_bottom: int = 200,
    hair_left: int = 150,
    hair_right: int = 490,
) -> HairMask:
    """Build a HairMask with a rectangular hair region (enough pixels to learn)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[hair_top:hair_bottom, hair_left:hair_right] = 1
    return HairMask(mask=mask, width=w, height=h)


def _empty_mask(h: int = 480, w: int = 640) -> HairMask:
    mask = np.zeros((h, w), dtype=np.uint8)
    return HairMask(mask=mask, width=w, height=h)


# Nose sits roughly in the middle of the face, below the hair
NOSE_PX = (320.0, 260.0)   # (x, y) in pixels — below the hair region top=50..200

# Fingertip far from any hair (bottom of frame) — will not trigger near_hair
FAR_FINGERTIP = [(320.0, 440.0)]

# Fingertip inside the hair mask region
HAIR_FINGERTIP = [(320.0, 120.0)]


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_not_learned_initially(self):
        zone = NoseAnchoredHairZone()
        assert zone.is_learned is False

    def test_zone_bounds_initially_none(self):
        zone = NoseAnchoredHairZone()
        offsets, count = zone.zone_bounds
        assert offsets is None
        assert count == 0

    def test_fallback_returns_false_when_not_learned(self):
        """When zone is not yet learned, the fallback path returns False."""
        zone = NoseAnchoredHairZone()
        hair_mask = _empty_mask()
        # Fingertips away from hair — but we simulate near_hair by providing hair
        # at the fingertip location manually with a mask that has a hole at the tip
        result = zone.update_and_check(hair_mask, NOSE_PX, FAR_FINGERTIP)
        assert result is False


# ---------------------------------------------------------------------------
# Learning (clean frames)
# ---------------------------------------------------------------------------

class TestLearning:
    def test_zone_learned_after_enough_clean_frames(self):
        zone = NoseAnchoredHairZone()
        hair_mask = _make_mask_with_hair()
        for _ in range(15):
            zone.update_and_check(hair_mask, NOSE_PX, FAR_FINGERTIP)
        assert zone.is_learned is True

    def test_zone_not_learned_after_too_few_frames(self):
        zone = NoseAnchoredHairZone()
        hair_mask = _make_mask_with_hair()
        for _ in range(5):
            zone.update_and_check(hair_mask, NOSE_PX, FAR_FINGERTIP)
        assert zone.is_learned is False

    def test_offsets_set_after_clean_frame(self):
        zone = NoseAnchoredHairZone()
        hair_mask = _make_mask_with_hair()
        zone.update_and_check(hair_mask, NOSE_PX, FAR_FINGERTIP)
        offsets, _ = zone.zone_bounds
        assert offsets is not None
        assert len(offsets) == 4

    def test_offsets_reasonable_values(self):
        """Top offset should be > 0 (nose is below hair top)."""
        zone = NoseAnchoredHairZone()
        hair_mask = _make_mask_with_hair(hair_top=50, hair_bottom=200)
        # NOSE_PX y=260, hair_top=50 -> top offset = 260-50 = 210
        for _ in range(12):
            zone.update_and_check(hair_mask, NOSE_PX, FAR_FINGERTIP)
        offsets, _ = zone.zone_bounds
        top, bottom, left, right = offsets
        assert top > 0
        assert left > 0
        assert right > 0


# ---------------------------------------------------------------------------
# Zone containment check
# ---------------------------------------------------------------------------

class TestZoneContainment:
    def setup_method(self):
        self.zone = NoseAnchoredHairZone()
        hair_mask = _make_mask_with_hair()
        # Train the zone with clean frames (fingertip far from hair)
        for _ in range(15):
            self.zone.update_and_check(hair_mask, NOSE_PX, FAR_FINGERTIP)

    def test_point_in_zone_returns_true(self):
        """A point clearly inside the learned zone returns True via fallback."""
        # We need near_hair=True to trigger the fallback path.
        # Construct a mask with no hair AT the fingertip but hair nearby
        h, w = 480, 640
        mask = np.zeros((h, w), dtype=np.uint8)
        # Hair to the side — so near_hair=True (within PROXIMITY_PX)
        mask[100:200, 280:300] = 1   # hair patch nearby
        hair_mask = HairMask(mask=mask, width=w, height=h)

        # Fingertip in the middle of the learned zone but NOT on the mask pixel
        # nose=(320,260), top~210 -> zone_top ~ 260-210=50, fingertip at y=120 is inside
        fingertip_in_zone = [(320.0, 120.0)]  # inside zone, but mask pixel is 0 there

        result = self.zone.update_and_check(hair_mask, NOSE_PX, fingertip_in_zone)
        assert result is True

    def test_point_outside_zone_returns_false_via_fallback(self):
        """A point outside the zone returns False when mask pixel is also 0."""
        h, w = 480, 640
        mask = np.zeros((h, w), dtype=np.uint8)
        # Hair far to the left so near_hair is triggered for our fingertip on the right
        mask[100:200, 0:50] = 1
        hair_mask = HairMask(mask=mask, width=w, height=h)

        # Fingertip well outside the zone (far right, near the hair strip on left)
        # near_hair for fingertip at x=50+PROXIMITY_PX=90 would be true; use x=30
        outside_fingertip = [(30.0, 120.0)]   # near the hair strip
        # The pixel (30, 120) is inside mask — so direct mask check fires -> True
        # We need a point near hair but NOT on hair AND not in zone
        outside_fingertip2 = [(30.0, 400.0)]  # near hair, but y=400 is far below zone
        result = self.zone.update_and_check(hair_mask, NOSE_PX, outside_fingertip2)
        # mask[400, 30] == 0, and y=400 is outside zone (zone bottom ~ 260+bottom_offset)
        assert result is False


# ---------------------------------------------------------------------------
# EMA smoothing
# ---------------------------------------------------------------------------

class TestEMASmoothing:
    def test_values_change_after_different_mask(self):
        """EMA: if we feed a different mask, offsets should shift."""
        zone = NoseAnchoredHairZone()
        mask1 = _make_mask_with_hair(hair_top=50, hair_bottom=200, hair_left=150, hair_right=490)
        for _ in range(20):
            zone.update_and_check(mask1, NOSE_PX, FAR_FINGERTIP)

        offsets_before, _ = zone.zone_bounds
        top_before = offsets_before[0]

        # Now feed a mask where hair is higher (smaller top offset)
        mask2 = _make_mask_with_hair(hair_top=10, hair_bottom=150, hair_left=150, hair_right=490)
        for _ in range(20):
            zone.update_and_check(mask2, NOSE_PX, FAR_FINGERTIP)

        offsets_after, _ = zone.zone_bounds
        top_after = offsets_after[0]

        # top offset should have increased (hair is now higher = more pixels above nose)
        assert top_after > top_before

    def test_values_change_smoothly_not_instantly(self):
        """Single frame update should not jump all the way to new value."""
        zone = NoseAnchoredHairZone()
        mask1 = _make_mask_with_hair(hair_top=50, hair_bottom=200)
        # Seed the zone
        for _ in range(10):
            zone.update_and_check(mask1, NOSE_PX, FAR_FINGERTIP)
        offsets_seed, _ = zone.zone_bounds
        top_seed = offsets_seed[0]

        # One frame with very different mask
        mask2 = _make_mask_with_hair(hair_top=5, hair_bottom=100)
        zone.update_and_check(mask2, NOSE_PX, FAR_FINGERTIP)
        offsets_new, _ = zone.zone_bounds
        top_new = offsets_new[0]

        # nose_y=260, mask2 top=5 -> raw top = 255
        # EMA: 0.05*255 + 0.95*top_seed
        raw_top = 260.0 - 5.0  # 255
        expected = 0.05 * raw_top + 0.95 * top_seed
        assert abs(top_new - expected) < 1.0  # within 1 pixel


# ---------------------------------------------------------------------------
# Update suppression when near hair
# ---------------------------------------------------------------------------

class TestUpdateSuppression:
    def test_zone_not_updated_when_near_hair(self):
        """When a fingertip is near hair pixels, _update_zone is NOT called."""
        zone = NoseAnchoredHairZone()

        # Seed with one clean frame to get initial offsets
        mask = _make_mask_with_hair()
        zone.update_and_check(mask, NOSE_PX, FAR_FINGERTIP)
        offsets_before, count_before = zone.zone_bounds

        # Now call with fingertip INSIDE the hair region (near hair)
        # HAIR_FINGERTIP = (320, 120) which is inside mask (hair_top=50..200)
        zone.update_and_check(mask, NOSE_PX, HAIR_FINGERTIP)
        offsets_after, count_after = zone.zone_bounds

        # frame_count should NOT have incremented
        assert count_after == count_before

    def test_zone_updated_when_not_near_hair(self):
        """When fingertips are away from hair, _update_zone IS called."""
        zone = NoseAnchoredHairZone()
        mask = _make_mask_with_hair()

        # Call with fingertip far from hair
        zone.update_and_check(mask, NOSE_PX, FAR_FINGERTIP)
        _, count_after = zone.zone_bounds
        assert count_after == 1

    def test_near_hair_still_checks_mask_directly(self):
        """Even when near_hair, if fingertip IS on a mask pixel, returns True."""
        zone = NoseAnchoredHairZone()
        mask = _make_mask_with_hair()
        # Fingertip lands directly on a hair pixel
        on_hair_fingertip = [(320.0, 120.0)]  # y=120 is in hair (50..200)
        result = zone.update_and_check(mask, NOSE_PX, on_hair_fingertip)
        assert result is True

    def test_no_hair_pixels_returns_false(self):
        """Empty mask, far fingertip → False."""
        zone = NoseAnchoredHairZone()
        result = zone.update_and_check(_empty_mask(), NOSE_PX, FAR_FINGERTIP)
        assert result is False

    def test_none_nose_skips_zone_update(self):
        """If nose_px is None, zone is not updated but check still proceeds."""
        zone = NoseAnchoredHairZone()
        mask = _make_mask_with_hair()
        zone.update_and_check(mask, None, FAR_FINGERTIP)
        offsets, count = zone.zone_bounds
        assert offsets is None  # no update happened
        assert count == 0
