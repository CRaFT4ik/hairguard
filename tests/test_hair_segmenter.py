"""Tests for HairMask (pure unit tests — no model required).

Coverage:
- HairMask.contains() returns True for hair pixels
- HairMask.contains() returns False for non-hair pixels
- HairMask.detected is True when mask has hair pixels
- HairMask.detected is False for empty mask
- Edge cases: x=0, y=0, x=1, y=1
- Coordinate clamping stays within bounds
"""
from __future__ import annotations

import numpy as np
import pytest

from hairguard.hair_segmenter import HairMask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(hair_rows: tuple[int, int], hair_cols: tuple[int, int], h: int = 480, w: int = 640) -> HairMask:
    """Build a HairMask with a rectangular hair region."""
    mask = np.zeros((h, w), dtype=np.uint8)
    r0, r1 = hair_rows
    c0, c1 = hair_cols
    mask[r0:r1, c0:c1] = 1
    return HairMask(mask=mask, width=w, height=h)


# ---------------------------------------------------------------------------
# detected flag
# ---------------------------------------------------------------------------

class TestDetectedFlag:
    def test_detected_true_when_hair_pixels_present(self):
        hm = _make_mask((100, 200), (100, 300))
        assert hm.detected is True

    def test_detected_false_for_empty_mask(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        hm = HairMask(mask=mask, width=640, height=480)
        assert hm.detected is False

    def test_detected_true_for_single_pixel(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[240, 320] = 1
        hm = HairMask(mask=mask, width=640, height=480)
        assert hm.detected is True


# ---------------------------------------------------------------------------
# contains()
# ---------------------------------------------------------------------------

class TestContains:
    """HairMask with hair in rows 100..200, cols 200..400 (h=480, w=640)."""

    def setup_method(self):
        # Hair region: y in [100/480, 200/480) ≈ [0.208, 0.417)
        #              x in [200/640, 400/640) ≈ [0.313, 0.625)
        self.hm = _make_mask((100, 200), (200, 400), h=480, w=640)

    def test_contains_true_for_hair_pixel(self):
        # Centre of the hair rectangle: row=150, col=300
        # x = 300/640 ≈ 0.469, y = 150/480 ≈ 0.313
        assert self.hm.contains(300 / 640, 150 / 480) is True

    def test_contains_false_outside_hair(self):
        # Far below and right of hair region
        assert self.hm.contains(0.9, 0.9) is False

    def test_contains_false_above_hair(self):
        # Row 50 is above the hair region (rows 100..200)
        assert self.hm.contains(0.46, 50 / 480) is False

    def test_contains_false_left_of_hair(self):
        # Col 50 is left of hair region (cols 200..400)
        assert self.hm.contains(50 / 640, 150 / 480) is False

    # Edge-case coordinates

    def test_contains_top_left_corner(self):
        """x=0, y=0 must not raise and returns bool."""
        result = self.hm.contains(0.0, 0.0)
        assert isinstance(result, bool)
        # Pixel (0,0) is not in the hair region
        assert result is False

    def test_contains_bottom_right_corner(self):
        """x=1, y=1 must not raise — clamps to last pixel."""
        result = self.hm.contains(1.0, 1.0)
        assert isinstance(result, bool)

    def test_contains_x_exactly_zero(self):
        result = self.hm.contains(0.0, 0.5)
        assert isinstance(result, bool)

    def test_contains_y_exactly_zero(self):
        result = self.hm.contains(0.5, 0.0)
        assert isinstance(result, bool)

    def test_contains_x_exactly_one(self):
        # Should not raise even at boundary
        result = self.hm.contains(1.0, 0.5)
        assert isinstance(result, bool)

    def test_contains_y_exactly_one(self):
        result = self.hm.contains(0.5, 1.0)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# get_mask()
# ---------------------------------------------------------------------------

class TestGetMask:
    def test_returns_numpy_array(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        hm = HairMask(mask=mask, width=640, height=480)
        assert isinstance(hm.get_mask(), np.ndarray)

    def test_mask_shape_matches_dimensions(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        hm = HairMask(mask=mask, width=640, height=480)
        assert hm.get_mask().shape == (480, 640)

    def test_mask_values_preserved(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        hm = HairMask(mask=mask, width=10, height=10)
        assert hm.get_mask()[5, 5] == 1
        assert hm.get_mask()[0, 0] == 0


# ---------------------------------------------------------------------------
# Dimension properties
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_width_and_height_stored(self):
        mask = np.zeros((240, 320), dtype=np.uint8)
        hm = HairMask(mask=mask, width=320, height=240)
        assert hm.width == 320
        assert hm.height == 240
