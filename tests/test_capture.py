"""Tests for hairlab.capture — the crop geometry used when storing clips.

Clips keep a square patch around the head and the hand rather than the whole
frame: it is what the model needs, it keeps clips small, and it leaves the
rest of the room out of the dataset.
"""
from __future__ import annotations

import numpy as np
import pytest

from hairlab.capture import crop_box_for_hand, crop_frame, square_crop_box

W, H = 640, 480


class _FakeLandmark:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y, self.z = x, y, 0.0


class _FakeHand:
    """A hand whose landmarks fill a square of ``size`` centred on (cx, cy)."""

    def __init__(self, cx: float, cy: float, size: float) -> None:
        half = size / 2
        self.landmarks = {
            i: _FakeLandmark(cx + dx * half, cy + dy * half)
            for i, (dx, dy) in enumerate(
                [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)]
            )
        }


def _box_side(box) -> int:
    return box[2] - box[0]


def test_box_contains_every_point() -> None:
    points = [(0.4, 0.3), (0.6, 0.5)]
    x0, y0, x1, y1 = square_crop_box(points, W, H)
    assert x0 <= 0.4 * W and x1 >= 0.6 * W
    assert y0 <= 0.3 * H and y1 >= 0.5 * H


def test_box_is_square() -> None:
    x0, y0, x1, y1 = square_crop_box([(0.2, 0.2), (0.7, 0.3)], W, H)
    assert (x1 - x0) == (y1 - y0)


def test_box_stays_inside_the_frame() -> None:
    """Points at the very corner must not produce a box hanging off the edge."""
    x0, y0, x1, y1 = square_crop_box([(0.0, 0.0), (0.02, 0.02)], W, H)
    assert x0 >= 0 and y0 >= 0
    assert x1 <= W and y1 <= H


def test_box_stays_inside_the_frame_at_the_far_corner() -> None:
    x0, y0, x1, y1 = square_crop_box([(0.99, 0.99)], W, H)
    assert x0 >= 0 and y0 >= 0
    assert x1 <= W and y1 <= H


def test_points_outside_the_frame_are_clamped() -> None:
    """MediaPipe happily reports landmarks slightly outside the image."""
    x0, y0, x1, y1 = square_crop_box([(-0.2, 0.5), (1.3, 0.5)], W, H)
    assert x0 >= 0 and x1 <= W
    assert y0 >= 0 and y1 <= H


def test_no_points_falls_back_to_a_centre_box() -> None:
    x0, y0, x1, y1 = square_crop_box([], W, H)
    assert x1 > x0 and y1 > y0
    assert x0 >= 0 and y1 <= H
    # Roughly centred
    assert abs(((x0 + x1) / 2) - W / 2) < W * 0.1


def test_a_single_point_still_yields_a_usable_box() -> None:
    x0, y0, x1, y1 = square_crop_box([(0.5, 0.5)], W, H)
    assert (x1 - x0) >= 64


# ---------------------------------------------------------------------------
# crop_box_for_hand — the crop follows the hand, at a consistent scale
# ---------------------------------------------------------------------------

def test_crop_is_centred_on_the_hand() -> None:
    box = crop_box_for_hand(_FakeHand(0.3, 0.6, 0.1), W, H)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    assert abs(cx - 0.3 * W) < 4
    assert abs(cy - 0.6 * H) < 4


def test_crop_contains_the_whole_hand() -> None:
    hand = _FakeHand(0.5, 0.5, 0.2)
    x0, y0, x1, y1 = crop_box_for_hand(hand, W, H)
    for lm in hand.landmarks.values():
        assert x0 <= lm.x * W <= x1
        assert y0 <= lm.y * H <= y1


def test_a_bigger_hand_gets_a_bigger_crop() -> None:
    """Scale must follow the hand, not be fixed — the user moves nearer and further."""
    small = crop_box_for_hand(_FakeHand(0.5, 0.5, 0.1), W, H)
    large = crop_box_for_hand(_FakeHand(0.5, 0.5, 0.2), W, H)
    assert _box_side(large) > _box_side(small) * 1.5


def test_the_hand_fills_the_same_fraction_at_any_distance() -> None:
    """The point of the whole design: a 224px clip always frames the hand alike.

    The old crop spanned head and hands together, so the hand could fill the
    frame in one clip and be a thumbnail in the next.
    """
    fractions = []
    for size in (0.08, 0.12, 0.18, 0.25):
        box = crop_box_for_hand(_FakeHand(0.5, 0.5, size), W, H)
        fractions.append((size * W) / _box_side(box))
    assert max(fractions) - min(fractions) < 0.05


def test_crop_follows_the_given_hand_not_another_one() -> None:
    left = crop_box_for_hand(_FakeHand(0.2, 0.5, 0.1), W, H)
    right = crop_box_for_hand(_FakeHand(0.8, 0.5, 0.1), W, H)
    assert left[0] < right[0]
    assert abs(_box_side(left) - _box_side(right)) <= 1


def test_crop_for_a_hand_at_the_edge_stays_inside_the_frame() -> None:
    box = crop_box_for_hand(_FakeHand(0.02, 0.02, 0.15), W, H)
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= W and box[3] <= H


def test_crop_for_a_hand_without_landmarks_falls_back() -> None:
    class _Empty:
        landmarks: dict = {}

    box = crop_box_for_hand(_Empty(), W, H)
    assert _box_side(box) > 0


def test_clips_are_at_least_ten_seconds_apart() -> None:
    """One long episode should yield a few varied samples, not a burst.

    Near-duplicate clips cost labelling time and teach the model nothing.
    """
    from hairlab.capture import DEFAULT_COOLDOWN_SEC

    assert DEFAULT_COOLDOWN_SEC >= 10.0


def test_crop_frame_returns_the_requested_size_marker() -> None:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    out = crop_frame(frame, (10, 10, 210, 210), size=224)
    assert out.shape == (224, 224, 3)


def test_crop_frame_keeps_the_selected_region() -> None:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[100:200, 100:200] = 255
    out = crop_frame(frame, (100, 100, 200, 200), size=64)
    assert out.mean() > 250
