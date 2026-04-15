"""Tests for hairguard.face_detector — TDD suite."""
from __future__ import annotations

import pytest

from hairguard.face_detector import FaceLandmark, FaceResult, FOREHEAD_INDICES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(points: dict[int, tuple[float, float]], detected: bool = True) -> FaceResult:
    """Build a FaceResult from a plain {index: (x, y)} mapping."""
    landmarks = {idx: FaceLandmark(x=x, y=y) for idx, (x, y) in points.items()}
    return FaceResult(landmarks=landmarks, detected=detected)


def _forehead_result() -> FaceResult:
    """Return a FaceResult with all FOREHEAD_INDICES populated at known positions.

    Indices are spread across x=0.1..0.9, y=0.2 (a flat hairline for easy maths).
    We deliberately insert them in reverse index order so sorting is observable.
    """
    n = len(FOREHEAD_INDICES)
    # Assign x values spread over 0.1 .. 0.9, y fixed at 0.2
    xs = [0.1 + 0.8 * i / (n - 1) for i in range(n)]
    # Shuffle index→x mapping so that FOREHEAD_INDICES order != x order
    shuffled_indices = list(reversed(FOREHEAD_INDICES))
    points = {idx: (xs[i], 0.2) for i, idx in enumerate(shuffled_indices)}
    return _make_result(points)


# ---------------------------------------------------------------------------
# FaceLandmark
# ---------------------------------------------------------------------------

class TestFaceLandmark:
    def test_creation(self):
        lm = FaceLandmark(x=0.5, y=0.3)
        assert lm.x == 0.5
        assert lm.y == 0.3

    def test_is_dataclass(self):
        from dataclasses import fields
        f_names = {f.name for f in fields(FaceLandmark)}
        assert f_names == {"x", "y"}


# ---------------------------------------------------------------------------
# FaceResult basic
# ---------------------------------------------------------------------------

class TestFaceResultCreation:
    def test_detected_true(self):
        r = _forehead_result()
        assert r.detected is True

    def test_detected_false(self):
        r = FaceResult(landmarks={}, detected=False)
        assert r.detected is False
        assert r.landmarks == {}

    def test_landmarks_accessible_by_index(self):
        r = _make_result({10: (0.5, 0.2)})
        assert isinstance(r.landmarks[10], FaceLandmark)
        assert r.landmarks[10].x == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# forehead_line
# ---------------------------------------------------------------------------

class TestForeheadLine:
    def test_returns_list_of_tuples(self):
        r = _forehead_result()
        line = r.forehead_line()
        assert isinstance(line, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in line)

    def test_sorted_by_x(self):
        r = _forehead_result()
        line = r.forehead_line()
        xs = [p[0] for p in line]
        assert xs == sorted(xs), "forehead_line() must be sorted ascending by x"

    def test_length_matches_forehead_indices(self):
        r = _forehead_result()
        line = r.forehead_line()
        assert len(line) == len(FOREHEAD_INDICES)

    def test_empty_when_not_detected(self):
        r = FaceResult(landmarks={}, detected=False)
        assert r.forehead_line() == []

    def test_missing_indices_skipped(self):
        """If some FOREHEAD_INDICES are absent from landmarks, they are skipped."""
        # Only provide half the indices
        half = FOREHEAD_INDICES[: len(FOREHEAD_INDICES) // 2]
        points = {idx: (float(i) / 10, 0.2) for i, idx in enumerate(half)}
        r = _make_result(points)
        line = r.forehead_line()
        assert len(line) == len(half)


# ---------------------------------------------------------------------------
# forehead_y_at_x
# ---------------------------------------------------------------------------

class TestForeheadYAtX:
    def _flat_result(self) -> FaceResult:
        """Flat hairline at y=0.2 spanning x=0.0 to x=1.0 (two-point version)."""
        return _make_result(
            {
                FOREHEAD_INDICES[0]: (0.0, 0.2),
                FOREHEAD_INDICES[1]: (1.0, 0.2),
            }
        )

    def _sloped_result(self) -> FaceResult:
        """Diagonal: y = x (x in [0,1])."""
        return _make_result(
            {
                FOREHEAD_INDICES[0]: (0.0, 0.0),
                FOREHEAD_INDICES[1]: (1.0, 1.0),
            }
        )

    def test_flat_line_at_midpoint(self):
        r = self._flat_result()
        assert r.forehead_y_at_x(0.5) == pytest.approx(0.2)

    def test_flat_line_at_left_endpoint(self):
        r = self._flat_result()
        assert r.forehead_y_at_x(0.0) == pytest.approx(0.2)

    def test_flat_line_at_right_endpoint(self):
        r = self._flat_result()
        assert r.forehead_y_at_x(1.0) == pytest.approx(0.2)

    def test_sloped_interpolation(self):
        r = self._sloped_result()
        assert r.forehead_y_at_x(0.25) == pytest.approx(0.25)
        assert r.forehead_y_at_x(0.75) == pytest.approx(0.75)

    def test_returns_none_outside_range_left(self):
        r = _make_result(
            {
                FOREHEAD_INDICES[0]: (0.3, 0.2),
                FOREHEAD_INDICES[1]: (0.7, 0.2),
            }
        )
        assert r.forehead_y_at_x(0.1) is None

    def test_returns_none_outside_range_right(self):
        r = _make_result(
            {
                FOREHEAD_INDICES[0]: (0.3, 0.2),
                FOREHEAD_INDICES[1]: (0.7, 0.2),
            }
        )
        assert r.forehead_y_at_x(0.9) is None

    def test_returns_none_when_no_forehead(self):
        r = FaceResult(landmarks={}, detected=False)
        assert r.forehead_y_at_x(0.5) is None


# ---------------------------------------------------------------------------
# is_above_forehead
# ---------------------------------------------------------------------------

class TestIsAboveForehead:
    def _flat_result(self) -> FaceResult:
        """Flat hairline at y=0.5 spanning x=0.2..0.8."""
        return _make_result(
            {
                FOREHEAD_INDICES[0]: (0.2, 0.5),
                FOREHEAD_INDICES[1]: (0.8, 0.5),
            }
        )

    def test_above_forehead_returns_true(self):
        """y=0.1 is above (smaller y) than forehead at y=0.5."""
        r = self._flat_result()
        assert r.is_above_forehead(0.5, 0.1) is True

    def test_below_forehead_returns_false(self):
        """y=0.9 is below (larger y) than forehead at y=0.5."""
        r = self._flat_result()
        assert r.is_above_forehead(0.5, 0.9) is False

    def test_on_forehead_line_returns_false(self):
        """Point exactly on the line is NOT above it."""
        r = self._flat_result()
        assert r.is_above_forehead(0.5, 0.5) is False

    def test_outside_range_left_uses_edge_point(self):
        """x=0.05 is left of forehead (starts at 0.2), clamp to leftmost point y=0.5."""
        r = self._flat_result()
        # y=0.1 < 0.5 → above even at edge
        assert r.is_above_forehead(0.05, 0.1) is True

    def test_outside_range_right_uses_edge_point(self):
        """x=0.95 is right of forehead (ends at 0.8), clamp to rightmost point y=0.5."""
        r = self._flat_result()
        # y=0.9 > 0.5 → below even at edge
        assert r.is_above_forehead(0.95, 0.9) is False

    def test_no_forehead_returns_false(self):
        """When face not detected / no forehead, default to False."""
        r = FaceResult(landmarks={}, detected=False)
        assert r.is_above_forehead(0.5, 0.1) is False

    def test_sloped_above(self):
        """Diagonal hairline y=x; point (0.5, 0.3) → y=0.3 < hairline_y=0.5 → above."""
        r = _make_result(
            {
                FOREHEAD_INDICES[0]: (0.0, 0.0),
                FOREHEAD_INDICES[1]: (1.0, 1.0),
            }
        )
        assert r.is_above_forehead(0.5, 0.3) is True

    def test_sloped_below(self):
        """Diagonal hairline y=x; point (0.5, 0.7) → y=0.7 > hairline_y=0.5 → below."""
        r = _make_result(
            {
                FOREHEAD_INDICES[0]: (0.0, 0.0),
                FOREHEAD_INDICES[1]: (1.0, 1.0),
            }
        )
        assert r.is_above_forehead(0.5, 0.7) is False
