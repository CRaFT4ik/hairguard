"""HairZone — hair region geometry.

Combines FaceMesh forehead contour + Pose ear positions to define the
"hair region": the area above the forehead where hair-touching triggers
detection.

Public API
----------
HairZone(face, pose, hair_zone_offset=0.4)
    .contains(x, y)   — True if (x, y) is in the hair zone
    .forehead_line     — [(x, y), ...] for visualization
    .bounds            — (x_min, x_max, y_ceiling, y_forehead_avg)
"""
from __future__ import annotations

from hairguard.face_detector import FaceResult
from hairguard.detector import PoseResult


class HairZone:
    """Defines the hair region above the forehead and between the ears.

    Parameters
    ----------
    face:
        Result from FaceDetector.detect().
    pose:
        Result from PoseDetector.process().
    hair_zone_offset:
        Fraction of shoulder-width used to compute the upper ceiling of the
        hair zone.  Default 0.4.
    """

    def __init__(
        self,
        face: FaceResult,
        pose: PoseResult,
        hair_zone_offset: float = 0.4,
    ) -> None:
        self._face = face
        self._pose = pose
        self._offset = hair_zone_offset

        # --- Lateral bounds from ears (+15% margin) -------------------------
        lm = pose.landmarks
        left_ear = lm.get("left_ear")
        right_ear = lm.get("right_ear")

        if left_ear is not None and right_ear is not None:
            ear_margin = abs(right_ear.x - left_ear.x) * 0.15
            self._x_min: float = min(left_ear.x, right_ear.x) - ear_margin
            self._x_max: float = max(left_ear.x, right_ear.x) + ear_margin
        else:
            # Degenerate fallback: whole width
            self._x_min = 0.0
            self._x_max = 1.0

        # --- Forehead / nose proxy y-mean ------------------------------------
        forehead_y_mean = self._compute_forehead_y_mean()

        # --- Ceiling: forehead_y_mean - shoulder_width * offset, clamped >=0 --
        left_shoulder = lm.get("left_shoulder")
        right_shoulder = lm.get("right_shoulder")
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        else:
            shoulder_width = 0.0

        self._y_forehead_avg: float = forehead_y_mean
        self._y_ceiling: float = max(0.0, forehead_y_mean - shoulder_width * self._offset)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_forehead_y_mean(self) -> float:
        """Return the mean y of the forehead line, or nose.y as fallback."""
        if self._face.detected:
            line = self._face.forehead_line()
            if line:
                return sum(y for _, y in line) / len(line)

        # Fallback: use nose.y as a forehead proxy
        nose = self._pose.landmarks.get("nose")
        if nose is not None:
            return nose.y

        # Last-resort default
        return 0.5

    def _is_above_forehead(self, x: float, y: float) -> bool:
        """Return True if (x, y) is above the forehead / nose-proxy line."""
        if self._face.detected:
            line = self._face.forehead_line()
            if line:
                return self._face.is_above_forehead(x, y)

        # Fallback: compare against nose.y
        nose = self._pose.landmarks.get("nose")
        if nose is not None:
            return y < nose.y

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contains(self, x: float, y: float) -> bool:
        """Return True if (x, y) falls inside the hair zone.

        Conditions (all must hold):
        1. x is between the ear lateral bounds (inclusive of margin).
        2. (x, y) is above the forehead line (or nose proxy when no face).
        3. y is not above the ceiling (y >= y_ceiling).
        """
        if not (self._x_min <= x <= self._x_max):
            return False
        if not self._is_above_forehead(x, y):
            return False
        if y < self._y_ceiling:
            return False
        return True

    @property
    def forehead_line(self) -> list[tuple[float, float]]:
        """Forehead contour points for visualization, sorted by x.

        Returns the FaceMesh forehead line when a face is detected, otherwise
        an empty list (no useful contour available in fallback mode).
        """
        if self._face.detected:
            return self._face.forehead_line()
        return []

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_ceiling, y_forehead_avg) for visualization."""
        return (self._x_min, self._x_max, self._y_ceiling, self._y_forehead_avg)
