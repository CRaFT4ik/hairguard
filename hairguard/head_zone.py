"""HairZone — rotated ellipse hair region based on head orientation.

Uses 5 landmarks (nose, 2 ears, 2 shoulders) to compute a rotated ellipse
that tracks the hair zone regardless of head tilt, pitch, yaw, or camera angle.

See project/docs/hair-zone-geometry.md for full mathematical derivation.

Public API
----------
HairZone(face, pose, hair_zone_offset=0.4)
    .contains(x, y)       — True if (x, y) is in the hair zone
    .ellipse_params        — (center, semi_w, semi_h, head_horizontal, up_vec) for visualization
    .forehead_line         — [(x, y), ...] from FaceMesh for visualization
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from hairguard.face_detector import FaceResult
from hairguard.detector import PoseResult, Landmark


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _length(v: tuple[float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


def _normalize(v: tuple[float, float]) -> tuple[float, float]:
    ln = _length(v)
    if ln < 1e-9:
        return (0.0, 0.0)
    return (v[0] / ln, v[1] / ln)


def _midpoint(a: Landmark, b: Landmark) -> tuple[float, float]:
    return ((a.x + b.x) / 2, (a.y + b.y) / 2)


# Confidence threshold for landmark reliability
_CONF_THRESHOLD = 0.5


class HairZone:
    """Rotated ellipse hair zone based on head orientation.

    Computes orientation from nose + ears + shoulders, then places
    an ellipse above the head center aligned with the head axes.
    """

    def __init__(
        self,
        face: FaceResult,
        pose: PoseResult,
        hair_zone_offset: float = 0.4,
    ) -> None:
        self._face = face
        self._pose = pose

        lm = pose.landmarks
        nose = lm.get("nose")
        left_ear = lm.get("left_ear")
        right_ear = lm.get("right_ear")
        left_sh = lm.get("left_shoulder")
        right_sh = lm.get("right_shoulder")

        # --- Step 1: Camera baseline from shoulders ---
        shoulders_ok = (
            left_sh is not None and right_sh is not None
            and left_sh.confidence >= _CONF_THRESHOLD
            and right_sh.confidence >= _CONF_THRESHOLD
        )
        if shoulders_ok:
            shoulder_vec = (right_sh.x - left_sh.x, right_sh.y - left_sh.y)
            self._camera_horizontal = _normalize(shoulder_vec)
            self._shoulder_distance = _length(shoulder_vec)
            self._shoulder_center = _midpoint(left_sh, right_sh)
        else:
            self._camera_horizontal = (1.0, 0.0)  # screen horizontal fallback
            self._shoulder_distance = 0.3  # reasonable default
            self._shoulder_center = (0.5, 0.7)

        # --- Step 2: Head orientation vectors ---
        ears_ok = (
            left_ear is not None and right_ear is not None
            and left_ear.confidence >= _CONF_THRESHOLD
            and right_ear.confidence >= _CONF_THRESHOLD
        )

        if ears_ok:
            ear_vec = (right_ear.x - left_ear.x, right_ear.y - left_ear.y)
            ear_distance = _length(ear_vec)
        else:
            ear_vec = None
            ear_distance = 0.0

        # Guard: if ears foreshortened, fall back to camera horizontal
        min_ear_dist = self._shoulder_distance * 0.2
        if ear_vec is not None and ear_distance >= min_ear_dist:
            self._head_horizontal = _normalize(ear_vec)
        else:
            self._head_horizontal = self._camera_horizontal
            ear_distance = 0.0  # signal to use shoulder-based scale

        # --- Step 3: Head center and scale ---
        if ears_ok and ear_distance >= min_ear_dist:
            self._head_center = _midpoint(left_ear, right_ear)
        elif nose is not None and nose.confidence >= _CONF_THRESHOLD:
            # Both ears occluded — use nose as head center
            self._head_center = (nose.x, nose.y)
        else:
            self._head_center = (0.5, 0.4)  # last resort

        self._head_scale = max(ear_distance, self._shoulder_distance * 0.45)

        # --- Step 4: Up vector (toward crown) ---
        hh = self._head_horizontal
        # 90° CCW in screen coords (Y down) = visual "up"
        up_candidate = (-hh[1], hh[0])

        # Determine sign: head is above shoulders
        down_vec = (
            self._shoulder_center[0] - self._head_center[0],
            self._shoulder_center[1] - self._head_center[1],
        )
        down_len = _length(down_vec)

        if down_len < 0.05 * self._head_scale:
            # Top-down camera — head and shoulders overlap
            up_candidate = (0.0, -1.0)  # screen-up fallback
        elif _dot(up_candidate, down_vec) > 0:
            # up_candidate points toward shoulders — flip
            up_candidate = (-up_candidate[0], -up_candidate[1])

        self._up_vec = _normalize(up_candidate)

        # --- Step 5: Pitch detection (decomposed) ---
        if nose is not None and nose.confidence >= _CONF_THRESHOLD:
            face_vec = (nose.x - self._head_center[0], nose.y - self._head_center[1])
            # Signed pitch: negative dot with up_vec means nose is above ears (looking at ceiling)
            pitch_component = -_dot(face_vec, self._up_vec)
            pitch_ratio = pitch_component / self._head_scale if self._head_scale > 1e-6 else 0.3
            pitch_ratio = max(-0.3, min(0.6, pitch_ratio))
        else:
            pitch_ratio = 0.3  # neutral default

        # Expand when leaning forward (back of head exposed)
        pitch_factor = 1.0 + max(0.0, 0.3 - pitch_ratio) * 1.5

        # --- Step 6: Hair zone ellipse ---
        base_offset = self._head_scale * 0.7
        pitch_offset_bonus = max(0.0, 0.3 - pitch_ratio) * self._head_scale * 0.4

        self._hair_center = (
            self._head_center[0] + self._up_vec[0] * (base_offset + pitch_offset_bonus),
            self._head_center[1] + self._up_vec[1] * (base_offset + pitch_offset_bonus),
        )

        self._semi_width = self._head_scale * 1.0    # full head width along ear axis
        self._semi_height = self._head_scale * 1.2 * pitch_factor  # taller than wide (forehead to nape)

        # Detect head yaw: when ears are foreshortened relative to shoulders,
        # head is turned sideways. Forehead line becomes unreliable.
        if self._shoulder_distance > 0.01:
            self._yaw_ratio = ear_distance / self._shoulder_distance
        else:
            self._yaw_ratio = 0.45  # neutral default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contains(self, x: float, y: float) -> bool:
        """Return True if (x, y) is inside the rotated hair zone ellipse.

        Optionally also checks FaceMesh forehead boundary if available.
        """
        dx = x - self._hair_center[0]
        dy = y - self._hair_center[1]

        # Project onto ellipse axes
        proj_w = _dot((dx, dy), self._head_horizontal)
        proj_h = _dot((dx, dy), self._up_vec)

        sw = self._semi_width
        sh = self._semi_height

        if sw < 1e-9 or sh < 1e-9:
            return False

        in_ellipse = (proj_w / sw) ** 2 + (proj_h / sh) ** 2 <= 1.0

        if not in_ellipse:
            return False

        # Forehead filter: exclude face area, but only when facing camera.
        # When head is turned sideways (yaw_ratio < 0.3 = ears foreshortened),
        # forehead landmarks become unreliable — skip the filter.
        if self._face.detected and self._yaw_ratio > 0.3:
            if not self._face.is_above_forehead(x, y):
                return False

        return True

    @property
    def ellipse_params(self) -> dict:
        """Return ellipse parameters for visualization.

        Keys: center, semi_width, semi_height, head_horizontal, up_vec,
              head_center, forehead_line
        """
        return {
            "center": self._hair_center,
            "semi_width": self._semi_width,
            "semi_height": self._semi_height,
            "head_horizontal": self._head_horizontal,
            "up_vec": self._up_vec,
            "head_center": self._head_center,
        }

    @property
    def forehead_line(self) -> list[tuple[float, float]]:
        """Forehead contour points for visualization."""
        if self._face.detected:
            return self._face.forehead_line()
        return []

    # Keep backward-compatible bounds for status panel
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return approximate (x_min, x_max, y_ceiling, y_forehead_avg)."""
        cx, cy = self._hair_center
        return (
            cx - self._semi_width,
            cx + self._semi_width,
            cy - self._semi_height,
            cy,
        )


class HairZoneSmoother:
    """Applies temporal EMA smoothing to HairZone parameters across frames.

    Smooths hair_center, semi_width, semi_height, head_horizontal, and up_vec
    to eliminate per-frame jitter from landmark noise.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._prev_center: Optional[tuple[float, float]] = None
        self._prev_sw: Optional[float] = None
        self._prev_sh: Optional[float] = None
        self._prev_hh: Optional[tuple[float, float]] = None
        self._prev_uv: Optional[tuple[float, float]] = None

    def smooth(self, zone: HairZone) -> HairZone:
        """Apply EMA smoothing to the zone's ellipse parameters in-place."""
        a = self._alpha

        p = zone.ellipse_params
        cx, cy = p["center"]
        sw = p["semi_width"]
        sh = p["semi_height"]
        hh = p["head_horizontal"]
        uv = p["up_vec"]

        if self._prev_center is None:
            # First frame — no smoothing
            self._prev_center = (cx, cy)
            self._prev_sw = sw
            self._prev_sh = sh
            self._prev_hh = hh
            self._prev_uv = uv
            return zone

        # EMA on scalars and 2D vectors
        cx = a * cx + (1 - a) * self._prev_center[0]
        cy = a * cy + (1 - a) * self._prev_center[1]
        sw = a * sw + (1 - a) * self._prev_sw
        sh = a * sh + (1 - a) * self._prev_sh

        hh_x = a * hh[0] + (1 - a) * self._prev_hh[0]
        hh_y = a * hh[1] + (1 - a) * self._prev_hh[1]
        hh_len = math.sqrt(hh_x * hh_x + hh_y * hh_y)
        if hh_len > 1e-9:
            hh = (hh_x / hh_len, hh_y / hh_len)

        uv_x = a * uv[0] + (1 - a) * self._prev_uv[0]
        uv_y = a * uv[1] + (1 - a) * self._prev_uv[1]
        uv_len = math.sqrt(uv_x * uv_x + uv_y * uv_y)
        if uv_len > 1e-9:
            uv = (uv_x / uv_len, uv_y / uv_len)

        # Update zone internals with smoothed values
        zone._hair_center = (cx, cy)
        zone._semi_width = sw
        zone._semi_height = sh
        zone._head_horizontal = hh
        zone._up_vec = uv

        # Store for next frame
        self._prev_center = (cx, cy)
        self._prev_sw = sw
        self._prev_sh = sh
        self._prev_hh = hh
        self._prev_uv = uv

        return zone
