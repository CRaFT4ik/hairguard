"""HairSegmenter — MediaPipe hair segmentation model wrapper.

Wraps MediaPipe's hair_segmenter.tflite model to produce a per-pixel
hair mask from a camera frame.

Public API
----------
HairSegmenter()
    .segment(frame) -> HairMask
    .close()

HairMask(mask, width, height)
    .contains(x, y)    — True if normalized point falls on a hair pixel
    .get_mask()        — raw 2D numpy array (uint8, 1=hair)
    .detected          — True if any hair pixels are present

NoseAnchoredHairZone()
    .update_and_check(hair_mask, nose_px, fingertips_px) -> bool
    .is_learned        — True once enough clean frames have been seen
    .zone_bounds       — (offsets, frame_count) for debug display
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "hair_segmenter/float32/latest/hair_segmenter.tflite"
)
_MODEL_DIR = Path.home() / ".hairguard" / "models"
_MODEL_PATH = _MODEL_DIR / "hair_segmenter.tflite"


def _ensure_model() -> str:
    """Return local model path, downloading if necessary."""
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[hair_segmenter] Downloading hair segmenter model…")
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    print(f"[hair_segmenter] Model saved to {_MODEL_PATH}")
    return str(_MODEL_PATH)


# ---------------------------------------------------------------------------
# HairMask
# ---------------------------------------------------------------------------

class HairMask:
    """Binary hair pixel mask for one frame.

    Parameters
    ----------
    mask:
        2D uint8 numpy array (H x W) where non-zero = hair pixel.
    width:
        Frame width in pixels.
    height:
        Frame height in pixels.
    """

    def __init__(self, mask: np.ndarray, width: int, height: int) -> None:
        self._mask = mask
        self.width = width
        self.height = height
        self.detected: bool = bool(mask.any())

    def contains(self, x: float, y: float) -> bool:
        """Return True if normalized point (x, y) falls on a hair pixel.

        Coordinates are clamped to valid pixel range.
        """
        px = int(x * self.width)
        py = int(y * self.height)
        px = max(0, min(px, self.width - 1))
        py = max(0, min(py, self.height - 1))
        return bool(self._mask[py, px])

    def get_mask(self) -> np.ndarray:
        """Return raw 2D mask (uint8, non-zero = hair)."""
        return self._mask

    def dilate(self, pixels: int = 10) -> "HairMask":
        """Expand the hair mask outward by `pixels`. Returns new HairMask."""
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
        dilated = cv2.dilate(self._mask, kernel, iterations=1)
        return HairMask(mask=dilated, width=self.width, height=self.height)

    def subtract_face(self, nose_x: float, nose_y: float,
                      ear_left: tuple[float, float] | None = None,
                      ear_right: tuple[float, float] | None = None) -> "HairMask":
        """Remove face area from hair mask using pose landmarks.

        Builds an ellipse centered on the nose, sized from ear positions,
        and zeros out those pixels from the mask.
        """
        import cv2

        nose_px = int(nose_x * self.width)
        nose_py = int(nose_y * self.height)

        # Estimate face size from ears or fallback
        if ear_left is not None and ear_right is not None:
            ear_dist = ((ear_left[0] - ear_right[0]) * self.width) ** 2 + \
                       ((ear_left[1] - ear_right[1]) * self.height) ** 2
            ear_dist = ear_dist ** 0.5
            face_w = int(ear_dist * 0.30)  # face width (narrower than ear-to-ear)
            face_h = int(ear_dist * 0.35)  # face height (nose to chin approx)
        else:
            # Rough fallback
            face_w = int(self.width * 0.08)
            face_h = int(self.height * 0.10)

        if face_w < 5 or face_h < 5:
            return self

        result = self._mask.copy()
        cv2.ellipse(result, (nose_px, nose_py), (face_w, face_h), 0, 0, 360, 0, -1)
        return HairMask(mask=result, width=self.width, height=self.height)


# ---------------------------------------------------------------------------
# NoseAnchoredHairZone
# ---------------------------------------------------------------------------


class NoseAnchoredHairZone:
    """Learns hair extent relative to nose from clean mask frames.

    Continuously updates 4 offsets (top, bottom, left, right) from nose
    via exponential moving average. When hand is near hair and the mask is
    corrupted, falls back to the learned zone for containment checks.
    """

    ALPHA = 0.05        # EMA smoothing factor (slow adaptation)
    PROXIMITY_PX = 40   # pixels — if fingertip is this close to any hair pixel, mask is "near"

    def __init__(self) -> None:
        # Learned offsets from nose in PIXELS: (top, bottom, left, right)
        # top    = how many pixels above nose hair extends
        # bottom = how many pixels below nose hair extends
        # left   = how many pixels left of nose hair extends
        # right  = how many pixels right of nose hair extends
        self._offsets: Optional[tuple[float, float, float, float]] = None
        self._frame_count: int = 0

    def update_and_check(
        self,
        hair_mask: "HairMask",
        nose_px: Optional[tuple[float, float]],
        fingertips_px: list[tuple[float, float]],
    ) -> bool:
        """Main per-frame call. Returns True if ANY fingertip is on hair.

        Logic:
        1. Check if any fingertip is near a hair pixel (within PROXIMITY_PX)
        2. If NOT near: mask is clean -> update learned zone from mask -> use mask for check
        3. If near: mask may be corrupted -> DON'T update zone -> use BOTH mask AND learned zone
        """
        mask = hair_mask.get_mask()
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        h, w = mask.shape[:2]

        # Check proximity: are fingertips near any hair pixel?
        near_hair = self._any_near_hair(fingertips_px, mask, w, h)

        # Update learned zone only when mask is clean (fingers not near hair)
        if not near_hair and nose_px is not None:
            self._update_zone(mask, nose_px, w, h)

        # Check each fingertip
        for fx, fy in fingertips_px:
            fx = max(0, min(int(fx), w - 1))
            fy = max(0, min(int(fy), h - 1))

            # Primary: direct mask check (always try, works when mask is good)
            if mask[fy, fx] > 0:
                return True

            # Fallback: nose-anchored zone (useful when hand corrupts mask)
            # Only use fallback after zone has stabilized (>= 10 clean frames)
            if near_hair and self.is_learned and self._point_in_zone(fx, fy, nose_px):
                return True

        return False

    def _any_near_hair(
        self,
        fingertips_px: list[tuple[float, float]],
        mask: np.ndarray,
        w: int,
        h: int,
    ) -> bool:
        """Check if any fingertip is within PROXIMITY_PX of a hair pixel."""
        p = self.PROXIMITY_PX
        for fx, fy in fingertips_px:
            fx, fy = int(fx), int(fy)
            y0 = max(0, fy - p)
            y1 = min(h, fy + p)
            x0 = max(0, fx - p)
            x1 = min(w, fx + p)
            if mask[y0:y1, x0:x1].any():
                return True
        return False

    def _update_zone(
        self,
        mask: np.ndarray,
        nose_px: tuple[float, float],
        w: int,
        h: int,
    ) -> None:
        """Learn hair extent from clean mask via EMA."""
        ys, xs = np.where(mask > 0)
        if len(ys) < 100:
            return  # not enough hair pixels

        nx, ny = int(nose_px[0]), int(nose_px[1])
        offsets = (
            float(ny - ys.min()),   # top: nose_y - hair_top
            float(ys.max() - ny),   # bottom: hair_bottom - nose_y
            float(nx - xs.min()),   # left: nose_x - hair_left
            float(xs.max() - nx),   # right: hair_right - nose_x
        )

        if self._offsets is None:
            self._offsets = offsets
        else:
            a = self.ALPHA
            self._offsets = tuple(
                a * new + (1 - a) * old
                for new, old in zip(offsets, self._offsets)
            )
        self._frame_count += 1

    def _point_in_zone(
        self,
        px: int,
        py: int,
        nose_px: Optional[tuple[float, float]],
    ) -> bool:
        """Check if pixel (px, py) is inside the learned zone."""
        if self._offsets is None or nose_px is None:
            return False
        nx, ny = int(nose_px[0]), int(nose_px[1])
        top, bottom, left, right = self._offsets
        return (
            ny - top <= py <= ny + bottom
            and nx - left <= px <= nx + right
        )

    @property
    def is_learned(self) -> bool:
        return self._offsets is not None and self._frame_count >= 10

    @property
    def zone_bounds(self) -> tuple[Optional[tuple[float, float, float, float]], int]:
        """Return (offsets, frame_count) for debug display."""
        return self._offsets, self._frame_count


# ---------------------------------------------------------------------------
# HairSegmenter
# ---------------------------------------------------------------------------


class HairSegmenter:
    """Wraps MediaPipe Tasks ImageSegmenter with the hair_segmenter model.

    Uses IMAGE running mode (stateless, no timestamp required).
    The model outputs confidence masks (float 0-1); values >= 0.5 are
    treated as hair pixels.
    """

    # Hair channel index in the model's output confidence masks.
    # The hair_segmenter model has 2 channels: 0=background, 1=hair.
    _HAIR_CHANNEL = 1

    def __init__(self) -> None:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            ImageSegmenter,
            ImageSegmenterOptions,
            RunningMode,
        )

        self._cv2 = cv2
        self._mp = mp

        model_path = _ensure_model()
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = ImageSegmenter.create_from_options(options)

    def segment(self, frame) -> HairMask:
        """Segment hair in a BGR frame and return a HairMask.

        Parameters
        ----------
        frame:
            BGR numpy array from cv2 (e.g. from CameraManager.read_frame()).

        Returns
        -------
        HairMask
            Binary hair pixel mask at original frame resolution.
        """
        cv2 = self._cv2
        mp = self._mp

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._segmenter.segment(mp_image)

        # result.confidence_masks is a list of mp.Image objects, one per class.
        # Channel 1 = hair confidence (float32, values 0-1).
        if (
            result.confidence_masks is not None
            and len(result.confidence_masks) > self._HAIR_CHANNEL
        ):
            hair_conf = result.confidence_masks[self._HAIR_CHANNEL].numpy_view()
            # Threshold at 0.5 → binary mask, resize to original frame size
            binary = (hair_conf >= 0.5).astype(np.uint8)
            if binary.shape[0] != h or binary.shape[1] != w:
                binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            # Fallback: empty mask (no detection)
            binary = np.zeros((h, w), dtype=np.uint8)

        return HairMask(mask=binary, width=w, height=h)

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._segmenter.close()
        except Exception:
            pass
