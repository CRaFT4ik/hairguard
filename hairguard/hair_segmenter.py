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
