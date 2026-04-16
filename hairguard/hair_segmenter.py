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
# HairSegmenter
# ---------------------------------------------------------------------------
# StableMaskTracker
# ---------------------------------------------------------------------------


class StableMaskTracker:
    """Maintains a stable hair mask that doesn't jitter when hands occlude hair.

    Strategy:
      - Continuously saves a "clean" snapshot of the mask every frame (when no hand near head).
        This snapshot is always 1+ frames behind — so when hand starts affecting landmarks,
        we already have the last clean state.
      - When hand is near head: use the frozen snapshot. Don't warp immediately — the snapshot
        is already at the right position. Only warp if nose moves significantly (real head movement).
      - When hand leaves: wait recovery_frames for segmenter to recover, then resume live.
    """

    _WARP_THRESHOLD = 0.02  # normalized nose displacement before we bother warping

    def __init__(self, recovery_frames: int = 3) -> None:
        self._clean_mask: Optional[np.ndarray] = None
        self._clean_nose: Optional[tuple[float, float]] = None
        self._frozen_mask: Optional[np.ndarray] = None  # snapshot at freeze moment
        self._frozen_nose: Optional[tuple[float, float]] = None
        self._recovery_frames = recovery_frames
        self._frames_since_hand_left = 999
        self._was_frozen = False

    def update(
        self,
        live_mask: HairMask,
        pose_landmarks: dict,
        hand_near_head: bool,
    ) -> HairMask:
        current_nose = self._get_nose(pose_landmarks)

        if hand_near_head:
            self._frames_since_hand_left = 0
        else:
            self._frames_since_hand_left += 1

        is_frozen = self._frames_since_hand_left <= self._recovery_frames

        # Always update clean snapshot when hands are away
        if not is_frozen:
            self._clean_mask = live_mask.get_mask().copy()
            self._clean_nose = current_nose
            self._was_frozen = False
            return live_mask

        # --- Frozen mode ---

        # First frame of freeze: capture the clean snapshot as our frozen baseline
        if not self._was_frozen:
            if self._clean_mask is not None:
                self._frozen_mask = self._clean_mask.copy()
                self._frozen_nose = self._clean_nose
            else:
                # No clean mask yet (hand was up from the start)
                self._frozen_mask = live_mask.get_mask().copy()
                self._frozen_nose = current_nose
            self._was_frozen = True

        if self._frozen_mask is None:
            return live_mask

        # Warp only if nose moved significantly (real head movement, not jitter)
        if current_nose and self._frozen_nose:
            dx = current_nose[0] - self._frozen_nose[0]
            dy = current_nose[1] - self._frozen_nose[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > self._WARP_THRESHOLD:
                warped = self._warp_by_nose(
                    self._frozen_mask, self._frozen_nose, current_nose,
                    live_mask.width, live_mask.height,
                )
                return HairMask(mask=warped, width=live_mask.width, height=live_mask.height)

        # No significant movement — use frozen mask as-is
        return HairMask(mask=self._frozen_mask, width=live_mask.width, height=live_mask.height)

    def _get_nose(self, lm: dict) -> Optional[tuple[float, float]]:
        nose = lm.get("nose")
        if nose is not None and (nose.confidence or 0) >= 0.5:
            return (nose.x, nose.y)
        return None

    def _warp_by_nose(
        self,
        mask: np.ndarray,
        src_nose: tuple[float, float],
        dst_nose: tuple[float, float],
        width: int,
        height: int,
    ) -> np.ndarray:
        import cv2
        tx = (dst_nose[0] - src_nose[0]) * width
        ty = (dst_nose[1] - src_nose[1]) * height
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(mask, M, (width, height), borderValue=0)


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
