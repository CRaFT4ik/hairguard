"""FaceDetector — MediaPipe FaceLandmarker wrapper + hairline boundary.

Public API
----------
FaceLandmark(x, y)          — single normalized landmark
FaceResult(landmarks, detected)
    .forehead_line()        — sorted list of (x, y) upper-arc points
    .forehead_y_at_x(x)     — linear interpolation along hairline
    .is_above_forehead(x,y) — True if point is above the hairline
FaceDetector(min_confidence) — detector wrapping MediaPipe FaceLandmarker
    .detect(frame, timestamp_ms) — returns FaceResult
    .close()                — release resources
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_DIR = Path.home() / ".hairguard" / "models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"


def _ensure_model() -> str:
    """Return local model path, downloading if necessary."""
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[face_detector] Downloading FaceLandmarker model…")
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    print(f"[face_detector] Model saved to {_MODEL_PATH}")
    return str(_MODEL_PATH)


# ---------------------------------------------------------------------------
# FOREHEAD_INDICES — upper arc of FACE_OVAL (hairline boundary)
# NOT pre-sorted by x; forehead_line() sorts them.
# ---------------------------------------------------------------------------

FOREHEAD_INDICES: list[int] = [
    21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FaceLandmark:
    """Single face mesh landmark, coordinates normalized 0-1."""
    x: float
    y: float


@dataclass
class FaceResult:
    """Result from one FaceDetector.detect() call.

    landmarks: mapping from MediaPipe landmark index → FaceLandmark
    detected:  False when no face was found
    """
    landmarks: dict[int, FaceLandmark]
    detected: bool

    # ------------------------------------------------------------------
    # Forehead / hairline helpers
    # ------------------------------------------------------------------

    def forehead_line(self) -> list[tuple[float, float]]:
        """Return the upper FACE_OVAL arc as (x, y) points sorted by x.

        Points are taken from FOREHEAD_INDICES. Any index absent from
        self.landmarks is silently skipped (graceful partial detection).
        Returns an empty list when no face was detected.
        """
        if not self.detected or not self.landmarks:
            return []

        points: list[tuple[float, float]] = []
        for idx in FOREHEAD_INDICES:
            lm = self.landmarks.get(idx)
            if lm is not None:
                points.append((lm.x, lm.y))

        points.sort(key=lambda p: p[0])
        return points

    def forehead_y_at_x(self, x: float) -> Optional[float]:
        """Linear interpolation of forehead y at the given x coordinate.

        Returns None if x is outside the forehead range or if no forehead
        points are available.
        """
        line = self.forehead_line()
        if len(line) < 2:
            return None

        x_min, x_max = line[0][0], line[-1][0]
        if x < x_min or x > x_max:
            return None

        # Find bracketing segment
        for i in range(len(line) - 1):
            x0, y0 = line[i]
            x1, y1 = line[i + 1]
            if x0 <= x <= x1:
                if x1 == x0:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        # Exact match with last point
        return line[-1][1]

    def is_above_forehead(self, x: float, y: float) -> bool:
        """Return True if (x, y) is above the hairline (y < forehead_y).

        In image coordinates, smaller y = higher in the image = above the
        hairline.  When x is outside the forehead range the nearest edge
        point's y is used as the threshold.  Returns False if no face is
        detected or no forehead points are available.
        """
        line = self.forehead_line()
        if not line:
            return False

        forehead_y = self.forehead_y_at_x(x)

        if forehead_y is None:
            # x is outside the covered range — clamp to nearest edge
            x_min = line[0][0]
            x_max = line[-1][0]
            if x < x_min:
                forehead_y = line[0][1]
            else:
                forehead_y = line[-1][1]

        return y < forehead_y


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------

class FaceDetector:
    """Wraps MediaPipe Tasks FaceLandmarker (VIDEO mode).

    Parameters
    ----------
    min_confidence:
        Minimum detection / tracking confidence threshold (0-1).
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        model_path = _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms: Optional[int] = None) -> FaceResult:
        """Run face detection on a BGR frame and return a FaceResult.

        Parameters
        ----------
        frame:
            BGR numpy array (e.g. from cv2.VideoCapture).
        timestamp_ms:
            Timestamp in milliseconds.  If omitted, time.monotonic() is used.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return FaceResult(landmarks={}, detected=False)

        raw_face = result.face_landmarks[0]  # first detected face
        landmarks: dict[int, FaceLandmark] = {
            idx: FaceLandmark(x=lm.x, y=lm.y)
            for idx, lm in enumerate(raw_face)
        }

        return FaceResult(landmarks=landmarks, detected=True)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
