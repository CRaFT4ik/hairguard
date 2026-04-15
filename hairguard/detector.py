from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmark,
    RunningMode,
)

_KEY_LANDMARKS = {
    "left_wrist": PoseLandmark.LEFT_WRIST,
    "right_wrist": PoseLandmark.RIGHT_WRIST,
    "nose": PoseLandmark.NOSE,
    "left_ear": PoseLandmark.LEFT_EAR,
    "right_ear": PoseLandmark.RIGHT_EAR,
    "left_shoulder": PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
}

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
_MODEL_DIR = Path.home() / ".hairguard" / "models"
_MODEL_PATH = _MODEL_DIR / "pose_landmarker_full.task"


def _ensure_model() -> str:
    """Download the pose landmarker model if not cached."""
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[detector] Downloading pose landmarker model...")
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    print(f"[detector] Model saved to {_MODEL_PATH}")
    return str(_MODEL_PATH)


@dataclass
class Landmark:
    x: float          # normalized 0-1
    y: float          # normalized 0-1
    confidence: float  # visibility score


@dataclass
class PoseResult:
    landmarks: dict[str, Landmark]
    raw_landmarks: object
    detected: bool


class PoseDetector:
    """Wraps MediaPipe Tasks PoseLandmarker to detect body pose from BGR frames."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        model_path = _ensure_model()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_ts = 0

    def process(self, frame, timestamp_ms: int | None = None) -> PoseResult:
        """Run pose detection on a BGR frame and return a PoseResult.

        Parameters
        ----------
        frame:
            OpenCV BGR ndarray.
        timestamp_ms:
            Monotonic millisecond timestamp.  If None, time.monotonic()*1000 is used.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts = int(timestamp_ms) if timestamp_ms is not None else int(time.monotonic() * 1000)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return PoseResult(landmarks={}, raw_landmarks=None, detected=False)

        pose = result.pose_landmarks[0]  # first (only) detected pose
        landmarks: dict[str, Landmark] = {}
        for name, idx in _KEY_LANDMARKS.items():
            lm = pose[idx]
            landmarks[name] = Landmark(
                x=lm.x,
                y=lm.y,
                confidence=lm.visibility,
            )

        return PoseResult(landmarks=landmarks, raw_landmarks=result, detected=True)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
