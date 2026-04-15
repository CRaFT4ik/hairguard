from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp

# The six landmarks we care about, keyed by MediaPipe PoseLandmark name.
_KEY_LANDMARKS = (
    "left_wrist",
    "right_wrist",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
)


@dataclass
class Landmark:
    x: float          # normalized 0-1
    y: float          # normalized 0-1
    confidence: float # visibility score from MediaPipe


@dataclass
class PoseResult:
    landmarks: dict[str, Landmark]           # key landmarks only
    raw_landmarks: object                     # full MediaPipe NormalizedLandmarkList (or None)
    detected: bool                            # True when MediaPipe found a pose


class PoseDetector:
    """Wraps mediapipe.solutions.pose to detect body pose from BGR frames."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        # Build a lookup from lowercase landmark name -> enum value index
        self._name_to_index: dict[str, int] = {
            lm.name.lower(): lm.value
            for lm in mp.solutions.pose.PoseLandmark
        }

    def process(self, frame) -> PoseResult:
        """Run pose detection on a BGR frame and return a PoseResult."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        raw = result.pose_landmarks  # None if no pose detected

        if raw is None:
            return PoseResult(landmarks={}, raw_landmarks=None, detected=False)

        lm_list = raw.landmark
        landmarks: dict[str, Landmark] = {}
        for name in _KEY_LANDMARKS:
            idx = self._name_to_index.get(name)
            if idx is None:
                continue
            lm = lm_list[idx]
            landmarks[name] = Landmark(
                x=lm.x,
                y=lm.y,
                confidence=lm.visibility,
            )

        return PoseResult(landmarks=landmarks, raw_landmarks=raw, detected=True)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()
