"""hairguard.hand_detector — MediaPipe HandLandmarker wrapper (Tasks API, VIDEO mode).

Detects hands in BGR frames and returns HandResult objects with 21 landmarks.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Landmark index constants
# ---------------------------------------------------------------------------

THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20
MIDDLE_MCP  = 9
RING_MCP    = 13
PINKY_MCP   = 17
WRIST       = 0

_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_NAME = "hand_landmarker.task"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HandLandmark:
    """A single hand landmark in normalized image coordinates."""
    x: float
    y: float
    z: float
    presence: float = 1.0


@dataclass
class HandResult:
    """One detected hand with 21 landmarks indexed by MediaPipe landmark ID."""

    landmarks: dict[int, HandLandmark]
    handedness: str   # "Left" or "Right"
    score: float      # detection confidence

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def thumb_tip(self) -> HandLandmark:
        return self.landmarks[THUMB_TIP]

    def index_tip(self) -> HandLandmark:
        return self.landmarks[INDEX_TIP]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def hand_scale(self) -> float:
        """Euclidean distance from wrist[0] to middle_mcp[9].

        Provides a stable, scale-invariant reference length for this hand.
        """
        w  = self.landmarks[WRIST]
        m  = self.landmarks[MIDDLE_MCP]
        dx = m.x - w.x
        dy = m.y - w.y
        dz = m.z - w.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def pinch_ratio(self) -> float:
        """Ratio of thumb-tip to index-tip distance over hand_scale.

        Smaller values mean thumb and index are closer together (pinching).
        """
        t  = self.thumb_tip()
        i  = self.index_tip()
        dx = t.x - i.x
        dy = t.y - i.y
        dz = t.z - i.z
        dist  = math.sqrt(dx * dx + dy * dy + dz * dz)
        scale = self.hand_scale()
        if scale == 0.0:
            return 0.0
        return dist / scale

    def is_pinching(self, threshold: float = 0.4) -> bool:
        """Return True when thumb and index tips are close enough to be pinching.

        Both landmarks must have presence >= 0.5 to be considered reliable.
        """
        t = self.thumb_tip()
        i = self.index_tip()
        if (t.presence or 0.0) < 0.5 or (i.presence or 0.0) < 0.5:
            return False
        return self.pinch_ratio() < threshold

    def is_gripping(self, curl_threshold: int = 2) -> bool:
        """Return True when at least curl_threshold fingers are curled.

        Curl detection uses hand-relative coordinates: each finger tip and its
        MCP joint are projected onto the wrist->middle_mcp axis.  If the tip
        projection is less than the MCP projection the finger is curled (tip is
        closer to the wrist than the MCP along the hand axis).

        Only middle, ring and pinky fingers are checked (index and thumb are
        intentionally excluded so a grip can coexist with a pinch gesture).
        """
        curl_pairs = [
            (MIDDLE_TIP, MIDDLE_MCP),
            (RING_TIP,   RING_MCP),
            (PINKY_TIP,  PINKY_MCP),
        ]

        w  = self.landmarks[WRIST]
        m  = self.landmarks[MIDDLE_MCP]

        # Unit vector along wrist→middle_mcp axis
        ax = m.x - w.x
        ay = m.y - w.y
        az = m.z - w.z
        axis_len = math.sqrt(ax * ax + ay * ay + az * az)
        if axis_len == 0.0:
            return False
        ax /= axis_len
        ay /= axis_len
        az /= axis_len

        def _proj(lm: HandLandmark) -> float:
            """Scalar projection of landmark (relative to wrist) onto axis."""
            rx = lm.x - w.x
            ry = lm.y - w.y
            rz = lm.z - w.z
            return rx * ax + ry * ay + rz * az

        curled = 0
        for tip_idx, mcp_idx in curl_pairs:
            tip = self.landmarks[tip_idx]
            mcp = self.landmarks[mcp_idx]
            if _proj(tip) < _proj(mcp):
                curled += 1

        return curled >= curl_threshold


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def _ensure_model() -> str:
    """Local path to the hand landmarker model. See ``hairguard.assets``."""
    from hairguard import assets

    return assets.ensure_task_file(_MODEL_NAME, _MODEL_URL)


# ---------------------------------------------------------------------------
# HandDetector
# ---------------------------------------------------------------------------

class HandDetector:
    """Wraps MediaPipe Tasks HandLandmarker to detect hands from BGR frames.

    Usage::

        detector = HandDetector(max_hands=2, min_confidence=0.5)
        results  = detector.detect(bgr_frame)
        detector.close()
    """

    def __init__(self, max_hands: int = 2, min_confidence: float = 0.5) -> None:
        # Defer heavy imports so the module is importable without mediapipe
        # installed (e.g. in CI environments that only run pure-Python tests).
        import cv2  # noqa: F401  — validated here, used in detect()
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker as _HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        self._cv2 = cv2
        self._mp  = mp

        model_path = _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = _HandLandmarker.create_from_options(options)
        self._last_ts = 0

    def detect(self, frame, timestamp_ms: int | None = None) -> list[HandResult]:
        """Detect hands in a BGR frame and return a list of HandResult objects.

        Parameters
        ----------
        frame:
            OpenCV BGR ndarray.
        timestamp_ms:
            Monotonic millisecond timestamp.  If None, time.monotonic()*1000 is used.
        """
        cv2 = self._cv2
        mp  = self._mp

        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        # Ensure strictly monotonic
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        ts_ms = timestamp_ms

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        raw      = self._landmarker.detect_for_video(mp_image, ts_ms)

        results: list[HandResult] = []

        if not raw.hand_landmarks:
            return results

        for hand_lms, hand_world_lms, classifications in zip(
            raw.hand_landmarks,
            raw.hand_world_landmarks,
            raw.handedness,
        ):
            lm_dict: dict[int, HandLandmark] = {}
            for idx, lm in enumerate(hand_lms):
                lm_dict[idx] = HandLandmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    presence=getattr(lm, "presence", None) or getattr(lm, "visibility", None) or 1.0,
                )

            top = classifications[0]
            results.append(HandResult(
                landmarks=lm_dict,
                handedness=top.display_name,
                score=top.score,
            ))

        return results

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
