# HairGuard v2: Precision Hair-Touching Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the wrist-proximity detection (v1) with a multi-layered system that triggers ONLY when fingertips (thumb + index) are physically on the hair — not on the face, not holding an object, not just near the head.

**Architecture:** Three MediaPipe models run per frame: PoseLandmarker (body/head position), HandLandmarker (21-point hand skeleton with accurate fingertips), and FaceLandmarker (468-point face mesh for precise forehead/hairline boundary). A 7-layer analyzer pipeline filters: hand detected → head located → fingertips above hairline → pinch gesture → fingers not gripping an object → debounce. Only when ALL layers pass does the alert trigger.

**Tech Stack:** Python 3.11+, MediaPipe Tasks API (PoseLandmarker full, HandLandmarker, FaceLandmarker), OpenCV, existing HairGuard infrastructure (overlay, stats, tray, CLI)

**Branch:** `v2/hand-detection` (already created)

---

## Known Limitations

**Back-of-head pulling (occipital region):** When the user reaches behind their head, the camera cannot see the fingers and HandLandmarker loses tracking entirely. This is an inherent limitation of a single-camera frontal setup. Mitigation: if Pose detects a wrist near/above the head but HandLandmarker returns no hands, treat this as a supplementary warning signal (lower confidence, longer debounce). This is documented as a future enhancement, not a v2 blocker.

---

## Research Basis

This plan is based on three independent research investigations:

1. **Existing trichotillomania CV solutions:** No open-source solution exists. Commercial apps (HabitCam, TricTrak) are closed-source. Roboflow blog describes a YOLO-based approach with "hard negative class" training. Wearable solutions (HabitAware Keen) have high false-positive rates.

2. **MediaPipe Hands accuracy:** HandLandmarker (Tasks API) gives accurate fingertip positions (landmarks 4=thumb tip, 8=index tip). The model hallucinates positions for occluded fingers — must filter by `presence_score`. Holistic is deprecated; separate models are more reliable. Running Hands + Pose + FaceMesh totals ~53ms per frame at 10 FPS.

3. **Pinch/gesture detection:** Pinch = distance(landmark 4, landmark 8) < threshold (~0.05 normalized). Not built into MediaPipe — trivial geometry. Distinguishing pinch-on-hair from holding-an-object: check if middle/ring/pinky fingers are relaxed (extended) vs. curled (gripping). Temporal rotation signal (wrist micro-rotations over 10-20 frames) adds accuracy for twisting detection.

---

## File Structure

### Files to CREATE (new):

| File | Responsibility |
|---|---|
| `hairguard/hand_detector.py` | Wraps MediaPipe HandLandmarker. Detects hands, returns 21 landmarks per hand with confidence. Provides fingertip positions, pinch detection, finger curl state. |
| `hairguard/face_detector.py` | Wraps MediaPipe FaceLandmarker. Returns face mesh landmarks. Computes hairline boundary (top arc of FACE_OVAL contour). |
| `hairguard/head_zone.py` | Combines Pose + FaceMesh data to define "hair zone" — the region above the forehead line, between ears, extended to crown. Handles head tilt by using FaceMesh forehead contour which tracks with head orientation. |
| `tests/test_analyzer_v2.py` | Unit tests for the new 7-layer analyzer |
| `tests/test_hand_detector.py` | Unit tests for hand detection + pinch + curl |
| `tests/test_head_zone.py` | Unit tests for hair zone geometry |

### Files to MODIFY (existing):

| File | Changes |
|---|---|
| `hairguard/detector.py` | Keep PoseLandmarker for body/shoulders. No changes needed — it already returns nose, ears, shoulders, wrists. |
| `hairguard/analyzer.py` | Complete rewrite of `_detect()`. New 7-layer pipeline. State machine (IDLE/TRIGGERED/ALERTING/COOLDOWN) stays the same. |
| `hairguard/config.py` | Add new config fields: `detection.pinch_threshold`, `detection.hair_zone_offset`, `detection.grip_curl_threshold`, `detection.models` (pose/hand/face model selection). |
| `hairguard/app.py` | Initialize HandDetector + FaceDetector alongside PoseDetector. Pass all three results to analyzer. Update tick to run all three models. |
| `hairguard/debugview.py` | Draw 21-point hand skeleton with connections, hair zone boundary, pinch indicator, curl state per finger, fingertip-in-zone indicator. |
| `pyproject.toml` | No new deps — `mediapipe` already includes all three models. |

### Files UNCHANGED:

`camera.py`, `overlay.py`, `phrases.py`, `stats.py`, `db.py`, `tray.py`, `pid.py`, `autostart.py`, `cli.py` (except adding `debug` subcommand if not already present)

---

## Model Details

### HandLandmarker

- **Model:** `hand_landmarker.task` (~5.5MB, auto-downloaded on first run)
- **URL:** `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
- **Key landmarks:**
  - `THUMB_TIP` = index 4
  - `INDEX_FINGER_TIP` = index 8
  - `MIDDLE_FINGER_TIP` = index 12
  - `RING_FINGER_TIP` = index 16
  - `PINKY_TIP` = index 20
  - `THUMB_MCP` = index 2 (for curl baseline)
  - `INDEX_FINGER_MCP` = index 5
  - `MIDDLE_FINGER_MCP` = index 9
  - `RING_FINGER_MCP` = index 13
  - `PINKY_MCP` = index 17
  - `WRIST` = index 0

### FaceLandmarker

- **Model:** `face_landmarker.task` (~5MB, auto-downloaded)
- **URL:** `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- **Key landmarks for hairline:** FACE_OVAL upper arc — indices 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
  - The **top segment** of this oval (roughly indices 10, 338, 297, 332, 284, 251, 389 and their mirror) defines the forehead-hairline boundary
  - Specifically: landmarks around index 10 (top center of forehead), 338 (right forehead), 109 (left forehead) form the upper contour

### PoseLandmarker (existing)

- **Model:** `pose_landmarker_full.task` (already downloaded, 9MB)
- **Used for:** shoulders (normalization), ears (lateral head bounds), nose (depth reference), wrists (fallback if HandLandmarker loses tracking)

---

## Detection Pipeline: 7 Layers

```
Frame
  │
  ├─→ HandLandmarker ─→ hands[] with 21 landmarks each
  ├─→ PoseLandmarker ─→ pose landmarks (shoulders, ears, nose, wrists)
  └─→ FaceLandmarker ─→ face mesh (468 points, FACE_OVAL contour)
        │
        ▼
Layer 1: Hand present?
  │ HandLandmarker returned at least one hand with sufficient confidence?
  │ If NO → IDLE, skip remaining layers
  ▼
Layer 2: Head located?
  │ PoseLandmarker found ears + nose with confidence >= threshold?
  │ FaceLandmarker found face mesh?
  │ If NO → IDLE, skip remaining layers
  ▼
Layer 3: Compute hair zone
  │ Top of FACE_OVAL contour = forehead line
  │ Hair zone = region ABOVE forehead line, bounded horizontally by ears
  │ Extended upward by hair_zone_offset * shoulder_width
  │ This zone adapts to head tilt because FaceMesh tracks the face
  ▼
Layer 4: Fingertips in hair zone?
  │ Are THUMB_TIP (4) and/or INDEX_FINGER_TIP (8) inside the hair zone polygon?
  │ Use presence_score >= 0.7 filter (low = hallucinated position)
  │ If NO fingertips in hair zone → IDLE
  ▼
Layer 5: Pinch detected?
  │ Compute hand_scale = distance(WRIST[0], MIDDLE_MCP[9]) — stable reference for hand size
  │ pinch_ratio = distance(THUMB_TIP, INDEX_FINGER_TIP) / hand_scale
  │ If pinch_ratio < pinch_threshold (default 0.4) → pinching
  │ Normalizing by hand_scale makes threshold independent of camera distance
  │ If not pinching (open hand near head, just resting) → IDLE
  ▼
Layer 6: Not gripping an object?
  │ Check middle, ring, pinky finger curl using HAND-RELATIVE coordinates:
  │   Compute finger_axis = vector from WRIST[0] to MIDDLE_MCP[9]
  │   For each finger: project TIP and MCP onto finger_axis
  │   If TIP projection < MCP projection (tip is closer to wrist along axis) → curled
  │   This works regardless of hand orientation (reaching up, sideways, etc.)
  │ If 2+ of {middle, ring, pinky} are curled → gripping an object
  │ When twisting hair: other fingers are typically relaxed/extended
  │ When holding comb/pen: other fingers wrap around the handle (curled)
  │ If gripping detected → IDLE
  ▼
Layer 7: Debounce
  │ State machine: IDLE → TRIGGERED → ALERTING → COOLDOWN → IDLE
  │ TRIGGERED → ALERTING after trigger_delay_sec (default 0.8s)
  │ ALERTING → COOLDOWN when layers 1-6 stop passing
  │ COOLDOWN → IDLE after dismiss_delay_sec (default 2.5s)
  │ Same state machine as v1
  ▼
ALERT (show overlay)
```

---

## Updated Config

```yaml
camera: 0
detection:
  # Timing
  trigger_delay_sec: 0.8
  dismiss_delay_sec: 2.5
  fps: 10
  min_confidence: 0.5

  # Pinch detection
  pinch_threshold: 0.4           # ratio of thumb-index distance to hand scale (wrist-to-middle-MCP)
  min_presence_score: 0.7        # minimum landmark presence to trust position

  # Hair zone
  hair_zone_offset: 0.4          # how far above forehead to extend hair zone (fraction of shoulder width)

  # Grip filter
  grip_curl_threshold: 2         # if this many of {middle, ring, pinky} are curled → gripping an object

alert:
  opacity: 0.7
  text_color: "#ffffff"
pause:
  auto_resume_minutes: 15
```

Old v1 fields removed: `distance_threshold`, `fallback_frame_percent`.

---

## Tasks

### Task 1: HandDetector — MediaPipe HandLandmarker wrapper

**Files:**
- Create: `hairguard/hand_detector.py`
- Create: `tests/test_hand_detector.py`

- [ ] **Step 1: Write failing test for HandResult dataclass**

```python
# tests/test_hand_detector.py
from hairguard.hand_detector import HandResult, HandLandmark

def test_hand_result_creation():
    landmarks = {i: HandLandmark(x=0.5, y=0.5, z=0.0, presence=0.9) for i in range(21)}
    result = HandResult(landmarks=landmarks, handedness="Right", score=0.95)
    assert result.handedness == "Right"
    assert len(result.landmarks) == 21
    assert result.landmarks[4].presence == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hand_detector.py::test_hand_result_creation -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hairguard.hand_detector'`

- [ ] **Step 3: Write failing test for pinch detection**

```python
# tests/test_hand_detector.py (append)
def test_is_pinching_true():
    """Thumb tip and index tip close together relative to hand size = pinching."""
    landmarks = {i: HandLandmark(x=0.5, y=0.5, z=0.0, presence=0.9) for i in range(21)}
    landmarks[0] = HandLandmark(x=0.50, y=0.60, z=0.0, presence=0.95)  # wrist
    landmarks[9] = HandLandmark(x=0.50, y=0.40, z=0.0, presence=0.95)  # middle MCP (hand_scale ref)
    landmarks[4] = HandLandmark(x=0.50, y=0.30, z=0.0, presence=0.95)  # thumb tip
    landmarks[8] = HandLandmark(x=0.52, y=0.31, z=0.0, presence=0.95)  # index tip — close to thumb
    result = HandResult(landmarks=landmarks, handedness="Right", score=0.95)
    # pinch_ratio = ~0.02 / 0.20 = ~0.1, well below 0.4
    assert result.is_pinching(threshold=0.4) is True

def test_is_pinching_false_open_hand():
    """Thumb and index far apart relative to hand size = not pinching."""
    landmarks = {i: HandLandmark(x=0.5, y=0.5, z=0.0, presence=0.9) for i in range(21)}
    landmarks[0] = HandLandmark(x=0.50, y=0.60, z=0.0, presence=0.95)  # wrist
    landmarks[9] = HandLandmark(x=0.50, y=0.40, z=0.0, presence=0.95)  # middle MCP
    landmarks[4] = HandLandmark(x=0.30, y=0.30, z=0.0, presence=0.95)  # thumb tip — far left
    landmarks[8] = HandLandmark(x=0.70, y=0.20, z=0.0, presence=0.95)  # index tip — far right
    result = HandResult(landmarks=landmarks, handedness="Right", score=0.95)
    # pinch_ratio = ~0.41 / 0.20 = ~2.05, well above 0.4
    assert result.is_pinching(threshold=0.4) is False
```

- [ ] **Step 4: Write failing test for grip detection**

```python
# tests/test_hand_detector.py (append)
def test_is_gripping_true():
    """Middle, ring, pinky curled (tip closer to wrist along finger axis) = gripping."""
    landmarks = {i: HandLandmark(x=0.5, y=0.5, z=0.0, presence=0.9) for i in range(21)}
    # Wrist at bottom, middle MCP above — finger axis points upward
    landmarks[0] = HandLandmark(x=0.5, y=0.70, z=0.0, presence=0.9)   # wrist
    landmarks[9] = HandLandmark(x=0.5, y=0.50, z=0.0, presence=0.9)   # middle MCP
    # Curled: tips are closer to wrist than their MCPs along the axis
    landmarks[12] = HandLandmark(x=0.5, y=0.58, z=0.0, presence=0.9)  # middle tip (curled back toward wrist)
    landmarks[13] = HandLandmark(x=0.45, y=0.50, z=0.0, presence=0.9) # ring MCP
    landmarks[16] = HandLandmark(x=0.45, y=0.58, z=0.0, presence=0.9) # ring tip (curled)
    landmarks[17] = HandLandmark(x=0.55, y=0.50, z=0.0, presence=0.9) # pinky MCP
    landmarks[20] = HandLandmark(x=0.55, y=0.58, z=0.0, presence=0.9) # pinky tip (curled)
    result = HandResult(landmarks=landmarks, handedness="Right", score=0.95)
    assert result.is_gripping(curl_threshold=2) is True

def test_is_gripping_false_relaxed():
    """Middle, ring, pinky extended (tip farther from wrist along axis) = relaxed."""
    landmarks = {i: HandLandmark(x=0.5, y=0.5, z=0.0, presence=0.9) for i in range(21)}
    landmarks[0] = HandLandmark(x=0.5, y=0.70, z=0.0, presence=0.9)   # wrist
    landmarks[9] = HandLandmark(x=0.5, y=0.50, z=0.0, presence=0.9)   # middle MCP
    # Extended: tips are farther from wrist than MCPs
    landmarks[12] = HandLandmark(x=0.5, y=0.30, z=0.0, presence=0.9)  # middle tip (extended)
    landmarks[13] = HandLandmark(x=0.45, y=0.50, z=0.0, presence=0.9) # ring MCP
    landmarks[16] = HandLandmark(x=0.45, y=0.30, z=0.0, presence=0.9) # ring tip (extended)
    landmarks[17] = HandLandmark(x=0.55, y=0.50, z=0.0, presence=0.9) # pinky MCP
    landmarks[20] = HandLandmark(x=0.55, y=0.30, z=0.0, presence=0.9) # pinky tip (extended)
    result = HandResult(landmarks=landmarks, handedness="Right", score=0.95)
    assert result.is_gripping(curl_threshold=2) is False
```

- [ ] **Step 5: Implement hand_detector.py**

```python
# hairguard/hand_detector.py
from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
_MODEL_DIR = Path.home() / ".hairguard" / "models"
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"

# Finger tip and MCP (knuckle) indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17


def _ensure_model() -> str:
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[hand_detector] Downloading hand landmarker model...")
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    print(f"[hand_detector] Model saved to {_MODEL_PATH}")
    return str(_MODEL_PATH)


@dataclass
class HandLandmark:
    x: float           # normalized 0-1
    y: float           # normalized 0-1
    z: float           # relative depth
    presence: float    # presence score (0-1, how likely this landmark is visible)


@dataclass
class HandResult:
    landmarks: dict[int, HandLandmark]   # index 0-20 -> landmark
    handedness: str                       # "Left" or "Right"
    score: float                          # detection confidence

    def thumb_tip(self) -> HandLandmark:
        return self.landmarks[THUMB_TIP]

    def index_tip(self) -> HandLandmark:
        return self.landmarks[INDEX_TIP]

    def hand_scale(self) -> float:
        """Distance from wrist to middle MCP — stable reference for hand size."""
        w = self.landmarks[0]   # WRIST
        m = self.landmarks[MIDDLE_MCP]
        dist = math.sqrt((w.x - m.x) ** 2 + (w.y - m.y) ** 2)
        return max(dist, 1e-6)  # avoid division by zero

    def pinch_ratio(self) -> float:
        """Thumb-index distance normalized by hand scale. Smaller = closer to pinch."""
        t = self.landmarks[THUMB_TIP]
        i = self.landmarks[INDEX_TIP]
        raw = math.sqrt((t.x - i.x) ** 2 + (t.y - i.y) ** 2)
        return raw / self.hand_scale()

    def is_pinching(self, threshold: float = 0.4) -> bool:
        """True if thumb tip and index tip are close relative to hand size."""
        t = self.landmarks[THUMB_TIP]
        i = self.landmarks[INDEX_TIP]
        if t.presence < 0.5 or i.presence < 0.5:
            return False
        return self.pinch_ratio() < threshold

    def is_gripping(self, curl_threshold: int = 2) -> bool:
        """True if curl_threshold or more of {middle, ring, pinky} are curled.

        Uses hand-relative coordinates: projects fingertip and MCP onto the
        wrist-to-middle-MCP axis. If tip projection < MCP projection (tip is
        closer to wrist), the finger is curled. Works regardless of hand orientation.
        """
        w = self.landmarks[0]  # WRIST
        m = self.landmarks[MIDDLE_MCP]
        # Finger axis: wrist -> middle MCP
        ax = m.x - w.x
        ay = m.y - w.y
        axis_len_sq = ax * ax + ay * ay
        if axis_len_sq < 1e-12:
            return False

        curl_pairs = [
            (MIDDLE_TIP, MIDDLE_MCP),
            (RING_TIP, RING_MCP),
            (PINKY_TIP, PINKY_MCP),
        ]
        curled = 0
        for tip_idx, mcp_idx in curl_pairs:
            tip = self.landmarks.get(tip_idx)
            mcp = self.landmarks.get(mcp_idx)
            if tip is None or mcp is None:
                continue
            if tip.presence < 0.5 or mcp.presence < 0.5:
                continue
            # Project onto wrist->middle_mcp axis
            tip_proj = ((tip.x - w.x) * ax + (tip.y - w.y) * ay) / axis_len_sq
            mcp_proj = ((mcp.x - w.x) * ax + (mcp.y - w.y) * ay) / axis_len_sq
            if tip_proj < mcp_proj:  # tip closer to wrist along axis = curled
                curled += 1
        return curled >= curl_threshold


class HandDetector:
    """Wraps MediaPipe HandLandmarker to detect hands and fingertips."""

    def __init__(self, max_hands: int = 2, min_confidence: float = 0.5) -> None:
        model_path = _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts = 0

    def detect(self, frame) -> list[HandResult]:
        """Detect hands in a BGR frame. Returns list of HandResult (0-2 hands)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts = int(time.monotonic() * 1000)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        hands: list[HandResult] = []
        if not result.hand_landmarks:
            return hands

        for i, hand_lms in enumerate(result.hand_landmarks):
            landmarks = {}
            for idx, lm in enumerate(hand_lms):
                landmarks[idx] = HandLandmark(
                    x=lm.x, y=lm.y, z=lm.z,
                    presence=lm.presence if hasattr(lm, 'presence') else lm.visibility,
                )
            handedness = "Unknown"
            if result.handedness and i < len(result.handedness):
                handedness = result.handedness[i][0].category_name
            score = result.handedness[i][0].score if result.handedness and i < len(result.handedness) else 0.0
            hands.append(HandResult(landmarks=landmarks, handedness=handedness, score=score))

        return hands

    def close(self) -> None:
        self._landmarker.close()
```

- [ ] **Step 6: Run all tests to verify they pass**

Run: `python -m pytest tests/test_hand_detector.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add hairguard/hand_detector.py tests/test_hand_detector.py
git commit -m "Task 1: Add HandDetector with pinch and grip detection"
```

---

### Task 2: FaceDetector — MediaPipe FaceLandmarker wrapper + hairline boundary

**Files:**
- Create: `hairguard/face_detector.py`
- Create: `tests/test_face_detector.py`

- [ ] **Step 1: Write failing test for FaceResult**

```python
# tests/test_face_detector.py
from hairguard.face_detector import FaceResult, FaceLandmark

def test_face_result_creation():
    landmarks = {i: FaceLandmark(x=0.5, y=0.5) for i in range(468)}
    result = FaceResult(landmarks=landmarks, detected=True)
    assert result.detected is True
    assert len(result.landmarks) == 468
```

- [ ] **Step 2: Write failing test for hairline boundary**

```python
# tests/test_face_detector.py (append)
def test_forehead_line_basic():
    """Forehead line should be a list of (x, y) points above the face center."""
    # Create minimal face landmarks with known forehead positions
    landmarks = {i: FaceLandmark(x=0.5, y=0.5) for i in range(468)}
    # Set forehead-top landmarks to known positions
    # Index 10 = top center of forehead
    landmarks[10] = FaceLandmark(x=0.50, y=0.20)
    # Index 338 = right forehead
    landmarks[338] = FaceLandmark(x=0.60, y=0.22)
    # Index 109 = left forehead
    landmarks[109] = FaceLandmark(x=0.40, y=0.22)
    result = FaceResult(landmarks=landmarks, detected=True)
    line = result.forehead_line()
    assert len(line) > 0
    # All points should have y < 0.5 (upper half of face)
    for x, y in line:
        assert y < 0.5

def test_point_above_forehead():
    """A point above the forehead line should be in hair zone."""
    landmarks = {i: FaceLandmark(x=0.5, y=0.5) for i in range(468)}
    landmarks[10] = FaceLandmark(x=0.50, y=0.30)   # top center forehead
    landmarks[338] = FaceLandmark(x=0.60, y=0.32)
    landmarks[109] = FaceLandmark(x=0.40, y=0.32)
    result = FaceResult(landmarks=landmarks, detected=True)
    # Point clearly above forehead
    assert result.is_above_forehead(0.50, 0.15) is True
    # Point clearly below forehead (on face)
    assert result.is_above_forehead(0.50, 0.50) is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_face_detector.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement face_detector.py**

```python
# hairguard/face_detector.py
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

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_MODEL_DIR = Path.home() / ".hairguard" / "models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"

# FACE_OVAL upper arc landmark indices (top of forehead contour).
# Full upper arc from left temple across the top to right temple.
# These are NOT pre-sorted by x — forehead_line() sorts them.
_FOREHEAD_INDICES = [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]


def _ensure_model() -> str:
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[face_detector] Downloading face landmarker model...")
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    print(f"[face_detector] Model saved to {_MODEL_PATH}")
    return str(_MODEL_PATH)


@dataclass
class FaceLandmark:
    x: float  # normalized 0-1
    y: float  # normalized 0-1


@dataclass
class FaceResult:
    landmarks: dict[int, FaceLandmark]  # index 0-467 -> landmark
    detected: bool

    def forehead_line(self) -> list[tuple[float, float]]:
        """Return the forehead/hairline boundary as a list of (x, y) points.

        These points trace the top edge of the face oval — where the
        forehead ends and the hair begins. Sorted left to right.
        """
        points = []
        for idx in _FOREHEAD_INDICES:
            lm = self.landmarks.get(idx)
            if lm is not None:
                points.append((lm.x, lm.y))
        # Sort by x for reliable interpolation regardless of index order
        points.sort(key=lambda p: p[0])
        return points

    def forehead_y_at_x(self, x: float) -> Optional[float]:
        """Interpolate the forehead y-coordinate at a given x position.

        Returns the y value of the forehead line at horizontal position x,
        or None if x is outside the forehead range. Uses linear interpolation
        between the nearest forehead landmarks.
        """
        line = self.forehead_line()
        if len(line) < 2:
            return None

        # Find the two points that bracket x
        for i in range(len(line) - 1):
            x1, y1 = line[i]
            x2, y2 = line[i + 1]
            if x1 <= x <= x2 or x2 <= x <= x1:
                # Linear interpolation
                if abs(x2 - x1) < 1e-6:
                    return y1
                t = (x - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)

        return None

    def is_above_forehead(self, x: float, y: float) -> bool:
        """Return True if (x, y) is above the forehead line (in hair territory).

        In normalized coords, "above" means smaller y value.
        """
        forehead_y = self.forehead_y_at_x(x)
        if forehead_y is None:
            # x is outside forehead range — check against leftmost/rightmost
            line = self.forehead_line()
            if not line:
                return False
            # Use nearest edge point
            if x < line[0][0]:
                forehead_y = line[0][1]
            else:
                forehead_y = line[-1][1]
        return y < forehead_y


class FaceDetector:
    """Wraps MediaPipe FaceLandmarker to detect face mesh and compute hairline."""

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
        self._frame_ts = 0

    def detect(self, frame) -> FaceResult:
        """Detect face in BGR frame. Returns FaceResult with 468 landmarks."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts = int(time.monotonic() * 1000)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return FaceResult(landmarks={}, detected=False)

        face = result.face_landmarks[0]
        landmarks = {}
        for idx, lm in enumerate(face):
            landmarks[idx] = FaceLandmark(x=lm.x, y=lm.y)

        return FaceResult(landmarks=landmarks, detected=True)

    def close(self) -> None:
        self._landmarker.close()
```

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/test_face_detector.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add hairguard/face_detector.py tests/test_face_detector.py
git commit -m "Task 2: Add FaceDetector with forehead/hairline boundary"
```

---

### Task 3: HeadZone — hair region computation

**Files:**
- Create: `hairguard/head_zone.py`
- Create: `tests/test_head_zone.py`

- [ ] **Step 1: Write failing tests for hair zone**

```python
# tests/test_head_zone.py
from hairguard.head_zone import HairZone
from hairguard.face_detector import FaceResult, FaceLandmark
from hairguard.detector import PoseResult, Landmark

def _make_face(forehead_y: float = 0.25) -> FaceResult:
    """Create a face with forehead at given y."""
    landmarks = {i: FaceLandmark(x=0.5, y=0.5) for i in range(468)}
    # Forehead line across top
    for i, idx in enumerate([54, 103, 67, 109, 10, 338, 297, 332, 284]):
        x = 0.3 + i * 0.05  # spread from 0.30 to 0.70
        landmarks[idx] = FaceLandmark(x=x, y=forehead_y)
    return FaceResult(landmarks=landmarks, detected=True)

def _make_pose() -> PoseResult:
    """Create a pose with ears and shoulders."""
    return PoseResult(
        landmarks={
            "left_ear": Landmark(x=0.35, y=0.30, confidence=1.0),
            "right_ear": Landmark(x=0.65, y=0.30, confidence=1.0),
            "nose": Landmark(x=0.50, y=0.40, confidence=1.0),
            "left_shoulder": Landmark(x=0.30, y=0.60, confidence=1.0),
            "right_shoulder": Landmark(x=0.70, y=0.60, confidence=1.0),
        },
        raw_landmarks=None,
        detected=True,
    )

def test_point_in_hair_zone():
    """A point above the forehead and between ears should be in hair zone."""
    zone = HairZone(_make_face(forehead_y=0.25), _make_pose(), hair_zone_offset=0.4)
    # Above forehead, between ears
    assert zone.contains(0.50, 0.10) is True

def test_point_on_face_not_in_hair_zone():
    """A point on the face (below forehead) should NOT be in hair zone."""
    zone = HairZone(_make_face(forehead_y=0.25), _make_pose(), hair_zone_offset=0.4)
    # Below forehead — on the face
    assert zone.contains(0.50, 0.40) is False

def test_point_outside_ear_bounds():
    """A point above forehead but outside ear range should NOT be in hair zone."""
    zone = HairZone(_make_face(forehead_y=0.25), _make_pose(), hair_zone_offset=0.4)
    # Way to the left of head
    assert zone.contains(0.05, 0.10) is False

def test_tilted_head():
    """When head tilts forward, forehead line moves down; hair zone adapts."""
    face_tilted = _make_face(forehead_y=0.40)  # forehead at y=0.40 (tilted forward)
    zone = HairZone(face_tilted, _make_pose(), hair_zone_offset=0.4)
    # Point at y=0.35 — above the tilted forehead
    assert zone.contains(0.50, 0.35) is True
    # Point at y=0.45 — below the tilted forehead
    assert zone.contains(0.50, 0.45) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_head_zone.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement head_zone.py**

```python
# hairguard/head_zone.py
from __future__ import annotations

from typing import Optional

from hairguard.detector import PoseResult, Landmark
from hairguard.face_detector import FaceResult


class HairZone:
    """Defines the 'hair region' above the forehead, bounded by ears.

    Combines FaceMesh forehead contour (adapts to head tilt) with Pose
    ear positions (lateral bounds) and shoulder width (scale).
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

        # Compute lateral bounds from ears
        left_ear = pose.landmarks.get("left_ear")
        right_ear = pose.landmarks.get("right_ear")
        if left_ear and right_ear:
            ear_margin = abs(right_ear.x - left_ear.x) * 0.15
            self._x_min = min(left_ear.x, right_ear.x) - ear_margin
            self._x_max = max(left_ear.x, right_ear.x) + ear_margin
        else:
            self._x_min = 0.0
            self._x_max = 1.0

        # Compute vertical ceiling (how far above forehead to accept)
        # Ceiling = forehead mean y - (shoulder_width * offset), clamped to 0
        forehead_pts = face.forehead_line() if face.detected else []
        forehead_y_mean = sum(y for _, y in forehead_pts) / len(forehead_pts) if forehead_pts else 0.3
        left_sh = pose.landmarks.get("left_shoulder")
        right_sh = pose.landmarks.get("right_shoulder")
        if left_sh and right_sh:
            shoulder_width = abs(right_sh.x - left_sh.x)
            self._y_ceiling = max(0.0, forehead_y_mean - shoulder_width * hair_zone_offset)
        else:
            self._y_ceiling = 0.0

    def contains(self, x: float, y: float) -> bool:
        """Return True if (x, y) is in the hair zone.

        Hair zone = above forehead line, between ears (with margin),
        and not above the ceiling (crown limit).
        """
        # Check horizontal bounds (between ears + margin)
        if x < self._x_min or x > self._x_max:
            return False

        # Check above forehead using FaceMesh
        if self._face.detected:
            if not self._face.is_above_forehead(x, y):
                return False
        else:
            # No face mesh — fall back to nose position
            nose = self._pose.landmarks.get("nose")
            if nose and y >= nose.y:
                return False

        # Check not too far above head (ceiling)
        if y < self._y_ceiling:
            return False

        return True

    @property
    def forehead_line(self) -> list[tuple[float, float]]:
        """Return the forehead boundary points for visualization."""
        if self._face.detected:
            return self._face.forehead_line()
        return []

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_ceiling, y_forehead_avg) for visualization."""
        forehead_pts = self.forehead_line
        y_avg = sum(y for _, y in forehead_pts) / len(forehead_pts) if forehead_pts else 0.5
        return (self._x_min, self._x_max, self._y_ceiling, y_avg)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_head_zone.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add hairguard/head_zone.py tests/test_head_zone.py
git commit -m "Task 3: Add HairZone — hair region geometry from FaceMesh + Pose"
```

---

### Task 4: Rewrite GestureAnalyzer — 7-layer detection pipeline

**Files:**
- Modify: `hairguard/analyzer.py` (complete rewrite of `_detect()`)
- Create: `tests/test_analyzer_v2.py`

- [ ] **Step 1: Write failing tests for new analyzer**

```python
# tests/test_analyzer_v2.py
import time
from unittest.mock import MagicMock
from hairguard.analyzer import GestureAnalyzer, AnalysisResult
from hairguard.detector import PoseResult, Landmark
from hairguard.hand_detector import HandResult, HandLandmark
from hairguard.face_detector import FaceResult, FaceLandmark
from hairguard.head_zone import HairZone

def _make_config(**overrides):
    """Create a mock config."""
    defaults = {
        ("detection", "trigger_delay_sec"): 0.8,
        ("detection", "dismiss_delay_sec"): 2.5,
        ("detection", "min_confidence"): 0.5,
        ("detection", "pinch_threshold"): 0.06,
        ("detection", "min_presence_score"): 0.7,
        ("detection", "hair_zone_offset"): 0.4,
        ("detection", "grip_curl_threshold"): 2,
    }
    defaults.update(overrides)
    config = MagicMock()
    config.get = lambda *keys: defaults.get(keys, 0.5)
    return config

def _make_hand_on_hair(pinching=True, gripping=False):
    """Hand with thumb+index at forehead-height (in hair zone), optionally pinching."""
    landmarks = {i: HandLandmark(x=0.5, y=0.5, z=0.0, presence=0.95) for i in range(21)}
    if pinching:
        landmarks[4] = HandLandmark(x=0.48, y=0.12, z=0.0, presence=0.95)  # thumb tip — in hair zone
        landmarks[8] = HandLandmark(x=0.50, y=0.13, z=0.0, presence=0.95)  # index tip — close to thumb
    else:
        landmarks[4] = HandLandmark(x=0.40, y=0.12, z=0.0, presence=0.95)  # thumb
        landmarks[8] = HandLandmark(x=0.60, y=0.12, z=0.0, presence=0.95)  # index — far from thumb
    if gripping:
        # Curl middle, ring, pinky
        for tip, mcp in [(12, 9), (16, 13), (20, 17)]:
            landmarks[mcp] = HandLandmark(x=0.5, y=0.40, z=0.0, presence=0.95)
            landmarks[tip] = HandLandmark(x=0.5, y=0.60, z=0.0, presence=0.95)
    else:
        # Extend middle, ring, pinky
        for tip, mcp in [(12, 9), (16, 13), (20, 17)]:
            landmarks[mcp] = HandLandmark(x=0.5, y=0.40, z=0.0, presence=0.95)
            landmarks[tip] = HandLandmark(x=0.5, y=0.20, z=0.0, presence=0.95)
    return HandResult(landmarks=landmarks, handedness="Right", score=0.95)

def _make_face():
    landmarks = {i: FaceLandmark(x=0.5, y=0.5) for i in range(468)}
    for i, idx in enumerate([54, 103, 67, 109, 10, 338, 297, 332, 284]):
        landmarks[idx] = FaceLandmark(x=0.3 + i * 0.05, y=0.25)
    return FaceResult(landmarks=landmarks, detected=True)

def _make_pose():
    return PoseResult(
        landmarks={
            "left_ear": Landmark(x=0.35, y=0.30, confidence=1.0),
            "right_ear": Landmark(x=0.65, y=0.30, confidence=1.0),
            "nose": Landmark(x=0.50, y=0.40, confidence=1.0),
            "left_shoulder": Landmark(x=0.30, y=0.60, confidence=1.0),
            "right_shoulder": Landmark(x=0.70, y=0.60, confidence=1.0),
        },
        raw_landmarks=None,
        detected=True,
    )

def test_pinch_on_hair_triggers():
    """Pinching fingers in hair zone should result in hand_near_head=True."""
    config = _make_config()
    analyzer = GestureAnalyzer(config)
    hands = [_make_hand_on_hair(pinching=True, gripping=False)]
    face = _make_face()
    pose = _make_pose()
    result = analyzer.analyze(pose, hands, face, 480, 640)
    assert result.hand_near_head is True

def test_open_hand_near_head_does_not_trigger():
    """Open hand near head (not pinching) should NOT trigger."""
    config = _make_config()
    analyzer = GestureAnalyzer(config)
    hands = [_make_hand_on_hair(pinching=False, gripping=False)]
    face = _make_face()
    pose = _make_pose()
    result = analyzer.analyze(pose, hands, face, 480, 640)
    assert result.hand_near_head is False

def test_gripping_object_near_head_does_not_trigger():
    """Pinching but gripping (holding comb) should NOT trigger."""
    config = _make_config()
    analyzer = GestureAnalyzer(config)
    hands = [_make_hand_on_hair(pinching=True, gripping=True)]
    face = _make_face()
    pose = _make_pose()
    result = analyzer.analyze(pose, hands, face, 480, 640)
    assert result.hand_near_head is False

def test_no_hands_stays_idle():
    """No hands detected → IDLE."""
    config = _make_config()
    analyzer = GestureAnalyzer(config)
    result = analyzer.analyze(_make_pose(), [], _make_face(), 480, 640)
    assert result.should_alert is False
    assert result.state == "IDLE"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analyzer_v2.py -v`
Expected: FAIL — `analyze()` signature mismatch (old analyzer has different params)

- [ ] **Step 3: Rewrite analyzer.py**

The full rewrite of `analyzer.py`. State machine stays the same (IDLE/TRIGGERED/ALERTING/COOLDOWN). The `_detect()` method is completely replaced with the 7-layer pipeline. The `analyze()` signature changes to accept `pose_result`, `hand_results`, `face_result`, `frame_height`, `frame_width`.

Key changes to the existing file:
- `analyze()` now accepts: `pose_result: PoseResult, hand_results: list[HandResult], face_result: FaceResult, frame_height: int, frame_width: int`
- `_detect()` implements layers 1-6
- State machine (layers 7) stays in `analyze()` as before
- Remove old imports of `fallback_frame_percent`, `distance_threshold`
- Add new config reads: `pinch_threshold`, `min_presence_score`, `hair_zone_offset`, `grip_curl_threshold`

```python
# hairguard/analyzer.py — full replacement
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from hairguard.config import Config
from hairguard.detector import PoseResult, Landmark
from hairguard.hand_detector import HandResult
from hairguard.face_detector import FaceResult
from hairguard.head_zone import HairZone


class _State(Enum):
    IDLE = auto()
    TRIGGERED = auto()
    ALERTING = auto()
    COOLDOWN = auto()


@dataclass
class AnalysisResult:
    should_alert: bool
    hand_near_head: bool
    state: str
    episode_started: bool
    episode_ended: bool
    episode_duration: float
    # v2 debug info
    pinch_distance: Optional[float] = None
    fingertips_in_zone: bool = False
    is_gripping: bool = False


class GestureAnalyzer:
    """7-layer detection pipeline for hair-touching gestures.

    Layer 1: Hand present?
    Layer 2: Head located?
    Layer 3: Compute hair zone (FaceMesh forehead + Pose ears)
    Layer 4: Fingertips in hair zone?
    Layer 5: Pinch detected? (thumb + index close)
    Layer 6: Not gripping an object? (middle/ring/pinky relaxed)
    Layer 7: Debounce state machine
    """

    def __init__(self, config: Config) -> None:
        self._trigger_delay_sec: float = config.get("detection", "trigger_delay_sec")
        self._dismiss_delay_sec: float = config.get("detection", "dismiss_delay_sec")
        self._min_confidence: float = config.get("detection", "min_confidence")
        self._pinch_threshold: float = config.get("detection", "pinch_threshold")
        self._min_presence: float = config.get("detection", "min_presence_score")
        self._hair_zone_offset: float = config.get("detection", "hair_zone_offset")
        self._grip_curl_threshold: int = int(config.get("detection", "grip_curl_threshold"))

        self._state = _State.IDLE
        self._state_entered_at: float = 0.0
        self._episode_started_at: float = 0.0

    def analyze(
        self,
        pose_result: PoseResult,
        hand_results: list[HandResult],
        face_result: FaceResult,
        frame_height: int,
        frame_width: int,
    ) -> AnalysisResult:
        now = time.monotonic()
        detection = self._detect(pose_result, hand_results, face_result)

        hand_near_head = detection["detected"]

        episode_started = False
        episode_ended = False
        episode_duration = 0.0

        # State machine (same as v1)
        if self._state is _State.IDLE:
            if hand_near_head:
                self._transition(_State.TRIGGERED, now)

        elif self._state is _State.TRIGGERED:
            if not hand_near_head:
                self._transition(_State.IDLE, now)
            elif now - self._state_entered_at >= self._trigger_delay_sec:
                self._transition(_State.ALERTING, now)
                self._episode_started_at = now
                episode_started = True

        elif self._state is _State.ALERTING:
            if not hand_near_head:
                self._transition(_State.COOLDOWN, now)

        elif self._state is _State.COOLDOWN:
            if hand_near_head:
                self._transition(_State.ALERTING, now)
            elif now - self._state_entered_at >= self._dismiss_delay_sec:
                episode_duration = now - self._episode_started_at
                episode_ended = True
                self._transition(_State.IDLE, now)

        should_alert = self._state in (_State.ALERTING, _State.COOLDOWN)

        return AnalysisResult(
            should_alert=should_alert,
            hand_near_head=hand_near_head,
            state=self._state.name,
            episode_started=episode_started,
            episode_ended=episode_ended,
            episode_duration=episode_duration,
            pinch_distance=detection.get("pinch_distance"),
            fingertips_in_zone=detection.get("fingertips_in_zone", False),
            is_gripping=detection.get("is_gripping", False),
        )

    def _transition(self, new_state: _State, now: float) -> None:
        self._state = new_state
        self._state_entered_at = now

    def _detect(
        self,
        pose_result: PoseResult,
        hand_results: list[HandResult],
        face_result: FaceResult,
    ) -> dict:
        """Run layers 1-6 of the detection pipeline."""
        result = {
            "detected": False,
            "pinch_distance": None,
            "fingertips_in_zone": False,
            "is_gripping": False,
        }

        # Layer 1: Hand present?
        if not hand_results:
            return result

        # Layer 2: Head located?
        if not pose_result.detected:
            return result

        # Layer 3: Compute hair zone
        zone = HairZone(face_result, pose_result, self._hair_zone_offset)

        # Check each detected hand
        for hand in hand_results:
            thumb = hand.thumb_tip()
            index = hand.index_tip()

            # Filter by presence score
            if thumb.presence < self._min_presence or index.presence < self._min_presence:
                continue

            # Layer 4: Fingertips in hair zone?
            thumb_in = zone.contains(thumb.x, thumb.y)
            index_in = zone.contains(index.x, index.y)
            if not (thumb_in or index_in):
                continue
            result["fingertips_in_zone"] = True

            # Layer 5: Pinch detected? (normalized by hand size)
            pinch_r = hand.pinch_ratio()
            result["pinch_distance"] = pinch_r
            if pinch_r >= self._pinch_threshold:
                continue

            # Layer 6: Not gripping an object?
            gripping = hand.is_gripping(self._grip_curl_threshold)
            result["is_gripping"] = gripping
            if gripping:
                continue

            # All layers passed!
            result["detected"] = True
            return result

        return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_analyzer_v2.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add hairguard/analyzer.py tests/test_analyzer_v2.py
git commit -m "Task 4: Rewrite analyzer with 7-layer detection pipeline"
```

---

### Task 5: Update Config — new v2 detection parameters

**Files:**
- Modify: `hairguard/config.py`

- [ ] **Step 1: Update DEFAULTS dict**

Replace the `detection` section in `DEFAULTS`:

```python
"detection": {
    "trigger_delay_sec": 0.8,
    "dismiss_delay_sec": 2.5,
    "fps": 10,
    "min_confidence": 0.5,
    "pinch_threshold": 0.06,
    "min_presence_score": 0.7,
    "hair_zone_offset": 0.4,
    "grip_curl_threshold": 2,
},
```

- [ ] **Step 2: Update _FIELD_RULES**

Replace detection field rules:

```python
(("detection", "trigger_delay_sec"), float, (0.0, None)),
(("detection", "dismiss_delay_sec"), float, (0.0, None)),
(("detection", "fps"), (int, float), (1, None)),
(("detection", "min_confidence"), float, (0.0, 1.0)),
(("detection", "pinch_threshold"), float, (0.1, 2.0)),
(("detection", "min_presence_score"), float, (0.0, 1.0)),
(("detection", "hair_zone_offset"), float, (0.0, 2.0)),
(("detection", "grip_curl_threshold"), (int, float), (0, 3)),
```

Remove old rules for `distance_threshold` and `fallback_frame_percent`.

- [ ] **Step 3: Delete old user config to force regeneration**

The user's `~/.hairguard/config.yaml` has old v1 fields. On next start, missing new fields will be filled from defaults. But old fields (`distance_threshold`, `fallback_frame_percent`) will remain harmlessly.

- [ ] **Step 4: Commit**

```bash
git add hairguard/config.py
git commit -m "Task 5: Update config defaults for v2 detection parameters"
```

---

### Task 6: Update app.py — wire new detectors into main loop

**Files:**
- Modify: `hairguard/app.py`

- [ ] **Step 1: Update imports and __init__**

In `__init__`, add:
```python
from hairguard.hand_detector import HandDetector
from hairguard.face_detector import FaceDetector
```

Create instances:
```python
self._hand_detector = HandDetector(max_hands=2, min_confidence=min_conf)
self._face_detector = FaceDetector(min_confidence=min_conf)
```

- [ ] **Step 2: Update _tick() detection pipeline**

Replace the detection section in `_tick()`. Use a shared base timestamp with offsets to avoid MediaPipe "timestamp must be monotonically increasing" errors when multiple detectors process in the same millisecond:

```python
# --- Detection pipeline ---
frame_height = frame.shape[0]
frame_width = frame.shape[1] if len(frame.shape) > 1 else 640

# Shared timestamp base — offset each detector to avoid collisions
ts_base = int(time.monotonic() * 1000)
pose_result = self._detector.process(frame, timestamp_ms=ts_base)
hand_results = self._hand_detector.detect(frame, timestamp_ms=ts_base + 1)
face_result = self._face_detector.detect(frame, timestamp_ms=ts_base + 2)
result = self._analyzer.analyze(pose_result, hand_results, face_result, frame_height, frame_width)
```

This requires adding an optional `timestamp_ms` parameter to each detector's `process()`/`detect()` method. If provided, use it instead of computing `time.monotonic()` internally. Update `detector.py`, `hand_detector.py`, and `face_detector.py` accordingly — each `detect()` method should accept `timestamp_ms: int | None = None` and use `timestamp_ms or int(time.monotonic() * 1000)`.

Also update `_TICK_MS` to derive from config FPS:

```python
# In __init__:
fps = float(self._config.get("detection", "fps") or 10)
self._TICK_MS = max(50, int(1000 / fps))
```

- [ ] **Step 3: Update shutdown()**

Add cleanup:
```python
try:
    self._hand_detector.close()
except Exception:
    pass
try:
    self._face_detector.close()
except Exception:
    pass
```

- [ ] **Step 4: Commit**

```bash
git add hairguard/app.py
git commit -m "Task 6: Wire HandDetector + FaceDetector into main loop"
```

---

### Task 7: Rewrite debugview.py — visualize hand skeleton, hair zone, pinch

**Files:**
- Modify: `hairguard/debugview.py`

- [ ] **Step 1: Plan the new debug view layout**

The debug view should show:
- Camera feed as background
- **Hand skeleton:** all 21 points per hand with MediaPipe HAND_CONNECTIONS lines, color-coded by state
- **Fingertip highlights:** larger circles on thumb_tip (4) and index_tip (8) — GREEN if outside zone, YELLOW if in zone, RED if pinching in zone
- **Hair zone:** draw the forehead line (from FaceMesh) as a colored curve. Fill the area above it with a semi-transparent overlay to show the hair region. Draw vertical bounds from ears.
- **Pinch indicator:** when thumb+index are close, draw a line between them with distance label. Color: green (>threshold), red (<threshold)
- **Grip indicator:** show "GRIP" label if middle/ring/pinky are curled
- **Status panel:** same as before but with additional v2 info: pinch distance, fingertips_in_zone, is_gripping

- [ ] **Step 2: Implement new debugview.py**

This is a complete rewrite of the drawing logic. The main loop structure stays the same (OpenCV window, camera capture, key handling). The `draw_overlay()` function is rewritten to use HandDetector, FaceDetector, and PoseDetector.

Key changes:
- Import `HandDetector`, `FaceDetector`, `HairZone`
- In the main loop: call all three detectors per frame
- Draw hand skeleton using `mp.solutions.hands.HAND_CONNECTIONS` mapping (or hardcode the 21 connection pairs)
- Draw forehead line from FaceResult
- Draw hair zone as semi-transparent region
- Show pinch distance as a number on screen
- Show finger curl states

Hand skeleton connections (hardcoded, since we use Tasks API not solutions):
```python
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),    # thumb
    (0,5),(5,6),(6,7),(7,8),    # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),       # palm
]
```

- [ ] **Step 3: Manual test**

Run: `python -m hairguard.debugview`

Verify:
- Hand skeleton drawn with 21 points + connections
- Forehead line visible as a curve
- Hair zone shaded above forehead
- Pinch distance shown when thumb + index are close
- "GRIP" indicator when holding an object
- Status panel shows v2 fields
- Pressing 'q' exits

- [ ] **Step 4: Commit**

```bash
git add hairguard/debugview.py
git commit -m "Task 7: Rewrite debugview for v2 — hand skeleton, hair zone, pinch indicator"
```

---

### Task 8: Integration test — full end-to-end

**Files:**
- No new files, manual testing

- [ ] **Step 1: Test main app**

```bash
python -m hairguard start
```

Verify:
- App starts without errors
- Three models download on first run (if not cached)
- Tray icon appears
- Touch hair with pinching gesture → overlay appears after 0.8s
- Open hand near head → NO trigger
- Holding comb near head → NO trigger
- Touch forehead → NO trigger
- Remove hand → overlay disappears after 2.5s
- Stats recorded: `python -m hairguard stats`

- [ ] **Step 2: Test debug view**

```bash
python -m hairguard.debugview
```

Verify all visualizations work as described in Task 7.

- [ ] **Step 3: Performance check**

In debug view, verify FPS stays above 8-10 on the user's machine. If below 8:
- Try `max_hands=1` in HandDetector
- Try reducing FPS in config

- [ ] **Step 4: Commit any fixes**

```bash
# Only add files that were actually changed during testing fixes
git add hairguard/ tests/
git commit -m "Task 8: Integration fixes from end-to-end testing"
```

---

## Execution Notes

- Tasks 1, 2 can run in parallel (no shared dependencies)
- Task 3 depends on Tasks 1 + 2
- Task 4 depends on Tasks 1 + 2 + 3
- Task 5 has no dependencies (config only)
- Task 6 depends on Tasks 1 + 2 + 4 + 5
- Task 7 depends on Tasks 1 + 2 + 3
- Task 8 depends on all

```
Task 1 (HandDetector) ──┐
                        ├──→ Task 3 (HairZone) ──→ Task 4 (Analyzer) ──→ Task 6 (app.py) ──→ Task 8
Task 2 (FaceDetector) ──┘                    └──→ Task 7 (debugview) ──┘
Task 5 (Config) ──────────────────────────────────────────────────────────┘
```
