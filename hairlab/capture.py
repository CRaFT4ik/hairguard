"""hairlab.capture — turns the app's per-frame detections into stored clips.

``CaptureService`` is the seam between the running application and the
dataset: the app hands it a frame plus the detections it already computed,
and it decides (via ClipCollector) whether that moment becomes a clip.

Two things here decide whether the dataset is usable at all.

*The crop follows the hand.* It is centred on the hand found in the hair
zone and sized from that hand's own span, so the hand fills the same
fraction of every clip whether the user sits near the camera or far from
it. Cropping around head-and-hands together — the obvious first idea —
produces clips where the hand fills the frame in one sample and is a
thumbnail in the next, and no amount of training fixes that.

*Landmarks travel with the pixels.* The first model trains on normalised
landmark sequences, not on images, so the coordinates are the training
input and the frames are what the human labels by. Both are written, in
full-frame normalised coordinates, while the detections are still exact —
re-deriving them later from a crop is strictly worse.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

from hairlab.clipstore import ClipStore
from hairlab.collector import ClipCollector
from hairlab.writer import BackgroundClipWriter

# The label taxonomy. Shared by the labelling UI and, later, training.
# Order matters: it is the order of the buttons and of the keyboard shortcuts.
LABELS: list[dict[str, str]] = [
    {"id": "twisting", "title": "Кручу", "hint": "Верчу/тяну прядь"},
    {"id": "resting", "title": "Просто держу", "hint": "Рука лежит, без движения пальцев"},
    {"id": "scratching", "title": "Чешу", "hint": "Чешу кожу головы"},
    {"id": "object", "title": "Расчёска/предмет", "hint": "В руке расчёска или другой предмет"},
    {"id": "unclear", "title": "Не разобрать", "hint": "Непонятно — исключить из обучения"},
]

LABEL_IDS = [item["id"] for item in LABELS]

DEFAULT_CROP_SIZE = 256
DEFAULT_WINDOW_FRAMES = 20   # 2 seconds at the app's default 10 fps
# Minimum gap between clips. A single long episode should contribute a few
# varied samples, not a burst of near-duplicates that cost labelling time and
# teach the model nothing new.
DEFAULT_COOLDOWN_SEC = 10.0

# Added on every side of the hand's own bounding box. 0.4 leaves the hand at
# a little over half the frame width, with the surrounding hair in view.
_HAND_MARGIN = 0.4
_HEAD_MARGIN = 0.35   # used only by the head+hand fallback

_POSE_KEYS = ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder")


# ---------------------------------------------------------------------------
# Crop geometry (pure)
# ---------------------------------------------------------------------------

def square_crop_box(
    points: Sequence[tuple[float, float]],
    frame_width: int,
    frame_height: int,
    margin: float = _HEAD_MARGIN,
    min_size: int = 96,
) -> tuple[int, int, int, int]:
    """Square pixel box around normalised points, clamped to the frame.

    Landmarks may fall slightly outside the image — MediaPipe reports those
    rather than dropping them — so points are clamped before use. With no
    points at all the box falls back to the centre of the frame.
    """
    limit = min(frame_width, frame_height)

    if points:
        xs = [min(max(x, 0.0), 1.0) * frame_width for x, _ in points]
        ys = [min(max(y, 0.0), 1.0) * frame_height for _, y in points]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        size = span * (1.0 + 2 * margin)
    else:
        cx, cy = frame_width / 2, frame_height / 2
        size = limit * 0.6

    size = max(min_size, min(size, limit))
    half = size / 2

    # Keep the box inside the frame by shifting it, never by shrinking it.
    cx = min(max(cx, half), frame_width - half)
    cy = min(max(cy, half), frame_height - half)

    x0, y0 = int(round(cx - half)), int(round(cy - half))
    side = int(round(size))
    return x0, y0, x0 + side, y0 + side


def hand_points(hand: Any) -> list[tuple[float, float]]:
    """Normalised (x, y) of every landmark on one hand."""
    return [(lm.x, lm.y) for lm in getattr(hand, "landmarks", {}).values()]


def crop_box_for_hand(
    hand: Any,
    frame_width: int,
    frame_height: int,
    margin: float = _HAND_MARGIN,
) -> tuple[int, int, int, int]:
    """Square box centred on one hand and scaled to that hand's own span."""
    return square_crop_box(hand_points(hand), frame_width, frame_height, margin=margin)


def crop_frame(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    size: int = DEFAULT_CROP_SIZE,
) -> np.ndarray:
    """Cut ``box`` out of ``frame`` and resize it to ``size`` x ``size``.

    Enlarging a small box invents detail rather than adding it, so
    ``INTER_AREA`` is used when shrinking and the caller records the source
    box in metadata to make genuine resolution auditable.
    """
    x0, y0, x1, y1 = box
    patch = frame[max(0, y0):y1, max(0, x0):x1]
    if patch.size == 0:
        patch = frame
    shrinking = patch.shape[0] >= size
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(patch, (size, size), interpolation=interpolation)


# ---------------------------------------------------------------------------
# Landmark capture
# ---------------------------------------------------------------------------

def _hand_landmarks(hand: Any) -> list[list[float]]:
    """21 x [x, y, z] in full-frame normalised coordinates, in landmark order."""
    landmarks = getattr(hand, "landmarks", {}) or {}
    return [
        [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
        for _, lm in sorted(landmarks.items())
    ]


def _pose_landmarks(pose_result: Any) -> dict[str, list[float]]:
    """Head and shoulder points — the reference frame motion is normalised into."""
    landmarks = getattr(pose_result, "landmarks", None) or {}
    out: dict[str, list[float]] = {}
    for name in _POSE_KEYS:
        lm = landmarks.get(name)
        if lm is not None:
            out[name] = [round(lm.x, 5), round(lm.y, 5), round(lm.confidence, 3)]
    return out


def _forehead_line(face_result: Any) -> list[list[float]]:
    """The hairline arc, when the face detector found one."""
    if not getattr(face_result, "detected", False):
        return []
    try:
        return [[round(x, 5), round(y, 5)] for x, y in face_result.forehead_line()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class CaptureService:
    """Collects labelled-later clips from the running detection loop.

    Collection is off until :meth:`start` is called and every frame held in
    memory is dropped by :meth:`stop`, so the switch in the tray is the whole
    truth about whether anything is being recorded.
    """

    def __init__(
        self,
        store: Optional[ClipStore] = None,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        crop_size: int = DEFAULT_CROP_SIZE,
        fps: float = 10.0,
    ) -> None:
        self._store = store if store is not None else ClipStore()
        self._crop_size = crop_size
        self._fps = float(fps)
        # Clips are encoded on a worker thread. The detection loop also owns
        # the tkinter event loop and the alert overlay, and must not be held
        # up by twenty JPEG writes.
        self._writer = BackgroundClipWriter(self._store)
        self._collector = ClipCollector(
            sink=self._writer,
            window_frames=window_frames,
            cooldown_sec=cooldown_sec,
        )
        self._collector.clip_meta = {
            "fps_configured": self._fps,
            "crop_size": crop_size,
        }

    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._collector.enabled

    @property
    def saved_count(self) -> int:
        return self._collector.saved_count

    @property
    def clips_root(self):
        return self._store.root

    @property
    def buffered_frames(self) -> int:
        """Frames held so far towards the next clip."""
        return self._collector.buffered_frames

    @property
    def window_frames(self) -> int:
        """Frames a complete clip needs."""
        return self._collector.window_frames

    def start(self) -> None:
        self._collector.start()

    def stop(self) -> None:
        self._collector.stop()

    def toggle(self) -> bool:
        """Flip collection on/off. Returns the new state."""
        if self.enabled:
            self.stop()
        else:
            self.start()
        return self.enabled

    def close(self) -> None:
        """Stop collecting and finish writing whatever was already accepted."""
        self.stop()
        try:
            self._writer.close()
        except Exception:
            pass

    # ------------------------------------------------------------------

    def on_tick(
        self,
        frame: np.ndarray,
        pose_result: Any,
        hand_results: Iterable[Any],
        in_zone: bool,
        hand_index: Optional[int] = None,
        face_result: Any = None,
    ) -> Optional[str]:
        """Offer one analysed frame. Returns a clip id if one was just stored."""
        if not self.enabled:
            return None

        if not in_zone:
            # Let the collector reset; never crop or retain this frame.
            return self._collector.offer(frame, in_zone=False)

        hands = list(hand_results or [])
        hand = hands[hand_index] if hand_index is not None and hand_index < len(hands) else None

        height, width = frame.shape[:2]
        if hand is not None:
            box = crop_box_for_hand(hand, width, height)
        else:
            # No specific hand identified: fall back to head and hands together
            # rather than dropping a sample the geometry already approved.
            points = [(lm.x, lm.y) for h in hands for lm in getattr(h, "landmarks", {}).values()]
            points += [
                (lm.x, lm.y)
                for name in ("nose", "left_ear", "right_ear")
                if (lm := (getattr(pose_result, "landmarks", None) or {}).get(name)) is not None
            ]
            box = square_crop_box(points, width, height)

        patch = crop_frame(frame, box, size=self._crop_size)

        frame_meta: dict[str, Any] = {
            "t": round(time.monotonic(), 4),
            "box": list(box),
            "source_px": box[2] - box[0],
            "hand_index": hand_index,
        }
        if hand is not None:
            frame_meta["hand"] = _hand_landmarks(hand)
        pose_points = _pose_landmarks(pose_result)
        if pose_points:
            frame_meta["pose"] = pose_points
        forehead = _forehead_line(face_result)
        if forehead:
            frame_meta["forehead"] = forehead

        return self._collector.offer(patch, in_zone=True, frame_meta=frame_meta)
