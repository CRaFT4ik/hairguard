from __future__ import annotations

import os
import time
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Frame classification
# ---------------------------------------------------------------------------
# A camera index that opens successfully is not proof of a usable camera.
# Two failure modes occur on real machines, and read() reports success for
# both: a device that returns all-zero frames, and a virtual camera serving
# one frozen placeholder image. Measured on this project's hardware:
#
#   source                  frame std   mean |frame[i] - frame[i-1]|
#   blank device              0.00               0.0000
#   OBS virtual placeholder  38.88               0.0000
#   real webcam              62.8                1.49 - 1.87
#
# Brightness alone cannot separate the placeholder from a real sensor, but
# movement can: a live sensor always carries noise, so consecutive frames
# never match exactly. The thresholds below sit in the wide gap between
# those columns.

_BLANK_STD = 1.0    # below this the frame carries no content at all
_LIVE_DIFF = 0.1    # below this consecutive frames are effectively frozen

_PROBE_WARMUP = 8    # frames discarded while exposure settles after open
_PROBE_SAMPLES = 5   # frames actually classified
_MAX_PROBE_INDEX = 5  # probing absent indices is slow on Windows


def classify_frames(frames: Sequence[np.ndarray]) -> str:
    """Classify a burst of consecutive frames as "blank", "static" or "live".

    Needs at least two frames: liveness is a property of how frames differ
    from one another, so a single frame can never prove it.
    """
    if len(frames) < 2:
        return "blank"

    samples = [f.astype(np.int16) for f in frames]

    if max(float(f.std()) for f in samples) < _BLANK_STD:
        return "blank"

    max_diff = max(
        float(np.abs(samples[i] - samples[i - 1]).mean())
        for i in range(1, len(samples))
    )
    if max_diff < _LIVE_DIFF:
        return "static"

    return "live"


def probe_camera(index: int) -> Optional[str]:
    """Open a camera, sample it and classify it. None if it cannot be opened."""
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        for _ in range(_PROBE_WARMUP):
            cap.read()
            time.sleep(0.03)

        frames = []
        for _ in range(_PROBE_SAMPLES):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            time.sleep(0.04)
    finally:
        cap.release()

    return classify_frames(frames)


def find_working_camera(max_index: int = _MAX_PROBE_INDEX) -> Optional[int]:
    """Return the lowest camera index that shows a live image, else None.

    Stops at the first live camera, so the common case (a working camera at
    index 0 or 1) costs about a second rather than a full sweep.
    """
    for index in range(max_index + 1):
        verdict = probe_camera(index)
        if verdict is None:
            continue
        print(f"[camera] Проверка камеры {index}: {verdict}")
        if verdict == "live":
            return index
    return None


def resolve_camera_index(
    configured: object,
    finder: Callable[[], Optional[int]] = find_working_camera,
) -> int:
    """Turn a configured camera setting into a concrete index.

    An explicit index is always honoured, even if that camera turns out to be
    blind — the user asked for it by number. Anything else ("auto", missing)
    triggers a probe, falling back to 0 so a machine with no usable camera
    still starts instead of crashing at login.
    """
    if isinstance(configured, int) and not isinstance(configured, bool):
        return configured

    found = finder()
    if found is None:
        print("[camera] Рабочая камера не найдена, использую индекс 0.")
        return 0
    print(f"[camera] Выбрана камера {found}.")
    return found


class CameraManager:
    @staticmethod
    def list_cameras() -> list[dict]:
        os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
        results = []
        for index in range(10):
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if cap.isOpened():
                name = f"Camera {index}"
                backend = cap.getBackendName()
                if backend:
                    name = f"Camera {index} ({backend})"
                results.append({"index": index, "name": name})
                cap.release()
        return results

    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self._index = camera_index
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        # Ask for a capture resolution before the first read. Webcams that
        # cannot deliver it silently keep their default, so the result is
        # read back rather than assumed.
        if width and height:
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            except Exception:
                pass
        # Ask the backend for a 1-frame internal buffer so read() returns
        # the freshest available frame. Not all backends honor this, but
        # when supported it prevents accumulated stale frames when the
        # consumer reads slower than the camera produces.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    @property
    def resolution(self) -> tuple[int, int]:
        """Resolution the camera actually settled on."""
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read_frame(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __enter__(self) -> "CameraManager":
        return self

    def __exit__(self, *_) -> None:
        self.release()
