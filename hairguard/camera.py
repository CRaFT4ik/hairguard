from __future__ import annotations

import os

import cv2
import numpy as np


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

    def __init__(self, camera_index: int = 0) -> None:
        self._index = camera_index
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

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
