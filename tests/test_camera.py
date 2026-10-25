"""Tests for hairguard.camera — frame classification and camera index resolution.

These tests exercise the pure functions only; they never open a real camera.
Frames are synthesised with numpy to model the three source kinds observed on
real hardware:

  blank  — a device that opens and returns frames of all-zero pixels
  static — a virtual camera serving one frozen placeholder image
  live   — a real sensor, whose frames always carry a little noise
"""
from __future__ import annotations

import numpy as np
import pytest

from hairguard.camera import classify_frames, resolve_camera_index

_SHAPE = (48, 64, 3)


def _blank_frame() -> np.ndarray:
    return np.zeros(_SHAPE, dtype=np.uint8)


def _picture(seed: int = 0) -> np.ndarray:
    """A frame with real content: a bright half and a dark half."""
    frame = np.zeros(_SHAPE, dtype=np.uint8)
    frame[:, : _SHAPE[1] // 2] = 200
    frame[0, 0] = seed % 256
    return frame


def _noisy(rng: np.random.Generator) -> np.ndarray:
    """A frame with content plus sensor noise, like a real webcam."""
    base = _picture().astype(np.int16)
    noise = rng.integers(-3, 4, size=_SHAPE, dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# classify_frames
# ---------------------------------------------------------------------------

def test_all_zero_frames_are_blank() -> None:
    assert classify_frames([_blank_frame() for _ in range(5)]) == "blank"


def test_identical_repeated_picture_is_static() -> None:
    """A virtual camera serving a frozen placeholder: bright, but never changes."""
    frame = _picture()
    assert classify_frames([frame.copy() for _ in range(5)]) == "static"


def test_frames_with_sensor_noise_are_live() -> None:
    rng = np.random.default_rng(1234)
    assert classify_frames([_noisy(rng) for _ in range(5)]) == "live"


def test_single_frame_is_blank_because_liveness_is_unprovable() -> None:
    assert classify_frames([_picture()]) == "blank"


def test_no_frames_is_blank() -> None:
    assert classify_frames([]) == "blank"


def test_bright_frames_with_no_movement_are_not_live() -> None:
    """Guards the real failure: OBS placeholder had std=38.9 yet was useless."""
    frame = _picture()
    assert classify_frames([frame.copy() for _ in range(6)]) != "live"


def test_motion_between_frames_is_live() -> None:
    """A scene that changes is live even without per-pixel noise."""
    frames = []
    for i in range(5):
        f = _blank_frame()
        f[:, : 10 + i * 5] = 180
        frames.append(f)
    assert classify_frames(frames) == "live"


# ---------------------------------------------------------------------------
# resolve_camera_index
# ---------------------------------------------------------------------------

def test_explicit_index_is_returned_unchanged() -> None:
    """An explicit index is honoured without probing, even if it is blind."""
    def _never_called() -> int | None:
        raise AssertionError("must not probe when an index is configured")

    assert resolve_camera_index(2, finder=_never_called) == 2


def test_explicit_index_zero_is_honoured() -> None:
    """0 is falsy — it must not be mistaken for 'unset'."""
    def _never_called() -> int | None:
        raise AssertionError("must not probe when an index is configured")

    assert resolve_camera_index(0, finder=_never_called) == 0


def test_auto_triggers_the_finder() -> None:
    assert resolve_camera_index("auto", finder=lambda: 3) == 3


def test_missing_value_triggers_the_finder() -> None:
    assert resolve_camera_index(None, finder=lambda: 1) == 1


def test_auto_falls_back_to_zero_when_nothing_is_found() -> None:
    """A headless machine must not crash the app at startup."""
    assert resolve_camera_index("auto", finder=lambda: None) == 0
