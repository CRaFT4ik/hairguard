"""hairlab.collector — decides which moments are worth keeping as clips.

The rule is deliberately dumb about gestures: it keeps any unbroken stretch
of frames during which the hand was inside the hair zone. Judging *what* the
hand was doing is the job of the human labeller, and later of the trained
model — a collector that already knew would only ever confirm what the
current detector believes.

Three properties matter and are enforced here:

  consent    Nothing is buffered until start() is called, and stop()
             discards whatever was held in memory.
  continuity A saved clip is one unbroken stretch at the head, never
             stitched together from separate visits.
  restraint  After a clip is saved a cooldown passes before the next one,
             so a long episode yields samples rather than a flood.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np


class ClipSink(Protocol):
    """Destination for a finished clip."""

    def save(self, frames: Sequence[np.ndarray], meta: dict[str, Any]) -> str:
        """Persist the clip and return its identifier."""


class ClipCollector:
    """Buffers frames while the hand is at the head and emits fixed-length clips.

    Parameters
    ----------
    sink:
        Where finished clips go.
    window_frames:
        How many consecutive in-zone frames make up one clip.
    cooldown_sec:
        Minimum delay between two saved clips.
    clock:
        Monotonic time source, injectable for tests.
    """

    def __init__(
        self,
        sink: ClipSink,
        window_frames: int,
        cooldown_sec: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sink = sink
        self._window = int(window_frames)
        self._cooldown = float(cooldown_sec)
        self._clock = clock

        self._enabled = False
        self._buffer: list[np.ndarray] = []
        self._meta: list[dict[str, Any]] = []
        self._ready_at = 0.0
        self._saved_count = 0

        # Merged into the metadata of every clip this collector saves.
        self.clip_meta: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Consent
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def saved_count(self) -> int:
        """Clips saved since this collector was created."""
        return self._saved_count

    @property
    def buffered_frames(self) -> int:
        """Frames held so far towards the next clip."""
        return len(self._buffer)

    @property
    def window_frames(self) -> int:
        """Frames a complete clip needs."""
        return self._window

    def start(self) -> None:
        """Begin collecting. Starts from an empty buffer."""
        self._discard()
        self._enabled = True

    def stop(self) -> None:
        """Stop collecting and drop every frame still held in memory."""
        self._enabled = False
        self._discard()

    # ------------------------------------------------------------------
    # Frame intake
    # ------------------------------------------------------------------

    def offer(
        self,
        frame: np.ndarray,
        in_zone: bool,
        frame_meta: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Offer one frame. Returns a clip id when this frame completed a clip.

        ``in_zone`` is the geometric verdict for this frame: fingertips inside
        the hair zone. Frames without it are not buffered at all, so footage of
        the user simply sitting at the camera is never retained.
        """
        if not self._enabled or not in_zone:
            self._discard()
            return None

        if self._clock() < self._ready_at:
            # Still cooling down from the previous clip.
            return None

        self._buffer.append(frame)
        self._meta.append(frame_meta or {})

        if len(self._buffer) < self._window:
            return None

        clip_id = self._sink.save(self._buffer, self._build_meta())
        self._saved_count += 1
        self._ready_at = self._clock() + self._cooldown
        self._discard()
        return clip_id

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_meta(self) -> dict[str, Any]:
        """Clip metadata, including the frame rate actually achieved.

        The configured fps is a request the detection loop rarely meets — it
        runs three models per frame. Playback speed and any time-derived
        feature have to use the measured rate, so it is computed from the
        frame timestamps rather than copied from the configuration.
        """
        meta: dict[str, Any] = {**self.clip_meta, "frames": self._meta}

        times = [m["t"] for m in self._meta if isinstance(m.get("t"), (int, float))]
        if len(times) >= 2:
            span = times[-1] - times[0]
            if span > 0:
                meta["duration_sec"] = round(span, 4)
                meta["fps_measured"] = round((len(times) - 1) / span, 3)
        return meta

    def _discard(self) -> None:
        self._buffer = []
        self._meta = []
