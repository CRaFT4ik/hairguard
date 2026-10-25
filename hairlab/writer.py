"""hairlab.writer — write clips without stalling the detection loop.

Encoding twenty JPEGs and a metadata file takes long enough to be felt by a
loop that also owns the tkinter event loop and the fullscreen alert. Dataset
collection is a side activity; the alert is the reason the app exists. So
the caller only reserves an id — one mkdir — and a worker thread does the
encoding.

Errors are reported through a callback rather than raised. A full disk is a
reason to lose a clip, never a reason to stop warning the user.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional, Sequence

import numpy as np

from hairlab.clipstore import ClipStore

# Clips are seconds apart; a backlog this deep already means something is
# badly wrong, and dropping is better than growing memory without bound.
_MAX_PENDING = 16


class BackgroundClipWriter:
    """Reserves clip ids on the caller's thread, writes them on its own."""

    def __init__(
        self,
        store: Optional[ClipStore] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self._store = store if store is not None else ClipStore()
        self._on_error = on_error
        self._queue: queue.Queue = queue.Queue(maxsize=_MAX_PENDING)
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="clip-writer", daemon=True
        )
        self._thread.start()

    @property
    def root(self):
        return self._store.root

    # ------------------------------------------------------------------

    def save(self, frames: Sequence[np.ndarray], meta: dict[str, Any]) -> str:
        """Accept a clip and return its id at once. Satisfies ClipSink."""
        clip_id, _ = self._store.reserve()
        try:
            self._queue.put_nowait((clip_id, list(frames), dict(meta)))
        except queue.Full:
            self._report(RuntimeError("очередь записи клипов переполнена, клип пропущен"))
        return clip_id

    def flush(self, timeout: float = 10.0) -> None:
        """Wait for queued clips to be written. For tests and shutdown."""
        deadline = threading.Event()
        if not self._queue.empty():
            self._queue.join()
        deadline.set()

    def close(self, timeout: float = 10.0) -> None:
        """Finish pending work and stop the worker."""
        if self._closed:
            return
        self._closed = True
        self.flush(timeout)
        self._queue.put(None)
        self._thread.join(timeout)

    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            clip_id, frames, meta = item
            try:
                self._store.write(clip_id, frames, meta)
            except Exception as exc:
                self._report(exc)
            finally:
                self._queue.task_done()

    def _report(self, exc: Exception) -> None:
        if self._on_error is not None:
            try:
                self._on_error(exc)
                return
            except Exception:
                pass
        print(f"[hairlab] Не удалось записать клип: {exc}")
