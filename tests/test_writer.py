"""Tests for hairlab.writer — clips are written off the detection thread.

Encoding twenty JPEGs is slow enough to be felt by a loop that also owns
the tkinter event loop and the alert overlay. Dataset collection is a side
activity and must never delay, or break, the thing the app exists for.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from hairlab.clipstore import ClipStore
from hairlab.writer import BackgroundClipWriter


def _frames(n: int = 3) -> list[np.ndarray]:
    return [np.full((8, 8, 3), 10 * i, dtype=np.uint8) for i in range(n)]


@pytest.fixture
def store(tmp_path) -> ClipStore:
    return ClipStore(root=tmp_path / "clips")


@pytest.fixture
def writer(store: ClipStore):
    w = BackgroundClipWriter(store)
    yield w
    w.close()


def test_save_returns_an_id_immediately(writer: BackgroundClipWriter) -> None:
    assert writer.save(_frames(), {}) != ""


def test_the_clip_really_lands_on_disk(writer: BackgroundClipWriter, store: ClipStore) -> None:
    clip_id = writer.save(_frames(3), {"fps_measured": 6.0})
    writer.flush()

    assert store.exists(clip_id)
    assert store.frame_count(clip_id) == 3
    assert store.read_meta(clip_id)["fps_measured"] == 6.0


def test_the_caller_is_not_made_to_wait(store: ClipStore) -> None:
    """The point of the whole class: the detection loop hands off and moves on."""
    started = threading.Event()
    release = threading.Event()

    class SlowStore(ClipStore):
        def write(self, clip_id, frames, meta):
            started.set()
            release.wait(5)
            super().write(clip_id, frames, meta)

    writer = BackgroundClipWriter(SlowStore(root=store.root))
    try:
        began = time.monotonic()
        writer.save(_frames(), {})
        elapsed = time.monotonic() - began

        assert started.wait(2), "фоновая запись не началась"
        assert elapsed < 0.5, f"вызывающий поток простоял {elapsed:.2f}s"
    finally:
        release.set()
        writer.close()


def test_a_failing_write_does_not_escape_to_the_caller(store: ClipStore) -> None:
    """A full disk must not take the alert overlay down with it."""
    errors: list[Exception] = []

    class BrokenStore(ClipStore):
        def write(self, clip_id, frames, meta):
            raise OSError("диск переполнен")

    writer = BackgroundClipWriter(BrokenStore(root=store.root), on_error=errors.append)
    try:
        writer.save(_frames(), {})     # must not raise
        writer.flush()
    finally:
        writer.close()

    assert errors and isinstance(errors[0], OSError)


def test_several_clips_are_all_written(writer: BackgroundClipWriter, store: ClipStore) -> None:
    ids = [writer.save(_frames(2), {}) for _ in range(5)]
    writer.flush()

    assert sorted(store.clip_ids()) == sorted(ids)


def test_close_waits_for_pending_work(store: ClipStore) -> None:
    """Shutting down must not throw away a clip already accepted."""
    writer = BackgroundClipWriter(store)
    clip_id = writer.save(_frames(2), {})
    writer.close()

    assert store.exists(clip_id)
