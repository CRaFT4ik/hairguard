"""Tests for hairlab.collector — the clip capture state machine.

The collector decides *when* a clip is worth keeping. It is deliberately
free of camera and disk concerns: frames arrive through offer(), finished
clips leave through a sink, and time comes from an injected clock. That
makes every rule below testable without hardware.
"""
from __future__ import annotations

import numpy as np
import pytest

from hairlab.collector import ClipCollector

WINDOW = 4
COOLDOWN = 5.0


class FakeSink:
    """Records the clips a collector decides to keep."""

    def __init__(self) -> None:
        self.clips: list[list[np.ndarray]] = []
        self.metas: list[dict] = []

    def save(self, frames, meta) -> str:
        self.clips.append(list(frames))
        self.metas.append(meta)
        return f"clip{len(self.clips)}"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def sink() -> FakeSink:
    return FakeSink()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def collector(sink: FakeSink, clock: FakeClock) -> ClipCollector:
    return ClipCollector(sink=sink, window_frames=WINDOW, cooldown_sec=COOLDOWN, clock=clock)


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def _feed(collector: ClipCollector, count: int, *, in_zone: bool = True, start: int = 0) -> None:
    for i in range(count):
        collector.offer(_frame(start + i), in_zone=in_zone)


# ---------------------------------------------------------------------------
# Consent: nothing is captured unless collection was explicitly started
# ---------------------------------------------------------------------------

def test_collector_starts_disabled(collector: ClipCollector) -> None:
    assert collector.enabled is False


def test_disabled_collector_never_saves(collector: ClipCollector, sink: FakeSink) -> None:
    _feed(collector, WINDOW * 3)
    assert sink.clips == []


def test_stopping_discards_buffered_frames(collector: ClipCollector, sink: FakeSink) -> None:
    """Frames held in memory must not survive a stop — that is the point of the switch."""
    collector.start()
    _feed(collector, WINDOW - 1)
    collector.stop()
    collector.start()
    _feed(collector, 1)
    assert sink.clips == []


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

def test_saves_a_clip_after_a_full_window_in_zone(collector: ClipCollector, sink: FakeSink) -> None:
    collector.start()
    _feed(collector, WINDOW)
    assert len(sink.clips) == 1


def test_saved_clip_holds_exactly_the_window_in_order(collector: ClipCollector, sink: FakeSink) -> None:
    collector.start()
    _feed(collector, WINDOW)
    saved = sink.clips[0]
    assert len(saved) == WINDOW
    assert [int(f[0, 0, 0]) for f in saved] == list(range(WINDOW))


def test_nothing_saved_before_the_window_is_full(collector: ClipCollector, sink: FakeSink) -> None:
    collector.start()
    _feed(collector, WINDOW - 1)
    assert sink.clips == []


def test_out_of_zone_frame_resets_progress(collector: ClipCollector, sink: FakeSink) -> None:
    """A clip must be one unbroken stretch of hand-at-head, not an accumulation."""
    collector.start()
    _feed(collector, WINDOW - 1)
    collector.offer(_frame(99), in_zone=False)
    _feed(collector, WINDOW - 1)
    assert sink.clips == []


def test_out_of_zone_frames_are_never_buffered(collector: ClipCollector, sink: FakeSink) -> None:
    """Frames with no hand at the head must not end up inside a saved clip."""
    collector.start()
    _feed(collector, 10, in_zone=False, start=90)
    _feed(collector, WINDOW)
    assert [int(f[0, 0, 0]) for f in sink.clips[0]] == list(range(WINDOW))


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

def test_cooldown_blocks_a_second_clip(collector: ClipCollector, sink: FakeSink) -> None:
    collector.start()
    _feed(collector, WINDOW * 2)
    assert len(sink.clips) == 1


def test_clip_can_be_saved_again_after_cooldown(
    collector: ClipCollector, sink: FakeSink, clock: FakeClock
) -> None:
    collector.start()
    _feed(collector, WINDOW)
    clock.advance(COOLDOWN + 0.1)
    _feed(collector, WINDOW)
    assert len(sink.clips) == 2


# ---------------------------------------------------------------------------
# Metadata: the clip must record how fast it was actually captured
# ---------------------------------------------------------------------------

def test_clip_metadata_carries_the_per_frame_entries(
    collector: ClipCollector, sink: FakeSink
) -> None:
    collector.start()
    for i in range(WINDOW):
        collector.offer(_frame(i), in_zone=True, frame_meta={"i": i})
    assert [m["i"] for m in sink.metas[0]["frames"]] == list(range(WINDOW))


def test_measured_frame_rate_comes_from_the_timestamps(
    collector: ClipCollector, sink: FakeSink
) -> None:
    """The configured fps is a request; what the loop achieved is a fact.

    Playback speed and any time-derived feature must use the real rate, not
    the number someone put in a config file.
    """
    collector.start()
    for i in range(WINDOW):
        collector.offer(_frame(i), in_zone=True, frame_meta={"t": 100.0 + i * 0.25})

    meta = sink.metas[0]
    assert meta["duration_sec"] == pytest.approx(0.75)
    assert meta["fps_measured"] == pytest.approx(4.0)


def test_clip_metadata_keeps_the_collector_wide_fields(
    collector: ClipCollector, sink: FakeSink
) -> None:
    collector.clip_meta = {"crop_size": 256}
    collector.start()
    _feed(collector, WINDOW)
    assert sink.metas[0]["crop_size"] == 256


def test_no_measured_rate_without_timestamps(
    collector: ClipCollector, sink: FakeSink
) -> None:
    collector.start()
    _feed(collector, WINDOW)
    assert "fps_measured" not in sink.metas[0]
