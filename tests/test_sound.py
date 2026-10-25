"""Tests for hairguard.sound — the audible part of an alert.

The sound repeats for as long as the hand stays at the head: a single chirp
is easy to ignore, a repeating one is not, and stopping it is the reward for
moving the hand away.

Two constraints shape the design. Playback blocks for the length of the
sound, and the caller is the detection loop that owns the tkinter event loop
and the alert window — so it runs on a worker thread. And it has to stop
promptly, because a sound that outlives the behaviour it announces trains
the user to tune it out.
"""
from __future__ import annotations

import threading
import time

import pytest

from hairguard.sound import AlertSound

GAP = 0.02


def _counting_backend(calls: list[int]):
    def backend() -> None:
        calls.append(1)
    return backend


def test_starting_repeats_the_sound() -> None:
    calls: list[int] = []
    sound = AlertSound(enabled=True, backend=_counting_backend(calls), gap_sec=GAP)
    try:
        sound.start()
        time.sleep(0.25)
    finally:
        sound.stop()

    assert len(calls) > 2, f"прозвучало всего {len(calls)} раз — цикла нет"


def test_stopping_ends_the_sound() -> None:
    calls: list[int] = []
    sound = AlertSound(enabled=True, backend=_counting_backend(calls), gap_sec=GAP)
    sound.start()
    time.sleep(0.1)
    sound.stop()

    after_stop = len(calls)
    time.sleep(0.15)
    assert len(calls) == after_stop, "звук продолжился после остановки"


def test_stopping_is_prompt() -> None:
    """A sound that outlives the behaviour teaches the user to ignore it."""
    sound = AlertSound(enabled=True, backend=lambda: time.sleep(0.05), gap_sec=GAP)
    sound.start()
    time.sleep(0.1)

    began = time.monotonic()
    sound.stop()
    assert time.monotonic() - began < 0.5


def test_starting_twice_does_not_stack_two_loops() -> None:
    calls: list[int] = []
    sound = AlertSound(enabled=True, backend=_counting_backend(calls), gap_sec=0.05)
    try:
        sound.start()
        sound.start()
        sound.start()
        time.sleep(0.16)
    finally:
        sound.stop()

    # One loop at ~0.05s per cycle manages a handful; three would triple it.
    assert len(calls) <= 6, f"похоже, запустилось несколько циклов: {len(calls)}"


def test_it_can_be_started_again_after_stopping() -> None:
    calls: list[int] = []
    sound = AlertSound(enabled=True, backend=_counting_backend(calls), gap_sec=GAP)

    sound.start()
    time.sleep(0.08)
    sound.stop()
    first = len(calls)

    sound.start()
    time.sleep(0.08)
    sound.stop()

    assert len(calls) > first


def test_disabled_sound_stays_silent() -> None:
    calls: list[int] = []
    sound = AlertSound(enabled=False, backend=_counting_backend(calls), gap_sec=GAP)
    sound.start()
    time.sleep(0.1)
    sound.stop()

    assert calls == []


def test_active_reports_whether_it_is_sounding() -> None:
    sound = AlertSound(enabled=True, backend=lambda: None, gap_sec=GAP)
    assert sound.active is False
    sound.start()
    assert sound.active is True
    sound.stop()
    assert sound.active is False


def test_a_broken_sound_device_gives_up_instead_of_spinning() -> None:
    """No speakers means silence — not a thread burning a core in a tight loop."""
    calls: list[int] = []

    def broken() -> None:
        calls.append(1)
        raise OSError("нет звукового устройства")

    sound = AlertSound(enabled=True, backend=broken, gap_sec=GAP)
    try:
        sound.start()
        time.sleep(0.2)
    finally:
        sound.stop()

    assert len(calls) == 1, f"повторял сломанный бэкенд {len(calls)} раз"


def test_a_broken_device_does_not_reach_the_caller() -> None:
    def broken() -> None:
        raise OSError("нет звукового устройства")

    sound = AlertSound(enabled=True, backend=broken, gap_sec=GAP)
    sound.start()      # must not raise
    time.sleep(0.05)
    sound.stop()


def test_a_default_backend_is_available_on_every_platform() -> None:
    assert callable(AlertSound(enabled=True)._backend)
