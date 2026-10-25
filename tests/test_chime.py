"""Tests for the synthesised alert chime.

The previous sound used winsound.Beep, which emits a square wave and cuts it
off mid-cycle. Both are audible defects: the square wave is harsh, and the
abrupt start and end leave a step in the waveform that the speaker reproduces
as a click.

Both are properties of the samples, so both can be asserted here.
"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from hairguard.sound import SAMPLE_RATE, synthesize_chime


@pytest.fixture(scope="module")
def samples() -> np.ndarray:
    with wave.open(io.BytesIO(synthesize_chime()), "rb") as handle:
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == SAMPLE_RATE
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float64)


def test_it_is_a_readable_wav_file() -> None:
    with wave.open(io.BytesIO(synthesize_chime()), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getnframes() > 0


def test_it_starts_at_silence(samples: np.ndarray) -> None:
    """A non-zero first sample is the click at the start."""
    assert abs(samples[0]) < 1


def test_it_ends_at_silence(samples: np.ndarray) -> None:
    """A non-zero last sample is the click at the end."""
    assert abs(samples[-1]) < 1


def test_the_opening_ramps_up_instead_of_jumping(samples: np.ndarray) -> None:
    """The attack must be gradual — that is what removes the pop."""
    attack = samples[: int(0.010 * SAMPLE_RATE)]
    early = np.abs(attack[: len(attack) // 4]).max()
    late = np.abs(attack[-len(attack) // 4:]).max()
    assert early < late


def test_the_ending_fades_out_instead_of_stopping(samples: np.ndarray) -> None:
    tail = samples[-int(0.030 * SAMPLE_RATE):]
    first_half = np.abs(tail[: len(tail) // 2]).max()
    second_half = np.abs(tail[len(tail) // 2:]).max()
    assert second_half < first_half


def test_no_sample_to_sample_jumps(samples: np.ndarray) -> None:
    """A step between neighbouring samples is heard as a crack."""
    biggest_step = np.abs(np.diff(samples)).max()
    full_scale = 32767
    assert biggest_step < full_scale * 0.25, "волна рвётся — будет треск"


def test_it_does_not_clip(samples: np.ndarray) -> None:
    """Clipping turns a smooth tone back into a harsh one."""
    assert np.abs(samples).max() < 32767


def test_it_is_actually_audible(samples: np.ndarray) -> None:
    assert np.abs(samples).max() > 32767 * 0.2


def test_it_is_short_enough_to_repeat(samples: np.ndarray) -> None:
    duration = len(samples) / SAMPLE_RATE
    assert 0.2 < duration < 1.0


def test_it_is_deterministic() -> None:
    assert synthesize_chime() == synthesize_chime()
