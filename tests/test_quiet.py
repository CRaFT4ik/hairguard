"""Tests for hairguard.quiet — silencing native library chatter.

MediaPipe and TensorFlow Lite write their startup notes straight to the
operating system's stderr, below Python, so neither logging configuration
nor a replaced ``sys.stderr`` can stop them. Only the file descriptor can.

``capfd`` captures at that same descriptor level, which is what makes it the
right instrument here — ``capsys`` would not see these writes at all.
"""
from __future__ import annotations

import os

import pytest

from hairguard.quiet import suppressed_native_stderr


def _write_like_a_c_library(text: str) -> None:
    """Write the way MediaPipe does: straight to the descriptor."""
    os.write(2, text.encode())


@pytest.fixture(autouse=True)
def _not_verbose(monkeypatch):
    monkeypatch.delenv("HAIRGUARD_VERBOSE", raising=False)


def test_native_writes_are_swallowed(capfd) -> None:
    with suppressed_native_stderr():
        _write_like_a_c_library("noise from mediapipe\n")
    assert "noise" not in capfd.readouterr().err


def test_stderr_works_again_afterwards(capfd) -> None:
    with suppressed_native_stderr():
        _write_like_a_c_library("swallowed\n")
    _write_like_a_c_library("this must survive\n")

    err = capfd.readouterr().err
    assert "this must survive" in err
    assert "swallowed" not in err


def test_stderr_is_restored_even_after_an_error(capfd) -> None:
    """A failing detector must not leave the process mute."""
    with pytest.raises(RuntimeError):
        with suppressed_native_stderr():
            raise RuntimeError("boom")

    _write_like_a_c_library("still speaking\n")
    assert "still speaking" in capfd.readouterr().err


def test_python_stdout_is_untouched(capfd) -> None:
    """Model-download notices and app messages go to stdout and must remain."""
    with suppressed_native_stderr():
        print("модель скачивается")
    assert "модель скачивается" in capfd.readouterr().out


def test_verbose_env_keeps_the_native_logs(capfd, monkeypatch) -> None:
    """Debugging must stay possible without editing code."""
    monkeypatch.setenv("HAIRGUARD_VERBOSE", "1")
    with suppressed_native_stderr():
        _write_like_a_c_library("wanted while debugging\n")
    assert "wanted while debugging" in capfd.readouterr().err
