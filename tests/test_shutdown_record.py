"""Tests for the app leaving a record of how it ended.

Found by running the packaged build and watching the tray icon disappear
with nothing in the log at all — not a traceback, not a goodbye, nothing.
A tray application that exits leaves no window behind to mark the occasion,
so without a line here "it stopped working at some point today" and "it
crashed" are the same observation, and neither is actionable.

Two halves. A line on the way out says it was asked to stop and by what. A
``faulthandler`` armed at the log says it was not: a segfault in MediaPipe
or a graphics driver kills the process without running any Python, so
nothing written in Python can record it, but a C-level stack still can.
"""
from __future__ import annotations

import sys

import pytest

from hairguard import diagnostics
from hairguard.app import HairGuardApp


class Dummy:
    """Absorbs whatever shutdown calls on it."""

    def __getattr__(self, name):
        def anything(*args, **kwargs):
            return None

        return anything


@pytest.fixture
def app():
    instance = HairGuardApp.__new__(HairGuardApp)
    for name in ("_sound", "_capture", "_camera", "_detector", "_hand_detector",
                 "_face_detector", "_twist_model", "_tray", "_pid_file", "_db",
                 "_debug_feed"):
        setattr(instance, name, Dummy())
    instance._tick_count = 4242
    return instance


@pytest.fixture
def log(tmp_path, monkeypatch):
    path = tmp_path / "hairguard.log"
    monkeypatch.setenv(diagnostics.LOG_ENV, str(path))
    yield path
    diagnostics.uninstall()


# ---------------------------------------------------------------------------
# The way out
# ---------------------------------------------------------------------------


def test_shutting_down_is_recorded(app, capsys) -> None:
    app.shutdown()

    assert "Завершение работы" in capsys.readouterr().out


def test_the_reason_is_recorded(app, capsys) -> None:
    """"It exited" and "it was killed" are different problems."""
    app.shutdown("«Выход» в меню трея")

    assert "«Выход» в меню трея" in capsys.readouterr().out


def test_how_long_it_ran_is_recorded(app, capsys) -> None:
    """A tray app that dies after four ticks and one that dies after a
    million are not the same bug."""
    app.shutdown()

    assert "4242" in capsys.readouterr().out


def test_it_is_said_once_even_though_shutdown_is_called_twice(app, capsys) -> None:
    """``run()`` calls it in a finally and callers call it directly."""
    app.shutdown()
    app.shutdown()

    assert capsys.readouterr().out.count("Завершение работы") == 1


def test_the_tray_exit_names_itself(app) -> None:
    import queue

    app._tray_events = queue.Queue()
    app._stop_reason = "unset"

    assert app._handle_tray_event("exit") is False
    assert "трея" in app._stop_reason


def test_a_signal_names_itself(app) -> None:
    import queue

    app._tray_events = queue.Queue()

    app._handle_signal(15, None)

    assert "15" in app._stop_reason
    assert app._tray_events.get_nowait() == "exit"


def test_a_broken_log_does_not_leave_the_camera_open(app, monkeypatch) -> None:
    """The record is the lesser of shutdown's two jobs, and it goes first.

    Written as a test because the first version had it outside a try/except:
    a log file that could not be written — a full disk, a read-only home —
    would have aborted shutdown at line one, leaving the webcam held and the
    database locked.
    """
    released: list[int] = []
    app._camera = type("C", (), {"release": lambda self: released.append(1)})()

    def broken_print(*args, **kwargs):
        raise OSError("log is gone")

    monkeypatch.setattr("builtins.print", broken_print)

    app.shutdown()  # must not raise

    assert released == [1]


# ---------------------------------------------------------------------------
# Crashes that run no Python
# ---------------------------------------------------------------------------


def test_installing_arms_a_crash_handler(log) -> None:
    """MediaPipe is 28 MB of C++ called on every frame."""
    import faulthandler

    was_enabled = faulthandler.is_enabled()
    if was_enabled:
        faulthandler.disable()
    try:
        diagnostics.install(console=None)

        assert faulthandler.is_enabled()
    finally:
        diagnostics.uninstall()
        if was_enabled:
            faulthandler.enable()


def test_a_native_crash_leaves_a_stack_in_the_log(log, tmp_path) -> None:
    """Exercised for real, in a child process, because the only way to know
    this works is to crash something."""
    import subprocess
    import textwrap

    script = textwrap.dedent(
        f"""
        import ctypes, faulthandler, os, sys
        sys.path.insert(0, {str(tmp_path.parent.parent)!r})
        os.environ["HAIRGUARD_LOG"] = {str(log)!r}
        from hairguard import diagnostics
        diagnostics.install(console=None)
        print("about to die")
        ctypes.string_at(0)        # segfault, no Python unwinding
        """
    )
    repo = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    finished = subprocess.run(
        [sys.executable, "-c", script.replace(str(tmp_path.parent.parent), repo)],
        capture_output=True,
        text=True,
        cwd=repo,
    )

    assert finished.returncode != 0, "the child was supposed to crash"
    written = log.read_text(encoding="utf-8", errors="replace")
    assert "about to die" in written
    assert "Current thread" in written or "Fatal" in written, written[-800:]


def test_uninstalling_disarms_what_it_armed(log) -> None:
    """A handler still pointed at a closed file descriptor would write a
    crash report into whatever got that descriptor next."""
    import faulthandler

    was_enabled = faulthandler.is_enabled()
    if was_enabled:
        faulthandler.disable()
    try:
        diagnostics.install(console=None)
        assert faulthandler.is_enabled()

        diagnostics.uninstall()

        assert not faulthandler.is_enabled()
    finally:
        if was_enabled:
            faulthandler.enable()


def test_a_handler_somebody_else_armed_is_left_alone(log) -> None:
    """pytest arms one, and so might an embedding application. Pointing it
    somewhere else and then disabling it would take away a facility this
    module did not provide."""
    import faulthandler

    if not faulthandler.is_enabled():
        faulthandler.enable()

    diagnostics.install(console=None)
    diagnostics.uninstall()

    assert faulthandler.is_enabled()
