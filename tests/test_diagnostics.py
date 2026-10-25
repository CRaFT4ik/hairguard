"""Tests for the log file that replaces the console in a packaged app.

Every diagnostic this app has is a ``print``: which detector loaded and why,
whether clips are being recorded, that the camera disappeared, what the
config rejected. A double-clickable build has no console, so those lines
would go nowhere — and "it does nothing when I start it" with no way to ask
why is the worst state this app can be in for someone who runs it daily and
cannot read a traceback.

So the lines go to a file instead, the frozen app points at it, and a crash
before the tray even appears gets a dialog rather than silence.
"""
from __future__ import annotations

import io
import sys

import pytest

from hairguard import diagnostics


@pytest.fixture
def log(tmp_path, monkeypatch):
    """Redirect the log somewhere disposable, and always unwrap afterwards."""
    path = tmp_path / "hairguard.log"
    monkeypatch.setenv(diagnostics.LOG_ENV, str(path))
    yield path
    diagnostics.uninstall()


# ---------------------------------------------------------------------------
# Where the log lives
# ---------------------------------------------------------------------------


def test_the_log_sits_beside_the_config_and_the_database(monkeypatch, tmp_path):
    """One directory to look in. ~/.hairguard already holds everything else
    a person might be asked to send."""
    monkeypatch.delenv(diagnostics.LOG_ENV, raising=False)
    monkeypatch.setattr(diagnostics.Path, "home", classmethod(lambda cls: tmp_path))

    assert diagnostics.log_path() == tmp_path / ".hairguard" / "hairguard.log"


def test_the_log_location_can_be_overridden(log):
    assert diagnostics.log_path() == log


# ---------------------------------------------------------------------------
# Capturing what the app says
# ---------------------------------------------------------------------------


def test_installing_captures_print_into_the_log(log):
    diagnostics.install(console=None)

    print("[app] Детектор: модель twistnet")
    diagnostics.flush()

    assert "Детектор: модель twistnet" in log.read_text(encoding="utf-8")


def test_the_log_records_a_session_banner(log):
    """The first question about any log is which build wrote it."""
    diagnostics.install(console=None)
    diagnostics.flush()

    text = log.read_text(encoding="utf-8")
    assert "HairGuard" in text
    assert sys.platform in text
    from hairguard import __version__

    assert __version__ in text


def test_every_line_is_timestamped(log):
    """Episodes, camera losses and model fallbacks are only diagnosable
    against each other in time."""
    diagnostics.install(console=None)

    print("first")
    print("second")
    diagnostics.flush()

    lines = [
        line
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.endswith(("first", "second"))
    ]
    assert len(lines) == 2
    for line in lines:
        stamp = line.split(" ", 1)[0]
        hours, minutes, seconds = stamp.split(":")
        assert len(hours) == 2 and len(minutes) == 2 and len(seconds) == 2


def test_partial_writes_are_held_until_the_line_ends(log):
    """``print(..., end="")`` is used for progress-ish output; a timestamp
    per fragment would make the file unreadable."""
    diagnostics.install(console=None)

    print("Проверка камеры 0: ", end="")
    print("live")
    diagnostics.flush()

    matching = [
        line
        for line in log.read_text(encoding="utf-8").splitlines()
        if "Проверка камеры" in line
    ]
    assert matching == [matching[0]]
    assert matching[0].endswith("Проверка камеры 0: live")


def test_stderr_is_captured_too(log):
    """Tracebacks are the half of the output that matters most here."""
    diagnostics.install(console=None)

    print("тревога", file=sys.stderr)
    diagnostics.flush()

    assert "тревога" in log.read_text(encoding="utf-8")


def test_the_console_still_gets_everything_when_there_is_one(log):
    """Running from a terminal must not become quieter than it was."""
    console = io.StringIO()
    diagnostics.install(console=console)

    print("видно и там и там")
    diagnostics.flush()

    assert "видно и там и там" in console.getvalue()
    assert "видно и там и там" in log.read_text(encoding="utf-8")


def test_a_missing_console_is_not_an_error(log):
    """A windowed build has no stdout at all. ``print`` must still work, and
    must not take the detection tick down with it."""
    diagnostics.install(console=None)

    print("no console here")  # the real test is that this does not raise
    diagnostics.flush()

    assert "no console here" in log.read_text(encoding="utf-8")


def test_installing_twice_does_not_nest_the_capture(log):
    """Called from both entry points and from the viewer; a doubled tee
    would write every line twice and grow the file for nothing."""
    diagnostics.install(console=None)
    first = sys.stdout
    diagnostics.install(console=None)

    print("однажды")
    diagnostics.flush()

    assert sys.stdout is first
    assert log.read_text(encoding="utf-8").count("однажды") == 1


def test_uninstalling_restores_the_original_streams(log):
    original = sys.stdout
    diagnostics.install(console=None)
    assert sys.stdout is not original

    diagnostics.uninstall()

    assert sys.stdout is original


# ---------------------------------------------------------------------------
# Keeping the file bounded
# ---------------------------------------------------------------------------


def test_an_oversized_log_is_rotated_rather_than_grown(log):
    """This runs all day, every day, for months."""
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("x" * 2048, encoding="utf-8")

    diagnostics.install(console=None, max_bytes=1024)
    print("после ротации")
    diagnostics.flush()

    previous = log.with_suffix(log.suffix + ".1")
    assert previous.exists()
    assert previous.read_text(encoding="utf-8").startswith("xxx")
    assert "после ротации" in log.read_text(encoding="utf-8")
    assert "xxx" not in log.read_text(encoding="utf-8")


def test_rotation_keeps_only_one_previous_log(log):
    log.parent.mkdir(parents=True, exist_ok=True)
    previous = log.with_suffix(log.suffix + ".1")
    previous.write_text("старое", encoding="utf-8")
    log.write_text("y" * 2048, encoding="utf-8")

    diagnostics.install(console=None, max_bytes=1024)
    diagnostics.flush()

    assert "старое" not in previous.read_text(encoding="utf-8")
    assert not log.with_suffix(log.suffix + ".2").exists()


def test_a_small_log_is_appended_to(log):
    """Yesterday's start-up is often what explains today's behaviour."""
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("вчерашняя строка\n", encoding="utf-8")

    diagnostics.install(console=None, max_bytes=1_000_000)
    print("сегодняшняя строка")
    diagnostics.flush()

    text = log.read_text(encoding="utf-8")
    assert "вчерашняя строка" in text
    assert "сегодняшняя строка" in text


# ---------------------------------------------------------------------------
# Crashing where nobody can see
# ---------------------------------------------------------------------------


def test_report_fatal_writes_the_traceback_to_the_log(log):
    diagnostics.install(console=None)

    try:
        raise RuntimeError("камера 0 не открывается")
    except RuntimeError as exc:
        message = diagnostics.report_fatal(exc, dialog=False)

    diagnostics.flush()
    text = log.read_text(encoding="utf-8")
    assert "камера 0 не открывается" in text
    assert "Traceback" in text
    assert "RuntimeError" in text
    assert "камера 0 не открывается" in message


def test_report_fatal_names_the_log_in_what_it_shows_the_user(log):
    """The dialog is the only thing a person with no terminal will see, so it
    has to end with somewhere to look."""
    diagnostics.install(console=None)
    shown: list[tuple[str, str]] = []

    message = diagnostics.report_fatal(
        ValueError("модель не загрузилась"),
        dialog=True,
        show=lambda title, body: shown.append((title, body)),
    )

    assert str(log) in message
    assert shown and str(log) in shown[0][1]
    assert "модель не загрузилась" in shown[0][0] + shown[0][1]


def test_report_fatal_survives_a_dialog_that_cannot_be_shown(log):
    """Headless, no display, tkinter missing — the log entry still happened
    and that is the part that matters."""
    diagnostics.install(console=None)

    def broken_show(title, body):
        raise RuntimeError("no display")

    message = diagnostics.report_fatal(
        ValueError("boom"), dialog=True, show=broken_show
    )

    diagnostics.flush()
    assert "boom" in message
    assert "boom" in log.read_text(encoding="utf-8")


def test_report_fatal_works_without_an_exception_object(log):
    diagnostics.install(console=None)

    try:
        raise KeyError("нет такого ключа")
    except KeyError:
        message = diagnostics.report_fatal(dialog=False)

    assert "KeyError" in message


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------


def test_tail_returns_the_last_lines(log):
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(f"строка {n}" for n in range(100)), encoding="utf-8")

    assert diagnostics.tail(limit=3) == ["строка 97", "строка 98", "строка 99"]


def test_tail_of_a_missing_log_is_empty(log):
    assert diagnostics.tail(limit=5) == []


# ---------------------------------------------------------------------------
# Knowing which kind of process this is
# ---------------------------------------------------------------------------


def test_is_frozen_is_false_when_running_from_source():
    assert diagnostics.is_frozen() is False


def test_is_frozen_follows_the_bundler_flag(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert diagnostics.is_frozen() is True
