"""Tests for the entry points a packaged app needs.

Three things change when the app stops being ``python -m hairguard start``
typed in a source directory.

Double-clicking has to mean "start": the windowed executable gets no
arguments, and argparse's answer to no arguments is a usage error printed to
a console that does not exist.

Starting it twice has to say so. The PID guard already refuses, loudly, to a
console nobody is watching; from an icon that looks like nothing happened.

And ``hairguard debug`` has to stop being a trap. It opens its own camera,
which the running app is holding, and on Windows a second open of a held
camera returns black frames instead of failing — so the command appears to
work and shows an empty picture. The tray's window is the right answer while
the app is running, and this command now says so.
"""
from __future__ import annotations

import sys

import pytest

from hairguard import cli, entry


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def test_the_viewer_subcommand_exists() -> None:
    """The app spawns itself with this to put up the diagnostic window."""
    from hairguard.debugfeed import VIEWER_COMMAND

    args = cli.build_parser().parse_args([VIEWER_COMMAND])

    assert args.func is cli.cmd_debug_viewer


def test_the_viewer_subcommand_is_not_advertised() -> None:
    """Nobody types it, and a command list is a place a person looks for
    things they are meant to use."""
    from hairguard.debugfeed import VIEWER_COMMAND

    help_text = cli.build_parser().format_help()

    assert VIEWER_COMMAND not in help_text
    assert "SUPPRESS" not in help_text


def test_the_log_subcommand_exists() -> None:
    args = cli.build_parser().parse_args(["log"])

    assert args.func is cli.cmd_log


def test_doctor_is_advertised() -> None:
    """Unlike the viewer, this is one to reach for, so it has to be findable."""
    assert "doctor" in cli.build_parser().format_help()


def test_no_arguments_is_still_an_error_on_the_command_line() -> None:
    """Only the windowed build treats "no arguments" as a request to start."""
    with pytest.raises(SystemExit):
        cli.main([])


def test_a_default_turns_no_arguments_into_start(monkeypatch) -> None:
    started: list[object] = []
    monkeypatch.setattr(cli, "cmd_start", lambda args: started.append(args))

    cli.main([], default_argv=["start"])

    assert len(started) == 1


def test_an_explicit_command_beats_the_default(monkeypatch) -> None:
    """``HairGuard.exe stop`` must stop, not start."""
    called: list[str] = []
    monkeypatch.setattr(cli, "cmd_stop", lambda args: called.append("stop"))
    monkeypatch.setattr(cli, "cmd_start", lambda args: called.append("start"))

    cli.main(["stop"], default_argv=["start"])

    assert called == ["stop"]


# ---------------------------------------------------------------------------
# hairguard debug, while the app is running
# ---------------------------------------------------------------------------


class FakePid:
    def __init__(self, running: bool, pid: int = 1234) -> None:
        self._running = running
        self._pid = pid

    def is_running(self) -> bool:
        return self._running

    def read(self) -> int | None:
        return self._pid if self._running else None


def test_debug_refuses_to_open_a_second_camera_on_the_running_app(
    monkeypatch, capsys
) -> None:
    opened: list[object] = []
    monkeypatch.setattr(cli, "_pid_file", lambda: FakePid(running=True))
    monkeypatch.setattr(
        "hairguard.debugview.main", lambda camera=None: opened.append(camera)
    )

    code = cli.cmd_debug(cli.build_parser().parse_args(["debug"]))

    assert opened == [], "it opened a camera the app was already holding"
    out = capsys.readouterr().out
    assert "трее" in out, "it did not point at the tray entry"
    assert "чёрные" in out or "черные" in out, "it did not say why"
    assert code != 0


def test_debug_opens_normally_when_the_app_is_not_running(monkeypatch) -> None:
    """Which is what the command is for: looking at the pipeline with no app
    in the way."""
    opened: list[object] = []
    monkeypatch.setattr(cli, "_pid_file", lambda: FakePid(running=False))
    monkeypatch.setattr(
        "hairguard.debugview.main", lambda camera=None: opened.append(camera)
    )

    cli.cmd_debug(cli.build_parser().parse_args(["debug", "--camera", "2"]))

    assert opened == [2]


def test_debug_can_be_forced(monkeypatch) -> None:
    """The refusal is a guard, not a rule: two cameras on one machine is a
    real configuration and the user may know better."""
    opened: list[object] = []
    monkeypatch.setattr(cli, "_pid_file", lambda: FakePid(running=True))
    monkeypatch.setattr(
        "hairguard.debugview.main", lambda camera=None: opened.append(camera)
    )

    cli.cmd_debug(cli.build_parser().parse_args(["debug", "--force"]))

    assert opened == [None]


# ---------------------------------------------------------------------------
# hairguard log
# ---------------------------------------------------------------------------


def test_log_prints_the_path_and_the_last_lines(monkeypatch, capsys, tmp_path) -> None:
    """The one command to ask for when someone says "it does nothing"."""
    from hairguard import diagnostics

    path = tmp_path / "hairguard.log"
    path.write_text("\n".join(f"строка {n}" for n in range(50)), encoding="utf-8")
    monkeypatch.setenv(diagnostics.LOG_ENV, str(path))

    cli.cmd_log(cli.build_parser().parse_args(["log", "--tail", "3"]))

    out = capsys.readouterr().out
    assert str(path) in out
    assert "строка 49" in out
    assert "строка 46" not in out


def test_log_says_so_when_there_is_no_log(monkeypatch, capsys, tmp_path) -> None:
    from hairguard import diagnostics

    monkeypatch.setenv(diagnostics.LOG_ENV, str(tmp_path / "absent.log"))

    cli.cmd_log(cli.build_parser().parse_args(["log"]))

    assert "пуст" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The windowed entry point
# ---------------------------------------------------------------------------


@pytest.fixture
def quiet_log(tmp_path, monkeypatch):
    from hairguard import diagnostics

    monkeypatch.setenv(diagnostics.LOG_ENV, str(tmp_path / "hairguard.log"))
    yield tmp_path / "hairguard.log"
    diagnostics.uninstall()


def test_the_windowed_entry_starts_the_app(monkeypatch, quiet_log) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(entry, "_already_running", lambda: None)
    monkeypatch.setattr(cli, "main", lambda argv, default_argv=None: seen.append(default_argv))

    assert entry.gui_main([]) == 0
    assert seen == [["start"]]


def test_the_windowed_entry_captures_output_into_the_log(monkeypatch, quiet_log) -> None:
    """With no console, the startup lines — which detector loaded, which
    camera — would otherwise be lost, and they are the whole diagnosis."""
    monkeypatch.setattr(entry, "_already_running", lambda: None)
    monkeypatch.setattr(
        cli, "main", lambda argv, default_argv=None: print("[app] Детектор: модель")
    )

    entry.gui_main([])

    assert "Детектор: модель" in quiet_log.read_text(encoding="utf-8")


def test_a_crash_before_the_tray_appears_gets_a_dialog(monkeypatch, quiet_log) -> None:
    """The failure mode this exists for: a double-click that does nothing,
    with no console to explain it."""
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(entry, "_already_running", lambda: None)
    monkeypatch.setattr(
        "hairguard.diagnostics.show_error",
        lambda title, message: shown.append((title, message)) or True,
    )

    def explode(argv, default_argv=None):
        raise RuntimeError("Cannot open camera 0")

    monkeypatch.setattr(cli, "main", explode)

    code = entry.gui_main([])

    assert code != 0
    assert shown, "the user was shown nothing at all"
    assert "Cannot open camera 0" in shown[0][1]
    assert str(quiet_log) in shown[0][1]
    text = quiet_log.read_text(encoding="utf-8")
    assert "Traceback" in text


def test_a_clean_exit_is_not_reported_as_a_crash(monkeypatch, quiet_log) -> None:
    shown: list[object] = []
    monkeypatch.setattr(entry, "_already_running", lambda: None)
    monkeypatch.setattr(
        "hairguard.diagnostics.show_error", lambda t, m: shown.append((t, m))
    )

    def quit_cleanly(argv, default_argv=None):
        raise SystemExit(0)

    monkeypatch.setattr(cli, "main", quit_cleanly)

    assert entry.gui_main([]) == 0
    assert shown == []


def test_double_clicking_a_running_app_says_it_is_already_running(
    monkeypatch, quiet_log
) -> None:
    """From an icon, the PID guard's console message is invisible: the second
    double-click just looks like nothing happened."""
    shown: list[tuple[str, str]] = []
    started: list[int] = []
    monkeypatch.setattr(entry, "_already_running", lambda: 4321)
    monkeypatch.setattr(
        entry, "show_notice", lambda title, message: shown.append((title, message))
    )
    monkeypatch.setattr(cli, "main", lambda *a, **k: started.append(1))

    assert entry.gui_main([]) == 0
    assert started == [], "it started a second instance"
    assert shown and "4321" in shown[0][1]


def test_the_running_check_never_removes_the_pid_file(monkeypatch, tmp_path) -> None:
    """Deleting that file has caused a real incident here: it is the only
    thing standing between one camera and two instances fighting over it."""
    from hairguard.pid import PidFile

    path = tmp_path / "hairguard.pid"
    path.write_text(str(99999999), encoding="utf-8")
    monkeypatch.setattr(entry, "PidFile", lambda: PidFile(path))

    entry._already_running()

    assert path.exists()


def test_the_console_entry_keeps_the_console(monkeypatch, quiet_log, capsys) -> None:
    """`hairguard model` in a terminal must still answer in the terminal."""
    monkeypatch.setattr(cli, "main", lambda argv, default_argv=None: print("Детектор: модель"))

    entry.console_main(["model"])

    assert "Детектор: модель" in capsys.readouterr().out
    assert "Детектор: модель" in quiet_log.read_text(encoding="utf-8")


def test_the_viewer_is_launched_from_the_console_entry(monkeypatch, quiet_log) -> None:
    """The feed spawns the console binary, so this path has to reach the
    viewer rather than the tray app."""
    from hairguard.debugfeed import VIEWER_COMMAND

    ran: list[int] = []
    monkeypatch.setattr("hairguard.debugviewer.main", lambda: ran.append(1) or 0)

    cli.main([VIEWER_COMMAND])

    assert ran == [1]
