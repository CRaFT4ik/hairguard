"""hairguard.entry — the two ways a packaged HairGuard is started.

A bundle ships two executables from one tree of libraries. ``HairGuard`` is
windowed: it is the icon a person double-clicks, it takes no arguments, and
it has no console at all. ``hairguard`` is the console one: it is the whole
command line, unchanged, and it exists because a packaged app that cannot be
asked what went wrong is worse than a script.

What the windowed entry adds over simply calling the CLI is three answers to
things that only became problems once there was no terminal.

*Output needs somewhere to go.* Everything this app knows how to tell you it
tells you with ``print``. See :mod:`hairguard.diagnostics`.

*A crash before the tray appears needs saying out loud.* A camera that will
not open, a missing runtime library in a fresh bundle, an unreadable config:
from an icon, all of those look identical to a double-click that did
nothing. So the traceback goes to the log and a sentence of it goes to a
dialog box.

*A second double-click needs an answer.* The PID guard already refuses to
start twice, which is right, and says so to a console nobody is watching.
From an icon the refusal is invisible and indistinguishable from the first
failure mode. It is now a dialog naming the process that already holds the
camera — and nothing here touches that PID file, which is the single thing
standing between one webcam and two instances fighting over it.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from hairguard import cli, diagnostics
from hairguard.pid import PidFile


def show_notice(title: str, message: str) -> bool:
    """An informational dialog. Best effort, like everything in a dying app."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showinfo(title, message)
        finally:
            root.destroy()
        return True
    except Exception:
        return False


def _already_running() -> Optional[int]:
    """The PID of a live instance, or None.

    Read-only, deliberately and permanently: removing
    ``~/.hairguard/hairguard.pid`` has caused a real incident here.
    """
    try:
        guard = PidFile()
        if guard.is_running():
            return guard.read()
    except Exception:
        return None
    return None


def gui_main(argv: Optional[Sequence[str]] = None) -> int:
    """The windowed entry point. No console, no arguments, no traceback."""
    diagnostics.install(console=None)

    arguments = list(sys.argv[1:] if argv is None else argv)

    # Only a bare double-click gets the "already running" dialog. An explicit
    # command — someone wired `HairGuard.exe stop` to a shortcut — is passed
    # straight through.
    if not arguments:
        pid = _already_running()
        if pid is not None:
            print(f"[entry] HairGuard уже запущен (PID {pid}) — второй экземпляр не нужен.")
            show_notice(
                "HairGuard уже запущен",
                f"HairGuard работает (PID {pid}).\n\n"
                "Значок находится в системном трее, рядом с часами. "
                "Второй экземпляр не запускается: камеру можно "
                "открыть только одному.",
            )
            return 0

    try:
        return int(cli.main(arguments, default_argv=["start"]) or 0)
    except SystemExit as exc:
        # argparse and the PID guard both leave this way. A non-zero code
        # here has already been explained to the log by whoever raised it.
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(f"[entry] {code}")
        return 1
    except KeyboardInterrupt:
        return 0
    except BaseException as exc:  # noqa: BLE001 - last line before silence
        diagnostics.report_fatal(exc)
        return 1
    finally:
        diagnostics.flush()


def console_main(argv: Optional[Sequence[str]] = None) -> int:
    """The console entry point: the command line, plus a copy in the log.

    The log copy is what makes a packaged app diagnosable after the fact —
    the console scrolls away, and the question is usually about yesterday.
    """
    diagnostics.install()
    try:
        return int(cli.main(argv) or 0)
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else (code if isinstance(code, int) else 1)
    except KeyboardInterrupt:
        print()
        return 130
    finally:
        diagnostics.flush()


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(console_main())
