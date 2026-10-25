"""hairguard.diagnostics — the log file that stands in for the console.

Everything this app knows how to tell you, it tells you with ``print``.
Which detector loaded and why it was that one; that the camera vanished when
the laptop slept and came back; that a config field was out of range and got
replaced; that a clip was saved. Those lines are the app's entire account of
itself, and a double-clickable build has no console for them to land in — on
Windows a windowed executable has no standard output at all.

Losing them is not acceptable here. A constraint on this work was that a
packaged app must not be harder to diagnose than the source one, and the
person this is for runs it every day and will not be reading a traceback in
a terminal. So the streams are tee'd into ``~/.hairguard/hairguard.log``,
beside the config and the database, timestamped, bounded, and reachable from
the tray and from ``hairguard log``. When the app dies before the tray even
appears — a missing DLL, a camera that refuses, an unreadable model — the
traceback goes to that file *and* to a dialog box, because at that moment
there is nothing else on screen to ask.

The console is kept when there is one: running from a terminal must not
become quieter than it was before any of this existed.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

#: Overrides the log location. Used by the tests, and a way out for anyone
#: whose home directory is not where they want a log.
LOG_ENV = "HAIRGUARD_LOG"

LOG_NAME = "hairguard.log"

#: Rotated at half a megabyte, keeping one previous file. Months of daily
#: use at a handful of lines per start-up plus the occasional camera-loss
#: report; generous for diagnosis, small enough to attach to a message.
DEFAULT_MAX_BYTES = 512_000

_UNSET = object()


def is_frozen() -> bool:
    """True inside a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def log_path() -> Path:
    override = os.environ.get(LOG_ENV)
    if override:
        return Path(override)
    return Path.home() / ".hairguard" / LOG_NAME


def rotate(path: Path, max_bytes: int) -> None:
    """Move an oversized log aside, keeping exactly one previous file."""
    try:
        if not path.is_file() or path.stat().st_size < max_bytes:
            return
        previous = path.with_suffix(path.suffix + ".1")
        previous.unlink(missing_ok=True)
        path.replace(previous)
    except OSError:
        # A log that cannot be rotated is not a reason to refuse to start.
        pass


class _Tee:
    """Writes to a file and, when there is one, to the original stream.

    Buffers until a newline so that each complete line gets one timestamp
    rather than one per fragment — ``print(..., end="")`` is used in a few
    places and a stamp per piece would make the file unreadable.
    """

    def __init__(self, sink: TextIO, console: Optional[TextIO], tag: str) -> None:
        self._sink = sink
        self._console = console
        self._tag = tag
        self._buffer = ""
        self._lock = threading.Lock()

    # -- the parts of the file protocol that callers of print() rely on --

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        console = self._console
        if console is not None:
            try:
                console.write(text)
            except Exception:
                # A dead console must not take the log down with it.
                self._console = None
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit(line)
        return len(text)

    def flush(self) -> None:
        console = self._console
        if console is not None:
            try:
                console.flush()
            except Exception:
                self._console = None
        with self._lock:
            if self._buffer:
                self._emit(self._buffer)
                self._buffer = ""
        try:
            self._sink.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        console = self._console
        try:
            return bool(console is not None and console.isatty())
        except Exception:
            return False

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def fileno(self) -> int:
        # Deliberately refused. Something asking for a file descriptor wants
        # to bypass this object, and handing it the log file's descriptor
        # would interleave raw bytes with timestamped lines.
        raise OSError("hairguard log has no file descriptor")

    def _emit(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        try:
            self._sink.write(f"{stamp} {self._tag}{line}\n")
            self._sink.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


def install(
    *,
    path: Optional[Path] = None,
    console: Any = _UNSET,
    max_bytes: int = DEFAULT_MAX_BYTES,
    banner: bool = True,
) -> Path:
    """Tee stdout and stderr into the log file. Idempotent.

    ``console`` defaults to the real streams, which is what a terminal run
    wants; pass None to write only to the file. A windowed build has no
    streams to keep, so there the default already resolves to None.
    """
    target = Path(path) if path is not None else log_path()
    if _state:
        return _state["path"]

    target.parent.mkdir(parents=True, exist_ok=True)
    rotate(target, max_bytes)
    try:
        sink = open(target, "a", encoding="utf-8", errors="replace")
    except OSError as exc:
        # No log is survivable; refusing to start over it is not.
        print(f"[diagnostics] Журнал {target} не открывается: {exc}")
        return target

    out_console = sys.stdout if console is _UNSET else console
    err_console = sys.stderr if console is _UNSET else console

    _state.update(
        path=target,
        sink=sink,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # A segfault inside MediaPipe or a GPU driver kills the process without
    # running a single line of Python, so nothing above would record it: the
    # tray icon simply disappears. This at least leaves a C-level stack in
    # the log, which is the difference between "it vanished" and a bug report.
    try:
        import faulthandler

        if not faulthandler.is_enabled():
            faulthandler.enable(file=sink)
            _state["faulthandler"] = True
    except Exception:
        pass
    sys.stdout = _Tee(sink, out_console, "")  # type: ignore[assignment]
    sys.stderr = _Tee(sink, err_console, "! ")  # type: ignore[assignment]

    if banner:
        from hairguard import __version__

        kind = "упакованное" if is_frozen() else "из исходников"
        # Written to the file directly rather than printed. It is a header
        # for the log, not output: on the console it would prefix every
        # `hairguard model` with two lines of nothing anybody asked for, and
        # it would corrupt `doctor --json`, which something else parses.
        try:
            sink.write(
                f"=== HairGuard {__version__} "
                f"· {time.strftime('%Y-%m-%d %H:%M:%S')} · {sys.platform} "
                f"· {kind} · pid {os.getpid()} ===\n"
                f"=== {sys.executable} ===\n"
            )
            sink.flush()
        except Exception:
            pass
    return target


def uninstall() -> None:
    """Restore the original streams. Safe to call when nothing is installed."""
    if not _state:
        return
    try:
        flush()
    except Exception:
        pass
    if _state.get("faulthandler"):
        # The file it writes to is about to close.
        try:
            import faulthandler

            faulthandler.disable()
        except Exception:
            pass
    sys.stdout = _state.get("stdout") or sys.__stdout__
    sys.stderr = _state.get("stderr") or sys.__stderr__
    sink = _state.get("sink")
    if sink is not None:
        try:
            sink.close()
        except Exception:
            pass
    _state.clear()


def flush() -> None:
    """Push any buffered partial line into the file."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass


def installed() -> bool:
    return bool(_state)


# ---------------------------------------------------------------------------
# Dialogs and crashes
# ---------------------------------------------------------------------------


def show_error(title: str, message: str) -> bool:
    """Put a message where someone with no terminal will see it.

    Best effort by definition: the usual reason the app is dying is that the
    GUI toolkit it needs for the dialog is itself unhappy.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showerror(title, message)
        finally:
            root.destroy()
        return True
    except Exception:
        return False


def report_fatal(
    exc: Optional[BaseException] = None,
    *,
    dialog: bool = True,
    show: Optional[Callable[[str, str], Any]] = None,
    title: str = "HairGuard не запустился",
) -> str:
    """Record a crash and, optionally, show it. Returns what the user is told.

    The log gets the whole traceback; the dialog gets the sentence plus the
    path to the log, because "it does nothing when I double-click it" with
    nowhere to look is the state this whole module exists to prevent.
    """
    if exc is None:
        exc = sys.exc_info()[1]

    detail = (
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if exc is not None
        else "нет сведений об исключении"
    )
    summary = (
        f"{type(exc).__name__}: {exc}" if exc is not None else "неизвестная ошибка"
    )
    path = _state.get("path") or log_path()

    print(f"[fatal] {summary}", file=sys.stderr)
    print(detail, file=sys.stderr)
    flush()

    message = f"{summary}\n\nПодробности в журнале:\n{path}"
    if dialog:
        shower = show if show is not None else show_error
        try:
            shower(title, message)
        except Exception:
            pass
    return message


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------


def tail(limit: int = 40, *, path: Optional[Path] = None) -> list[str]:
    """The last ``limit`` lines of the log, or an empty list if there is none."""
    target = Path(path) if path is not None else log_path()
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    return lines[-limit:] if limit > 0 else lines


def reveal() -> bool:
    """Open the log in whatever the desktop uses for text files.

    Reached from the tray, which is the only route a user without a terminal
    has to the thing that explains what just happened.
    """
    path = _state.get("path") or log_path()
    try:
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("", encoding="utf-8")
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess

            subprocess.Popen(["open", str(path)])
        else:
            import subprocess

            subprocess.Popen(["xdg-open", str(path)])
        return True
    except Exception as exc:
        print(f"[diagnostics] Не удалось открыть журнал: {exc}")
        return False
