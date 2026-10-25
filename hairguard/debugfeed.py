"""hairguard.debugfeed — the app's half of the diagnostic window.

When something looks wrong, the only question worth asking is what the
detector is seeing *right now*. Answering it has one hard constraint and one
hard prohibition.

The constraint: the running app holds the webcam. A window that opened its
own camera would, on Windows, get black frames rather than an error — the
same trap that once had this project's camera auto-detection calling a live
sensor blank because another instance held it. So the viewer is a separate
process that is given no camera at all, and every pixel and number it draws
is pushed to it from the detection tick that already decided. Fidelity comes
free, and the wrong thing is unreachable rather than merely discouraged.

The prohibition: that same tick drives the tkinter event loop, the fullscreen
overlay and the alert chime. Clip encoding on the main thread and a
notification library stealing the overlay's topmost status have each broken
this app's core once. So :meth:`DebugFeed.publish` stores a reference and
sets an event — no encoding, no I/O, no allocation worth measuring — and a
daemon thread does the JPEG and the pipe write. A viewer that stops reading
blocks that thread, where blocking is free; frames pile up nowhere because
only the newest is kept. A viewer that dies closes the feed. Nothing the
viewer can do arrives at the tick as an exception or a delay.

Chosen over two alternatives. A cv2 window driven from the tick itself is
the obvious shape and was rejected outright: ``waitKey`` in the detection
loop, HighGUI sharing the main thread with tkinter, and a window-manager
crash taking the alert down with it. A loopback socket, which would allow
attaching a viewer to an already-running app, was rejected because it means
a listening port and an authentication token on a machine where the thing
being served is a live webcam feed; a pipe to a child process cannot be
connected to by anything else and needs no secret.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from hairguard import debugwire

#: The hidden subcommand that turns a hairguard process into the viewer.
VIEWER_COMMAND = "debug-viewer"

#: How long a closing feed waits for the viewer to notice end-of-feed and
#: leave on its own before it is terminated.
_GRACE_SEC = 0.3

#: Idle wake-up interval for the writer thread. Only used to notice that the
#: viewer has exited while nothing is being published — a paused app, for
#: instance, which publishes nothing at all.
_POLL_SEC = 0.25


def viewer_command() -> list[str]:
    """The argv that starts a viewer for this build of the app.

    Frozen builds ship two executables from one bundle: the windowed one the
    user double-clicks, and a console one for diagnosis. The viewer is
    spawned from the console binary when it can be found, because a windowed
    Windows executable is not guaranteed a usable standard input, and
    standard input is the whole channel.
    """
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        for name in ("hairguard-cli.exe", "hairguard-cli"):
            candidate = executable.parent / name
            if candidate.is_file():
                return [str(candidate), VIEWER_COMMAND]
        return [str(executable), VIEWER_COMMAND]
    return [sys.executable, "-m", "hairguard", VIEWER_COMMAND]


def _stderr_target() -> Any:
    """Send the child's stderr to the log file, or nowhere.

    A viewer that fails before it can install its own logging — a missing
    DLL in a fresh bundle, say — would otherwise write its traceback to a
    handle that does not exist in a windowed build. This is the difference
    between "the menu entry does nothing" and a line in the log saying why.
    """
    try:
        from hairguard import diagnostics

        path = diagnostics.log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "ab")
    except Exception:
        return subprocess.DEVNULL


def spawn_viewer(command: Optional[list[str]] = None) -> subprocess.Popen:
    """Start the viewer process with a pipe for its standard input."""
    argv = command if command is not None else viewer_command()
    kwargs: dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.DEVNULL,
        "stderr": _stderr_target(),
        "close_fds": True,
    }
    if sys.platform == "win32":
        # No console window, even though the console binary is what runs.
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    return subprocess.Popen(argv, **kwargs)


class DebugFeed:
    """A viewer process and the one-slot mailbox feeding it."""

    def __init__(
        self,
        *,
        spawn: Optional[Callable[[], Any]] = None,
        encode: Optional[Callable[..., bytes]] = None,
        log: Callable[[str], None] = print,
        max_width: int = debugwire.DEFAULT_MAX_WIDTH,
        quality: int = debugwire.DEFAULT_QUALITY,
    ) -> None:
        self._spawn = spawn if spawn is not None else spawn_viewer
        self._encode = encode if encode is not None else debugwire.encode
        self._log = log
        self._max_width = max_width
        self._quality = quality

        self._slot_lock = threading.Lock()
        self._proc_lock = threading.Lock()
        self._pending: Optional[debugwire.Snapshot] = None
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[Any] = None
        self._alive = False
        self._started = False
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True from :meth:`open` until the viewer closes or fails.

        Deliberately true before the process exists: the detection tick asks
        this to decide whether to build a snapshot, and the answer must not
        depend on winning a race with ``Popen``.
        """
        return self._alive

    @property
    def started(self) -> bool:
        """True once the viewer process has actually been spawned."""
        return self._started

    @property
    def error(self) -> Optional[str]:
        """Why the feed closed, or None if it closed cleanly or is open."""
        return self._error

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Ask for a viewer. Returns immediately; the spawn happens on the thread.

        ``subprocess.Popen`` is tens of milliseconds, and the caller is the
        detection tick — which, at the moment the user reaches for the tray,
        may be holding a fullscreen alert on screen.
        """
        if self._alive:
            return
        self._error = None
        self._started = False
        self._alive = True
        self._wake.clear()
        thread = threading.Thread(target=self._run, daemon=True, name="debug-feed")
        self._thread = thread
        thread.start()

    def close(self, timeout: float = 1.0) -> None:
        """Stop the feed and the viewer. Safe to call twice, and from any thread.

        Runs on the way out of the tkinter mainloop among other places, so it
        does not wait on anything indefinitely: a viewer wedged inside a
        window-manager call must not hold the app's exit open.
        """
        self._alive = False
        self._wake.set()
        thread, self._thread = self._thread, None
        # Closing the pipe is what unsticks a writer blocked inside write(),
        # so it happens before the join rather than after it.
        self._shutdown_proc(timeout=timeout)
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout)
        with self._slot_lock:
            self._pending = None
        self._started = False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, snapshot: debugwire.Snapshot) -> None:
        """Offer one tick of detector state. Never blocks, never raises.

        Keeps a reference and nothing more. The frame is not copied: OpenCV
        hands out a fresh array per ``read()``, and the detectors only read
        from it, so the worst case is that the viewer draws a frame a moment
        newer than the landmarks — invisible, and cheaper than a 2.7 MB copy
        inside the detection loop.
        """
        if not self._alive:
            return
        with self._slot_lock:
            self._pending = snapshot
        self._wake.set()

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            proc = self._spawn()
        except Exception as exc:
            self._fail(f"окно диагностики не запустилось: {exc}")
            return

        with self._proc_lock:
            self._proc = proc
        self._started = True
        self._log("[debug] Окно диагностики открыто.")

        try:
            while self._alive:
                self._wake.wait(_POLL_SEC)
                self._wake.clear()
                if not self._alive:
                    return
                if proc.poll() is not None:
                    # The user closed the window. Not a fault.
                    self._alive = False
                    self._log("[debug] Окно диагностики закрыто.")
                    return
                with self._slot_lock:
                    snapshot, self._pending = self._pending, None
                if snapshot is None:
                    continue
                try:
                    payload = self._encode(
                        snapshot, max_width=self._max_width, quality=self._quality
                    )
                except Exception as exc:
                    self._fail(f"кадр диагностики не закодирован: {exc}")
                    return
                try:
                    debugwire.write_message(proc.stdin, payload)
                except Exception as exc:
                    self._fail(f"окно диагностики не принимает кадры: {exc}")
                    return
        finally:
            self._shutdown_proc(timeout=_GRACE_SEC)

    def _fail(self, message: str) -> None:
        self._alive = False
        self._error = message
        try:
            self._log(f"[debug] {message}")
        except Exception:
            pass

    def _shutdown_proc(self, timeout: float = 1.0) -> None:
        with self._proc_lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return

        grace = max(0.0, min(_GRACE_SEC, timeout))
        # Closing the pipe is the polite signal: the viewer reads end-of-feed
        # and tears its own window down. Terminating is the fallback.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        if self._wait(proc, grace):
            return
        try:
            proc.terminate()
        except Exception:
            pass
        if self._wait(proc, grace):
            return
        try:
            proc.kill()
        except Exception:
            pass
        self._wait(proc, grace)

    @staticmethod
    def _wait(proc: Any, timeout: float) -> bool:
        """True if the process is gone. Never raises."""
        try:
            if proc.poll() is not None:
                return True
        except Exception:
            return True
        try:
            proc.wait(timeout=timeout)
            return True
        except Exception:
            return False
