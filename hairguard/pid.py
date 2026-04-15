"""PID file management and instance guard for HairGuard."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

HAIRGUARD_DIR = Path.home() / ".hairguard"
DEFAULT_PID_PATH = HAIRGUARD_DIR / "hairguard.pid"


def _process_is_alive(pid: int) -> bool:
    """Return True if a process with the given PID is alive."""
    try:
        if sys.platform == "win32":
            import ctypes

            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle == 0:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        else:
            os.kill(pid, 0)
            return True
    except (OSError, PermissionError):
        # OSError/PermissionError with errno EPERM means process exists but
        # we lack permission to signal it — still alive.
        import errno as _errno

        try:
            os.kill(pid, 0)
        except OSError as e:
            if e.errno == _errno.EPERM:
                return True
            return False
        except Exception:
            return False
        return False


class PidFile:
    """Manages a PID file for single-instance enforcement."""

    def __init__(self, pid_path: Optional[Path | str] = None) -> None:
        if pid_path is None:
            self._path = DEFAULT_PID_PATH
        else:
            self._path = Path(pid_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> bool:
        """Try to acquire the PID file for the current process.

        Returns True if successfully acquired (no other instance running).
        Returns False if another live instance already holds the PID file.
        Removes stale PID files (process no longer exists) automatically.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        existing_pid = self.read()
        if existing_pid is not None:
            if _process_is_alive(existing_pid):
                return False
            # Stale PID — remove it and continue.
            self._path.unlink(missing_ok=True)

        self._path.write_text(str(os.getpid()), encoding="utf-8")
        return True

    def release(self) -> None:
        """Remove the PID file."""
        self._path.unlink(missing_ok=True)

    def read(self) -> Optional[int]:
        """Read the PID from the file. Returns None if the file does not exist
        or contains invalid data."""
        if not self._path.exists():
            return None
        try:
            return int(self._path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return None

    def is_running(self) -> bool:
        """Return True if a PID file exists and the recorded process is alive."""
        pid = self.read()
        if pid is None:
            return False
        return _process_is_alive(pid)
