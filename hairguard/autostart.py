"""Autostart management for HairGuard.

Windows: creates/removes a .bat launcher in the user's Startup folder.
Linux:   creates/removes a .desktop file in ~/.config/autostart/.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_APP_NAME = "HairGuard"
_BAT_NAME = "hairguard.bat"
_DESKTOP_NAME = "hairguard.desktop"


def _windows_startup_dir() -> Path:
    """Return the Windows per-user Startup folder path."""
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidate = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        if candidate.exists():
            return candidate

    startup = os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup")
    return Path(startup)


def _linux_autostart_dir() -> Path:
    return Path.home() / ".config" / "autostart"


def _bat_path() -> Path:
    return _windows_startup_dir() / _BAT_NAME


def _desktop_path() -> Path:
    return _linux_autostart_dir() / _DESKTOP_NAME


def _hairguard_command() -> tuple[list[str], Optional[str]]:
    """Return (argv, cwd) for invoking the hairguard CLI from autostart.

    Prefers an installed `hairguard` script when available; otherwise
    falls back to `python -m hairguard` with the project directory as
    cwd, so the launcher works even when the package was not pip-installed
    (e.g. when the user runs the app via `python -m hairguard start`).
    """
    scripts_dir = Path(sys.executable).parent

    if sys.platform == "win32":
        candidate = scripts_dir / "hairguard.exe"
        if candidate.exists():
            return ([str(candidate)], None)
    else:
        candidate = scripts_dir / "hairguard"
        if candidate.exists():
            return ([str(candidate)], None)

    # Fallback: invoke the module via the same Python interpreter.
    import hairguard as _pkg

    package_parent = Path(_pkg.__file__).resolve().parent.parent
    return ([sys.executable, "-m", "hairguard"], str(package_parent))


def _windows_quote(arg: str) -> str:
    """Quote a single argument for a Windows .bat `start` line."""
    if not arg:
        return '""'
    if any(ch in arg for ch in ' \t"&|<>^()'):
        escaped = arg.replace('"', '""')
        return f'"{escaped}"'
    return arg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def autostart_on() -> None:
    """Enable HairGuard autostart for the current user."""
    argv, cwd = _hairguard_command()
    full_argv = argv + ["start", "--background"]

    if sys.platform == "win32":
        bat = _bat_path()
        bat.parent.mkdir(parents=True, exist_ok=True)

        quoted = " ".join(_windows_quote(a) for a in full_argv)
        lines = ["@echo off"]
        if cwd:
            # Working directory is required for `python -m hairguard` when
            # the package is not installed system-wide.
            lines.append(f'cd /d "{cwd}"')
        # `start ""` opens a new process without inheriting the console,
        # so the user is not left with a stray cmd window at boot.
        lines.append(f'start "" {quoted}')
        bat.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
        print(f"Автозапуск включён: {bat}")
    else:
        desktop = _desktop_path()
        desktop.parent.mkdir(parents=True, exist_ok=True)

        exec_str = " ".join(shlex.quote(a) for a in full_argv)
        path_line = f"Path={cwd}\n" if cwd else ""
        desktop.write_text(
            "[Desktop Entry]\n"
            f"Name={_APP_NAME}\n"
            "Comment=Hair-pulling detection assistant\n"
            f"Exec={exec_str}\n"
            f"{path_line}"
            "Icon=\n"
            "Terminal=false\n"
            "Type=Application\n"
            "Categories=Utility;\n"
            "X-GNOME-Autostart-enabled=true\n",
            encoding="utf-8",
        )
        print(f"Автозапуск включён: {desktop}")


def autostart_off() -> None:
    """Disable HairGuard autostart for the current user."""
    if sys.platform == "win32":
        bat = _bat_path()
        if bat.exists():
            bat.unlink()
            print(f"Автозапуск отключён: {bat}")
        else:
            print("Автозапуск уже отключён.")
    else:
        desktop = _desktop_path()
        if desktop.exists():
            desktop.unlink()
            print(f"Автозапуск отключён: {desktop}")
        else:
            print("Автозапуск уже отключён.")


def autostart_status() -> bool:
    """Return True if the autostart entry exists."""
    if sys.platform == "win32":
        return _bat_path().exists()
    return _desktop_path().exists()
