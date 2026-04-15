"""Autostart management for HairGuard.

Windows: creates/removes a .bat launcher in the user's Startup folder.
Linux:   creates/removes a .desktop file in ~/.config/autostart/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_APP_NAME = "HairGuard"
_BAT_NAME = "hairguard.bat"
_DESKTOP_NAME = "hairguard.desktop"


def _windows_startup_dir() -> Path:
    """Return the Windows per-user Startup folder path."""
    # APPDATA is e.g. C:\Users\<user>\AppData\Roaming
    # Startup folder is under ..\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidate = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        if candidate.exists():
            return candidate

    # Fallback: use expandvars with the shell keyword
    startup = os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup")
    return Path(startup)


def _linux_autostart_dir() -> Path:
    return Path.home() / ".config" / "autostart"


def _bat_path() -> Path:
    return _windows_startup_dir() / _BAT_NAME


def _desktop_path() -> Path:
    return _linux_autostart_dir() / _DESKTOP_NAME


def _hairguard_executable() -> str:
    """Best-effort path to the hairguard executable / invocation."""
    # If running inside a virtualenv the script will be on PATH already.
    # We try to find it explicitly so the .bat / .desktop works even when the
    # venv is not activated by the shell that runs the startup entry.
    scripts_dir = Path(sys.executable).parent
    if sys.platform == "win32":
        candidate = scripts_dir / "hairguard.exe"
        if candidate.exists():
            return str(candidate)
        candidate = scripts_dir / "hairguard"
        if candidate.exists():
            return str(candidate)
    else:
        candidate = scripts_dir / "hairguard"
        if candidate.exists():
            return str(candidate)
    # Fallback: hope it's on PATH
    return "hairguard"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def autostart_on() -> None:
    """Enable HairGuard autostart for the current user."""
    exe = _hairguard_executable()

    if sys.platform == "win32":
        bat = _bat_path()
        bat.parent.mkdir(parents=True, exist_ok=True)
        bat.write_text(
            f'@echo off\nstart "" "{exe}" start --background\n',
            encoding="utf-8",
        )
        print(f"Автозапуск включён: {bat}")
    else:
        desktop = _desktop_path()
        desktop.parent.mkdir(parents=True, exist_ok=True)
        desktop.write_text(
            "[Desktop Entry]\n"
            f"Name={_APP_NAME}\n"
            "Comment=Hair-pulling detection assistant\n"
            f"Exec={exe} start --background\n"
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
