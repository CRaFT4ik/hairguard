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
_MACOS_LABEL = "com.eldar.hairguard"
_PLIST_NAME = f"{_MACOS_LABEL}.plist"


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


def _macos_launchagents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def _bat_path() -> Path:
    return _windows_startup_dir() / _BAT_NAME


def _desktop_path() -> Path:
    return _linux_autostart_dir() / _DESKTOP_NAME


def _plist_path() -> Path:
    return _macos_launchagents_dir() / _PLIST_NAME


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _is_frozen() -> bool:
    """True inside a PyInstaller bundle, where sys.executable is the app."""
    return bool(getattr(sys, "frozen", False))


def _bundle_launcher() -> Optional[Path]:
    """The windowed executable of this bundle, if there is one beside us.

    A bundle ships two binaries: the windowed one for the icon and a console
    one for diagnosis. Autostart wants the windowed one — registering the
    console binary would flash a terminal at every login, which for a tray
    utility that is supposed to disappear is a daily annoyance with no
    upside.
    """
    if not _is_frozen():
        return None
    here = Path(sys.executable).resolve()
    for name in ("HairGuard.exe", "HairGuard"):
        candidate = here.parent / name
        if candidate.is_file():
            return candidate
    return here


def _launch_args() -> list[str]:
    """What to ask the launcher to do at login.

    ``--background`` exists to escape a console by re-launching detached. A
    windowed bundle has no console to escape, so the extra hop would buy
    nothing and add one more place for the launch to fail unseen.
    """
    if _is_frozen():
        return ["start"]
    return ["start", "--background"]


def _hairguard_command() -> tuple[list[str], Optional[str]]:
    """Return (argv, cwd) for invoking HairGuard from autostart.

    Inside a bundle this is the bundle's own launcher: ``sys.executable`` is
    the app, not an interpreter, so the ``python -m hairguard`` fallback
    below would write ``HairGuard.exe -m hairguard start`` into the login
    entry — a command line argparse rejects, failing silently at a login
    months from now and looking like the app had simply stopped working.

    Otherwise: an installed `hairguard` script when there is one, else
    `python -m hairguard` with the project directory as cwd, so the launcher
    works even when the package was never pip-installed.
    """
    launcher = _bundle_launcher()
    if launcher is not None:
        return ([str(launcher)], None)

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
    full_argv = argv + _launch_args()

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
    elif sys.platform == "darwin":
        plist = _plist_path()
        plist.parent.mkdir(parents=True, exist_ok=True)

        program_args = "".join(
            f"        <string>{_xml_escape(a)}</string>\n" for a in full_argv
        )
        wd_block = (
            f"    <key>WorkingDirectory</key>\n"
            f"    <string>{_xml_escape(cwd)}</string>\n"
            if cwd
            else ""
        )
        plist.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0">\n'
            "<dict>\n"
            f"    <key>Label</key>\n    <string>{_MACOS_LABEL}</string>\n"
            f"    <key>ProgramArguments</key>\n    <array>\n{program_args}    </array>\n"
            f"{wd_block}"
            "    <key>RunAtLoad</key>\n    <true/>\n"
            "    <key>KeepAlive</key>\n    <false/>\n"
            "</dict>\n"
            "</plist>\n",
            encoding="utf-8",
        )
        print(f"Автозапуск включён: {plist}")
        print(
            "Чтобы запустить сейчас (без перезагрузки): "
            f"launchctl load -w \"{plist}\""
        )
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
    elif sys.platform == "darwin":
        plist = _plist_path()
        if plist.exists():
            plist.unlink()
            print(f"Автозапуск отключён: {plist}")
            print(
                "Если запускали через launchctl — выгрузите: "
                f"launchctl unload \"{plist}\""
            )
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
    if sys.platform == "darwin":
        return _plist_path().exists()
    return _desktop_path().exists()
