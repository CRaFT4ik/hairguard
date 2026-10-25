# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for HairGuard — one bundle, two executables.

Deliberately thin. What may and may not go in lives in ``bundle.buildconf``,
which is tested without running a build; this file only assembles it.

Two ``EXE`` targets share one ``COLLECT``, so the windowed launcher and the
console tool sit side by side over a single copy of 250 MB of libraries. The
alternative — two bundles — would double the download and let the two drift
apart.

``onedir``, not ``onefile``. A single file would be prettier to hand over,
but it unpacks the whole tree into a temporary directory on every launch:
seconds of delay for an app that starts at login, and 250 MB rewritten to
disk each time. A folder starts immediately and is honest about what it is.

Run it through ``python bundle/build.py``, which checks the sources first,
checks the result afterwards and measures it. Running PyInstaller on this
spec directly works and skips both checks.
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

SPEC_DIR = Path(SPECPATH).resolve()
REPO_ROOT = SPEC_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bundle import buildconf  # noqa: E402

MODELS_DIR = Path(
    os.environ.get("HAIRGUARD_BUILD_MODELS", Path.home() / ".hairguard" / "models")
)
ICON = os.environ.get("HAIRGUARD_BUILD_ICON") or None

platform_key = (
    "win32" if sys.platform == "win32"
    else "darwin" if sys.platform == "darwin"
    else "linux"
)

hidden_imports = buildconf.HIDDEN_IMPORTS + buildconf.PLATFORM_HIDDEN_IMPORTS.get(
    platform_key, []
)

# MediaPipe's .fbs/.txt resources. There is no PyInstaller hook for
# mediapipe, so nothing collects these by default.
datas = collect_data_files("mediapipe") + buildconf.datas(REPO_ROOT, MODELS_DIR)

# The 28 MB shared library mediapipe actually is, loaded by a path computed
# at runtime. Declared as a binary rather than a data file so that its own
# dependencies are followed too.
binaries = buildconf.mediapipe_native_libraries()

analysis = Analysis(
    [str(SPEC_DIR / "entry_gui.py"), str(SPEC_DIR / "entry_cli.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=buildconf.EXCLUDES,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)


def scripts_for(stem):
    """The runtime hooks, then one entry script.

    ``analysis.scripts`` is not just the scripts that were asked for: it
    begins with PyInstaller's own runtime hooks, which set up ``sys.path``,
    the inspect module and so on. Indexing it — ``analysis.scripts[0]`` —
    builds an executable whose entry point is a runtime hook, which then
    exits zero having done nothing. That is what the first attempt at this
    produced: two binaries that ran, printed nothing and reported success.
    """
    hooks = [entry for entry in analysis.scripts if not entry[0].startswith("entry_")]
    chosen = [entry for entry in analysis.scripts if entry[0] == stem]
    if not chosen:
        raise SystemExit(f"entry script {stem!r} missing from the analysis")
    return hooks + chosen


gui_exe = EXE(
    pyz,
    scripts_for("entry_gui"),
    exclude_binaries=True,
    name=buildconf.APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # No console: this is the icon, and a terminal flashing up at every
    # login would defeat the point of a tray application. Everything the app
    # would have printed goes to ~/.hairguard/hairguard.log instead — see
    # hairguard.diagnostics.
    console=False,
    disable_windowed_traceback=False,
    icon=ICON,
)

console_exe = EXE(
    pyz,
    scripts_for("entry_cli"),
    exclude_binaries=True,
    name=buildconf.CONSOLE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    icon=ICON,
)

collect = COLLECT(
    gui_exe,
    console_exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=buildconf.APP_NAME,
)

if sys.platform == "darwin":
    # A .app is the only thing macOS treats as double-clickable, and the
    # keys below are not decoration:
    #
    #   NSCameraUsageDescription — without it the system kills the process
    #   the moment it opens the webcam, with no prompt and no error.
    #
    #   LSUIElement — keeps the app out of the Dock and the ⌘-Tab switcher.
    #   It lives in the menu bar; a Dock icon for something with no window
    #   is clutter the user cannot get rid of.
    #
    # Unverified: this was written on Windows and no part of it has been run
    # on macOS. See the README.
    app = BUNDLE(
        collect,
        name=f"{buildconf.APP_NAME}.app",
        icon=ICON,
        bundle_identifier=buildconf.BUNDLE_IDENTIFIER,
        info_plist={
            "CFBundleName": buildconf.APP_NAME,
            "CFBundleDisplayName": buildconf.APP_NAME,
            "CFBundleShortVersionString": os.environ.get(
                "HAIRGUARD_BUILD_VERSION", "0.1.0"
            ),
            "NSCameraUsageDescription": (
                "HairGuard смотрит в камеру, чтобы заметить накручивание "
                "волос и прервать его. Видео остаётся на этом компьютере."
            ),
            "NSHighResolutionCapable": True,
            "LSUIElement": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
