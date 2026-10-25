#!/bin/sh
# Put this folder's HairGuard in the applications menu.
#
# Copied into the built folder by bundle/build.py. It writes a .desktop entry
# pointing at wherever you unpacked the folder, because that path is not
# known until you choose it.
#
# Nothing is installed system-wide and nothing needs root: the entry goes in
# ~/.local/share/applications, which is the per-user location.
#
# Written on Windows and not run on Linux. See the README.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
BIN="$HERE/HairGuard"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ENTRY="$APPS/hairguard.desktop"

if [ ! -x "$BIN" ]; then
    echo "error: $BIN is missing or not executable." >&2
    echo "Run this script from inside the unpacked HairGuard folder." >&2
    exit 1
fi

ICON="$HERE/hairguard.png"
[ -f "$ICON" ] || ICON=camera-web

mkdir -p "$APPS"
cat > "$ENTRY" <<ENTRY_END
[Desktop Entry]
Type=Application
Name=HairGuard
Comment=Interrupts compulsive hair-pulling using the webcam
Exec=$BIN
Icon=$ICON
Terminal=false
Categories=Utility;
StartupNotify=false
ENTRY_END
chmod +x "$ENTRY"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS" >/dev/null 2>&1 || true
fi

echo "HairGuard is now in your applications menu."
echo "  entry:   $ENTRY"
echo "  program: $BIN"
echo
echo "Start it at login:   $HERE/hairguard-cli autostart on"
echo "Check what it sees:  $HERE/hairguard-cli doctor"
echo
echo "On Wayland the fullscreen overlay needs gtk-layer-shell; on X11 it does"
echo "not. If the overlay never appears, run '$HERE/hairguard-cli log' first."
