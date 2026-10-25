"""overlay.py — Alert overlay for HairGuard.

Factory function ``create_overlay(config)`` returns the appropriate backend:
  - Linux + Wayland ($XDG_SESSION_TYPE == "wayland"): GtkOverlay if
    gi + gtk_layer_shell are available, else TkOverlay with a warning.
  - Everything else: TkOverlay.

The overlay owner (Stage 6 app) drives the tkinter event loop by calling
``root.mainloop()`` (or scheduling via ``root.after()``).  This module never
starts a mainloop itself.
"""

from __future__ import annotations

import os
import platform
import sys
import tkinter as tk
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hairguard.config import Config

# ---------------------------------------------------------------------------
# TkOverlay
# ---------------------------------------------------------------------------


class TkOverlay:
    """Fullscreen semi-transparent frameless overlay powered by tkinter.

    The caller owns the tkinter mainloop.  ``get_root()`` exposes the Tk root
    so the app can schedule callbacks via ``root.after()``.
    """

    _BG_COLOR = "#1a1a1a"  # near-black; alpha applied separately

    def __init__(self, config: "Config") -> None:
        self._opacity: float = float(config.get("alert", "opacity") or 0.7)
        self._text_color: str = str(config.get("alert", "text_color") or "#ffffff")
        self._visible: bool = False

        # Enable DPI awareness on Windows before creating any tk windows
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
            except Exception:
                try:
                    import ctypes
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass

        self._root = tk.Tk()
        self._root.withdraw()  # hide root; we use a Toplevel for the overlay

        self._window = tk.Toplevel(self._root)
        self._window.withdraw()

        self._setup_window()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        win = self._window
        root = self._root

        # --- geometry & decoration ---
        win.overrideredirect(True)
        win.attributes("-topmost", True)

        # Full-screen size
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        win.geometry(f"{sw}x{sh}+0+0")

        # --- background ---
        win.configure(bg=self._BG_COLOR)

        # --- transparency ---
        # On Windows, apply alpha BEFORE any WS_EX flags to avoid
        # the WS_EX_LAYERED + alpha rendering conflict.
        # On Linux, -alpha works reliably.
        if not sys.platform == "win32":
            try:
                win.attributes("-alpha", self._opacity)
            except tk.TclError:
                pass

        # --- phrase label ---
        self._label = tk.Label(
            win,
            text="",
            font=("Arial", 72, "bold"),
            fg=self._text_color,
            bg=self._BG_COLOR,
            wraplength=sw - 200,
            justify="center",
        )
        self._label.place(relx=0.5, rely=0.5, anchor="center")

        # --- platform-specific setup ---
        win.update_idletasks()  # ensure window ID is assigned
        if sys.platform == "win32":
            self._apply_win32_overlay()
        elif sys.platform.startswith("linux"):
            self._apply_linux_notification_type()

    def _apply_win32_overlay(self) -> None:
        """Windows: set WS_EX_NOACTIVATE + topmost, then apply alpha via SetLayeredWindowAttributes."""
        try:
            import ctypes

            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOPMOST = 0x00000008
            WS_EX_TRANSPARENT = 0x00000020
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            LWA_ALPHA = 0x02

            hwnd = self._window.winfo_id()
            user32 = ctypes.windll.user32

            # Add NOACTIVATE + LAYERED + TOPMOST to extended style
            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ex_style |= WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_LAYERED
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)

            # Set alpha via Win32 API (not tkinter -alpha which conflicts)
            alpha_byte = int(self._opacity * 255)
            user32.SetLayeredWindowAttributes(hwnd, 0, alpha_byte, LWA_ALPHA)

            # Ensure topmost without activating
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[overlay] Win32 overlay setup failed: {exc}")

    def _apply_linux_notification_type(self) -> None:
        """Try to mark the window as _NET_WM_WINDOW_TYPE_NOTIFICATION via xprop."""
        try:
            import subprocess
            wid = hex(self._window.winfo_id())
            subprocess.Popen(
                [
                    "xprop",
                    "-id", wid,
                    "-f", "_NET_WM_WINDOW_TYPE", "32a",
                    "-set", "_NET_WM_WINDOW_TYPE",
                    "_NET_WM_WINDOW_TYPE_NOTIFICATION",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass  # xprop not available — silently skip
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[overlay] xprop call failed: {exc}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def show(self, phrase: str) -> None:
        """Display the overlay with *phrase* as the centred text."""
        self._label.config(text=phrase)
        self._window.deiconify()
        self._window.lift()
        self._visible = True
        self._raise_topmost()

    def ensure_on_top(self) -> None:
        """Re-assert topmost while the alert is up.

        Topmost is not a promise the window system keeps: anything that
        raises itself afterwards — a notification, an installer, another
        fullscreen app — leaves the alert behind it, which to the user is
        indistinguishable from the alert not working at all. Claiming it
        once at show() was not enough.
        """
        if self._visible:
            self._raise_topmost()

    def _raise_topmost(self) -> None:
        if sys.platform != "win32":
            return
        # Re-apply topmost without stealing focus
        try:
            import ctypes
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            hwnd = self._window.winfo_id()
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:  # noqa: BLE001
            pass

    def hide(self) -> None:
        """Hide the overlay."""
        self._window.withdraw()
        self._visible = False

    def is_visible(self) -> bool:
        """Return True if the overlay window is currently shown."""
        return self._visible

    def get_root(self) -> tk.Tk:
        """Return the tkinter root so the app can schedule ``root.after()`` callbacks."""
        return self._root


# ---------------------------------------------------------------------------
# GtkOverlay  (optional — Wayland only)
# ---------------------------------------------------------------------------


def _build_gtk_overlay(config: "Config") -> "GtkOverlay | None":
    """Attempt to import gi + gtk_layer_shell and build a GtkOverlay.

    Returns None if the required libraries are not available.
    """
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk, GLib  # noqa: F401
        import gi.repository.GtkLayerShell as GtkLayerShell  # noqa: F401
    except (ImportError, ValueError):
        return None

    class GtkOverlay:  # type: ignore[no-redef]
        """Wayland overlay backed by gtk-layer-shell."""

        def __init__(self, cfg: "Config") -> None:
            from gi.repository import Gtk, Gdk
            import gi.repository.GtkLayerShell as gls

            self._opacity: float = float(cfg.get("alert", "opacity") or 0.7)
            self._text_color: str = str(cfg.get("alert", "text_color") or "#ffffff")
            self._visible = False

            self._win = Gtk.Window(type=Gtk.WindowType.TOPLEVEL)

            # gtk-layer-shell setup
            gls.init_for_window(self._win)
            gls.set_layer(self._win, gls.Layer.OVERLAY)
            gls.set_keyboard_mode(self._win, gls.KeyboardMode.NONE)
            for edge in (gls.Edge.LEFT, gls.Edge.RIGHT, gls.Edge.TOP, gls.Edge.BOTTOM):
                gls.set_anchor(self._win, edge, True)

            self._win.set_opacity(self._opacity)
            self._win.set_app_paintable(True)

            # Label
            self._label = Gtk.Label(label="")
            css = Gtk.CssProvider()
            css.load_from_data(
                f"label {{ color: {self._text_color}; font-size: 72px; font-weight: bold; }}".encode()
            )
            self._label.get_style_context().add_provider(
                css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
            self._win.add(self._label)
            self._win.connect("delete-event", lambda w, e: self.hide() or True)

        def show(self, phrase: str) -> None:
            self._label.set_text(phrase)
            self._win.show_all()
            self._visible = True

        def hide(self) -> None:
            self._win.hide()
            self._visible = False

        def is_visible(self) -> bool:
            return self._visible

        def ensure_on_top(self) -> None:
            """Layer-shell keeps this window on top for us."""

    return GtkOverlay(config)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_overlay(config: "Config") -> TkOverlay:
    """Create and return the appropriate overlay backend for the current platform.

    On Linux with Wayland, attempts GtkOverlay first.  If the required
    libraries (gi + gtk_layer_shell) are not installed, falls back to
    TkOverlay with a warning printed to stderr.

    On all other platforms (Windows, Linux/X11, macOS), returns TkOverlay.

    The returned object always provides:
        ``show(phrase: str) -> None``
        ``hide() -> None``
        ``is_visible() -> bool``

    TkOverlay additionally provides:
        ``get_root() -> tkinter.Tk``
    """
    is_wayland = (
        sys.platform.startswith("linux")
        and os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )

    if is_wayland:
        gtk_overlay = _build_gtk_overlay(config)
        if gtk_overlay is not None:
            return gtk_overlay  # type: ignore[return-value]
        warnings.warn(
            "[overlay] gtk-layer-shell not available on Wayland. "
            "Falling back to TkOverlay (overlay may not stay on top). "
            "Install with: pip install hairguard[wayland]",
            stacklevel=2,
        )

    return TkOverlay(config)
