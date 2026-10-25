"""System tray icon for HairGuard.

Runs pystray in a daemon thread so that tkinter can own the main thread.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Lazy imports — pystray and Pillow are optional at import time so that the
# module can still be imported in environments where they're not installed
# (e.g. CI or headless testing).  The actual tray functionality will fail
# with a clear ImportError only when TrayApp.start() is called.
# ---------------------------------------------------------------------------


def _make_icon_image(size: int = 64) -> "PIL.Image.Image":  # type: ignore[name-defined]
    """Generate a simple colored circle as the tray icon."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Green circle with a white border
    draw.ellipse([2, 2, size - 2, size - 2], fill=(34, 139, 34, 255), outline=(255, 255, 255, 200), width=2)
    return img


# ---------------------------------------------------------------------------
# TrayApp
# ---------------------------------------------------------------------------


class TrayApp:
    """Manages the system tray icon and its menu.

    Parameters
    ----------
    on_pause:
        Called (from tray thread) when the user clicks Pause.
    on_resume:
        Called (from tray thread) when the user clicks Resume or the
        auto-resume timer fires.
    on_exit:
        Called (from tray thread) when the user clicks Exit.
    stats_getter:
        Callable that returns a short today-stats string shown in the
        tooltip and the stats notification.
    config:
        Optional Config instance.  Used to read ``pause.auto_resume_minutes``.
    on_toggle_collect:
        Called (from tray thread) when the user toggles dataset collection.
        When omitted the collection entry is left out of the menu.
    collect_status_getter:
        Returns ``(enabled, clips_saved)`` for the collection menu entry.
    detector_status_getter:
        Returns a one-line description of the detector currently deciding.
        Shown as a disabled menu entry, so which of the two detectors is
        running is something the user can read rather than infer. Omitted
        from the menu when not supplied.
    """

    def __init__(
        self,
        on_pause: Callable[[], None],
        on_resume: Callable[[], None],
        on_exit: Callable[[], None],
        stats_getter: Callable[[], str],
        config: Optional[object] = None,  # hairguard.config.Config
        on_toggle_collect: Optional[Callable[[], None]] = None,
        collect_status_getter: Optional[Callable[[], tuple[bool, int]]] = None,
        detector_status_getter: Optional[Callable[[], str]] = None,
    ) -> None:
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_exit = on_exit
        self._stats_getter = stats_getter
        self._config = config
        self._on_toggle_collect = on_toggle_collect
        self._collect_status_getter = collect_status_getter
        self._detector_status_getter = detector_status_getter

        self._paused = False
        self._pause_timer: Optional[threading.Timer] = None
        self._pause_end: Optional[float] = None  # monotonic time

        self._icon: Optional[object] = None  # pystray.Icon
        self._thread: Optional[threading.Thread] = None

        # Tooltip text (updated periodically)
        self._tooltip = "HairGuard"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the tray icon in a daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="tray")
        self._thread.start()

    def stop(self) -> None:
        """Stop the pystray icon (safe to call from any thread)."""
        icon = self._icon
        self._icon = None  # signal tooltip updater thread to stop
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def update_tooltip(self, text: str) -> None:
        """Update the tray tooltip text."""
        self._tooltip = text
        if self._icon is not None:
            try:
                self._icon.title = text
            except Exception:
                pass

    def set_paused(self, is_paused: bool) -> None:
        """Reflect external pause/resume state in the menu."""
        self._paused = is_paused
        self._rebuild_menu()

    def refresh_menu(self) -> None:
        """Re-read the menu labels. Called after state changes elsewhere."""
        self._rebuild_menu()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_resume_minutes(self) -> int:
        if self._config is not None:
            try:
                return int(self._config.get("pause", "auto_resume_minutes"))  # type: ignore[arg-type]
            except Exception:
                pass
        return 15

    def _minutes_remaining(self) -> Optional[int]:
        if self._pause_end is None:
            return None
        remaining = self._pause_end - time.monotonic()
        if remaining <= 0:
            return 0
        import math
        return max(1, math.ceil(remaining / 60))

    def _pause_label(self) -> str:
        if not self._paused:
            return "Пауза"
        mins = self._minutes_remaining()
        if mins is not None and mins > 0:
            return f"На паузе ({mins} мин осталось)"
        return "На паузе"

    def _collect_label(self, item: object = None) -> str:  # noqa: ARG002
        """Live text for the collection entry — pystray re-reads it on open.

        pystray resolves callable menu text as ``text(item)``, so the menu
        item is accepted and ignored.
        """
        if self._collect_status_getter is None:
            return "Сбор данных: выкл"
        try:
            enabled, saved = self._collect_status_getter()
        except Exception:
            return "Сбор данных: выкл"
        if enabled:
            return f"⏺ Сбор данных: ВКЛ ({saved} клипов) — остановить"
        return f"Сбор данных: выкл ({saved} за сеанс) — начать"

    def _detector_label(self, item: object = None) -> str:  # noqa: ARG002
        """Live text for the detector entry — pystray re-reads it on open."""
        if self._detector_status_getter is None:
            return "Детектор: неизвестно"
        try:
            return self._detector_status_getter()
        except Exception:
            return "Детектор: неизвестно"

    def _rebuild_menu(self) -> None:
        """Rebuild and reapply the pystray menu."""
        if self._icon is None:
            return
        try:
            import pystray

            self._icon.menu = self._build_menu(pystray)
            self._icon.update_menu()
        except Exception:
            pass

    def _build_menu(self, pystray_mod: object) -> object:
        """Build a pystray.Menu from current state."""
        pystray = pystray_mod  # type: ignore[assignment]

        if self._paused:
            toggle_item = pystray.MenuItem(
                self._pause_label(),
                self._handle_resume,
            )
        else:
            toggle_item = pystray.MenuItem(
                "Пауза",
                self._handle_pause,
            )

        items = [
            toggle_item,
            pystray.MenuItem("Статистика", self._handle_stats),
        ]

        if self._detector_status_getter is not None:
            # Not clickable: there is nothing to do here, only something to
            # know. Switching detectors is a config change and a restart,
            # deliberately — it is not a thing to fumble from a menu while
            # an alert is up.
            items.append(
                pystray.MenuItem(self._detector_label, None, enabled=False)
            )

        if self._on_toggle_collect is not None:
            items += [
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(self._collect_label, self._handle_toggle_collect),
            ]

        items += [
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", self._handle_exit),
        ]
        return pystray.Menu(*items)

    # ------------------------------------------------------------------
    # Menu action handlers (called from tray thread)
    # ------------------------------------------------------------------

    def _handle_pause(self, icon: object, item: object) -> None:  # noqa: ARG002
        self._paused = True
        minutes = self._auto_resume_minutes()
        self._pause_end = time.monotonic() + minutes * 60

        # Cancel any previous timer
        if self._pause_timer is not None:
            self._pause_timer.cancel()

        self._pause_timer = threading.Timer(minutes * 60, self._auto_resume)
        self._pause_timer.daemon = True
        self._pause_timer.start()

        self._rebuild_menu()
        self._on_pause()

    def _handle_resume(self, icon: object, item: object) -> None:  # noqa: ARG002
        self._cancel_pause_timer()
        self._paused = False
        self._pause_end = None
        self._rebuild_menu()
        self._on_resume()

    def _auto_resume(self) -> None:
        """Called by the threading.Timer when auto-resume fires."""
        self._paused = False
        self._pause_end = None
        self._pause_timer = None
        self._rebuild_menu()
        self._on_resume()

    def _handle_toggle_collect(self, icon: object, item: object) -> None:  # noqa: ARG002
        if self._on_toggle_collect is not None:
            self._on_toggle_collect()
        self._rebuild_menu()

    def _handle_stats(self, icon: object, item: object) -> None:  # noqa: ARG002
        stats_text = self._stats_getter()
        try:
            from plyer import notification

            notification.notify(
                title="HairGuard — Статистика",
                message=stats_text,
                app_name="HairGuard",
                timeout=10,
            )
        except Exception as exc:
            print(f"[tray] Не удалось показать уведомление: {exc}")
            print(f"[tray] {stats_text}")

    def _handle_exit(self, icon: object, item: object) -> None:  # noqa: ARG002
        self._cancel_pause_timer()
        self.stop()
        self._on_exit()

    def _cancel_pause_timer(self) -> None:
        if self._pause_timer is not None:
            self._pause_timer.cancel()
            self._pause_timer = None

    # ------------------------------------------------------------------
    # Tooltip updater thread
    # ------------------------------------------------------------------

    def _tooltip_updater(self) -> None:
        """Periodically refresh the tray tooltip with live stats."""
        while self._icon is not None:
            try:
                stats = self._stats_getter()
                text = f"HairGuard — Сегодня: {stats}"
                self.update_tooltip(text)
            except Exception:
                pass
            time.sleep(30)

    # ------------------------------------------------------------------
    # Main tray thread entry point
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            import pystray
        except ImportError:
            print("[tray] pystray не установлен — трей недоступен.")
            return

        img = _make_icon_image()
        menu = self._build_menu(pystray)

        self._icon = pystray.Icon(
            name="hairguard",
            icon=img,
            title=self._tooltip,
            menu=menu,
        )

        # Start background tooltip updater
        tooltip_thread = threading.Thread(
            target=self._tooltip_updater, daemon=True, name="tray-tooltip"
        )
        tooltip_thread.start()

        self._icon.run()
