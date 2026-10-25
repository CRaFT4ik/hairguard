"""Tests for the tray's diagnostic entries.

Two things become unreachable the moment the app stops being a command in a
terminal: seeing what the detector sees, and reading what the app said about
itself. Both have to be in the tray menu, because the tray menu is the only
user interface a packaged tray app has.

``TrayApp._build_menu`` takes the pystray module as an argument, so a
stand-in is enough and no tray is ever created.
"""
from __future__ import annotations

from hairguard.tray import TrayApp


class FakeMenuItem:
    def __init__(self, text, action=None, **kwargs) -> None:
        self.text = text
        self.action = action
        self.enabled = kwargs.get("enabled", True)

    def resolved_text(self) -> str:
        # pystray resolves callable text as ``self._text(self)``.
        return self.text(self) if callable(self.text) else str(self.text)


class FakeMenu:
    SEPARATOR = object()

    def __init__(self, *items) -> None:
        self.items = items


class FakePystray:
    MenuItem = FakeMenuItem
    Menu = FakeMenu


def _tray(**kwargs) -> TrayApp:
    return TrayApp(
        on_pause=lambda: None,
        on_resume=lambda: None,
        on_exit=lambda: None,
        stats_getter=lambda: "",
        **kwargs,
    )


def _items(tray: TrayApp) -> list[FakeMenuItem]:
    menu = tray._build_menu(FakePystray)
    return [item for item in menu.items if isinstance(item, FakeMenuItem)]


def _entry(tray: TrayApp, needle: str) -> FakeMenuItem:
    for item in _items(tray):
        if needle in item.resolved_text():
            return item
    raise AssertionError(
        f"no entry matching {needle!r} in {[i.resolved_text() for i in _items(tray)]}"
    )


# ---------------------------------------------------------------------------
# The debug window
# ---------------------------------------------------------------------------


def test_the_debug_entry_is_absent_without_a_handler() -> None:
    """Nothing to open means nothing in the menu, the way collection works."""
    texts = [item.resolved_text() for item in _items(_tray())]

    assert not any("Диагностик" in text for text in texts)


def test_the_debug_entry_offers_to_open_when_closed() -> None:
    tray = _tray(on_toggle_debug=lambda: None, debug_status_getter=lambda: False)

    text = _entry(tray, "Диагностик").resolved_text()

    assert "открыть" in text


def test_the_debug_entry_offers_to_close_when_open() -> None:
    """Open it, look, close it again — all from the same place."""
    tray = _tray(on_toggle_debug=lambda: None, debug_status_getter=lambda: True)

    text = _entry(tray, "Диагностик").resolved_text()

    assert "закрыть" in text


def test_the_debug_label_is_read_fresh_each_time_the_menu_opens() -> None:
    """The window can close itself — the user clicks its X — so a label
    baked in at build time would lie."""
    state = {"open": False}
    tray = _tray(
        on_toggle_debug=lambda: None,
        debug_status_getter=lambda: state["open"],
    )
    item = _entry(tray, "Диагностик")

    assert "открыть" in item.resolved_text()
    state["open"] = True
    assert "закрыть" in item.resolved_text()


def test_clicking_the_debug_entry_calls_the_handler() -> None:
    calls: list[int] = []
    tray = _tray(
        on_toggle_debug=lambda: calls.append(1),
        debug_status_getter=lambda: False,
    )

    _entry(tray, "Диагностик").action(object(), object())

    assert calls == [1]


def test_a_failing_status_getter_does_not_break_the_menu() -> None:
    """The menu is the app's only interface; it opens even when something
    behind it is broken."""

    def explode() -> bool:
        raise RuntimeError("no idea")

    tray = _tray(on_toggle_debug=lambda: None, debug_status_getter=explode)

    assert "Диагностик" in _entry(tray, "Диагностик").resolved_text()


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


def test_the_log_entry_is_present_by_default() -> None:
    """A packaged app has no console, so the log is the only account of
    itself it can give. It is not optional."""
    texts = [item.resolved_text() for item in _items(_tray())]

    assert any("журнал" in text.lower() for text in texts)


def test_clicking_the_log_entry_opens_the_log() -> None:
    opened: list[int] = []
    tray = _tray(on_open_log=lambda: opened.append(1))

    _entry(tray, "журнал").action(object(), object())

    assert opened == [1]


def test_the_default_log_handler_reaches_diagnostics(monkeypatch) -> None:
    """Wired by default rather than by the app, so it still works when the
    app crashed before it could wire anything."""
    from hairguard import diagnostics

    opened: list[int] = []
    monkeypatch.setattr(diagnostics, "reveal", lambda: opened.append(1) or True)
    tray = _tray()

    _entry(tray, "журнал").action(object(), object())

    assert opened == [1]
