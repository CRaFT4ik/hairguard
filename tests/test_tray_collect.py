"""Tests for the tray's dataset-collection entry.

Starting and stopping collection has to be reachable from the application
itself, not from the labelling page, so this covers the menu entry and the
text it shows. ``TrayApp._build_menu`` takes the pystray module as an
argument, so a stand-in is enough and no tray is ever created.
"""
from __future__ import annotations

from hairguard.tray import TrayApp


class FakeMenuItem:
    def __init__(self, text, action=None, **kwargs) -> None:  # noqa: ARG002
        self.text = text
        self.action = action

    def resolved_text(self) -> str:
        # pystray resolves callable text as ``self._text(self)`` — it passes
        # the menu item in. Matching that here is what makes this stand-in
        # worth having.
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


def _entry_texts(tray: TrayApp) -> list[str]:
    menu = tray._build_menu(FakePystray)
    return [
        item.resolved_text()
        for item in menu.items
        if isinstance(item, FakeMenuItem)
    ]


def test_collection_entry_is_absent_without_a_handler() -> None:
    texts = _entry_texts(_tray())
    assert not any("Сбор данных" in t for t in texts)


def test_collection_entry_is_present_when_wired() -> None:
    tray = _tray(on_toggle_collect=lambda: None, collect_status_getter=lambda: (False, 0))
    assert any("Сбор данных" in t for t in _entry_texts(tray))


def test_entry_offers_to_start_when_idle() -> None:
    tray = _tray(on_toggle_collect=lambda: None, collect_status_getter=lambda: (False, 0))
    text = next(t for t in _entry_texts(tray) if "Сбор данных" in t)
    assert "выкл" in text
    assert "начать" in text


def test_entry_offers_to_stop_and_shows_the_count_while_running() -> None:
    tray = _tray(on_toggle_collect=lambda: None, collect_status_getter=lambda: (True, 42))
    text = next(t for t in _entry_texts(tray) if "Сбор данных" in t)
    assert "ВКЛ" in text
    assert "42" in text
    assert "остановить" in text


def test_clicking_the_entry_calls_the_handler() -> None:
    calls: list[str] = []
    tray = _tray(
        on_toggle_collect=lambda: calls.append("toggled"),
        collect_status_getter=lambda: (False, 0),
    )
    menu = tray._build_menu(FakePystray)
    entry = next(
        item for item in menu.items
        if isinstance(item, FakeMenuItem) and "Сбор данных" in item.resolved_text()
    )
    entry.action(None, None)
    assert calls == ["toggled"]


def test_a_broken_status_getter_does_not_break_the_menu() -> None:
    """The tray must survive a collector that errors rather than vanish."""
    def boom() -> tuple[bool, int]:
        raise RuntimeError("nope")

    tray = _tray(on_toggle_collect=lambda: None, collect_status_getter=boom)
    assert any("Сбор данных" in t for t in _entry_texts(tray))


# ---------------------------------------------------------------------------
# The detector entry: which of the two is running must be readable
# ---------------------------------------------------------------------------

def test_the_detector_entry_is_absent_without_a_getter() -> None:
    assert not any("Детектор" in t for t in _entry_texts(_tray()))


def test_the_detector_entry_shows_what_is_running() -> None:
    """A silent fallback to the rule would look like the model being bad."""
    tray = _tray(detector_status_getter=lambda: "Детектор: модель twistnet")
    assert any("модель twistnet" in t for t in _entry_texts(tray))


def test_the_detector_entry_is_read_fresh_each_time_the_menu_opens() -> None:
    """The model can demote itself mid-session; a cached label would lie."""
    state = ["Детектор: модель twistnet"]
    tray = _tray(detector_status_getter=lambda: state[0])
    assert any("модель" in t for t in _entry_texts(tray))
    state[0] = "Детектор: правило"
    assert any("правило" in t for t in _entry_texts(tray))


def test_the_detector_entry_is_not_clickable() -> None:
    """Switching detectors is a config change and a restart, on purpose."""
    tray = _tray(detector_status_getter=lambda: "Детектор: правило")
    menu = tray._build_menu(FakePystray)
    entry = next(
        item for item in menu.items
        if isinstance(item, FakeMenuItem) and "Детектор" in item.resolved_text()
    )
    assert entry.action is None


def test_a_broken_detector_getter_does_not_break_the_menu() -> None:
    def boom() -> str:
        raise RuntimeError("nope")

    tray = _tray(detector_status_getter=boom)
    assert any("Детектор" in t for t in _entry_texts(tray))
