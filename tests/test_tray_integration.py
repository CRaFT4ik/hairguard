"""The whole chain from a tray click to collection starting, on real pystray.

The stand-in used in test_tray_collect keeps the menu logic honest, but a
stand-in is only as good as its author's understanding — an earlier version
resolved callable menu text without passing the item, which is not what
pystray does, and the resulting crash reached the running application.

So this exercises the real pystray Menu and MenuItem objects: the text is
resolved through the real property and the action is invoked the way
pystray invokes it. Only the OS-level icon is absent.
"""
from __future__ import annotations

import queue

import pytest

pystray = pytest.importorskip("pystray")

from hairguard.app import HairGuardApp
from hairguard.tray import TrayApp


class FakeCapture:
    def __init__(self) -> None:
        self.enabled = False
        self.saved_count = 0
        self.clips_root = "/tmp/clips"
        self.buffered_frames = 0
        self.window_frames = 20

    def start(self) -> None:
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False


@pytest.fixture
def wiring(tmp_path):
    """A tray and an app halves wired together the way run() wires them."""
    from hairlab.state import CollectionState

    events: queue.Queue = queue.Queue()
    capture = FakeCapture()

    app = HairGuardApp.__new__(HairGuardApp)
    app._capture = capture
    app._tray_events = events
    app._paused = False
    app._overlay = None
    app._collect_state = CollectionState(path=tmp_path / "collect.state")
    app._collect_ticks = 0

    tray = TrayApp(
        on_pause=lambda: events.put("pause"),
        on_resume=lambda: events.put("resume"),
        on_exit=lambda: events.put("exit"),
        stats_getter=lambda: "",
        on_toggle_collect=app._on_toggle_collect,
        collect_status_getter=app._get_collect_status,
    )
    app._tray = tray
    return app, tray, capture, events


def _collection_item(tray: TrayApp):
    """The collection entry, found through pystray's own text resolution."""
    menu = tray._build_menu(pystray)
    for item in menu.items:
        if isinstance(item, pystray.MenuItem) and "Сбор данных" in item.text:
            return item
    raise AssertionError("в меню нет пункта сбора данных")


def test_the_menu_builds_on_real_pystray(wiring) -> None:
    """Building the menu must not raise — this is what crashed the tray thread."""
    _, tray, _, _ = wiring
    assert _collection_item(tray) is not None


def test_real_pystray_resolves_the_entry_text(wiring) -> None:
    _, tray, _, _ = wiring
    assert "начать" in _collection_item(tray).text


def test_clicking_the_real_menu_item_queues_the_event(wiring) -> None:
    _, tray, _, events = wiring
    _collection_item(tray)(None)   # pystray calls item(icon)
    assert events.get_nowait() == "collect_toggle"


def test_click_through_to_collection_starting(wiring) -> None:
    """Click, then let the main loop drain the queue, as the app does each tick."""
    app, tray, capture, events = wiring

    _collection_item(tray)(None)
    while not events.empty():
        app._handle_tray_event(events.get_nowait())

    assert capture.enabled is True


def test_the_entry_text_follows_the_new_state(wiring) -> None:
    app, tray, capture, events = wiring

    _collection_item(tray)(None)
    while not events.empty():
        app._handle_tray_event(events.get_nowait())

    text = _collection_item(tray).text
    assert "ВКЛ" in text
    assert "остановить" in text


def test_clicking_twice_stops_collection(wiring) -> None:
    app, tray, capture, events = wiring

    for _ in range(2):
        _collection_item(tray)(None)
        while not events.empty():
            app._handle_tray_event(events.get_nowait())

    assert capture.enabled is False
