"""Tests for the app's tray-event handling.

This is the path a tray click travels: the tray thread queues an event and
the main thread applies it on the next tick. It was the one link in the
collection feature never covered, so it gets covered here — without a
camera, a tray or a tkinter window, by exercising the handler on an
instance assembled with only the attributes it touches.
"""
from __future__ import annotations

import pytest

from hairguard.app import HairGuardApp
from hairlab.state import CollectionState


class FakeCapture:
    """Stands in for CaptureService, with the surface the app actually uses."""

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


class FakeTray:
    def __init__(self) -> None:
        self.refreshes = 0

    def refresh_menu(self) -> None:
        self.refreshes += 1


class FakeSound:
    def __init__(self) -> None:
        self.active = False
        self.starts = 0

    def start(self) -> None:
        if not self.active:
            self.starts += 1
        self.active = True

    def stop(self) -> None:
        self.active = False


class FakeOverlay:
    def __init__(self) -> None:
        self.visible = False

    def is_visible(self) -> bool:
        return self.visible

    def hide(self) -> None:
        self.visible = False


@pytest.fixture
def app(tmp_path) -> HairGuardApp:
    """An app instance with just enough wired for event handling."""
    instance = HairGuardApp.__new__(HairGuardApp)
    instance._capture = FakeCapture()
    instance._tray = FakeTray()
    instance._overlay = FakeOverlay()
    instance._paused = False
    instance._collect_state = CollectionState(path=tmp_path / "collect.state")
    instance._collect_ticks = 0
    instance._collect_hint_shown = False
    instance._sound = FakeSound()
    instance._alert_gone_at = None
    return instance


# ---------------------------------------------------------------------------
# The alert sound follows the hand
# ---------------------------------------------------------------------------

def test_sound_starts_while_the_hand_is_detected(app: HairGuardApp) -> None:
    app._update_alert_sound("ALERTING", now=100.0)
    assert app._sound.active is True


def test_sound_stops_once_the_hand_is_gone(app: HairGuardApp) -> None:
    """Stopping the noise is the reward for moving the hand away."""
    app._update_alert_sound("ALERTING", now=100.0)
    app._update_alert_sound("COOLDOWN", now=100.1)
    app._update_alert_sound("COOLDOWN", now=101.0)
    assert app._sound.active is False


def test_a_brief_tracking_dropout_does_not_cut_the_sound(app: HairGuardApp) -> None:
    """Hand tracking flickers; the sound must not stutter with it."""
    app._update_alert_sound("ALERTING", now=100.0)
    app._update_alert_sound("COOLDOWN", now=100.1)
    app._update_alert_sound("ALERTING", now=100.2)
    assert app._sound.active is True


def test_the_sound_is_not_restarted_while_it_is_already_going(app: HairGuardApp) -> None:
    for tick in range(5):
        app._update_alert_sound("ALERTING", now=100.0 + tick * 0.1)
    assert app._sound.starts == 1


def test_it_sounds_again_for_a_new_episode(app: HairGuardApp) -> None:
    app._update_alert_sound("ALERTING", now=100.0)
    app._update_alert_sound("IDLE", now=101.0)
    app._update_alert_sound("IDLE", now=102.0)
    app._update_alert_sound("ALERTING", now=110.0)
    assert app._sound.starts == 2


def test_no_sound_when_nothing_is_detected(app: HairGuardApp) -> None:
    app._update_alert_sound("IDLE", now=100.0)
    assert app._sound.active is False


# ---------------------------------------------------------------------------
# Startup: recording state must be obvious and must survive a restart
# ---------------------------------------------------------------------------

def test_a_previously_enabled_switch_resumes_on_startup(app: HairGuardApp, capsys) -> None:
    """Collecting a dataset takes weeks; a switch that forgets never records."""
    app._collect_state.write(True)
    app._announce_collection_state()

    assert app._capture.enabled is True
    assert "ЗАПИСЬ КЛИПОВ ВКЛЮЧЕНА" in capsys.readouterr().out


def test_startup_says_so_when_recording_is_off(app: HairGuardApp, capsys) -> None:
    app._collect_state.write(False)
    app._announce_collection_state()

    assert app._capture.enabled is False
    assert "выключена" in capsys.readouterr().out


def test_detection_without_recording_explains_the_missing_clip(
    app: HairGuardApp, capsys
) -> None:
    """The moment a clip is expected is the moment its absence must be explained."""
    app._hint_collection_is_off()
    assert "запись выключена" in capsys.readouterr().out


def test_the_hint_is_shown_only_once(app: HairGuardApp, capsys) -> None:
    app._hint_collection_is_off()
    capsys.readouterr()
    app._hint_collection_is_off()
    assert capsys.readouterr().out == ""


def test_no_hint_while_recording(app: HairGuardApp, capsys) -> None:
    app._capture.start()
    app._hint_collection_is_off()
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Collection toggle
# ---------------------------------------------------------------------------

def test_collect_toggle_starts_collection(app: HairGuardApp) -> None:
    app._handle_tray_event("collect_toggle")
    assert app._capture.enabled is True


def test_collect_toggle_twice_stops_collection(app: HairGuardApp) -> None:
    app._handle_tray_event("collect_toggle")
    app._handle_tray_event("collect_toggle")
    assert app._capture.enabled is False


def test_toggling_refreshes_the_tray_entry(app: HairGuardApp) -> None:
    """Otherwise the menu keeps offering to start what is already running."""
    app._handle_tray_event("collect_toggle")
    assert app._tray.refreshes == 1


def test_toggle_is_harmless_without_a_collector(app: HairGuardApp) -> None:
    app._capture = None
    assert app._handle_tray_event("collect_toggle") is True


def test_collect_status_reports_the_collector_state(app: HairGuardApp) -> None:
    app._handle_tray_event("collect_toggle")
    app._capture.saved_count = 7
    assert app._get_collect_status() == (True, 7)


def test_collect_status_without_a_collector(app: HairGuardApp) -> None:
    app._capture = None
    assert app._get_collect_status() == (False, 0)


# ---------------------------------------------------------------------------
# The shared switch: the command line must reach a running app
# ---------------------------------------------------------------------------

def test_toggling_records_the_new_state_on_disk(app: HairGuardApp) -> None:
    """Otherwise nobody can tell afterwards whether collection ever started."""
    app._handle_tray_event("collect_toggle")
    assert app._collect_state.read() is True


def test_toggling_off_records_that_too(app: HairGuardApp) -> None:
    app._handle_tray_event("collect_toggle")
    app._handle_tray_event("collect_toggle")
    assert app._collect_state.read() is False


def test_sync_starts_collection_when_the_switch_says_on(app: HairGuardApp) -> None:
    app._collect_state.write(True)
    app._sync_collection_state()
    assert app._capture.enabled is True


def test_sync_stops_collection_when_the_switch_says_off(app: HairGuardApp) -> None:
    app._collect_state.write(True)
    app._sync_collection_state()
    app._collect_state.write(False)
    app._sync_collection_state()
    assert app._capture.enabled is False


def test_sync_is_quiet_when_nothing_changed(app: HairGuardApp) -> None:
    app._collect_state.write(True)
    app._sync_collection_state()
    app._sync_collection_state()
    assert app._capture.enabled is True
    assert app._tray.refreshes == 1


def test_sync_without_a_collector_is_harmless(app: HairGuardApp) -> None:
    app._capture = None
    app._collect_state.write(True)
    app._sync_collection_state()


# ---------------------------------------------------------------------------
# The other events must keep working
# ---------------------------------------------------------------------------

def test_pause_event_pauses_and_hides_the_overlay(app: HairGuardApp) -> None:
    app._overlay.visible = True
    app._handle_tray_event("pause")
    assert app._paused is True
    assert app._overlay.visible is False


def test_resume_event_resumes(app: HairGuardApp) -> None:
    app._paused = True
    app._handle_tray_event("resume")
    assert app._paused is False


def test_exit_event_asks_for_shutdown(app: HairGuardApp) -> None:
    assert app._handle_tray_event("exit") is False


def test_other_events_keep_the_app_running(app: HairGuardApp) -> None:
    assert app._handle_tray_event("pause") is True
    assert app._handle_tray_event("something-else") is True
