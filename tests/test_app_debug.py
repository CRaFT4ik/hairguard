"""Tests for the app's half of the diagnostic window: the tray click and the
per-tick publish.

The wiring is the risky part. The detection tick owns the tkinter event loop,
the fullscreen overlay and the alert chime, and this is the third feature to
hang off it; the previous two each broke it once. So what is asserted here is
mostly about what the tick does *not* do — build snapshots nobody asked for,
let a diagnostic exception escape, or get to the debug window before it gets
to the alert.

Assembled with ``__new__`` and only the attributes the methods touch, the way
``tests/test_app_events.py`` does it: no camera, no tray, no tkinter.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from hairguard import debugwire
from hairguard.analyzer import AnalysisResult
from hairguard.app import HairGuardApp
from hairguard.detector import PoseResult
from hairguard.face_detector import FaceResult


class FakeFeed:
    """Stands in for DebugFeed, with the surface the app uses."""

    def __init__(self) -> None:
        self.active = False
        self.opens = 0
        self.closes = 0
        self.published: list[debugwire.Snapshot] = []
        self.raise_on_publish: BaseException | None = None

    def open(self) -> None:
        self.opens += 1
        self.active = True

    def close(self, timeout: float = 1.0) -> None:
        self.closes += 1
        self.active = False

    def publish(self, snapshot) -> None:
        if self.raise_on_publish is not None:
            raise self.raise_on_publish
        self.published.append(snapshot)


class FakeTray:
    def __init__(self) -> None:
        self.refreshes = 0

    def refresh_menu(self) -> None:
        self.refreshes += 1


class FakeModel:
    threshold = 0.9660314917564392
    seq_len = 20


class FakeAnalyzer:
    detector_name = "model"


def make_result(**overrides) -> AnalysisResult:
    fields = dict(
        should_alert=True,
        hand_near_head=True,
        state="ALERTING",
        episode_started=False,
        episode_ended=False,
        episode_duration=0.0,
        model_score=0.981,
        detector="model",
        fingertips_in_zone=True,
        consecutive_frames=4,
    )
    fields.update(overrides)
    return AnalysisResult(**fields)


class FakeConfig:
    def __init__(self, values: dict | None = None) -> None:
        self._values = values or {
            ("detection", "combined_threshold"): 0.55,
            ("detection", "hair_zone_offset"): 0.4,
        }

    def get(self, *keys):
        return self._values.get(keys)


@pytest.fixture
def app() -> HairGuardApp:
    instance = HairGuardApp.__new__(HairGuardApp)
    instance._debug_feed = FakeFeed()
    instance._tray = FakeTray()
    instance._config = FakeConfig()
    instance._twist_model = FakeModel()
    instance._analyzer = FakeAnalyzer()
    instance._detector_status = "модель twistnet (порог 0.966)"
    instance._tick_count = 7
    instance._tick_fps = 6.75
    instance._paused = False
    instance._capture = None
    return instance


def pipeline_frame():
    return (
        np.zeros((48, 64, 3), dtype=np.uint8),
        PoseResult(landmarks={}, raw_landmarks=None, detected=False),
        [],
        FaceResult(landmarks={}, detected=False),
    )


# ---------------------------------------------------------------------------
# Opening and closing from the tray
# ---------------------------------------------------------------------------


def test_the_debug_tray_event_opens_the_window(app) -> None:
    assert app._handle_tray_event("debug") is True

    assert app._debug_feed.opens == 1
    assert app._debug_view_open() is True


def test_the_debug_tray_event_closes_an_open_window(app) -> None:
    """Same entry both ways — open it, look, close it again."""
    app._handle_tray_event("debug")

    app._handle_tray_event("debug")

    assert app._debug_feed.closes == 1
    assert app._debug_view_open() is False


def test_toggling_the_window_refreshes_the_menu(app) -> None:
    """So the entry says "close" immediately after it said "open"."""
    app._handle_tray_event("debug")

    assert app._tray.refreshes >= 1


def test_the_debug_event_does_not_stop_the_app(app) -> None:
    """``_handle_tray_event`` returning False means shut down. Only "exit"
    may do that."""
    assert app._handle_tray_event("debug") is True
    assert app._handle_tray_event("debug") is True


def test_the_tray_callback_queues_rather_than_acting(app) -> None:
    """Called on the pystray thread, applied on the main thread: spawning a
    process and touching tkinter state from the tray thread is how the
    overlay got broken before."""
    import queue

    app._tray_events = queue.Queue()

    app._on_toggle_debug()

    assert app._tray_events.get_nowait() == "debug"
    assert app._debug_feed.opens == 0


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def test_nothing_is_published_while_the_window_is_closed(app) -> None:
    """Which is almost always. The tick must pay nothing for a window nobody
    opened — not even the cost of building a snapshot."""
    frame, pose, hands, face = pipeline_frame()

    app._publish_debug_frame(frame, pose, hands, face, make_result())

    assert app._debug_feed.published == []


def test_the_published_snapshot_carries_the_ticks_own_verdict(app) -> None:
    """The window's whole value is that it shows what the running detector
    decided, so it is handed the result object, not a recomputation."""
    app._handle_tray_event("debug")
    frame, pose, hands, face = pipeline_frame()
    result = make_result(model_score=0.4242, state="TRIGGERED")

    app._publish_debug_frame(frame, pose, hands, face, result)

    snapshot = app._debug_feed.published[-1]
    assert snapshot.result is result
    assert snapshot.frame is frame
    assert snapshot.pose is pose
    assert snapshot.face is face


def test_the_published_snapshot_says_which_detector_is_deciding(app) -> None:
    """A model can demote itself to the rule mid-session. The window has to
    show which one is answering now, not which one loaded at startup."""
    app._handle_tray_event("debug")
    frame, pose, hands, face = pipeline_frame()

    app._publish_debug_frame(frame, pose, hands, face, make_result())

    snapshot = app._debug_feed.published[-1]
    assert "модель" in snapshot.detector_status
    assert snapshot.model_threshold == pytest.approx(0.9660314917564392)


def test_the_published_snapshot_carries_the_thresholds_the_panel_prints(app) -> None:
    app._handle_tray_event("debug")
    frame, pose, hands, face = pipeline_frame()

    app._publish_debug_frame(frame, pose, hands, face, make_result())

    snapshot = app._debug_feed.published[-1]
    assert snapshot.combined_threshold == pytest.approx(0.55)
    assert snapshot.hair_zone_offset == pytest.approx(0.4)
    assert snapshot.fps == pytest.approx(6.75)
    assert snapshot.tick == 7


def test_no_model_means_no_model_threshold(app) -> None:
    app._twist_model = None
    app._analyzer = type("A", (), {"detector_name": "rule"})()
    app._detector_status = "правило"
    app._handle_tray_event("debug")
    frame, pose, hands, face = pipeline_frame()

    app._publish_debug_frame(frame, pose, hands, face, make_result(detector="rule"))

    assert app._debug_feed.published[-1].model_threshold is None


def test_a_failing_publish_cannot_escape_into_the_tick(app) -> None:
    """The one thing a diagnostic must never do is take down the thing it is
    diagnosing."""
    app._handle_tray_event("debug")
    app._debug_feed.raise_on_publish = RuntimeError("окно сломалось")
    frame, pose, hands, face = pipeline_frame()

    app._publish_debug_frame(frame, pose, hands, face, make_result())  # no raise


def test_a_failing_publish_closes_the_window(app) -> None:
    """Rather than raising once per tick for the rest of the session."""
    app._handle_tray_event("debug")
    app._debug_feed.raise_on_publish = RuntimeError("окно сломалось")
    frame, pose, hands, face = pipeline_frame()

    app._publish_debug_frame(frame, pose, hands, face, make_result())

    assert app._debug_view_open() is False


# ---------------------------------------------------------------------------
# Order of business inside the tick
# ---------------------------------------------------------------------------


def test_the_alert_is_raised_before_the_debug_window_is_fed() -> None:
    """Asserted on the source rather than trusted to review.

    This app exists to interrupt a behaviour inside a second or so. Twice now
    a side feature placed ahead of the overlay has cost it that: clip
    encoding on the main thread, and a notification library that stole the
    overlay's topmost status. The diagnostic publish is the third such
    passenger, and it goes last.
    """
    lines = inspect.getsource(HairGuardApp._tick).splitlines()
    overlay = next(i for i, line in enumerate(lines) if "self._overlay.show(" in line)
    publish = next(
        i for i, line in enumerate(lines) if "_publish_debug_frame(" in line
    )

    assert overlay < publish, "the debug publish got ahead of the alert"


def test_the_debug_window_is_closed_on_shutdown(app) -> None:
    """A viewer process outliving the app it was watching would sit there
    showing a frozen frame as though it were live."""
    app._handle_tray_event("debug")
    closed: list[int] = []
    app._debug_feed.close = lambda timeout=1.0: closed.append(1)

    HairGuardApp._close_debug_view(app)

    assert closed == [1]


# ---------------------------------------------------------------------------
# The frame rate the panel shows
# ---------------------------------------------------------------------------


def test_the_frame_rate_is_measured_from_real_tick_spacing() -> None:
    """The configured fps is 10 and the loop achieves about 6.75. The panel
    has to show the second number — the model's reaction time is quoted in
    frames, so the real rate is what converts it to seconds."""
    app = HairGuardApp.__new__(HairGuardApp)
    app._tick_fps = 0.0
    app._last_tick_at = None

    now = 1000.0
    for _ in range(80):
        app._update_tick_rate(now)
        now += 0.148  # ~6.75 fps

    assert app._tick_fps == pytest.approx(6.75, abs=0.3)


def test_the_first_tick_cannot_invent_a_frame_rate() -> None:
    app = HairGuardApp.__new__(HairGuardApp)
    app._tick_fps = 0.0
    app._last_tick_at = None

    app._update_tick_rate(500.0)

    assert app._tick_fps == 0.0


def test_a_long_gap_does_not_leave_a_stale_rate_on_screen() -> None:
    """The laptop sleeps, the camera goes, the app waits. Reporting the
    pre-sleep rate afterwards would hide exactly that."""
    app = HairGuardApp.__new__(HairGuardApp)
    app._tick_fps = 0.0
    app._last_tick_at = None

    now = 0.0
    for _ in range(40):
        app._update_tick_rate(now)
        now += 0.148
    app._update_tick_rate(now + 300.0)

    assert app._tick_fps < 1.0
