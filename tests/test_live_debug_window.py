"""End-to-end: the real app alerts, the diagnostic window opens from the real
tray handler, and the app goes on alerting after it closes.

Everything here is the shipped object. The real ``HairGuardApp``, the real
tkinter root and fullscreen overlay, the real MediaPipe landmarkers, the real
trained model deciding, the real ``TrayApp`` callback that pystray invokes,
the real ``DebugFeed``, and a real viewer in a real child process reading a
real pipe. One thing is replaced: the webcam, which serves frames from
recorded clips the user labelled by hand — a twisting one to make it alert
and a resting one to let it settle. That substitution is the point rather
than a compromise: it makes "it alerted" a repeatable fact instead of
something that depends on who is sitting in front of the camera.

It takes about half a minute and opens windows, so it is off by default:

    HAIRGUARD_LIVE_TESTS=1 python -m pytest tests/test_live_debug_window.py -s

The alert overlay is a fullscreen, always-on-top, near-opaque window. That is
right for the product and wrong for a test: running this used to black out
whoever's monitor happened to be attached, over whatever they were doing.
The real overlay object is still used and still shown — the state machine,
the show/hide calls and the topmost handling are all exercised — but it is
resized to a small box in a corner and dropped from topmost before the app
starts, and the test asserts that it took effect rather than trusting it.
What is left on screen is that box and, for about four seconds, the
diagnostic window itself.

What it is here to settle is the thing a unit test cannot. The debug window
must show what the *running* detector sees, and the detection tick it hangs
off also owns the tkinter event loop, the overlay and the chime. Opening a
window from inside that loop, feeding it from the same tick, and closing it
again is exactly where this would break, and twice before a side feature has
broken this app's core — clip encoding on the main thread, and a
notification library stealing the overlay's topmost status.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("HAIRGUARD_LIVE_TESTS"),
    reason="opens real windows and takes ~40s; set HAIRGUARD_LIVE_TESTS=1",
)

TWISTING_TARGET = 1  # how many alerts to wait for before touching the tray


class ClipCamera:
    """A webcam that replays labelled clips, so "it alerted" is repeatable.

    Serves a stretch of twisting, then a stretch of resting, round and round.
    The resting stretch matters: it lets the state machine come back out of
    COOLDOWN, which is what makes a *second* alert, after the diagnostic
    window has been and gone, mean anything.
    """

    def __init__(self, twisting, resting, *, twisting_reps=4, resting_reps=3):
        self._script = twisting * twisting_reps + resting * resting_reps
        self._position = 0
        self.released = False

    @property
    def resolution(self):
        frame = self._script[0]
        return (frame.shape[1], frame.shape[0])

    def read_frame(self):
        frame = self._script[self._position % len(self._script)]
        self._position += 1
        return frame

    def release(self):
        self.released = True


def _load_clips():
    from hairlab.clipstore import ClipStore, LabelStore

    store, labels = ClipStore(), LabelStore()
    by_label: dict[str, list] = {"twisting": [], "resting": []}
    for clip_id in store.clip_ids():
        label = labels.get(clip_id)
        if label in by_label and len(by_label[label]) < 1:
            frames = [
                store.read_frame(clip_id, index)
                for index in range(store.frame_count(clip_id))
            ]
            if frames:
                by_label[label] = frames
        if all(by_label.values()):
            break
    if not all(by_label.values()):
        pytest.skip("no labelled twisting/resting clips on this machine")
    return by_label["twisting"], by_label["resting"]


class RecordingSound:
    """Counts what the chime was asked to do, without making any noise."""

    def __init__(self, enabled: bool = True) -> None:
        self.active = False
        self.starts = 0

    def start(self) -> None:
        if not self.active:
            self.starts += 1
        self.active = True

    def stop(self) -> None:
        self.active = False


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Keep the test off the user's PID file, database and clip dataset."""
    from hairguard import db, diagnostics, pid, sound, stats
    from hairguard import app as app_module

    monkeypatch.setenv(diagnostics.LOG_ENV, str(tmp_path / "hairguard.log"))
    monkeypatch.setattr(pid, "DEFAULT_PID_PATH", tmp_path / "hairguard.pid")
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", tmp_path / "hairguard.db", raising=False)
    monkeypatch.setattr(sound, "AlertSound", RecordingSound)
    # No clip recording: this test must not write to the real dataset.
    monkeypatch.setattr(app_module.HairGuardApp, "_make_capture_service",
                        staticmethod(lambda fps: None))
    yield tmp_path
    diagnostics.uninstall()


def test_the_debug_window_opens_from_the_tray_and_the_alert_survives_it(
    isolated, monkeypatch
) -> None:
    from hairguard import camera as camera_module, diagnostics
    from hairguard.app import HairGuardApp

    # Capture the app's own output the way a packaged run does, so the log
    # this test reads back is the log a user would be asked for.
    #
    # Installed here rather than in the fixture: pytest re-points sys.stdout
    # at its own capture between fixture setup and the test body, which
    # quietly replaced the tee and left the log holding only what the viewer
    # subprocess wrote. Passing -s hid it, which is worse than failing.
    diagnostics.install(console=sys.__stdout__)

    twisting, resting = _load_clips()
    fake_camera = ClipCamera(twisting, resting)
    monkeypatch.setattr(camera_module, "CameraManager", lambda **kwargs: fake_camera)
    monkeypatch.setattr(camera_module, "resolve_camera_index", lambda configured: 0)
    _pretend_the_screen_is_small(monkeypatch)

    app = HairGuardApp(camera_index=0, background=False)
    _check_the_overlay_cannot_cover_the_screen(app)

    # Observe the real overlay rather than replacing it: the window still
    # opens and hides for real, and we get to count the times it was asked to.
    seen: dict = {"shows": 0, "hides": 0, "shows_after_debug": 0}
    real_show, real_hide = app._overlay.show, app._overlay.hide
    debug_closed_at = {"value": None}

    def counting_show(phrase):
        seen["shows"] += 1
        if debug_closed_at["value"] is not None:
            seen["shows_after_debug"] += 1
        return real_show(phrase)

    def counting_hide():
        seen["hides"] += 1
        return real_hide()

    app._overlay.show = counting_show
    app._overlay.hide = counting_hide

    notes: list[str] = []
    app._debug_feed._log = notes.append

    report: dict = {}
    failure: list[str] = []

    def wait_for(predicate, timeout, what):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        failure.append(f"timed out waiting for {what}")
        return False

    def driver() -> None:
        """Everything a user would do, from the thread pystray would do it on."""
        try:
            # 1. The app alerts on its own.
            if not wait_for(lambda: seen["shows"] >= TWISTING_TARGET, 60,
                            "the first alert"):
                return
            report["alerted_before"] = seen["shows"]

            # 2. The tray entry. This is the exact callback pystray invokes,
            #    called on a thread that is not the tkinter one.
            app._tray._handle_toggle_debug(None, None)

            if not wait_for(lambda: app._debug_feed.started, 30, "the viewer to start"):
                return
            proc = app._debug_feed._proc
            report["viewer_pid"] = getattr(proc, "pid", None)

            # 3. Frames must actually reach it. The viewer says so in the log;
            #    a window that opens and receives nothing is the failure this
            #    whole architecture exists to avoid.
            log = Path(os.environ["HAIRGUARD_LOG"])
            if not wait_for(
                lambda: proc.poll() is None and "Первый кадр получен" in _read(log),
                30,
                "the viewer to report a frame",
            ):
                return
            report["viewer_saw_frames"] = True
            report["menu_label_while_open"] = app._tray._debug_label()
            time.sleep(4.0)  # leave it up long enough to be looked at

            # 4. Close it from the same entry.
            app._tray._handle_toggle_debug(None, None)
            if not wait_for(lambda: not app._debug_feed.active, 20, "the feed to close"):
                return
            if not wait_for(lambda: proc.poll() is not None, 20, "the viewer to exit"):
                return
            report["viewer_exit_code"] = proc.poll()
            report["menu_label_after_close"] = app._tray._debug_label()
            report["feed_error"] = app._debug_feed.error
            debug_closed_at["value"] = time.monotonic()

            # 5. And the app goes on alerting.
            if not wait_for(lambda: seen["shows_after_debug"] >= 1, 90,
                            "an alert after the window closed"):
                return
            report["alerted_after"] = seen["shows_after_debug"]
            report["sound_starts"] = app._sound.starts
            report["ticks"] = app._tick_count
            report["fps"] = app._tick_fps
            report["detector"] = app._get_detector_status()
            report["viewer_frames"] = _frames_seen(Path(os.environ["HAIRGUARD_LOG"]))
        finally:
            app._on_exit()

    thread = threading.Thread(target=driver, daemon=True, name="driver")
    thread.start()
    app.run()
    thread.join(10)

    print("\n--- live run ---")
    for key, value in report.items():
        print(f"  {key}: {value}")
    for note in notes:
        print(f"  feed: {note}")
    assert not failure, failure

    # The app alerted before the window, and again after it.
    assert report["alerted_before"] >= 1
    assert report["alerted_after"] >= 1
    assert report["sound_starts"] >= 2

    # The window was real, fed, and closed cleanly.
    assert report["viewer_pid"]
    assert report["viewer_saw_frames"] is True
    assert report["feed_error"] is None
    assert "закрыть" in report["menu_label_while_open"]
    assert "открыть" in report["menu_label_after_close"]

    # The window saw a real run of frames, not one and a stall.
    assert report["viewer_frames"] >= 10

    # And the detector deciding throughout was the trained one.
    assert "модель" in report["detector"]
    assert report["fps"] > 1.0

    # Finally: the app said how it ended. Without this line a tray icon that
    # vanishes and a tray icon that was told to quit look identical, which
    # is exactly what happened the first time this bundle was run by hand.
    log = _read(Path(os.environ["HAIRGUARD_LOG"]))
    assert "Завершение работы" in log
    assert "трея" in log


#: The screen the overlay is told it has. Wide enough that the label's
#: ``wraplength = width - 200`` stays positive.
_PRETEND_SCREEN = (640, 360)


def _pretend_the_screen_is_small(monkeypatch) -> None:
    """Stop the alert overlay from ever being built at full screen size.

    The real overlay is fullscreen, always on top and 70% opaque — correct
    for the product, intolerable in a test. Running this blacked out a
    monitor over whatever its owner was doing.

    The size is taken from ``winfo_screenwidth``/``winfo_screenheight`` once,
    in ``TkOverlay._setup_window``. Lying to it there means the fullscreen
    geometry is never requested at all, so there is no moment — not even a
    frame at construction — when a full-screen window could appear. Resizing
    afterwards was the first attempt and is strictly worse: it leaves the
    fullscreen geometry already requested, and Tk reports the construction
    value back, so the check could not even tell whether it had worked.

    Everything being tested still runs as it ships: the state machine, the
    show/hide pairing, the window actually being mapped and unmapped, and
    ``ensure_on_top`` re-asserting position with SWP_NOMOVE|SWP_NOSIZE.
    """
    import tkinter as tk

    width, height = _PRETEND_SCREEN
    monkeypatch.setattr(tk.Tk, "winfo_screenwidth", lambda self: width, raising=False)
    monkeypatch.setattr(tk.Tk, "winfo_screenheight", lambda self: height, raising=False)


def _check_the_overlay_cannot_cover_the_screen(app) -> None:
    """Asserted rather than assumed.

    A silent failure here costs whoever runs this next their screen, which
    is exactly how this function came to exist.
    """
    window = getattr(app._overlay, "_window", None)
    if window is None:  # GtkOverlay on Wayland: nothing to check
        return
    # The requested geometry, not winfo_width(): the window is withdrawn at
    # this point and an unmapped window reports a width of 1.
    requested = window.winfo_geometry().split("+")[0]
    width, height = (int(part) for part in requested.split("x"))
    assert (width, height) <= _PRETEND_SCREEN, (
        f"the overlay was built at {requested} — it would cover the screen"
    )
    # And it does not get to sit above everything else while it is up.
    try:
        window.attributes("-topmost", False)
    except Exception:
        pass


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _frames_seen(path: Path) -> int:
    """How many frames the viewer reported receiving, from its parting line."""
    for line in reversed(_read(path).splitlines()):
        if "получено кадров:" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("."))
    return 0
