"""Tests for the debug feed — the app's half of the diagnostic window.

The product requirement is narrow and the failure modes are not. The window
has to show what the running detector sees; the detection tick it hangs off
also drives the tkinter event loop, the fullscreen overlay and the alert
chime. This project has twice had its core broken by a side feature — clip
encoding on the main thread, and a notification library stealing the
overlay's topmost status — so the feed is written so that the diagnostic
cannot reach the thing being diagnosed.

Everything below is one of two claims. Either: publishing costs the
detection tick nothing it can notice, whatever the viewer is doing. Or: when
the viewer breaks, the feed is what closes.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from hairguard import debugfeed, debugwire
from hairguard.analyzer import AnalysisResult
from hairguard.detector import PoseResult
from hairguard.face_detector import FaceResult


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeStdin:
    """A pipe that records what it was given, and can be made to hang."""

    def __init__(self) -> None:
        self.messages: list[bytes] = []
        self.closed = False
        self.block = threading.Event()
        self.block.set()  # not blocking by default
        self.entered_write = threading.Event()
        self.fail_with: BaseException | None = None
        self._lock = threading.Lock()

    def write(self, data: bytes) -> int:
        self.entered_write.set()
        self.block.wait()
        if self.fail_with is not None:
            raise self.fail_with
        if self.closed:
            raise ValueError("write to closed file")
        with self._lock:
            self.messages.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        if self.closed:
            raise ValueError("flush of closed file")

    def close(self) -> None:
        self.closed = True
        self.block.set()

    def payloads(self) -> list[bytes]:
        """Reassemble the length-prefixed stream into whole payloads."""
        import io

        with self._lock:
            blob = b"".join(self.messages)
        stream = io.BytesIO(blob)
        out = []
        while True:
            payload = debugwire.read_message(stream)
            if payload is None:
                return out
            out.append(payload)


class FakeProc:
    """Stands in for the viewer subprocess."""

    def __init__(self) -> None:
        self.stdin = FakeStdin()
        self.pid = 4321
        self.terminated = 0
        self.killed = 0
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def exit(self, code: int = 0) -> None:
        self._returncode = code

    def terminate(self) -> None:
        self.terminated += 1
        self._returncode = -1

    def kill(self) -> None:
        self.killed += 1
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            raise TimeoutError
        return self._returncode


def make_snapshot(tick: int = 0, width: int = 64) -> debugwire.Snapshot:
    return debugwire.Snapshot(
        frame=np.zeros((48, width, 3), dtype=np.uint8),
        pose=PoseResult(landmarks={}, raw_landmarks=None, detected=False),
        hands=[],
        face=FaceResult(landmarks={}, detected=False),
        result=AnalysisResult(
            should_alert=False,
            hand_near_head=False,
            state="IDLE",
            episode_started=False,
            episode_ended=False,
            episode_duration=0.0,
        ),
        fps=6.75,
        combined_threshold=0.55,
        hair_zone_offset=0.4,
        model_threshold=0.9,
        detector_status="модель",
        tick=tick,
    )


@pytest.fixture
def feed_and_proc():
    """A feed wired to a fake viewer, already running."""
    proc = FakeProc()
    notes: list[str] = []
    feed = debugfeed.DebugFeed(spawn=lambda: proc, log=notes.append)
    feed.open()
    assert _wait_until(lambda: feed.started, 2.0), "viewer was never spawned"
    yield feed, proc, notes
    feed.close(timeout=1.0)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


# ---------------------------------------------------------------------------
# Costing the detection tick nothing
# ---------------------------------------------------------------------------


def test_publishing_without_a_viewer_does_nothing():
    """The tray entry is closed almost always, and the tick pays nothing for
    a feature nobody opened."""
    feed = debugfeed.DebugFeed(spawn=lambda: pytest.fail("spawned unasked"))

    assert feed.active is False
    feed.publish(make_snapshot())  # must not raise, must not spawn

    assert feed.active is False


def test_opening_does_not_spawn_on_the_calling_thread():
    """Popen is tens of milliseconds and the caller is the detection tick,
    which at that moment may be holding a fullscreen alert on screen."""
    release = threading.Event()
    spawned = threading.Event()

    def slow_spawn():
        spawned.set()
        release.wait(2.0)
        return FakeProc()

    feed = debugfeed.DebugFeed(spawn=slow_spawn)
    started = time.monotonic()
    feed.open()
    elapsed = time.monotonic() - started

    assert elapsed < 0.05
    assert spawned.wait(2.0)
    assert feed.active is True  # requested, and not yet failed
    release.set()
    feed.close(timeout=2.0)


def test_publish_does_not_block_on_a_stalled_viewer(feed_and_proc):
    """A viewer that stops reading fills the pipe. The write blocks on the
    feed's own thread; the tick must not notice."""
    feed, proc, _ = feed_and_proc
    proc.stdin.block.clear()
    feed.publish(make_snapshot(tick=1))
    assert proc.stdin.entered_write.wait(2.0)

    started = time.monotonic()
    for tick in range(2, 60):
        feed.publish(make_snapshot(tick=tick))
    elapsed = time.monotonic() - started

    assert elapsed < 0.25, f"publish took {elapsed:.3f}s against a stalled viewer"
    proc.stdin.block.set()


def test_a_stalled_viewer_loses_frames_rather_than_queueing_them(feed_and_proc):
    """Backlogged diagnostic frames would be stale by the time they drew and
    unbounded in the meantime. Only the newest tick is worth sending."""
    feed, proc, _ = feed_and_proc
    proc.stdin.block.clear()
    feed.publish(make_snapshot(tick=1))
    assert proc.stdin.entered_write.wait(2.0)

    for tick in range(2, 40):
        feed.publish(make_snapshot(tick=tick))
    proc.stdin.block.set()

    assert _wait_until(lambda: len(proc.stdin.payloads()) >= 2)
    time.sleep(0.1)
    ticks = [debugwire.decode(p).tick for p in proc.stdin.payloads()]
    assert len(ticks) < 10, f"frames were queued, not dropped: {ticks}"
    assert ticks[-1] == 39, "the newest tick is the one that must arrive"


def test_the_published_snapshot_arrives_intact(feed_and_proc):
    feed, proc, _ = feed_and_proc

    feed.publish(make_snapshot(tick=77))

    assert _wait_until(lambda: proc.stdin.payloads())
    restored = debugwire.decode(proc.stdin.payloads()[-1])
    assert restored.tick == 77
    assert restored.detector_status == "модель"


def test_encoding_happens_off_the_detection_tick(feed_and_proc):
    """Named rather than implied: the JPEG encode is the expensive part, and
    it must happen on the feed's thread. Clip encoding on the main thread is
    a mistake this project has already made once."""
    feed, proc, _ = feed_and_proc
    encoded_on: list[str] = []

    def slow_encode(snapshot, **kwargs):
        encoded_on.append(threading.current_thread().name)
        time.sleep(0.05)
        return debugwire.encode(snapshot, **kwargs)

    feed._encode = slow_encode  # the seam the real feed uses
    publisher = threading.current_thread().name

    started = time.monotonic()
    feed.publish(make_snapshot(tick=5))
    elapsed = time.monotonic() - started

    assert elapsed < 0.02
    assert _wait_until(lambda: bool(encoded_on))
    assert encoded_on[0] != publisher


# ---------------------------------------------------------------------------
# Breaking without taking anything with it
# ---------------------------------------------------------------------------


def test_the_user_closing_the_window_closes_the_feed(feed_and_proc):
    """Closing the diagnostic window is the normal way out of it, and the
    app has to go back to publishing nothing."""
    feed, proc, _ = feed_and_proc

    proc.exit(0)
    feed.publish(make_snapshot(tick=1))

    assert _wait_until(lambda: feed.active is False)
    assert feed.error is None  # a clean exit is not a fault


def test_a_broken_pipe_closes_the_feed_without_raising(feed_and_proc):
    """The viewer being killed must surface as a closed feed, never as an
    exception travelling up through the detection tick."""
    feed, proc, _ = feed_and_proc
    proc.stdin.fail_with = BrokenPipeError(32, "broken pipe")

    feed.publish(make_snapshot(tick=1))  # must not raise

    assert _wait_until(lambda: feed.active is False)
    assert feed.error is not None
    assert "pipe" in feed.error.lower() or "32" in feed.error


def test_an_encoding_failure_closes_the_feed_and_is_reported(feed_and_proc):
    """A frame that cannot be encoded is a bug in the diagnostic, so the
    diagnostic is what stops — loudly enough to find in the log."""
    feed, proc, notes = feed_and_proc

    def exploding_encode(snapshot, **kwargs):
        raise ValueError("кадр не кодируется")

    feed._encode = exploding_encode
    feed.publish(make_snapshot(tick=1))

    assert _wait_until(lambda: feed.active is False)
    assert "не кодируется" in (feed.error or "")
    assert any("не кодируется" in note for note in notes)


def test_a_viewer_that_never_starts_leaves_the_app_running():
    """An exec failure is not a reason for anything to stop except the feed."""
    notes: list[str] = []

    def failing_spawn():
        raise FileNotFoundError("viewer not found")

    feed = debugfeed.DebugFeed(spawn=failing_spawn, log=notes.append)
    feed.open()

    assert _wait_until(lambda: feed.active is False)
    assert "viewer not found" in (feed.error or "")
    feed.publish(make_snapshot())  # still a no-op, still does not raise
    feed.close()


def test_publishing_after_close_is_a_no_op(feed_and_proc):
    feed, proc, _ = feed_and_proc

    feed.close(timeout=1.0)
    before = len(proc.stdin.payloads())
    feed.publish(make_snapshot(tick=9))
    time.sleep(0.05)

    assert len(proc.stdin.payloads()) == before
    assert feed.active is False


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------


def test_close_terminates_the_viewer(feed_and_proc):
    feed, proc, _ = feed_and_proc

    feed.close(timeout=1.0)

    assert proc.terminated >= 1
    assert feed.active is False


def test_close_is_idempotent(feed_and_proc):
    feed, proc, _ = feed_and_proc

    feed.close(timeout=1.0)
    feed.close(timeout=1.0)
    feed.close(timeout=1.0)

    assert feed.active is False


def test_close_returns_promptly_even_with_the_writer_stuck(feed_and_proc):
    """Shutdown runs on the way out of the tkinter mainloop. A diagnostic
    window cannot be allowed to hold the app's exit open."""
    feed, proc, _ = feed_and_proc
    proc.stdin.block.clear()
    feed.publish(make_snapshot(tick=1))
    assert proc.stdin.entered_write.wait(2.0)

    started = time.monotonic()
    feed.close(timeout=0.3)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"close blocked for {elapsed:.2f}s"
    assert proc.terminated >= 1
    proc.stdin.block.set()


def test_reopening_after_a_close_works(feed_and_proc):
    """Open it, look, close it, look again — the whole point of putting it in
    the tray rather than in a command."""
    feed, first, _ = feed_and_proc
    feed.close(timeout=1.0)

    second = FakeProc()
    feed._spawn = lambda: second
    feed.open()
    assert _wait_until(lambda: feed.started)
    feed.publish(make_snapshot(tick=3))

    assert _wait_until(lambda: second.stdin.payloads())
    assert debugwire.decode(second.stdin.payloads()[-1]).tick == 3


# ---------------------------------------------------------------------------
# Finding a viewer to spawn
# ---------------------------------------------------------------------------


def test_from_source_the_viewer_runs_through_the_interpreter() -> None:
    command = debugfeed.viewer_command()

    assert command == [sys.executable, "-m", "hairguard", debugfeed.VIEWER_COMMAND]


def test_a_bundle_spawns_its_console_binary(tmp_path, monkeypatch) -> None:
    """Not the windowed one. A windowed Windows executable is not guaranteed
    a usable standard input, and standard input is the entire channel."""
    console = tmp_path / ("hairguard-cli.exe" if sys.platform == "win32" else "hairguard-cli")
    console.write_bytes(b"MZ")
    windowed = tmp_path / ("HairGuard.exe" if sys.platform == "win32" else "HairGuard")
    windowed.write_bytes(b"MZ")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(windowed))

    command = debugfeed.viewer_command()

    assert command == [str(console), debugfeed.VIEWER_COMMAND]


def test_a_bundle_without_a_console_binary_spawns_itself(tmp_path, monkeypatch) -> None:
    """Degraded but not broken: a window that may not get frames beats a menu
    entry that does nothing."""
    windowed = tmp_path / ("HairGuard.exe" if sys.platform == "win32" else "HairGuard")
    windowed.write_bytes(b"MZ")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(windowed))

    assert debugfeed.viewer_command() == [str(windowed), debugfeed.VIEWER_COMMAND]


# ---------------------------------------------------------------------------
# The architectural claim, asserted rather than trusted
# ---------------------------------------------------------------------------


def test_neither_the_feed_nor_the_viewer_mentions_a_camera():
    """The design is that a second camera open is impossible, not merely
    avoided: on Windows it would return black frames rather than failing,
    so the viewer is given no camera to open.

    Checked at the source level because an import reached on only one branch
    is exactly the kind that survives review.
    """
    import ast

    for module in ("debugfeed", "debugviewer", "debugwire", "debugdraw"):
        path = Path(__file__).resolve().parent.parent / "hairguard" / f"{module}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        imported: set[str] = set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)

        assert "hairguard.camera" not in imported, f"{module} imports the camera module"
        assert "CameraManager" not in names, f"{module} reaches for the camera"
        assert "VideoCapture" not in names, f"{module} opens a capture device"


def test_the_viewer_process_cannot_reach_a_camera_at_all():
    """The stronger form of the same claim, over the whole import graph.

    The drawing code used to live beside a ``CameraManager`` import, which
    would have left the viewer process one attribute lookup away from
    opening the device the app is holding. Asserted in a fresh interpreter
    because this test suite imports the camera module for other reasons.
    """
    import subprocess

    probe = (
        "import sys; import hairguard.debugviewer; "
        "mods = [m for m in sys.modules if 'camera' in m]; "
        "print(mods)"
    )
    output = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert output.returncode == 0, output.stderr
    assert output.stdout.strip() == "[]", (
        f"the viewer process imports camera code: {output.stdout.strip()}"
    )
