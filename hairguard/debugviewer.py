"""hairguard.debugviewer — the diagnostic window, fed by the running app.

This process owns a window and nothing else. It has no camera, no detector
and no model; every frame and every number it draws arrived on its standard
input from the detection tick of the app that spawned it (see
``hairguard.debugfeed``). That is the point: the question the window exists
to answer is "what is the detector seeing *right now*", and anything it
sourced itself would be a different answer dressed as that one.

Two things are deliberate in the loop below.

*The reader is a thread, the drawing is the main loop.* Blocking on the pipe
and drawing in the same loop would be shorter, and would freeze the window
whenever the app stops publishing — which it does on pause, exactly when a
user is most likely to reach for the close button. So frames land in a
one-slot mailbox and the display loop runs on its own clock, staying
responsive to ESC and to the window's close button whether or not anything
is arriving.

*The hair zone is drawn unsmoothed.* The standalone debug command applies an
EMA to the ellipse so it does not jitter, which means it has always drawn a
slightly different zone from the one the analyzer gates on. Here the zone is
recomputed from the landmarks the analyzer used, with no smoothing, so what
is on screen is the gate — jitter included.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import BinaryIO, Optional

import numpy as np

from hairguard import debugwire
from hairguard.debugdraw import (
    PANEL_HEIGHT,
    draw_footer,
    draw_hair_zone,
    draw_hand_skeletons,
    draw_pinch_indicator,
    draw_pose_landmarks,
    draw_status_panel,
)

#: The title bar, which Windows and X11 draw themselves and can render.
WINDOW_TITLE = "HairGuard — что видит детектор"

#: Display loop period. Frames arrive at the detection rate (~7 fps); the
#: loop runs faster so the keyboard and the close button stay live.
_FRAME_WAIT_MS = 30

#: No frame for this long means the app has stopped publishing — paused,
#: wedged, or without a camera. Said out loud rather than shown as a stale
#: picture that looks current.
_STALE_SEC = 2.0

#: Narrower frames are enlarged before anything is drawn on them. The status
#: panel is a fixed 150 px tall and two columns wide; on the 256-pixel clips
#: this project's dataset is made of, it covers the picture and the columns
#: write over each other.
MIN_DISPLAY_WIDTH = 720


class _Reader:
    """Pulls snapshots off a stream into a one-slot mailbox."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._latest: Optional[debugwire.Snapshot] = None
        self._received = 0
        self._error: Optional[str] = None
        self._ended = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="debug-reader")

    def start(self) -> None:
        self._thread.start()

    @property
    def ended(self) -> bool:
        return self._ended.is_set()

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def received(self) -> int:
        return self._received

    def take(self) -> Optional[debugwire.Snapshot]:
        """The newest snapshot, or None if none has arrived since the last call."""
        with self._lock:
            snapshot, self._latest = self._latest, None
            return snapshot

    def _run(self) -> None:
        try:
            while True:
                payload = debugwire.read_message(self._stream)
                if payload is None:
                    return
                snapshot = debugwire.decode(payload)
                with self._lock:
                    self._latest = snapshot
                    self._received += 1
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ended.set()


def _placeholder(width: int = 960, height: int = 540) -> np.ndarray:
    return np.full((height, width, 3), 28, dtype=np.uint8)


def _footer_lines(
    snapshot: Optional[debugwire.Snapshot],
    age: Optional[float],
    received: int,
) -> list[str]:
    """The two lines along the bottom edge. ASCII only, and that is not style.

    ``cv2.putText`` draws with Hershey vector fonts, which contain no
    Cyrillic: every Russian character handed to it comes out as ``?``. This
    window used to caption itself in Russian like the rest of the app, and
    rendered a status line reading ``???????? ??????``. The panel above has
    always been English for the same reason. The Russian description still
    travels in the snapshot and still reaches the log, which is UTF-8 and
    renders it properly.
    """
    if snapshot is None:
        return [
            "Waiting for frames from HairGuard...",
            "This window shows what the running detector sees. It has no camera.",
        ]

    if snapshot.result.detector == "model":
        threshold = snapshot.model_threshold
        deciding = (
            f"Detector: model (fires at {threshold:.2f})"
            if threshold is not None
            else "Detector: model"
        )
    else:
        deciding = "Detector: rule (threshold)"

    if age is not None and age > _STALE_SEC:
        return [
            deciding,
            f"No frames for {age:.0f}s - HairGuard is paused or has no video.",
        ]
    return [deciding, f"tick {snapshot.tick} - {received} frames - ESC closes"]


def _fit(frame: np.ndarray) -> np.ndarray:
    """Enlarge a small frame so the status panel has room. Never shrinks."""
    width = int(frame.shape[1])
    if width >= MIN_DISPLAY_WIDTH:
        return np.array(frame, copy=True)
    import cv2

    scale = MIN_DISPLAY_WIDTH / width
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(
        frame, (MIN_DISPLAY_WIDTH, height), interpolation=cv2.INTER_LINEAR
    )


def render(snapshot: debugwire.Snapshot) -> np.ndarray:
    """Draw one snapshot, returning a new image. Pure, so it is testable."""
    display = _fit(np.asarray(snapshot.frame))
    draw_hair_zone(
        display,
        snapshot.face,
        snapshot.pose,
        snapshot.hair_zone_offset,
        # No smoother: see the module docstring.
        smoother=None,
    )
    draw_pose_landmarks(display, snapshot.pose)
    draw_hand_skeletons(display, snapshot.hands, snapshot.result, snapshot.combined_threshold)
    draw_pinch_indicator(display, snapshot.hands, snapshot.combined_threshold)
    draw_status_panel(
        display,
        snapshot.result,
        snapshot.fps,
        snapshot.combined_threshold,
        model_threshold=snapshot.model_threshold,
        hint="ESC / Q: close",
        title="HairGuard - what the detector sees",
    )
    return display


def main(stream: Optional[BinaryIO] = None) -> int:
    """Run the viewer until the feed ends or the user closes the window."""
    import cv2

    if stream is None:
        stream = getattr(sys.stdin, "buffer", None)
    if stream is None:
        print(
            "[debugviewer] Нет входного потока — это окно запускается "
            "приложением из меню в трее, а не вручную."
        )
        return 2

    reader = _Reader(stream)
    reader.start()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 640)
    print("[debugviewer] Окно открыто, жду кадры от приложения.")

    latest: Optional[debugwire.Snapshot] = None
    last_arrival: Optional[float] = None

    try:
        while True:
            snapshot = reader.take()
            if snapshot is not None:
                # Announced because "the window opened but shows nothing" and
                # "the detector is seeing nothing" look identical on screen,
                # and only one of them is a bug in this plumbing.
                if latest is None:
                    print(f"[debugviewer] Первый кадр получен (tick {snapshot.tick}).")
                latest = snapshot
                last_arrival = time.monotonic()

            if latest is None:
                display = _placeholder()
            else:
                display = render(latest)

            age = None if last_arrival is None else time.monotonic() - last_arrival
            draw_footer(display, _footer_lines(latest, age, reader.received))
            cv2.imshow(WINDOW_TITLE, display)

            key = cv2.waitKey(_FRAME_WAIT_MS) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            try:
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if reader.ended and reader.take() is None:
                # The app is gone. Nothing further will ever arrive, and a
                # window showing a frozen frame would read as a live one.
                break
    finally:
        if reader.error:
            print(f"[debugviewer] Поток кадров оборвался: {reader.error}")
        print(f"[debugviewer] Окно закрыто, получено кадров: {reader.received}.")
        try:
            cv2.destroyWindow(WINDOW_TITLE)
        except Exception:
            pass
        try:
            cv2.waitKey(1)  # let the window manager actually take it down
        except Exception:
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
