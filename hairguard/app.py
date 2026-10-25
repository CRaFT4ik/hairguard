"""app.py — HairGuardApp: wires all components into the main application loop.

Threading model:
  1. Main thread   — tkinter mainloop + detection tick (via root.after)
  2. Tray thread   — pystray daemon thread (started by TrayApp.start())
  3. Camera thread — captures frames into queue.Queue (daemon thread)
"""

from __future__ import annotations

import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional


class HairGuardApp:
    """Main application class that wires all HairGuard components together."""

    # Update tray tooltip every N ticks (~6 seconds at 5 FPS)
    _TOOLTIP_EVERY_N_TICKS = 30
    # Re-read the shared collection switch about once a second
    _COLLECT_SYNC_EVERY_N_TICKS = 10
    # Grace before the alert sound stops, so a dropped frame of hand
    # tracking does not chop it into stutters
    _SOUND_LINGER_SEC = 0.6

    def __init__(
        self,
        camera_index: Optional[int] = None,
        background: bool = False,
        collect: Optional[bool] = None,
    ) -> None:
        from hairguard.config import Config
        from hairguard.pid import PidFile
        from hairguard.db import Database
        from hairguard.stats import StatsRecorder
        from hairguard.phrases import PhraseManager
        from hairguard.camera import CameraManager, resolve_camera_index
        from hairguard.detector import PoseDetector
        from hairguard.hand_detector import HandDetector
        from hairguard.face_detector import FaceDetector
        from hairguard.analyzer import GestureAnalyzer
        from hairguard.overlay import create_overlay
        from hairguard.tray import TrayApp

        # --- Config ---
        self._config = Config()

        # Resolve camera index: CLI arg > config file. A configured "auto"
        # (or a missing setting) probes the attached cameras and picks the
        # first one showing a live image.
        if camera_index is None:
            camera_index = resolve_camera_index(self._config.get("camera"))
        self._camera_index = camera_index

        # --- Instance guard (actual acquire happens in run()) ---
        self._pid_file = PidFile()

        # --- Database + stats ---
        self._db = Database()
        self._stats = StatsRecorder(self._db)

        # --- Phrase manager ---
        self._phrases = PhraseManager()

        # --- Camera ---
        # Captured at the configured resolution; detection runs on a
        # downscaled copy so the extra pixels cost nothing at runtime and
        # only serve the dataset crops.
        self._camera_size = (
            int(self._config.get("camera_width") or 1280),
            int(self._config.get("camera_height") or 720),
        )
        self._process_width = int(self._config.get("detection", "process_width") or 640)
        self._camera = CameraManager(
            camera_index=self._camera_index,
            width=self._camera_size[0],
            height=self._camera_size[1],
        )
        print(f"[app] Камера {self._camera_index}: {self._camera.resolution[0]}x{self._camera.resolution[1]}")

        # --- Detector + Analyzer ---
        min_conf: float = float(self._config.get("detection", "min_confidence") or 0.5)
        # Building the landmarkers and running the first inference is where
        # MediaPipe prints its dozen startup notes. Silenced so that the
        # messages the user can act on are not buried under them.
        from hairguard.quiet import suppressed_native_stderr

        with suppressed_native_stderr():
            self._detector = PoseDetector(min_confidence=min_conf)
            self._hand_detector = HandDetector(max_hands=2, min_confidence=min_conf)
            self._face_detector = FaceDetector(min_confidence=min_conf)
            self._warm_up_detectors()
        self._analyzer = GestureAnalyzer(self._config)

        # Derive tick interval from configured FPS
        fps = float(self._config.get("detection", "fps") or 10)
        self._TICK_MS = max(50, int(1000 / fps))

        # --- Alert sound (plays on its own thread) ---
        from hairguard.sound import AlertSound

        self._sound = AlertSound(enabled=bool(self._config.get("alert", "sound")))
        self._alert_gone_at: Optional[float] = None

        # --- Overlay (creates the tkinter root) ---
        self._overlay = create_overlay(self._config)
        # Obtain the tkinter root so we can schedule callbacks
        if hasattr(self._overlay, "get_root"):
            self._root = self._overlay.get_root()
        else:
            # GtkOverlay on Wayland — create a minimal hidden Tk root for the event loop
            import tkinter as tk
            self._root = tk.Tk()
            self._root.withdraw()

        # --- Dataset collection ---
        # The switch is shared with the command line through a file, so it can
        # be set either way and inspected afterwards.
        #
        # It deliberately survives a restart. An earlier version reset it to
        # off on every start, reasoning that a login-launched app must never
        # resume recording by itself — but collecting a dataset takes weeks,
        # and a switch that silently forgets is a switch that never records.
        # Instead the state is announced loudly at startup, so recording is
        # never a surprise.
        self._capture = self._make_capture_service(fps)
        self._collect_state = self._make_collect_state()
        self._collect_hint_shown = False

        if collect is not None and self._collect_state is not None:
            try:
                self._collect_state.write(bool(collect))
            except Exception:
                pass

        # --- Tray ---
        self._tray = TrayApp(
            on_pause=self._on_pause,
            on_resume=self._on_resume,
            on_exit=self._on_exit,
            stats_getter=self._get_stats_summary,
            config=self._config,
            on_toggle_collect=self._on_toggle_collect if self._capture else None,
            collect_status_getter=self._get_collect_status if self._capture else None,
        )

        # --- Frame queue (maxsize=2 — drop old frames if detection is slow) ---
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        # Tray events queue (thread-safe communication from tray thread to main thread)
        self._tray_events: queue.Queue = queue.Queue()

        # --- State flags ---
        self._running = False
        self._paused = False
        self._tick_count = 0

        # Episode tracking: store when the current episode started
        self._episode_started_at: Optional[datetime] = None

        # --- Signal handlers ---
        # SIGINT (Ctrl-C) and SIGTERM both trigger a clean shutdown via tkinter
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle_signal)
            except (OSError, ValueError):
                # Not always possible (e.g. non-main thread), ignore
                pass

    @staticmethod
    def _make_collect_state():
        """The shared collection switch, or None if hairlab is unavailable."""
        try:
            from hairlab.state import CollectionState

            return CollectionState()
        except Exception:
            return None

    def _warm_up_detectors(self) -> None:
        """Run one inference per model on a blank frame.

        MediaPipe emits part of its startup logging on the first inference
        rather than at construction, so this happens while stderr is still
        silenced. It also pays the one-off graph setup cost before the first
        real frame instead of during it.
        """
        import numpy as np

        blank = np.zeros((self._camera_size[1], self._camera_size[0], 3), dtype=np.uint8)
        try:
            self._detector.process(blank, timestamp_ms=1)
            self._hand_detector.detect(blank, timestamp_ms=1)
            self._face_detector.detect(blank, timestamp_ms=1)
        except Exception:
            # A warm-up failure is not a reason to refuse to start.
            pass

    @staticmethod
    def _make_capture_service(fps: float):
        """Build the dataset collector, or None if hairlab is unavailable.

        The app must keep running for users who only installed hairguard, so
        a missing hairlab simply hides the collection entry from the tray.
        """
        try:
            from hairlab.capture import CaptureService

            # A two-second clip, whatever the configured frame rate.
            return CaptureService(window_frames=max(4, int(round(2.0 * fps))), fps=fps)
        except Exception as exc:
            print(f"[app] Сбор данных недоступен: {exc}")
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Acquire PID file, start threads, run tkinter mainloop."""
        if not self._pid_file.acquire():
            print("HairGuard уже запущен. Используйте 'hairguard stop' для остановки.")
            sys.exit(1)

        self._running = True
        self._announce_collection_state()

        # Start camera capture thread
        cam_thread = threading.Thread(
            target=self._camera_loop,
            daemon=True,
            name="camera",
        )
        cam_thread.start()

        # Start tray icon thread
        self._tray.start()

        # Schedule the first detection tick
        self._root.after(self._TICK_MS, self._tick)

        try:
            self._root.mainloop()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean up all resources. Safe to call multiple times."""
        self._running = False

        # Silence the alert before anything else: a beep that outlives the
        # app it belongs to has no way of being stopped.
        try:
            self._sound.stop()
        except Exception:
            pass

        # Stop collecting, drop buffered frames, and let the writer finish
        # whatever it already accepted rather than losing a clip on exit.
        try:
            if self._capture is not None:
                self._capture.close()
        except Exception:
            pass

        # Release camera
        try:
            self._camera.release()
        except Exception:
            pass

        # Close MediaPipe detector
        try:
            self._detector.close()
        except Exception:
            pass
        try:
            self._hand_detector.close()
        except Exception:
            pass
        try:
            self._face_detector.close()
        except Exception:
            pass

        # PhraseManager saves on every get_next_phrase(); no explicit save needed
        # but call it as a no-op safety measure if it ever gains one.

        # Stop tray
        try:
            self._tray.stop()
        except Exception:
            pass

        # Release PID file
        try:
            self._pid_file.release()
        except Exception:
            pass

        # Close database
        try:
            self._db.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Camera thread
    # ------------------------------------------------------------------

    def _camera_loop(self) -> None:
        """Daemon thread: continuously reads frames and puts them in the queue.

        Survives camera disappearance (system sleep, USB unplug). After a
        short streak of read failures we close the capture and re-open it
        on a backoff schedule, retrying forever until the user explicitly
        stops the app.
        """
        from hairguard.camera import CameraManager

        fps: float = float(self._config.get("detection", "fps") or 10)
        sleep_sec = 1.0 / fps
        consecutive_failures = 0
        # ~2 seconds of failures before assuming the camera is gone
        reopen_after = max(int(2.0 * fps), 5)
        # Backoff between reopen attempts — grows on repeated failures
        reopen_backoff = 1.0
        max_reopen_backoff = 30.0
        announced_lost = False

        while self._running:
            try:
                frame = self._camera.read_frame()
            except Exception:
                frame = None

            if frame is not None:
                if announced_lost:
                    print("[app] Камера снова доступна.")
                    announced_lost = False
                consecutive_failures = 0
                reopen_backoff = 1.0
                # Drop the oldest frame if the queue is full, then put the new one
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Race condition — ignore
                time.sleep(sleep_sec)
                continue

            consecutive_failures += 1
            if consecutive_failures < reopen_after:
                time.sleep(sleep_sec)
                continue

            # Camera looks gone. Drop any stale queued frames so the next
            # tick after recovery does not analyze pre-sleep video.
            while True:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

            if not announced_lost:
                print(
                    "[app] Камера недоступна (возможно, система ушла в сон). "
                    "Жду подключения..."
                )
                announced_lost = True

            try:
                self._camera.release()
            except Exception:
                pass

            # Sleep the backoff in small chunks so shutdown is responsive.
            slept = 0.0
            while slept < reopen_backoff and self._running:
                time.sleep(min(0.25, reopen_backoff - slept))
                slept += 0.25
            if not self._running:
                return

            try:
                self._camera = CameraManager(
                    camera_index=self._camera_index,
                    width=self._camera_size[0],
                    height=self._camera_size[1],
                )
                consecutive_failures = 0
                reopen_backoff = 1.0
            except Exception:
                reopen_backoff = min(reopen_backoff * 2.0, max_reopen_backoff)

    # ------------------------------------------------------------------
    # Detection tick (main thread, scheduled via root.after)
    # ------------------------------------------------------------------

    def _downscale_for_detection(self, frame):
        """Shrink a frame to the configured processing width, keeping aspect."""
        width = frame.shape[1]
        if width <= self._process_width:
            return frame
        import cv2

        scale = self._process_width / width
        return cv2.resize(
            frame,
            (self._process_width, max(1, int(round(frame.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _handle_tray_event(self, event: str) -> bool:
        """Apply one tray event on the main thread. False means shut down."""
        if event == "pause":
            self._paused = True
            self._hide_overlay_if_visible()
            self._silence_alert()
        elif event == "resume":
            self._paused = False
        elif event == "collect_toggle":
            self._toggle_collection()
        elif event == "exit":
            return False
        return True

    def _tick(self) -> None:
        """Called every 200 ms on the main thread via root.after()."""
        if not self._running:
            return

        self._tick_count += 1

        # Drain tray events (thread-safe)
        while not self._tray_events.empty():
            try:
                event = self._tray_events.get_nowait()
            except queue.Empty:
                break
            if not self._handle_tray_event(event):
                self._running = False
                self._root.quit()
                return

        # Adopt the shared collection switch every second or so, so the
        # command line can start and stop a running app.
        if self._tick_count % self._COLLECT_SYNC_EVERY_N_TICKS == 0:
            try:
                self._sync_collection_state()
            except Exception as exc:
                print(f"[app] Не удалось прочитать состояние сбора: {exc}")

        # If paused, skip detection but keep rescheduling
        if self._paused:
            self._root.after(self._TICK_MS, self._tick)
            return

        # Drain the queue and keep only the newest frame so detection
        # always runs on the freshest camera data. DebugView reads frames
        # directly in a tight loop, so to keep engine behaviour identical
        # here we must not feed the analyzer a stale queued frame.
        frame = None
        while True:
            try:
                frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        if frame is None:
            self._root.after(self._TICK_MS, self._tick)
            return

        # --- Detection pipeline ---
        frame_height = frame.shape[0] if hasattr(frame, "shape") else 480
        frame_width = frame.shape[1] if hasattr(frame, "shape") and len(frame.shape) > 1 else 640

        # Detection runs on a downscaled copy: landmarks are normalised, so
        # they apply equally to the full-resolution frame the clips are cut
        # from, while the models keep working on the pixel count they were
        # tuned for.
        detect_frame = self._downscale_for_detection(frame)

        ts_base = int(time.monotonic() * 1000)
        pose_result = self._detector.process(detect_frame, timestamp_ms=ts_base)
        hand_results = self._hand_detector.detect(detect_frame, timestamp_ms=ts_base + 1)
        face_result = self._face_detector.detect(detect_frame, timestamp_ms=ts_base + 2)
        result = self._analyzer.analyze(pose_result, hand_results, face_result, frame_height, frame_width)

        # --- Overlay management ---
        # Deliberately ahead of dataset collection. The alert is what the app
        # is for; collection is a side activity that must never delay it or
        # take it down.
        # The sound follows the hand rather than the overlay: it repeats
        # while the hand is at the head and stops when it leaves.
        self._update_alert_sound(result.state, now=time.monotonic())

        if result.should_alert and not self._overlay.is_visible():
            self._hint_collection_is_off()
            self._overlay.show(self._phrases.get_next_phrase())
        elif not result.should_alert and self._overlay.is_visible():
            self._overlay.hide()
        elif result.should_alert:
            # Keep claiming the top of the window stack while the alert is up.
            try:
                self._overlay.ensure_on_top()
            except AttributeError:
                pass

        # --- Episode tracking ---
        if result.episode_started:
            self._episode_started_at = datetime.now()

        if result.episode_ended and self._episode_started_at is not None:
            try:
                self._stats.record_episode(
                    started_at=self._episode_started_at,
                    duration_sec=result.episode_duration,
                )
            except Exception as exc:
                print(f"[app] Не удалось сохранить эпизод: {exc}")
            self._episode_started_at = None

        # --- Dataset collection (last, and never allowed to matter) ---
        # Driven by the geometric verdict only: the clip is kept whenever the
        # hand is at the head, and what it was doing is decided later by hand.
        # Encoding happens on a worker thread, so this call only crops.
        if self._capture is not None and self._capture.enabled:
            try:
                clip_id = self._capture.on_tick(
                    frame,
                    pose_result,
                    hand_results,
                    in_zone=result.fingertips_in_zone,
                    hand_index=result.in_zone_hand_index,
                    face_result=face_result,
                )
                if clip_id:
                    print(
                        f"[app] Клип сохранён ({self._capture.saved_count} за сеанс): {clip_id}"
                    )
            except Exception as exc:
                print(f"[app] Сбор клипа не удался: {exc}")

        # --- Periodic tray tooltip update ---
        if self._tick_count % self._TOOLTIP_EVERY_N_TICKS == 0:
            try:
                self._tray.update_tooltip(
                    f"HairGuard — {self._get_stats_summary()}"
                )
            except Exception:
                pass

        # Reschedule
        self._root.after(self._TICK_MS, self._tick)

    # ------------------------------------------------------------------
    # Tray callbacks (called from tray thread — must not touch tkinter directly)
    # ------------------------------------------------------------------

    def _on_pause(self) -> None:
        """Called from tray thread when user clicks Pause."""
        self._tray_events.put("pause")

    def _on_resume(self) -> None:
        """Called from tray thread when user clicks Resume or auto-resume fires."""
        self._tray_events.put("resume")

    def _on_toggle_collect(self) -> None:
        """Called from tray thread when the user toggles dataset collection."""
        self._tray_events.put("collect_toggle")

    def _update_alert_sound(self, state: str, now: float) -> None:
        """Sound while the hand is at the head; fall silent when it moves away.

        Tied to ALERTING rather than to the overlay, which lingers through
        the dismissal delay: the noise should end the moment the hand does,
        because that is what makes moving it away worth doing.

        Hand tracking drops a frame now and then, so a short grace period
        keeps a flicker from chopping the sound into stutters.
        """
        if state == "ALERTING":
            self._alert_gone_at = None
            self._sound.start()
            return

        if not self._sound.active:
            return

        if self._alert_gone_at is None:
            self._alert_gone_at = now
        elif now - self._alert_gone_at >= self._SOUND_LINGER_SEC:
            self._sound.stop()
            self._alert_gone_at = None

    def _announce_collection_state(self) -> None:
        """Say plainly, at startup, whether clips are being recorded.

        Recording is a thing a person must never have to guess about — in
        either direction. Off and silent means a dataset that never gets
        collected; on and silent means a camera quietly writing you to disk.
        """
        if self._capture is None or self._collect_state is None:
            return

        if self._collect_state.read():
            self._apply_collection(True, record=False, announce=False)
            print("=" * 62)
            print("  ЗАПИСЬ КЛИПОВ ВКЛЮЧЕНА")
            print(f"  Клипы: {self._capture.clips_root}")
            print("  Выключить: hairguard collect off")
            print("=" * 62)
        else:
            print("=" * 62)
            print("  Запись клипов выключена — клипы сохраняться НЕ будут.")
            print("  Включить: hairguard collect on")
            print("=" * 62)

    def _hint_collection_is_off(self) -> None:
        """Mention the switch the first time detection fires without recording.

        This is the moment the user expects a clip, so it is the moment the
        absence of one has to be explained.
        """
        if self._collect_hint_shown or self._capture is None:
            return
        if self._capture.enabled:
            return
        self._collect_hint_shown = True
        print(
            "[app] Обнаружено кручение, но запись выключена — клип не сохранён. "
            "Включить: hairguard collect on"
        )

    def _get_collect_status(self) -> tuple[bool, int]:
        """(collecting, clips saved this session) for the tray menu label."""
        if self._capture is None:
            return False, 0
        return self._capture.enabled, self._capture.saved_count

    def _toggle_collection(self) -> None:
        """Flip collection, from the tray. Must run on the main thread."""
        if self._capture is None:
            return
        self._apply_collection(not self._capture.enabled)

    def _sync_collection_state(self) -> None:
        """Adopt the shared switch, so `hairguard collect on` reaches us.

        The tray is not the only way in, and when clips failed to appear
        there was no way to tell a missed click from a broken menu. The file
        is checked periodically and applied when it disagrees with reality.
        """
        if self._capture is None or self._collect_state is None:
            return
        wanted = self._collect_state.read()
        if wanted != self._capture.enabled:
            self._apply_collection(wanted, record=False)

    def _apply_collection(self, enabled: bool, record: bool = True, announce: bool = True) -> None:
        """Start or stop collecting and say so. Must run on the main thread.

        A switch that records the user has to be loud about it, so the new
        state is announced three ways: the console, a desktop notification
        and the tray menu entry.
        """
        if self._capture is None:
            return

        if enabled:
            self._capture.start()
        else:
            self._capture.stop()

        if record and self._collect_state is not None:
            try:
                self._collect_state.write(enabled)
            except Exception as exc:
                print(f"[app] Не удалось записать состояние сбора: {exc}")

        self._collect_hint_shown = False

        try:
            self._tray.refresh_menu()
        except Exception:
            pass

        if not announce:
            # Startup prints its own banner; no need to say it twice.
            return

        # No desktop notification here: plyer creates a window of its own on
        # Windows, and nothing about dataset collection is worth the risk of
        # disturbing the fullscreen alert. The console line and the tray entry
        # already say it.
        if enabled:
            print(f"[app] Запись клипов ВКЛЮЧЕНА. Клипы: {self._capture.clips_root}")
        else:
            print(
                f"[app] Запись клипов выключена. Сохранено за сеанс: "
                f"{self._capture.saved_count}"
            )

    def _on_exit(self) -> None:
        """Called from tray thread or signal handler."""
        self._tray_events.put("exit")

    def _silence_alert(self) -> None:
        """Stop the alert sound. Pausing or quitting must not leave it ringing."""
        try:
            self._sound.stop()
        except Exception:
            pass
        self._alert_gone_at = None

    def _hide_overlay_if_visible(self) -> None:
        """Hide overlay if visible. Must be called on main thread."""
        if self._overlay.is_visible():
            self._overlay.hide()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame: object) -> None:
        """Handle SIGTERM/SIGINT: initiate clean shutdown via tkinter."""
        self._on_exit()

    # ------------------------------------------------------------------
    # Stats summary for tray tooltip
    # ------------------------------------------------------------------

    def _get_stats_summary(self) -> str:
        """Return today's stats as a short string for the tray tooltip."""
        try:
            from datetime import date
            today = date.today()
            from datetime import datetime as _dt
            from_dt = _dt(today.year, today.month, today.day, 0, 0, 0)
            to_dt = _dt(today.year, today.month, today.day, 23, 59, 59, 999999)

            episodes = self._stats.get_episodes(from_dt, to_dt)
            count = len(episodes)
            total_sec = int(sum(ep.duration_sec for ep in episodes))
            minutes, secs = divmod(total_sec, 60)

            if minutes:
                time_str = f"{minutes}м {secs}с"
            else:
                time_str = f"{secs}с"

            return f"Сегодня: {count} эпизодов, {time_str}"
        except Exception:
            return "Сегодня: нет данных"
