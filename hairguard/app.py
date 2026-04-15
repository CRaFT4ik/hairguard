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

    def __init__(self, camera_index: Optional[int] = None, background: bool = False) -> None:
        from hairguard.config import Config
        from hairguard.pid import PidFile
        from hairguard.db import Database
        from hairguard.stats import StatsRecorder
        from hairguard.phrases import PhraseManager
        from hairguard.camera import CameraManager
        from hairguard.detector import PoseDetector
        from hairguard.hand_detector import HandDetector
        from hairguard.hair_segmenter import HairSegmenter
        from hairguard.analyzer import GestureAnalyzer
        from hairguard.overlay import create_overlay
        from hairguard.tray import TrayApp

        # --- Config ---
        self._config = Config()

        # Resolve camera index: CLI arg > config file > default 0
        if camera_index is None:
            camera_index = self._config.get("camera") or 0
        self._camera_index = camera_index

        # --- Instance guard (actual acquire happens in run()) ---
        self._pid_file = PidFile()

        # --- Database + stats ---
        self._db = Database()
        self._stats = StatsRecorder(self._db)

        # --- Phrase manager ---
        self._phrases = PhraseManager()

        # --- Camera ---
        self._camera = CameraManager(camera_index=self._camera_index)

        # --- Detector + Analyzer ---
        min_conf: float = float(self._config.get("detection", "min_confidence") or 0.5)
        self._detector = PoseDetector(min_confidence=min_conf)
        self._hand_detector = HandDetector(max_hands=2, min_confidence=min_conf)
        self._hair_segmenter = HairSegmenter()
        self._analyzer = GestureAnalyzer(self._config)

        # Derive tick interval from configured FPS
        fps = float(self._config.get("detection", "fps") or 10)
        self._TICK_MS = max(50, int(1000 / fps))

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

        # --- Tray ---
        self._tray = TrayApp(
            on_pause=self._on_pause,
            on_resume=self._on_resume,
            on_exit=self._on_exit,
            stats_getter=self._get_stats_summary,
            config=self._config,
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Acquire PID file, start threads, run tkinter mainloop."""
        if not self._pid_file.acquire():
            print("HairGuard уже запущен. Используйте 'hairguard stop' для остановки.")
            sys.exit(1)

        self._running = True

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
            self._hair_segmenter.close()
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
        """Daemon thread: continuously reads frames and puts them in the queue."""
        fps: float = float(self._config.get("detection", "fps") or 10)
        sleep_sec = 1.0 / fps
        consecutive_failures = 0
        max_failures = 30  # ~6 seconds at 5 FPS

        while self._running:
            frame = self._camera.read_frame()
            if frame is not None:
                consecutive_failures = 0
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
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print("[app] Камера отключена или недоступна. Завершаю работу.")
                    self._on_exit()
                    return

            time.sleep(sleep_sec)

    # ------------------------------------------------------------------
    # Detection tick (main thread, scheduled via root.after)
    # ------------------------------------------------------------------

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
            if event == "pause":
                self._paused = True
                self._hide_overlay_if_visible()
            elif event == "resume":
                self._paused = False
            elif event == "exit":
                self._running = False
                self._root.quit()
                return

        # If paused, skip detection but keep rescheduling
        if self._paused:
            self._root.after(self._TICK_MS, self._tick)
            return

        # Get the latest frame (non-blocking)
        try:
            frame = self._frame_queue.get_nowait()
        except queue.Empty:
            self._root.after(self._TICK_MS, self._tick)
            return

        # --- Detection pipeline ---
        frame_height = frame.shape[0] if hasattr(frame, "shape") else 480
        frame_width = frame.shape[1] if hasattr(frame, "shape") and len(frame.shape) > 1 else 640

        ts_base = int(time.monotonic() * 1000)
        pose_result = self._detector.process(frame, timestamp_ms=ts_base)
        hand_results = self._hand_detector.detect(frame, timestamp_ms=ts_base + 1)
        hair_mask = self._hair_segmenter.segment(frame)
        result = self._analyzer.analyze(pose_result, hand_results, hair_mask, frame_height, frame_width)

        # --- Overlay management ---
        if result.should_alert and not self._overlay.is_visible():
            self._overlay.show(self._phrases.get_next_phrase())
        elif not result.should_alert and self._overlay.is_visible():
            self._overlay.hide()

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

    def _on_exit(self) -> None:
        """Called from tray thread or signal handler."""
        self._tray_events.put("exit")

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
