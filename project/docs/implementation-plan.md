# HairGuard — Implementation Plan

Based on: `project/docs/design-spec.md`

## Stage 1: Project scaffold + Config + Camera

**Goal:** Runnable project skeleton — can list cameras and load config.

**Files to create:**
- `pyproject.toml` — project metadata, dependencies, `[project.scripts]` entry point
- `hairguard/__init__.py` — version string
- `hairguard/__main__.py` — `python -m hairguard` support
- `hairguard/cli.py` — argparse skeleton with all subcommands (start, stop, status, cameras, stats, config, autostart). Each subcommand dispatches to its module. Unimplemented commands print "Not yet implemented".
- `hairguard/config.py` — load/save/validate `~/.hairguard/config.yaml`, create defaults on first run
- `hairguard/camera.py` — `CameraManager`: enumerate cameras via OpenCV (`cv2.VideoCapture` probe), select by index, capture frames, release

**Verification:**
- `pip install -e .` succeeds
- `hairguard cameras` lists connected cameras
- `hairguard config` prints default config
- Config validation rejects bad values gracefully

---

## Stage 2: Detection pipeline

**Goal:** PoseDetector + GestureAnalyzer detect "hand at head" from camera frames.

**Files to create:**
- `hairguard/detector.py` — `PoseDetector`: wraps `mediapipe.solutions.pose`, processes a frame, returns landmarks with confidence scores. Configurable `min_confidence`.
- `hairguard/analyzer.py` — `GestureAnalyzer`:
  - Computes distance(wrist, nearest_ear) / distance(left_shoulder, right_shoulder)
  - Shoulder occlusion fallback: if either shoulder confidence < threshold, use absolute distance as % of frame height (`fallback_frame_percent` from config)
  - Debounce logic: tracks state machine (idle -> triggered -> alerting -> cooldown)
  - `trigger_delay_sec` before alerting
  - **GestureAnalyzer owns ALL timing logic** including `dismiss_delay_sec`. It returns `should_alert: bool` which already accounts for dismiss delay. The app layer simply shows/hides overlay based on this flag — no additional timing logic in the main loop.
  - Returns: `AnalysisResult` with `should_alert: bool`, `hand_detected_sec: float`, `episode_started: bool`, `episode_ended: bool`, `episode_duration: float`

**Verification:**
- Unit test: GestureAnalyzer state machine with mock landmarks — all transitions: idle -> triggered -> alerting -> cooldown -> idle
- Unit test: shoulder occlusion fallback logic (one shoulder missing)
- Unit test: dismiss delay — `should_alert` stays True for `dismiss_delay_sec` after hand leaves
- Manual test: run detector on live camera, print "ALERT" to console when hand at head

---

## Stage 3: Alert overlay + Phrases (can run in parallel with Stage 4)

**Goal:** Fullscreen semi-transparent warning appears when hand detected.

**Files to create:**
- `hairguard/phrases.py` — list of 200+ humorous/motivational phrases in Russian (user's primary language). Shuffle manager with persistence to `~/.hairguard/phrase_state.json`. `get_next_phrase()` returns next unseen phrase, reshuffles when all exhausted.
- `hairguard/overlay.py` — `AlertOverlay`:
  - Factory function `create_overlay()` that returns tkinter or GTK backend based on platform
  - **TkOverlay:** frameless, fullscreen, semi-transparent, topmost. Platform-specific focus handling:
    - Windows: `ctypes` + `WS_EX_NOACTIVATE` + `WS_EX_TOPMOST` on HWND via `SetWindowLong`
    - Linux X11: `overrideredirect` + `_NET_WM_WINDOW_TYPE_NOTIFICATION`
  - **GtkOverlay** (optional, Wayland): `gtk-layer-shell` with `LAYER_OVERLAY`. Graceful fallback to TkOverlay if `gtk-layer-shell` not installed (with warning at startup).
  - Methods: `show(phrase)`, `hide()`, `is_visible()`
  - Text: large, centered, white on dark semi-transparent background

**Verification:**
- Manual test (Windows): overlay appears/disappears, does not steal focus from active window
- Manual test (Linux X11, if available): overlay appears on top, no focus steal
- Manual test (Wayland, if available): gtk-layer-shell overlay works; fallback warning shown if not installed
- Phrase shuffler returns all 200+ phrases before repeating
- Phrase state survives app restart (kill and restart, check next phrase is not a repeat)

---

## Stage 4: Statistics (can run in parallel with Stage 3)

**Goal:** Record episodes to SQLite, query via CLI.

**Files to create:**
- `hairguard/db.py` — SQLite connection manager. Creates `~/.hairguard/hairguard.db` on first run. Schema versioning via `schema_version` table. Migration runner that applies missing migrations sequentially.
- `hairguard/stats.py` — two classes:
  - `StatsRecorder`: `start_episode()`, `end_episode(duration)` — writes to SQLite
  - `StatsCLI`: handles all `--date`, `--from/--to`, `--last`, `--hour`, `--compare` flags. Outputs:
    - Episode count, total time, average duration
    - Hourly histogram (Unicode bar chars)
    - Trend comparison (colorama: green/red)
    - Default (no args): today + comparison with yesterday

**Verification:**
- Unit test: StatsRecorder writes/reads episodes correctly
- Unit test: StatsCLI date parsing — `today`, `yesterday`, `7d`, `2w`, `1m`, date ranges
- Unit test: period comparison and trend calculation
- Unit test: migration runner — simulate v1->v2 migration (add a dummy column, verify it applies)
- Manual test: `hairguard stats` with sample data shows formatted colored output

---

## Stage 5: Tray + CLI commands + Autostart

**Goal:** System tray icon, all CLI commands working, autostart.

**Depends on:** Stages 3 and 4 complete.

**Files to create:**
- `hairguard/tray.py` — `TrayApp`: runs pystray in daemon thread. Menu items:
  - Pause/Resume (with 15-min auto-resume timer, shows remaining time)
  - Detailed stats (triggers system notification via `plyer`)
  - Exit (signals main thread to shut down via callback)
  - Tooltip: "Today: N episodes, Xm Ys"
- `hairguard/pid.py` — PID file management (`~/.hairguard/hairguard.pid`): write on start, read on stop/status, stale detection (check if PID process exists), instance guard
- `hairguard/autostart.py` — `autostart on/off`:
  - Windows: create/remove shortcut in `shell:startup` via direct COM (`win32com.client` from `pywin32`) — no extra dependency beyond what's commonly available, or fall back to creating a `.bat` file
  - Linux: create/remove `.desktop` file in `~/.config/autostart/`
- Update `hairguard/cli.py` — wire all subcommands: `start`, `stop`, `status`, `cameras`, `stats`, `config`, `config --edit` (opens `$EDITOR`, falls back to `notepad` on Windows / `nano` on Linux), `autostart`

**Verification:**
- Tray icon appears, pause/resume works, tooltip shows stats
- `plyer` notification shows stats summary when "Detailed stats" clicked
- `hairguard stop` terminates running instance, PID file removed
- `hairguard status` shows running/stopped state correctly
- Instance guard: second `hairguard start` prints error and exits
- `hairguard autostart on` creates appropriate file per platform
- `hairguard start --camera 1` passes camera index to CameraManager
- `hairguard config --edit` opens editor

---

## Stage 6: Integration — Main loop + Wiring

**Goal:** All components wired together into a working application.

**Depends on:** Stage 5 complete (all components ready).

**File to create/modify:**
- `hairguard/app.py` — `HairGuardApp` main class:
  - Sets up tkinter root window (hidden, for event loop)
  - Creates AlertOverlay via factory
  - Starts camera thread (captures frames into `queue.Queue`)
  - Starts tray thread (pystray in daemon thread, communicates via queue/events)
  - Writes PID file, checks instance guard
  - Main loop via `root.after(200, tick)` (5 FPS = 200ms):
    1. Read frame from queue (non-blocking)
    2. Run PoseDetector on frame
    3. Run GestureAnalyzer (returns `AnalysisResult`)
    4. If `should_alert` and overlay not visible: `overlay.show(get_next_phrase())`
    5. If not `should_alert` and overlay visible: `overlay.hide()`
    6. If `episode_ended`: `stats_recorder.end_episode(duration)`
    7. Check tray events (pause/resume/exit)
  - Graceful shutdown: SIGTERM/SIGINT handler, releases camera, saves phrase state, deletes PID file
- Update `hairguard/cli.py` — `start` command creates and runs `HairGuardApp`

**Note:** The app layer does NOT implement any timing/delay logic. GestureAnalyzer's `should_alert` already accounts for trigger_delay and dismiss_delay. The app simply maps `should_alert` to show/hide.

**Verification:**
- Full end-to-end: start app, raise hand to head, overlay appears after 1.5s with phrase, remove hand, overlay disappears after 2.5s
- Episode recorded in DB with correct duration
- Tray pause stops detection, overlay hidden, auto-resumes after 15 min
- `hairguard stop` from another terminal cleanly shuts down (no zombie threads, camera released)
- Multiple start/stop cycles — no resource leaks
- `hairguard start --camera 1` uses correct camera
- No thread deadlocks on exit (verify with timeout)

---

## Review checkpoints

- **After Stage 1:** Independent review — project structure, config validation, camera enumeration
- **After Stage 2:** Independent review — detection accuracy, state machine correctness, dismiss delay ownership
- **After Stage 3:** Independent review — overlay cross-platform behavior, phrase system, focus handling
- **After Stage 4:** Independent review — SQL correctness, migration runner, stats output formatting
- **After Stage 5:** Independent review — tray threading safety, PID management, CLI completeness, plyer notifications
- **After Stage 6:** Final comprehensive review — full integration, edge cases, cross-platform, no timing duplication

## Parallelism note

Stages 3 and 4 have no shared dependencies (both depend only on Stage 1 output). They can be implemented by parallel agents to save time.
