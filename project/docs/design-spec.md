# HairGuard — Design Spec

## Purpose

Desktop application for combating trichotillomania. Uses computer vision to detect when the user's hand is near their head (hair-pulling/twisting gesture) and displays a full-screen overlay warning. Tracks episodes over time so the user can monitor recovery progress.

Target platforms: Windows, Linux.

## Architecture

**Tech stack:** Python 3.11+, MediaPipe Pose, OpenCV, pystray, tkinter, SQLite

### Components

| Component | Responsibility |
|---|---|
| **CameraManager** | Enumerate connected cameras, let user select, capture frames |
| **PoseDetector** | Wraps MediaPipe Pose — receives frame, returns 33 skeleton landmarks |
| **GestureAnalyzer** | Computes relative wrist-to-ear distance normalized by shoulder width. Decides "hand at head" or not. Handles debounce timing |
| **AlertOverlay** | Frameless fullscreen semi-transparent tkinter window on top of all windows. Shows randomized phrases |
| **StatsRecorder** | Writes episodes to SQLite (start time, duration) |
| **StatsCLI** | Flexible CLI for querying statistics with date ranges, comparisons, colored output |
| **TrayApp** | System tray icon via pystray: pause (with auto-resume), quick stats tooltip, detailed stats, exit |
| **Config** | Loads/saves config.yaml with all tunable parameters |

### Threading model

The application uses three threads:

1. **Main thread** — runs `tkinter` mainloop (required by tkinter on all platforms). The AlertOverlay lives here. The main thread schedules periodic callbacks via `root.after()` to drive the detection loop.
2. **Tray thread** — `pystray` runs in a daemon thread. Tray menu actions post events to tkinter's main thread via `root.event_generate()` or a thread-safe `queue.Queue`.
3. **Camera thread** — captures frames in a daemon thread, puts them into a `queue.Queue`. The detection callback on the main thread reads from this queue.

This avoids the tkinter + pystray main-thread conflict: tkinter owns the main thread, pystray runs in a background thread.

### Data flow

```
Camera thread -> Queue -> Main thread (PoseDetector -> GestureAnalyzer -> AlertOverlay)
                                              |
                                        StatsRecorder -> SQLite
```

## Detection Logic

### Approach: MediaPipe Pose (skeleton-based)

MediaPipe Pose provides 33 body landmarks. We use:
- **Wrists** (left_wrist, right_wrist)
- **Ears** (left_ear, right_ear) — proxy for head area
- **Shoulders** (left_shoulder, right_shoulder) — for computing base scale

**Relative threshold:** distance(wrist, ear) / distance(left_shoulder, right_shoulder). If ratio < threshold (default 0.6) — hand is at head. Normalization by shoulder width ensures detection works regardless of distance from camera or posture (including slouching in chair).

**Shoulder occlusion handling:** Both shoulders must have confidence >= `min_confidence` for the ratio calculation. If either shoulder is below confidence threshold, fall back to absolute pixel distance between wrist and nearest ear, compared to a percentage of frame height (default 15%). This prevents division by zero and handles partial body visibility.

> **IMPORTANT — Fallback consideration:** If during testing MediaPipe Pose proves insufficient (e.g., poor accuracy with partial torso visibility), the architecture supports swapping to MediaPipe Hands + Face detection (Approach B). The GestureAnalyzer interface stays the same — only the internal detection logic changes. This should be evaluated during initial testing phase.

> **Note on user posture:** The user frequently slouches/slides down in their chair. The relative distance approach (normalizing by shoulder width) handles this. However, this should be explicitly tested during development — capture frames in various slouched positions and verify detection accuracy.

### Debounce

- **Trigger delay:** Alert appears only after hand is at head for N seconds (default 1.5s) — prevents false positives from scratching nose, adjusting glasses
- **Dismiss delay:** Alert disappears 2.5 seconds after hand leaves head — gives time for conscious awareness

### Confidence filter

MediaPipe returns confidence per landmark. Frames where key landmarks (wrists, ears) have confidence < 0.5 are ignored to avoid false positives. Shoulder confidence is handled separately (see shoulder occlusion above).

### Frame rate

5 FPS — sufficient for detection, minimal CPU load (~2-4% of one core).

## Alert Overlay

- Frameless fullscreen window via tkinter, semi-transparent (~70% opacity)
- Always on top, does NOT capture focus (no accidental clicks on underlying apps)
- Large centered text with randomized phrases
- **200+ unique phrases** — at 30 touches/day, no repeats for ~7 days. Phrases cycle randomly without repeats until all are exhausted, then restart. Phrases should be humorous/surprising to maintain engagement. Content is intentionally not listed in this spec — the user wants to be surprised.
- **Phrase cycle persistence:** Current position in the phrase shuffle is stored in `~/.hairguard/phrase_state.json` (shuffled indices list + current index). Survives restarts.
- Configurable: opacity, text color

### Focus behavior (platform-specific)

The overlay must appear on top without stealing keyboard/mouse focus from the active application:

- **Windows:** Use `ctypes` to call `SetWindowLong` with `WS_EX_NOACTIVATE | WS_EX_TOPMOST` on the tkinter window's HWND (obtained via `winfo_id()`). Additionally use `overrideredirect(True)` and `attributes('-topmost', True)`.
- **Linux (X11):** Use `overrideredirect(True)` combined with `_NET_WM_WINDOW_TYPE_NOTIFICATION` via `xprop` or python-xlib. This tells the window manager to treat it as a notification — always on top, no focus.
- **Linux (Wayland):** Use `gtk-layer-shell` via PyGObject as the overlay backend instead of tkinter. `gtk-layer-shell` provides `LAYER_OVERLAY` which is the Wayland-native way to create always-on-top overlays without focus. The AlertOverlay component has a factory method that selects the tkinter or GTK backend based on `$XDG_SESSION_TYPE`.

### Wayland support

Wayland does not allow applications to position themselves on top or set always-on-top. The solution:

- Detect Wayland via `$XDG_SESSION_TYPE == "wayland"`
- Use `gtk-layer-shell` (available as `python3-gi` + `libgtk-layer-shell`) for the overlay
- `gtk-layer-shell` provides `LAYER_OVERLAY` — designed exactly for this use case
- Fallback: if `gtk-layer-shell` is not installed, show a warning at startup with install instructions and fall back to basic tkinter (which may not stay on top on Wayland)
- Add `PyGObject` as an optional dependency (`pip install hairguard[wayland]`)

## Statistics

### Storage

SQLite database at `~/.hairguard/hairguard.db`.

**Table `episodes`:**
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `started_at` TEXT (ISO 8601 timestamp)
- `duration_sec` REAL

**Table `schema_version`:**
- `version` INTEGER — current schema version (starts at 1)

On startup, check `schema_version` and run migrations if needed. This allows adding columns/tables in future versions without breaking existing installs.

### CLI Interface

```bash
# Default — today + comparison with yesterday
hairguard stats

# Specific date
hairguard stats --date 2026-04-15

# Date range
hairguard stats --from 2026-04-01 --to 2026-04-15

# Last N days/weeks
hairguard stats --last 7d
hairguard stats --last 4w

# Specific hour on a date
hairguard stats --date today --hour 14

# Compare two periods
hairguard stats --compare --last 2w
```

### Output includes

- Episode count
- Total time
- Average episode duration
- Hourly histogram (bar chart in terminal)
- Trend comparison (green = improvement, red = worse) via colorama

### Quick stats via tray

- **Tooltip on hover:** "Today: 14 episodes, 5m 20s"
- **Menu item "Detailed stats":** Runs `hairguard stats` and shows output in a system notification (multi-line) via `plyer` library. This avoids the complexity of spawning platform-specific terminals.

## System Tray

Via `pystray` + `Pillow`:

- **Pause / Resume** — pauses detection. Auto-resumes after 15 minutes. Menu shows remaining time: "Paused (12 min left)"
- **Stats tooltip** — today's summary on hover
- **Detailed stats** — shows stats as system notification
- **Exit**

## CLI Commands

```bash
hairguard start              # Start detection (with tray icon)
hairguard start --camera 1   # Start with specific camera
hairguard start --background # Start as background process (writes PID to ~/.hairguard/hairguard.pid)
hairguard stop               # Stop running instance (reads PID file, sends SIGTERM/TerminateProcess)
hairguard status             # Show if running, PID, uptime
hairguard cameras            # List available cameras
hairguard stats              # Show statistics (see Statistics section)
hairguard config             # Show current config
hairguard config --edit      # Open config.yaml in $EDITOR (falls back to notepad/nano)
hairguard autostart on       # Add to system autostart
hairguard autostart off      # Remove from autostart
```

### Instance guard

On `hairguard start`, check if `~/.hairguard/hairguard.pid` exists and if the process is alive. If already running, print message and exit. PID file is deleted on clean shutdown and stale PID files (process no longer exists) are ignored.

## Autostart

```bash
hairguard autostart on / off
```

- **Windows:** Creates/removes shortcut in `shell:startup` (runs `hairguard start --background`)
- **Linux:** Creates/removes `.desktop` file in `~/.config/autostart/`

## Configuration

File: `~/.hairguard/config.yaml` (created on first run with defaults)

```yaml
camera: 0
detection:
  distance_threshold: 0.6
  fallback_frame_percent: 0.15
  trigger_delay_sec: 1.5
  dismiss_delay_sec: 2.5
  fps: 5
  min_confidence: 0.5
alert:
  opacity: 0.7
  text_color: "#ffffff"
pause:
  auto_resume_minutes: 15
```

### Config validation

On load, validate all values against expected types and ranges. Invalid config prints a clear error message with the problematic field and falls back to defaults for that field.

## Project Structure

```
hairguard/
├── hairguard/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── cli.py               # argparse commands
│   ├── camera.py            # CameraManager
│   ├── detector.py          # PoseDetector (MediaPipe)
│   ├── analyzer.py          # GestureAnalyzer (threshold logic)
│   ├── overlay.py           # AlertOverlay (tkinter + Wayland factory)
│   ├── phrases.py           # 200+ phrases + shuffle state
│   ├── stats.py             # StatsRecorder + StatsCLI
│   ├── tray.py              # TrayApp (pystray)
│   ├── config.py            # config.yaml load/save/validate
│   ├── app.py               # HairGuardApp — main loop, wires all components
│   ├── autostart.py         # autostart management
│   ├── pid.py               # PID file management + instance guard
│   └── db.py                # SQLite connection + schema migrations
├── tests/
├── pyproject.toml
└── README.md
```

## Dependencies

- `mediapipe` — pose detection
- `opencv-python` — camera capture
- `pystray` + `Pillow` — system tray icon
- `pyyaml` — config file
- `colorama` — colored terminal output (Windows-compatible)
- `plyer` — cross-platform system notifications
- `tkinter` — overlay (built into Python)

### Optional (Linux Wayland)

- `PyGObject` + `gtk-layer-shell` — Wayland overlay support (`pip install hairguard[wayland]`)

## Installation

```bash
pip install hairguard
# or from repo
pip install -e .

# With Wayland support (Linux)
pip install hairguard[wayland]
```
