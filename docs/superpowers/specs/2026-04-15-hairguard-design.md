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

### Data flow

```
Camera -> CameraManager -> PoseDetector -> GestureAnalyzer -> AlertOverlay
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

> **IMPORTANT — Fallback consideration:** If during testing MediaPipe Pose proves insufficient (e.g., poor accuracy with partial torso visibility), the architecture supports swapping to MediaPipe Hands + Face detection (Approach B). The GestureAnalyzer interface stays the same — only the internal detection logic changes. This should be evaluated during initial testing phase.

> **Note on user posture:** The user frequently slouches/slides down in their chair. The relative distance approach (normalizing by shoulder width) handles this. However, this should be explicitly tested during development — capture frames in various slouched positions and verify detection accuracy.

### Debounce

- **Trigger delay:** Alert appears only after hand is at head for N seconds (default 1.5s) — prevents false positives from scratching nose, adjusting glasses
- **Dismiss delay:** Alert disappears 2.5 seconds after hand leaves head — gives time for conscious awareness

### Confidence filter

MediaPipe returns confidence per landmark. Frames where key landmarks (wrists, ears, shoulders) have confidence < 0.5 are ignored to avoid false positives.

### Frame rate

5 FPS — sufficient for detection, minimal CPU load (~2-4% of one core).

## Alert Overlay

- Frameless fullscreen window via tkinter, semi-transparent (~70% opacity)
- Always on top, does NOT capture focus (no accidental clicks on underlying apps)
- Large centered text with randomized phrases
- **200+ unique phrases** — at 30 touches/day, no repeats for ~7 days. Phrases cycle randomly without repeats until all are exhausted, then restart. Phrases should be humorous/surprising to maintain engagement. Content is intentionally not listed in this spec — the user wants to be surprised.
- Configurable: opacity, text color

### Cross-platform

- **Windows:** tkinter `topmost` + `alpha` attribute for transparency
- **Linux (X11):** tkinter + `_NET_WM_STATE_ABOVE` — works reliably
- **Linux (Wayland):** Known limitation — fullscreen overlay may require compositor-specific handling. Needs testing; may need fallback approach

## Statistics

### Storage

SQLite database at `~/.hairguard/hairguard.db`, table `episodes`:
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `started_at` TEXT (ISO 8601 timestamp)
- `duration_sec` REAL

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
- **Menu item "Detailed stats":** Opens terminal with full `hairguard stats` output

## System Tray

Via `pystray` + `Pillow`:

- **Pause / Resume** — pauses detection. Auto-resumes after 15 minutes. Menu shows remaining time: "Paused (12 min left)"
- **Stats tooltip** — today's summary on hover
- **Detailed stats** — opens terminal with `hairguard stats`
- **Exit**

## CLI Commands

```bash
hairguard start              # Start detection (foreground)
hairguard start --camera 1   # Start with specific camera
hairguard cameras            # List available cameras
hairguard stats              # Show statistics (see Statistics section)
hairguard config             # Show current config
hairguard config --edit      # Open config.yaml in system editor
hairguard autostart on       # Add to system autostart
hairguard autostart off      # Remove from autostart
```

## Autostart

```bash
hairguard autostart on / off
```

- **Windows:** Creates/removes shortcut in `shell:startup`
- **Linux:** Creates/removes `.desktop` file in `~/.config/autostart/`

## Configuration

File: `~/.hairguard/config.yaml` (created on first run with defaults)

```yaml
camera: 0
detection:
  distance_threshold: 0.6
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
│   ├── overlay.py           # AlertOverlay (tkinter)
│   ├── phrases.py           # 200+ phrases
│   ├── stats.py             # StatsRecorder + StatsCLI
│   ├── tray.py              # TrayApp (pystray)
│   ├── config.py            # config.yaml load/save
│   └── autostart.py         # autostart management
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
- `tkinter` — overlay (built into Python)

## Installation

```bash
pip install hairguard
# or from repo
pip install -e .
```
