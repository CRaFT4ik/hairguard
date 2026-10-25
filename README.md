# HairGuard

🇷🇺 [Русская версия](README.ru.md)

Desktop app to help combat trichotillomania using computer vision.

HairGuard watches your webcam in the background and fires a fullscreen alert the moment your fingertips touch your hair zone — before the habit completes. It tracks every episode so you can see your progress over time.

---

## How It Works

Three MediaPipe models run on every frame:

- **PoseLandmarker** — locates head, shoulders, and wrists
- **HandLandmarker** — tracks all 21 hand landmarks, including fingertips
- **FaceLandmarker** — maps the 468-point face mesh to compute the precise forehead/hairline boundary

A 6-layer pipeline filters each frame: hand present → head located → fingertips above hairline → pinch gesture detected → not gripping an object → debounce. An alert fires only when all layers pass simultaneously.

---

## Features

- Fullscreen semi-transparent overlay alert with 200+ unique rotating phrases
- Statistics tracking with daily summaries, trend comparison, and hourly histogram
- System tray icon — pause with auto-resume, quick stats on hover
- Debug view to inspect the detection pipeline live
- Autostart on login (Windows and Linux)
- Single running instance enforced via PID file

---

## Installation

Python 3.11+ required.

```bash
# From PyPI
pip install hairguard

# From source
git clone <repo>
cd hair
pip install -e .

# Linux Wayland — overlay support
pip install hairguard[wayland]
```

---

## Quick Start

```bash
# Start detection with tray icon
python -m hairguard start

# Open debug view (live camera + landmark overlay)
python -m hairguard.debugview
```

---

## CLI Commands

```bash
hairguard start                  # Start detection (foreground + tray)
hairguard start --camera 1       # Use a specific camera index
hairguard start --background     # Start as background process
hairguard stop                   # Stop running instance
hairguard status                 # Show if running, PID, uptime
hairguard cameras                # List available cameras
hairguard collect on             # Start dataset collection (see hairlab/)
hairguard collect off            # Stop it
hairguard collect status         # Is it recording, and how many clips so far
hairguard stats                  # Today's stats vs. yesterday
hairguard stats --last 7d        # Last 7 days
hairguard stats --from 2026-04-01 --to 2026-04-15
hairguard stats --compare --last 2w
hairguard config                 # Print current config
hairguard config --edit          # Open config.yaml in $EDITOR
hairguard autostart on           # Add to login autostart
hairguard autostart off          # Remove from autostart
hairguard debug                  # Launch debug view
```

---

## Configuration

Config file is created at `~/.hairguard/config.yaml` on first run.

```yaml
camera: auto                     # Camera index, or "auto" to detect one
                                 #   (see: hairguard cameras)
camera_width: 1280               # Capture resolution. Detail on the hand is
camera_height: 720               #   what limits dataset quality

detection:
  trigger_delay_sec: 0.8         # Seconds hand must be in hair zone before alert fires
  dismiss_delay_sec: 2.5         # Seconds alert stays after hand leaves
  fps: 10                        # Frames per second processed
  process_width: 1280            # Width the detectors see; lower it to save CPU
  min_confidence: 0.5            # Minimum landmark confidence to trust a detection
  pinch_threshold: 0.4           # Thumb-index distance / hand scale — below = pinching
  min_presence_score: 0.7        # Minimum presence score to trust fingertip position
  hair_zone_offset: 0.4          # How far above forehead to extend hair zone
                                 #   (fraction of shoulder width)

alert:
  opacity: 0.7                   # Overlay opacity (0.0–1.0)
  text_color: "#ffffff"          # Alert text color
  sound: true                    # Repeating chirp while your hand is at your
                                 #   head; stops the moment you move it away

pause:
  auto_resume_minutes: 15        # Tray pause auto-resumes after this many minutes
```

Invalid values are reported at startup; the affected field falls back to its default.

### Camera selection

With `camera: auto`, HairGuard probes each index at startup and picks the first one
showing a live image. Opening a camera successfully is not enough — a device can
return all-black frames, and a virtual camera can serve a single frozen placeholder.
Both look like a working camera to `read()`, so the probe samples several consecutive
frames and keeps only a source that both carries content and changes between frames,
as any real sensor does through its noise. The probe adds a few seconds to startup.

Setting an explicit index skips probing entirely and is always honoured, even if that
camera turns out to be blind.

---

## Detection Pipeline

Each frame passes through 6 layers before an alert is triggered:

| Layer | Check |
|-------|-------|
| 1 | Hand detected with sufficient confidence |
| 2 | Head located (ears + face mesh found) |
| 3 | Hair zone computed from FaceMesh forehead contour |
| 4 | Thumb tip or index fingertip inside the hair zone |
| 5 | Pinch gesture detected (thumb-index distance below threshold) |
| 6 | Debounce — sustained for `trigger_delay_sec` |

The grip filter (originally layer 6 in v2 design) distinguishes twisting from holding an object by checking whether the middle/ring/pinky fingers are curled. If 2 or more are curled, the frame is treated as object-gripping and ignored.

When HandLandmarker loses tracking (e.g., hand behind the head), Pose wrist proximity acts as a supplementary low-confidence signal.

---

## Debug View

```bash
python -m hairguard.debugview
# or
hairguard debug
```

Shows a live camera window with:

- 21-point hand skeleton with connections
- Hair zone boundary polygon
- Pinch indicator and finger curl state per finger
- Fingertip-in-zone highlight
- Layer-by-layer pass/fail status

---

## Statistics

```bash
hairguard stats                        # Today vs. yesterday
hairguard stats --date 2026-04-10      # Specific date
hairguard stats --last 7d              # Last 7 days
hairguard stats --last 4w              # Last 4 weeks
hairguard stats --compare --last 2w    # Compare two consecutive periods
```

Output includes episode count, total time, average duration, and an hourly histogram. Trends are color-coded (green = improvement).

The tray icon tooltip shows a live today summary: "Today: 14 episodes, 5m 20s".

Episodes are stored in `~/.hairguard/hairguard.db` (SQLite).

---

## Platform Support

| Platform | Status |
|----------|--------|
| Windows 10/11 | Full support |
| Linux (X11) | Full support |
| Linux (Wayland) | Supported via `gtk-layer-shell` — install with `pip install hairguard[wayland]` |

On Wayland without `gtk-layer-shell`, the overlay falls back to basic tkinter which may not stay on top. A warning is shown at startup.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| mediapipe | Pose, Hand, and Face landmark detection |
| opencv-python | Camera capture |
| pystray + Pillow | System tray icon |
| pyyaml | Config file |
| colorama | Colored terminal output |
| plyer | System notifications |
| tkinter | Alert overlay (built into Python) |

Optional: `PyGObject` — Wayland overlay backend (`pip install hairguard[wayland]`).
