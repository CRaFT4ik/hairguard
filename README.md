# HairGuard

🇷🇺 [Русская версия](README.ru.md)

A desktop tool for people working to interrupt trichotillomania — compulsive
hair-pulling.

The behaviour is usually automatic. It starts without the person noticing and
is often over before they do. HairGuard watches a webcam in the background,
recognises hair-twisting as it begins, and interrupts with a fullscreen
message and a repeating chime that stops the moment the hand moves away.
Everything runs on the machine; no video leaves it and none is stored unless
you explicitly turn on recording.

It is not a treatment. It is an interruption, meant to be used alongside one.

---

## How detection works

Three MediaPipe models run on each frame — pose, hands and face mesh — giving
the position of the head, the hairline and all 21 landmarks of each hand.

A geometric gate asks the cheap question first: are the fingertips inside the
hair zone above the hairline? Only then does the detector ask the hard one:
is this twisting a strand, or just a hand resting on the head?

That second question is answered by a small temporal convolutional network
trained on labelled examples. It reads a sliding window of hand and head
geometry — normalised into a head-relative frame, so it does not matter how
far away you sit or how your head is tilted — and returns a probability.
A state machine then requires the signal to be sustained before alerting, and
holds the alert briefly after the hand leaves.

Measured on capture days the model never saw during training:

| | catches | false alarms on a resting hand | time to alert |
|---|---|---|---|
| Geometric rule alone | 37.7% | 10.3% | 1.20 s |
| Trained model | **80.7%** | **6.9%** | **0.72 s** |

A trained model ships with the package, in [`hairguard/models/`](hairguard/models),
so detection works on first run. It contains network weights and a threshold —
no video, no images, no personal data of any kind. If you train your own it
takes precedence automatically.

If no model can be loaded at all, the app runs on the geometric rule and says
so at startup, so a fallback is never mistaken for the model performing badly.

### The bundled model is personal to whoever trained it

Those numbers come from one person's data: their hair, their camera, their
habits. Hair-pulling varies enormously between people — scalp, brows, lashes,
one hand or both, twisting or plucking — so treat the bundled model as a
starting point that works out of the box, not as the ceiling.

Training one on your own data is the intended path, and the tooling for it is
included.

---

## Installing

### The application

Download the archive for your system, unpack it, and run `HairGuard`. It puts
an icon in the system tray or menu bar and stays there. No Python, no
terminal, nothing else to install — the trained model and the three MediaPipe
landmark models are inside, so it detects on the first run and works offline.

| | what you get | how to start it |
|---|---|---|
| **Windows** | a folder with `HairGuard.exe` | double-click `HairGuard.exe` |
| **macOS** | `HairGuard.app` | drag to Applications, then open it — [read this first](#macos) |
| **Linux** | a folder with `HairGuard` | run `./install-desktop-entry.sh` once, then launch it from the menu |

To have it start with the machine:

```
hairguard-cli autostart on      # Windows: hairguard-cli.exe autostart on
```

`hairguard-cli` sits in the same folder and is the whole command line. You
never need it to use the app; it is there for when something is wrong.

### From source

Python 3.11+.

```bash
git clone https://github.com/CRaFT4ik/hairguard
cd hairguard
pip install -e .

# To collect data and train a model of your own
pip install -e ".[lab]" -e ".[train]"
```

---

## Quick start

```bash
python -m hairguard start          # detection, with a tray icon
python -m hairguard cameras        # list cameras if autodetect picks wrong
python -m hairguard stats          # episodes today vs yesterday
python -m hairguard doctor         # what is installed, and which detector will run
python -m hairguard stop
```

At startup the app prints which camera it chose and which detector is active.
Everything it prints also goes to `~/.hairguard/hairguard.log`.

### The tray menu

| entry | |
|---|---|
| Пауза | stops detection, resuming by itself after 15 minutes |
| Статистика | today's episodes |
| Детектор | which of the two detectors is deciding right now |
| Сбор данных | start or stop recording clips for training |
| **Диагностика** | opens a window showing what the detector is seeing |
| **Открыть журнал** | opens `~/.hairguard/hairguard.log` |

---

## When something looks wrong

The app has no console, so it writes one instead. Three things to reach for,
in order.

**Open the diagnostic window from the tray.** It draws the hand skeleton, the
hair zone, and every value the detector is deciding on — including the model's
score against its threshold, frame by frame. "What is it actually seeing right
now" is usually the only question worth asking, and this answers it.

It has no camera of its own. The running app already holds the webcam, and on
Windows opening a held camera a second time does not fail — it quietly returns
black frames. So the window is a separate process fed from the detection loop
itself: what it shows is what the detector decided on, and it cannot stall or
crash the alert.

(The `hairguard debug` command is the other way round: it opens its own camera
and runs its own copy of the pipeline, which is right when the app is *not*
running. It refuses to start while the app is, and says why.)

**Ask it what it has.**

```
hairguard-cli doctor
```

Which detector will decide and why, whether the feature extractor the model is
validated against is importable, whether `onnxruntime` is there, where each
MediaPipe model came from, and whether an instance is already running. It never
opens the camera, so it is safe to run while the app is working.

**Read the log.**

```
hairguard-cli log --tail 50
```

`~/.hairguard/hairguard.log`, timestamped, rotated at half a megabyte with one
previous file kept. If the app dies before the tray icon appears, the traceback
is in there and a dialog box says so.

---

## Configuration

`~/.hairguard/config.yaml`, created on first run.

```yaml
camera: auto                     # index, or "auto" to find a working one
camera_width: 1280               # detail on the hand limits model quality
camera_height: 720

detection:
  fps: 10
  trigger_delay_sec: 0.8         # sustained detection before alerting
  dismiss_delay_sec: 2.5         # alert persists this long after the hand leaves
  model:
    enabled: true                # false falls back to the geometric rule
    min_frames: 3                # frames of evidence before the model is asked
    trigger_delay_sec: 0.3

alert:
  opacity: 0.7
  sound: true                    # repeats while the hand is at the head
```

Invalid values are reported at startup and fall back to their defaults.

---

## Building a model for yourself

The `hairlab` package covers the whole path from webcam to a running model.

**1. Collect.** Recording is off until you switch it on, and the state is
announced loudly at every start.

```bash
python -m hairguard collect on
```

Clips of about two seconds are kept whenever fingertips are in the hair zone —
with no judgement about what the hand was doing, since judging it here would
only teach the model what the existing detector already believes. Each clip is
a square crop that follows the hand, plus the landmarks behind it. Roughly
200 KB per clip.

**2. Label.** A browser interface plays each clip as a loop for keyboard
labelling: twisting, resting, scratching, holding an object, or unclear.

```bash
python -m hairlab label
python -m hairlab stats     # progress, and whether the data is usable
```

A few hundred clips per class is a workable minimum. Expect a couple of weeks
of ordinary use to reach that.

**3. Train.**

```bash
python -m hairlab train --cv      # honest estimate, cross-validated by day
python -m hairlab train           # final model, written to ~/.hairguard/trained
```

Two things about the evaluation are worth knowing, because getting them wrong
produces numbers that look excellent and mean nothing.

Splits are by **whole capture day**, never random. Clips seconds apart are
nearly identical, so a random split trains and tests on the same episode.

The reported metric is the **false-alarm rate per kind of negative** —
separately for a resting hand, scratching, and holding an object — not overall
accuracy. Accuracy averages away the one failure that matters: a tool that
interrupts you while you are simply resting your hand on your head will be
turned off, whatever its accuracy says.

The operating point is likewise not chosen to maximise F1. F1 prices a missed
detection and a false alarm equally; for something that interrupts you with a
fullscreen message they are nowhere near equal. The shipped threshold is the
lowest one that keeps false alarms on a resting hand under 5%.

---

## Privacy

- All processing is local. Nothing is uploaded. The downloaded application
  makes no network requests at all — the MediaPipe models are inside it; a
  source install fetches them once on first run.
- The diagnostic window is a child process talking to the app over a pipe,
  not a socket. Nothing listens on a port, and no other program on the
  machine can attach to the video.
- No video is recorded unless you turn collection on.
- Collected clips, labels, episode history and the trained model live in
  `~/.hairguard/` and are never written into the repository.
- Episode statistics are counts and timestamps in a local SQLite file.

---

## Platform support

| Platform | Status |
|----------|--------|
| Windows 10/11 | Full support. The bundle is built and tested here. |
| Linux (X11) | Full support from source. The bundle's build configuration is written but unverified. |
| Linux (Wayland) | Overlay via `gtk-layer-shell` — `pip install hairguard[wayland]` |
| macOS | Autostart and statistics supported; overlay less tested. The `.app` is unverified — see below. |

### macOS

Everything in this section is reasoned from the documentation rather than
observed: the bundles were built on Windows and no part of the macOS path has
been run. Expect to hit at least the first of these.

**Gatekeeper will refuse an unsigned app.** The `.app` is not signed and not
notarised, so a copy downloaded with a browser carries a quarantine attribute
and macOS will say it is damaged and should be moved to the Bin. It is not
damaged; that is the message macOS gives for "unsigned and quarantined". Clear
the attribute:

```bash
xattr -dr com.apple.quarantine /Applications/HairGuard.app
```

Right-click → Open also works on some versions. The real fix is a paid Apple
Developer account, `codesign --deep --options runtime` and `notarytool`, which
is not something this project has.

**Build it on the Mac you will run it on.** An `.app` built on Apple Silicon
will not run on Intel and the reverse is also true: the OpenCV and MediaPipe
wheels carry architecture-specific binaries, and the bundler copies whichever
one it finds. There is no universal build here.

**The camera prompt has to be answered.** The bundle declares
`NSCameraUsageDescription`, without which macOS terminates the process the
instant it touches the webcam — silently, with no prompt. With it, the first
run asks. If you refuse, or if it never asks, the app sees nothing: System
Settings → Privacy & Security → Camera. The log will show the camera failing
to open.

**It stays out of the Dock.** `LSUIElement` is set, so it lives in the menu
bar only — correct for something with no window, but it also means closing it
is a menu-bar action, not ⌘Q.

Accessibility and Screen Recording permissions are *not* needed: the overlay
is an ordinary always-on-top window and nothing reads the screen or the
keyboard.

### Linux

Also unverified — written on Windows and not run. The bundle is a folder;
`install-desktop-entry.sh` inside it writes a `.desktop` entry into
`~/.local/share/applications` pointing at wherever you unpacked it, which is
what makes it appear in the applications menu. Nothing needs root.

Two things to expect. The fullscreen overlay uses X11 window hints and needs
`gtk-layer-shell` under Wayland, which the bundle does not carry — under
Wayland use the source install with `pip install hairguard[wayland]`. And
glibc is not bundled: a folder built on a newer distribution will not start on
an older one, so build on the oldest system you intend to support.

---

## Building the application

```bash
pip install -e ".[package]"
python bundle/build.py
```

The result is in `dist/`, with a zip beside it. It takes about a minute and
produces roughly 300 MB unpacked, 137 MB zipped, of which OpenCV is 118 MB and
onnxruntime 32 MB.

Run it on the platform you are building for; there is no cross-compiling.
Everything that decides what goes in lives in
[`bundle/buildconf.py`](bundle/buildconf.py).

The build refuses rather than producing something subtly wrong. It will not
start without the trained model, because an app that falls back to the
geometric rule starts normally and looks fine while catching half as much. It
will not finish if anything from the training stack got swallowed — torch
alone is 2.8 GB, and the app needs none of it to run what torch produced.
Afterwards it launches the built binary with its home directory pointed at an
empty folder, so it cannot find the developer's own model, and makes it report
which detector it would use and whether it can see everything it needs.

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal use, research,
education, charities and public health organisations.

**Commercial use requires a separate paid agreement.** Contact
eldar.tim@gmail.com.
