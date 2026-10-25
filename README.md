# HairGuard

[Русская версия](README.ru.md)

HairGuard catches the moment you start twisting your hair and interrupts
you: a message covers the screen and a sound plays until your hand moves
away. It is for people dealing with trichotillomania, where the hand usually
reaches for the head on its own, without the person noticing.

It lives in the system tray, watches your webcam, and does all the work on
your machine. Nothing is uploaded, and no video is saved unless you turn
recording on yourself.

This is not treatment. It is a way to notice the moment.

## Getting started

Download the archive, unpack it, run it. No Python, no install step: every
model it needs is already inside.

| System | How to start it |
|--------|-----------------|
| Windows | double-click `HairGuard.exe` |
| macOS | drag `HairGuard.app` to Applications, [read the Gatekeeper note first](#macos) |
| Linux | run `./install-desktop-entry.sh` once, then launch from your menu |

An icon appears in the tray (the menu bar on macOS). That is all you need to
do.

To start it with your system:

```
hairguard-cli autostart on
```

`hairguard-cli` sits in the same folder. You do not need it for normal use,
only when something goes wrong.

## The tray menu

| Item | What it does |
|------|--------------|
| Pause | stops alerts, resumes itself after 15 minutes |
| Statistics | episodes so far today |
| Detector | which detector is deciding right now |
| Collect data | record clips for training your own model |
| Diagnostics | a window showing what the detector sees |
| Open log | `~/.hairguard/hairguard.log` |

## When it does not work

The app has no console, so it keeps a log. Three steps, in order.

**Open Diagnostics from the tray.** It draws the hand skeleton, the hair
zone, and every number the decision rests on, frame by frame. There is
usually only one question worth asking, which is what the thing is actually
seeing. This answers it.

The window has no camera of its own; the app sends it frames. That is
deliberate. The running app holds the camera, and on Windows a second open
of a busy camera does not fail, it quietly returns black frames.

**Ask what it has.**

```
hairguard-cli doctor
```

Tells you which detector will run and why, whether the models are in place,
and whether an instance is already running. It never opens the camera, so it
is safe to run while the app is working.

**Read the log.**

```
hairguard-cli log --tail 50
```

If the app dies before the tray icon appears, the traceback is in there.

## Configuration

`~/.hairguard/config.yaml`, created on first run.

```yaml
camera: auto                     # camera index, or "auto"
camera_width: 1280
camera_height: 720

detection:
  fps: 10
  trigger_delay_sec: 0.8         # how long detection must hold before alerting
  dismiss_delay_sec: 2.5         # how long the alert stays after the hand leaves
  model:
    enabled: true                # false falls back to the geometric rule
    min_frames: 3                # frames to gather before asking the model
    trigger_delay_sec: 0.3

alert:
  opacity: 0.7
  sound: true                    # repeats while the hand is at the head
```

Bad values are replaced with defaults, and it says so at startup.

## How it works

Three MediaPipe models run on every frame and give pose, hands, and the face
mesh. Then come two questions.

The cheap one: are the fingertips above the hairline? If not, nothing else
runs.

The hard one: is this twisting a strand, or just a hand resting on a head?
A small neural network trained on labelled examples answers that. It looks
at how the hand moved over the last two seconds, expressed relative to the
head, so how far you sit from the camera and how you tilt your head do not
matter.

Measured on capture days the model never saw during training:

| | Catches | False alarms | Time to alert |
|---|---|---|---|
| Geometric rule | 37.7% | 10.3% | 1.20 s |
| Trained model | 80.7% | 6.9% | 0.72 s |

The model ships in [`hairguard/models/`](hairguard/models) and works
immediately. It contains weights and a threshold, no video and no personal
data. If it fails to load, the app runs on the geometric rule and says so at
startup, so a fallback cannot be mistaken for the model doing badly.

One caveat worth stating plainly. Those numbers come from one person's data:
his hair, his camera, his habits. Trichotillomania looks different from
person to person: scalp, eyebrows, eyelashes, one hand or both. Treat the
bundled model as a starting point rather than a ceiling, and train your own.

## Training your own model

**1. Collect clips.** Recording stays off until you turn it on.

```
python -m hairguard collect on
```

Two-second clips are saved whenever fingertips enter the hair zone, with no
judgement about what the hand was doing. Judging at this stage would teach
the new model whatever the old one already believes. About 200 KB per clip.

**2. Label them.** A browser interface loops each clip; you label from the
keyboard: twisting, resting, scratching, object in hand, unclear.

```
python -m hairlab label
python -m hairlab stats
```

A few hundred clips per class is the working minimum. Expect about two weeks
of ordinary use to gather them.

**3. Train.**

```
python -m hairlab train --cv      # honest evaluation
python -m hairlab train           # the model you keep
```

Two things are easy to get wrong here, and getting them wrong produces
beautiful numbers that mean nothing.

Data is split **by whole capture days**, never at random. Clips recorded
seconds apart are nearly identical, so a random split trains and tests you
on the same episode.

The metric that matters is the **false alarm rate per type of negative**:
resting, scratching, object in hand, each counted separately. Not overall
accuracy. Accuracy averages away the exact mistake that gets the app
uninstalled, which is interrupting you when you simply leaned on your hand.

The threshold is not chosen by maximising F1 either. F1 treats a miss and a
false alarm as equally bad, and they are nowhere near equal when the thing
covers your screen. It picks the lowest threshold that keeps false alarms on
a resting hand under 5%.

## Privacy

Everything runs locally and nothing is uploaded. The downloaded app does not
use the network at all; the models are inside it.

No video is recorded until you turn collection on. Clips, labels, episode
history, and your trained model live in `~/.hairguard/` and never reach the
repository.

The diagnostics window is a child process that receives frames through a
pipe rather than a socket. No port is listening, and nothing else on the
machine can connect to your video.

## From source

Python 3.11 or newer.

```bash
git clone https://github.com/CRaFT4ik/hairguard
cd hairguard
pip install -e .

# if you plan to train your own model
pip install -e ".[lab]" -e ".[train]"

python -m hairguard start
python -m hairguard cameras        # list cameras if auto-detection picked wrong
python -m hairguard stats
python -m hairguard doctor
python -m hairguard stop
```

## Platforms

| Platform | State |
|----------|-------|
| Windows 10/11 | Full support, bundle built and tested |
| Linux (X11) | Works from source, bundle config written but untested |
| Linux (Wayland) | Needs `gtk-layer-shell`: `pip install hairguard[wayland]` |
| macOS | Works from source, `.app` untested |

### macOS

Everything in this section is derived from documentation, not tested on a
real machine: the bundles were built on Windows.

**Gatekeeper will refuse to open it.** The app is unsigned, and macOS will
claim it is damaged and should go to the trash. It is not damaged; that is
how macOS reports quarantine. Clear the flag:

```bash
xattr -dr com.apple.quarantine /Applications/HairGuard.app
```

Right-click and Open sometimes works too. The proper fix is a paid Apple
Developer account and a signature, which this project does not have.

**Build on the same Mac you will run it on.** An Apple Silicon build will
not run on Intel or the other way round, and there is no universal build.

**Allow camera access** when asked on first launch. If you decline, the app
sees nothing, and the log will show the camera failing to open. Fix it in
System Settings, Privacy and Security, Camera.

There is no Dock icon; the app lives in the menu bar only. Screen recording
and accessibility permissions are not needed.

### Linux

Also untested. The bundle folder contains `install-desktop-entry.sh`, which
adds the app to your menu. No root required.

On Wayland the fullscreen window needs `gtk-layer-shell`, which is not in
the bundle, so install from source there. glibc is not bundled either: a
folder built on a recent distribution will not start on an old one, so build
on the oldest system you want to support.

## Building

```bash
pip install -e ".[package]"
python bundle/build.py
```

The result lands in `dist/` next to a zip. It takes about a minute and
produces roughly 300 MB unpacked, 137 MB zipped, of which OpenCV is 118 MB.

The build refuses to run rather than producing something quietly wrong. It
will not start without a trained model, because an app without one starts
fine and looks fine while catching half as much. It will not finish if the
training stack got bundled: torch alone is 2.8 GB, and running a finished
model does not need it. At the end it launches the binary with an empty home
directory, so it cannot find the developer's model, and asks it what it
would choose.

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Free for personal use, research,
education, charities, and healthcare organisations.

Commercial use requires a separate paid agreement: eldar.tim@gmail.com
