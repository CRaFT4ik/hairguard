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

## Installation

Python 3.11+ required.

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
python -m hairguard stop
```

At startup the app prints which camera it chose and which detector is active.

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

- All processing is local. Nothing is uploaded, and the app makes no network
  requests except to download the MediaPipe models on first run.
- No video is recorded unless you turn collection on.
- Collected clips, labels, episode history and the trained model live in
  `~/.hairguard/` and are never written into the repository.
- Episode statistics are counts and timestamps in a local SQLite file.

---

## Platform support

| Platform | Status |
|----------|--------|
| Windows 10/11 | Full support |
| Linux (X11) | Full support |
| Linux (Wayland) | Overlay via `gtk-layer-shell` — `pip install hairguard[wayland]` |
| macOS | Autostart and statistics supported; overlay less tested |

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal use, research,
education, charities and public health organisations.

**Commercial use requires a separate paid agreement.** Contact
eldar.tim@gmail.com.
