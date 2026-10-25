"""hairguard.sound — the audible part of an alert.

The overlay only works if the user is looking at the screen, and the whole
point of the alert is to interrupt someone who is absorbed in something
else. A sound reaches them anyway.

Two constraints shape this module. Playback blocks for the length of the
sound, and the caller is the detection loop that owns the tkinter event
loop and the alert window — so it plays on a worker thread. And a machine
with no audio device must fall silent, never fail: losing the beep is a
small thing, losing the alert is not.
"""
from __future__ import annotations

import functools
import io
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable, Optional

SAMPLE_RATE = 44100

# Two notes a fourth apart, the shape of a doorbell: rising reads as a
# summons rather than an error, and two notes are recognisable as "the app"
# instead of "a system noise".
_NOTES = ((784.0, 0.0), (1046.5, 0.16))   # G5, C6 — second note starts later
_NOTE_LENGTH = 0.34
# Ends exactly where the second note does: trailing silence would only delay
# the next repeat.
_TOTAL_LENGTH = _NOTES[-1][1] + _NOTE_LENGTH

# A tone reads as harsh when it is a bare square wave and as a click when it
# starts or stops abruptly. Harmonics give it a bell-like body; the envelope
# makes both ends start and finish at exactly zero.
_HARMONICS = ((1.0, 1.00), (2.0, 0.28), (3.01, 0.12), (4.98, 0.05))
_ATTACK_SEC = 0.012
_DECAY_RATE = 5.5
_FADE_OUT_SEC = 0.05
_PEAK = 0.62          # of full scale, leaving headroom so nothing clips


def _note(frequency: float, start: float, length: float, total: int) -> "np.ndarray":
    """One bell-like note placed into a buffer of ``total`` samples."""
    import numpy as np

    buffer = np.zeros(total, dtype=np.float64)
    begin = int(start * SAMPLE_RATE)
    count = min(int(length * SAMPLE_RATE), total - begin)
    if count <= 0:
        return buffer

    t = np.arange(count) / SAMPLE_RATE

    wave_form = np.zeros(count, dtype=np.float64)
    for ratio, weight in _HARMONICS:
        wave_form += weight * np.sin(2 * np.pi * frequency * ratio * t)
    wave_form /= sum(weight for _, weight in _HARMONICS)

    # Raised-cosine attack: no step at the start, so no pop.
    envelope = np.exp(-_DECAY_RATE * t)
    attack = int(_ATTACK_SEC * SAMPLE_RATE)
    if attack > 1:
        rise = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, attack))
        envelope[:attack] *= rise

    buffer[begin:begin + count] = wave_form * envelope
    return buffer


def synthesize_chime() -> bytes:
    """Build the alert chime as an in-memory 16-bit mono WAV.

    Generated rather than shipped as a file so there is no asset to lose,
    and generated rather than beeped because ``winsound.Beep`` produces a
    square wave with no envelope — harsh to listen to, and audibly clicking
    at both ends where the waveform steps away from and back to silence.
    """
    import numpy as np

    total = int(_TOTAL_LENGTH * SAMPLE_RATE)
    signal = np.zeros(total, dtype=np.float64)
    for frequency, start in _NOTES:
        signal += _note(frequency, start, _NOTE_LENGTH, total)

    # Fade the tail to exactly zero: whatever the decay left behind must not
    # be cut off mid-cycle.
    fade = int(_FADE_OUT_SEC * SAMPLE_RATE)
    if fade > 1:
        signal[-fade:] *= np.linspace(1.0, 0.0, fade)
    signal[0] = 0.0
    signal[-1] = 0.0

    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak * _PEAK

    samples = np.round(signal * 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


@functools.lru_cache(maxsize=1)
def _chime_bytes() -> bytes:
    return synthesize_chime()


@functools.lru_cache(maxsize=1)
def _chime_path() -> str:
    """The chime as a file, for players that cannot take a buffer."""
    path = Path(tempfile.gettempdir()) / "hairguard_alert.wav"
    path.write_bytes(_chime_bytes())
    return str(path)


def _windows_backend() -> None:
    import winsound

    # SND_MEMORY plays the buffer directly — no temp file, and no gap while
    # a player process starts up.
    winsound.PlaySound(_chime_bytes(), winsound.SND_MEMORY)


def _macos_backend() -> None:
    subprocess.run(
        ["afplay", _chime_path()],
        check=False,
        timeout=5,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _linux_backend() -> None:
    for command in (["paplay", _chime_path()], ["aplay", "-q", _chime_path()]):
        try:
            if subprocess.run(
                command, check=False, timeout=5,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0:
                return
        except (OSError, subprocess.SubprocessError):
            continue
    # Last resort: the terminal bell.
    sys.stdout.write("\a")
    sys.stdout.flush()


def default_backend() -> Callable[[], None]:
    """The way this platform makes a noise."""
    if sys.platform == "win32":
        return _windows_backend
    if sys.platform == "darwin":
        return _macos_backend
    return _linux_backend


# Silence between repeats. Long enough that the chirp stays a chirp rather
# than becoming a siren, short enough that it cannot be waited out.
DEFAULT_GAP_SEC = 0.6

# The stop flag is checked this often while waiting, so the sound never
# outlasts the hand by more than one tone plus this.
_STOP_CHECK_SEC = 0.05


class AlertSound:
    """Repeats a short sound while an alert is up, off the caller's thread.

    The sound keeps going until :meth:`stop` — which is the point. A single
    chirp is easy to ignore; one that only ends when the hand moves away
    makes moving the hand away the obvious thing to do.
    """

    def __init__(
        self,
        enabled: bool = True,
        backend: Optional[Callable[[], None]] = None,
        gap_sec: float = DEFAULT_GAP_SEC,
    ) -> None:
        self._enabled = bool(enabled)
        self._backend = backend if backend is not None else default_backend()
        self._gap_sec = float(gap_sec)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def active(self) -> bool:
        """True while the sound is repeating."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Begin repeating the sound. Returns at once; safe to call again."""
        if not self._enabled:
            return

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="alert-sound", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop repeating. Waits only for the tone already sounding."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            self._thread = None

    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._backend()
            except Exception as exc:  # noqa: BLE001
                # No audio device, no mixer, no permission. It will not fix
                # itself, so give up rather than retry in a tight loop —
                # and never let it disturb the alert itself.
                print(f"[sound] Звук не воспроизведён: {exc}")
                return

            # Wait out the gap in slices so stopping stays responsive.
            waited = 0.0
            while waited < self._gap_sec:
                if self._stop.wait(min(_STOP_CHECK_SEC, self._gap_sec - waited)):
                    return
                waited += _STOP_CHECK_SEC
