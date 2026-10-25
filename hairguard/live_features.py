"""hairguard.live_features — the detection loop's frames, shaped for the model.

This module is the seam where train/serve skew would live, so it is written
to make skew impossible rather than unlikely.

The model was trained on ``hairlab.features.clip_features`` reading a stored
clip's ``meta.json``. At inference time the same numbers have to come out of
the objects the detection loop already holds. There are two ways that can go
wrong, and each gets a structural answer rather than a careful one.

*The per-frame record could differ from the stored one.* Landmark ordering,
coordinate convention, which keys exist, and — least visible and most
damaging — the rounding, since ``hairlab.capture`` writes coordinates to five
decimals and timestamps to four. A velocity divided by a differently rounded
``t`` is a different velocity. So this module does not restate the format: it
calls the very functions ``hairlab.capture`` uses to write it. If that module
ever moves them, the import fails loudly and the app falls back to the rule
instead of quietly scoring the wrong numbers.

*The window could be assembled differently from a clip.* ``clip_features``
computes row *i* from frame *i* and frame *i-1*, with row 0 given no
predecessor at all. :class:`FeatureWindow` reproduces that exactly while
caching: as the window slides, a frame's in-window predecessor never changes,
so only the row that has just become row 0 — the one that must lose its
predecessor — needs recomputing. Two ``frame_features`` calls per tick
instead of twenty, for a tensor that is bit-identical to ``clip_features``.
``tests/test_live_features.py`` proves that on every recorded clip.

Two consequences of matching the training path, both deliberate:

  - Only frames with a hand in the hair zone enter the window, because the
    collector only ever buffered those; the model has never seen anything
    else and has no defined meaning outside the zone.
  - A window is discarded the moment the hand leaves, because a stored clip
    is one unbroken visit. A window stitched across a gap is not a sample of
    the distribution the model was fitted to.

Filling a window that is not full yet
-------------------------------------
The app interrupts a behaviour, so how long it waits is most of its value,
and waiting for twenty real frames is nearly three seconds at the 6.75 fps
this loop achieves. ``hairlab.truncation`` measured what the model does when
asked earlier, cross-validated over capture days, and the answer turned on
*what fills the empty slots* far more than on how many are empty.

``clip_features`` pads short clips with zero rows, which is right for
training — an all-zero row has ``hand_present`` clear and reads as "no data"
rather than inventing a frozen hand. **At inference it destroys the model.**
The network standardises its input first, so a zero row becomes
``-mean/std``: a large constant vector nothing in training resembled. Global
average pooling over time then lets those rows dominate, and the logit
saturates — ROC AUC exactly 0.500, every clip scoring 1.0 below k=17 and 0.0
above it. A detector that has stopped reading its input.

Repeating the *last real frame* instead keeps it intact. A hand holding still
is something the model saw several hundred of, and the repeated rows are all
identical — a frame against a copy of itself has zero velocity — so they cost
one extra ``frame_features`` call, broadcast. Measured recall falls only from
86.7% at twenty frames to 78.0% at three, with the false-alarm rate on "hand
resting on head" slightly *better* (3.2% against 3.6%), which is still well
ahead of the 53.9% rule this replaces on both axes, at a ninth of the wait.

Hence ``min_frames``: below it nothing is scored, at or above it the window is
repeat-padded. It defaults to the full window, so partial scoring is
something a caller asks for rather than something they get by accident, and
``frame_features`` is never handed a zero row by this module.
"""
from __future__ import annotations

import collections
from typing import Any, Optional

import numpy as np

from hairlab.features import FEATURE_DIM, frame_features

# Imported rather than reimplemented: these two decide the stored frame
# format, including the rounding, and being the same code is the only way to
# be sure of agreeing with it. They are private to hairlab.capture because
# nothing was meant to depend on the format — which is exactly what the
# inference path has to do.
from hairlab.capture import _hand_landmarks, _pose_landmarks

__all__ = ["FeatureWindow", "frame_record"]

# Timestamps are stored to four decimals by hairlab.capture. Velocities
# divide by the difference between two of them, so the rounding is part of
# the feature definition and not a display concern.
_T_DECIMALS = 4


def frame_record(hand: Any, pose_result: Any, t: float) -> dict[str, Any]:
    """The frame dict ``hairlab.capture`` would have stored for this tick.

    ``hand`` is the single hand the analyzer found inside the hair zone, or
    None when it could not identify one — mirroring the stored clips, where a
    frame without an identified hand simply has no ``hand`` key and reads as
    "no hand" to the extractor.

    The stored record also carries ``box``, ``source_px`` and ``forehead``.
    None of them is a feature input (see the ``hairlab.features`` docstring on
    why the hairline is deliberately unused), and computing a crop box for a
    frame nobody will ever look at would be work for nothing, so they are left
    out. That the omission cannot change the tensor is checked against real
    clips rather than assumed.
    """
    record: dict[str, Any] = {"t": round(float(t), _T_DECIMALS)}
    if hand is not None:
        record["hand"] = _hand_landmarks(hand)
    pose_points = _pose_landmarks(pose_result)
    if pose_points:
        record["pose"] = pose_points
    return record


class FeatureWindow:
    """A sliding window of frame records, yielding the model's input tensor.

    Push one record per in-zone tick; :meth:`push` returns a
    ``(seq_len, FEATURE_DIM)`` float32 array once ``min_frames`` records
    exist, repeat-padding until the window is genuinely full, and None before
    that. Call :meth:`reset` the moment the hand leaves the zone.

    ``min_frames`` defaults to ``seq_len`` — no partial scoring unless asked
    for. See the module docstring for why the padding is a repeat and never a
    zero.
    """

    def __init__(self, seq_len: int = 20, min_frames: Optional[int] = None) -> None:
        if int(seq_len) < 1:
            raise ValueError(f"seq_len must be at least 1, got {seq_len}")
        self._seq_len = int(seq_len)
        if min_frames is None:
            min_frames = self._seq_len
        if int(min_frames) < 1:
            raise ValueError(f"min_frames must be at least 1, got {min_frames}")
        # Clamped rather than refused: a configuration asking for more frames
        # than the window holds means "wait for a full window", which is a
        # coherent request and the conservative one.
        self._min_frames = min(int(min_frames), self._seq_len)
        self._records: collections.deque = collections.deque(maxlen=self._seq_len)
        # Row i of this deque is frame_features(record_i, record_{i-1}) —
        # correct for every window position except the first, which has to
        # forget its predecessor and is recomputed on each read.
        self._rows: collections.deque = collections.deque(maxlen=self._seq_len)

    # ------------------------------------------------------------------

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def min_frames(self) -> int:
        """Records needed before the window is worth scoring."""
        return self._min_frames

    @property
    def filled(self) -> int:
        """Records held so far towards a full window."""
        return len(self._records)

    def reset(self) -> None:
        """Drop the window. The next visit starts with no history behind it."""
        self._records.clear()
        self._rows.clear()

    def push(self, record: dict[str, Any]) -> Optional[np.ndarray]:
        """Add one frame. Returns the window tensor once it is worth scoring."""
        previous = self._records[-1] if self._records else None
        self._rows.append(frame_features(record, previous))
        self._records.append(record)

        if len(self._records) < self._min_frames:
            return None
        return self._tensor()

    # ------------------------------------------------------------------

    def _tensor(self) -> np.ndarray:
        """The window as ``clip_features`` would have computed it.

        Equal, row for row, to ``clip_features(records + [records[-1]] * gap)``
        — the repeat-padded frame list ``hairlab.truncation`` measured. Proven
        against it on recorded clips rather than argued for.

        A fresh array every time: the caller may keep it, and the next tick
        must not change what they are holding.
        """
        count = len(self._records)
        out = np.empty((self._seq_len, FEATURE_DIM), dtype=np.float32)
        # The oldest frame's cached row was computed against a predecessor
        # that has now slid out of the window, so it is recomputed with none
        # — which is what clip_features does for row 0.
        out[0] = frame_features(self._records[0], None)
        for index in range(1, count):
            out[index] = self._rows[index]
        if count < self._seq_len:
            # Repeat-padding. Every remaining row is the last real frame
            # against a copy of itself, so they are all identical and one
            # call fills them: a hand holding still, which is in the
            # distribution, rather than a zero row, which is not.
            last = self._records[-1]
            out[count:] = frame_features(last, last)
        return out
