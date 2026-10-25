"""hairlab.features — landmark frames turned into model input.

The first model trains on landmark sequences, not pixels, so this module is
where the dataset actually gets its information content. Two ideas carry it.

*Everything is measured in the head's own reference frame.* Raw MediaPipe
coordinates are relative to the image, so the same gesture produces entirely
different numbers when the user sits further left, leans towards the camera
or tilts their head. Those are three invariances the model would otherwise
have to learn from a few hundred clips, and it cannot. Instead the head
supplies an origin (the midpoint of the ears), a unit of length (the ear-to-
ear distance) and an orientation (the roll of the ear line), and the hand is
expressed in that frame. Head motion then cancels for free: a nod moves the
hand across the image but not relative to the head, which is correct — it is
not hand motion.

*Finger shape is measured against the hand's own size.* Position belongs in
head units, but a pinch does not: whether the thumb and index finger are
touching is a fact about the hand, so those ratios are divided by the palm
length. The two scales are kept deliberately separate.

On top of the normalised coordinates sit features that name the gesture
directly — pinch, spread, curl, palm angle, and velocities relative to the
head. The one that matters most is ``fingertip_residual_speed``: fingertips
moving while the wrist stays put. That is what twisting *is*, and what tells
it apart from a hand that merely rests on the head or drags across it.

The hairline each clip also carries is deliberately *not* used. The face
landmarker finds it in only 47% of frames, and it fails exactly when the head
is turned — which is what reaching for a strand looks like. Its absence
therefore predicts twisting on its own, while describing the detector rather
than the gesture, and where it is present it does not separate twisting from
scratching at all. See the tests for the measurements.

Velocities are divided by the real elapsed time between frames, never by a
frame count. The capture loop is nominally 10 fps but measures between 5.5
and 7.3, drifting with CPU load both across clips and within one, so a
per-frame delta would encode machine load as if it were gesture speed.

Nothing here may emit a NaN or an inf. A single non-finite value propagates
through a training step into every weight, and the loss goes to nan some
epochs later with no trace of the frame that caused it. So every quantity is
bounded, every divisor has a floor, and the result is scrubbed on the way out.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

# MediaPipe hand topology. The feature layout is positional, so a landmark
# list that is not the full 21-point hand is treated as no hand at all.
_HAND_POINTS = 21
_WRIST = 0
_MIDDLE_MCP = 9

_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Base-to-tip chains. The thumb starts at the CMC joint, which is where its
# bending actually happens; the others start at the knuckle.
_FINGER_CHAINS: dict[str, tuple[int, ...]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_TIPS = {name: chain[-1] for name, chain in _FINGER_CHAINS.items()}

_POSE_EARS = ("left_ear", "right_ear")
_POSE_SHOULDERS = ("left_shoulder", "right_shoulder")

# Below this the pose landmark is a guess rather than a measurement, and a
# guessed ear defines a reference frame that jumps between frames.
_MIN_POSE_CONF = 0.3

# An ear-to-ear span this small is not a measurement: either the detector
# collapsed or the head is turned fully side-on.
_MIN_EAR_SPAN = 0.02
# Rough anthropometric ratio, used only when the ears are unavailable. Its
# exact value just sets a constant scale factor the model absorbs; what
# matters is that it tracks the user's distance from the camera.
_EAR_SPAN_PER_SHOULDER_SPAN = 0.40
# A side-on head foreshortens the ear line while the head itself stays the
# same size, which would inflate every coordinate. The shoulders are far more
# stable at an angle, so they provide a floor rather than a replacement.
_EAR_SPAN_FLOOR_RATIO = 0.5
_DEFAULT_HEAD_SCALE = 0.20
_MIN_HAND_SCALE = 1e-3
_EPS = 1e-6

# Bounds. Positions and distances are in head-widths, velocities in
# head-widths per second; anything past these is a detector failure, not a
# gesture, and is clamped so it cannot dominate a training batch.
MAX_COORD = 20.0
MAX_RATIO = 20.0
MAX_VELOCITY = 20.0

# The capture loop runs near 10 fps but measures as low as 6, so velocities
# are divided by the real elapsed time. A step shorter than this is clock
# noise; one longer than this is a stall, and what happened in between is
# unknown, so the velocity is reported as zero rather than invented.
_MIN_DT = 1.0 / 120.0
_MAX_DT = 1.0


def _build_feature_names() -> list[str]:
    names = ["hand_present", "pose_present"]
    for i in range(_HAND_POINTS):
        names += [f"hand_{i:02d}_x", f"hand_{i:02d}_y", f"hand_{i:02d}_z"]
    names += ["pinch_thumb_index", "pinch_thumb_middle", "spread_index_pinky"]
    names += [f"curl_{finger}" for finger in _FINGERS]
    names += ["palm_angle_sin", "palm_angle_cos", "hand_scale", "head_dist"]
    for finger in _FINGERS:
        names += [f"vel_{finger}_x", f"vel_{finger}_y", f"vel_{finger}_z"]
    names += ["vel_wrist_x", "vel_wrist_y", "vel_wrist_z"]
    names += ["fingertip_residual_speed"]
    return names


FEATURE_NAMES: list[str] = _build_feature_names()
FEATURE_DIM: int = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Parsing — every reader below tolerates a frame the capture side wrote badly
# ---------------------------------------------------------------------------

def _finite(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _hand_array(raw: Any) -> Optional[np.ndarray]:
    """21 x 3 of raw normalised coordinates, or None if unusable."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    if len(raw) < _HAND_POINTS:
        return None

    out = np.zeros((_HAND_POINTS, 3), dtype=np.float64)
    for i in range(_HAND_POINTS):
        point = raw[i]
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
            return None
        if len(point) < 2:
            return None
        x, y = _as_float(point[0]), _as_float(point[1])
        # z is relative depth; a missing or broken one is better treated as
        # the wrist plane than as a reason to drop the whole hand.
        z = _as_float(point[2]) if len(point) > 2 else 0.0
        out[i] = (
            x if x is not None else 0.0,
            y if y is not None else 0.0,
            z if z is not None else 0.0,
        )
    return out


def _pose_point(pose: Any, name: str) -> Optional[np.ndarray]:
    if not isinstance(pose, dict):
        return None
    point = pose.get(name)
    if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
        return None
    if len(point) < 2:
        return None
    x, y = _as_float(point[0]), _as_float(point[1])
    if x is None or y is None:
        return None
    conf = _as_float(point[2]) if len(point) > 2 else 1.0
    if conf is None or conf < _MIN_POSE_CONF:
        return None
    return np.array([x, y], dtype=np.float64)


# ---------------------------------------------------------------------------
# The head-relative reference frame
# ---------------------------------------------------------------------------

class _Head(NamedTuple):
    origin: np.ndarray   # full-frame normalised coordinates
    scale: float         # head width, in the same units; always > 0
    roll: float          # radians, angle of the right-ear -> left-ear line
    present: bool        # was there a real head landmark to anchor on?


def _head_reference(pose: Any, hand: Optional[np.ndarray]) -> _Head:
    """Origin, unit of length and orientation of the head.

    The fallbacks matter more than the happy path: ears disappear whenever
    the user turns side-on, which is exactly when they are twisting hair
    behind their head. Falling back to the nose and the shoulders keeps those
    clips usable instead of dropping them.
    """
    left_ear = _pose_point(pose, "left_ear")
    right_ear = _pose_point(pose, "right_ear")
    nose = _pose_point(pose, "nose")
    shoulders = [p for p in (_pose_point(pose, n) for n in _POSE_SHOULDERS) if p is not None]

    origin: Optional[np.ndarray] = None
    scale: Optional[float] = None
    roll = 0.0

    if left_ear is not None and right_ear is not None:
        origin = (left_ear + right_ear) / 2.0
        span_vec = left_ear - right_ear
        span = float(math.hypot(span_vec[0], span_vec[1]))
        if span >= _MIN_EAR_SPAN:
            # Taken in a fixed landmark order, so the angle is stable even
            # though the camera image is mirrored.
            roll = float(math.atan2(span_vec[1], span_vec[0]))
            scale = span

    shoulder_span = 0.0
    if len(shoulders) == 2:
        delta = shoulders[0] - shoulders[1]
        shoulder_span = float(math.hypot(delta[0], delta[1]))
    shoulder_estimate = shoulder_span * _EAR_SPAN_PER_SHOULDER_SPAN

    if scale is not None:
        scale = max(scale, shoulder_estimate * _EAR_SPAN_FLOOR_RATIO)
    elif shoulder_estimate >= _MIN_EAR_SPAN:
        scale = shoulder_estimate

    if origin is None and nose is not None:
        origin = nose
    present = origin is not None
    if origin is None and len(shoulders) == 2:
        origin = (shoulders[0] + shoulders[1]) / 2.0

    if scale is None and hand is not None:
        # No head at all: fall back to the hand's own span, which makes the
        # features hand-relative rather than head-relative. Degraded, but a
        # consistent degradation the model can still read something from.
        extent = hand[:, :2].max(axis=0) - hand[:, :2].min(axis=0)
        span = float(max(extent[0], extent[1]))
        scale = span if span >= _MIN_EAR_SPAN else None

    if origin is None:
        origin = (hand[_WRIST, :2].copy() if hand is not None
                  else np.array([0.5, 0.5], dtype=np.float64))
    if scale is None or not math.isfinite(scale) or scale < _MIN_EAR_SPAN:
        scale = _DEFAULT_HEAD_SCALE

    return _Head(_finite(origin), float(scale), roll, present)


class _Normalised(NamedTuple):
    hand: Optional[np.ndarray]   # 21 x 3 in head units, de-rolled
    head: _Head
    t: Optional[float]


def _normalise(frame: Any) -> _Normalised:
    if not isinstance(frame, dict):
        frame = {}
    hand_raw = _hand_array(frame.get("hand"))
    head = _head_reference(frame.get("pose"), hand_raw)

    # Rotating by -roll puts the ear line on the x axis, so a tilted head and
    # an upright one yield the same coordinates.
    cos_r, sin_r = math.cos(-head.roll), math.sin(-head.roll)

    def to_head_frame(xy: np.ndarray) -> np.ndarray:
        local = (xy - head.origin) / head.scale
        x, y = local[..., 0], local[..., 1]
        return np.stack([cos_r * x - sin_r * y, sin_r * x + cos_r * y], axis=-1)

    hand = None
    if hand_raw is not None:
        hand = np.empty((_HAND_POINTS, 3), dtype=np.float64)
        hand[:, :2] = to_head_frame(hand_raw[:, :2])
        # In-plane roll leaves depth alone; z only needs the same unit.
        hand[:, 2] = hand_raw[:, 2] / head.scale
        hand = np.clip(_finite(hand), -MAX_COORD, MAX_COORD)

    return _Normalised(hand, head, _as_float(frame.get("t")))


# ---------------------------------------------------------------------------
# Derived gesture features
# ---------------------------------------------------------------------------

def _curl(hand: np.ndarray, chain: tuple[int, ...]) -> float:
    """How folded a finger is, on a scale of 0 (straight) to 1 (fully bent).

    Base-to-tip distance over the length of the finger walked joint by joint.
    By the triangle inequality the ratio cannot exceed 1, so the feature is
    bounded without clipping, and it needs no angles — which would have to be
    unwrapped and would be far noisier on MediaPipe's z.
    """
    points = hand[list(chain)]
    segments = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if segments < _EPS:
        return 1.0
    reach = float(np.linalg.norm(points[-1] - points[0]))
    return float(np.clip(1.0 - reach / segments, 0.0, 1.0))


def _velocity(
    current: Optional[np.ndarray],
    previous: Optional[np.ndarray],
    dt: Optional[float],
) -> np.ndarray:
    """Per-landmark velocity in head units per second, or zeros if unknown."""
    if current is None or previous is None or dt is None:
        return np.zeros((_HAND_POINTS, 3), dtype=np.float64)
    return np.clip((current - previous) / dt, -MAX_VELOCITY, MAX_VELOCITY)


def _timestep(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None:
        return None
    dt = current - previous
    if dt <= 0.0 or dt > _MAX_DT:
        return None
    return max(dt, _MIN_DT)


def frame_features(frame: dict, previous: Optional[dict]) -> np.ndarray:
    """Feature vector for one frame, given the frame before it (or None).

    ``previous`` is the raw neighbouring frame dict; it is normalised into its
    own head frame, so the velocities that come out describe motion relative
    to the head rather than across the image.
    """
    out = np.zeros(FEATURE_DIM, dtype=np.float64)
    current = _normalise(frame)
    prior = _normalise(previous) if previous is not None else None

    hand = current.hand
    out[0] = 1.0 if hand is not None else 0.0
    out[1] = 1.0 if current.head.present else 0.0

    cursor = 2
    if hand is not None:
        out[cursor:cursor + _HAND_POINTS * 3] = hand.reshape(-1)
    cursor += _HAND_POINTS * 3

    if hand is not None:
        # The palm length is the hand's own yardstick, so finger-shape ratios
        # stay the same whether the hand is near the camera or far from it.
        palm = float(np.linalg.norm(hand[_MIDDLE_MCP] - hand[_WRIST]))
        denominator = max(palm, _MIN_HAND_SCALE)

        def ratio(a: int, b: int) -> float:
            distance = float(np.linalg.norm(hand[a] - hand[b]))
            return float(np.clip(distance / denominator, 0.0, MAX_RATIO))

        out[cursor] = ratio(_TIPS["thumb"], _TIPS["index"])
        out[cursor + 1] = ratio(_TIPS["thumb"], _TIPS["middle"])
        out[cursor + 2] = ratio(_TIPS["index"], _TIPS["pinky"])
        cursor += 3

        for finger in _FINGERS:
            out[cursor] = _curl(hand, _FINGER_CHAINS[finger])
            cursor += 1

        # Palm direction as sin/cos rather than an angle: the raw angle jumps
        # by 2*pi at the wrap, and a cliff in feature space is a cliff the
        # model has to memorise its way around.
        palm_vec = hand[_MIDDLE_MCP, :2] - hand[_WRIST, :2]
        angle = math.atan2(palm_vec[1], palm_vec[0]) if palm > _EPS else 0.0
        out[cursor] = math.sin(angle)
        out[cursor + 1] = math.cos(angle)
        # Hand size in head units: the model's only clue about how far the
        # hand is from the camera, which the ratios above deliberately hide.
        out[cursor + 2] = min(palm, MAX_COORD)
        centroid = hand[:, :2].mean(axis=0)
        out[cursor + 3] = min(float(np.linalg.norm(centroid)), MAX_COORD)
        cursor += 4
    else:
        cursor += 3 + len(_FINGERS) + 4

    dt = _timestep(current.t, prior.t if prior is not None else None)
    velocity = _velocity(hand, prior.hand if prior is not None else None, dt)
    for finger in _FINGERS:
        out[cursor:cursor + 3] = velocity[_TIPS[finger]]
        cursor += 3
    out[cursor:cursor + 3] = velocity[_WRIST]
    cursor += 3

    # Fingertip motion with the whole hand's motion subtracted. Twisting is
    # fingers working against a wrist that stays put; resting a hand on the
    # head and dragging it across are both wrist motion with still fingers.
    residual = velocity[[_TIPS[f] for f in _FINGERS]] - velocity[_WRIST]
    out[cursor] = min(float(np.linalg.norm(residual, axis=1).mean()), MAX_VELOCITY)
    cursor += 1

    assert cursor == FEATURE_DIM, f"wrote {cursor} of {FEATURE_DIM} features"
    # Belt and braces: whatever slipped through above, nothing non-finite
    # leaves this function.
    return _finite(out).astype(np.float32)


def clip_features(frames: list[dict], seq_len: int = 20) -> np.ndarray:
    """One clip as a (seq_len, FEATURE_DIM) sequence.

    Short clips are zero-padded at the end rather than padded by repeating
    the last frame: an all-zero row has ``hand_present`` and ``pose_present``
    clear, so it reads as "no data" instead of inventing a hand that froze.
    Long clips are truncated from the front, keeping the ordering
    deterministic so a given clip always yields the same tensor.
    """
    out = np.zeros((max(seq_len, 0), FEATURE_DIM), dtype=np.float32)
    if not frames:
        return out
    for i in range(min(len(frames), out.shape[0])):
        out[i] = frame_features(frames[i], frames[i - 1] if i > 0 else None)
    return out
