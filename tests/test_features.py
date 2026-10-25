"""Tests for hairlab.features — landmarks turned into a training input.

The model sees landmark sequences, never pixels, so these features *are* the
dataset. Two properties are therefore tested harder than anything else:

*Invariance.* A scene translated, scaled or rolled must yield an identical
feature vector. Whatever the model has to learn, it must not have to learn
that sitting further left is the same gesture.

*Finiteness.* One NaN poisons a whole training batch and the loss turns to
nan several epochs in, far from the frame that caused it. So every degenerate
input the capture side can produce is checked here instead.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hairlab.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    MAX_VELOCITY,
    clip_features,
    frame_features,
)

# ---------------------------------------------------------------------------
# Synthetic scenes
# ---------------------------------------------------------------------------

# A hand in its own local frame: wrist at the origin, fingers extended along
# -y (up, in image coordinates), thumb out to the left. Every finger chain is
# collinear, so a straight finger has a curl of exactly zero.
_HAND_LOCAL: list[tuple[float, float, float]] = [
    (0.00, 0.00, 0.00),    # 0  wrist
    (-0.15, -0.10, 0.01),  # 1  thumb CMC
    (-0.30, -0.20, 0.02),  # 2  thumb MCP
    (-0.40, -0.30, 0.03),  # 3  thumb IP
    (-0.50, -0.40, 0.04),  # 4  thumb tip
    (-0.10, -0.50, 0.00),  # 5  index MCP
    (-0.10, -0.70, 0.00),  # 6  index PIP
    (-0.10, -0.85, 0.00),  # 7  index DIP
    (-0.10, -1.00, 0.00),  # 8  index tip
    (0.05, -0.52, 0.00),   # 9  middle MCP
    (0.05, -0.74, 0.00),   # 10
    (0.05, -0.90, 0.00),   # 11
    (0.05, -1.05, 0.00),   # 12 middle tip
    (0.20, -0.50, -0.01),  # 13 ring MCP
    (0.20, -0.70, -0.01),  # 14
    (0.20, -0.85, -0.01),  # 15
    (0.20, -0.98, -0.01),  # 16 ring tip
    (0.33, -0.45, -0.02),  # 17 pinky MCP
    (0.33, -0.60, -0.02),  # 18
    (0.33, -0.72, -0.02),  # 19
    (0.33, -0.83, -0.02),  # 20 pinky tip
]


def _index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _feat(vec: np.ndarray, name: str) -> float:
    return float(vec[_index(name)])


def _place_hand(
    local: list[tuple[float, float, float]],
    origin: tuple[float, float] = (0.70, 0.50),
    size: float = 0.20,
    angle: float = 0.0,
) -> list[list[float]]:
    """Put a local hand into full-frame normalised coordinates."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [
        [
            origin[0] + size * (cos_a * x - sin_a * y),
            origin[1] + size * (sin_a * x + cos_a * y),
            size * z,
        ]
        for x, y, z in local
    ]


def _head_pose(
    centre: tuple[float, float] = (0.50, 0.50),
    ear_span: float = 0.20,
    roll: float = 0.0,
    conf: float = 0.99,
) -> dict[str, list[float]]:
    """The five pose points, with the ear line rolled by ``roll`` radians."""
    half = ear_span / 2
    dx, dy = half * math.cos(roll), half * math.sin(roll)
    # right_ear -> left_ear is the reference direction, so +roll tips that way.
    return {
        "nose": [centre[0], centre[1] + 0.1 * ear_span, conf],
        "left_ear": [centre[0] + dx, centre[1] + dy, conf],
        "right_ear": [centre[0] - dx, centre[1] - dy, conf],
        "left_shoulder": [centre[0] + ear_span, centre[1] + 1.4 * ear_span, conf],
        "right_shoulder": [centre[0] - ear_span, centre[1] + 1.4 * ear_span, conf],
    }


def _frame(
    hand: list[list[float]] | None = None,
    pose: dict[str, list[float]] | None = None,
    t: float = 0.0,
    forehead: list[list[float]] | None = None,
) -> dict:
    """A frame dict shaped exactly like the one hairlab.capture writes."""
    frame: dict = {"t": t, "box": [0, 0, 256, 256], "source_px": 256, "hand_index": 0}
    if hand is None:
        hand = _place_hand(_HAND_LOCAL)
    if pose is None:
        pose = _head_pose()
    if hand is not False:
        frame["hand"] = hand
    if pose is not False:
        frame["pose"] = pose
    if forehead is not None:
        frame["forehead"] = forehead
    return frame


def _transform(frame: dict, scale: float = 1.0, angle: float = 0.0,
               tx: float = 0.0, ty: float = 0.0) -> dict:
    """Apply one similarity transform to every coordinate in a frame.

    This is the operation the features must be blind to: the same gesture,
    seen from nearer or further away, off to one side, head tilted.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def xy(x: float, y: float) -> tuple[float, float]:
        return (tx + scale * (cos_a * x - sin_a * y),
                ty + scale * (sin_a * x + cos_a * y))

    out = dict(frame)
    if "hand" in frame:
        out["hand"] = [[*xy(p[0], p[1]), scale * p[2]] for p in frame["hand"]]
    if "pose" in frame:
        out["pose"] = {
            name: [*xy(p[0], p[1]), *p[2:]] for name, p in frame["pose"].items()
        }
    if "forehead" in frame:
        out["forehead"] = [list(xy(p[0], p[1])) for p in frame["forehead"]]
    return out


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_feature_dim_matches_the_names() -> None:
    assert FEATURE_DIM == len(FEATURE_NAMES)


def test_feature_names_are_unique() -> None:
    """Duplicated names make a trained model impossible to interpret."""
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_frame_features_shape_and_dtype() -> None:
    vec = frame_features(_frame(), None)
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# Finiteness, against every degenerate input the capture side can write
# ---------------------------------------------------------------------------

def test_a_normal_frame_is_finite() -> None:
    assert np.all(np.isfinite(frame_features(_frame(), None)))


def test_missing_pose_is_finite() -> None:
    vec = frame_features(_frame(pose=False), None)
    assert np.all(np.isfinite(vec))
    assert _feat(vec, "pose_present") == 0.0


def test_missing_hand_is_finite() -> None:
    vec = frame_features(_frame(hand=False), None)
    assert np.all(np.isfinite(vec))
    assert _feat(vec, "hand_present") == 0.0


def test_present_flags_are_set_for_a_full_frame() -> None:
    vec = frame_features(_frame(), None)
    assert _feat(vec, "hand_present") == 1.0
    assert _feat(vec, "pose_present") == 1.0


def test_coincident_ears_are_finite() -> None:
    """Head scale collapses to zero and the roll angle is undefined."""
    pose = _head_pose(ear_span=0.0)
    assert np.all(np.isfinite(frame_features(_frame(pose=pose), None)))


def test_a_pose_with_only_a_nose_is_finite() -> None:
    pose = {"nose": [0.5, 0.5, 0.99]}
    assert np.all(np.isfinite(frame_features(_frame(pose=pose), None)))


def test_a_wholly_degenerate_scene_is_finite() -> None:
    """Every landmark on the same point: no scale, no direction, no extent."""
    hand = [[0.5, 0.5, 0.0] for _ in range(21)]
    pose = {name: [0.5, 0.5, 0.99] for name in
            ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder")}
    assert np.all(np.isfinite(frame_features(_frame(hand=hand, pose=pose), None)))


def test_an_empty_frame_is_finite() -> None:
    assert np.all(np.isfinite(frame_features({}, None)))


def test_nan_landmarks_are_scrubbed() -> None:
    """A NaN must not escape into the batch, wherever it came from."""
    hand = _place_hand(_HAND_LOCAL)
    hand[0] = [float("nan")] * 3
    hand[8] = [float("inf"), float("-inf"), 0.0]
    vec = frame_features(_frame(hand=hand), None)
    assert np.all(np.isfinite(vec))


def test_nan_in_the_pose_is_scrubbed() -> None:
    pose = _head_pose()
    pose["left_ear"] = [float("nan"), float("nan"), 0.99]
    assert np.all(np.isfinite(frame_features(_frame(pose=pose), None)))


def test_a_short_hand_is_rejected_not_crashed() -> None:
    """Truncated landmark lists should degrade, not raise."""
    vec = frame_features(_frame(hand=[[0.5, 0.5, 0.0]] * 7), None)
    assert np.all(np.isfinite(vec))
    assert _feat(vec, "hand_present") == 0.0


def test_landmarks_outside_the_frame_are_finite() -> None:
    """MediaPipe reports out-of-image landmarks rather than dropping them."""
    hand = _place_hand(_HAND_LOCAL, origin=(-0.4, 1.6), size=0.9)
    assert np.all(np.isfinite(frame_features(_frame(hand=hand), None)))


# ---------------------------------------------------------------------------
# Invariance — the whole point of the head-relative reference frame
# ---------------------------------------------------------------------------

def test_translation_invariance() -> None:
    base = _frame()
    moved = _transform(base, tx=0.21, ty=-0.13)
    np.testing.assert_allclose(
        frame_features(base, None), frame_features(moved, None), atol=2e-3
    )


def test_scale_invariance() -> None:
    """The user leaning towards the camera is not a different gesture."""
    base = _frame()
    nearer = _transform(base, scale=1.9)
    np.testing.assert_allclose(
        frame_features(base, None), frame_features(nearer, None), atol=2e-3
    )


def test_roll_invariance() -> None:
    """A tilted head must produce the numbers of an upright one."""
    base = _frame()
    tilted = _transform(base, angle=0.6)
    np.testing.assert_allclose(
        frame_features(base, None), frame_features(tilted, None), atol=2e-3
    )


def test_combined_similarity_invariance() -> None:
    base = _frame()
    other = _transform(base, scale=0.55, angle=-1.2, tx=0.3, ty=0.2)
    np.testing.assert_allclose(
        frame_features(base, None), frame_features(other, None), atol=2e-3
    )


def test_velocity_is_invariant_too() -> None:
    """Velocities live in the same normalised frame, so they transform alike."""
    prev = _frame(t=0.0)
    cur = _frame(hand=_place_hand(_HAND_LOCAL, origin=(0.76, 0.46)), t=0.1)
    args = dict(scale=0.7, angle=0.9, tx=0.1, ty=-0.05)
    np.testing.assert_allclose(
        frame_features(cur, prev),
        frame_features(_transform(cur, **args), _transform(prev, **args)),
        atol=3e-3,
    )


def test_the_head_centre_is_the_origin_and_head_width_the_unit() -> None:
    """A wrist one head-width to the right of the ears' midpoint reads as x=1."""
    frame = _frame(
        hand=_place_hand(_HAND_LOCAL, origin=(0.70, 0.50), size=0.05),
        pose=_head_pose(centre=(0.50, 0.50), ear_span=0.20),
    )
    vec = frame_features(frame, None)
    assert _feat(vec, "hand_00_x") == pytest.approx(1.0, abs=1e-3)
    assert _feat(vec, "hand_00_y") == pytest.approx(0.0, abs=1e-3)
    assert _feat(vec, "head_dist") > 0.5


def test_the_reference_frame_falls_back_to_the_nose() -> None:
    """Ears drop out at side angles; the features must still be head-relative."""
    pose = {"nose": [0.5, 0.5, 0.99],
            "left_shoulder": [0.7, 0.78, 0.9],
            "right_shoulder": [0.3, 0.78, 0.9]}
    near = _frame(hand=_place_hand(_HAND_LOCAL, origin=(0.55, 0.50), size=0.05), pose=pose)
    far = _frame(hand=_place_hand(_HAND_LOCAL, origin=(0.80, 0.50), size=0.05), pose=pose)
    assert _feat(frame_features(far, None), "head_dist") > _feat(
        frame_features(near, None), "head_dist"
    )


# ---------------------------------------------------------------------------
# Gesture features
# ---------------------------------------------------------------------------

def test_pinch_ratio_drops_when_the_thumb_meets_the_index() -> None:
    """Holding a strand is a pinch; the open hand of a scratch is not."""
    open_hand = _place_hand(_HAND_LOCAL)
    pinched = list(_HAND_LOCAL)
    pinched[4] = (-0.10, -0.98, 0.00)   # thumb tip onto the index tip
    pinch_vec = frame_features(_frame(hand=_place_hand(pinched)), None)
    open_vec = frame_features(_frame(hand=open_hand), None)
    assert _feat(pinch_vec, "pinch_thumb_index") < _feat(open_vec, "pinch_thumb_index")
    assert _feat(pinch_vec, "pinch_thumb_index") < 0.15


def test_pinch_ratio_is_distance_invariant() -> None:
    """Measured against the hand's own size, not the frame's."""
    small = _frame(hand=_place_hand(_HAND_LOCAL, size=0.10))
    large = _frame(hand=_place_hand(_HAND_LOCAL, size=0.30))
    assert _feat(frame_features(small, None), "pinch_thumb_index") == pytest.approx(
        _feat(frame_features(large, None), "pinch_thumb_index"), abs=1e-3
    )


def test_curl_is_zero_for_a_straight_finger() -> None:
    assert _feat(frame_features(_frame(), None), "curl_index") == pytest.approx(0.0, abs=1e-3)


def test_curl_rises_when_a_finger_bends() -> None:
    bent = list(_HAND_LOCAL)
    bent[6] = (-0.10, -0.70, 0.00)
    bent[7] = (-0.12, -0.62, 0.00)
    bent[8] = (-0.14, -0.52, 0.00)   # tip folded back towards the knuckle
    vec = frame_features(_frame(hand=_place_hand(bent)), None)
    assert _feat(vec, "curl_index") > 0.3


def test_curl_is_bounded() -> None:
    """A fully folded finger must not run past 1, whatever the geometry."""
    folded = list(_HAND_LOCAL)
    for i in (6, 7, 8):
        folded[i] = (-0.10, -0.50, 0.00)   # every joint on the knuckle
    vec = frame_features(_frame(hand=_place_hand(folded)), None)
    assert 0.0 <= _feat(vec, "curl_index") <= 1.0


def test_spread_grows_when_the_fingers_fan_out() -> None:
    fanned = list(_HAND_LOCAL)
    fanned[20] = (0.80, -0.70, -0.02)   # pinky swung away from the index
    narrow = frame_features(_frame(hand=_place_hand(_HAND_LOCAL)), None)
    wide = frame_features(_frame(hand=_place_hand(fanned)), None)
    assert _feat(wide, "spread_index_pinky") > _feat(narrow, "spread_index_pinky")


def test_palm_angle_tracks_the_hand_turning_against_the_head() -> None:
    """Rolling the head is cancelled out; rolling only the hand is not."""
    upright = frame_features(_frame(hand=_place_hand(_HAND_LOCAL, angle=0.0)), None)
    turned = frame_features(_frame(hand=_place_hand(_HAND_LOCAL, angle=1.1)), None)
    assert abs(_feat(upright, "palm_angle_sin") - _feat(turned, "palm_angle_sin")) > 0.3


def test_palm_angle_is_continuous_across_the_wrap() -> None:
    """sin/cos instead of a raw angle, so ±pi is not a cliff in feature space."""
    a = frame_features(_frame(hand=_place_hand(_HAND_LOCAL, angle=math.pi - 0.02)), None)
    b = frame_features(_frame(hand=_place_hand(_HAND_LOCAL, angle=-math.pi + 0.02)), None)
    for name in ("palm_angle_sin", "palm_angle_cos"):
        assert abs(_feat(a, name) - _feat(b, name)) < 0.1


def test_hand_scale_grows_as_the_hand_approaches_the_camera() -> None:
    small = frame_features(_frame(hand=_place_hand(_HAND_LOCAL, size=0.10)), None)
    large = frame_features(_frame(hand=_place_hand(_HAND_LOCAL, size=0.20)), None)
    assert _feat(large, "hand_scale") > _feat(small, "hand_scale") * 1.5


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

def test_velocities_are_zero_without_a_previous_frame() -> None:
    vec = frame_features(_frame(), None)
    for name in FEATURE_NAMES:
        if name.startswith("vel_"):
            assert _feat(vec, name) == 0.0


def test_velocity_follows_the_motion() -> None:
    prev = _frame(t=0.0)
    cur = _frame(hand=_place_hand(_HAND_LOCAL, origin=(0.78, 0.50)), t=0.1)
    vec = frame_features(cur, prev)
    assert _feat(vec, "vel_wrist_x") > 1.0
    assert _feat(vec, "vel_wrist_y") == pytest.approx(0.0, abs=0.2)


def test_velocity_uses_elapsed_time_not_frame_count() -> None:
    """Measured fps drifts well below the configured 10, so dt is not constant."""
    prev = _frame(t=0.0)
    moved = _place_hand(_HAND_LOCAL, origin=(0.74, 0.50))
    slow = frame_features(_frame(hand=moved, t=0.2), prev)
    fast = frame_features(_frame(hand=moved, t=0.1), prev)
    assert _feat(fast, "vel_wrist_x") == pytest.approx(2 * _feat(slow, "vel_wrist_x"), rel=0.05)


def test_a_zero_timestep_does_not_divide_by_zero() -> None:
    prev = _frame(t=5.0)
    cur = _frame(hand=_place_hand(_HAND_LOCAL, origin=(0.78, 0.50)), t=5.0)
    assert np.all(np.isfinite(frame_features(cur, prev)))


def test_a_backwards_timestep_does_not_invert_the_motion() -> None:
    prev = _frame(t=5.0)
    cur = _frame(hand=_place_hand(_HAND_LOCAL, origin=(0.78, 0.50)), t=4.0)
    assert np.all(np.isfinite(frame_features(cur, prev)))


def test_velocity_is_clamped() -> None:
    """A dropped frame must not emit a spike that dominates the batch."""
    prev = _frame(t=0.0)
    cur = _frame(hand=_place_hand(_HAND_LOCAL, origin=(5.0, -4.0)), t=0.0005)
    vec = frame_features(cur, prev)
    assert np.all(np.isfinite(vec))
    for name in FEATURE_NAMES:
        if name.startswith("vel_"):
            assert abs(_feat(vec, name)) <= MAX_VELOCITY


def test_velocity_is_zero_when_the_previous_frame_had_no_hand() -> None:
    cur = _frame(hand=_place_hand(_HAND_LOCAL, origin=(0.78, 0.50)), t=0.1)
    vec = frame_features(cur, _frame(hand=False, t=0.0))
    assert _feat(vec, "vel_wrist_x") == 0.0


def test_a_rigidly_moving_hand_has_no_residual_finger_motion() -> None:
    """The feature that separates twisting from merely moving the hand."""
    prev = _frame(t=0.0)
    cur = _frame(hand=_place_hand(_HAND_LOCAL, origin=(0.76, 0.46)), t=0.1)
    assert _feat(frame_features(cur, prev), "fingertip_residual_speed") == pytest.approx(
        0.0, abs=0.05
    )


def test_fingers_moving_over_a_still_wrist_show_a_residual() -> None:
    rolling = list(_HAND_LOCAL)
    rolling[4] = (-0.10, -0.98, 0.00)   # thumb closes onto the index
    rolling[8] = (-0.30, -0.80, 0.00)   # index swings across
    prev = _frame(t=0.0)
    cur = _frame(hand=_place_hand(rolling), t=0.1)
    assert _feat(frame_features(cur, prev), "fingertip_residual_speed") > 0.3


def test_head_motion_does_not_count_as_hand_motion() -> None:
    """A nod moves the hand in the image but not relative to the head."""
    prev = _frame(t=0.0)
    cur = _transform(_frame(t=0.1), tx=0.05, ty=0.04)
    vec = frame_features(cur, prev)
    assert _feat(vec, "vel_wrist_x") == pytest.approx(0.0, abs=0.05)
    assert _feat(vec, "vel_wrist_y") == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# The hairline is deliberately unused
# ---------------------------------------------------------------------------

def test_the_hairline_is_not_a_feature() -> None:
    """Measured on the real dataset, the hairline is a trap, not a feature.

    The face landmarker finds it in 47% of frames, and it fails precisely
    when the head is turned — which is what reaching for a strand looks like.
    So its absence predicts twisting (AUC 0.57 on its own) while describing
    the detector rather than the gesture, and a model would happily learn
    "landmarker failed, therefore twisting". Where it *is* present it earns
    nothing either: mean fingertip-to-hairline distance is 0.69 head-widths
    for twisting against 0.68 for scratching, the one pair it was supposed to
    separate. This test exists so it does not get helpfully added back.
    """
    assert not any("forehead" in name for name in FEATURE_NAMES)


def test_a_forehead_line_changes_no_feature() -> None:
    line = [[0.45 + 0.01 * i, 0.40] for i in range(12)]
    np.testing.assert_array_equal(
        frame_features(_frame(forehead=line), None), frame_features(_frame(), None)
    )


def test_a_broken_forehead_line_is_still_ignored() -> None:
    """It is still written into every clip, so it must stay harmless."""
    for line in ([], [[float("nan"), 0.4]], "nonsense", [[0.4]]):
        assert np.all(np.isfinite(frame_features(_frame(forehead=line), None)))


# ---------------------------------------------------------------------------
# clip_features
# ---------------------------------------------------------------------------

def _clip(n: int, dx: float = 0.0) -> list[dict]:
    return [
        _frame(hand=_place_hand(_HAND_LOCAL, origin=(0.70 + dx * i, 0.50)), t=0.1 * i)
        for i in range(n)
    ]


def test_clip_features_shape() -> None:
    assert clip_features(_clip(20)).shape == (20, FEATURE_DIM)


def test_clip_features_dtype() -> None:
    assert clip_features(_clip(20)).dtype == np.float32


def test_clip_features_pads_a_short_clip() -> None:
    out = clip_features(_clip(5), seq_len=20)
    assert out.shape == (20, FEATURE_DIM)
    # Padding is an explicit "no data" row, not a repeat that invents motion.
    assert np.all(out[5:] == 0.0)


def test_clip_features_truncates_a_long_clip() -> None:
    out = clip_features(_clip(30), seq_len=20)
    assert out.shape == (20, FEATURE_DIM)
    np.testing.assert_allclose(out, clip_features(_clip(20), seq_len=20), atol=1e-6)


def test_clip_features_honours_a_custom_seq_len() -> None:
    assert clip_features(_clip(20), seq_len=8).shape == (8, FEATURE_DIM)


def test_clip_features_starts_with_zero_velocity() -> None:
    out = clip_features(_clip(20, dx=0.01))
    assert out[0, _index("vel_wrist_x")] == 0.0


def test_clip_features_chains_consecutive_frames() -> None:
    out = clip_features(_clip(20, dx=0.01))
    assert out[1, _index("vel_wrist_x")] > 0.1
    assert out[5, _index("vel_wrist_x")] > 0.1


def test_clip_features_of_an_empty_clip() -> None:
    out = clip_features([])
    assert out.shape == (20, FEATURE_DIM)
    assert np.all(out == 0.0)


def test_clip_features_is_finite_for_garbage() -> None:
    frames = [{}, {"hand": None}, {"pose": {}}, {"hand": [], "pose": None},
              {"t": "nonsense"}, _frame()]
    out = clip_features(frames)
    assert out.shape == (20, FEATURE_DIM)
    assert np.all(np.isfinite(out))


def test_clip_features_survives_a_gap_in_the_hand_track() -> None:
    """The hand detector loses the hand mid-clip often enough to matter."""
    frames = _clip(6)
    frames[3] = _frame(hand=False, t=0.3)
    out = clip_features(frames)
    assert np.all(np.isfinite(out))
    assert out[3, _index("hand_present")] == 0.0
    assert out[4, _index("hand_present")] == 1.0
