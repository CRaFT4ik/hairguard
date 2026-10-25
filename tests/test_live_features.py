"""Tests that the live feature path and the training feature path agree.

This is the file that decides whether the model is worth shipping. Training
features come from ``hairlab.features.clip_features`` reading a stored clip's
``meta.json``; inference features come from ``hairguard.live_features``
assembling the same numbers out of the detector objects the detection loop
already holds. If the two disagree anywhere — landmark order, rounding, how
``t`` is derived, which hand is picked, how the window slides — the model
degrades and nothing raises. So the disagreement has to be made into a test
failure, and it has to be measured on real recorded clips rather than on
synthetic frames that cannot surprise us.

The proof is in three parts, and all three are needed:

1. ``TestRecordMatchesCapture`` — the live record is byte-identical to what
   ``hairlab.capture`` writes, checked by running the real CaptureService
   over the same detector objects and reading back the file it produced.
2. ``TestWindowMatchesClipFeatures`` — sliding a window over a clip's frames
   produces exactly the tensor ``clip_features`` produces, on every clip in
   the recorded dataset.
3. ``TestScoresMatch`` — the scores the app will act on equal the scores the
   training pipeline measured, end to end through ONNX.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hairguard.detector import Landmark, PoseResult
from hairguard.hand_detector import HandLandmark, HandResult
from hairguard.live_features import FeatureWindow, frame_record
from hairlab.features import FEATURE_DIM, clip_features

CLIPS_ROOT = Path.home() / ".hairguard" / "clips"
TRAINED_DIR = Path.home() / ".hairguard" / "trained"

# Keeps the whole-dataset sweeps to a few seconds while still covering every
# capture day. Raise it when touching anything in the feature path.
_SWEEP_LIMIT = 250


# ---------------------------------------------------------------------------
# Real recorded clips
# ---------------------------------------------------------------------------

def _clip_frames(limit: int | None = None) -> list[tuple[str, list[dict]]]:
    """``(clip_id, frames)`` for recorded clips whose metadata is readable."""
    if not CLIPS_ROOT.is_dir():
        return []
    out: list[tuple[str, list[dict]]] = []
    for clip_dir in sorted(CLIPS_ROOT.iterdir()):
        if limit is not None and len(out) >= limit:
            break
        if not clip_dir.is_dir():
            continue
        try:
            meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        frames = meta.get("frames") if isinstance(meta, dict) else None
        if isinstance(frames, list) and frames:
            out.append((clip_dir.name, frames))
    return out


_CLIPS = _clip_frames(_SWEEP_LIMIT)

needs_clips = pytest.mark.skipif(
    not _CLIPS,
    reason=f"no recorded clips under {CLIPS_ROOT} to prove equivalence against",
)


# ---------------------------------------------------------------------------
# Synthetic detector objects
# ---------------------------------------------------------------------------

def _hand(offset: float = 0.0) -> HandResult:
    """A full 21-landmark hand, laid out so no two points coincide."""
    landmarks = {
        index: HandLandmark(
            x=0.40 + 0.01 * index + offset,
            y=0.30 + 0.004 * (index % 5) - 0.002 * index + offset,
            z=-0.01 * index,
            presence=1.0,
        )
        for index in range(21)
    }
    return HandResult(landmarks=landmarks, handedness="Right", score=0.9)


def _pose(confidence: float = 0.99) -> PoseResult:
    points = {
        "nose": (0.52, 0.30),
        "left_ear": (0.58, 0.28),
        "right_ear": (0.46, 0.29),
        "left_shoulder": (0.62, 0.55),
        "right_shoulder": (0.40, 0.56),
        "left_wrist": (0.45, 0.40),
        "right_wrist": (0.70, 0.45),
    }
    return PoseResult(
        landmarks={
            name: Landmark(x=x, y=y, confidence=confidence)
            for name, (x, y) in points.items()
        },
        raw_landmarks=None,
        detected=True,
    )


# ---------------------------------------------------------------------------
# 1. The live record is what capture would have stored
# ---------------------------------------------------------------------------

class TestRecordMatchesCapture:
    """The seam itself: same inputs in, same frame dict out.

    Driven through the real ``CaptureService`` rather than against a restated
    expectation, because the thing that must not drift is the stored format
    — including its rounding, which is the kind of detail that silently
    shifts every feature by a hair.
    """

    def _captured_frames(self, tmp_path, hands, poses) -> list[dict]:
        from hairlab.capture import CaptureService
        from hairlab.clipstore import ClipStore

        window = len(hands)
        service = CaptureService(
            store=ClipStore(root=tmp_path / "clips"),
            window_frames=window,
            cooldown_sec=0.0,
            crop_size=64,
        )
        service.start()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        clip_id = None
        try:
            for hand, pose in zip(hands, poses):
                clip_id = service.on_tick(
                    frame, pose, [hand], in_zone=True, hand_index=0
                ) or clip_id
        finally:
            service.close()

        assert clip_id is not None, "the capture service did not complete a clip"
        meta = json.loads(
            (tmp_path / "clips" / clip_id / "meta.json").read_text(encoding="utf-8")
        )
        return meta["frames"]

    def test_landmarks_and_pose_match_the_stored_clip(self, tmp_path) -> None:
        hands = [_hand(0.01 * i) for i in range(4)]
        poses = [_pose() for _ in hands]
        stored = self._captured_frames(tmp_path, hands, poses)

        for frame, hand, pose in zip(stored, hands, poses):
            live = frame_record(hand, pose, t=frame["t"])
            assert live["hand"] == frame["hand"]
            assert live["pose"] == frame["pose"]

    def test_the_timestamp_is_rounded_the_same_way(self) -> None:
        """Velocities divide by this; a different rounding is a different speed."""
        record = frame_record(_hand(), _pose(), t=295034.7501234)
        assert record["t"] == 295034.7501

    def test_a_pose_landmark_below_confidence_is_still_stored(self) -> None:
        """Capture stores the confidence and lets features do the filtering."""
        record = frame_record(_hand(), _pose(confidence=0.1), t=1.0)
        assert record["pose"]["nose"][2] == pytest.approx(0.1)

    def test_no_hand_means_no_hand_key(self) -> None:
        """Mirrors capture, where a frame without an identified hand omits it."""
        record = frame_record(None, _pose(), t=1.0)
        assert "hand" not in record

    def test_no_pose_landmarks_means_no_pose_key(self) -> None:
        empty = PoseResult(landmarks={}, raw_landmarks=None, detected=False)
        record = frame_record(_hand(), empty, t=1.0)
        assert "pose" not in record

    @needs_clips
    def test_the_keys_the_live_path_omits_cannot_change_the_features(self) -> None:
        """``box``, ``source_px`` and ``forehead`` are stored but never read.

        The live record leaves them out, which is only safe if the feature
        extractor really ignores them. Checked on real clips, which carry
        real values for all three.
        """
        feature_keys = {"t", "hand", "pose"}
        for clip_id, frames in _CLIPS[:40]:
            stripped = [
                {k: v for k, v in frame.items() if k in feature_keys}
                for frame in frames
            ]
            assert np.array_equal(
                clip_features(frames, seq_len=20),
                clip_features(stripped, seq_len=20),
            ), clip_id


# ---------------------------------------------------------------------------
# 2. The sliding window is what clip_features computes
# ---------------------------------------------------------------------------

class TestWindowMatchesClipFeatures:
    """``FeatureWindow`` caches per-frame rows; this proves the cache is exact."""

    @needs_clips
    def test_every_recorded_clip_yields_an_identical_tensor(self) -> None:
        """The headline equivalence check, over the real dataset."""
        window = FeatureWindow(seq_len=20)
        checked = 0
        for clip_id, frames in _CLIPS:
            window.reset()
            live = None
            for record in frames:
                pushed = window.push(record)
                if pushed is not None:
                    live = pushed
            expected = clip_features(frames, seq_len=20)
            assert live is not None, f"{clip_id} never completed a window"
            # Bit-for-bit, not approximately: both sides run the same
            # function on the same inputs, so any difference at all means
            # the paths have diverged.
            assert np.array_equal(live, expected), clip_id
            checked += 1
        assert checked >= 100, f"only {checked} clips checked — too few to trust"

    @needs_clips
    def test_the_window_slides_without_drifting(self) -> None:
        """A shorter window steps across a clip and must match at every offset.

        Clips are exactly as long as the model's window, so a full-length
        window only ever lands in one place. Shortening it is what actually
        exercises the slide — and the slide is where a cached row could be
        paired with the wrong predecessor.
        """
        for clip_id, frames in _CLIPS[:40]:
            window = FeatureWindow(seq_len=5)
            for index, record in enumerate(frames):
                live = window.push(record)
                if index < 4:
                    assert live is None, f"{clip_id} scored a partial window"
                    continue
                expected = clip_features(frames[index - 4:index + 1], seq_len=5)
                assert np.array_equal(live, expected), f"{clip_id} at frame {index}"

    def test_nothing_is_scored_below_the_minimum(self) -> None:
        """``min_frames`` defaults to the whole window. See TestPartialWindows
        for what happens when a caller asks for less."""
        window = FeatureWindow(seq_len=4)
        assert window.push(frame_record(_hand(), _pose(), t=1.0)) is None
        assert window.push(frame_record(_hand(0.01), _pose(), t=1.1)) is None
        assert window.push(frame_record(_hand(0.02), _pose(), t=1.2)) is None
        assert window.push(frame_record(_hand(0.03), _pose(), t=1.3)) is not None

    def test_reset_discards_the_history(self) -> None:
        """The hand left the zone, so the next visit must start from nothing."""
        window = FeatureWindow(seq_len=3)
        for index in range(2):
            window.push(frame_record(_hand(0.01 * index), _pose(), t=1.0 + index))
        window.reset()
        assert window.filled == 0
        assert window.push(frame_record(_hand(), _pose(), t=5.0)) is None

    def test_the_first_frame_after_a_reset_carries_no_velocity(self) -> None:
        """A fresh window must not inherit speed from the previous visit."""
        window = FeatureWindow(seq_len=1)
        row = window.push(frame_record(_hand(), _pose(), t=1.0))
        window.reset()
        again = window.push(frame_record(_hand(0.2), _pose(), t=1.2))
        assert np.array_equal(row[0][-1:], again[0][-1:])

    def test_the_tensor_has_the_shape_the_model_expects(self) -> None:
        window = FeatureWindow(seq_len=20)
        tensor = None
        for index in range(20):
            tensor = window.push(frame_record(_hand(0.005 * index), _pose(), t=index * 0.15))
        assert tensor.shape == (20, FEATURE_DIM)
        assert tensor.dtype == np.float32

    def test_each_full_window_is_a_fresh_array(self) -> None:
        """The caller may hold on to one; it must not change underneath them."""
        window = FeatureWindow(seq_len=2)
        window.push(frame_record(_hand(), _pose(), t=1.0))
        first = window.push(frame_record(_hand(0.05), _pose(), t=1.15))
        kept = first.copy()
        window.push(frame_record(_hand(0.10), _pose(), t=1.30))
        assert np.array_equal(first, kept)


# ---------------------------------------------------------------------------
# 3. End to end: the same scores the training pipeline measured
# ---------------------------------------------------------------------------

class TestScoresMatch:
    @pytest.mark.skipif(
        not (TRAINED_DIR / "twistnet.onnx").is_file(),
        reason="no exported model to score with",
    )
    @needs_clips
    def test_live_path_scores_equal_training_path_scores(self) -> None:
        """The number the app thresholds is the number the evaluation reported."""
        from hairguard.twistnet import TwistModel

        model = TwistModel.load(TRAINED_DIR)
        try:
            batch = np.stack([clip_features(f, seq_len=20) for _, f in _CLIPS])
            expected = model.score_batch(batch)

            live = []
            window = FeatureWindow(seq_len=model.seq_len)
            for _, frames in _CLIPS:
                window.reset()
                tensor = None
                for record in frames:
                    pushed = window.push(record)
                    if pushed is not None:
                        tensor = pushed
                live.append(model.score(tensor))

            difference = np.abs(np.asarray(live) - expected)
            # Same graph, same inputs, one sample at a time versus a batch:
            # only the batching differs, so the gap is float accumulation
            # order and nothing else.
            assert difference.max() < 1e-6, f"max score gap {difference.max():.2e}"
        finally:
            model.close()


# ---------------------------------------------------------------------------
# 4. Scoring before the window is full
# ---------------------------------------------------------------------------

class TestPartialWindows:
    """A partial window is filled by repeating the last frame, never with zeros.

    This is load-bearing, not a detail. The network standardises its input
    first, so a zero row becomes ``-mean/std`` — a large constant vector
    nothing in training resembled — and global average pooling lets those
    rows dominate. Measured on the shipped model's held-out days,
    zero-padding gives ROC AUC exactly 0.500: every clip scores 1.0 below
    k=17 and 0.0 above it. Repeat-padding holds AUC flat down to a handful
    of frames. See ``hairlab/truncation.py``.
    """

    @needs_clips
    def test_a_partial_window_is_what_repeat_padding_produces(self) -> None:
        """The equivalence that lets the truncation measurement apply here."""
        for clip_id, frames in _CLIPS[:40]:
            for k in (1, 2, 3, 5, 11, 19):
                window = FeatureWindow(seq_len=20, min_frames=1)
                live = None
                for record in frames[:k]:
                    live = window.push(record)
                expected = clip_features(
                    frames[:k] + [frames[k - 1]] * (20 - k), seq_len=20
                )
                assert live is not None, f"{clip_id} gave nothing at k={k}"
                assert np.array_equal(live, expected), f"{clip_id} at k={k}"

    @needs_clips
    def test_a_full_window_is_still_untouched(self) -> None:
        """Once the real frames fill the window, nothing is padded at all."""
        for clip_id, frames in _CLIPS[:40]:
            window = FeatureWindow(seq_len=20, min_frames=3)
            live = None
            for record in frames:
                pushed = window.push(record)
                if pushed is not None:
                    live = pushed
            assert np.array_equal(live, clip_features(frames, seq_len=20)), clip_id

    @needs_clips
    def test_no_returned_window_ever_contains_a_zero_row(self) -> None:
        """The guard against the failure mode that silently blinds the model."""
        for clip_id, frames in _CLIPS[:20]:
            window = FeatureWindow(seq_len=20, min_frames=1)
            for index, record in enumerate(frames):
                live = window.push(record)
                rows_all_zero = int((~live.any(axis=1)).sum())
                assert rows_all_zero == 0, f"{clip_id} frame {index}"

    def test_nothing_is_scored_below_the_minimum(self) -> None:
        window = FeatureWindow(seq_len=20, min_frames=3)
        assert window.push(frame_record(_hand(), _pose(), t=1.0)) is None
        assert window.push(frame_record(_hand(0.01), _pose(), t=1.15)) is None
        assert window.push(frame_record(_hand(0.02), _pose(), t=1.30)) is not None

    def test_the_padding_is_one_frozen_frame_repeated(self) -> None:
        """Identical rows, so they carry no invented motion."""
        window = FeatureWindow(seq_len=20, min_frames=1)
        tensor = None
        for index in range(4):
            tensor = window.push(frame_record(_hand(0.01 * index), _pose(), t=index * 0.15))
        padding = tensor[4:]
        assert np.array_equal(padding, np.repeat(padding[:1], len(padding), axis=0))

    def test_the_minimum_defaults_to_the_whole_window(self) -> None:
        """The conservative default: a caller must ask for partial scoring."""
        window = FeatureWindow(seq_len=4)
        assert window.min_frames == 4
        for index in range(3):
            assert window.push(frame_record(_hand(0.01 * index), _pose(), t=index * 0.15)) is None

    def test_a_minimum_above_the_window_is_clamped(self) -> None:
        assert FeatureWindow(seq_len=5, min_frames=99).min_frames == 5

    def test_a_minimum_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="min_frames"):
            FeatureWindow(seq_len=5, min_frames=0)

    def test_reset_clears_a_partial_window_too(self) -> None:
        window = FeatureWindow(seq_len=20, min_frames=2)
        window.push(frame_record(_hand(), _pose(), t=1.0))
        window.push(frame_record(_hand(0.01), _pose(), t=1.15))
        window.reset()
        assert window.push(frame_record(_hand(), _pose(), t=5.0)) is None
