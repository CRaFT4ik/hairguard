"""Tests for the truncation experiment's window assembly.

``padded_windows`` is the whole experiment. It answers "what would the model
have seen k frames into a visit?", and it answers it by indexing cached
feature rows rather than by re-extracting features — which is what makes a
sweep over every k and every padding scheme affordable, and also what makes
an off-by-one here able to answer a *different* question without anything
looking wrong.

So the indexing is checked against ``clip_features`` applied to the
correspondingly padded frame list, on real recorded clips, for every scheme
and a spread of k. That is the same standard the live feature path is held
to in ``tests/test_live_features.py``, and for the same reason: a
measurement nobody can trust is worse than no measurement, because decisions
get made on it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hairlab.features import FEATURE_DIM, clip_features
from hairlab.truncation import (
    MEASURED_FPS,
    SCHEMES,
    cache_rows,
    clip_rows,
    padded_windows,
    seconds_in,
)

CLIPS_ROOT = Path.home() / ".hairguard" / "clips"
SEQ = 20


def _clips(limit: int) -> list[tuple[str, list[dict]]]:
    if not CLIPS_ROOT.is_dir():
        return []
    out: list[tuple[str, list[dict]]] = []
    for clip_dir in sorted(CLIPS_ROOT.iterdir()):
        if len(out) >= limit:
            break
        if not clip_dir.is_dir():
            continue
        try:
            meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        frames = meta.get("frames") if isinstance(meta, dict) else None
        if isinstance(frames, list) and len(frames) >= SEQ:
            out.append((clip_dir.name, frames))
    return out


_CLIPS = _clips(30)

needs_clips = pytest.mark.skipif(
    not _CLIPS, reason=f"no recorded clips under {CLIPS_ROOT}"
)


def _pad(frames: list[dict], k: int, scheme: str) -> list[dict]:
    """The frame list each scheme is claiming to be equivalent to."""
    if scheme == "zero-end":
        return frames[:k]
    if scheme == "repeat-last":
        return frames[:k] + [frames[k - 1]] * (SEQ - k)
    return [frames[0]] * (SEQ - k) + frames[:k]


# ---------------------------------------------------------------------------
# The assembly is what it claims to be
# ---------------------------------------------------------------------------

@needs_clips
class TestWindowsMatchClipFeatures:
    @pytest.fixture(scope="class")
    def rows(self):
        return cache_rows([cid for cid, _ in _CLIPS], CLIPS_ROOT, seq_len=SEQ)

    @pytest.mark.parametrize("scheme", SCHEMES)
    @pytest.mark.parametrize("k", [1, 2, 3, 5, 10, 19, 20])
    def test_every_scheme_at_every_k(self, rows, scheme: str, k: int) -> None:
        built = padded_windows(rows, k, scheme)
        for row, (clip_id, frames) in enumerate(_CLIPS):
            expected = clip_features(_pad(frames, k, scheme), seq_len=SEQ)
            assert np.array_equal(built[row], expected), f"{scheme} k={k} {clip_id}"

    def test_a_full_window_is_the_untouched_clip(self, rows) -> None:
        """Whatever the scheme, k == seq_len must leave nothing padded."""
        for scheme in SCHEMES:
            built = padded_windows(rows, SEQ, scheme)
            for row, (clip_id, frames) in enumerate(_CLIPS):
                assert np.array_equal(
                    built[row], clip_features(frames, seq_len=SEQ)
                ), f"{scheme} {clip_id}"

    def test_zero_padding_really_does_leave_zeros(self, rows) -> None:
        """The thing that destroys the model, pinned down so it stays visible."""
        built = padded_windows(rows, 6, "zero-end")
        assert np.count_nonzero(built[:, 6:]) == 0

    def test_repeat_last_pads_with_a_frozen_hand(self, rows) -> None:
        """Identical consecutive frames, so the padded rows carry no velocity."""
        built = padded_windows(rows, 6, "repeat-last")
        for row in range(len(_CLIPS)):
            block = built[row, 6:]
            assert np.array_equal(block, np.repeat(block[:1], len(block), axis=0))
        assert np.count_nonzero(built[:, 6:]) > 0


# ---------------------------------------------------------------------------
# Row caching
# ---------------------------------------------------------------------------

@needs_clips
def test_clip_rows_reports_the_frames_it_found() -> None:
    _, frames = _CLIPS[0]
    _, _, _, count = clip_rows(frames, seq_len=SEQ)
    assert count == SEQ


@needs_clips
def test_frozen_rows_differ_from_real_ones() -> None:
    """Otherwise the padding would be indistinguishable from real motion."""
    _, frames = _CLIPS[0]
    _, A, S, _ = clip_rows(frames, seq_len=SEQ)
    assert not np.array_equal(A[5], S[5])


# ---------------------------------------------------------------------------
# Guards and arithmetic
# ---------------------------------------------------------------------------

def _tiny_rows():
    from hairlab.truncation import Rows

    shape = (2, SEQ, FEATURE_DIM)
    return Rows(
        np.zeros((2, FEATURE_DIM), dtype=np.float32),
        np.ones(shape, dtype=np.float32),
        np.full(shape, 2.0, dtype=np.float32),
        np.full(2, SEQ, dtype=np.int64),
    )


def test_an_unknown_scheme_is_refused() -> None:
    with pytest.raises(ValueError, match="scheme"):
        padded_windows(_tiny_rows(), 5, "make-something-up")


@pytest.mark.parametrize("k", [0, -1, 21])
def test_k_outside_the_window_is_refused(k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        padded_windows(_tiny_rows(), k, "repeat-last")


def test_the_first_frame_costs_no_waiting() -> None:
    assert seconds_in(1) == 0.0


def test_a_full_window_is_about_three_seconds() -> None:
    """The number the whole experiment exists to question."""
    assert seconds_in(20, MEASURED_FPS) == pytest.approx(2.81, abs=0.02)
