"""Tests for hairlab.dataset — labelled clips assembled into arrays.

Everything here runs against clips written into ``tmp_path``. The real
dataset in ``~/.hairguard`` is the only copy of several hundred hand-labelled
clips, so no test is allowed near it, and the loader is checked to be
read-only even for the label database it is pointed at.

The loader's real job is tolerating a dataset that has been collected and
relabelled by hand for weeks: half-written clips, labels for clips since
deleted, directories with no metadata.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from hairlab.dataset import (
    DROPPED_LABELS,
    POSITIVE_LABEL,
    SEQ_LEN,
    ClipDataset,
    load_dataset,
)
from hairlab.features import FEATURE_DIM, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Synthetic clips
# ---------------------------------------------------------------------------

_POSE = {
    "nose": [0.540, 0.727, 0.999],
    "left_ear": [0.520, 0.712, 0.998],
    "right_ear": [0.474, 0.710, 0.999],
    "left_shoulder": [0.572, 0.919, 0.988],
    "right_shoulder": [0.369, 0.929, 0.988],
}


def _frames(count: int = 20, drift: float = 0.004) -> list[dict]:
    """A plausible clip: a hand near the head, moving a little per frame."""
    out = []
    for i in range(count):
        shift = drift * i
        out.append({
            "t": 1000.0 + 0.15 * i,
            "box": [496, 411, 792, 707],
            "source_px": 296,
            "hand_index": 0,
            "hand": [
                [0.47 + shift + 0.01 * j, 0.89 - 0.015 * j, 0.001 * j]
                for j in range(21)
            ],
            "pose": _POSE,
        })
    return out


def _write_clip(root: Path, clip_id: str, frames: list[dict] | None = None) -> None:
    clip_dir = root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "clip_id": clip_id,
        "frame_count": len(frames if frames is not None else _frames()),
        "created_at": "2026-09-19T00:29:52+00:00",
        "crop_size": 256,
        "duration_sec": 2.89,
        "fps_configured": 10.0,
        "fps_measured": 6.574,
        "frames": _frames() if frames is None else frames,
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # Frames exist on disk but the loader must never need them.
    (clip_dir / "frame_000.jpg").write_bytes(b"not really a jpeg")


def _write_labels(db_path: Path, labels: dict[str, str]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                clip_id     TEXT PRIMARY KEY,
                label       TEXT NOT NULL,
                labelled_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO labels (clip_id, label, labelled_at) VALUES (?, ?, ?)",
            [(cid, label, "2026-09-20T10:00:00+00:00") for cid, label in labels.items()],
        )


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "clips", tmp_path / "labels.db"


def _build(dataset_dir: tuple[Path, Path], labels: dict[str, str], **kwargs) -> ClipDataset:
    clips_root, db_path = dataset_dir
    for clip_id in labels:
        _write_clip(clips_root, clip_id)
    _write_labels(db_path, labels)
    return load_dataset(clips_root=clips_root, labels_db=db_path, **kwargs)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_sequence_length_matches_the_capture_window() -> None:
    """Clips are 20 frames; a mismatch here silently pads every sample."""
    assert SEQ_LEN == 20


def test_the_positive_class_is_twisting() -> None:
    assert POSITIVE_LABEL == "twisting"


def test_unclear_is_dropped_by_default() -> None:
    assert "unclear" in DROPPED_LABELS


# ---------------------------------------------------------------------------
# Shapes and types
# ---------------------------------------------------------------------------

def test_shapes(dataset_dir) -> None:
    data = _build(dataset_dir, {"20260918T0001_0": "twisting", "20260918T0002_0": "resting"})
    assert data.X.shape == (2, SEQ_LEN, FEATURE_DIM)
    assert data.y.shape == (2,)
    assert data.sublabels.shape == (2,)
    assert data.groups.shape == (2,)
    assert data.clip_ids.shape == (2,)


def test_dtypes(dataset_dir) -> None:
    data = _build(dataset_dir, {"20260918T0001_0": "twisting"})
    assert data.X.dtype == np.float32
    assert data.y.dtype == np.int64


def test_x_is_finite(dataset_dir) -> None:
    data = _build(dataset_dir, {"20260918T0001_0": "twisting", "20260919T0002_0": "object"})
    assert np.all(np.isfinite(data.X))


def test_a_custom_seq_len_is_honoured(dataset_dir) -> None:
    data = _build(dataset_dir, {"20260918T0001_0": "twisting"}, seq_len=12)
    assert data.X.shape == (1, 12, FEATURE_DIM)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def test_twisting_is_the_positive_class(dataset_dir) -> None:
    data = _build(dataset_dir, {"20260918T0001_0": "twisting"})
    assert data.y.tolist() == [1]


@pytest.mark.parametrize("label", ["resting", "scratching", "object"])
def test_every_other_kept_label_is_negative(dataset_dir, label: str) -> None:
    """Hand at the head but not twisting — the hard negatives."""
    data = _build(dataset_dir, {"20260918T0001_0": label})
    assert data.y.tolist() == [0]


def test_sublabels_keep_the_original_label(dataset_dir) -> None:
    """Needed to tell "called scratching a twist" from "called resting a twist"."""
    data = _build(dataset_dir, {
        "20260918T0001_0": "twisting",
        "20260918T0002_0": "scratching",
        "20260918T0003_0": "object",
    })
    assert sorted(data.sublabels.tolist()) == ["object", "scratching", "twisting"]


def test_sublabels_line_up_with_the_labels(dataset_dir) -> None:
    data = _build(dataset_dir, {
        "20260918T0001_0": "twisting",
        "20260918T0002_0": "resting",
    })
    for sub, y in zip(data.sublabels.tolist(), data.y.tolist()):
        assert y == (1 if sub == POSITIVE_LABEL else 0)


# ---------------------------------------------------------------------------
# What gets dropped
# ---------------------------------------------------------------------------

def test_unclear_clips_are_dropped(dataset_dir) -> None:
    data = _build(dataset_dir, {
        "20260918T0001_0": "twisting",
        "20260918T0002_0": "unclear",
    })
    assert data.clip_ids.tolist() == ["20260918T0001_0"]


def test_unlabelled_clips_are_dropped(dataset_dir) -> None:
    clips_root, db_path = dataset_dir
    _write_clip(clips_root, "20260918T0001_0")
    _write_clip(clips_root, "20260918T0002_0")   # never labelled
    _write_labels(db_path, {"20260918T0001_0": "twisting"})
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    assert data.clip_ids.tolist() == ["20260918T0001_0"]


def test_a_label_for_a_deleted_clip_is_ignored(dataset_dir) -> None:
    """Clips get pruned from disk; their rows stay in the database."""
    clips_root, db_path = dataset_dir
    _write_clip(clips_root, "20260918T0001_0")
    _write_labels(db_path, {
        "20260918T0001_0": "twisting",
        "20260918T9999_0": "resting",   # no such directory
    })
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    assert data.clip_ids.tolist() == ["20260918T0001_0"]


def test_corrupt_metadata_is_skipped(dataset_dir) -> None:
    clips_root, db_path = dataset_dir
    _write_clip(clips_root, "20260918T0001_0")
    broken = clips_root / "20260918T0002_0"
    broken.mkdir(parents=True)
    (broken / "meta.json").write_text("{not json at all", encoding="utf-8")
    _write_labels(db_path, {"20260918T0001_0": "twisting", "20260918T0002_0": "twisting"})
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    assert data.clip_ids.tolist() == ["20260918T0001_0"]


def test_a_clip_directory_without_metadata_is_skipped(dataset_dir) -> None:
    """meta.json is written last, so its absence means a half-written clip."""
    clips_root, db_path = dataset_dir
    _write_clip(clips_root, "20260918T0001_0")
    (clips_root / "20260918T0002_0").mkdir(parents=True)
    _write_labels(db_path, {"20260918T0001_0": "twisting", "20260918T0002_0": "twisting"})
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    assert data.clip_ids.tolist() == ["20260918T0001_0"]


def test_a_clip_with_no_frames_is_skipped(dataset_dir) -> None:
    clips_root, db_path = dataset_dir
    _write_clip(clips_root, "20260918T0001_0")
    _write_clip(clips_root, "20260918T0002_0", frames=[])
    _write_labels(db_path, {"20260918T0001_0": "twisting", "20260918T0002_0": "twisting"})
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    assert data.clip_ids.tolist() == ["20260918T0001_0"]


def test_a_clip_whose_hand_was_never_found_is_kept(dataset_dir) -> None:
    """The human labelled it from the pixels; a poor hand track is signal too."""
    clips_root, db_path = dataset_dir
    frames = [{"t": 1000.0 + 0.15 * i, "pose": _POSE} for i in range(20)]
    _write_clip(clips_root, "20260918T0001_0", frames=frames)
    _write_labels(db_path, {"20260918T0001_0": "twisting"})
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    assert data.clip_ids.tolist() == ["20260918T0001_0"]
    assert np.all(data.X[0, :, FEATURE_NAMES.index("hand_present")] == 0.0)


def test_garbage_inside_the_frames_does_not_crash(dataset_dir) -> None:
    clips_root, db_path = dataset_dir
    frames = [{"t": "nonsense"}, {"hand": None}, {"pose": {}}, {}]
    _write_clip(clips_root, "20260918T0001_0", frames=frames)
    _write_labels(db_path, {"20260918T0001_0": "twisting"})
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    assert data.X.shape == (1, SEQ_LEN, FEATURE_DIM)
    assert np.all(np.isfinite(data.X))


def test_an_unknown_label_is_dropped(dataset_dir) -> None:
    """The taxonomy may grow; an unrecognised value must not become a negative."""
    data = _build(dataset_dir, {
        "20260918T0001_0": "twisting",
        "20260918T0002_0": "some_future_label",
    })
    assert data.clip_ids.tolist() == ["20260918T0001_0"]


# ---------------------------------------------------------------------------
# Grouping and ordering
# ---------------------------------------------------------------------------

def test_groups_are_the_capture_day(dataset_dir) -> None:
    """Clips from one sitting are near-duplicates, so splits must group by day."""
    data = _build(dataset_dir, {
        "20260918T230139_077998": "twisting",
        "20260919T003725_086605": "resting",
    })
    assert sorted(data.groups.tolist()) == ["20260918", "20260919"]


def test_clips_from_the_same_day_share_a_group(dataset_dir) -> None:
    data = _build(dataset_dir, {
        "20260918T230139_077998": "twisting",
        "20260918T230146_984035": "resting",
    })
    assert len(set(data.groups.tolist())) == 1


def test_rows_are_ordered_by_clip_id(dataset_dir) -> None:
    """A stable order makes a seeded split reproducible across runs."""
    data = _build(dataset_dir, {
        "20260919T0003_0": "twisting",
        "20260918T0001_0": "resting",
        "20260918T0002_0": "object",
    })
    assert data.clip_ids.tolist() == sorted(data.clip_ids.tolist())


def test_rows_stay_aligned_across_every_array(dataset_dir) -> None:
    clips_root, db_path = dataset_dir
    labels = {f"2026091{d}T000{i}_0": ("twisting" if i % 2 else "resting")
              for d in (8, 9) for i in range(3)}
    for clip_id in labels:
        _write_clip(clips_root, clip_id)
    _write_labels(db_path, labels)
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    for i, clip_id in enumerate(data.clip_ids.tolist()):
        assert data.groups[i] == clip_id[:8]
        assert data.sublabels[i] == labels[clip_id]
        assert data.y[i] == (1 if labels[clip_id] == "twisting" else 0)


# ---------------------------------------------------------------------------
# Empty and missing inputs
# ---------------------------------------------------------------------------

def test_no_labels_yields_an_empty_but_well_shaped_dataset(dataset_dir) -> None:
    """Callers do ``data.X.shape[-1]``, so the feature axis must survive."""
    clips_root, db_path = dataset_dir
    _write_clip(clips_root, "20260918T0001_0")
    _write_labels(db_path, {})
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    assert data.X.shape == (0, SEQ_LEN, FEATURE_DIM)
    assert data.y.shape == (0,)
    assert data.X.dtype == np.float32
    assert data.y.dtype == np.int64


def test_a_missing_clips_root_yields_an_empty_dataset(dataset_dir) -> None:
    clips_root, db_path = dataset_dir
    _write_labels(db_path, {"20260918T0001_0": "twisting"})
    data = load_dataset(clips_root=clips_root / "nowhere", labels_db=db_path)
    assert data.X.shape == (0, SEQ_LEN, FEATURE_DIM)


def test_a_missing_label_database_yields_an_empty_dataset(dataset_dir) -> None:
    clips_root, db_path = dataset_dir
    _write_clip(clips_root, "20260918T0001_0")
    data = load_dataset(clips_root=clips_root, labels_db=db_path)
    assert data.X.shape == (0, SEQ_LEN, FEATURE_DIM)


def test_loading_never_creates_the_label_database(dataset_dir) -> None:
    """sqlite3.connect happily creates an empty file; the loader must not."""
    clips_root, db_path = dataset_dir
    _write_clip(clips_root, "20260918T0001_0")
    load_dataset(clips_root=clips_root, labels_db=db_path)
    assert not db_path.exists()


def test_loading_does_not_modify_the_label_database(dataset_dir) -> None:
    clips_root, db_path = dataset_dir
    _write_clip(clips_root, "20260918T0001_0")
    _write_labels(db_path, {"20260918T0001_0": "twisting"})
    before = db_path.read_bytes()
    load_dataset(clips_root=clips_root, labels_db=db_path)
    assert db_path.read_bytes() == before


def test_the_defaults_point_at_the_app_data_directory() -> None:
    """Called with no arguments it must find the real dataset, not cwd."""
    from hairlab.dataset import DEFAULT_CLIPS_ROOT, DEFAULT_LABEL_DB

    assert DEFAULT_CLIPS_ROOT == Path.home() / ".hairguard" / "clips"
    assert DEFAULT_LABEL_DB == Path.home() / ".hairguard" / "labels.db"
