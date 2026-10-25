"""hairlab.dataset — labelled clips assembled into training arrays.

One clip is one sample: a (SEQ_LEN, FEATURE_DIM) sequence of head-relative
landmark features and a binary target, twisting against everything else.

The negatives are the point of the dataset. Clips are only captured when a
hand is already at the head, so "not twisting" here means resting, scratching
or holding a comb — the cases a geometric rule cannot separate and the reason
a model is being trained at all. ``sublabels`` keeps the original verdict so
a confusion matrix can say which of them the model actually struggles with.

Two things are deliberate.

*Nothing here writes.* The label database is opened read-only; it is the only
copy of several hundred hand-labelled clips and no amount of convenience is
worth risking it. ``sqlite3.connect`` silently creates an empty database file
when handed a path that does not exist, which would turn a typo in a path
into a mysteriously empty dataset, so a missing file is checked for first.

*Partial data is normal.* The dataset is collected and pruned by hand over
weeks, so there are labels for clips long since deleted, clip directories
whose metadata was never written, and clips the hand detector lost halfway
through. All of those are skipped or kept on their merits rather than
allowed to abort a load.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from hairlab.capture import LABEL_IDS
from hairlab.clipstore import DEFAULT_LABEL_DB, DEFAULT_ROOT
from hairlab.features import FEATURE_DIM, clip_features

SEQ_LEN = 20
POSITIVE_LABEL = "twisting"
DROPPED_LABELS = ("unclear",)

# Re-exported under a clearer name: this module juggles two defaults, and
# ``DEFAULT_ROOT`` alone does not say which of them it is.
DEFAULT_CLIPS_ROOT: Path = DEFAULT_ROOT

# Derived from the labelling UI's taxonomy rather than restated, so a label
# added to the buttons becomes a training negative automatically instead of
# being silently dropped here.
KEPT_LABELS: tuple[str, ...] = tuple(
    label for label in LABEL_IDS if label not in DROPPED_LABELS
)

_META_NAME = "meta.json"
# Clip ids are UTC timestamps, so the first eight characters are the capture
# date. Clips from one sitting show the same hair, light and camera angle, so
# splitting without grouping by day leaks the test set into training.
_GROUP_CHARS = 8


@dataclass
class ClipDataset:
    """Feature tensors plus everything needed to split and audit them."""

    X: np.ndarray          # (N, SEQ_LEN, FEATURE_DIM) float32
    y: np.ndarray          # (N,) int64 — 1 for twisting, 0 otherwise
    sublabels: np.ndarray  # (N,) str — the original label, for error analysis
    groups: np.ndarray     # (N,) str — capture day, for grouped splitting
    clip_ids: np.ndarray   # (N,) str

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _read_labels(labels_db: Path) -> dict[str, str]:
    """Every stored verdict, read without touching the file."""
    if not labels_db.is_file():
        return {}
    uri = f"{labels_db.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        rows = conn.execute("SELECT clip_id, label FROM labels").fetchall()
    return {
        str(clip_id): str(label)
        for clip_id, label in rows
        if clip_id is not None and label is not None
    }


def _read_frames(clip_dir: Path) -> Optional[list[dict]]:
    """The clip's landmark frames, or None if it cannot be trusted.

    ``meta.json`` is written after the images, so its absence means capture
    was interrupted. Its presence is the only signal that a clip is complete.
    """
    try:
        meta: Any = json.loads((clip_dir / _META_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    frames = meta.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    # A frame that is not a dict would be read as "no landmarks" rather than
    # crashing feature extraction, which is the right call for one bad frame
    # inside an otherwise good clip.
    return frames


def load_dataset(
    clips_root: Path | str | None = None,
    labels_db: Path | str | None = None,
    seq_len: int = SEQ_LEN,
) -> ClipDataset:
    """Load every usable labelled clip, ordered by clip id.

    Clips that are unlabelled, labelled ``unclear``, labelled with something
    outside the taxonomy, or whose metadata is missing or corrupt are left
    out. A missing clips directory or label database yields an empty dataset
    with the right shape rather than an error: on a fresh machine there is
    simply nothing labelled yet.

    The row order is sorted by clip id — which, ids being timestamps, is also
    chronological — so a seeded split is reproducible between runs.
    """
    root = Path(clips_root) if clips_root is not None else DEFAULT_CLIPS_ROOT
    db_path = Path(labels_db) if labels_db is not None else DEFAULT_LABEL_DB

    labels = _read_labels(db_path)

    sequences: list[np.ndarray] = []
    targets: list[int] = []
    sublabels: list[str] = []
    groups: list[str] = []
    clip_ids: list[str] = []

    for clip_id in sorted(labels):
        label = labels[clip_id]
        if label not in KEPT_LABELS:
            continue
        # A crafted id must not reach outside the clips directory.
        clip_dir = root / Path(clip_id).name
        if not clip_dir.is_dir():
            continue
        frames = _read_frames(clip_dir)
        if frames is None:
            continue

        sequences.append(clip_features(frames, seq_len=seq_len))
        targets.append(1 if label == POSITIVE_LABEL else 0)
        sublabels.append(label)
        groups.append(clip_id[:_GROUP_CHARS])
        clip_ids.append(clip_id)

    if sequences:
        X = np.stack(sequences).astype(np.float32, copy=False)
    else:
        X = np.zeros((0, max(seq_len, 0), FEATURE_DIM), dtype=np.float32)

    return ClipDataset(
        X=X,
        y=np.asarray(targets, dtype=np.int64),
        sublabels=np.asarray(sublabels, dtype=np.str_),
        groups=np.asarray(groups, dtype=np.str_),
        clip_ids=np.asarray(clip_ids, dtype=np.str_),
    )
