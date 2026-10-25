"""Tests for hairlab.clipstore — clips on disk and labels in SQLite.

Everything runs against a tmp_path, so the user's real clip directory is
never touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from hairlab.clipstore import ClipStore, LabelStore

FRAME_SHAPE = (16, 16, 3)


def _frame(value: int) -> np.ndarray:
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


@pytest.fixture
def store(tmp_path) -> ClipStore:
    return ClipStore(root=tmp_path / "clips")


@pytest.fixture
def labels(tmp_path) -> LabelStore:
    return LabelStore(db_path=tmp_path / "labels.db")


# ---------------------------------------------------------------------------
# ClipStore
# ---------------------------------------------------------------------------

def test_saved_clip_appears_in_the_listing(store: ClipStore) -> None:
    clip_id = store.save([_frame(10), _frame(20)], {"fps": 10})
    assert store.clip_ids() == [clip_id]


def test_frames_survive_the_round_trip(store: ClipStore) -> None:
    clip_id = store.save([_frame(10), _frame(200)], {})
    assert store.frame_count(clip_id) == 2
    first = store.read_frame(clip_id, 0)
    assert first.shape == FRAME_SHAPE
    # JPEG is lossy; a flat patch survives well within this tolerance.
    assert abs(int(first.mean()) - 10) <= 3


def test_frames_keep_their_order(store: ClipStore) -> None:
    clip_id = store.save([_frame(0), _frame(120), _frame(240)], {})
    means = [int(store.read_frame(clip_id, i).mean()) for i in range(3)]
    assert means[0] < means[1] < means[2]


def test_metadata_round_trips(store: ClipStore) -> None:
    clip_id = store.save([_frame(1)], {"fps": 10, "camera": 1})
    meta = store.read_meta(clip_id)
    assert meta["fps"] == 10
    assert meta["camera"] == 1


def test_metadata_records_the_frame_count(store: ClipStore) -> None:
    clip_id = store.save([_frame(1), _frame(2), _frame(3)], {})
    assert store.read_meta(clip_id)["frame_count"] == 3


def test_consecutive_saves_get_distinct_ids(store: ClipStore) -> None:
    first = store.save([_frame(1)], {})
    second = store.save([_frame(2)], {})
    assert first != second
    assert sorted(store.clip_ids()) == sorted([first, second])


def test_deleting_a_clip_removes_it(store: ClipStore) -> None:
    clip_id = store.save([_frame(1)], {})
    store.delete(clip_id)
    assert store.clip_ids() == []


def test_empty_store_lists_nothing(store: ClipStore) -> None:
    assert store.clip_ids() == []


# ---------------------------------------------------------------------------
# LabelStore
# ---------------------------------------------------------------------------

def test_label_round_trips(labels: LabelStore) -> None:
    labels.set("clip1", "twisting")
    assert labels.get("clip1") == "twisting"


def test_unlabelled_clip_returns_none(labels: LabelStore) -> None:
    assert labels.get("nope") is None


def test_relabelling_replaces_the_previous_verdict(labels: LabelStore) -> None:
    labels.set("clip1", "twisting")
    labels.set("clip1", "scratching")
    assert labels.get("clip1") == "scratching"
    assert labels.counts() == {"scratching": 1}


def test_counts_group_by_label(labels: LabelStore) -> None:
    labels.set("a", "twisting")
    labels.set("b", "twisting")
    labels.set("c", "resting")
    assert labels.counts() == {"twisting": 2, "resting": 1}


def test_unset_removes_a_label(labels: LabelStore) -> None:
    labels.set("a", "twisting")
    labels.unset("a")
    assert labels.get("a") is None


def test_last_labelled_is_reported_for_undo(labels: LabelStore) -> None:
    labels.set("a", "twisting")
    labels.set("b", "resting")
    assert labels.last_labelled() == "b"


def test_last_labelled_is_none_on_empty_store(labels: LabelStore) -> None:
    assert labels.last_labelled() is None


def test_labels_persist_across_instances(tmp_path) -> None:
    path = tmp_path / "labels.db"
    LabelStore(db_path=path).set("a", "twisting")
    assert LabelStore(db_path=path).get("a") == "twisting"
