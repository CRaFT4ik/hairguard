"""Tests for the labelling API.

Each test drives a real FastAPI app over a temporary clip store, so the
request/response contract the UI depends on is exercised end to end.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hairlab.clipstore import ClipStore, LabelStore
from hairlab.server.app import create_app


@pytest.fixture
def store(tmp_path) -> ClipStore:
    return ClipStore(root=tmp_path / "clips")


@pytest.fixture
def labels(tmp_path) -> LabelStore:
    return LabelStore(db_path=tmp_path / "labels.db")


@pytest.fixture
def client(store: ClipStore, labels: LabelStore) -> TestClient:
    return TestClient(create_app(store, labels))


def _add_clip(store: ClipStore, frames: int = 3) -> str:
    return store.save(
        [np.full((8, 8, 3), 40 + i * 20, dtype=np.uint8) for i in range(frames)],
        {"fps": 10},
    )


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def test_label_definitions_are_served(client: TestClient) -> None:
    body = client.get("/api/labels/definitions").json()
    ids = [item["id"] for item in body["labels"]]
    assert "twisting" in ids
    assert "unclear" in ids


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

def test_queue_is_empty_without_clips(client: TestClient) -> None:
    body = client.get("/api/queue").json()
    assert body["clips"] == []
    assert body["remaining"] == 0


def test_queue_offers_an_unlabelled_clip(client: TestClient, store: ClipStore) -> None:
    clip_id = _add_clip(store)
    body = client.get("/api/queue").json()
    assert [c["clip_id"] for c in body["clips"]] == [clip_id]
    assert body["clips"][0]["frame_count"] == 3
    assert len(body["clips"][0]["frames"]) == 3


def test_playback_uses_the_measured_frame_rate(client: TestClient, store: ClipStore) -> None:
    """Playing back at the configured fps would show the clip at the wrong speed."""
    store.save(
        [np.full((8, 8, 3), 40, dtype=np.uint8) for _ in range(3)],
        {"fps_configured": 10.0, "fps_measured": 6.5},
    )
    clip = client.get("/api/queue").json()["clips"][0]
    assert clip["fps"] == 6.5


def test_playback_falls_back_to_the_configured_rate(client: TestClient, store: ClipStore) -> None:
    store.save([np.full((8, 8, 3), 40, dtype=np.uint8) for _ in range(3)], {"fps_configured": 8.0})
    clip = client.get("/api/queue").json()["clips"][0]
    assert clip["fps"] == 8.0


def test_queue_respects_its_limit(client: TestClient, store: ClipStore) -> None:
    for _ in range(4):
        _add_clip(store)
    body = client.get("/api/queue?limit=2").json()
    assert len(body["clips"]) == 2
    assert body["remaining"] == 4


def test_labelled_clips_leave_the_queue(client: TestClient, store: ClipStore) -> None:
    clip_id = _add_clip(store)
    client.post("/api/labels", json={"clip_id": clip_id, "label": "twisting"})
    assert client.get("/api/queue").json()["clips"] == []


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def test_a_label_is_stored(client: TestClient, store: ClipStore, labels: LabelStore) -> None:
    clip_id = _add_clip(store)
    response = client.post("/api/labels", json={"clip_id": clip_id, "label": "resting"})
    assert response.status_code == 200
    assert labels.get(clip_id) == "resting"


def test_unknown_label_is_rejected(client: TestClient, store: ClipStore) -> None:
    clip_id = _add_clip(store)
    response = client.post("/api/labels", json={"clip_id": clip_id, "label": "dancing"})
    assert response.status_code == 422


def test_labelling_a_missing_clip_is_rejected(client: TestClient) -> None:
    response = client.post("/api/labels", json={"clip_id": "nope", "label": "twisting"})
    assert response.status_code == 404


def test_undo_returns_the_clip_to_the_queue(client: TestClient, store: ClipStore) -> None:
    clip_id = _add_clip(store)
    client.post("/api/labels", json={"clip_id": clip_id, "label": "twisting"})
    assert client.post("/api/labels/undo").status_code == 200
    assert [c["clip_id"] for c in client.get("/api/queue").json()["clips"]] == [clip_id]


def test_undo_with_nothing_to_undo_is_rejected(client: TestClient) -> None:
    assert client.post("/api/labels/undo").status_code == 404


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def test_a_frame_is_served_as_jpeg(client: TestClient, store: ClipStore) -> None:
    clip_id = _add_clip(store)
    response = client.get(f"/api/clips/{clip_id}/frames/0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_missing_frame_index_is_rejected(client: TestClient, store: ClipStore) -> None:
    clip_id = _add_clip(store)
    assert client.get(f"/api/clips/{clip_id}/frames/99").status_code == 404


def test_frames_of_unknown_clip_are_rejected(client: TestClient) -> None:
    assert client.get("/api/clips/nope/frames/0").status_code == 404


def test_clip_id_cannot_escape_the_store(client: TestClient) -> None:
    """A traversal attempt must not reach outside the clip directory."""
    assert client.get("/api/clips/..%2F..%2Fetc/frames/0").status_code == 404


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_track_progress(client: TestClient, store: ClipStore) -> None:
    first = _add_clip(store)
    _add_clip(store)
    client.post("/api/labels", json={"clip_id": first, "label": "twisting"})

    stats = client.get("/api/stats").json()
    assert stats["total"] == 2
    assert stats["labelled"] == 1
    assert stats["remaining"] == 1
    assert stats["counts"] == {"twisting": 1}
    assert stats["can_undo"] is True


def test_stats_on_an_empty_store(client: TestClient) -> None:
    stats = client.get("/api/stats").json()
    assert stats == {"total": 0, "labelled": 0, "remaining": 0, "counts": {}, "can_undo": False}


# ---------------------------------------------------------------------------
# UI hosting
# ---------------------------------------------------------------------------

def test_the_ui_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "HairLab" in response.text
