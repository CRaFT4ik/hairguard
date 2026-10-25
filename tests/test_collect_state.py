"""Tests for hairlab.state — the shared on/off switch for collection.

The tray is not the only way to reach the running app, and it gave no way
to check afterwards whether collection had actually started. The state lives
in a file so that the tray, the command line and a puzzled user looking at
the disk all see the same answer.
"""
from __future__ import annotations

import pytest

from hairlab.state import CollectionState


@pytest.fixture
def state(tmp_path) -> CollectionState:
    return CollectionState(path=tmp_path / "collect.state")


def test_collection_is_off_when_nothing_was_ever_written(state: CollectionState) -> None:
    assert state.read() is False


def test_turning_on_is_readable_back(state: CollectionState) -> None:
    state.write(True)
    assert state.read() is True


def test_turning_off_is_readable_back(state: CollectionState) -> None:
    state.write(True)
    state.write(False)
    assert state.read() is False


def test_another_process_sees_the_same_answer(tmp_path) -> None:
    """The CLI and the running app are separate processes."""
    path = tmp_path / "collect.state"
    CollectionState(path=path).write(True)
    assert CollectionState(path=path).read() is True


def test_a_corrupted_file_reads_as_off(state: CollectionState) -> None:
    """Never start recording because a file got mangled."""
    state.write(True)
    state.path.write_text("garbage", encoding="utf-8")
    assert state.read() is False


def test_an_unreadable_location_reads_as_off(tmp_path) -> None:
    assert CollectionState(path=tmp_path / "missing-dir" / "collect.state").read() is False


def test_writing_creates_the_directory(tmp_path) -> None:
    state = CollectionState(path=tmp_path / "fresh" / "collect.state")
    state.write(True)
    assert state.read() is True
