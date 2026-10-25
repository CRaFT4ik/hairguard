"""Tests for where the MediaPipe landmark models come from.

Three of them, 21 MB together, downloaded from Google's CDN into
``~/.hairguard/models`` on first run. That is fine for a developer with a
terminal and a network; it is a poor first impression for someone who has
just dragged an app into Applications, and it is a hard failure on an
aeroplane or behind a corporate proxy. The three downloads also happen
before the tray icon exists, so what the user sees while they run is
nothing.

So a build bundles them, and the resolution order below is what makes a
bundled copy, a cached copy and a download all work without the app ever
having to be told which situation it is in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hairguard import assets


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    user = tmp_path / "user"
    bundled = tmp_path / "bundled"
    user.mkdir()
    bundled.mkdir()
    monkeypatch.setattr(assets, "user_dir", lambda: user)
    monkeypatch.setattr(assets, "bundled_dir", lambda: bundled)
    return {"user": user, "bundled": bundled}


def _downloader(log: list[tuple[str, str]]):
    def download(url: str, destination: str) -> None:
        log.append((url, destination))
        Path(destination).write_bytes(b"downloaded")

    return download


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


def test_a_cached_file_is_used_and_nothing_is_downloaded(dirs) -> None:
    (dirs["user"] / "hand_landmarker.task").write_bytes(b"cached")
    log: list[tuple[str, str]] = []

    path = assets.ensure_task_file(
        "hand_landmarker.task", "https://example/x", download=_downloader(log)
    )

    assert Path(path) == dirs["user"] / "hand_landmarker.task"
    assert log == []


def test_a_bundled_file_is_used_when_nothing_is_cached(dirs) -> None:
    """The point of bundling: a fresh machine detects on the first launch,
    offline, with no wait and no explaining."""
    (dirs["bundled"] / "hand_landmarker.task").write_bytes(b"bundled")
    log: list[tuple[str, str]] = []

    path = assets.ensure_task_file(
        "hand_landmarker.task", "https://example/x", download=_downloader(log)
    )

    assert Path(path) == dirs["bundled"] / "hand_landmarker.task"
    assert log == []


def test_the_cache_wins_over_the_bundle(dirs) -> None:
    """A cached file is either what the app downloaded itself or something
    the user put there on purpose. The bundled copy is pinned to a build and
    can only ever be older."""
    (dirs["user"] / "hand_landmarker.task").write_bytes(b"cached")
    (dirs["bundled"] / "hand_landmarker.task").write_bytes(b"bundled")

    path = assets.ensure_task_file("hand_landmarker.task", "https://example/x")

    assert Path(path).read_bytes() == b"cached"


def test_downloading_is_the_last_resort(dirs) -> None:
    log: list[tuple[str, str]] = []

    path = assets.ensure_task_file(
        "face_landmarker.task", "https://example/face", download=_downloader(log)
    )

    assert Path(path) == dirs["user"] / "face_landmarker.task"
    assert log == [("https://example/face", str(dirs["user"] / "face_landmarker.task"))]
    assert Path(path).read_bytes() == b"downloaded"


def test_the_cache_directory_is_created_before_downloading(tmp_path, monkeypatch) -> None:
    user = tmp_path / "nowhere" / "models"
    monkeypatch.setattr(assets, "user_dir", lambda: user)
    monkeypatch.setattr(assets, "bundled_dir", lambda: tmp_path / "absent")

    assets.ensure_task_file(
        "pose_landmarker_full.task", "https://example/pose", download=_downloader([])
    )

    assert (user / "pose_landmarker_full.task").is_file()


# ---------------------------------------------------------------------------
# Failing usefully
# ---------------------------------------------------------------------------


def test_a_failed_download_says_what_was_missing_and_from_where(dirs) -> None:
    """This runs before the tray icon exists, so the message in the log is
    the only account of why nothing appeared."""

    def refuse(url: str, destination: str) -> None:
        raise OSError("getaddrinfo failed")

    with pytest.raises(assets.AssetUnavailable) as caught:
        assets.ensure_task_file(
            "hand_landmarker.task", "https://example/hand", download=refuse
        )

    message = str(caught.value)
    assert "hand_landmarker.task" in message
    assert "https://example/hand" in message
    assert "getaddrinfo failed" in message


def test_a_half_written_download_is_not_left_behind(dirs) -> None:
    """A truncated .task file would be found by the next run, used, and fail
    somewhere far less obvious than here."""

    def fail_midway(url: str, destination: str) -> None:
        Path(destination).write_bytes(b"half a model")
        raise OSError("connection reset")

    with pytest.raises(assets.AssetUnavailable):
        assets.ensure_task_file(
            "hand_landmarker.task", "https://example/hand", download=fail_midway
        )

    assert not (dirs["user"] / "hand_landmarker.task").exists()


# ---------------------------------------------------------------------------
# Where the bundled copy lives
# ---------------------------------------------------------------------------


def test_the_bundled_directory_sits_inside_the_package() -> None:
    """Which is what makes it resolve under PyInstaller: the bundler rewrites
    the package's path, and the assets are placed relative to it."""
    assert assets.bundled_dir() == Path(assets.__file__).parent / "assets"


def test_a_source_checkout_has_no_bundled_assets() -> None:
    """They are 21 MB of binaries copied in at build time, not committed."""
    assert not assets.bundled_dir().exists()


# ---------------------------------------------------------------------------
# What the detectors do with it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module, filename",
    [
        ("hairguard.detector", "pose_landmarker_full.task"),
        ("hairguard.hand_detector", "hand_landmarker.task"),
        ("hairguard.face_detector", "face_landmarker.task"),
    ],
)
def test_each_detector_resolves_its_model_through_assets(
    module, filename, monkeypatch
) -> None:
    """All three had their own copy of the same download-if-missing block.
    One of them silently not looking in the bundle would mean one download
    on a machine that was supposed to need none."""
    import importlib

    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        assets,
        "ensure_task_file",
        lambda name, url, **kwargs: asked.append((name, url)) or f"/models/{name}",
    )

    resolved = importlib.import_module(module)._ensure_model()

    assert asked and asked[0][0] == filename
    assert asked[0][1].startswith("https://")
    assert resolved == f"/models/{filename}"
