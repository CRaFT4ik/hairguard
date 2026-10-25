"""Tests for the two places a bundled app re-launches itself.

Both were written against ``python -m hairguard`` and both are wrong inside
a bundle, where ``sys.executable`` is the app rather than an interpreter:
``[sys.executable, "-m", "hairguard", "start"]`` becomes
``HairGuard.exe -m hairguard start``, which argparse rejects.

That matters more than it looks. Autostart is the whole reason this app gets
run at all — the friction of opening a terminal to start something you need
every day is, eventually, the thing that means it does not get started. An
autostart entry that writes a broken command line would fail silently at
login, months from now, and look like the app had stopped working.
"""
from __future__ import annotations

import sys

import pytest

from hairguard import autostart, cli


@pytest.fixture
def frozen_bundle(tmp_path, monkeypatch):
    """A directory shaped like a built bundle, with sys.executable inside it."""
    windowed = tmp_path / ("HairGuard.exe" if sys.platform == "win32" else "HairGuard")
    console = tmp_path / ("hairguard.exe" if sys.platform == "win32" else "hairguard")
    windowed.write_bytes(b"MZ")
    console.write_bytes(b"MZ")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(console))
    return {"dir": tmp_path, "windowed": windowed, "console": console}


# ---------------------------------------------------------------------------
# Autostart
# ---------------------------------------------------------------------------


def test_autostart_launches_the_bundle_itself(frozen_bundle) -> None:
    argv, cwd = autostart._hairguard_command()

    assert argv == [str(frozen_bundle["windowed"])]
    assert "-m" not in argv
    assert cwd is None


def test_autostart_prefers_the_windowed_binary(frozen_bundle) -> None:
    """Registering the console one would pop a terminal at every login."""
    argv, _ = autostart._hairguard_command()

    assert "HairGuard" in argv[0]


def test_autostart_falls_back_to_this_executable(frozen_bundle, monkeypatch) -> None:
    """A single-file build, or a renamed one: whatever is running is still a
    better guess than an interpreter that is not there."""
    frozen_bundle["windowed"].unlink()

    argv, cwd = autostart._hairguard_command()

    assert argv == [str(frozen_bundle["console"])]
    assert cwd is None


def test_the_bundle_is_not_asked_to_detach_itself(frozen_bundle) -> None:
    """``--background`` re-launches the process to escape the console. A
    windowed build has no console to escape, and the extra hop would only
    add somewhere for the launch to go wrong."""
    assert autostart._launch_args() == ["start"]


def test_a_source_install_still_detaches(monkeypatch) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    assert autostart._launch_args() == ["start", "--background"]


def test_the_written_autostart_entry_points_at_the_bundle(
    frozen_bundle, monkeypatch, tmp_path
) -> None:
    """End to end on the artefact the user's machine will actually read."""
    startup = tmp_path / "startup"
    startup.mkdir()
    monkeypatch.setattr(autostart, "_windows_startup_dir", lambda: startup)
    monkeypatch.setattr(autostart, "_linux_autostart_dir", lambda: startup)
    monkeypatch.setattr(autostart, "_macos_launchagents_dir", lambda: startup)

    autostart.autostart_on()

    written = "".join(p.read_text(encoding="utf-8") for p in startup.iterdir())
    assert "HairGuard" in written
    assert "-m hairguard" not in written
    assert "--background" not in written
    assert str(frozen_bundle["windowed"]) in written


# ---------------------------------------------------------------------------
# hairguard start --background
# ---------------------------------------------------------------------------


def test_background_start_re_launches_the_bundle(frozen_bundle, monkeypatch) -> None:
    """Kept working rather than removed: `hairguard start --background` is in
    the README and in the autostart entries already on disk."""
    launched: list[list[str]] = []

    class FakeProc:
        pid = 777

    def fake_popen(cmd, **kwargs):
        launched.append(cmd)
        return FakeProc()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)

    cli._start_background(
        cli.build_parser().parse_args(["start", "--background", "--camera", "1"])
    )

    assert launched == [[str(frozen_bundle["console"]), "start", "--camera", "1"]]
    assert "-m" not in launched[0]


def test_background_start_from_source_still_uses_the_interpreter(monkeypatch) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    class FakeProc:
        pid = 778

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda cmd, **kwargs: launched.append(cmd) or FakeProc(),
    )

    cli._start_background(cli.build_parser().parse_args(["start", "--background"]))

    assert launched == [[sys.executable, "-m", "hairguard", "start"]]
