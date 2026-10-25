"""Tests for what ``pip install hairguard`` has to pull in.

The packaged bundle is one way the app reaches a machine; the other is a
source install, and it had a hole. ``onnxruntime`` was listed only under the
``train`` extra, so ``pip install hairguard`` produced an app that started
cleanly, printed one line saying the model was unavailable, and then ran the
geometric rule — 53.9% of episodes caught against the model's 86.7%. Nobody
installing it would have had a reason to look.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _names(requirements: list[str]) -> set[str]:
    return {
        requirement.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        for requirement in requirements
    }


def test_onnxruntime_is_a_runtime_dependency(project) -> None:
    """It is what runs the model. Without it the app falls back to the rule."""
    assert "onnxruntime" in _names(project["dependencies"])


def test_the_training_stack_is_not_a_runtime_dependency(project) -> None:
    """torch is 2.8 GB and the app does not need it to run what torch made:
    the exported ONNX graph carries its own input normalisation."""
    runtime = _names(project["dependencies"])

    for package in ("torch", "pytorch-lightning", "torchmetrics", "scikit-learn"):
        assert package not in runtime


def test_training_still_has_everything_it_needs(project) -> None:
    train = _names(project["optional-dependencies"]["train"])

    assert {"torch", "pytorch-lightning", "onnx", "onnxscript"} <= train


def test_building_a_bundle_is_an_extra(project) -> None:
    """So that `pip install hairguard` does not drag a bundler onto the
    machine of someone who only wants to run the thing."""
    assert "pyinstaller" in _names(project["optional-dependencies"]["package"])


def test_there_is_a_console_entry_point(project) -> None:
    assert project["scripts"]["hairguard"] == "hairguard.entry:console_main"


def test_there_is_a_windowless_entry_point(project) -> None:
    """A source install should also be able to produce an icon that does not
    open a terminal — the same thing the bundle's HairGuard.exe is."""
    assert project["gui-scripts"]["hairguard-gui"] == "hairguard.entry:gui_main"


def test_the_trained_model_ships_as_package_data() -> None:
    """So that even a source install detects properly on day one, before the
    user has collected a dataset of their own."""
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    package_data = config["tool"]["setuptools"]["package-data"]

    assert any("models" in pattern for pattern in package_data["hairguard"])


def test_both_packages_are_installed_together() -> None:
    """``hairlab.features`` is the extractor the model is validated against.
    An install carrying only ``hairguard`` is why ``hairguard.labpath``
    exists at all."""
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    include = config["tool"]["setuptools"]["packages"]["find"]["include"]

    assert "hairguard*" in include
    assert "hairlab*" in include
