"""Tests for finding hairlab when the install did not map it.

Which detector the app runs must not depend on the directory it was started
from. The editable install on the development machine lists only
``hairguard`` as a top-level package, so ``hairlab`` — which holds the
feature extractor the model was trained with — is importable from the
repository root and nowhere else. A console script launched at login would
therefore quietly get the geometric rule and its 54% recall.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from hairguard.labpath import ensure_hairlab_importable

ROOT = Path(__file__).resolve().parent.parent


def test_nothing_is_done_when_hairlab_already_imports() -> None:
    """The normal case: a correct install must not be touched."""
    before = list(sys.path)
    assert ensure_hairlab_importable() is None
    assert sys.path == before


def test_it_is_found_beside_hairguard_when_the_path_lacks_it(monkeypatch) -> None:
    """Simulates the console-script launch that has no repository root on sys.path."""
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".") != ROOT])
    for name in list(sys.modules):
        if name == "hairlab" or name.startswith("hairlab."):
            monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()

    note = ensure_hairlab_importable()
    assert note is not None
    assert "sys.path" in note
    assert importlib.util.find_spec("hairlab") is not None


def test_it_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".") != ROOT])
    for name in list(sys.modules):
        if name == "hairlab" or name.startswith("hairlab."):
            monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()

    assert ensure_hairlab_importable() is not None
    assert ensure_hairlab_importable() is None
    assert sys.path.count(str(ROOT)) == 1


def test_the_repair_is_appended_not_prepended(monkeypatch) -> None:
    """A repair must never get the chance to shadow a real installed package."""
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".") != ROOT])
    for name in list(sys.modules):
        if name == "hairlab" or name.startswith("hairlab."):
            monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()

    ensure_hairlab_importable()
    assert sys.path[-1] == str(ROOT)


def test_it_gives_up_quietly_when_there_is_nothing_beside_hairguard(
    monkeypatch, tmp_path
) -> None:
    """A real hairguard-only install has nothing to find and must not crash."""
    import hairguard.labpath as labpath

    monkeypatch.setattr(labpath, "_importable", lambda: False)
    monkeypatch.setattr(labpath, "__file__", str(tmp_path / "pkg" / "labpath.py"))
    assert labpath.ensure_hairlab_importable() is None
