"""Tests for what goes into a build and what is refused.

Two failures are worth this much machinery.

The first is size. Training needs torch, which is 2.8 GB on this machine,
and a bundler that follows imports will swallow it whole — ``hairlab.model``
imports torch at module scope and sits in the same repository as the feature
extractor the app genuinely needs. A 3 GB download for a tray utility is not
a shippable artefact. The app does its inference through onnxruntime, 40 MB,
and needs neither torch nor lightning to run what they produced.

The second is quieter and worse. Without ``~/.hairguard/trained`` or a
bundled equivalent the app falls back to the geometric rule, which catches
53.9% of episodes against the model's 86.7% — and it does this while
starting normally and looking fine. On the developer's machine those files
happen to exist. A build that left them behind would ship something that
appears to work and misses half of what it should catch. Refusing to build
is better, so the preflight refuses, and the finished bundle is checked
again before it is called done.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bundle import buildconf


# ---------------------------------------------------------------------------
# Keeping the training stack out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    ["torch", "pytorch_lightning", "torchmetrics", "sklearn", "onnx", "onnxscript"],
)
def test_the_training_stack_is_excluded(module) -> None:
    assert module in buildconf.EXCLUDES


@pytest.mark.parametrize(
    "module",
    ["hairlab.model", "hairlab.train", "hairlab.dataset", "hairlab.evaluation",
     "hairlab.truncation"],
)
def test_the_training_modules_are_excluded(module) -> None:
    """The route torch would actually take into the bundle: these live in the
    same package as the feature extractor the app imports."""
    assert module in buildconf.EXCLUDES


def test_onnxruntime_is_not_excluded() -> None:
    """It is the thing that runs the model. Easy to sweep up with the rest of
    the names containing "onnx", and the app is useless without it."""
    assert "onnxruntime" not in buildconf.EXCLUDES
    assert "onnxruntime" in buildconf.HIDDEN_IMPORTS


def test_matplotlib_is_not_excluded() -> None:
    """29 MB of plotting library in a webcam tray utility, and it has to stay.

    ``mediapipe.tasks.python.vision.__init__`` imports ``drawing_styles`` →
    ``drawing_utils`` → ``import matplotlib.pyplot``, at module scope. The
    app imports ``PoseLandmarker`` from that package, so Python runs that
    ``__init__`` on the way. Nothing here ever draws a plot.

    This was excluded in the first build and the bundle died on its first
    import, four frames inside mediapipe. The test exists because the entry
    looks exactly like something a reasonable person should delete.
    """
    assert "matplotlib" not in buildconf.EXCLUDES
    assert "matplotlib" not in buildconf.FORBIDDEN_COMPONENTS
    # Its test suite, though, is 5 MB of reference images.
    assert "matplotlib.tests" in buildconf.EXCLUDES


def test_the_labelling_server_is_excluded() -> None:
    """fastapi and uvicorn belong to `hairlab label`, which is a developer
    tool and runs from a source checkout."""
    for module in ("fastapi", "uvicorn", "hairlab.server"):
        assert module in buildconf.EXCLUDES


def test_the_feature_extractor_is_kept() -> None:
    """``hairlab.features`` is what the model was trained with, and
    ``twistnet`` refuses to load a model whose feature width disagrees with
    it. Excluded, the bundle would start on the rule and say so politely."""
    assert "hairlab.features" in buildconf.HIDDEN_IMPORTS
    assert "hairlab.features" not in buildconf.EXCLUDES


@pytest.mark.parametrize(
    "module",
    ["hairlab.capture", "hairlab.state", "hairlab.clipstore", "hairlab.collector",
     "hairlab.writer"],
)
def test_the_collection_modules_are_kept(module) -> None:
    """Dataset collection is in the tray menu of the shipped app: it is how
    a user trains a model on their own hands, which the README tells them to
    do. ``labpath`` finds hairlab with ``find_spec``, so static analysis
    cannot be relied on to notice these."""
    assert module in buildconf.HIDDEN_IMPORTS


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


@pytest.fixture
def buildable(tmp_path):
    """A source tree and a model cache with everything a build needs."""
    repo = tmp_path / "repo"
    models = tmp_path / "cache"
    (repo / "hairguard" / "models").mkdir(parents=True)
    (repo / "hairlab").mkdir()
    (repo / "hairlab" / "features.py").write_text("FEATURE_DIM = 96", encoding="utf-8")
    for name in buildconf.TRAINED_MODEL_FILES:
        (repo / "hairguard" / "models" / name).write_bytes(b"x" * 64)
    models.mkdir()
    for name in buildconf.MEDIAPIPE_TASKS:
        (models / name).write_bytes(b"y" * 64)
    return {"repo": repo, "models": models}


def test_preflight_passes_on_a_complete_tree(buildable) -> None:
    assert buildconf.preflight(buildable["repo"], buildable["models"]) == []


def test_preflight_refuses_without_the_trained_model(buildable) -> None:
    """The headline requirement. Shipping without it means shipping a
    detector that catches half as much, silently."""
    (buildable["repo"] / "hairguard" / "models" / "twistnet.onnx").unlink()

    problems = buildconf.preflight(buildable["repo"], buildable["models"])

    assert problems
    joined = " ".join(problems)
    assert "twistnet.onnx" in joined
    assert "правил" in joined, "it did not say what shipping anyway would mean"


def test_preflight_refuses_without_the_onnx_weight_sidecar(buildable) -> None:
    """The graph is 4.7 KB and its weights are 395 KB in a separate file.
    Shipping the graph alone would load, fail its own trial inference and
    fall back to the rule."""
    (buildable["repo"] / "hairguard" / "models" / "twistnet.onnx.data").unlink()

    problems = buildconf.preflight(buildable["repo"], buildable["models"])

    assert any("twistnet.onnx.data" in problem for problem in problems)


def test_preflight_refuses_without_the_landmark_models(buildable) -> None:
    (buildable["models"] / "hand_landmarker.task").unlink()

    problems = buildconf.preflight(buildable["repo"], buildable["models"])

    assert any("hand_landmarker.task" in problem for problem in problems)


def test_the_landmark_models_can_be_left_out_deliberately(buildable) -> None:
    """They are downloadable at runtime, so a build without them is degraded
    only in needing a network on first run — a real choice, unlike shipping
    without the trained model, which is not."""
    (buildable["models"] / "hand_landmarker.task").unlink()

    problems = buildconf.preflight(
        buildable["repo"], buildable["models"], require_landmark_models=False
    )

    assert problems == []


def test_preflight_refuses_without_the_feature_extractor(buildable) -> None:
    """Without ``hairlab/features.py`` beside the app there is nothing to
    check the model's feature width against, and it would be refused at
    load time."""
    (buildable["repo"] / "hairlab" / "features.py").unlink()

    problems = buildconf.preflight(buildable["repo"], buildable["models"])

    assert any("features" in problem for problem in problems)


# ---------------------------------------------------------------------------
# What gets copied in
# ---------------------------------------------------------------------------


def _targets(entries) -> dict[str, str]:
    return {Path(source).name: target for source, target in entries}


def test_the_trained_model_is_placed_where_twistnet_looks(buildable) -> None:
    """``twistnet.BUNDLED_DIR`` is ``Path(__file__).parent / "models"``, which
    inside a bundle resolves relative to the rewritten package path."""
    targets = _targets(buildconf.datas(buildable["repo"], buildable["models"]))

    assert targets["model.json"] == "hairguard/models"
    assert targets["twistnet.onnx"] == "hairguard/models"
    assert targets["twistnet.onnx.data"] == "hairguard/models"


def test_the_torch_checkpoint_is_not_shipped(buildable) -> None:
    """400 KB of pickled torch tensors that only ``hairlab`` can read. The
    app runs the ONNX export; the checkpoint is for retraining."""
    (buildable["repo"] / "hairguard" / "models" / "twistnet.ckpt").write_bytes(b"z")

    names = _targets(buildconf.datas(buildable["repo"], buildable["models"]))

    assert "twistnet.ckpt" not in names


def test_the_landmark_models_are_placed_where_assets_looks(buildable) -> None:
    targets = _targets(buildconf.datas(buildable["repo"], buildable["models"]))

    for name in buildconf.MEDIAPIPE_TASKS:
        assert targets[name] == "hairguard/assets"


def test_mediapipes_native_library_is_found_and_placed_where_it_looks() -> None:
    """This build of mediapipe is a ctypes wrapper around one 28 MB shared
    library, loaded via ``importlib.resources.files`` — a path computed at
    runtime that no import-following bundler can see. Left out, the bundle
    imports mediapipe fine and fails on the first detector."""
    libraries = buildconf.mediapipe_native_libraries()

    assert libraries, "mediapipe's shared library was not located"
    for source, target in libraries:
        assert Path(source).is_file()
        assert target == "mediapipe/tasks/c"


def test_preflight_refuses_without_mediapipes_native_library(
    buildable, monkeypatch
) -> None:
    monkeypatch.setattr(buildconf, "mediapipe_native_libraries", lambda: [])

    problems = buildconf.preflight(buildable["repo"], buildable["models"])

    assert any("MediaPipe" in problem for problem in problems)


def test_absent_landmark_models_are_simply_not_listed(buildable) -> None:
    """So that a deliberate build without them still produces a spec that
    PyInstaller can run."""
    (buildable["models"] / "face_landmarker.task").unlink()

    names = _targets(buildconf.datas(buildable["repo"], buildable["models"]))

    assert "face_landmarker.task" not in names
    assert "hand_landmarker.task" in names


# ---------------------------------------------------------------------------
# Checking the finished artefact
# ---------------------------------------------------------------------------


@pytest.fixture
def built(tmp_path):
    """A dist tree shaped like a good build."""
    root = tmp_path / "HairGuard"
    internal = root / "_internal"
    (internal / "hairguard" / "models").mkdir(parents=True)
    (internal / "hairguard" / "assets").mkdir(parents=True)
    for name in buildconf.TRAINED_MODEL_FILES:
        (internal / "hairguard" / "models" / name).write_bytes(b"x" * 64)
    for name in buildconf.MEDIAPIPE_TASKS:
        (internal / "hairguard" / "assets" / name).write_bytes(b"y" * 64)
    (root / f"{buildconf.APP_NAME}.exe").write_bytes(b"MZ")
    (root / f"{buildconf.CONSOLE_NAME}.exe").write_bytes(b"MZ")
    return root


def test_verification_passes_on_a_good_tree(built) -> None:
    assert buildconf.verify_bundle(built) == []


def test_verification_catches_a_torch_directory(built) -> None:
    """The failure this whole exclude list exists to prevent, asserted
    against the artefact rather than against the configuration — the
    configuration is a claim, the tree on disk is the fact."""
    (built / "_internal" / "torch" / "lib").mkdir(parents=True)
    (built / "_internal" / "torch" / "lib" / "torch_cpu.dll").write_bytes(b"x" * 1024)

    problems = buildconf.verify_bundle(built)

    assert any("torch" in problem for problem in problems)


def test_verification_catches_a_stray_lightning_package(built) -> None:
    (built / "_internal" / "pytorch_lightning").mkdir(parents=True)
    (built / "_internal" / "pytorch_lightning" / "__init__.pyc").write_bytes(b"x")

    assert any("lightning" in p for p in buildconf.verify_bundle(built))


def test_verification_is_not_fooled_by_onnxruntime(built) -> None:
    """"onnx" is a substring of "onnxruntime", and onnxruntime is required."""
    runtime = built / "_internal" / "onnxruntime" / "capi"
    runtime.mkdir(parents=True)
    (runtime / "onnxruntime_pybind11_state.pyd").write_bytes(b"x" * 1024)

    assert buildconf.verify_bundle(built) == []


def test_verification_catches_a_missing_trained_model(built) -> None:
    """Which would look like a working app that detects half as much."""
    (built / "_internal" / "hairguard" / "models" / "twistnet.onnx").unlink()

    problems = buildconf.verify_bundle(built)

    assert any("twistnet.onnx" in problem for problem in problems)


def test_verification_catches_a_missing_console_binary(built) -> None:
    """The console executable is how a packaged app stays diagnosable, and it
    is what the app spawns for the diagnostic window. Not an optional extra."""
    (built / f"{buildconf.CONSOLE_NAME}.exe").unlink()

    assert any(buildconf.CONSOLE_NAME in p for p in buildconf.verify_bundle(built))


def test_verification_catches_a_missing_windowed_binary(built) -> None:
    (built / f"{buildconf.APP_NAME}.exe").unlink()

    assert any(buildconf.APP_NAME in p for p in buildconf.verify_bundle(built))


def test_the_two_executables_differ_by_more_than_case() -> None:
    """Windows filenames are case-insensitive, so ``hairguard.exe`` beside
    ``HairGuard.exe`` is one file — the second one built replaces the first
    and the bundle ships with only a windowed binary, silently.

    That is exactly what the first build of this did.
    """
    assert buildconf.APP_NAME.lower() != buildconf.CONSOLE_NAME.lower()


def test_verification_catches_two_names_that_collide_on_windows(
    built, monkeypatch
) -> None:
    monkeypatch.setattr(buildconf, "CONSOLE_NAME", buildconf.APP_NAME.lower())

    assert any("регистр" in p for p in buildconf.verify_bundle(built))


def test_verification_reports_every_problem_at_once(built) -> None:
    """A build takes minutes. Finding one fault per run would be unkind."""
    (built / "_internal" / "hairguard" / "models" / "twistnet.onnx").unlink()
    (built / "_internal" / "torch").mkdir()
    (built / "_internal" / "torch" / "version.pyc").write_bytes(b"x")

    assert len(buildconf.verify_bundle(built)) >= 2


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def test_largest_components_are_reported_biggest_first(built) -> None:
    """"How big is it" is a deliverable, and "what dominates it" is the only
    useful follow-up."""
    big = built / "_internal" / "huge"
    big.mkdir()
    (big / "blob.bin").write_bytes(b"x" * 100_000)

    components = buildconf.largest_components(built)

    assert components[0][0] == "huge"
    assert components[0][1] >= 100_000
    assert [size for _, size in components] == sorted(
        (size for _, size in components), reverse=True
    )


def test_total_size_counts_the_whole_tree(built) -> None:
    assert buildconf.total_size(built) > 0
