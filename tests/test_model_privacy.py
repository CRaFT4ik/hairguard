"""The exported model must not carry the trainer's machine inside it.

The dynamo exporter records a Python stack trace for every node it emits,
absolute file paths included. A model trained on a developer's laptop and
then published therefore ships that developer's username and directory
layout to everyone who downloads it. This happened: a public artifact
contained 175 strings naming the trainer's home directory.

The graph does not need any of it to run, so the export strips it — and
this is where that stays true.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hairlab.model import TwistModule, export_onnx

PRINTABLE = re.compile(rb"[\x20-\x7e]{4,}")


def _strings(path) -> list[str]:
    return [chunk.decode("ascii", "replace") for chunk in PRINTABLE.findall(path.read_bytes())]


@pytest.fixture(scope="module")
def trained():
    """One module, exported once — so the agreement check compares like with like."""
    module = TwistModule(feature_dim=8)
    module.net.fit_normalisation(
        np.random.default_rng(0).normal(size=(4, 20, 8)).astype(np.float32)
    )
    module.eval()
    return module


@pytest.fixture(scope="module")
def exported(tmp_path_factory, trained):
    path = tmp_path_factory.mktemp("onnx") / "twistnet.onnx"
    export_onnx(trained, path, feature_dim=8, seq_len=20)
    return path


def test_no_filesystem_paths_are_embedded(exported) -> None:
    """Absolute paths name the machine that did the training."""
    offenders = [
        s for s in _strings(exported)
        if re.search(r"[A-Za-z]:\\|/home/|/Users/|\.py\b", s)
    ]
    assert not offenders, f"пути в модели: {offenders[:5]}"


def test_no_python_stack_traces_are_embedded(exported) -> None:
    offenders = [s for s in _strings(exported) if "line " in s and "File" in s]
    assert not offenders, f"трассировки в модели: {offenders[:3]}"


def test_nothing_about_the_training_machine_is_named(exported) -> None:
    """Install layout names the environment, and often the account, that built it.

    The project's own name is deliberate — it is the ONNX producer field and
    is public anyway. What must not survive is anything describing where the
    model was built.
    """
    text = " ".join(_strings(exported)).lower()
    for needle in ("site-packages", "localcache", "appdata", "/home/", "users"):
        assert needle not in text, f"в модели упомянуто {needle!r}"


def test_the_model_still_runs_after_stripping(exported) -> None:
    """Metadata removal must not touch the computation."""
    ort = pytest.importorskip("onnxruntime")

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    x = np.zeros((3, 20, 8), dtype=np.float32)
    outputs = session.run(None, {session.get_inputs()[0].name: x})

    scores = outputs[1].reshape(-1)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_stripping_matches_the_torch_model(exported, trained) -> None:
    """The scrubbed graph must still agree with the module it came from."""
    ort = pytest.importorskip("onnxruntime")

    x = np.random.default_rng(1).normal(size=(5, 20, 8)).astype(np.float32)
    with torch.no_grad():
        expected = torch.sigmoid(trained.net(torch.from_numpy(x))).numpy().reshape(-1)

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: x})[1].reshape(-1)

    assert np.allclose(expected, actual, rtol=1e-4, atol=1e-5)
