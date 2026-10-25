"""Tests for hairlab.model — the twisting classifier and how it is trained.

Everything here is synthetic: the tests must run without the clip dataset and
without a network.

The real dataset is 861 clips, which is small enough that regularisation and
the training protocol matter more than capacity. So these tests pin down the
things that silently go wrong at that size: that the normalisation statistics
travel *inside* the model (an exported ONNX with no sidecar scaler file cannot
drift out of sync with it), that early stopping hands back the best weights
rather than the last ones, that a seed reproduces a run, and that the ONNX
artifact really agrees with the torch model it came from.

The synthetic signal is an oscillation rather than a mean shift on purpose.
The network average-pools over time, so a mean shift would be learnable by the
linear head alone and would tell us nothing about the temporal convolutions —
which are the entire reason for choosing this architecture.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pytest
import torch

from hairlab.model import (
    DEFAULT_SEQ_LEN,
    TwistModule,
    TwistNet,
    export_onnx,
    load_checkpoint,
    predict_scores,
    save_checkpoint,
    train_model,
)

SEQ_LEN = DEFAULT_SEQ_LEN
FEATURE_DIM = 16   # tiny, so the training tests stay quick


def _synthetic(n: int, feature_dim: int = FEATURE_DIM, seed: int = 0):
    """Positives oscillate in the first few features; negatives are noise only."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.int64)
    X = rng.normal(0.0, 0.25, size=(n, SEQ_LEN, feature_dim)).astype(np.float32)

    t = np.arange(SEQ_LEN, dtype=np.float32)
    for i, label in enumerate(y):
        if label:
            # A random phase stops the model from keying on frame 0 alone.
            phase = rng.uniform(0.0, 2.0 * np.pi)
            wave = 2.0 * np.sin(2.0 * np.pi * 3.0 * t / SEQ_LEN + phase)
            X[i, :, :4] += wave[:, None]
    return X, y


@pytest.fixture(scope="module")
def dataset():
    X, y = _synthetic(240, seed=1)
    return X[:180], y[:180], X[180:], y[180:]


@pytest.fixture(scope="module")
def trained(dataset):
    """One trained model, shared by every test that needs real weights."""
    X_train, y_train, X_val, y_val = dataset
    return train_model(
        X_train, y_train, X_val, y_val,
        max_epochs=40, seed=0, batch_size=32, patience=40,
    )


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def test_forward_returns_one_logit_per_sample() -> None:
    net = TwistNet(feature_dim=FEATURE_DIM)
    out = net(torch.randn(7, SEQ_LEN, FEATURE_DIM))
    assert out.shape == (7,)
    assert torch.isfinite(out).all()


def test_forward_handles_a_single_sample() -> None:
    """Inference runs one clip at a time, so BatchNorm must not need a batch."""
    net = TwistNet(feature_dim=FEATURE_DIM).eval()
    with torch.no_grad():
        out = net(torch.randn(1, SEQ_LEN, FEATURE_DIM))
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_sequence_length_is_not_baked_in() -> None:
    """Padding and global pooling should leave the model length-agnostic."""
    net = TwistNet(feature_dim=FEATURE_DIM).eval()
    with torch.no_grad():
        assert net(torch.randn(3, 30, FEATURE_DIM)).shape == (3,)


def test_parameter_count_suits_a_small_dataset() -> None:
    """861 clips cannot support a large model; the brief's budget is 50k-200k."""
    net = TwistNet(feature_dim=80)
    total = sum(p.numel() for p in net.parameters())
    assert 50_000 <= total <= 200_000, total


# ---------------------------------------------------------------------------
# Input normalisation — the reason the exported model needs no scaler file
# ---------------------------------------------------------------------------

def test_normalisation_statistics_come_from_the_training_set() -> None:
    X, _ = _synthetic(64, seed=2)
    X = X * 7.0 + 3.0   # arbitrary scale and offset

    net = TwistNet(feature_dim=FEATURE_DIM)
    net.fit_normalisation(X)

    expected_mean = X.mean(axis=(0, 1))
    expected_std = X.std(axis=(0, 1))
    np.testing.assert_allclose(net.input_mean.numpy(), expected_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(net.input_std.numpy(), expected_std, rtol=1e-4, atol=1e-5)


def test_normalisation_is_applied_inside_forward() -> None:
    """Standardising by hand outside an un-fitted model must give the same answer.

    This is what makes the ONNX export self-contained: the division happens in
    the graph, not in whatever code happens to call it.
    """
    X, _ = _synthetic(32, seed=3)
    X = X * 5.0 - 2.0

    fitted = TwistNet(feature_dim=FEATURE_DIM).eval()
    fitted.fit_normalisation(X)

    plain = TwistNet(feature_dim=FEATURE_DIM).eval()
    plain.load_state_dict(fitted.state_dict())
    plain.reset_normalisation()   # back to mean 0, std 1

    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1))
    with torch.no_grad():
        from_model = fitted(torch.from_numpy(X))
        by_hand = plain(torch.from_numpy((X - mean) / std).float())

    np.testing.assert_allclose(from_model.numpy(), by_hand.numpy(), rtol=1e-4, atol=1e-4)


def test_normalisation_survives_a_state_dict_round_trip() -> None:
    """Persistent buffers, not attributes — otherwise a reloaded model is wrong."""
    X, _ = _synthetic(32, seed=4)
    X = X * 9.0 + 1.0

    fitted = TwistNet(feature_dim=FEATURE_DIM)
    fitted.fit_normalisation(X)
    state = fitted.state_dict()
    assert "input_mean" in state and "input_std" in state

    restored = TwistNet(feature_dim=FEATURE_DIM)
    restored.load_state_dict(state)
    np.testing.assert_allclose(restored.input_mean.numpy(), fitted.input_mean.numpy())
    np.testing.assert_allclose(restored.input_std.numpy(), fitted.input_std.numpy())


def test_a_constant_feature_does_not_produce_nans() -> None:
    """Some geometry features are constant when a landmark is always missing."""
    X, _ = _synthetic(32, seed=5)
    X[:, :, 2] = 0.5   # zero variance

    net = TwistNet(feature_dim=FEATURE_DIM).eval()
    net.fit_normalisation(X)
    with torch.no_grad():
        out = net(torch.from_numpy(X))
    assert torch.isfinite(out).all()


def test_a_constant_feature_normalises_to_zero() -> None:
    """Dividing by a floored std, so a constant column maps to 0, never to nan."""
    X, _ = _synthetic(32, seed=5)
    X[:, :, 2] = 0.5

    net = TwistNet(feature_dim=FEATURE_DIM)
    net.fit_normalisation(X)
    normalised = (torch.from_numpy(X) - net.input_mean) / net.input_std
    assert torch.isfinite(normalised).all()
    np.testing.assert_array_equal(normalised[:, :, 2].numpy(), 0.0)


def test_the_real_datasets_constant_columns_survive_training_and_export(tmp_path) -> None:
    """The real 96-feature matrix has four exactly-constant columns.

    ``hand_present`` and ``pose_present`` are always 1.0 because every clip in
    this snapshot has landmarks in every frame; ``hand_00_z`` is always 0.0
    because MediaPipe puts the z origin at the wrist, and ``vel_wrist_z``
    follows from that. All four are correct features that simply do not vary
    *yet*, so the model has to tolerate them rather than the feature set being
    trimmed to one snapshot of the data.

    Zero variance is the dangerous case: an unfloored 1/std is inf, which
    reaches the loss as nan and destroys training silently. The indices below
    are arbitrary — what matters is that some columns are exactly constant.
    """
    real_dim = 96
    rng = np.random.default_rng(12)
    X = rng.normal(0.0, 1.0, size=(64, SEQ_LEN, real_dim)).astype(np.float32)
    y = (rng.random(64) < 0.5).astype(np.int64)

    constant = {3: 1.0, 7: 1.0, 20: 0.0, 61: 0.0}   # *_present, *_z
    for index, value in constant.items():
        X[:, :, index] = value

    module = train_model(X[:48], y[:48], X[48:], y[48:], max_epochs=4, seed=0)

    # Nothing became nan on the way through training...
    for name, tensor in module.state_dict().items():
        assert torch.isfinite(tensor.float()).all(), name
    scores = predict_scores(module, X)
    assert np.isfinite(scores).all()

    # ...nor on the way through the exported graph, which is where a bad
    # scale would otherwise surface only in production.
    onnxruntime = pytest.importorskip("onnxruntime")
    path = tmp_path / "constant.onnx"
    export_onnx(module, path, feature_dim=real_dim)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    exported = np.asarray(session.run(None, {session.get_inputs()[0].name: X})[1]).reshape(-1)
    assert np.isfinite(exported).all()
    np.testing.assert_allclose(exported, scores, rtol=1e-4, atol=1e-5)


def _real_shaped(seed: int = 0):
    """The real dataset's shape and pathologies, without the real dataset.

    858 clips at 536/322, 96 features, the four exactly-constant columns, and
    per-feature scales spanning 1e-6 to 1e6 — because well-conditioned unit
    noise is precisely the input that hides scaling bugs.
    """
    rng = np.random.default_rng(seed)
    n, dim = 858, 96

    y = np.array([1] * 536 + [0] * 322, dtype=np.int64)
    rng.shuffle(y)

    X = rng.normal(0.0, 1.0, size=(n, SEQ_LEN, dim)).astype(np.float32)
    scales = (10.0 ** rng.integers(-6, 7, size=dim)).astype(np.float32)
    X = (X * scales + scales * rng.normal(0.0, 1.0, size=dim)).astype(np.float32)

    # Something learnable, at that feature's own scale.
    X[:, :, 0] += (y[:, None] * scales[0] * 2.0).astype(np.float32)
    for index, value in {3: 1.0, 7: 1.0, 20: 0.0, 61: 0.0}.items():
        X[:, :, index] = value
    return X.astype(np.float32), y


def test_predict_scores_stay_finite_on_real_shaped_data() -> None:
    """The guarantee that matters in a cross-validation run that takes minutes.

    A single nan score fails the whole run deep inside scikit-learn, far from
    whatever produced it, so it is much cheaper to find here.
    """
    X, y = _real_shaped(seed=0)
    # Enough epochs for a diverging run to actually blow up; this costs a few
    # seconds against a cross-validation run that costs minutes per fold.
    module = train_model(X[:686], y[:686], X[686:], y[686:], max_epochs=25, seed=0)

    scores = predict_scores(module, X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    # The weights themselves must be clean, not merely the scores.
    for name, tensor in module.state_dict().items():
        assert torch.isfinite(tensor.float()).all(), name


def test_normalisation_rejects_non_finite_features() -> None:
    """One nan cell would otherwise make the fitted statistics — and so every
    score the model ever produces — nan, with nothing raising on the way."""
    X, _ = _synthetic(32, seed=7)
    X[5, 2, 9] = np.nan

    net = TwistNet(feature_dim=FEATURE_DIM)
    with pytest.raises(ValueError, match=r"non-finite"):
        net.fit_normalisation(X)


def test_normalisation_names_the_offending_column() -> None:
    """A nan reaches here from the extractor, so the error has to point back."""
    X, _ = _synthetic(32, seed=7)
    X[1, 0, 9] = np.inf

    net = TwistNet(feature_dim=FEATURE_DIM)
    with pytest.raises(ValueError, match=r"column\(s\) \[9\]"):
        net.fit_normalisation(X)


def test_predict_scores_rejects_non_finite_input() -> None:
    """Better a named error than a nan score, which reads as a confident 'no'."""
    module = TwistModule(feature_dim=FEATURE_DIM)
    X, _ = _synthetic(8, seed=8)
    X[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=r"non-finite"):
        predict_scores(module, X)


def test_predict_scores_refuses_to_return_nan_from_corrupt_weights() -> None:
    """A diverged run must not quietly score every clip as 'not twisting'."""
    module = TwistModule(feature_dim=FEATURE_DIM)
    with torch.no_grad():
        module.net.head.bias.fill_(float("nan"))

    with pytest.raises(ValueError, match=r"non-finite"):
        predict_scores(module, _synthetic(8, seed=9)[0])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_training_learns_a_clear_signal(dataset, trained) -> None:
    _, _, X_val, y_val = dataset
    accuracy = ((predict_scores(trained, X_val) > 0.5).astype(np.int64) == y_val).mean()
    assert accuracy > 0.8, accuracy


def test_train_model_infers_the_feature_dimension(dataset) -> None:
    """FEATURE_DIM is the data layer's business, not a constant in this module."""
    X_train, y_train, X_val, y_val = dataset
    module = train_model(X_train, y_train, X_val, y_val, max_epochs=2, seed=0)
    assert module.hparams.feature_dim == X_train.shape[-1]


def test_training_fits_normalisation_on_the_training_split(dataset, trained) -> None:
    """Fitting on train only; validation data must not leak in through the stats."""
    X_train, _, _, _ = dataset
    np.testing.assert_allclose(
        trained.net.input_mean.detach().cpu().numpy(),
        X_train.mean(axis=(0, 1)),
        rtol=1e-4, atol=1e-5,
    )


def test_validation_loss_helper_agrees_with_the_monitored_metric(dataset, trained) -> None:
    """``validation_loss`` must reproduce the number early stopping acted on."""
    _, _, X_val, y_val = dataset
    assert trained.validation_loss(X_val, y_val) == pytest.approx(
        trained.best_val_loss, abs=1e-5
    )


def test_training_restores_the_best_weights() -> None:
    """The returned model must be the best-scoring epoch, not the last one.

    Random labels over a 32-sample training split force the failure mode this
    guards: the network memorises, validation loss turns around early and then
    climbs. On a signal the model can actually learn, the loss falls
    monotonically and the last epoch *is* the best one, so such a run cannot
    tell restoration apart from doing nothing.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(96, SEQ_LEN, FEATURE_DIM)).astype(np.float32)
    y = (rng.random(96) < 0.5).astype(np.int64)

    module = train_model(
        X[:32], y[:32], X[32:], y[32:],
        max_epochs=40, seed=0, patience=40, lr=3e-3,
    )

    history = module.val_loss_history
    assert len(history) == 40
    # The run really did end up worse than its best, or nothing is being tested.
    assert history[-1] > min(history) + 0.05
    assert module.best_val_loss == pytest.approx(min(history), abs=1e-6)
    # The weights we were handed score the minimum of the curve, not its end.
    assert module.validation_loss(X[32:], y[32:]) == pytest.approx(min(history), abs=1e-5)


def test_training_is_deterministic_for_a_given_seed(dataset) -> None:
    X_train, y_train, X_val, y_val = dataset
    kwargs = dict(max_epochs=3, seed=7, batch_size=32)
    first = train_model(X_train, y_train, X_val, y_val, **kwargs)
    second = train_model(X_train, y_train, X_val, y_val, **kwargs)
    np.testing.assert_array_equal(
        predict_scores(first, X_val), predict_scores(second, X_val)
    )


def test_a_different_seed_gives_a_different_model(dataset) -> None:
    """Guards against the determinism test passing because nothing trains."""
    X_train, y_train, X_val, y_val = dataset
    a = train_model(X_train, y_train, X_val, y_val, max_epochs=3, seed=7)
    b = train_model(X_train, y_train, X_val, y_val, max_epochs=3, seed=8)
    assert not np.allclose(predict_scores(a, X_val), predict_scores(b, X_val))


def test_pos_weight_shifts_the_loss(dataset) -> None:
    """Class imbalance is handled in the loss, so it has to reach the loss."""
    X_train, y_train, _, _ = dataset
    batch = (torch.from_numpy(X_train[:32]), torch.from_numpy(y_train[:32]))

    torch.manual_seed(0)
    neutral = TwistModule(feature_dim=FEATURE_DIM, pos_weight=1.0)
    torch.manual_seed(0)
    weighted = TwistModule(feature_dim=FEATURE_DIM, pos_weight=5.0)

    assert not torch.allclose(neutral.loss(*batch), weighted.loss(*batch))


# ---------------------------------------------------------------------------
# Inference and artifacts
# ---------------------------------------------------------------------------

def test_predict_scores_are_probabilities(dataset, trained) -> None:
    _, _, X_val, _ = dataset
    scores = predict_scores(trained, X_val)
    assert scores.shape == (len(X_val),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert np.isfinite(scores).all()


def test_predict_scores_leaves_the_model_in_eval_mode(trained) -> None:
    """Dropout firing during inference would make scores jitter run to run."""
    _ = predict_scores(trained, _synthetic(4, seed=6)[0])
    assert not trained.net.training


def test_checkpoint_round_trip_preserves_predictions(tmp_path, dataset, trained) -> None:
    _, _, X_val, _ = dataset
    path = tmp_path / "twist.ckpt"
    save_checkpoint(trained, path)

    reloaded = load_checkpoint(path)
    np.testing.assert_allclose(
        predict_scores(trained, X_val), predict_scores(reloaded, X_val),
        rtol=0, atol=1e-6,
    )


def test_onnx_export_matches_the_torch_model(tmp_path, dataset, trained) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    _, _, X_val, _ = dataset

    path = tmp_path / "twist.onnx"
    export_onnx(trained, path, feature_dim=X_val.shape[-1], seq_len=SEQ_LEN)
    assert path.exists()

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: X_val})

    # Second output is the probability — the graph applies the sigmoid itself,
    # so the app does not have to know it is looking at a logit.
    np.testing.assert_allclose(
        np.asarray(outputs[1]).reshape(-1), predict_scores(trained, X_val),
        rtol=1e-4, atol=1e-5,
    )


def test_onnx_export_bakes_in_the_normalisation(tmp_path) -> None:
    """The statistics must survive the trip into the graph.

    Export deep-copies the network and moves it to CPU, either of which could
    quietly drop a buffer, and constant folding rewrites the arithmetic that
    uses them. The test above would notice too, but only barely: its features
    are already near unit variance, so its statistics are close to identity.
    Here they are nowhere near it, so a graph that lost them cannot agree.

    Note this checks the exported graph against the torch model, so it says
    nothing about whether ``forward`` normalises at all — that is
    ``test_normalisation_is_applied_inside_forward``'s job.
    """
    onnxruntime = pytest.importorskip("onnxruntime")

    X, _ = _synthetic(24, seed=11)
    X = (X * 50.0 + 100.0).astype(np.float32)

    module = TwistModule(feature_dim=FEATURE_DIM)
    module.net.fit_normalisation(X)
    assert abs(float(module.net.input_mean[0])) > 10.0   # genuinely not identity

    path = tmp_path / "scaled.onnx"
    export_onnx(module, path, feature_dim=FEATURE_DIM)

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    scores = session.run(None, {session.get_inputs()[0].name: X})[1]
    np.testing.assert_allclose(
        np.asarray(scores).reshape(-1), predict_scores(module, X), rtol=1e-4, atol=1e-5
    )


def test_onnx_export_survives_a_non_utf8_console(tmp_path, trained, monkeypatch) -> None:
    """Exporting must not depend on what the terminal can encode.

    The exporter's own progress report contains emoji. On this project's
    Windows console (cp1251) printing it raises UnicodeEncodeError from inside
    torch.onnx.export, so a perfectly good export dies. pytest replaces stdout
    with a UTF-8 capture, which is exactly why this needs stating explicitly.
    """
    pytest.importorskip("onnxruntime")
    sink = io.TextIOWrapper(io.BytesIO(), encoding="cp1251", errors="strict")
    monkeypatch.setattr(sys, "stdout", sink)

    export_onnx(trained, tmp_path / "twist.onnx", feature_dim=FEATURE_DIM)
    assert (tmp_path / "twist.onnx").exists()


def test_onnx_declares_a_dynamic_batch_on_its_outputs(tmp_path, trained) -> None:
    """Declared output shapes must not lie about the batch dimension.

    torch.export specialises a size-1 dimension to the literal 1, so tracing
    with a batch of one leaves the outputs declared as ``{1}`` even though the
    input is symbolic and the values are correct for any batch. onnxruntime
    warns on every larger batch and anything reading the declared shape is
    misinformed. Checking values alone passes straight through this, which is
    exactly how it reached a real run before anyone noticed.
    """
    onnxruntime = pytest.importorskip("onnxruntime")

    path = tmp_path / "twist.onnx"
    export_onnx(trained, path, feature_dim=FEATURE_DIM, seq_len=SEQ_LEN)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    batch_dims = [output.shape[0] for output in session.get_outputs()]
    assert len(batch_dims) == 2
    for dim in batch_dims:
        # Symbolic dims come back as a name (or None); a concrete int here is
        # the bug — and `1` specifically is the one we hit.
        assert not isinstance(dim, int), f"output batch declared static as {dim}"

    # The input has always been dynamic; assert it so a fix cannot trade one
    # for the other.
    assert not isinstance(session.get_inputs()[0].shape[0], int)


def test_onnx_export_accepts_any_batch_size(tmp_path, trained) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")

    path = tmp_path / "twist.onnx"
    export_onnx(trained, path, feature_dim=FEATURE_DIM, seq_len=SEQ_LEN)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    for n in (1, 5):
        X = _synthetic(n, seed=10 + n)[0]
        assert np.asarray(session.run(None, {name: X})[1]).reshape(-1).shape == (n,)
