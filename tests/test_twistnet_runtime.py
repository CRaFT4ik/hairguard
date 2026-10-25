"""Tests for the ONNX runtime wrapper and the decision to use it at all.

Two things are being protected here.

*A missing or mismatched model must not stop the app.* The old rule still
works; it only catches half of what it should, which is far better than an
app that refuses to start. Every way loading can fail has to end in
``ModelUnavailable`` and a sentence a person can act on.

*A broken score must never read as "not twisting".* A NaN compares false
against every threshold, so it looks exactly like a confident negative —
which is the silent miss this whole project exists to fight. The runtime
raises instead, and the caller decides what a skipped window means.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hairguard.twistnet import ModelUnavailable, TwistModel, load_configured
from hairlab.features import FEATURE_DIM

TRAINED_DIR = Path.home() / ".hairguard" / "trained"

needs_model = pytest.mark.skipif(
    not (TRAINED_DIR / "twistnet.onnx").is_file(),
    reason=f"no exported model under {TRAINED_DIR}",
)


@pytest.fixture
def trained_copy(tmp_path) -> Path:
    """A writable copy of the real artefacts, for corrupting one at a time."""
    if not (TRAINED_DIR / "twistnet.onnx").is_file():
        pytest.skip(f"no exported model under {TRAINED_DIR}")
    target = tmp_path / "trained"
    target.mkdir()
    for name in ("model.json", "twistnet.onnx", "twistnet.onnx.data"):
        source = TRAINED_DIR / name
        if source.is_file():
            shutil.copy(source, target / name)
    return target


def _write_meta(directory: Path, **overrides) -> None:
    meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    meta.update(overrides)
    (directory / "model.json").write_text(json.dumps(meta), encoding="utf-8")


class _FakeConfig:
    """Answers only the keys the loader is allowed to read."""

    def __init__(self, **model_settings) -> None:
        self._model = {"enabled": True, "threshold": "auto", "trigger_delay_sec": 0.3}
        self._model.update(model_settings)

    def get(self, *keys: str):
        if keys[:2] == ("detection", "model"):
            return self._model[keys[2]]
        raise KeyError(keys)


# ---------------------------------------------------------------------------
# Loading the real artefacts
# ---------------------------------------------------------------------------

@needs_model
class TestLoad:
    def test_loads_the_trained_model(self) -> None:
        model = TwistModel.load(TRAINED_DIR)
        try:
            assert model.feature_dim == FEATURE_DIM
            assert model.seq_len == 20
            assert 0.0 < model.threshold < 1.0
        finally:
            model.close()

    def test_reports_where_it_came_from(self) -> None:
        """So "which detector is running" is answerable, not guessable."""
        model = TwistModel.load(TRAINED_DIR)
        try:
            assert model.info.directory == TRAINED_DIR
            assert model.info.trained_at
        finally:
            model.close()

    def test_an_explicit_threshold_overrides_the_file(self) -> None:
        model = TwistModel.load(TRAINED_DIR, threshold=0.5)
        try:
            assert model.threshold == pytest.approx(0.5)
        finally:
            model.close()


class TestLoadFailures:
    def test_a_missing_directory(self, tmp_path) -> None:
        with pytest.raises(ModelUnavailable, match="не найден"):
            TwistModel.load(tmp_path / "nope")

    def test_a_missing_onnx_file(self, tmp_path) -> None:
        (tmp_path / "model.json").write_text(
            json.dumps({"threshold": 0.5, "feature_dim": FEATURE_DIM, "seq_len": 20}),
            encoding="utf-8",
        )
        with pytest.raises(ModelUnavailable, match="twistnet.onnx"):
            TwistModel.load(tmp_path)

    def test_unreadable_metadata(self, trained_copy) -> None:
        (trained_copy / "model.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ModelUnavailable, match="model.json"):
            TwistModel.load(trained_copy)

    def test_a_feature_dim_the_extractor_no_longer_produces(self, trained_copy) -> None:
        """The guard that matters: features changed, the model did not.

        Nothing else in the system would notice. The extractor would hand
        over a differently shaped vector, or — worse, if the width happened
        to match — the same width with different meanings per column.
        """
        _write_meta(trained_copy, feature_dim=FEATURE_DIM + 1)
        with pytest.raises(ModelUnavailable, match="признак"):
            TwistModel.load(trained_copy)

    def test_metadata_disagreeing_with_the_graph(self, trained_copy) -> None:
        """model.json and the ONNX file must describe the same window."""
        _write_meta(trained_copy, seq_len=16)
        with pytest.raises(ModelUnavailable, match="16"):
            TwistModel.load(trained_copy)

    def test_a_threshold_outside_the_probability_range(self, trained_copy) -> None:
        _write_meta(trained_copy, threshold=1.5)
        with pytest.raises(ModelUnavailable, match="порог"):
            TwistModel.load(trained_copy)

    def test_a_missing_threshold(self, trained_copy) -> None:
        meta = json.loads((trained_copy / "model.json").read_text(encoding="utf-8"))
        del meta["threshold"]
        (trained_copy / "model.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ModelUnavailable, match="порог"):
            TwistModel.load(trained_copy)

    def test_a_corrupt_graph(self, trained_copy) -> None:
        (trained_copy / "twistnet.onnx").write_bytes(b"not an onnx graph")
        with pytest.raises(ModelUnavailable):
            TwistModel.load(trained_copy)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@needs_model
class TestScore:
    @pytest.fixture
    def model(self):
        instance = TwistModel.load(TRAINED_DIR)
        yield instance
        instance.close()

    def test_a_score_is_a_probability(self, model: TwistModel) -> None:
        window = np.zeros((model.seq_len, model.feature_dim), dtype=np.float32)
        score = model.score(window)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_a_batch_scores_every_row(self, model: TwistModel) -> None:
        batch = np.zeros((3, model.seq_len, model.feature_dim), dtype=np.float32)
        scores = model.score_batch(batch)
        assert scores.shape == (3,)

    def test_one_window_scores_the_same_alone_as_in_a_batch(
        self, model: TwistModel
    ) -> None:
        rng = np.random.default_rng(0)
        window = rng.normal(size=(model.seq_len, model.feature_dim)).astype(np.float32)
        single = model.score(window)
        batched = model.score_batch(window[None])[0]
        assert single == pytest.approx(batched, abs=1e-6)

    def test_a_nan_raises_rather_than_scoring_low(self, model: TwistModel) -> None:
        """A NaN score would be a confident "not twisting" — a silent miss."""
        window = np.zeros((model.seq_len, model.feature_dim), dtype=np.float32)
        window[3, 7] = np.nan
        with pytest.raises(ValueError, match="7"):
            model.score(window)

    def test_an_infinity_raises_too(self, model: TwistModel) -> None:
        window = np.zeros((model.seq_len, model.feature_dim), dtype=np.float32)
        window[0, 0] = np.inf
        with pytest.raises(ValueError):
            model.score(window)

    def test_the_wrong_window_length_raises(self, model: TwistModel) -> None:
        window = np.zeros((model.seq_len - 1, model.feature_dim), dtype=np.float32)
        with pytest.raises(ValueError, match="форм"):
            model.score(window)

    def test_the_wrong_feature_count_raises(self, model: TwistModel) -> None:
        window = np.zeros((model.seq_len, model.feature_dim - 1), dtype=np.float32)
        with pytest.raises(ValueError, match="форм"):
            model.score(window)

    def test_a_float64_window_is_accepted(self, model: TwistModel) -> None:
        """Callers should not have to know the graph's dtype."""
        window = np.zeros((model.seq_len, model.feature_dim), dtype=np.float64)
        assert 0.0 <= model.score(window) <= 1.0


# ---------------------------------------------------------------------------
# The configured decision, and saying it out loud
# ---------------------------------------------------------------------------

class TestLoadConfigured:
    @needs_model
    def test_enabled_and_present_gives_a_model(self) -> None:
        model, status = load_configured(_FakeConfig(), directory=TRAINED_DIR)
        try:
            assert model is not None
            assert "модель" in status.lower()
            # The threshold has to be in the line: it is the one number that
            # decides how often the user gets interrupted.
            assert f"{model.threshold:.3f}" in status
        finally:
            if model is not None:
                model.close()

    @needs_model
    def test_a_configured_threshold_wins(self) -> None:
        model, _ = load_configured(
            _FakeConfig(threshold=0.8), directory=TRAINED_DIR
        )
        try:
            assert model.threshold == pytest.approx(0.8)
        finally:
            model.close()

    def test_switched_off_in_config_falls_back_without_touching_the_disk(
        self, tmp_path
    ) -> None:
        model, status = load_configured(
            _FakeConfig(enabled=False), directory=tmp_path / "nope"
        )
        assert model is None
        assert "правил" in status.lower()

    def test_a_missing_model_falls_back_and_says_why(self, tmp_path) -> None:
        """The app must start. It must also not pretend the model is running."""
        model, status = load_configured(_FakeConfig(), directory=tmp_path / "nope")
        assert model is None
        assert "правил" in status.lower()
        assert "не найден" in status

    def test_a_broken_model_falls_back_and_says_why(self, tmp_path) -> None:
        (tmp_path / "model.json").write_text("{]", encoding="utf-8")
        model, status = load_configured(_FakeConfig(), directory=tmp_path)
        assert model is None
        assert "правил" in status.lower()

    @needs_model
    def test_an_unreadable_config_uses_the_defaults(self) -> None:
        """A config that cannot be read must not silently demote the detector.

        Falling back to the rule here would cost the user the model for a
        reason that has nothing to do with the model, so the built-in
        defaults are used instead — which is what a valid config would have
        produced anyway.
        """

        class Broken:
            def get(self, *keys):
                raise RuntimeError("no config")

        model, status = load_configured(Broken(), directory=TRAINED_DIR)
        try:
            assert model is not None, status
        finally:
            if model is not None:
                model.close()


@needs_model
class TestClose:
    def test_closing_twice_is_harmless(self) -> None:
        model = TwistModel.load(TRAINED_DIR)
        model.close()
        model.close()

    def test_scoring_after_close_says_so(self) -> None:
        """In the detection loop this becomes a skipped window, not a mystery."""
        model = TwistModel.load(TRAINED_DIR)
        window = np.zeros((model.seq_len, model.feature_dim), dtype=np.float32)
        model.close()
        with pytest.raises(ValueError, match="закрыт"):
            model.score(window)
