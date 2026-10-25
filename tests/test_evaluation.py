"""Tests for hairlab.evaluation — grouped splitting and the metrics that matter.

The property that earns most of this file: a capture day never appears in two
splits. Clips from one sitting are near-duplicates (same lighting, same hair,
often the same episode), so a single leaked day turns the test score into a
memory check and hides the one failure mode the project exists to fix.

Everything here is synthetic and seeded. No test reads the real dataset.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from hairlab.evaluation import (
    aggregate_folds,
    evaluate,
    format_composition,
    format_folds_report,
    format_report,
    grouped_cv_folds,
    grouped_split,
    split_composition,
)

_NEG_SUBS = ("resting", "scratching", "object")

# The real per-day distribution of the recorded set: 858 clips over 17 days,
# 536 positive. Only the shape is copied here — the clips are synthesised.
# Worth keeping verbatim: the days are wildly heterogeneous (positive share
# runs 14%..100%), five days hold 62% of everything, the largest single day is
# 13.5% of the dataset, and two days hold 1 and 3 clips.
_RECORDED_DAYS = [
    ("20260918", 3, 0),
    ("20260919", 51, 58),
    ("20260920", 17, 21),
    ("20260921", 56, 58),
    ("20260922", 89, 21),
    ("20260926", 43, 14),
    ("20260927", 3, 11),
    ("20261002", 72, 12),
    ("20261003", 57, 2),
    ("20261004", 20, 40),
    ("20261005", 4, 18),
    ("20261006", 1, 6),
    ("20261007", 6, 2),
    ("20261008", 1, 0),
    ("20261021", 7, 8),
    ("20261022", 73, 43),
    ("20261023", 33, 8),
]

# Same scale and skew as the recorded set: 17 days, 861 clips, ~63% positive,
# day sizes spanning 16..90, and one day that happens to hold no positives.
_REAL_SHAPE = [
    ("2026-03-01", 72, 18),
    ("2026-03-02", 40, 45),
    ("2026-03-03", 56, 24),
    ("2026-03-04", 30, 45),
    ("2026-03-05", 63, 7),
    ("2026-03-06", 39, 26),
    ("2026-03-07", 36, 24),
    ("2026-03-08", 44, 11),
    ("2026-03-09", 25, 25),
    ("2026-03-10", 36, 9),
    ("2026-03-11", 20, 20),
    ("2026-03-12", 28, 7),
    ("2026-03-13", 12, 18),
    ("2026-03-14", 20, 5),
    ("2026-03-15", 10, 10),
    ("2026-03-16", 16, 4),
    ("2026-03-17", 0, 16),
]


def _dataset(days):
    """``(day, n_pos, n_neg)`` triples -> ``groups, y, sublabels``.

    Negative sublabels cycle so every subtype appears on several days, which is
    what the recorded set looks like.
    """
    groups: list[str] = []
    y: list[int] = []
    subs: list[str] = []
    for day, n_pos, n_neg in days:
        groups += [str(day)] * (n_pos + n_neg)
        y += [1] * n_pos + [0] * n_neg
        subs += ["twisting"] * n_pos
        subs += [_NEG_SUBS[i % len(_NEG_SUBS)] for i in range(n_neg)]
    return np.asarray(groups), np.asarray(y), np.asarray(subs)


def _day_sets(groups, splits):
    return [set(np.asarray(groups)[idx].tolist()) for idx in splits]


def _pos_ratio(y, idx):
    sub = np.asarray(y)[idx]
    return float(sub.mean()) if sub.size else math.nan


# ---------------------------------------------------------------------------
# Splitting: the no-leak property
# ---------------------------------------------------------------------------

def test_no_day_appears_in_more_than_one_split() -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    splits = grouped_split(groups, y)
    train_days, val_days, test_days = _day_sets(groups, splits)
    assert not train_days & val_days
    assert not train_days & test_days
    assert not val_days & test_days


@pytest.mark.parametrize("seed", range(12))
def test_no_day_leaks_for_any_seed(seed: int) -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    train_days, val_days, test_days = _day_sets(groups, grouped_split(groups, y, seed=seed))
    assert len(train_days | val_days | test_days) == len(train_days) + len(val_days) + len(test_days)


@pytest.mark.parametrize("seed", range(25))
def test_no_day_leaks_on_random_dataset_shapes(seed: int) -> None:
    """Hammer the invariant across day counts, sizes and class mixes."""
    rng = np.random.default_rng(seed)
    n_days = int(rng.integers(1, 40))
    days = []
    for d in range(n_days):
        size = int(rng.integers(1, 100))
        n_pos = int(rng.integers(0, size + 1))
        days.append((f"day-{d}", n_pos, size - n_pos))
    groups, y, _ = _dataset(days)
    test_size = float(rng.uniform(0.1, 0.35))
    val_size = float(rng.uniform(0.0, 0.3))

    splits = grouped_split(groups, y, test_size=test_size, val_size=val_size, seed=seed)
    train_days, val_days, test_days = _day_sets(groups, splits)
    assert not train_days & val_days
    assert not train_days & test_days
    assert not val_days & test_days
    # And every clip of an assigned day travels with it.
    for day_set, idx in zip((train_days, val_days, test_days), splits):
        expected = np.isin(groups, list(day_set)).sum() if day_set else 0
        assert len(idx) == expected


def test_every_index_appears_exactly_once() -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    train, val, test = grouped_split(groups, y)
    combined = np.concatenate([train, val, test])
    assert sorted(combined.tolist()) == list(range(len(y)))


def test_indices_are_sorted_integer_arrays() -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    for idx in grouped_split(groups, y):
        assert isinstance(idx, np.ndarray)
        assert np.issubdtype(idx.dtype, np.integer)
        assert np.all(np.diff(idx) > 0)


# ---------------------------------------------------------------------------
# Splitting: determinism and balance
# ---------------------------------------------------------------------------

def test_deterministic_for_the_same_seed() -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    first = grouped_split(groups, y, seed=3)
    second = grouped_split(groups, y, seed=3)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_seed_changes_the_assignment() -> None:
    """With interchangeable days the seed must actually move days around."""
    days = [(f"day-{d}", 6 + d % 3, 4 - d % 3) for d in range(12)]
    groups, y, _ = _dataset(days)
    base = _day_sets(groups, grouped_split(groups, y, seed=0))
    assert any(
        _day_sets(groups, grouped_split(groups, y, seed=seed)) != base
        for seed in range(1, 10)
    )


def test_split_sizes_are_close_to_requested() -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    train, val, test = grouped_split(groups, y, test_size=0.2, val_size=0.15)
    n = len(y)
    # Whole days are indivisible and the largest is 90 of 861 clips, so a few
    # points of slack is the best that is available here.
    assert abs(len(test) / n - 0.2) < 0.04
    assert abs(len(val) / n - 0.15) < 0.04
    assert abs(len(train) / n - 0.65) < 0.04


def test_class_ratio_is_close_to_overall() -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    overall = float(np.mean(y))
    for idx in grouped_split(groups, y):
        assert abs(_pos_ratio(y, idx) - overall) < 0.05


def test_class_ratio_holds_across_seeds() -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    overall = float(np.mean(y))
    for seed in range(8):
        for idx in grouped_split(groups, y, seed=seed):
            assert abs(_pos_ratio(y, idx) - overall) < 0.06


# ---------------------------------------------------------------------------
# Splitting: awkward cases
# ---------------------------------------------------------------------------

def test_single_day_goes_entirely_to_train() -> None:
    """Nothing can be held out, so keep the data where a model can still be fit."""
    groups, y, _ = _dataset([("only-day", 10, 5)])
    train, val, test = grouped_split(groups, y)
    assert len(train) == 15
    assert len(val) == 0 and len(test) == 0


def test_two_days_fill_train_and_test_leaving_val_empty() -> None:
    groups, y, _ = _dataset([("a", 20, 10), ("b", 6, 4)])
    train, val, test = grouped_split(groups, y)
    assert len(train) == 30
    assert len(test) == 10
    assert len(val) == 0


def test_three_days_fill_all_three_splits() -> None:
    groups, y, _ = _dataset([("a", 20, 10), ("b", 6, 4), ("c", 5, 5)])
    train, val, test = grouped_split(groups, y)
    assert len(train) and len(val) and len(test)
    assert len(train) + len(val) + len(test) == len(y)


def test_dominant_day_stays_whole_and_does_not_starve_the_others() -> None:
    days = [("big", 400, 200), ("b", 10, 5), ("c", 8, 4), ("d", 6, 3)]
    groups, y, _ = _dataset(days)
    splits = grouped_split(groups, y)
    day_sets = _day_sets(groups, splits)
    assert sum("big" in days_in_split for days_in_split in day_sets) == 1
    assert all(len(idx) for idx in splits)


def test_single_class_day_does_not_break_the_split() -> None:
    days = [("a", 30, 0), ("b", 0, 20), ("c", 15, 15), ("d", 10, 5), ("e", 4, 6)]
    groups, y, _ = _dataset(days)
    splits = grouped_split(groups, y)
    train_days, val_days, test_days = _day_sets(groups, splits)
    assert not (train_days & val_days or train_days & test_days or val_days & test_days)
    assert all(len(idx) for idx in splits)


def test_zero_val_size_yields_an_empty_val_split() -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    train, val, test = grouped_split(groups, y, test_size=0.25, val_size=0.0)
    assert len(val) == 0
    assert len(train) + len(test) == len(y)
    assert len(test) > 0


def test_empty_input_returns_three_empty_splits() -> None:
    train, val, test = grouped_split(np.asarray([]), np.asarray([]))
    assert len(train) == len(val) == len(test) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": -0.1},
        {"val_size": -0.1},
        {"test_size": 0.7, "val_size": 0.4},
        {"test_size": 1.0, "val_size": 0.0},
    ],
)
def test_rejects_impossible_split_sizes(kwargs) -> None:
    groups, y, _ = _dataset(_REAL_SHAPE)
    with pytest.raises(ValueError):
        grouped_split(groups, y, **kwargs)


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        grouped_split(np.asarray(["a", "a", "b"]), np.asarray([1, 0]))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tiny_case():
    """Hand-countable case: 7 negatives of three subtypes, 3 positives."""
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    sublabels = ["resting"] * 4 + ["scratching"] * 2 + ["object"] + ["twisting"] * 3
    y_pred = [1, 0, 0, 0, 1, 1, 0, 1, 1, 0]
    y_score = [0.6, 0.4, 0.3, 0.2, 0.7, 0.8, 0.1, 0.9, 0.55, 0.45]
    return y_true, y_pred, y_score, sublabels


def test_false_positive_rate_is_reported_per_negative_subtype() -> None:
    metrics = evaluate(*_tiny_case())
    assert metrics["fpr_by_sublabel"]["resting"] == pytest.approx(0.25)
    assert metrics["fpr_by_sublabel"]["scratching"] == pytest.approx(1.0)
    assert metrics["fpr_by_sublabel"]["object"] == pytest.approx(0.0)
    assert metrics["fp_by_sublabel"] == {"resting": 1, "scratching": 2, "object": 0}
    assert metrics["n_by_sublabel"] == {"resting": 4, "scratching": 2, "object": 1}


def test_positive_sublabels_are_not_listed_as_negative_subtypes() -> None:
    metrics = evaluate(*_tiny_case())
    assert "twisting" not in metrics["fpr_by_sublabel"]


def test_subtype_with_no_samples_is_omitted() -> None:
    y_true = [0, 0, 1, 1]
    metrics = evaluate(y_true, [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.8], ["resting", "resting", "twisting", "twisting"])
    assert set(metrics["fpr_by_sublabel"]) == {"resting"}


def test_counts_and_confusion_matrix() -> None:
    metrics = evaluate(*_tiny_case())
    assert metrics["n"] == 10
    assert metrics["n_pos"] == 3
    assert metrics["n_neg"] == 7
    assert metrics["confusion"] == {"tn": 4, "fp": 3, "fn": 1, "tp": 2}
    assert metrics["fpr"] == pytest.approx(3 / 7)


def test_precision_recall_f1_for_the_positive_class() -> None:
    metrics = evaluate(*_tiny_case())
    assert metrics["precision"] == pytest.approx(0.4)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.6)


def test_perfect_predictions_score_perfectly() -> None:
    y_true = [1, 1, 0, 0, 0]
    sublabels = ["twisting", "twisting", "resting", "scratching", "object"]
    metrics = evaluate(y_true, y_true, [0.9, 0.8, 0.1, 0.2, 0.05], sublabels)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert all(rate == 0.0 for rate in metrics["fpr_by_sublabel"].values())


def test_always_firing_gives_a_full_false_positive_rate_everywhere() -> None:
    y_true = [1, 0, 0, 0]
    sublabels = ["twisting", "resting", "scratching", "object"]
    metrics = evaluate(y_true, [1, 1, 1, 1], [0.9, 0.9, 0.9, 0.9], sublabels)
    assert metrics["fpr_by_sublabel"] == {"resting": 1.0, "scratching": 1.0, "object": 1.0}
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(0.25)


def test_zero_predictions_do_not_divide_by_zero() -> None:
    y_true = [1, 1, 0, 0]
    metrics = evaluate(y_true, [0, 0, 0, 0], [0.3, 0.2, 0.1, 0.05], ["twisting", "twisting", "resting", "resting"])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_ranking_metrics_are_nan_when_one_class_is_missing() -> None:
    metrics = evaluate([1, 1, 1], [1, 0, 1], [0.9, 0.4, 0.8], ["twisting"] * 3)
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["average_precision"])


def test_ranking_metrics_ignore_the_passed_predictions() -> None:
    """AUC and AP come from the scores, so a broken threshold must not move them."""
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    good = evaluate(y_true, [1, 1, 0, 0], y_score, ["twisting", "twisting", "resting", "resting"])
    bad = evaluate(y_true, [0, 0, 0, 0], y_score, ["twisting", "twisting", "resting", "resting"])
    assert good["roc_auc"] == bad["roc_auc"] == pytest.approx(1.0)


def test_best_threshold_finds_an_operating_point_that_0_5_misses() -> None:
    y_true = [1] * 5 + [0] * 5
    y_score = [0.40, 0.38, 0.36, 0.34, 0.32, 0.20, 0.18, 0.16, 0.14, 0.12]
    sublabels = ["twisting"] * 5 + ["resting"] * 5
    metrics = evaluate(y_true, [0] * 10, y_score, sublabels)
    assert metrics["f1"] == 0.0
    best = metrics["best_f1"]
    assert best["f1"] == pytest.approx(1.0)
    assert best["threshold"] == pytest.approx(0.32)
    assert best["confusion"] == {"tn": 5, "fp": 0, "fn": 0, "tp": 5}
    assert best["fpr_by_sublabel"]["resting"] == 0.0


def test_best_threshold_metrics_match_that_threshold() -> None:
    y_true, y_pred, y_score, sublabels = _tiny_case()
    metrics = evaluate(y_true, y_pred, y_score, sublabels)
    threshold = metrics["best_f1"]["threshold"]
    recomputed = evaluate(y_true, (np.asarray(y_score) >= threshold).astype(int), y_score, sublabels)
    for key in ("precision", "recall", "f1", "confusion", "fpr_by_sublabel", "fp_by_sublabel"):
        assert metrics["best_f1"][key] == pytest.approx(recomputed[key])


def test_best_threshold_prefers_the_higher_of_two_equally_good_points() -> None:
    """Equal F1 for fewer alerts is the better operating point for a nagger."""
    y_true = [1, 1, 0, 0, 0, 0, 1, 1]
    y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    sublabels = ["twisting", "twisting"] + ["resting"] * 4 + ["twisting", "twisting"]
    metrics = evaluate(y_true, [0] * 8, y_score, sublabels)
    assert metrics["best_f1"]["f1"] == pytest.approx(2 / 3)
    assert metrics["best_f1"]["threshold"] == pytest.approx(0.8)


def test_best_threshold_is_nan_without_positives() -> None:
    metrics = evaluate([0, 0, 0], [0, 1, 0], [0.1, 0.6, 0.2], ["resting", "scratching", "object"])
    assert math.isnan(metrics["best_f1"]["threshold"])
    assert math.isnan(metrics["best_f1"]["f1"])


@pytest.mark.parametrize(
    "y_true, y_pred, y_score, sublabels",
    [
        ([1, 0], [1], [0.9, 0.1], ["twisting", "resting"]),
        ([1, 0], [1, 0], [0.9], ["twisting", "resting"]),
        ([1, 0], [1, 0], [0.9, 0.1], ["twisting"]),
    ],
)
def test_evaluate_rejects_mismatched_lengths(y_true, y_pred, y_score, sublabels) -> None:
    with pytest.raises(ValueError):
        evaluate(y_true, y_pred, y_score, sublabels)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_leads_with_the_per_subtype_false_positive_rates() -> None:
    text = format_report(evaluate(*_tiny_case())).lower()
    head = text.index("ложные срабатывания")
    assert head < text.index("precision")
    assert head < text.index("roc auc")
    # Accuracy is the metric that hides this failure; it must not come first.
    assert head < text.index("accuracy")


def test_report_names_every_negative_subtype() -> None:
    text = format_report(evaluate(*_tiny_case()))
    for subtype in _NEG_SUBS:
        assert subtype in text


def test_report_is_multiline_russian_text() -> None:
    text = format_report(evaluate(*_tiny_case()))
    assert isinstance(text, str)
    assert len(text.splitlines()) >= 10
    assert any("а" <= ch.lower() <= "я" for ch in text)


def test_report_shows_the_best_threshold() -> None:
    metrics = evaluate(*_tiny_case())
    text = format_report(metrics)
    assert "порог" in text.lower()
    assert f"{metrics['best_f1']['threshold']:.2f}" in text


def test_report_survives_missing_ranking_metrics() -> None:
    text = format_report(evaluate([1, 1, 1], [1, 0, 1], [0.9, 0.4, 0.8], ["twisting"] * 3))
    assert "н/д" in text


# ---------------------------------------------------------------------------
# Splitting the real per-day distribution
# ---------------------------------------------------------------------------

def test_recorded_fixture_matches_the_real_dataset() -> None:
    """Guard the fixture, so the tests below describe the actual recording."""
    groups, y, _ = _dataset(_RECORDED_DAYS)
    assert len(y) == 858
    assert int(y.sum()) == 536
    assert len(y) - int(y.sum()) == 322
    assert len(set(groups.tolist())) == 17


@pytest.mark.parametrize("seed", range(10))
def test_recorded_shape_never_leaks_a_day(seed: int) -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    splits = grouped_split(groups, y, seed=seed)
    train_days, val_days, test_days = _day_sets(groups, splits)
    assert not (train_days & val_days or train_days & test_days or val_days & test_days)
    for day_set, idx in zip((train_days, val_days, test_days), splits):
        assert len(idx) == (np.isin(groups, list(day_set)).sum() if day_set else 0)


@pytest.mark.parametrize("seed", range(10))
def test_recorded_shape_keeps_the_class_ratio_near_overall(seed: int) -> None:
    """Heterogeneous days make this the hard part; it must not be left to luck."""
    groups, y, _ = _dataset(_RECORDED_DAYS)
    overall = float(np.mean(y))
    for idx in grouped_split(groups, y, seed=seed):
        assert abs(_pos_ratio(y, idx) - overall) < 0.04


@pytest.mark.parametrize("seed", range(10))
def test_recorded_shape_gives_every_split_a_usable_size(seed: int) -> None:
    """The 1-clip and 3-clip days must not end up being the whole test split."""
    groups, y, _ = _dataset(_RECORDED_DAYS)
    for idx in grouped_split(groups, y, seed=seed):
        assert len(idx) / len(y) > 0.08


def test_the_largest_day_stays_whole() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    day_sets = _day_sets(groups, grouped_split(groups, y))
    assert sum("20261022" in days for days in day_sets) == 1


def test_a_single_clip_day_lands_in_exactly_one_split() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    splits = grouped_split(groups, y)
    owners = [i for i, idx in enumerate(splits) if "20261008" in set(groups[idx].tolist())]
    assert len(owners) == 1


# ---------------------------------------------------------------------------
# Realised composition
# ---------------------------------------------------------------------------

def test_composition_reports_what_each_split_actually_got() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    splits = grouped_split(groups, y)
    comp = split_composition(groups, y, splits)

    assert comp["n"] == 858
    assert comp["n_pos"] == 536
    assert comp["n_neg"] == 322
    assert comp["n_days"] == 17
    assert comp["pos_share"] == pytest.approx(536 / 858)

    assert [part["name"] for part in comp["splits"]] == ["train", "val", "test"]
    assert sum(part["n"] for part in comp["splits"]) == 858
    assert sum(part["n_days"] for part in comp["splits"]) == 17
    for part, idx in zip(comp["splits"], splits):
        assert part["n"] == len(idx)
        assert part["n_pos"] + part["n_neg"] == part["n"]
        assert part["pos_share"] == pytest.approx(part["n_pos"] / part["n"])
        assert part["frac"] == pytest.approx(part["n"] / 858)
        assert part["days"] == sorted(set(groups[idx].tolist()))
        assert part["n_days"] == len(part["days"])


def test_composition_accepts_fold_names() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    folds = grouped_cv_folds(groups, y, n_folds=3)
    comp = split_composition(groups, y, [test for _, test in folds], names=["f0", "f1", "f2"])
    assert [part["name"] for part in comp["splits"]] == ["f0", "f1", "f2"]


def test_composition_handles_an_empty_split() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    comp = split_composition(groups, y, grouped_split(groups, y, test_size=0.2, val_size=0.0))
    val = comp["splits"][1]
    assert val["n"] == 0 and val["n_days"] == 0 and val["days"] == []
    assert math.isnan(val["pos_share"])


def test_format_composition_is_readable_russian() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    text = format_composition(split_composition(groups, y, grouped_split(groups, y)))
    assert isinstance(text, str)
    for name in ("train", "val", "test"):
        assert name in text
    assert "дней" in text
    assert "858" in text


def test_a_badly_imbalanced_split_warns_instead_of_staying_silent() -> None:
    """Three days, each a different class mix: no assignment can be balanced."""
    groups, y, _ = _dataset([("a", 100, 0), ("b", 0, 100), ("c", 50, 50)])
    with pytest.warns(UserWarning, match="positive share"):
        grouped_split(groups, y)


def test_an_unavoidably_empty_split_warns() -> None:
    groups, y, _ = _dataset([("only", 10, 5)])
    with pytest.warns(UserWarning, match="empty"):
        grouped_split(groups, y)


def test_an_intentionally_empty_split_does_not_warn() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grouped_split(groups, y, test_size=0.2, val_size=0.0)


def test_the_recorded_shape_splits_without_complaint() -> None:
    """If this starts warning, the splitter regressed on the real distribution."""
    groups, y, _ = _dataset(_RECORDED_DAYS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grouped_split(groups, y)


# ---------------------------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------------------------

def test_cv_returns_the_requested_number_of_folds() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    folds = grouped_cv_folds(groups, y, n_folds=5)
    assert len(folds) == 5
    for train, test in folds:
        assert np.issubdtype(train.dtype, np.integer)
        assert np.issubdtype(test.dtype, np.integer)


@pytest.mark.parametrize("seed", range(6))
def test_cv_shares_no_day_between_train_and_test(seed: int) -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    for train, test in grouped_cv_folds(groups, y, n_folds=5, seed=seed):
        train_days = set(groups[train].tolist())
        test_days = set(groups[test].tolist())
        assert train_days and test_days
        assert not train_days & test_days
        # A day cannot be partially held out.
        assert len(test) == np.isin(groups, list(test_days)).sum()


def test_cv_train_and_test_partition_every_fold() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    for train, test in grouped_cv_folds(groups, y, n_folds=5):
        assert sorted(np.concatenate([train, test]).tolist()) == list(range(len(y)))


def test_cv_tests_every_clip_exactly_once_across_folds() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    tested = np.concatenate([test for _, test in grouped_cv_folds(groups, y, n_folds=5)])
    assert sorted(tested.tolist()) == list(range(len(y)))


def test_cv_folds_are_roughly_equal_in_size() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    sizes = [len(test) for _, test in grouped_cv_folds(groups, y, n_folds=5)]
    target = len(y) / 5
    # One day holds 13.5% of the clips, so a fold cannot land on the nose.
    assert max(abs(size - target) for size in sizes) < 0.25 * target


@pytest.mark.parametrize("seed", range(6))
def test_cv_folds_keep_the_class_ratio_near_overall(seed: int) -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    overall = float(np.mean(y))
    for _, test in grouped_cv_folds(groups, y, n_folds=5, seed=seed):
        assert abs(_pos_ratio(y, test) - overall) < 0.09


def test_cv_is_deterministic_for_the_same_seed() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    first = grouped_cv_folds(groups, y, seed=4)
    second = grouped_cv_folds(groups, y, seed=4)
    for (tr_a, te_a), (tr_b, te_b) in zip(first, second):
        assert np.array_equal(tr_a, tr_b) and np.array_equal(te_a, te_b)


def test_cv_seed_changes_the_folds() -> None:
    days = [(f"day-{d}", 6 + d % 3, 4 - d % 3) for d in range(12)]
    groups, y, _ = _dataset(days)
    base = [set(groups[test].tolist()) for _, test in grouped_cv_folds(groups, y, seed=0)]
    assert any(
        [set(groups[test].tolist()) for _, test in grouped_cv_folds(groups, y, seed=seed)] != base
        for seed in range(1, 10)
    )


def test_cv_leave_one_day_out() -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    folds = grouped_cv_folds(groups, y, n_folds=17)
    assert len(folds) == 17
    assert all(len(set(groups[test].tolist())) == 1 for _, test in folds)


def test_cv_tolerates_single_class_and_single_clip_days() -> None:
    days = [("a", 30, 0), ("b", 0, 20), ("c", 1, 0), ("d", 10, 5), ("e", 4, 6)]
    groups, y, _ = _dataset(days)
    for train, test in grouped_cv_folds(groups, y, n_folds=3):
        assert not set(groups[train].tolist()) & set(groups[test].tolist())
        assert len(test) > 0


@pytest.mark.parametrize("n_folds", [0, 1, -3])
def test_cv_rejects_degenerate_fold_counts(n_folds: int) -> None:
    groups, y, _ = _dataset(_RECORDED_DAYS)
    with pytest.raises(ValueError):
        grouped_cv_folds(groups, y, n_folds=n_folds)


def test_cv_rejects_more_folds_than_days() -> None:
    """Silently returning fewer folds would quietly corrupt the averaging."""
    groups, y, _ = _dataset([("a", 5, 5), ("b", 4, 4), ("c", 3, 3)])
    with pytest.raises(ValueError, match="days"):
        grouped_cv_folds(groups, y, n_folds=5)


def test_cv_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        grouped_cv_folds(np.asarray(["a", "b", "c"]), np.asarray([1, 0]))


# ---------------------------------------------------------------------------
# Aggregating folds
# ---------------------------------------------------------------------------

def test_aggregate_computes_mean_and_std_per_key() -> None:
    folds = [{"f1": 0.6, "recall": 1.0}, {"f1": 0.8, "recall": 0.0}]
    agg = aggregate_folds(folds)
    assert agg["n_folds"] == 2
    assert agg["f1"]["mean"] == pytest.approx(0.7)
    assert agg["f1"]["std"] == pytest.approx(0.1)
    assert agg["f1"]["min"] == pytest.approx(0.6)
    assert agg["f1"]["max"] == pytest.approx(0.8)
    assert agg["f1"]["n"] == 2
    assert agg["recall"]["std"] == pytest.approx(0.5)


def test_aggregate_recurses_into_nested_metric_dicts() -> None:
    folds = [
        {"confusion": {"tp": 10, "fp": 2}, "fpr_by_sublabel": {"resting": 0.2}},
        {"confusion": {"tp": 20, "fp": 4}, "fpr_by_sublabel": {"resting": 0.4}},
    ]
    agg = aggregate_folds(folds)
    assert agg["confusion"]["tp"]["mean"] == pytest.approx(15.0)
    assert agg["confusion"]["fp"]["std"] == pytest.approx(1.0)
    assert agg["fpr_by_sublabel"]["resting"]["mean"] == pytest.approx(0.3)


def test_aggregate_skips_nan_folds_and_says_how_many_counted() -> None:
    """A fold whose test split holds one class cannot produce an AUC."""
    folds = [{"roc_auc": math.nan}, {"roc_auc": 0.9}, {"roc_auc": 0.7}]
    agg = aggregate_folds(folds)
    assert agg["roc_auc"]["mean"] == pytest.approx(0.8)
    assert agg["roc_auc"]["n"] == 2


def test_aggregate_reports_nan_when_no_fold_measured_a_key() -> None:
    agg = aggregate_folds([{"roc_auc": math.nan}, {"roc_auc": math.nan}])
    assert math.isnan(agg["roc_auc"]["mean"])
    assert math.isnan(agg["roc_auc"]["std"])
    assert agg["roc_auc"]["n"] == 0


def test_aggregate_handles_keys_missing_from_some_folds() -> None:
    """A fold with no `object` clips simply has no rate to contribute."""
    folds = [
        {"fpr_by_sublabel": {"resting": 0.2, "object": 0.5}},
        {"fpr_by_sublabel": {"resting": 0.4}},
    ]
    agg = aggregate_folds(folds)
    assert agg["fpr_by_sublabel"]["object"]["mean"] == pytest.approx(0.5)
    assert agg["fpr_by_sublabel"]["object"]["n"] == 1
    assert agg["fpr_by_sublabel"]["resting"]["n"] == 2


def test_aggregate_merges_subtype_lists_in_canonical_order() -> None:
    folds = [{"subtypes": ["resting", "object"]}, {"subtypes": ["resting", "scratching"]}]
    assert aggregate_folds(folds)["subtypes"] == ["resting", "scratching", "object"]


def test_aggregate_of_one_fold_has_zero_spread() -> None:
    agg = aggregate_folds([{"f1": 0.75}])
    assert agg["f1"]["mean"] == pytest.approx(0.75)
    assert agg["f1"]["std"] == pytest.approx(0.0)
    assert agg["n_folds"] == 1


def test_aggregate_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError):
        aggregate_folds([])


def test_cross_validation_end_to_end() -> None:
    """Split, score every fold, aggregate — the spread is the error bar."""
    groups, y, subs = _dataset(_RECORDED_DAYS)
    rng = np.random.default_rng(0)
    centres = {"twisting": 0.7, "resting": 0.4, "scratching": 0.3, "object": 0.15}
    score = np.clip([rng.normal(centres[s], 0.2) for s in subs], 0.0, 1.0)

    per_fold = [
        evaluate(y[test], (score[test] >= 0.5).astype(int), score[test], subs[test])
        for _, test in grouped_cv_folds(groups, y, n_folds=5)
    ]
    agg = aggregate_folds(per_fold)

    assert agg["n_folds"] == 5
    assert 0.0 <= agg["f1"]["mean"] <= 1.0
    assert agg["f1"]["std"] >= 0.0
    assert agg["fpr_by_sublabel"]["resting"]["n"] == 5
    assert agg["best_f1"]["threshold"]["n"] == 5
    assert sum(fold["n"] for fold in per_fold) == 858


def test_cv_tolerates_one_dominant_day() -> None:
    """One day holding 95% of the clips cannot be split, only warned about."""
    days = [("big", 400, 200), ("b", 10, 5), ("c", 8, 4), ("d", 6, 3)]
    groups, y, _ = _dataset(days)
    with pytest.warns(UserWarning, match="size"):
        folds = grouped_cv_folds(groups, y, n_folds=3)

    assert len(folds) == 3
    for train, test in folds:
        assert not set(groups[train].tolist()) & set(groups[test].tolist())
        assert len(test) > 0
        assert sorted(np.concatenate([train, test]).tolist()) == list(range(len(y)))
    holders = [i for i, (_, test) in enumerate(folds) if "big" in set(groups[test].tolist())]
    assert len(holders) == 1


# ---------------------------------------------------------------------------
# Report over folds
# ---------------------------------------------------------------------------

def _aggregated_sample() -> dict:
    """Five-fold metrics for a model that struggles on resting hands."""
    groups, y, subs = _dataset(_RECORDED_DAYS)
    rng = np.random.default_rng(0)
    centres = {"twisting": 0.7, "resting": 0.4, "scratching": 0.3, "object": 0.15}
    score = np.clip([rng.normal(centres[s], 0.2) for s in subs], 0.0, 1.0)
    return aggregate_folds(
        [
            evaluate(y[test], (score[test] >= 0.5).astype(int), score[test], subs[test])
            for _, test in grouped_cv_folds(groups, y, n_folds=5)
        ]
    )


def test_folds_report_leads_with_the_per_subtype_false_positive_rates() -> None:
    text = format_folds_report(_aggregated_sample()).lower()
    head = text.index("ложные срабатывания")
    assert head < text.index("precision")
    assert head < text.index("roc auc")
    assert head < text.index("accuracy")


def test_folds_report_prints_value_plus_minus_sd_and_the_fold_count() -> None:
    text = format_folds_report(_aggregated_sample())
    assert "±" in text
    assert "фолд" in text.lower()
    assert "5" in text


def test_folds_report_names_every_negative_subtype() -> None:
    text = format_folds_report(_aggregated_sample())
    for subtype in _NEG_SUBS:
        assert subtype in text


def test_folds_report_is_multiline_russian_text() -> None:
    text = format_folds_report(_aggregated_sample())
    assert isinstance(text, str)
    assert len(text.splitlines()) >= 10
    assert any("а" <= ch.lower() <= "я" for ch in text)


def test_folds_report_shows_the_spread_of_the_best_threshold() -> None:
    text = format_folds_report(_aggregated_sample()).lower()
    assert "порог" in text


def test_folds_report_marks_unmeasurable_metrics() -> None:
    """A metric no fold could compute must read as unmeasurable, not as 0.000."""
    agg = aggregate_folds(
        [
            {"f1": 0.8, "roc_auc": math.nan, "fpr_by_sublabel": {"resting": 0.1}},
            {"f1": 0.9, "roc_auc": math.nan, "fpr_by_sublabel": {"resting": 0.2}},
        ]
    )
    text = format_folds_report(agg)
    assert "н/д" in text
    assert "0.850" in text   # f1 still rendered


def test_folds_report_says_how_many_folds_measured_a_metric() -> None:
    """`object` is rare enough that a fold can contain none of it."""
    agg = aggregate_folds(
        [
            {"fpr_by_sublabel": {"resting": 0.1, "object": 0.5}},
            {"fpr_by_sublabel": {"resting": 0.2}},
        ]
    )
    text = format_folds_report(agg)
    assert "1 из 2" in text
