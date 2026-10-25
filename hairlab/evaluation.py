"""hairlab.evaluation — splits that don't cheat and numbers that don't flatter.

Two decisions here, both about not fooling ourselves.

*Splits are cut by capture day, never by clip.* Clips recorded in one sitting
share lighting, hair state and very often the same episode, so two clips from
one day are closer to duplicates than to independent samples. Put one day on
both sides of the split and the test score measures recall of the training set;
it looks excellent and collapses the moment the model meets a new morning.
A whole day is therefore the atomic unit of every split.

*The headline metric is the false-positive rate per negative subtype.* The
detector being replaced fires when the user merely rests a hand on their head,
and that single failure is why the project exists. Overall accuracy averages it
away — with ~63% positives a model can look respectable while being wrong about
most resting hands — so ``format_report`` leads with ``resting``,
``scratching`` and ``object`` separately and prints accuracy last, as a
footnote.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Order of the returned index arrays.
SPLIT_NAMES = ("train", "val", "test")

# When there are fewer capture days than splits, somebody goes home empty. A
# model with no training data is useless; a test set is the next thing that
# cannot be faked; validation can always fall back to cross-validation inside
# train. So days are handed out in this order.
_FILL_PRIORITY = ("train", "test", "val")

# Greedy bin-packing is order-sensitive. Trial 0 uses the largest-day-first
# order (the one that generally packs best); the rest are seeded random orders,
# and the cheapest assignment of all of them wins. Cheap at 17 days, and it is
# what makes ``seed`` meaningful instead of decorative.
_RESTARTS = 24

# When the realised composition is this far off, say so out loud. The recorded
# set has days running from 14% to 100% positive, with five days holding 62% of
# all clips, so a split can come out badly skewed through no fault of the
# packer — and a skew that nobody noticed is the expensive kind.
_RATIO_WARN = 0.10   # absolute deviation of a split's positive share
_SIZE_WARN = 0.5     # relative deviation of a split's size from its target

# The negative subtypes, in the order a human wants to read them: the one the
# old detector fails on first, then the other two. Duplicated from
# ``hairlab.capture.LABELS`` rather than imported — importing capture drags in
# OpenCV and the whole detector stack, and this module has to run in a bare
# training environment.
NEGATIVE_SUBLABELS = ("resting", "scratching", "object")

_SUBLABEL_TITLES = {
    "twisting": "кручу прядь",
    "resting": "просто держу",
    "scratching": "чешу",
    "object": "расчёска/предмет",
    "unclear": "не разобрать",
}


# ---------------------------------------------------------------------------
# Grouped splitting
# ---------------------------------------------------------------------------

def _split_cost(counts: np.ndarray, target: np.ndarray) -> float:
    """How far one split sits from its per-class quota, relatively.

    Deviation is divided by the quota itself, so being ten clips over is a
    rounding error for train and a disaster for val. Both classes are scored
    separately: that is what keeps the positive/negative ratio near the
    overall one instead of merely getting the split sizes right.
    """
    scale = np.maximum(target, 1.0)
    return float(np.sum(((counts - target) / scale) ** 2))


def _greedy_assign(
    order: Sequence[Hashable],
    day_counts: Mapping[Hashable, np.ndarray],
    targets: Mapping[str, np.ndarray],
    priority: Sequence[str],
) -> tuple[dict[Hashable, str], float]:
    """Place whole days one at a time into the bin that needs them most.

    ``priority`` is every bin that may receive days, ordered by who gets fed
    first when days run short. Shared by the train/val/test split and by the
    cross-validation folds — the packing problem is the same one.
    """
    placed = {name: np.zeros(2, dtype=float) for name in priority}
    assignment: dict[Hashable, str] = {}
    remaining = len(order)

    for day in order:
        empty = [name for name in priority if placed[name].sum() == 0]
        if len(empty) >= remaining:
            # From here on every remaining day is needed to open an empty bin,
            # so stop optimising the ratio and fill by priority. This is what
            # keeps three days from all landing in train, and what guarantees
            # no cross-validation fold comes out without a test set.
            candidates = empty[:remaining]
        else:
            candidates = list(priority)

        best_name, best_delta = candidates[0], None
        for name in candidates:
            before = _split_cost(placed[name], targets[name])
            after = _split_cost(placed[name] + day_counts[day], targets[name])
            delta = after - before
            # Strict ``<`` over ``priority`` order: ties go to the earliest
            # bin, which also makes the outcome independent of dict ordering.
            if best_delta is None or delta < best_delta - 1e-12:
                best_name, best_delta = name, delta

        assignment[day] = best_name
        placed[best_name] = placed[best_name] + day_counts[day]
        remaining -= 1

    total = sum(_split_cost(placed[name], targets[name]) for name in priority)
    return assignment, total


def _best_assignment(
    day_counts: Mapping[Hashable, np.ndarray],
    targets: Mapping[str, np.ndarray],
    priority: Sequence[str],
    seed: int,
) -> dict[Hashable, str]:
    """Greedy packing from several seeded day orders, cheapest result kept."""
    days = list(day_counts)
    rng = np.random.default_rng(seed)
    best: dict[Hashable, str] | None = None
    best_cost = math.inf

    for trial in range(_RESTARTS):
        order = [days[i] for i in rng.permutation(len(days))]
        if trial == 0:
            # Descending size: the biggest days are the ones that cannot be
            # compensated for later, so they choose first. Stable sort, so the
            # shuffle above still breaks ties between equally sized days —
            # that is how the seed reaches this trial too.
            order.sort(key=lambda day: -day_counts[day].sum())
        assignment, cost = _greedy_assign(order, day_counts, targets, priority)
        if cost < best_cost - 1e-12:
            best, best_cost = assignment, cost

    assert best is not None
    return best


def _as_group_pair(groups: Sequence[Hashable], y: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    group_arr = np.asarray(groups)
    y_arr = np.asarray(y)
    if group_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"groups and y must be the same length, got {group_arr.shape[0]} and {y_arr.shape[0]}"
        )
    positive = (y_arr.astype(float) != 0).astype(int) if y_arr.size else y_arr.astype(int)
    return group_arr, positive


def _day_counts(group_arr: np.ndarray, positive: np.ndarray) -> dict[Hashable, np.ndarray]:
    """Per day ``[negatives, positives]``, in first-appearance order."""
    counts: dict[Hashable, np.ndarray] = {}
    for key, is_pos in zip(group_arr.tolist(), positive.tolist()):
        row = counts.get(key)
        if row is None:
            row = counts[key] = np.zeros(2, dtype=float)
        row[is_pos] += 1
    return counts


def grouped_split(
    groups: Sequence[Hashable],
    y: Sequence[Any],
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test without letting a day span two splits.

    ``groups`` is the capture day (or any grouping key) of each clip, ``y`` its
    0/1 label. Returns ascending index arrays in the order train, val, test.

    Strategy: whole days are packed greedily, largest day first, each into
    whichever split is furthest below its quota — and the quota is per class,
    so size and class balance are optimised together rather than one at the
    expense of the other. The pass is repeated from several seeded random day
    orders and the cheapest result kept. The no-leak property holds exactly;
    sizes and class ratio are approximations.

    Pass the result to :func:`split_composition` to see what each split
    actually received; a composition that is badly off warns on its own.

    Limits worth knowing. Days are indivisible, so exact fractions are
    impossible — on the recorded set (858 clips, 17 days, the largest holding
    13.5% of everything) expect a couple of points of slack, and far more if
    one day ever dominates. With fewer days than splits the leftovers come out
    empty, filled in the order train, test, val. The greedy pass is a
    heuristic, not an optimum, and with per-day positive share running from 14%
    to 100% there are datasets where no assignment of whole days is anywhere
    near balanced; that is what the warning is for, and it is the reason
    :func:`grouped_cv_folds` exists.
    """
    group_arr, positive = _as_group_pair(groups, y)
    if test_size < 0 or val_size < 0:
        raise ValueError("test_size and val_size must not be negative")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must leave something for train")

    empty = np.empty(0, dtype=np.int64)
    if group_arr.shape[0] == 0:
        return empty, empty.copy(), empty.copy()

    counts = _day_counts(group_arr, positive)
    totals = np.sum(list(counts.values()), axis=0)
    fractions = {"train": 1.0 - test_size - val_size, "val": val_size, "test": test_size}
    targets = {name: totals * fractions[name] for name in SPLIT_NAMES}
    priority = [name for name in _FILL_PRIORITY if fractions[name] > 0]

    assignment = _best_assignment(counts, targets, priority, seed)
    owner = np.asarray([assignment[key] for key in group_arr.tolist()])
    splits = tuple(
        np.flatnonzero(owner == name).astype(np.int64) for name in SPLIT_NAMES
    )

    _warn_if_skewed(split_composition(group_arr, positive, splits), fractions)
    return splits  # type: ignore[return-value]


def grouped_cv_folds(
    groups: Sequence[Hashable],
    y: Sequence[Any],
    n_folds: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Day-grouped cross-validation: ``[(train_idx, test_idx), ...]``.

    Every capture day lands in exactly one fold's test set, so each clip is
    tested exactly once across the folds and no day is ever on both sides of
    one fold. Folds are packed by the same per-class greedy as
    :func:`grouped_split`, with every fold targeting an equal share.

    This is the function to trust for a headline number. With per-day positive
    share running 14%..100% and five days holding 62% of the clips, a single
    20% test split is one or two big days and its score is largely luck; the
    spread across folds is the honest error bar. Feed each fold's metrics to
    :func:`aggregate_folds`.

    Raises if ``n_folds`` exceeds the number of days: returning fewer folds
    than asked for would quietly corrupt whatever averages the caller computes.
    """
    group_arr, positive = _as_group_pair(groups, y)
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")

    counts = _day_counts(group_arr, positive)
    if n_folds > len(counts):
        raise ValueError(
            f"n_folds={n_folds} exceeds the {len(counts)} capture days available; "
            "whole days are the atomic unit, so no more folds than days"
        )

    totals = np.sum(list(counts.values()), axis=0)
    names = [f"fold-{i}" for i in range(n_folds)]
    targets = {name: totals / n_folds for name in names}

    assignment = _best_assignment(counts, targets, names, seed)
    owner = np.asarray([assignment[key] for key in group_arr.tolist()])

    folds = [
        (
            np.flatnonzero(owner != name).astype(np.int64),
            np.flatnonzero(owner == name).astype(np.int64),
        )
        for name in names
    ]

    if n_folds < len(counts):
        # Leave-one-day-out folds are imbalanced by construction, not by any
        # fault of the packing, so there is nothing to report in that case.
        _warn_if_folds_skewed(
            split_composition(group_arr, positive, [test for _, test in folds], names=names)
        )
    return folds


# ---------------------------------------------------------------------------
# Realised composition
# ---------------------------------------------------------------------------

def split_composition(
    groups: Sequence[Hashable],
    y: Sequence[Any],
    splits: Sequence[Sequence[int]],
    names: Sequence[str] = SPLIT_NAMES,
) -> dict[str, Any]:
    """What each split actually ended up holding.

    Whole days are indivisible, so the realised composition is never exactly
    the requested one. Printing this next to a score is the difference between
    a number you can interpret and a number you merely have.
    """
    group_arr, positive = _as_group_pair(groups, y)
    if len(splits) != len(names):
        raise ValueError(f"got {len(splits)} splits and {len(names)} names")

    total = int(group_arr.shape[0])
    total_pos = int(positive.sum())
    parts: list[dict[str, Any]] = []
    for name, raw in zip(names, splits):
        idx = np.asarray(raw, dtype=np.int64).ravel()
        n = int(idx.size)
        n_pos = int(positive[idx].sum()) if n else 0
        days = sorted(set(group_arr[idx].tolist())) if n else []
        parts.append(
            {
                "name": str(name),
                "n": n,
                "frac": _ratio(n, total) if total else math.nan,
                "n_pos": n_pos,
                "n_neg": n - n_pos,
                "pos_share": (n_pos / n) if n else math.nan,
                "n_days": len(days),
                "days": days,
            }
        )

    return {
        "n": total,
        "n_pos": total_pos,
        "n_neg": total - total_pos,
        "pos_share": (total_pos / total) if total else math.nan,
        "n_days": len(set(group_arr.tolist())),
        "splits": parts,
    }


def _warn_if_skewed(composition: Mapping[str, Any], fractions: Mapping[str, float]) -> None:
    """Complain about a train/val/test split that came out unusable."""
    overall = composition["pos_share"]
    for part in composition["splits"]:
        name = part["name"]
        target = fractions.get(name, 0.0)
        if target <= 0:
            continue   # deliberately switched off by the caller
        if part["n"] == 0:
            warnings.warn(
                f"split {name!r} is empty: {composition['n_days']} capture day(s) "
                f"cannot fill train, val and test",
                stacklevel=3,
            )
            continue
        if abs(part["pos_share"] - overall) > _RATIO_WARN:
            warnings.warn(
                f"split {name!r} positive share {part['pos_share']:.2f} is far from the "
                f"overall {overall:.2f}; capture days are too heterogeneous to balance better",
                stacklevel=3,
            )
        if abs(part["frac"] - target) / target > _SIZE_WARN:
            warnings.warn(
                f"split {name!r} holds {part['frac']:.2f} of the clips, not the "
                f"requested {target:.2f}; whole days are indivisible",
                stacklevel=3,
            )


def _warn_if_folds_skewed(composition: Mapping[str, Any]) -> None:
    """One warning per problem for all the folds, not one per fold."""
    overall = composition["pos_share"]
    parts = composition["splits"]
    target = 1.0 / len(parts)

    skewed = [
        part["name"] for part in parts
        if part["n"] and abs(part["pos_share"] - overall) > _RATIO_WARN
    ]
    if skewed:
        warnings.warn(
            f"{len(skewed)} of {len(parts)} folds have a positive share more than "
            f"{_RATIO_WARN:.2f} from the overall {overall:.2f} ({', '.join(skewed)}); "
            "read the spread from aggregate_folds as a real error bar, not noise",
            stacklevel=3,
        )

    # Size matters as much as ratio: one dominant day makes a fold that is most
    # of the dataset, leaving its siblings too small to say anything.
    lopsided = [
        part["name"] for part in parts
        if abs(part["frac"] - target) / target > _SIZE_WARN
    ]
    if lopsided:
        warnings.warn(
            f"{len(lopsided)} of {len(parts)} folds are more than {100 * _SIZE_WARN:.0f}% off the "
            f"even size of {target:.2f} ({', '.join(lopsided)}); a single capture day most "
            "likely dominates the dataset, and these folds are not comparable",
            stacklevel=3,
        )


def format_composition(composition: Mapping[str, Any]) -> str:
    """Console table of the realised composition, for printing beside a score."""
    width = max([len(part["name"]) for part in composition["splits"]] + [6])
    lines = [
        f"Состав сплитов: {composition['n']} клипов, {composition['n_days']} дней, "
        f"доля положительных {composition['pos_share']:.3f}"
    ]
    for part in composition["splits"]:
        share = "н/д" if math.isnan(part["pos_share"]) else f"{part['pos_share']:.3f}"
        lines.append(
            f"  {part['name']:<{width}}  {part['n']:>5} клипов ({100 * part['frac']:4.1f}%)  "
            f"дней {part['n_days']:>3}  положительных {share}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _as_binary(values: Sequence[Any], name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return (arr.astype(float) != 0).astype(int)


def _ratio(numerator: float, denominator: float) -> float:
    """Rates with no samples are 0.0, not a crash — empty subtypes are normal."""
    return float(numerator) / float(denominator) if denominator else 0.0


def _point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sublabels: np.ndarray,
    subtypes: Sequence[str],
) -> dict[str, Any]:
    """Everything that depends on a single operating point."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)

    fpr_by_sublabel: dict[str, float] = {}
    fp_by_sublabel: dict[str, int] = {}
    n_by_sublabel: dict[str, int] = {}
    for subtype in subtypes:
        # Restricted to y_true == 0: a false positive is only definable on a
        # negative clip, whatever the sublabel claims.
        mask = (sublabels == subtype) & (y_true == 0)
        total = int(np.sum(mask))
        fired = int(np.sum(mask & (y_pred == 1)))
        n_by_sublabel[subtype] = total
        fp_by_sublabel[subtype] = fired
        fpr_by_sublabel[subtype] = _ratio(fired, total)

    return {
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "fpr": _ratio(fp, fp + tn),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr_by_sublabel": fpr_by_sublabel,
        "fp_by_sublabel": fp_by_sublabel,
        "n_by_sublabel": n_by_sublabel,
    }


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Score cut-off maximising F1, as ``y_score >= threshold``.

    Rolled by hand rather than taken from ``precision_recall_curve`` for the
    tie-break: among thresholds with identical F1 the highest one wins, because
    the same F1 with fewer alerts is strictly better for something that
    interrupts the user. Also sidesteps sklearn's single-class edge cases.
    """
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0 or y_score.size == 0:
        return math.nan

    order = np.argsort(-y_score, kind="stable")
    ranked_truth = y_true[order]
    ranked_score = y_score[order]

    tp = np.cumsum(ranked_truth == 1)
    fp = np.cumsum(ranked_truth == 0)
    # Only the last index of each run of equal scores is a reachable cut-off.
    last_of_run = np.append(np.flatnonzero(np.diff(ranked_score)), ranked_score.size - 1)

    tp_at, fp_at = tp[last_of_run], fp[last_of_run]
    denominator = 2 * tp_at + fp_at + (n_pos - tp_at)
    f1 = np.where(denominator > 0, 2 * tp_at / np.maximum(denominator, 1), 0.0)
    # Candidates are in descending score order and argmax takes the first
    # maximum, which is the highest threshold.
    return float(ranked_score[last_of_run[int(np.argmax(f1))]])


def evaluate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_score: Sequence[float],
    sublabels: Sequence[str],
) -> dict[str, Any]:
    """Score one split. See the module docstring for why the keys are ordered so.

    ``y_pred`` is taken as the operating point the caller actually ships —
    whatever threshold produced it — and ``y_score`` is used only for the
    ranking metrics and the threshold search, so a badly chosen threshold
    cannot drag ROC AUC down with it.
    """
    truth = _as_binary(y_true, "y_true")
    predicted = _as_binary(y_pred, "y_pred")
    score = np.asarray(y_score, dtype=float)
    sub = np.asarray(sublabels, dtype=object)

    lengths = {truth.shape[0], predicted.shape[0], score.shape[0], sub.shape[0]}
    if len(lengths) > 1:
        raise ValueError(
            "y_true, y_pred, y_score and sublabels must be the same length, got "
            f"{truth.shape[0]}, {predicted.shape[0]}, {score.shape[0]}, {sub.shape[0]}"
        )

    # Canonical subtypes first (stable report layout), then anything unexpected
    # that showed up in the labels, so a new label can never go unnoticed.
    present = {str(name) for name in np.unique(sub[truth == 0]).tolist()}
    subtypes = [name for name in NEGATIVE_SUBLABELS if name in present]
    subtypes += sorted(present - set(NEGATIVE_SUBLABELS))

    n_pos = int(np.sum(truth == 1))
    n_neg = int(np.sum(truth == 0))

    metrics: dict[str, Any] = {
        "n": int(truth.shape[0]),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "subtypes": subtypes,
    }
    metrics.update(_point_metrics(truth, predicted, sub, subtypes))

    # Both need two classes to mean anything; NaN says "not measurable here"
    # without forcing every caller into a None check.
    if n_pos and n_neg:
        metrics["roc_auc"] = float(roc_auc_score(truth, score))
        metrics["average_precision"] = float(average_precision_score(truth, score))
    else:
        metrics["roc_auc"] = math.nan
        metrics["average_precision"] = math.nan

    threshold = _best_f1_threshold(truth, score)
    at_best = score >= threshold if not math.isnan(threshold) else np.zeros_like(truth)
    best = _point_metrics(truth, at_best.astype(int), sub, subtypes)
    best["threshold"] = threshold
    if math.isnan(threshold):
        best["f1"] = math.nan
    metrics["best_f1"] = best
    return metrics


# ---------------------------------------------------------------------------
# Aggregating folds
# ---------------------------------------------------------------------------

def _stats(values: Sequence[float]) -> dict[str, float]:
    """mean/std/min/max over the folds that produced a usable number.

    ``std`` is the population spread (``ddof=0``) of the folds in hand, which
    is what sklearn prints too. Treat it as a descriptive error bar, not as an
    unbiased estimate of generalisation variance: consecutive folds share most
    of their training days, so the folds are not independent samples and the
    true variance is larger than this.
    """
    arr = np.asarray(values, dtype=float)
    usable = arr[~np.isnan(arr)]
    if usable.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    return {
        "mean": float(usable.mean()),
        "std": float(usable.std()),
        "min": float(usable.min()),
        "max": float(usable.max()),
        "n": int(usable.size),
    }


def _merge_label_lists(values: Sequence[Sequence[Any]]) -> list[str]:
    """Union of the folds' label lists, back in canonical reading order."""
    seen = {str(item) for value in values for item in value}
    ordered = [name for name in NEGATIVE_SUBLABELS if name in seen]
    return ordered + sorted(seen - set(NEGATIVE_SUBLABELS))


def _aggregate_level(dicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys: list[str] = []
    for fold in dicts:
        for key in fold:
            if key not in keys:
                keys.append(key)

    out: dict[str, Any] = {}
    for key in keys:
        # A key may be missing from some folds — a fold with no `object` clips
        # has no rate to contribute — so aggregate over whoever has it.
        values = [fold[key] for fold in dicts if key in fold]
        if all(isinstance(value, Mapping) for value in values):
            out[key] = _aggregate_level(values)
        elif all(isinstance(value, (list, tuple)) for value in values):
            out[key] = _merge_label_lists(values)
        elif any(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            numeric = [
                float(value)
                for value in values
                if isinstance(value, (int, float, np.integer, np.floating))
            ]
            out[key] = _stats(numeric)
        # Anything else (stray strings, objects) has no meaningful mean and is
        # dropped rather than guessed at.
    return out


def aggregate_folds(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine per-fold :func:`evaluate` dicts into mean and spread per key.

    Mirrors the input shape: every scalar becomes
    ``{"mean", "std", "min", "max", "n"}``, nested dicts (``confusion``,
    ``fpr_by_sublabel``, ``best_f1``) are recursed into, and ``n_folds`` is
    added at the top. ``n`` is how many folds actually produced a number —
    a fold whose test days happen to be all one class contributes no ROC AUC,
    and that is reported rather than silently averaged in as a zero.
    """
    folds = list(fold_metrics)
    if not folds:
        raise ValueError("aggregate_folds needs at least one fold's metrics")

    aggregated = _aggregate_level(folds)
    aggregated["n_folds"] = len(folds)
    return aggregated


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_pct(value: float) -> str:
    return "н/д" if value is None or math.isnan(value) else f"{100.0 * value:5.1f}%"


def _fmt_num(value: float) -> str:
    return "н/д" if value is None or math.isnan(value) else f"{value:.3f}"


def _title(subtype: str) -> str:
    return f"{_SUBLABEL_TITLES.get(subtype, subtype)} ({subtype})"


def format_report(metrics: dict) -> str:
    """Human-readable console report, false positives first.

    Deliberately not a table of everything: the first thing read must be the
    per-subtype false-positive rates, and accuracy is last so nobody quotes it.
    """
    subtypes = metrics.get("subtypes") or list(metrics.get("fpr_by_sublabel", {}))
    width = max([len(_title(name)) for name in subtypes] + [22])
    lines: list[str] = []

    lines.append("=" * (width + 22))
    lines.append("ОЦЕНКА КЛАССИФИКАТОРА НАКРУЧИВАНИЯ")
    lines.append(
        f"Клипов: {metrics['n']}  "
        f"(накручивание {metrics['n_pos']}, прочее {metrics['n_neg']})"
    )
    lines.append("=" * (width + 22))
    lines.append("")

    lines.append("ЛОЖНЫЕ СРАБАТЫВАНИЯ ПО ТИПАМ ОТРИЦАТЕЛЬНЫХ КЛИПОВ")
    lines.append("главная метрика: детектор не должен срабатывать, если рука просто у головы")
    for name in subtypes:
        fired = metrics["fp_by_sublabel"].get(name, 0)
        total = metrics["n_by_sublabel"].get(name, 0)
        rate = metrics["fpr_by_sublabel"].get(name, math.nan)
        lines.append(f"  {_title(name):<{width}}  {fired:>4} / {total:<5} {_fmt_pct(rate)}")
    confusion = metrics["confusion"]
    lines.append(
        f"  {'все отрицательные':<{width}}  "
        f"{confusion['fp']:>4} / {metrics['n_neg']:<5} {_fmt_pct(metrics['fpr'])}"
    )
    lines.append("")

    lines.append("ПОЛОЖИТЕЛЬНЫЙ КЛАСС (накручивание)")
    lines.append(f"  precision  {_fmt_num(metrics['precision'])}")
    lines.append(f"  recall     {_fmt_num(metrics['recall'])}")
    lines.append(f"  F1         {_fmt_num(metrics['f1'])}")
    lines.append("")

    lines.append("РАНЖИРОВАНИЕ (не зависит от порога)")
    lines.append(f"  ROC AUC            {_fmt_num(metrics['roc_auc'])}")
    lines.append(f"  average precision  {_fmt_num(metrics['average_precision'])}")
    lines.append("")

    lines.append("МАТРИЦА ОШИБОК")
    lines.append(f"  TN {confusion['tn']:>5}   FP {confusion['fp']:>5}")
    lines.append(f"  FN {confusion['fn']:>5}   TP {confusion['tp']:>5}")
    lines.append("")

    best = metrics["best_f1"]
    if math.isnan(best["threshold"]):
        lines.append("ЛУЧШИЙ ПОРОГ ПО F1: н/д (нет положительных примеров)")
    else:
        lines.append(f"ЛУЧШИЙ ПОРОГ ПО F1: {best['threshold']:.2f}")
        lines.append(
            f"  F1 {_fmt_num(best['f1'])}   "
            f"precision {_fmt_num(best['precision'])}   "
            f"recall {_fmt_num(best['recall'])}"
        )
        at_best = ", ".join(
            f"{_SUBLABEL_TITLES.get(name, name)} {_fmt_pct(best['fpr_by_sublabel'][name]).strip()}"
            for name in subtypes
        )
        if at_best:
            lines.append(f"  ложные срабатывания: {at_best}")
    lines.append("")

    lines.append(
        f"Общая точность (accuracy) {_fmt_num(metrics['accuracy'])} — справочно: "
        "она усредняет ровно те ошибки, из-за которых всё это и делается."
    )
    return "\n".join(lines)


_MISSING_STATS = {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0}


def _agg_stats(aggregated: Mapping[str, Any], *path: str) -> Mapping[str, float]:
    """Stats at ``path``, or an unmeasurable placeholder if nothing is there.

    Keeps the formatter total: a hand-assembled or partial aggregate still
    prints, with the gaps shown as gaps.
    """
    node: Any = aggregated
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING_STATS
        node = node[key]
    if isinstance(node, Mapping) and "mean" in node:
        return node
    return _MISSING_STATS


def _fmt_spread(stats: Mapping[str, float], n_folds: int, pct: bool = False) -> str:
    """``mean ± sd [min .. max]``, plus the fold count when some folds abstained."""
    if not stats["n"]:
        return "н/д"
    if pct:
        body = (
            f"{100 * stats['mean']:5.1f}% ± {100 * stats['std']:4.1f}%"
            f"   [{100 * stats['min']:5.1f}% .. {100 * stats['max']:5.1f}%]"
        )
    else:
        body = (
            f"{stats['mean']:.3f} ± {stats['std']:.3f}"
            f"   [{stats['min']:.3f} .. {stats['max']:.3f}]"
        )
    if stats["n"] < n_folds:
        # Silence here would read as a measurement; it is an absence of one.
        body += f"   ({stats['n']} из {n_folds} фолдов)"
    return body


def format_folds_report(aggregated: Mapping[str, Any]) -> str:
    """Console report over cross-validation folds: ``значение ± sd``.

    A sibling of :func:`format_report` rather than an extension of it. The two
    inputs are different shapes — one number per metric versus a distribution
    over folds — and the whole point of this one is the spread, which has no
    equivalent on a single split. Same priority order: per-subtype false
    positives first, accuracy last.
    """
    n_folds = int(aggregated.get("n_folds", 0) or 0)
    subtypes = aggregated.get("subtypes") or list(aggregated.get("fpr_by_sublabel", {}))
    width = max([len(_title(name)) for name in subtypes] + [22])

    lines = ["=" * (width + 34)]
    lines.append(f"ОЦЕНКА ПО {n_folds} ФОЛДАМ (разбиение по дням съёмки)")
    clips = _agg_stats(aggregated, "n")
    if clips["n"]:
        lines.append(
            f"Клипов в фолде: {clips['mean']:.0f} ± {clips['std']:.0f}  "
            f"[{clips['min']:.0f} .. {clips['max']:.0f}]"
        )
    lines.append("=" * (width + 34))
    lines.append("")

    lines.append("ЛОЖНЫЕ СРАБАТЫВАНИЯ ПО ТИПАМ ОТРИЦАТЕЛЬНЫХ КЛИПОВ")
    lines.append("главная метрика: детектор не должен срабатывать, если рука просто у головы")
    for name in subtypes:
        stats = _agg_stats(aggregated, "fpr_by_sublabel", name)
        lines.append(f"  {_title(name):<{width}}  {_fmt_spread(stats, n_folds, pct=True)}")
    lines.append(
        f"  {'все отрицательные':<{width}}  "
        f"{_fmt_spread(_agg_stats(aggregated, 'fpr'), n_folds, pct=True)}"
    )
    lines.append("")

    lines.append("ПОЛОЖИТЕЛЬНЫЙ КЛАСС (накручивание)")
    for label, key in (("precision", "precision"), ("recall", "recall"), ("F1", "f1")):
        lines.append(f"  {label:<10} {_fmt_spread(_agg_stats(aggregated, key), n_folds)}")
    lines.append("")

    lines.append("РАНЖИРОВАНИЕ (не зависит от порога)")
    lines.append(
        f"  {'ROC AUC':<18} {_fmt_spread(_agg_stats(aggregated, 'roc_auc'), n_folds)}"
    )
    lines.append(
        f"  {'average precision':<18} "
        f"{_fmt_spread(_agg_stats(aggregated, 'average_precision'), n_folds)}"
    )
    lines.append("")

    lines.append("ЛУЧШИЙ ПОРОГ ПО F1")
    lines.append(
        f"  {'порог':<10} {_fmt_spread(_agg_stats(aggregated, 'best_f1', 'threshold'), n_folds)}"
    )
    lines.append(
        f"  {'F1':<10} {_fmt_spread(_agg_stats(aggregated, 'best_f1', 'f1'), n_folds)}"
    )
    at_best = [
        (name, _agg_stats(aggregated, "best_f1", "fpr_by_sublabel", name)) for name in subtypes
    ]
    if any(stats["n"] for _, stats in at_best):
        lines.append("  ложные срабатывания на этом пороге:")
        for name, stats in at_best:
            if stats["n"]:
                lines.append(
                    f"    {_title(name):<{width}}  {_fmt_spread(stats, n_folds, pct=True)}"
                )
    lines.append("")

    lines.append(
        f"Общая точность (accuracy) {_fmt_spread(_agg_stats(aggregated, 'accuracy'), n_folds)}"
    )
    lines.append(
        "— справочно: она усредняет ровно те ошибки, из-за которых всё это и делается."
    )
    return "\n".join(lines)
