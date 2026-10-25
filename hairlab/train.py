"""hairlab.train — train the twisting detector and report honestly.

Two entry points, and the difference between them matters.

``python -m hairlab train --cv`` runs grouped cross-validation over capture
days and reports mean ± spread. That is the number to believe. Days differ
enormously in this dataset — positive share runs from 14% to 100% — so a
single split's score is largely decided by which days happened to land in
test. Measured on a fixed model, the resting false-positive rate swung
between 19% and 41% depending on the fold.

``python -m hairlab train`` trains one model on a single grouped split and
writes the artefacts to disk. Use it to produce the model you ship, after
cross-validation has told you what to expect from it.

Either way the headline is the false-positive rate per negative subtype,
not accuracy: the detector interrupts with a fullscreen alert and a
repeating sound, so firing while the user merely rests a hand on their head
is the expensive mistake.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Kept apart from ~/.hairguard/models, which holds the downloaded MediaPipe
# task files: mixing our artefacts with theirs makes it unclear what may be
# deleted to force a re-download.
DEFAULT_OUT = Path.home() / ".hairguard" / "trained"
DEFAULT_SEED = 0

# Operating point. The F1-optimal threshold is not the right default for
# something that interrupts the user: it buys recall with false alerts the
# user experiences as the app crying wolf. Instead the shipped threshold is
# the lowest one whose resting false-positive rate stays under this ceiling.
DEFAULT_FPR_CEILING = 0.05
CEILING_SUBLABEL = "resting"


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------

def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sublabels: np.ndarray,
    ceiling: float = DEFAULT_FPR_CEILING,
    sublabel: str = CEILING_SUBLABEL,
) -> tuple[float, bool]:
    """Lowest threshold whose false-positive rate on ``sublabel`` stays under ceiling.

    Returns (threshold, ceiling_met). Lowest, because among thresholds that
    are quiet enough we want the one that catches the most — recall is the
    current detector's actual weakness.
    """
    mask = np.asarray(sublabels) == sublabel
    y_score = np.asarray(y_score, dtype=float)
    if not mask.any():
        return 0.5, False

    negatives = y_score[mask]
    candidates = np.unique(np.concatenate([negatives, [0.0, 1.0]]))

    best: Optional[float] = None
    for threshold in np.sort(candidates):
        if (negatives >= threshold).mean() <= ceiling:
            best = float(threshold)
            break

    if best is None:
        # Even the strictest threshold cannot hold the ceiling.
        return float(np.nextafter(negatives.max(), np.inf)), False
    return best, True


# ---------------------------------------------------------------------------
# One train/evaluate cycle
# ---------------------------------------------------------------------------

def _fit_and_score(data, train_idx, val_idx, test_idx, *, seed: int, max_epochs: int):
    """Train on one split and return (metrics, module, threshold)."""
    from hairlab.model import predict_scores, train_model

    module = train_model(
        data.X[train_idx], data.y[train_idx],
        data.X[val_idx], data.y[val_idx],
        feature_dim=data.X.shape[2],
        max_epochs=max_epochs,
        seed=seed,
    )

    # The threshold is chosen on validation, never on test: picking it on the
    # same data you report would quietly tune the model to the test set.
    val_scores = predict_scores(module, data.X[val_idx])
    threshold, met = choose_threshold(
        data.y[val_idx], val_scores, data.sublabels[val_idx]
    )

    from hairlab.evaluation import evaluate

    test_scores = predict_scores(module, data.X[test_idx])
    metrics = evaluate(
        data.y[test_idx],
        (test_scores >= threshold).astype(int),
        test_scores,
        data.sublabels[test_idx],
    )
    metrics["threshold"] = threshold
    metrics["threshold_ceiling_met"] = met
    return metrics, module, threshold


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def run_cv(data, *, n_folds: int, seed: int, max_epochs: int) -> dict[str, Any]:
    from hairlab.evaluation import (
        aggregate_folds, format_composition, grouped_cv_folds, split_composition,
    )

    folds = grouped_cv_folds(data.groups, data.y, n_folds=n_folds, seed=seed)
    print(format_composition(
        split_composition(
            data.groups, data.y,
            [test for _, test in folds],
            names=[f"fold-{i}" for i in range(len(folds))],
        )
    ))

    per_fold = []
    for index, (train_idx, test_idx) in enumerate(folds):
        # No separate validation days inside a fold: carve the smallest
        # capture days out of train so whole days still never straddle.
        inner_train, inner_val = _carve_validation(data, train_idx, seed=seed + index)
        print(f"\n--- фолд {index + 1}/{len(folds)} "
              f"(обучение {len(inner_train)}, валидация {len(inner_val)}, тест {len(test_idx)}) ---")
        started = time.time()
        metrics, _, threshold = _fit_and_score(
            data, inner_train, inner_val, test_idx, seed=seed + index, max_epochs=max_epochs
        )
        print(f"    порог {threshold:.3f}, F1 {metrics['f1']:.3f}, "
              f"полнота {metrics['recall']:.3f}, "
              f"ложные на 'просто держу' {metrics['fpr_by_sublabel'].get('resting', float('nan')):.3f} "
              f"({time.time() - started:.0f}s)")
        per_fold.append(metrics)

    return aggregate_folds(per_fold)


def _carve_validation(data, train_idx: np.ndarray, *, seed: int, share: float = 0.18):
    """Split training indices into train/val without splitting a capture day."""
    from hairlab.evaluation import grouped_split

    sub_train, sub_val, _ = grouped_split(
        data.groups[train_idx], data.y[train_idx],
        test_size=0.0, val_size=share, seed=seed,
    )
    if len(sub_val) == 0:          # too few days to spare one
        sub_val = sub_train[: max(1, len(sub_train) // 5)]
        sub_train = sub_train[len(sub_val):]
    return train_idx[sub_train], train_idx[sub_val]


def run_single(data, *, seed: int, max_epochs: int, out_dir: Path) -> dict[str, Any]:
    from hairlab.evaluation import format_composition, grouped_split, split_composition
    from hairlab.model import export_onnx, save_checkpoint

    train_idx, val_idx, test_idx = grouped_split(data.groups, data.y, seed=seed)
    print(format_composition(
        split_composition(data.groups, data.y, [train_idx, val_idx, test_idx])
    ))

    metrics, module, threshold = _fit_and_score(
        data, train_idx, val_idx, test_idx, seed=seed, max_epochs=max_epochs
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(module, out_dir / "twistnet.ckpt")
    export_onnx(module, out_dir / "twistnet.onnx", feature_dim=data.X.shape[2],
                seq_len=data.X.shape[1])
    (out_dir / "model.json").write_text(json.dumps({
        "threshold": threshold,
        "feature_dim": int(data.X.shape[2]),
        "seq_len": int(data.X.shape[1]),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }, indent=2), encoding="utf-8")
    print(f"\nСохранено в {out_dir}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hairlab train", description="Обучение детектора накручивания волос"
    )
    parser.add_argument("--cv", action="store_true",
                        help="Кросс-валидация по дням вместо одного разбиения")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from hairlab.dataset import load_dataset

    print("Загружаю датасет...")
    data = load_dataset()
    print(f"  {data.X.shape[0]} клипов, {data.X.shape[1]} кадров, {data.X.shape[2]} признаков")
    print(f"  положительных {int(data.y.sum())}, отрицательных {int((data.y == 0).sum())}, "
          f"дней съёмки {len(set(data.groups))}\n")

    if args.cv:
        aggregated = run_cv(data, n_folds=args.folds, seed=args.seed,
                            max_epochs=args.max_epochs)
        print("\n" + "=" * 62)
        print(json.dumps(aggregated, indent=2, ensure_ascii=False, default=float))
    else:
        from hairlab.evaluation import format_report

        metrics = run_single(data, seed=args.seed, max_epochs=args.max_epochs,
                             out_dir=args.out)
        print("\n" + format_report(metrics))


if __name__ == "__main__":
    sys.exit(main())
