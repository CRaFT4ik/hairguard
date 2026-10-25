"""hairlab.truncation — how early in a visit the model can be trusted.

The app interrupts a behaviour, so reaction time is most of its value. The
model scores a fixed 20-frame window, and the detection loop measures 5.5 to
7.3 frames per second, so waiting for a full window means roughly three
seconds of hand-at-head before anything can appear on screen. The rule it
replaced fired in about one. That trade is only worth making if a full window
is genuinely necessary, and that is a measurable question, not a matter of
taste: every labelled clip is a full window, so truncating one to its first
*k* frames and filling the rest exactly as the live path would produces
precisely the input the model would see *k* frames into a real visit.

What fills the rest turns out to matter far more than *k* does.

*Zero-padding destroys the model.* ``clip_features`` pads short clips with
zero rows, which is right for training — an all-zero row has
``hand_present`` clear and reads as "no data" rather than inventing a frozen
hand. At inference it is fatal. The network standardises its input first, so
a zero row becomes ``-mean/std``: a large constant vector nothing in training
resembled. Global average pooling over time then lets those rows dominate the
pooled representation, and the logit saturates. Measured on the shipped
model's held-out days, every clip scores 1.0 for k <= 16 and 0.0 for k >= 18
— ROC AUC exactly 0.500, a detector that has stopped reading its input.

*Repeat-padding keeps it intact.* Filling the remaining slots with the last
real frame — a hand holding still, which is something the model saw several
hundred of — leaves ranking quality flat all the way down to a handful of
frames. The padded rows are all identical (a frame against a copy of itself
has zero velocity), so they are in-distribution and cheap: one extra
``frame_features`` call, broadcast.

The conclusion the numbers support is not "the window can be shortened" but
something more specific and more useful: *most of what this model knows is
hand shape, not hand motion*. 74 of the 96 features are static, the clips are
all conditioned on a hand already at the head, and a pinch on a strand simply
looks different from a palm resting on the scalp. Motion adds real recall,
but it saturates within a few frames rather than needing two seconds.

Run it with ``python -m hairlab truncation``. Cross-validated over capture
days, because days differ enormously here and a single split's answer to this
question would be decided by which days landed in test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from hairlab.features import FEATURE_DIM, frame_features

# How the remaining window slots are filled when fewer than seq_len real
# frames exist. "zero-end" is what clip_features does and is included to
# document that it must never be used at inference time.
SCHEMES = ("zero-end", "repeat-last", "repeat-first")
DEFAULT_SCHEME = "repeat-last"

DEFAULT_SEQ_LEN = 20
DEFAULT_FOLDS = 5
DEFAULT_SEED = 0

# Frames per second the detection loop actually achieves, measured across the
# recorded clips (min 5.50, median 6.75, max 7.28). Used only to turn a frame
# count into the seconds the user would wait, which is the number that matters.
MEASURED_FPS = 6.75


# ---------------------------------------------------------------------------
# Every row a clip can contribute
# ---------------------------------------------------------------------------

class Rows:
    """Cached feature rows for every clip, enough to build any padded window.

    A padded window only ever needs three kinds of row::

        Z[c]    = frame_features(f[0], None)       row 0 of any window
        A[c, i] = frame_features(f[i], f[i-1])     real frame, real predecessor
        S[c, i] = frame_features(f[i], f[i])       frozen frame, zero velocity

    Caching them turns a sweep over k and padding schemes from thousands of
    extractor calls into array indexing, and — because every scheme is then
    an indexing of the same rows — makes it cheap to assert against
    ``clip_features`` that the indexing is right.
    """

    def __init__(self, Z: np.ndarray, A: np.ndarray, S: np.ndarray, n: np.ndarray) -> None:
        self.Z, self.A, self.S, self.n = Z, A, S, n

    @property
    def seq_len(self) -> int:
        return int(self.A.shape[1])

    def __len__(self) -> int:
        return int(self.A.shape[0])


def clip_rows(frames: Sequence[dict], seq_len: int = DEFAULT_SEQ_LEN):
    """``(Z, A, S)`` for one clip's frames. See :class:`Rows`."""
    count = min(len(frames), seq_len)
    Z = frame_features(frames[0], None)
    A = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
    S = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
    for index in range(count):
        A[index] = Z if index == 0 else frame_features(frames[index], frames[index - 1])
        S[index] = frame_features(frames[index], frames[index])
    return Z, A, S, count


def cache_rows(
    clip_ids: Sequence[str],
    clips_root: Path,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> Rows:
    """Read every clip once and cache its rows."""
    total = len(clip_ids)
    Z = np.zeros((total, FEATURE_DIM), dtype=np.float32)
    A = np.zeros((total, seq_len, FEATURE_DIM), dtype=np.float32)
    S = np.zeros((total, seq_len, FEATURE_DIM), dtype=np.float32)
    n = np.zeros(total, dtype=np.int64)

    for row, clip_id in enumerate(clip_ids):
        meta = json.loads(
            (clips_root / str(clip_id) / "meta.json").read_text(encoding="utf-8")
        )
        Z[row], A[row], S[row], n[row] = clip_rows(meta["frames"], seq_len=seq_len)
    return Rows(Z, A, S, n)


def padded_windows(rows: Rows, k: int, scheme: str = DEFAULT_SCHEME) -> np.ndarray:
    """``(N, seq_len, FEATURE_DIM)`` windows as they would look *k* frames in.

    Equal, row for row, to ``clip_features`` applied to the correspondingly
    padded frame list — which is what ``tests/test_truncation.py`` checks on
    real clips, because this indexing is the whole experiment and an off-by-one
    here would silently answer a different question.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown padding scheme {scheme!r}, expected one of {SCHEMES}")
    seq_len = rows.seq_len
    if not 1 <= k <= seq_len:
        raise ValueError(f"k must be in 1..{seq_len}, got {k}")

    out = np.zeros((len(rows), seq_len, FEATURE_DIM), dtype=np.float32)
    if scheme == "repeat-first":
        pad = seq_len - k
        # Front-filled: the real frames sit at the end, where "the window
        # ending now" puts them.
        out[:, :pad] = rows.S[:, 0][:, None, :]
        out[:, pad] = rows.S[:, 0]       # real f[0], its predecessor a copy
        if k > 1:
            out[:, pad + 1:] = rows.A[:, 1:k]
    else:
        out[:, :k] = rows.A[:, :k]
        if scheme == "repeat-last":
            out[:, k:] = rows.S[:, k - 1][:, None, :]
    # Row 0 never has a predecessor, whatever the scheme.
    out[:, 0] = rows.Z
    return out


def seconds_in(k: int, fps: float = MEASURED_FPS) -> float:
    """How long the user has had a hand at their head by frame *k*."""
    return max(0, k - 1) / fps


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def sweep_fold(
    module: Any,
    rows: Rows,
    y: np.ndarray,
    sublabels: np.ndarray,
    test_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    scheme: str,
    ks: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """Metrics per *k* for one trained fold, at that fold's own operating point.

    The threshold is chosen once, on validation, from *full* windows — the
    operating point the app actually ships — and then held fixed across every
    *k*. That is the decision-relevant question: the live window fills
    gradually, the app carries one threshold, so the threshold cannot be a
    function of how full the window happens to be.
    """
    from hairlab.evaluation import evaluate
    from hairlab.model import predict_scores
    from hairlab.train import choose_threshold

    full = padded_windows(rows, rows.seq_len, scheme)
    threshold, _ = choose_threshold(y[val_idx], predict_scores(module, full[val_idx]),
                                    sublabels[val_idx])

    out: dict[int, dict[str, Any]] = {}
    for k in ks:
        windows = padded_windows(rows, k, scheme)
        scores = predict_scores(module, windows[test_idx])
        metrics = evaluate(
            y[test_idx], (scores >= threshold).astype(int), scores, sublabels[test_idx]
        )
        metrics["threshold"] = threshold
        out[k] = metrics
    return out


def run_sweep(
    data: Any,
    rows: Rows,
    *,
    scheme: str = DEFAULT_SCHEME,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    max_epochs: int = 100,
    ks: Optional[Sequence[int]] = None,
) -> dict[int, dict[str, Any]]:
    """Cross-validated curve: ``{k: aggregated metrics}``.

    One model per fold, each trained on full windows exactly as the shipped
    model was, then evaluated at every *k* on days it never saw.
    """
    from hairlab.evaluation import aggregate_folds, grouped_cv_folds
    from hairlab.model import train_model
    from hairlab.train import _carve_validation

    if ks is None:
        ks = range(1, rows.seq_len + 1)
    ks = list(ks)

    folds = grouped_cv_folds(data.groups, data.y, n_folds=n_folds, seed=seed)
    per_fold: list[dict[int, dict[str, Any]]] = []

    for index, (train_idx, test_idx) in enumerate(folds):
        inner_train, inner_val = _carve_validation(data, train_idx, seed=seed + index)
        print(
            f"--- фолд {index + 1}/{len(folds)} "
            f"(обучение {len(inner_train)}, валидация {len(inner_val)}, "
            f"тест {len(test_idx)}) ---",
            flush=True,
        )
        module = train_model(
            data.X[inner_train], data.y[inner_train],
            data.X[inner_val], data.y[inner_val],
            feature_dim=data.X.shape[2], max_epochs=max_epochs, seed=seed + index,
        )
        fold = sweep_fold(
            module, rows, data.y, data.sublabels, test_idx, inner_val,
            scheme=scheme, ks=ks,
        )
        full = fold[rows.seq_len]
        print(
            f"    порог {full['threshold']:.3f}; на полном окне полнота "
            f"{full['recall']:.3f}, ложные на 'просто держу' "
            f"{full['fpr_by_sublabel'].get('resting', float('nan')):.3f}",
            flush=True,
        )
        per_fold.append(fold)

    return {k: aggregate_folds([fold[k] for fold in per_fold]) for k in ks}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(stats: Any) -> str:
    if not isinstance(stats, dict) or not stats.get("n"):
        return "     н/д    "
    return f"{100 * stats['mean']:5.1f}% ±{100 * stats['std']:4.1f}"


def format_curve(
    curve: dict[int, dict[str, Any]],
    *,
    scheme: str,
    fps: float = MEASURED_FPS,
) -> str:
    """The curve, false positives first and accuracy nowhere."""
    ks = sorted(curve)
    n_folds = int(curve[ks[0]].get("n_folds", 0) or 0)

    # ASCII only, deliberately. This console is cp1251, and printing a box
    # character here raises UnicodeEncodeError from inside the formatter —
    # turning a five-minute measurement into a crash for reasons that have
    # nothing to do with the measurement. hairlab.model learned the same
    # lesson from the ONNX exporter's emoji.
    lines = ["=" * 84]
    lines.append(
        f"НАСКОЛЬКО РАНО МОДЕЛИ МОЖНО ВЕРИТЬ - заполнение окна '{scheme}', "
        f"{n_folds} фолдов по дням съёмки"
    )
    lines.append(
        f"Порог выбран один раз на полном окне и не меняется с k, "
        f"как и в приложении. {fps:.2f} кадра/с."
    )
    lines.append("=" * 84)
    lines.append("")
    lines.append(
        "            |  ЛОЖНЫЕ: просто держу |  ложные: чешу   |"
        "  полнота        |  точность"
    )
    lines.append(
        "  k    сек  |  ГЛАВНАЯ МЕТРИКА      |                 |"
        "                 |"
    )
    lines.append("-" * 84)
    for k in ks:
        metrics = curve[k]
        resting = metrics.get("fpr_by_sublabel", {}).get("resting")
        scratching = metrics.get("fpr_by_sublabel", {}).get("scratching")
        lines.append(
            f" {k:2d}  {seconds_in(k, fps):5.2f}  |  {_pct(resting)}   | "
            f"{_pct(scratching)}  | {_pct(metrics.get('recall'))} | "
            f"{_pct(metrics.get('precision'))}"
        )
    lines.append("-" * 84)
    lines.append(
        "Разброс — стандартное отклонение по фолдам. Фолды делят большую часть "
        "обучающих дней,\nтак что это описательная оценка разброса, а не "
        "несмещённая оценка дисперсии."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hairlab truncation",
        description="Насколько рано в эпизоде модель уже можно спрашивать",
    )
    parser.add_argument("--scheme", choices=SCHEMES, default=DEFAULT_SCHEME,
                        help="чем заполнять незанятые кадры окна")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--json", type=Path, default=None,
                        help="куда сохранить метрики целиком")
    args = parser.parse_args(argv)

    from hairlab.dataset import DEFAULT_CLIPS_ROOT, load_dataset

    print("Загружаю датасет...", flush=True)
    data = load_dataset()
    print(f"  {data.X.shape[0]} клипов, дней съёмки {len(set(data.groups))}", flush=True)

    print("Кэширую признаки кадров...", flush=True)
    rows = cache_rows(data.clip_ids.tolist(), DEFAULT_CLIPS_ROOT,
                      seq_len=int(data.X.shape[1]))

    curve = run_sweep(data, rows, scheme=args.scheme, n_folds=args.folds,
                      seed=args.seed, max_epochs=args.max_epochs)

    # Written before anything is printed. Five folds of training is minutes
    # of work, and a formatter that raises must not be able to take the
    # result with it.
    if args.json is not None:
        args.json.write_text(
            json.dumps({str(k): v for k, v in curve.items()},
                       indent=2, ensure_ascii=False, default=float),
            encoding="utf-8",
        )
        print(f"\nМетрики целиком: {args.json}")

    print("\n" + format_curve(curve, scheme=args.scheme))


if __name__ == "__main__":
    sys.exit(main())
