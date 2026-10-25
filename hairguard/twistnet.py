"""hairguard.twistnet — the trained detector as the app sees it.

``hairlab`` trains the model and exports two artefacts; this module is the
only thing in the shipped app that knows they exist. ONNX is the runtime on
purpose: the graph carries its own input normalisation, so nothing has to be
shipped alongside it, and the app needs neither torch nor lightning to run
what they produced.

Three rules shape everything here.

*Failure is a fallback, never a crash.* Every way loading can go wrong — no
directory, no file, unreadable metadata, a graph that disagrees with its own
metadata, a feature width the extractor no longer produces — ends in
:class:`ModelUnavailable` carrying a sentence a person can act on. The app
catches it, starts on the existing geometric rule, and says which detector is
running. A model is an improvement to a working product, not a dependency of
it.

*A broken score must not read as "not twisting".* A NaN compares false
against every threshold, so it is indistinguishable from a confident
negative — the exact silent miss this project exists to fight. So non-finite
input is rejected by column, and non-finite output raises even when the input
was clean, because then the fault is in the weights.

*The feature width is checked, not assumed.* It is the one failure that
nothing else in the system would notice: change the extractor and the model
keeps accepting vectors whose columns have quietly come to mean something
else. ``feature_dim`` is therefore compared against the live
``hairlab.features`` at load time, and a mismatch refuses the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

# Where hairlab.train writes what it produced. Kept apart from
# ~/.hairguard/models, which holds the downloaded MediaPipe task files.
DEFAULT_DIR = Path.home() / ".hairguard" / "trained"

# A model ships with the package so the app detects on day one, before the
# user has collected anything. Their own model wins the moment they train
# one: hair-pulling differs enormously between people, and the bundled
# weights were fitted to someone else's hands.
BUNDLED_DIR = Path(__file__).parent / "models"


def _default_directory() -> Path:
    """Where to look for a model: the user's own first, then the bundled one."""
    if (DEFAULT_DIR / _META_NAME).is_file():
        return DEFAULT_DIR
    return BUNDLED_DIR

_META_NAME = "model.json"
_ONNX_NAME = "twistnet.onnx"
_SCORE_OUTPUT = "score"
_INPUT_NAME = "features"

# Built-in answers used when the configuration cannot be read. Losing the
# model because a config lookup raised would be a demotion for a reason that
# has nothing to do with the model.
_FALLBACK_SETTINGS = {"enabled": True, "threshold": "auto"}


class ModelUnavailable(RuntimeError):
    """The trained model cannot be used. The app continues on the rule."""


@dataclass(frozen=True)
class ModelInfo:
    """Everything needed to say out loud which detector is running."""

    directory: Path
    threshold: float
    feature_dim: int
    seq_len: int
    trained_at: str


def _read_meta(directory: Path) -> dict[str, Any]:
    path = directory / _META_NAME
    if not path.is_file():
        raise ModelUnavailable(f"{path} не найден")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ModelUnavailable(f"{_META_NAME} не читается: {exc}") from exc
    if not isinstance(meta, dict):
        raise ModelUnavailable(f"{_META_NAME} не похож на описание модели")
    return meta


def _positive_int(meta: dict[str, Any], key: str) -> int:
    value = meta.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ModelUnavailable(f"{_META_NAME}: поле {key!r} отсутствует или нечисловое")
    value = int(value)
    if value < 1:
        raise ModelUnavailable(f"{_META_NAME}: {key}={value} должно быть положительным")
    return value


def _threshold(meta: dict[str, Any], override: Optional[float]) -> float:
    raw = meta.get("threshold") if override is None else override
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ModelUnavailable(
            f"{_META_NAME}: порог отсутствует или нечисловой — без него неизвестно, "
            "когда модель считает, что это накручивание"
        )
    value = float(raw)
    if not (0.0 <= value <= 1.0) or value != value:
        raise ModelUnavailable(f"порог {value!r} вне диапазона вероятностей [0, 1]")
    return value


def _graph_window(session: Any) -> tuple[Optional[int], Optional[int]]:
    """``(seq_len, feature_dim)`` as the graph declares them, where fixed."""
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ModelUnavailable(
            f"ожидался один вход, в графе их {len(inputs)}"
        )
    shape = list(inputs[0].shape)
    if len(shape) != 3:
        raise ModelUnavailable(f"ожидался вход (batch, seq_len, features), получен {shape}")
    # The batch dimension is symbolic (a string); the other two are literals
    # in a correctly exported graph, but a None or a symbol there is not a
    # reason to refuse — the metadata is then the only word on the subject.
    return (
        shape[1] if isinstance(shape[1], int) else None,
        shape[2] if isinstance(shape[2], int) else None,
    )


def _finite_or_raise(array: np.ndarray, what: str) -> None:
    """Reject non-finite values by column, the way hairlab.model does.

    Re-implemented rather than imported: ``hairlab.model`` pulls in torch and
    pytorch-lightning at import time, and the shipped app must not need
    either to run what they trained.
    """
    bad = ~np.isfinite(array)
    if not bad.any():
        return
    columns = np.unique(np.nonzero(bad)[-1]).tolist()
    shown = columns[:10]
    more = "" if len(columns) <= 10 else f" (+{len(columns) - 10})"
    raise ValueError(
        f"{what}: {int(bad.sum())} нечисловых значений в признаках {shown}{more}. "
        "Оценка по таким данным читалась бы как уверенное «не накручивает»."
    )


class TwistModel:
    """The exported graph, loaded and validated, scoring one window at a time."""

    def __init__(self, session: Any, info: ModelInfo) -> None:
        self._session = session
        self._info = info

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        directory: Union[str, Path, None] = None,
        *,
        threshold: Optional[float] = None,
        expected_feature_dim: Optional[int] = None,
    ) -> "TwistModel":
        """Load and validate the artefacts in ``directory``.

        ``expected_feature_dim`` defaults to whatever ``hairlab.features``
        currently produces, which is the check that matters: the model and the
        extractor have to be the same vintage.
        """
        directory = Path(directory) if directory is not None else _default_directory()
        if not directory.is_dir():
            raise ModelUnavailable(f"каталог модели {directory} не найден")

        meta = _read_meta(directory)
        onnx_path = directory / _ONNX_NAME
        if not onnx_path.is_file():
            raise ModelUnavailable(f"{onnx_path} не найден")

        feature_dim = _positive_int(meta, "feature_dim")
        seq_len = _positive_int(meta, "seq_len")
        value = _threshold(meta, threshold)

        if expected_feature_dim is None:
            # The feature extractor lives in hairlab, which the install does
            # not always make importable. Repaired here rather than in the
            # caller, so every entry point is covered: which detector runs
            # must not depend on the directory the app was started from.
            # See hairguard.labpath.
            from hairguard.labpath import ensure_hairlab_importable

            repaired = ensure_hairlab_importable()
            try:
                from hairlab.features import FEATURE_DIM

                expected_feature_dim = FEATURE_DIM
            except Exception as exc:
                raise ModelUnavailable(f"hairlab.features недоступен: {exc}") from exc
            if repaired is not None:
                print(f"[twistnet] {repaired}")

        if feature_dim != expected_feature_dim:
            raise ModelUnavailable(
                f"модель обучена на {feature_dim} признаках, а извлекатель даёт "
                f"{expected_feature_dim} — переобучите модель, иначе её оценки "
                "будут относиться к другим данным"
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelUnavailable(f"onnxruntime не установлен: {exc}") from exc

        options = ort.SessionOptions()
        # One thread each. The work is 0.1 ms on a graph this size, and a
        # thread pool spun up inside the tkinter loop's own tick buys nothing
        # while adding somewhere for a stall to come from.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        try:
            session = ort.InferenceSession(
                str(onnx_path), options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ModelUnavailable(f"{onnx_path} не загружается: {exc}") from exc

        graph_seq, graph_dim = _graph_window(session)
        if graph_seq is not None and graph_seq != seq_len:
            raise ModelUnavailable(
                f"{_META_NAME} обещает окно {seq_len} кадров, а в графе их {graph_seq}"
            )
        if graph_dim is not None and graph_dim != feature_dim:
            raise ModelUnavailable(
                f"{_META_NAME} обещает {feature_dim} признаков, а в графе их {graph_dim}"
            )
        names = {output.name for output in session.get_outputs()}
        if _SCORE_OUTPUT not in names:
            raise ModelUnavailable(
                f"в графе нет выхода {_SCORE_OUTPUT!r} (есть {sorted(names)})"
            )

        model = cls(
            session,
            ModelInfo(
                directory=directory,
                threshold=value,
                feature_dim=feature_dim,
                seq_len=seq_len,
                trained_at=str(meta.get("trained_at") or "неизвестно"),
            ),
        )

        # One inference now, on a blank window, so the first real tick does
        # not pay for graph setup inside the detection loop.
        try:
            model.score(np.zeros((seq_len, feature_dim), dtype=np.float32))
        except Exception as exc:
            raise ModelUnavailable(f"пробный прогон не удался: {exc}") from exc
        return model

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def threshold(self) -> float:
        return self._info.threshold

    @property
    def seq_len(self) -> int:
        return self._info.seq_len

    @property
    def feature_dim(self) -> int:
        return self._info.feature_dim

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, window: np.ndarray) -> float:
        """Probability that this window is twisting. Raises rather than guessing."""
        array = np.asarray(window, dtype=np.float32)
        if array.shape == (1, self.seq_len, self.feature_dim):
            array = array[0]
        if array.shape != (self.seq_len, self.feature_dim):
            raise ValueError(
                f"окно неверной формы: ожидалось "
                f"({self.seq_len}, {self.feature_dim}), получено {array.shape}"
            )
        return float(self.score_batch(array[None])[0])

    def score_batch(self, batch: np.ndarray) -> np.ndarray:
        """Probabilities for ``(N, seq_len, feature_dim)`` windows, shape ``(N,)``."""
        session = self._session
        if session is None:
            raise ValueError("сессия модели уже закрыта")
        array = np.ascontiguousarray(np.asarray(batch, dtype=np.float32))
        if array.ndim != 3 or array.shape[1:] != (self.seq_len, self.feature_dim):
            raise ValueError(
                f"пачка неверной формы: ожидалось "
                f"(N, {self.seq_len}, {self.feature_dim}), получено {array.shape}"
            )
        _finite_or_raise(array, "признаки для модели")

        outputs = session.run([_SCORE_OUTPUT], {_INPUT_NAME: array})
        scores = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if not np.isfinite(scores).all():
            # The input was clean, so the graph's own weights or normalisation
            # statistics are not. Say so here rather than letting a nan travel
            # on as a confident negative.
            raise ValueError(
                f"{int((~np.isfinite(scores)).sum())} из {scores.size} оценок "
                "нечисловые при чистом входе — повреждены веса модели"
            )
        return scores

    def close(self) -> None:
        """Release the session. Safe to call twice."""
        self._session = None


# ---------------------------------------------------------------------------
# The configured decision, and the sentence that announces it
# ---------------------------------------------------------------------------

def _setting(config: Any, key: str) -> Any:
    try:
        return config.get("detection", "model", key)
    except Exception:
        return _FALLBACK_SETTINGS.get(key)


def load_configured(
    config: Any, directory: Union[str, Path, None] = None
) -> tuple[Optional[TwistModel], str]:
    """``(model, status)`` — the model if it can be used, and why either way.

    The status line is the product requirement, not decoration: which
    detector is active has to be visible rather than inferred, because the
    two differ by thirty points of recall and nothing on screen would
    otherwise distinguish them.
    """
    if not _setting(config, "enabled"):
        return None, "правило (модель отключена в настройках)"

    configured = _setting(config, "threshold")
    override: Optional[float] = None
    if isinstance(configured, (int, float)) and not isinstance(configured, bool):
        override = float(configured)

    try:
        model = TwistModel.load(directory, threshold=override)
    except ModelUnavailable as exc:
        return None, f"правило — модель недоступна: {exc}"
    except Exception as exc:  # pragma: no cover - defence in depth
        return None, f"правило — модель не загрузилась: {exc!r}"

    source = "из настроек" if override is not None else "из model.json"
    return model, (
        f"модель twistnet (порог {model.threshold:.3f} {source}, "
        f"окно {model.seq_len} кадров, обучена {model.info.trained_at})"
    )
