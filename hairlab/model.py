"""hairlab.model — the twisting classifier: architecture, training, export.

The model answers one question about a 2-second window: is the hand at the
head *twisting a strand*, or is it there for some other reason (resting,
scratching, holding a comb). Input is a landmark-derived feature sequence,
``(seq_len, feature_dim)``; output is a single logit.

Three decisions here follow from the dataset being small — 861 clips — rather
than from anything about the signal.

*A temporal CNN, not a transformer.* Twisting is a short periodic motion, so
the discriminative evidence is local in time and the same at every offset.
Convolutions get that invariance for free from weight sharing; attention has
to learn it from data we do not have. The stack is sized so its receptive
field covers the whole window (see ``_KERNELS``), because a model that can
only see a third of the clip cannot tell a rotation from a single twitch.

*Regularisation over capacity.* BatchNorm, channel dropout and weight decay
throughout, global average pooling instead of a flatten-and-dense head (which
would be most of the parameters and all of the overfitting), and early
stopping that hands back the best epoch rather than the last. At this sample
count the network starts memorising within a few dozen epochs.

*Normalisation lives inside the model.* The per-feature mean and standard
deviation are persistent buffers fitted on the training split, so the first
thing ``forward`` does is standardise its input. A separate scaler file would
be one more thing to ship, to version, and to get out of sync with the
weights; the exported ONNX graph carries the statistics in it and needs
nothing alongside.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 2 seconds at the app's 10 fps — the window hairlab.capture records.
DEFAULT_SEQ_LEN = 20

# Kernel widths per block. With a /2 pool after the first two blocks the
# receptive field reaches 21 frames, so the last block sees the entire clip.
_KERNELS = (5, 5, 3)

# Only guards features that are exactly constant (a landmark that was never
# present). Anything with real variance is left alone.
#
# A blanket floor rather than "leave a tiny std alone": a feature whose real
# std is 1e-9 divided by 1e-6 is attenuated, which is survivable, whereas
# dividing by its own 1e-9 would amplify quantisation noise into the signal.
_STD_FLOOR = 1e-6


def _require_finite(X: Any, what: str) -> np.ndarray:
    """Reject non-finite features at the boundary, naming the columns.

    One nan cell is enough to destroy everything downstream: it makes the
    fitted mean and std of its column nan, which makes *every* score nan, and
    nothing raises until the numbers reach something that checks — by which
    point the error surfaces several layers from the clip that caused it. The
    geometry features invite exactly this (an angle from ``acos`` of a cosine
    a hair outside [-1, 1], a ratio with a zero denominator, a normalised
    zero-length vector), so the cheap scan is worth running every time.
    """
    array = np.asarray(X, dtype=np.float32)
    bad = ~np.isfinite(array)
    if bad.any():
        columns = np.unique(np.nonzero(bad)[-1]).tolist()
        shown = columns[:10]
        more = "" if len(columns) <= 10 else f" (+{len(columns) - 10} more)"
        raise ValueError(
            f"{what} contains {int(bad.sum())} non-finite value(s) in feature "
            f"column(s) {shown}{more}. Fix the extractor: a nan here silently "
            f"turns every score into nan rather than failing anywhere useful."
        )
    return array


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class TwistNet(nn.Module):
    """1D temporal CNN over ``(batch, seq_len, feature_dim)`` -> logit ``(batch,)``."""

    def __init__(
        self,
        feature_dim: int,
        channels: Sequence[int] = (64, 96, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.channels = tuple(int(c) for c in channels)

        # Buffers, not attributes: they must travel in the state dict and in
        # the ONNX graph. Identity until fit_normalisation() is called, so an
        # untrained model is still usable in tests and sanity checks.
        self.register_buffer("input_mean", torch.zeros(self.feature_dim))
        self.register_buffer("input_std", torch.ones(self.feature_dim))

        blocks: list[nn.Module] = []
        in_channels = self.feature_dim
        for index, (out_channels, kernel) in enumerate(zip(self.channels, _KERNELS)):
            blocks.append(
                nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False)
            )
            blocks.append(nn.BatchNorm1d(out_channels))
            blocks.append(nn.GELU())
            # Channel dropout rather than element-wise: neighbouring frames of
            # a feature map are strongly correlated, so dropping single
            # elements leaves the signal recoverable and regularises little.
            blocks.append(nn.Dropout1d(dropout))
            if index < len(self.channels) - 1:
                blocks.append(nn.MaxPool1d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_channels, 1)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_normalisation(self, X: Union[np.ndarray, torch.Tensor]) -> None:
        """Set the input statistics from ``(N, seq_len, feature_dim)`` samples.

        Call with the training split only — fitting on everything leaks the
        validation set into the model through the scale of its own inputs.
        Statistics are accumulated in float64 so that a feature with a large
        offset and small variance survives the subtraction.
        """
        array = np.asarray(X)
        if array.ndim != 3 or array.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected (N, seq_len, {self.feature_dim}), got {tuple(array.shape)}"
            )
        tensor = torch.as_tensor(_require_finite(array, "training features")).to(
            dtype=torch.float64
        )

        mean = tensor.mean(dim=(0, 1))
        # correction=0: the population std of the sample we were handed, which
        # is what the normalisation is for. The unbiased estimate would make
        # the statistics depend on how many clips happened to be in the split.
        std = tensor.std(dim=(0, 1), correction=0).clamp_min(_STD_FLOOR)

        self.input_mean.copy_(mean.to(self.input_mean.dtype))
        self.input_std.copy_(std.to(self.input_std.dtype))

    @torch.no_grad()
    def reset_normalisation(self) -> None:
        """Back to mean 0 / std 1 — the identity transform."""
        self.input_mean.zero_()
        self.input_std.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standardise first, in the graph, so the exported model is complete.
        x = (x - self.input_mean) / self.input_std
        # (B, T, F) -> (B, F, T): time is the convolved axis, features are channels.
        x = self.features(x.transpose(1, 2))
        # Average over time rather than flattening: keeps the head tiny and
        # makes the score independent of where in the window the motion sits.
        x = x.mean(dim=-1)
        return self.head(self.head_dropout(x)).squeeze(-1)


class _ExportWrapper(nn.Module):
    """Adds the sigmoid so the exported graph emits a usable score.

    The app wants a probability it can threshold, not a logit it has to know
    the meaning of. Both come out, logit first, for callers that want either.
    """

    def __init__(self, net: TwistNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.net(x)
        return logit, torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------------

class TwistModule(pl.LightningModule):
    """Lightning wrapper: loss, optimiser, metrics."""

    def __init__(
        self,
        feature_dim: int,
        lr: float = 1e-3,
        pos_weight: Optional[float] = None,
        channels: Sequence[int] = (64, 96, 128),
        dropout: float = 0.3,
        weight_decay: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Ignore nothing: every hyperparameter has to be in the checkpoint for
        # load_checkpoint() to rebuild the same architecture.
        self.save_hyperparameters()
        self.net = TwistNet(feature_dim, channels=channels, dropout=dropout)

        # A buffer so Lightning moves it with the model; 1.0 means "no
        # reweighting", which keeps the loss call free of branching.
        self.register_buffer(
            "_pos_weight",
            torch.tensor([1.0 if pos_weight is None else float(pos_weight)]),
        )
        # Filled in by _RestoreBestWeights during and after fit(). The curve is
        # kept because the interesting question for a dataset this size is
        # *when* validation loss turned around, which a single number hides.
        self.best_val_loss: Optional[float] = None
        self.val_loss_history: list[float] = []

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """BCE on the logits, positives weighted for class imbalance."""
        logits = self(x)
        return nn.functional.binary_cross_entropy_with_logits(
            logits, y.to(logits.dtype), pos_weight=self._pos_weight
        )

    def training_step(self, batch, batch_idx):  # noqa: D102 - Lightning hook
        x, y = batch
        loss = self.loss(x, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):  # noqa: D102 - Lightning hook
        x, y = batch
        logits = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y.to(logits.dtype), pos_weight=self._pos_weight
        )
        # Threshold on the logit, not the probability: logit > 0 is exactly
        # probability > 0.5 and costs no sigmoid.
        accuracy = ((logits > 0).to(logits.dtype) == y.to(logits.dtype)).to(logits.dtype).mean()
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", accuracy, prog_bar=False)
        return loss

    def configure_optimizers(self):  # noqa: D102 - Lightning hook
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def validation_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """The same loss the training loop monitors, over a whole split.

        Lets a caller (or a test) check what it got back without standing up a
        Trainer. Matches the logged value exactly because validation runs
        full-batch, so there is no per-batch averaging to reproduce.
        """
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
            target = torch.as_tensor(np.asarray(y), device=device)
            return float(self.loss(x, target))
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class _RestoreBestWeights(pl.Callback):
    """Keeps the best-scoring weights in memory and puts them back at the end.

    Lightning's ModelCheckpoint would do this via disk; at this model size a
    deep copy is cheaper than a temp directory and leaves no files to clean up
    after a failed run.
    """

    def __init__(self, monitor: str = "val_loss") -> None:
        self.monitor = monitor
        self.best_score = math.inf
        self.best_state: Optional[dict[str, torch.Tensor]] = None

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return
        score = float(score)
        pl_module.val_loss_history.append(score)
        if score < self.best_score:
            self.best_score = score
            self.best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in pl_module.state_dict().items()}
            )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.best_state is None:
            return
        pl_module.load_state_dict(self.best_state)
        pl_module.best_val_loss = self.best_score


def _as_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    feature_dim: Optional[int] = None,
    max_epochs: int = 100,
    seed: int = 0,
    batch_size: int = 32,
    patience: int = 15,
    accelerator: str = "auto",
    devices: Any = "auto",
    **kwargs: Any,
) -> TwistModule:
    """Fit a ``TwistModule`` and return it with the best-epoch weights loaded.

    ``pos_weight`` defaults to ``n_negative / n_positive`` on the training
    split; pass it explicitly (``1.0`` to disable) to override. ``feature_dim``
    is read off the data when not given, so this module never has to know how
    many features the extractor produces.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train).astype(np.int64)
    y_val = np.asarray(y_val).astype(np.int64)

    if feature_dim is None:
        feature_dim = int(X_train.shape[-1])

    # Seed before the model exists: weight init has to be part of what the
    # seed pins down, or "same seed, same result" is not true.
    pl.seed_everything(seed, workers=True, verbose=False)

    if "pos_weight" not in kwargs:
        positives = int((y_train == 1).sum())
        negatives = int((y_train == 0).sum())
        kwargs["pos_weight"] = (negatives / positives) if positives else 1.0

    module = TwistModule(feature_dim=feature_dim, **kwargs)
    module.net.fit_normalisation(X_train)

    train_loader = DataLoader(
        _as_dataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        # Explicit generator: seed_everything alone does not pin the sampler
        # once anything else has drawn from the global RNG.
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    # Validation runs full-batch: the split is small, and an exact loss makes
    # early stopping compare like with like across epochs.
    val_loader = DataLoader(
        _as_dataset(X_val, y_val),
        batch_size=max(1, len(X_val)),
        shuffle=False,
        num_workers=0,
    )

    restore = _RestoreBestWeights(monitor="val_loss")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience),
            restore,
        ],
    )
    trainer.fit(module, train_loader, val_loader)
    return module


# ---------------------------------------------------------------------------
# Inference and artifacts
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_scores(module: TwistModule, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Probabilities in [0, 1], one per sample, shape ``(N,)``.

    Never returns a non-finite score: bad input is rejected by name, and a
    model whose own weights have gone nan raises rather than handing back
    numbers that compare false against every threshold and so read as a
    confident "not twisting".
    """
    module.eval()
    device = next(module.parameters()).device
    X = np.ascontiguousarray(_require_finite(X, "features passed to predict_scores"))

    chunks: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).to(device)
        chunks.append(torch.sigmoid(module(batch)).float().cpu().numpy())
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    scores = np.concatenate(chunks).reshape(-1)
    if not np.isfinite(scores).all():
        # Input was finite, so the weights or the fitted statistics are not.
        # Say so here instead of letting the caller discover it downstream.
        raise ValueError(
            f"{int((~np.isfinite(scores)).sum())} of {scores.size} scores are "
            "non-finite from finite input — the model's own weights or "
            "normalisation statistics are corrupt (diverged training, or a "
            "checkpoint saved from a diverged run)"
        )
    return scores


def export_onnx(
    module: TwistModule,
    path: Union[str, Path],
    feature_dim: int,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> None:
    """Write a self-contained ONNX graph: features in, (logit, score) out.

    ``feature_dim`` is checked against the model rather than trusted, so a
    stale call site fails here instead of producing an artifact that silently
    disagrees with the extractor.
    """
    if int(feature_dim) != module.net.feature_dim:
        raise ValueError(
            f"feature_dim {feature_dim} does not match the model's {module.net.feature_dim}"
        )

    # A CPU copy in eval mode: the graph must not carry a device, a trained
    # module normally lives on CUDA, and BatchNorm has to fold its running
    # statistics rather than normalise whatever batch it is handed. Copying
    # leaves the caller's module exactly as it was.
    wrapper = _ExportWrapper(copy.deepcopy(module.net)).cpu().eval()
    # Two samples, not one. torch.export specialises dimensions of size 0 or 1
    # to that literal value, and while marking the input dynamic below keeps
    # the *input* symbolic, the outputs are inferred from the trace and come
    # out declared as a fixed batch of 1. onnxruntime then warns on every
    # batch above one ("Expected shape from model of {1}...") and any consumer
    # reading the declared shape is simply misinformed. A batch of 2 has
    # nothing to specialise on, so the batch dimension stays symbolic
    # end to end. The values are identical either way — BatchNorm is folded.
    example = torch.zeros(2, int(seq_len), int(feature_dim))

    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        input_names=["features"],
        output_names=["logit", "score"],
        # Batch stays dynamic so the app can score one clip and an evaluation
        # run can score a whole split through the same file. Keyed by the
        # wrapper's forward parameter name, not by input_names.
        dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
        # The torch.export path, default from torch 2.9 on. The TorchScript
        # exporter still works but is deprecated and due for removal.
        dynamo=True,
        # 18 is the lowest opset this exporter implements directly; asking for
        # less only buys a lossy down-conversion pass.
        opset_version=18,
        # Silence the exporter's progress report. It contains emoji, and on a
        # non-UTF-8 console (cp1251 here) printing it raises
        # UnicodeEncodeError from inside the export call — turning a working
        # export into a crash for reasons that have nothing to do with us.
        verbose=False,
    )
    strip_source_metadata(path)


def strip_source_metadata(path: Union[str, Path]) -> int:
    """Remove source locations from an exported graph. Returns strings cleared.

    The dynamo exporter records, for every node, the Python stack trace that
    produced it — absolute file paths and all. On the machine that trained
    this model those paths read
    ``C:\\Users\\<name>\\...\\hairlab\\model.py``, so a model shipped to
    anyone else carries the trainer's username and directory layout with it.

    The model is meant to be distributed, so this is not a cosmetic concern:
    it is the trainer's personal data travelling inside an artifact whose
    whole purpose is to be handed to strangers. None of it is needed to run
    the graph.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    cleared = 0

    def wipe(obj) -> None:
        nonlocal cleared
        if getattr(obj, "doc_string", ""):
            obj.doc_string = ""
            cleared += 1
        props = getattr(obj, "metadata_props", None)
        if props:
            cleared += len(props)
            del props[:]

    wipe(model)
    wipe(model.graph)
    for node in model.graph.node:
        wipe(node)
    for container in (model.graph.input, model.graph.output, model.graph.value_info):
        for item in container:
            wipe(item)
    for initializer in model.graph.initializer:
        wipe(initializer)

    # The producer name is harmless; the version string is not identifying
    # either, but the training-time working directory can appear here.
    model.producer_name = "hairlab"
    model.producer_version = ""

    onnx.save(model, str(path))
    return cleared


def save_checkpoint(module: TwistModule, path: Union[str, Path]) -> None:
    """Write weights plus hyperparameters, with no Trainer involved.

    ``Trainer.save_checkpoint`` needs a live trainer, which a module that was
    just loaded does not have. The layout is Lightning's own, so
    ``load_from_checkpoint`` reads it back.
    """
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in module.state_dict().items()},
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        str(path),
    )


def load_checkpoint(path: Union[str, Path]) -> TwistModule:
    """Rebuild a module from a checkpoint written by :func:`save_checkpoint`."""
    module = TwistModule.load_from_checkpoint(str(path), map_location="cpu")
    module.eval()
    return module
