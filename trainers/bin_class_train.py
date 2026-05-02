"""Binary graph classification training for :mod:`sgnn.models` classifiers.

Training entry points accept ``weight_decay`` (default ``0.0``). It is passed to
:class:`torch.optim.NAdam` / :class:`torch.optim.SGD`, which apply **L2 regularization**
on the model parameters (weight decay, not added manually to the loss).

``mini_batch`` (default ``1.0``) caps each epoch to a fraction of the training set:
after ``ceil(n * mini_batch)`` graphs have been processed (at least one), the epoch
ends for that training step.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from ..graph_builder.graph_builder import SrcDstGraph
from ..graph_builder.graph_custom_data import GraphCustomData, GraphCustomDataLoader
from ..models.gnn import BasicGNNClassifier, log_parameter_grad_norm
from ..models.sgnn import BasicSGNNClassifier

OptimizerName = Literal["nadam", "sgd"]


def _graphs_per_epoch_cap(train_size: int, mini_batch: float) -> int:
    """
    Maximum number of training graphs to process per epoch.

    ``mini_batch`` is a fraction in ``(0, 1]``: each epoch stops after that many
    graphs (rounded up), e.g. ``0.1`` with 100 graphs → 10 forwards per epoch.
    ``1.0`` uses the full training set each epoch.
    """
    if train_size < 1:
        raise ValueError("train_size must be >= 1")
    if mini_batch <= 0.0:
        raise ValueError("mini_batch must be positive")
    if mini_batch > 1.0:
        raise ValueError("mini_batch must be <= 1.0")
    if mini_batch >= 1.0:
        return train_size
    return max(1, min(train_size, math.ceil(train_size * mini_batch)))


class GraphBinaryClassificationDataset(Dataset[GraphCustomData]):
    """
    :class:`torch.utils.data.Dataset` of :class:`~sgnn.graph_builder.graph_custom_data.GraphCustomData`
    graphs for binary labels in ``{0, 1}``.

    Pass ``labels`` aligned with ``graphs``, or ensure each graph has ``y`` set (shape ``[1]``,
    ``long``) before wrapping.

    Optional ``src_dst_graphs`` (same length as ``graphs``) supplies the matching
    :class:`~sgnn.graph_builder.graph_builder.SrcDstGraph` per sample for
    :class:`~sgnn.models.sgnn.BasicSGNNClassifier` ``time_chunk_x`` when training on
    cumulative-per-row graphs from :attr:`~sgnn.graph_builder.graph_builder.SrcDstGraph.graph_ls`.
    """

    def __init__(
        self,
        graphs: Sequence[GraphCustomData],
        labels: torch.Tensor | Sequence[int] | None = None,
        *,
        src_dst_graphs: Sequence[SrcDstGraph] | None = None,
    ) -> None:
        self._graphs = list(graphs)
        if not self._graphs:
            raise ValueError("graphs must be non-empty")

        if src_dst_graphs is not None:
            self._src_dst_graphs: Sequence[SrcDstGraph] | None = src_dst_graphs
            if len(self._src_dst_graphs) != len(self._graphs):
                raise ValueError(
                    f"src_dst_graphs length {len(self._src_dst_graphs)} does not match "
                    f"len(graphs)={len(self._graphs)}"
                )
        else:
            self._src_dst_graphs = None

        if labels is not None:
            y = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
            if y.numel() != len(self._graphs):
                raise ValueError(
                    f"labels length {y.numel()} does not match len(graphs)={len(self._graphs)}"
                )
            for v in y.tolist():
                if v not in (0, 1):
                    raise ValueError("Binary classification expects labels in {0, 1}")
            self._labels = y
        else:
            for i, g in enumerate(self._graphs):
                if getattr(g, "y", None) is None:
                    raise ValueError(
                        f"Graph at index {i} has no y; pass labels=... or set y on each graph"
                    )
            self._labels = None

    @property
    def src_dst_graphs(self) -> Sequence[SrcDstGraph] | None:
        return self._src_dst_graphs

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> GraphCustomData:
        g = self._graphs[idx]
        if self._labels is None:
            return g
        out = g.clone()
        out.y = torch.tensor(
            [int(self._labels[idx].item())],
            dtype=torch.long,
        )
        return out


def _make_optimizer(
    params: Any,
    optimizer_name: OptimizerName,
    lr: float,
    *,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    nadam_betas: tuple[float, float] = (0.9, 0.999),
    nadam_eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """``weight_decay`` implements L2 regularization via the optimizer (see PyTorch docs)."""
    if optimizer_name == "nadam":
        return torch.optim.NAdam(
            params,
            lr=lr,
            betas=nadam_betas,
            eps=nadam_eps,
            weight_decay=weight_decay,
        )
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def _binary_targets(batch: GraphCustomData) -> torch.Tensor:
    if batch.y is None:
        raise ValueError("Batched graphs must carry graph-level y for binary training")
    return batch.y.view(-1).long()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: GraphCustomDataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, _binary_targets(batch))
        total_loss += loss.item() * batch.num_graphs
        pred = logits.argmax(dim=-1)
        correct += int((pred == _binary_targets(batch)).sum().item())
        n += batch.num_graphs
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def _evaluate_sgnn(
    model: BasicSGNNClassifier,
    dataset: GraphBinaryClassificationDataset,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate SGNN when each sample has its own :class:`SrcDstGraph` (cumulative snapshots)."""
    if dataset.src_dst_graphs is None:
        raise ValueError("_evaluate_sgnn requires dataset.src_dst_graphs")
    model.eval()
    total_loss = 0.0
    correct = 0
    n = len(dataset)
    for i in range(n):
        data = dataset[i].to(device)
        G = dataset.src_dst_graphs[i]
        batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        logits = model(data.x, data.edge_index, batch_vec, graph_build=G)
        y = _binary_targets(data)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
    return total_loss / max(n, 1), correct / max(n, 1)


def _train_loop_sgnn(
    model: BasicSGNNClassifier,
    train_dataset: GraphBinaryClassificationDataset,
    val_dataset: GraphBinaryClassificationDataset | None,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    *,
    max_graphs_per_epoch: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Train SGNN one graph at a time with per-sample ``graph_build`` (from ``graph_ls``).
    """
    if train_dataset.src_dst_graphs is None:
        raise ValueError("_train_loop_sgnn requires train_dataset.src_dst_graphs")
    if val_dataset is not None and val_dataset.src_dst_graphs is None:
        raise ValueError("val_dataset must include src_dst_graphs when train_dataset has them")
    n = len(train_dataset)
    cap = max_graphs_per_epoch
    if cap < 1 or cap > n:
        raise ValueError(f"max_graphs_per_epoch must be in [1, len(train_dataset)]; got {cap}")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for j in range(cap):
            i = int(perm[j].item())
            data = train_dataset[i].to(device)
            G = train_dataset.src_dst_graphs[i]
            optimizer.zero_grad(set_to_none=True)
            batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            logits = model(data.x, data.edge_index, batch_vec, graph_build=G)
            y = _binary_targets(data)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            if j == 0:
                pass
                #log_parameter_grad_norm(model, epoch, epochs)
            optimizer.step()
            running += loss.item()
        train_loss = running / max(cap, 1)
        history["train_loss"].append(train_loss)

        if val_dataset is not None:
            v_loss, v_acc = _evaluate_sgnn(model, val_dataset, device)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
        else:
            history["val_loss"].append(float("nan"))
            history["val_acc"].append(float("nan"))

        if verbose:
            msg = f"[epoch {epoch + 1}/{epochs}] train_loss={train_loss:.6f}"
            if val_dataset is not None:
                msg += (
                    f" val_loss={history['val_loss'][-1]:.6f}"
                    f" val_acc={history['val_acc'][-1]:.6f}"
                )
            print(msg)

        if scheduler is not None:
            scheduler.step()

    return history


def _train_loop(
    model: nn.Module,
    train_loader: GraphCustomDataLoader,
    val_loader: GraphCustomDataLoader | None,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    *,
    max_graphs_per_epoch: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    cap = max_graphs_per_epoch
    train_n = len(train_loader.dataset)
    if cap < 1 or cap > train_n:
        raise ValueError(
            f"max_graphs_per_epoch must be in [1, len(train_dataset)]; got {cap}"
        )

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        first_batch_in_epoch = True
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, _binary_targets(batch))
            loss.backward()
            if first_batch_in_epoch:
                #log_parameter_grad_norm(model, epoch, epochs)
                first_batch_in_epoch = False
            optimizer.step()
            running += loss.item() * batch.num_graphs
            seen += batch.num_graphs
            if seen >= cap:
                break
        train_loss = running / max(seen, 1)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            v_loss, v_acc = _evaluate(model, val_loader, device)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
        else:
            history["val_loss"].append(float("nan"))
            history["val_acc"].append(float("nan"))

        if verbose:
            msg = f"[epoch {epoch + 1}/{epochs}] train_loss={train_loss:.6f}"
            if val_loader is not None:
                msg += (
                    f" val_loss={history['val_loss'][-1]:.6f}"
                    f" val_acc={history['val_acc'][-1]:.6f}"
                )
            print(msg)

        if scheduler is not None:
            scheduler.step()

    return history


def train_basic_gnn_classifier_binary(
    model: BasicGNNClassifier,
    train_dataset: GraphBinaryClassificationDataset,
    *,
    optimizer_name: OptimizerName = "nadam",
    device: torch.device | str | None = None,
    val_dataset: GraphBinaryClassificationDataset | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    nadam_betas: tuple[float, float] = (0.9, 0.999),
    nadam_eps: float = 1e-8,
    scheduler_fn: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
    | None = None,
    num_workers: int = 0,
    verbose: bool = True,
    mini_batch: float = 1.0,
) -> dict[str, list[float]]:
    """
    Train :class:`~sgnn.models.gnn.BasicGNNClassifier` for binary classification (``num_classes=2``).

    Uses :class:`GraphBinaryClassificationDataset` and
    :class:`~sgnn.graph_builder.graph_custom_data.GraphCustomDataLoader`. Optimizer is
    :class:`torch.optim.NAdam` or :class:`torch.optim.SGD` per ``optimizer_name``.

    ``weight_decay`` is forwarded to the optimizer as L2 regularization on parameters
    (``0.0`` disables it).

    ``mini_batch`` in ``(0, 1]`` limits each epoch to ``ceil(n * mini_batch)`` training
    graphs (minimum 1). ``1.0`` runs a full pass over the training set each epoch.

    Set ``verbose=False`` to silence per-epoch loss logging.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    max_graphs = _graphs_per_epoch_cap(len(train_dataset), mini_batch)
    train_loader = GraphCustomDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader: GraphCustomDataLoader | None = None
    if val_dataset is not None:
        val_loader = GraphCustomDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    opt = _make_optimizer(
        model.parameters(),
        optimizer_name,
        lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nadam_betas=nadam_betas,
        nadam_eps=nadam_eps,
    )
    sched = scheduler_fn(opt) if scheduler_fn is not None else None
    return _train_loop(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        opt,
        max_graphs_per_epoch=max_graphs,
        scheduler=sched,
        verbose=verbose,
    )


def train_basic_sgnn_classifier_binary(
    model: BasicSGNNClassifier,
    train_dataset: GraphBinaryClassificationDataset,
    *,
    optimizer_name: OptimizerName = "nadam",
    device: torch.device | str | None = None,
    val_dataset: GraphBinaryClassificationDataset | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    nadam_betas: tuple[float, float] = (0.9, 0.999),
    nadam_eps: float = 1e-8,
    scheduler_fn: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
    | None = None,
    num_workers: int = 0,
    verbose: bool = True,
    mini_batch: float = 1.0,
) -> dict[str, list[float]]:
    """
    Train :class:`~sgnn.models.sgnn.BasicSGNNClassifier` for binary classification.

    Same data path as :func:`train_basic_gnn_classifier_binary`. The model must already
    hold a template :class:`~sgnn.graph_builder.graph_builder.SrcDstGraph` ``G`` (used when
    ``graph_build`` is not passed). If ``train_dataset`` was built with ``src_dst_graphs``
    aligned to :attr:`~sgnn.graph_builder.graph_builder.SrcDstGraph.graph_ls`, training
    runs one forward per graph with the matching ``graph_build`` (``batch_size`` ignored).

    ``weight_decay`` is forwarded to the optimizer as L2 regularization on parameters
    (``0.0`` disables it).

    ``mini_batch`` in ``(0, 1]`` limits each epoch to ``ceil(n * mini_batch)`` training
    graphs (minimum 1). ``1.0`` runs a full pass over the training set each epoch.

    Set ``verbose=False`` to silence per-epoch loss logging.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    max_graphs = _graphs_per_epoch_cap(len(train_dataset), mini_batch)
    opt = _make_optimizer(
        model.parameters(),
        optimizer_name,
        lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nadam_betas=nadam_betas,
        nadam_eps=nadam_eps,
    )
    sched = scheduler_fn(opt) if scheduler_fn is not None else None

    if train_dataset.src_dst_graphs is not None:
        return _train_loop_sgnn(
            model,
            train_dataset,
            val_dataset,
            device,
            epochs,
            opt,
            max_graphs_per_epoch=max_graphs,
            scheduler=sched,
            verbose=verbose,
        )

    train_loader = GraphCustomDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader: GraphCustomDataLoader | None = None
    if val_dataset is not None:
        val_loader = GraphCustomDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return _train_loop(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        opt,
        max_graphs_per_epoch=max_graphs,
        scheduler=sched,
        verbose=verbose,
    )


def train_basic_gnn_classifier_binary_nadam(
    model: BasicGNNClassifier,
    train_dataset: GraphBinaryClassificationDataset,
    **kwargs: Any,  # forwarded to :func:`train_basic_gnn_classifier_binary`
) -> dict[str, list[float]]:
    return train_basic_gnn_classifier_binary(
        model, train_dataset, optimizer_name="nadam", **kwargs
    )


def train_basic_gnn_classifier_binary_sgd(
    model: BasicGNNClassifier,
    train_dataset: GraphBinaryClassificationDataset,
    **kwargs: Any,
) -> dict[str, list[float]]:
    return train_basic_gnn_classifier_binary(
        model, train_dataset, optimizer_name="sgd", **kwargs
    )


def train_basic_sgnn_classifier_binary_nadam(
    model: BasicSGNNClassifier,
    train_dataset: GraphBinaryClassificationDataset,
    **kwargs: Any,
) -> dict[str, list[float]]:
    return train_basic_sgnn_classifier_binary(
        model, train_dataset, optimizer_name="nadam", **kwargs
    )


def train_basic_sgnn_classifier_binary_sgd(
    model: BasicSGNNClassifier,
    train_dataset: GraphBinaryClassificationDataset,
    **kwargs: Any,
) -> dict[str, list[float]]:
    return train_basic_sgnn_classifier_binary(
        model, train_dataset, optimizer_name="sgd", **kwargs
    )
