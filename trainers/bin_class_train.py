"""Binary graph classification training for :mod:`sgnn.models` classifiers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from ..graph_builder.graph_custom_data import GraphCustomData, GraphCustomDataLoader
from ..models.gnn import BasicGNNClassifier
from ..models.sgnn import BasicSGNNClassifier

OptimizerName = Literal["nadam", "sgd"]


class GraphBinaryClassificationDataset(Dataset[GraphCustomData]):
    """
    :class:`torch.utils.data.Dataset` of :class:`~sgnn.graph_builder.graph_custom_data.GraphCustomData`
    graphs for binary labels in ``{0, 1}``.

    Pass ``labels`` aligned with ``graphs``, or ensure each graph has ``y`` set (shape ``[1]``,
    ``long``) before wrapping.
    """

    def __init__(
        self,
        graphs: Sequence[GraphCustomData],
        labels: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        self._graphs = list(graphs)
        if not self._graphs:
            raise ValueError("graphs must be non-empty")

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


def _train_loop(
    model: nn.Module,
    train_loader: GraphCustomDataLoader,
    val_loader: GraphCustomDataLoader | None,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    for _ in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, _binary_targets(batch))
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.num_graphs
            seen += batch.num_graphs
        history["train_loss"].append(running / max(seen, 1))

        if val_loader is not None:
            v_loss, v_acc = _evaluate(model, val_loader, device)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
        else:
            history["val_loss"].append(float("nan"))
            history["val_acc"].append(float("nan"))

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
) -> dict[str, list[float]]:
    """
    Train :class:`~sgnn.models.gnn.BasicGNNClassifier` for binary classification (``num_classes=2``).

    Uses :class:`GraphBinaryClassificationDataset` and
    :class:`~sgnn.graph_builder.graph_custom_data.GraphCustomDataLoader`. Optimizer is
    :class:`torch.optim.NAdam` or :class:`torch.optim.SGD` per ``optimizer_name``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
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
    return _train_loop(model, train_loader, val_loader, device, epochs, opt, scheduler=sched)


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
) -> dict[str, list[float]]:
    """
    Train :class:`~sgnn.models.sgnn.BasicSGNNClassifier` for binary classification.

    Same data path as :func:`train_basic_gnn_classifier_binary`. The model must already
    hold the :class:`~sgnn.graph_builder.graph_builder.SrcDstGraph` ``G`` required for
    time-chunking inside the forward pass.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
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
    return _train_loop(model, train_loader, val_loader, device, epochs, opt, scheduler=sched)


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
