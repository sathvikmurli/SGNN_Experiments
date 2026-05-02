"""Graph-level GNN classifier built with PyTorch Geometric."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.propagate = False


def log_parameter_grad_norm(model: nn.Module, epoch_index: int, num_epochs: int) -> None:
    """
    Compute total L2 norm of all parameter gradients (after ``backward``, before ``step``)
    and log it (``StreamHandler`` on this logger so INFO shows in Jupyter). Uses:

    ``grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]``,
    ``total_norm = torch.cat(grads).norm()`` (when ``grads`` is non-empty).
    """
    grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    if not grads:
        dev = next(model.parameters()).device
        total_norm = torch.tensor(0.0, device=dev)
    else:
        total_norm = torch.cat(grads).norm()
    msg = f"[epoch {epoch_index + 1}/{num_epochs}] total_grad_norm={float(total_norm):.6f}"
    logger.info(msg)


def raise_if_forward_nan(t: torch.Tensor, *, where: str) -> None:
    """
    Raise ``RuntimeError`` if ``t`` contains any NaN.

    Intended for use right after each layer (or pooling) output in
    :meth:`BasicGNNClassifier.forward`.
    """
    if torch.isnan(t).any():
        raise RuntimeError(f"NaN in BasicGNNClassifier.forward after {where}")


class BasicGNNClassifier(nn.Module):
    """
    Stacked GCN layers with global mean pooling and a linear classification head.

    Expects batched graphs: ``forward(x, edge_index, batch)`` returns logits of
    shape ``[num_graphs, num_classes]``. For a single graph, pass
    ``batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device)``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_conv_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_conv_layers < 1 or num_conv_layers > 4:
            raise ValueError("num_conv_layers must be between 1 and 4")

        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = (
            GCNConv(int(hidden_channels), int(hidden_channels/2))
            if num_conv_layers >= 2
            else None
        )
        self.conv3 = (
            GCNConv(int(hidden_channels/2), int(hidden_channels/4))
            if num_conv_layers >= 3
            else None
        )
        self.conv4 = (
            GCNConv(int(hidden_channels/4), int(hidden_channels/8))
            if num_conv_layers >= 4
            else None
        )

        self.classifier_lin1= nn.Linear(int(hidden_channels/2), int(hidden_channels/4))
        self.classifier_lin2= nn.Linear(int(hidden_channels/4), num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        raise_if_forward_nan(x, where="conv1")

        if self.conv2 is not None:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            raise_if_forward_nan(x, where="conv2")
        if self.conv3 is not None:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
            raise_if_forward_nan(x, where="conv3")
        if self.conv4 is not None:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv4(x, edge_index)
            raise_if_forward_nan(x, where="conv4")

        x = global_mean_pool(x, batch)
        raise_if_forward_nan(x, where="global_mean_pool")

        x = self.classifier_lin1(x)
        raise_if_forward_nan(x, where="classifier_lin1")

        x = F.relu(x)
        raise_if_forward_nan(x, where="relu_after_classifier_lin1")

        x = self.classifier_lin2(x)
        raise_if_forward_nan(x, where="classifier_lin2")

        x = F.sigmoid(x)
        raise_if_forward_nan(x, where="sigmoid")
        return x

    def load_saved_weights(
        self,
        optimizer_name: str,
        *,
        weights_dir: str | Path | None = None,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> Path:
        """
        Load weights saved by the notebook naming convention:
        ``gnn_<optimizer>_weights.pt``.

        If ``weights_dir`` is not provided, defaults to ``sgnn/saved_weights``.
        """
        base_dir = (
            Path(weights_dir)
            if weights_dir is not None
            else Path(__file__).resolve().parents[1] / "saved_weights"
        )
        weight_path = base_dir / f"gnn_{optimizer_name.lower()}_weights.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"GNN weights not found: {weight_path}")

        state = torch.load(weight_path, map_location=map_location)
        self.load_state_dict(state, strict=strict)
        return weight_path
