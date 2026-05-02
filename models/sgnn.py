"""Graph-level SGNN: GCN layers with a spiking MLP classifier (LIF between linears)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

import snntorch as snn
from snntorch import surrogate

from ..graph_builder.graph_builder import SrcDstGraph
from ..graph_builder.graph_custom_data import GraphCustomData

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


class BasicSGNNClassifier(nn.Module):
    """
    Stacked ``GCNConv`` layers (same pattern as :class:`BasicGNNClassifier`: ReLU and
    dropout between convs), then **time-chunk reordering** via
    :meth:`GraphCustomData.time_chunk_x` with the
    :class:`~sgnn.graph_builder.graph_builder.SrcDstGraph` ``G`` passed at init (same
    constraints as that method: feature dim must divide ``len(G.DEFAULT_FEATURES)``,
    node count must match ``G.graph``). Then :meth:`_node_blur` yields
    ``[num_graphs, timesteps, hidden]``, flattened to ``[num_graphs, timesteps * hidden]``
    for the classifier head.
    The **classifier** is three
    linear layers with **Leaky integrate-and-fire** (LIF) neurons from snnTorch
    between them: ``Linear → LIF → Linear → LIF → Linear``.

    Each LIF block runs ``num_steps`` micro-steps; the **mean spike rate** is passed
    to the next linear layer.

    Forward: ``(x, edge_index, batch, graph_build=...)`` → logits ``[num_graphs, num_classes]``.
    Pass ``graph_build`` for per-sample :class:`~sgnn.graph_builder.graph_builder.SrcDstGraph`
    when training on ``graph_ls``; otherwise ``self.G`` is used.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        G: SrcDstGraph,
        num_conv_layers: int = 2,
        dropout: float = 0.5,
        *,
        classifier_hidden: int | None = None,
        num_steps: int = 4,
        beta: float = 0.9,
        timesteps: int = 8,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if num_conv_layers < 1 or num_conv_layers > 4:
            raise ValueError("num_conv_layers must be between 1 and 4")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if timesteps < 1:
            raise ValueError("timesteps must be at least 1")

        self.G = G
        self.timesteps = timesteps
        self.threshold = float(threshold)
        self.dropout = dropout
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()

        h_clf = classifier_hidden if classifier_hidden is not None else hidden_channels

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = (
            GCNConv(hidden_channels, hidden_channels)
            if num_conv_layers >= 2
            else None
        )
        self.conv3 = (
            GCNConv(hidden_channels, hidden_channels)
            if num_conv_layers >= 3
            else None
        )
        self.conv4 = (
            GCNConv(hidden_channels, hidden_channels)
            if num_conv_layers >= 4
            else None
        )

        self.classifier_lin1 = nn.Linear(hidden_channels * timesteps, h_clf)
        self.classifier_lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.classifier_lin2 = nn.Linear(h_clf, h_clf)
        self.classifier_lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.classifier_lin3 = nn.Linear(h_clf, num_classes)

    def _lif_rate(self, lif: snn.Leaky, x: torch.Tensor) -> torch.Tensor:
        mem = lif.init_leaky()
        acc = torch.zeros_like(x)
        for _ in range(self.num_steps):
            spk, mem = lif(x, mem)
            acc = acc + spk
        return acc / float(self.num_steps)

    def _classifier_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier_lin1(x)
        x = self._lif_rate(self.classifier_lif1, x)
        x = self.classifier_lin2(x)
        x = self._lif_rate(self.classifier_lif2, x)
        return self.classifier_lin3(x)

    def _node_blur(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Per PyG graph in ``batch``, map node features to a fixed ``timesteps`` axis.

        Input: ``x`` of shape ``[total_nodes, node_features]``. Output:
        ``[num_graphs, timesteps, node_features]`` (graphs ordered by ascending graph id).

        * ``num_nodes < timesteps``: pad rows with zeros to ``(timesteps, F)``, then
          ``W @ graph_mat`` with ``W[i,k] = linspace(threshold, 0, T)[i]`` (constant
          along ``k``), shape ``(timesteps, timesteps)``.
        * ``num_nodes >= timesteps``: ``R = ceil(num_nodes / timesteps)``, pad to
          ``R * timesteps`` rows, reshape to ``(R, timesteps, F)``, combine with a
          ``(R, timesteps)`` weight matrix (rows linearly spaced from ``threshold`` to
          ``0``) via ``sum_r W[r,t] * G[r,t,f]`` → ``(timesteps, F)``.
        """
        if x.dim() != 2:
            raise ValueError("_node_blur expects x of shape [total_nodes, node_features]")
        if batch.shape[0] != x.shape[0]:
            raise ValueError("batch must have length total_nodes")

        T = self.timesteps
        thr = self.threshold
        device, dtype = x.device, x.dtype

        outs: list[torch.Tensor] = []
        for g in torch.unique(batch, sorted=True):
            mask = batch == g
            xg = x[mask]
            n, f = xg.shape

            if n < T:
                graph_mat = F.pad(xg, (0, 0, 0, T - n))
                row_w = torch.linspace(
                    thr, 0.0, steps=T, device=device, dtype=dtype
                )
                w = row_w[:, None].expand(T, T)
                out_g = w @ graph_mat
            else:
                r = (n + T - 1) // T
                pad_rows = r * T - n
                if pad_rows > 0:
                    xg = F.pad(xg, (0, 0, 0, pad_rows))
                graph_mat = xg.view(r, T, f)
                if r == 1:
                    row_w = torch.tensor([thr], device=device, dtype=dtype)
                else:
                    row_w = torch.linspace(thr, 0.0, steps=r, device=device, dtype=dtype)
                w_rt = row_w[:, None].expand(r, T)
                out_g = (w_rt.unsqueeze(-1) * graph_mat).sum(dim=0)

            outs.append(out_g)

        return torch.stack(outs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        *,
        graph_build: SrcDstGraph | None = None,
    ) -> torch.Tensor:
        G = graph_build if graph_build is not None else self.G
        x = self.conv1(x, edge_index)
        if self.conv2 is not None:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        if self.conv3 is not None:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
        if self.conv4 is not None:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv4(x, edge_index)

        data = GraphCustomData(x=x, edge_index=edge_index)
        x = data.time_chunk_x(G)

        x = self._node_blur(x, batch)
        x = torch.flatten(x, start_dim=1)
        return self._classifier_forward(x)

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
        ``sgnn_<optimizer>_weights.pt``.

        If ``weights_dir`` is not provided, defaults to ``sgnn/saved_weights``.
        """
        base_dir = (
            Path(weights_dir)
            if weights_dir is not None
            else Path(__file__).resolve().parents[1] / "saved_weights"
        )
        weight_path = base_dir / f"sgnn_{optimizer_name.lower()}_weights.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"SGNN weights not found: {weight_path}")

        state = torch.load(weight_path, map_location=map_location)
        self.load_state_dict(state, strict=strict)
        return weight_path

