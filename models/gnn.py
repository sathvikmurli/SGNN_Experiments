"""Graph-level GNN classifier built with PyTorch Geometric."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


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

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
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

        x = global_mean_pool(x, batch)
        return self.classifier(x)
