"""PyTorch Geometric :class:`Data` wrapper for graph tensors ``x`` and ``edge_index``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import from_networkx as pyg_from_networkx

from .graph_builder import SrcDstGraph


class GraphCustomData(Data):
    """
    Subclass of :class:`torch_geometric.data.Data` wired for node features ``x``
    and COO edges ``edge_index`` (shape ``[2, num_edges]``, ``long``).

    Use :meth:`pair` to unpack ``(x, edge_index)`` for tensors separately.
    """

    def __init__(
        self,
        x: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(x=x, edge_index=edge_index, **kwargs)

    def pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, edge_index)``; raises if either is missing."""

        if self.x is None or self.edge_index is None:
            raise ValueError("GraphCustomData requires both x and edge_index to be set")
        return self.x, self.edge_index

    def chunk_x(self, num_chunks: int, dim: int = 1) -> tuple[torch.Tensor, ...]:
        """
        Split ``self.x`` into ``num_chunks`` equal-sized pieces along ``dim``.

        The size along ``dim`` must be divisible by ``num_chunks``.
        """
        if self.x is None:
            raise ValueError("x must be set before chunk_x")
        if num_chunks < 1:
            raise ValueError("num_chunks must be at least 1")

        size = self.x.size(dim)
        if size % num_chunks != 0:
            raise ValueError(
                f"Cannot split dim {dim} of size {size} into {num_chunks} equal chunks"
            )

        chunk_size = size // num_chunks
        return torch.split(self.x, chunk_size, dim=dim)

    def time_chunk_x(
        self,
        graph_build: SrcDstGraph,
        *,
        dim: int = 1,
    ) -> torch.Tensor:
        """
        Reorder **rows** (nodes) by ``graph_build._id_to_timestamp`` ascending
        (oldest node first; nodes without a timestamp sort last).

        Along ``dim`` (default ``1``), ``x`` is treated as ``K`` contiguous blocks
        of length ``features_per_step`` (one block per time step), so
        ``x.size(dim) == K * features_per_step``. After reordering, row ``0`` is the
        oldest node and the first ``features_per_step`` slice along ``dim`` is its
        first block, the next ``features_per_step`` its second, and so on.

        Stores the reordered tensor on ``self.x_time_chunked`` (same shape as ``self.x``).
        Does not modify ``self.x`` or ``edge_index``. For GNN layers on the time-ordered
        rows, build a matching ``edge_index`` by remapping indices with the same
        permutation (oldest node becomes row ``0``).
        """

        features_per_step = len(graph_build.DEFAULT_FEATURES)
        if self.x is None:
            raise ValueError("x must be set before time_chunk_x")
        if features_per_step < 1:
            raise ValueError("features_per_step must be at least 1")
        if self.x.dim() != 2:
            raise ValueError("time_chunk_x expects a 2D x (num_nodes x num_features)")
        if dim not in (0, 1):
            raise ValueError("dim must be 0 or 1")

        if dim == 1:
            n, d = self.x.shape[0], self.x.shape[1]
            x_work = self.x
        else:
            d, n = self.x.shape[0], self.x.shape[1]
            x_work = self.x.transpose(0, 1)

        if d % features_per_step != 0:
            raise ValueError(
                f"Size along dim {dim} is {d}, not divisible by features_per_step={features_per_step}"
            )
        k = d // features_per_step

        if n != graph_build.graph.number_of_nodes():
            raise ValueError(
                f"x has {n} nodes but graph has {graph_build.graph.number_of_nodes()} nodes"
            )

        order = sorted(
            range(n),
            key=lambda i: graph_build._id_to_timestamp.get(i, float("inf")),
        )
        perm = torch.tensor(order, dtype=torch.long, device=x_work.device)

        x_ordered = x_work.index_select(0, perm)
        out_3d = x_ordered.reshape(n, k, features_per_step)
        out = out_3d.reshape(n, d).contiguous()

        x_time_chunked = out if dim == 1 else out.transpose(0, 1)
        self.x_time_chunked = x_time_chunked
        return x_time_chunked

    @classmethod
    def from_networkx(
        cls,
        G: nx.Graph,
        group_node_attrs: list[str] | None = None,
        **kwargs: Any,
    ) -> GraphCustomData:
        """
        Build from a NetworkX graph (e.g. ``SrcDstGraph().graph``).

        Node attributes in ``group_node_attrs`` become columns of ``x`` in that order.
        """
        base = pyg_from_networkx(G, group_node_attrs=group_node_attrs, **kwargs)
        return cls(x=base.x, edge_index=base.edge_index)


    @classmethod
    def from_networkx_time_chunked(
        cls,
        G: SrcDstGraph,
        group_node_attrs: list[str] | None = None,
        **kwargs: Any,
    ) -> GraphCustomData:
        """
        Build from a NetworkX graph (e.g. ``SrcDstGraph().graph``).
        """
        base = pyg_from_networkx(G.graph, group_node_attrs=G.DEFAULT_FEATURES, **kwargs)
        data = cls(x=base.x, edge_index=base.edge_index)
        data.time_chunk_x(G)
        return data


class GraphCustomDataLoader(PyGDataLoader):
    """
    Batches :class:`GraphCustomData` (or any ``Data`` subclass) using PyG's collate.

    ``dataset`` may be a ``list`` / sequence of graphs or a :class:`torch.utils.data.Dataset`
    whose items are :class:`GraphCustomData`. Optional tensor fields such as
    ``x_time_chunked`` are concatenated when present on every sample.
    """

    def __init__(
        self,
        dataset: Sequence[GraphCustomData] | Dataset,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset, batch_size=batch_size, **kwargs)
