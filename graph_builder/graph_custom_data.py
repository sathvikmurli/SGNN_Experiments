"""PyTorch Geometric :class:`Data` wrapper for graph tensors ``x`` and ``edge_index``."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import from_networkx as pyg_from_networkx

from .graph_builder import SrcDstGraph, load_src_dst_graph_from_graph_set


def graphs_from_src_dst_list(
    graph_ls: Sequence[SrcDstGraph],
    *,
    time_chunked: bool = False,
) -> list[GraphCustomData]:
    """
    Convert each :class:`~sgnn.graph_builder.graph_builder.SrcDstGraph` in ``graph_ls``
    (e.g. from :attr:`~sgnn.graph_builder.graph_builder.SrcDstGraph.graph_ls` after
    :meth:`~sgnn.graph_builder.graph_builder.SrcDstGraph.full_graph_process`, cumulative
    per row) into
    :class:`GraphCustomData` for batched GNN/SGNN training.

    If ``time_chunked`` is True, applies :meth:`GraphCustomData.from_networkx_time_chunked`
    (matches SGNN ``time_chunk_x`` layout on **input** features); otherwise uses
    :meth:`GraphCustomData.from_networkx` on ``sg.graph``.
    """
    out: list[GraphCustomData] = []
    for sg in graph_ls:
        if time_chunked:
            out.append(GraphCustomData.from_networkx_time_chunked(sg))
        else:
            out.append(
                GraphCustomData.from_networkx(
                    sg.graph, group_node_attrs=list(SrcDstGraph.DEFAULT_FEATURES)
                )
            )
    return out


def _timestamp_sort_key(graph_build: SrcDstGraph, node_id: int) -> float:
    """
    Map ``graph_build._id_to_timestamp[node_id]`` to a float for ordering.

    CSV-backed flows often store ``TimeStamp`` as strings; missing destinations use
    no entry and sort last. All values are coerced so :func:`sorted` never compares
    incompatible types (e.g. ``str`` vs ``float``).
    """
    raw = graph_build._id_to_timestamp.get(node_id)
    if raw is None:
        return float("inf")
    if isinstance(raw, (int, float)):
        v = float(raw)
        if math.isnan(v):
            return float("inf")
        return v
    if isinstance(raw, str):
        s = raw.strip()
        try:
            v = float(s)
            if math.isnan(v):
                return float("inf")
            return v
        except ValueError:
            pass
    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(ts):
        return float("inf")
    return float(ts.timestamp())


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
        (oldest node first; nodes without a timestamp sort last). Values may be
        numeric or string (e.g. from CSV); they are coerced to epoch seconds for ordering.

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

        order = sorted(range(n), key=lambda i: _timestamp_sort_key(graph_build, i))
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


_GRAPH_SET_TXT = re.compile(r"^graph_\d{6}\.txt$")


def count_graph_set_snapshots(graph_set_dir: str | Path) -> int:
    """Number of ``graph_NNNNNN.txt`` files in ``graph_set_dir``."""

    d = Path(graph_set_dir)
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and _GRAPH_SET_TXT.match(p.name))


class LazySrcDstGraphSequence(Sequence[SrcDstGraph]):
    """
    Random-access sequence that loads snapshot ``i`` from disk via
    :func:`~sgnn.graph_builder.graph_builder.load_src_dst_graph_from_graph_set`
    (no in-memory graph list).
    """

    def __init__(
        self,
        graph_set_dir: str | Path,
        num_graphs: int,
        *,
        directed: bool = True,
        multi_edges: bool = True,
        feature_ls: list[str] | None = None,
    ) -> None:
        self._dir = Path(graph_set_dir)
        self._n = num_graphs
        self._directed = directed
        self._multi_edges = multi_edges
        self._feature_ls = feature_ls

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> SrcDstGraph:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        return load_src_dst_graph_from_graph_set(
            idx,
            self._dir,
            directed=self._directed,
            multi_edges=self._multi_edges,
            feature_ls=self._feature_ls,
        )


class GraphSetDirectoryDataset(Dataset[GraphCustomData]):
    """
    PyTorch dataset that loads each cumulative snapshot from ``graph_set_dir`` written by
    :meth:`SrcDstGraph.full_graph_process` (``graph_{i:06d}.txt`` adjacency string plus
    ``graph_{i:06d}_data.npz``). Each :meth:`__getitem__` rebuilds NetworkX →
    :class:`GraphCustomData` for :class:`GraphCustomDataLoader` batching.

    Use :attr:`src_dst_graphs` (a :class:`LazySrcDstGraphSequence`) when training
    :class:`~sgnn.models.sgnn.BasicSGNNClassifier` with ``graph_build=``.
    """

    def __init__(
        self,
        graph_set_dir: str | Path,
        *,
        num_graphs: int | None = None,
        labels: torch.Tensor | Sequence[int] | None = None,
        feature_ls: list[str] | None = None,
        directed: bool = True,
        multi_edges: bool = True,
        time_chunked: bool = False,
    ) -> None:
        self._dir = Path(graph_set_dir)
        self._n = num_graphs if num_graphs is not None else count_graph_set_snapshots(self._dir)
        if self._n < 1:
            raise ValueError("graph_set_dir must contain at least one graph_######.txt snapshot")
        self._feature_ls = feature_ls
        self._directed = directed
        self._multi_edges = multi_edges
        self._time_chunked = time_chunked

        if labels is not None:
            self._labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
            if self._labels.numel() != self._n:
                raise ValueError(
                    f"labels length {self._labels.numel()} does not match num_graphs={self._n}"
                )
            for v in self._labels.tolist():
                if v not in (0, 1):
                    raise ValueError("Binary classification expects labels in {0, 1}")
        else:
            self._labels = None

        self._src_dst_lazy = LazySrcDstGraphSequence(
            self._dir,
            self._n,
            directed=directed,
            multi_edges=multi_edges,
            feature_ls=feature_ls,
        )

    @property
    def src_dst_graphs(self) -> LazySrcDstGraphSequence:
        return self._src_dst_lazy

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> GraphCustomData:
        sg = load_src_dst_graph_from_graph_set(
            idx,
            self._dir,
            directed=self._directed,
            multi_edges=self._multi_edges,
            feature_ls=self._feature_ls,
        )
        names = list(self._feature_ls or SrcDstGraph.DEFAULT_FEATURES)
        if self._time_chunked:
            data = GraphCustomData.from_networkx_time_chunked(sg)
        else:
            data = GraphCustomData.from_networkx(sg.graph, group_node_attrs=names)
        if self._labels is None:
            return data
        out = data.clone()
        out.y = torch.tensor([int(self._labels[idx].item())], dtype=torch.long)
        return out


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
