from __future__ import annotations

import copy
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


MAX_WORKERS_LIMIT = 8

_DEFAULT_GRAPH_SET_GLOB = ("graph_*.txt", "graph_*_data.npz")


def numpy_adjacency_to_string(adj: np.ndarray) -> str:
    """Serialize a dense adjacency matrix with :func:`numpy.array2string` (text-friendly)."""

    return np.array2string(np.asarray(adj))


def adjacency_string_to_numpy(adj_string: str) -> np.ndarray:
    """
    Parse a matrix string produced by :func:`numpy_adjacency_to_string` / ``array2string``
    back to a 2D :class:`numpy.ndarray`.
    """

    t = adj_string.strip().replace("[", " ").replace("]", " ")
    return np.loadtxt(io.StringIO(t))


def networkx_from_numpy_adjacency(
    adj: np.ndarray,
    *,
    directed: bool,
    multi_edges: bool,
) -> nx.Graph:
    """
    Build a NetworkX graph from a dense adjacency matrix (e.g. from
    :func:`networkx.to_numpy_array`, which sums parallel edges by default).
    """

    adj = np.asarray(adj)
    if directed:
        ctor: type[nx.Graph] = nx.MultiDiGraph if multi_edges else nx.DiGraph
    else:
        ctor = nx.MultiGraph if multi_edges else nx.Graph
    return nx.from_numpy_array(adj, create_using=ctor())


def clear_graph_set_dir(graph_set_dir: str | Path) -> None:
    """Remove ``graph_*.txt`` and ``graph_*_data.npz`` under ``graph_set_dir``."""

    d = Path(graph_set_dir)
    if not d.is_dir():
        return
    for pattern in _DEFAULT_GRAPH_SET_GLOB:
        for p in d.glob(pattern):
            p.unlink(missing_ok=True)


class SrcDstGraph:
    SOURCE_NAME = "SourceIP"
    DESTINATION_NAME = "DestinationIP"

    DEFAULT_FEATURES = [
        "Duration",
        "FlowBytesSent",
        "FlowSentRate",
        "FlowBytesReceived",
        "FlowReceivedRate",
        "PacketLengthVariance",
        "PacketLengthStandardDeviation",
        "PacketLengthMean",
        "PacketLengthMedian",
        "PacketLengthMode",
        "PacketLengthSkewFromMedian",
        "PacketLengthSkewFromMode",
        "PacketLengthCoefficientofVariation",
        "PacketTimeVariance",
        "PacketTimeStandardDeviation",
        "PacketTimeMean",
        "PacketTimeMedian",
        "PacketTimeMode",
        "PacketTimeSkewFromMedian",
        "PacketTimeSkewFromMode",
        "PacketTimeCoefficientofVariation",
        "ResponseTimeTimeVariance",
        "ResponseTimeTimeStandardDeviation",
        "ResponseTimeTimeMean",
        "ResponseTimeTimeMedian",
        "ResponseTimeTimeMode",
        "ResponseTimeTimeSkewFromMedian",
        "ResponseTimeTimeSkewFromMode",
        "ResponseTimeTimeCoefficientofVariation",
    ]

    TIMESTAMP_FEATURE = "TimeStamp"

    def __init__(self, directed: bool = True, multi_edges: bool = True):
        self._directed = directed
        self._multi_edges = multi_edges
        self._id_to_endpoint: dict[int, Any] = {}
        self._id_to_timestamp: dict[int, float] = {}
        self.graph_ls: list[SrcDstGraph] = []
        #: Directory where :meth:`full_graph_process` writes ``graph_*.txt`` / ``graph_*_data.npz`` when requested.
        self.graph_set_dir: Path | None = None
        #: Set in :meth:`full_graph_process` when ``max_workers > 1``: one deep copy of
        #: ``df.iloc[0].to_dict()`` per worker slot, used to merge consistent keys when
        #: materializing row dicts off the main thread.
        self._worker_first_row_copies: list[dict[str, Any]] | None = None

        if directed:
            if multi_edges:
                self.graph = nx.MultiDiGraph()
            else:
                self.graph = nx.DiGraph()
        else:
            if multi_edges:
                self.graph = nx.MultiGraph()
            else:
                self.graph = nx.Graph()

    def _id_for_endpoint(
        self,
        endpoint: Any,
        timestamp: float | None = None,
        *,
        id_to_endpoint: dict[int, Any] | None = None,
        id_to_timestamp: dict[int, float] | None = None,
    ) -> int:
        """
        Map endpoint identity to a dense int id. Pass ``id_to_endpoint`` / ``id_to_timestamp``
        to use caller-owned maps (e.g. thread-local); defaults use instance maps.
        """
        ep_map = self._id_to_endpoint if id_to_endpoint is None else id_to_endpoint
        ts_map = self._id_to_timestamp if id_to_timestamp is None else id_to_timestamp
        for node_id, ep in ep_map.items():
            if ep == endpoint:
                return node_id
        node_id = len(ep_map)
        ep_map[node_id] = endpoint
        if timestamp is not None:
            ts_map[node_id] = timestamp
        return node_id

    @staticmethod
    def _row_has_columns(row: pd.Series | dict, names: tuple[str, ...]) -> bool:
        if isinstance(row, pd.Series):
            return all(n in row.index for n in names)
        return all(n in row for n in names)

    @staticmethod
    def _apply_flow_to(sg: SrcDstGraph, row: pd.Series | dict, feature_ls: list[str]) -> None:
        """
        Add one flow's endpoints and edge to ``sg``. Safe to call concurrently on **distinct**
        ``SrcDstGraph`` instances (each has its own id maps and graph).
        """
        if not SrcDstGraph._row_has_columns(row, (SrcDstGraph.SOURCE_NAME, SrcDstGraph.DESTINATION_NAME)):
            cols = getattr(row, "columns", list(row.keys()) if isinstance(row, dict) else "?")
            raise ValueError(
                f"Source and destination columns must be present in the row: {cols}"
            )
        source = row[SrcDstGraph.SOURCE_NAME]
        destination = row[SrcDstGraph.DESTINATION_NAME]

        source_id = sg._id_for_endpoint(source, row[SrcDstGraph.TIMESTAMP_FEATURE])
        destination_id = sg._id_for_endpoint(destination)

        feature_kwargs = {feature: row[feature] for feature in feature_ls}
        sg.graph.add_node(source_id, **feature_kwargs)
        sg.graph.add_node(destination_id, **feature_kwargs)

        sg.graph.add_edge(source_id, destination_id)

    def _process_row(self, row: pd.Series | dict, feature_ls: list[str] = [], ret_graph: bool = False) -> None:
        self._apply_flow_to(self, row, feature_ls)
        if ret_graph:
            return self._copy_graph_state()

    def _copy_graph_state(self) -> SrcDstGraph:
        """Deep copy of graph topology, node attrs, and endpoint/id maps (for snapshots)."""
        snap = SrcDstGraph(directed=self._directed, multi_edges=self._multi_edges)
        snap.graph = copy.deepcopy(self.graph)
        snap._id_to_endpoint = copy.deepcopy(self._id_to_endpoint)
        snap._id_to_timestamp = copy.deepcopy(self._id_to_timestamp)
        return snap

    @staticmethod
    def _materialize_row_dicts_for_chunk(
        df: pd.DataFrame,
        indices: np.ndarray,
        seed_row_dict: dict[str, Any],
    ) -> list[tuple[int, dict[str, Any]]]:
        """Merge each row with *seed* so all columns align; safe for concurrent workers."""
        out: list[tuple[int, dict[str, Any]]] = []
        for i in indices:
            ii = int(i)
            merged = {**seed_row_dict, **df.iloc[ii].to_dict()}
            out.append((ii, merged))
        return out

    def full_graph_process(
        self,
        df: pd.DataFrame,
        feature_ls: list[str] = [],
        max_workers: int = 1,
        debug: bool = True,
        graph_set_dir: str | Path | None = None,
        clear_graph_set: bool = True,
    ) -> None:
        """
        For each dataframe row in order, append a **cumulative** :class:`SrcDstGraph` to
        :attr:`graph_ls`: row ``i`` includes all flows from rows ``0..i`` (shared endpoints
        map to the same node ids). After the call, :attr:`graph` on ``self`` matches the
        final cumulative graph (same as the last entry in :attr:`graph_ls`).

        When ``max_workers > 1``, row dicts are built in parallel using
        :attr:`_worker_first_row_copies` (one deep copy of the first row per worker slot);
        cumulative graph updates still run sequentially on the main thread.

        If ``graph_set_dir`` is set, each snapshot ``i`` writes a dense adjacency string
        (``nx.to_numpy_array`` + ``np.array2string``) to ``graph_{i:06d}.txt`` and node
        features / timestamps to ``graph_{i:06d}_data.npz`` so snapshots can be reloaded
        with :func:`load_src_dst_graph_from_graph_set` (see
        :class:`~sgnn.graph_builder.graph_custom_data.GraphSetDirectoryDataset`).
        """
        self.graph_ls = []
        self._worker_first_row_copies = None
        self.graph_set_dir = Path(graph_set_dir).resolve() if graph_set_dir is not None else None
        if self.graph_set_dir is not None:
            self.graph_set_dir.mkdir(parents=True, exist_ok=True)
            if clear_graph_set:
                clear_graph_set_dir(self.graph_set_dir)

        if max_workers > MAX_WORKERS_LIMIT:
            raise ValueError(
                f"Max workers must be less than 8: {max_workers}, hardset for caution, verify for your system"
            )

        if max_workers == 1:
            accum = SrcDstGraph(directed=self._directed, multi_edges=self._multi_edges)
            self.graph_ls = []
            for i, (_, row) in enumerate(df.iterrows()):
                accum._process_row(row, feature_ls)
                self.graph_ls.append(accum._copy_graph_state())
                if self.graph_set_dir is not None:
                    _write_graph_snapshot_to_graph_set(accum, i, self.graph_set_dir, feature_ls)
                if debug and (i + 1) % 10_000 == 0:
                    print(
                        f"full_graph_process (max_workers=1): {len(self.graph_ls)} cumulative graphs "
                        f"built after {i + 1} rows"
                    )
            self.graph = accum.graph
            self._id_to_endpoint = accum._id_to_endpoint
            self._id_to_timestamp = accum._id_to_timestamp
            return

        # max_workers > 1
        n = len(df)
        if n == 0:
            return

        workers = min(max_workers, n)
        first_row_dict = df.iloc[0].to_dict()
        self._worker_first_row_copies = [copy.deepcopy(first_row_dict) for _ in range(workers)]

        index_chunks = np.array_split(np.arange(n, dtype=int), workers)
        merged_pairs: list[tuple[int, dict[str, Any]]] = []

        if debug:
            print(f"Processing {n} rows with {workers} workers")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    SrcDstGraph._materialize_row_dicts_for_chunk,
                    df,
                    chunk,
                    self._worker_first_row_copies[wi],
                )
                for wi, chunk in enumerate(index_chunks)
                if len(chunk) > 0
            ]
            for fut in as_completed(futures):
                merged_pairs.extend(fut.result())

        merged_pairs.sort(key=lambda p: p[0])
        row_dicts_ordered = [rd for _, rd in merged_pairs]

        accum = SrcDstGraph(directed=self._directed, multi_edges=self._multi_edges)
        for i, row_dict in enumerate(row_dicts_ordered):
            SrcDstGraph._apply_flow_to(accum, row_dict, feature_ls)
            self.graph_ls.append(accum._copy_graph_state())
            if self.graph_set_dir is not None:
                _write_graph_snapshot_to_graph_set(accum, i, self.graph_set_dir, feature_ls)
        self.graph = accum.graph
        self._id_to_endpoint = accum._id_to_endpoint
        self._id_to_timestamp = accum._id_to_timestamp


def _feature_names_for_snapshot(feature_ls: list[str]) -> list[str]:
    return feature_ls if feature_ls else list(SrcDstGraph.DEFAULT_FEATURES)


def _write_graph_snapshot_to_graph_set(
    accum: SrcDstGraph,
    index: int,
    graph_set_dir: Path,
    feature_ls: list[str],
) -> None:
    """Write ``graph_{index:06d}.txt`` (adjacency string) and ``graph_{index:06d}_data.npz``."""

    G = accum.graph
    nodelist = sorted(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=nodelist)
    adj_string = numpy_adjacency_to_string(adj_matrix)
    base = graph_set_dir / f"graph_{index:06d}"
    base.with_suffix(".txt").write_text(adj_string, encoding="utf-8")

    names = _feature_names_for_snapshot(feature_ls)
    feat_rows: list[list[float]] = []
    ts_list: list[Any] = []
    for nid in nodelist:
        nd = G.nodes[nid]
        feat_rows.append([float(nd.get(name, float("nan"))) for name in names])
        ts_list.append(accum._id_to_timestamp.get(nid))

    np.savez(
        base.with_name(f"graph_{index:06d}_data.npz"),
        features=np.array(feat_rows, dtype=np.float64),
        timestamps=np.array(ts_list, dtype=object),
        nodelist=np.array(nodelist, dtype=np.int64),
        feature_names=np.array(names, dtype=object),
    )


def load_src_dst_graph_from_graph_set(
    index: int,
    graph_set_dir: str | Path,
    *,
    directed: bool = True,
    multi_edges: bool = True,
    feature_ls: list[str] | None = None,
) -> SrcDstGraph:
    """
    Load cumulative graph snapshot ``index`` from ``graph_set_dir``: adjacency text plus
    ``_data.npz`` (features and timestamps). Rebuilds a :class:`SrcDstGraph` whose
    integer node ids match rows/columns of the saved adjacency (0..n-1).
    """

    d = Path(graph_set_dir)
    base_name = f"graph_{index:06d}"
    adj_str = (d / f"{base_name}.txt").read_text(encoding="utf-8")
    adj = adjacency_string_to_numpy(adj_str)
    data = np.load(d / f"{base_name}_data.npz", allow_pickle=True)
    features = np.asarray(data["features"])
    timestamps = data["timestamps"]
    nodelist = np.asarray(data["nodelist"]).reshape(-1).astype(int, copy=False)
    saved_names = data["feature_names"]
    if saved_names is not None and len(saved_names):
        names = [str(x) for x in saved_names.tolist()]
    else:
        names = _feature_names_for_snapshot(feature_ls or [])

    sg = SrcDstGraph(directed=directed, multi_edges=multi_edges)
    G = networkx_from_numpy_adjacency(adj, directed=directed, multi_edges=multi_edges)

    for i, nid in enumerate(nodelist):
        for j, name in enumerate(names):
            if j < features.shape[1]:
                G.nodes[nid][name] = float(features[i, j])

    sg.graph = G
    for i, nid in enumerate(nodelist):
        ts = timestamps[i]
        if ts is not None:
            sg._id_to_timestamp[int(nid)] = ts
        sg._id_to_endpoint[int(nid)] = int(nid)

    return sg
