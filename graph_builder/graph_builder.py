from typing import Any, Optional

import networkx as nx
import pandas as pd
import numpy as np


MAX_WORKERS_LIMIT = 8


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
        self._id_to_endpoint: dict[int, Any] = {}
        self._id_to_timestamp: dict[int, float] = {}

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

    def _id_for_endpoint(self, endpoint: Any, timestamp: float | None = None) -> int:
        """Map source/destination identity to a dense int id matching PyG node order."""
        for node_id, ep in self._id_to_endpoint.items():
            if ep == endpoint:
                return node_id
        node_id = len(self._id_to_endpoint)
        self._id_to_endpoint[node_id] = endpoint
        if timestamp is not None:
            self._id_to_timestamp[node_id] = timestamp
        return node_id

    def _process_row(self, row: pd.Series | dict, feature_ls: list[str] = []) -> None:
        if self.SOURCE_NAME not in row or self.DESTINATION_NAME not in row:
            raise ValueError(f"Source and destination columns must be present in the row: {row.columns}")
        source = row[self.SOURCE_NAME]
        destination = row[self.DESTINATION_NAME]

        source_id = self._id_for_endpoint(source, row[self.TIMESTAMP_FEATURE])
        destination_id = self._id_for_endpoint(destination)

        feature_kwargs = {feature: row[feature] for feature in feature_ls}
        self.graph.add_node(source_id, **feature_kwargs)
        self.graph.add_node(destination_id, **feature_kwargs)

        self.graph.add_edge(source_id, destination_id)

    def full_graph_process(
        self, df: pd.DataFrame, feature_ls: list[str] = [], max_workers: int = 1
    ) -> None:

        if max_workers == 1:
            df.apply(lambda x: self._process_row(x, feature_ls), axis=1)
        elif max_workers > 1:
            # parallelize later
            pass

        elif max_workers > MAX_WORKERS_LIMIT:
            raise ValueError(
                f"Max workers must be less than 8: {max_workers}, hardset for caution, verify for your system"
            )