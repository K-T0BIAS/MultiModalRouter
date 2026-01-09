# dataclasses.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


from dataclasses import dataclass
from enum import Enum
from abc import abstractmethod, ABC
from typing import Iterable, Optional


class OptimizationMetric(Enum):
    DISTANCE = "distance"


class EdgeMetadata:
    __slots__ = ['transportMode', 'metrics']

    def __init__(self, transportMode: str = None, **metrics):
        self.transportMode = transportMode
        self.metrics = metrics  # e.g., {"distance": 12.3, "time": 15} NOTE distance is required by the graph

    def getMetric(self, metric: OptimizationMetric | str):
        if isinstance(metric, str):
            value = self.metrics.get(metric)
            if value is None:
                raise KeyError(f"Metric '{metric}' not found in EdgeMetadata")
            return value

        value = self.metrics.get(metric.value)
        if value is None:
            raise KeyError(f"Metric '{metric.value}' not found in EdgeMetadata")
        return value

    def copy(self):
        return EdgeMetadata(transportMode=self.transportMode, **self.metrics)

    @property
    def allMetrics(self):
        return self.metrics.copy()

    def __str__(self):
        return f"transportMode={self.transportMode}, metrics={self.metrics}"


class Hub:
    """Base hub class - using regular class instead of dataclass for __slots__ compatibility"""
    __slots__ = ['coords', 'id', 'outgoing', 'hubType']

    def __init__(self, coords: list[float], id: str, hubType: str):
        self.coords: list[float] = coords
        self.id = id
        self.hubType = hubType
        # dict like {mode -> {dest_id -> EdgeMetadata}}
        self.outgoing: dict[str, dict[str, EdgeMetadata]] = {}

    def addOutgoing(self, mode: str, dest_id: str, metrics: EdgeMetadata):
        if mode not in self.outgoing:
            self.outgoing[mode] = {}
        self.outgoing[mode][dest_id] = metrics

    def getMetrics(self, mode: str, dest_id: str) -> EdgeMetadata:
        return self.outgoing.get(mode, {}).get(dest_id, None)

    def getMetric(self, mode: str, dest_id: str, metric: str) -> float:
        connection = self.outgoing.get(mode, {}).get(dest_id)
        return getattr(connection, metric, None) if connection else None

    def clone(self) -> "Hub":
        new = Hub(self.coords[:], self.id, self.hubType)

        for mode, dests in self.outgoing.items():
            for dest_id, meta in dests.items():
                new.addOutgoing(mode, dest_id, meta.copy())

        return new

    def __hash__(self):
        return hash((self.hubType, self.id))


@dataclass
class Route:
    """Route class can use dataclass since it doesn't need __slots__"""
    path: list[tuple[str, str]]
    totalMetrics: EdgeMetadata
    optimizedMetric: OptimizationMetric

    @property
    def optimizedValue(self):
        return self.totalMetrics.getMetric(self.optimizedMetric)

    @property
    def flatPath(self, toStr=True):
        """Flatten the path into a list of hub IDs"""
        if not self.path:
            return []
        # get all source hubs plus the final destination
        path = [edge for edge in self.path]
        if not toStr:
            return path
        pathStr = ""
        for i, edge in enumerate(path):
            if i == 0:
                pathStr += f"Start: {edge[0]}"
                continue

            if len(edge) > 2 and isinstance(edge[2], EdgeMetadata):
                pathStr += f"\n\tEdge: ({str(edge[2])})\n-> {edge[0]}"
            else:
                pathStr += f"{edge[0]} -> {edge[1]}"
        return pathStr

    def asGraph(self, graph):
        """
        Creates a new RouteGraph with only the hubs in the route.
        It replicates the settings from the original graph.

        ### NOTE:
            * the graph in the argument should be the same as the graph from which the route was created.
            * hubs that are present in the route, but not found in the graph will be skipped

        :param:
            graph: the graph to replicate settings from

        :returns:
            a new RouteGraph with the specified settings and the added route
        """
        from . import RouteGraph
        subGraph = RouteGraph(
            maxDistance=graph.maxDrivingDistance,
            transportModes=graph.TransportModes,
            compressed=graph.compressed,
            extraMetricsKeys=graph.extraMetricsKeys,
            drivingEnabled=graph.drivingEnabled,
            sourceCoordKeys=graph.sourceCoordKeys,
            destCoordKeys=graph.destCoordKeys
        )
        # gets the hubs from the route (if not present in graph the hub will be dropped)
        hubs: list[Hub] = [graph.getHubById(edge[0]) for edge in self.path if graph.getHubById(edge[0])]

        copies = [hub.clone() for hub in hubs]

        # add all hubs to subGraph
        for hub in copies:
            subGraph.addHub(hub)

        # add links between consecutive hubs
        for prev, curr in zip(copies, copies[1:]):
            transpMode = graph.TransportModes[prev.hubType]

            meta = prev.getMetrics(transpMode, curr.id)
            if meta is None:
                # recompute distance
                distance = graph._hubToHubDistances([curr], [prev])[0][0].item()
                meta = EdgeMetadata(transportMode=transpMode, distance=distance)
            else:
                meta = meta.copy()

            subGraph._addLink(prev, curr, transpMode, **meta.metrics)

        return subGraph


@dataclass
class VerboseRoute(Route):
    """Uses base Route class but adds additional info to hold the edge metadata for every leg"""
    path: list[tuple[str, str, EdgeMetadata]]


@dataclass(frozen=True)
class PathNode:
    hub_id: str
    mode: str
    edge: EdgeMetadata
    prev: Optional["PathNode"]

    @property
    def length(self) -> int:
        return 0 if self.prev is None else self.prev.length + 1

    def __iter__(self) -> Iterable["PathNode"]:
        node = self
        stack = []
        while node:
            stack.append(node)
            node = node.prev
        yield from reversed(stack)


class Filter(ABC):

    @abstractmethod
    def filterEdge(self, edge: EdgeMetadata) -> bool:
        """
        Return True if you want to keep the edge else False

        Args:
            edge (EdgeMetadata): Edge to filter

        Returns:
            bool: True if you want to keep the edge
        """
        pass

    @abstractmethod
    def filterHub(self, hub: Hub) -> bool:
        """
        Return True if you want to keep the hub else False

        Args:
            hub (Hub): Hub to filter

        Returns:
            bool: True if you want to keep the hub
        """
        pass

    def filter(self, start: Hub, end: Hub, edge: EdgeMetadata, path: PathNode | None = None) -> bool:
        return self.filterHub(start) and self.filterHub(end) and self.filterEdge(edge)
