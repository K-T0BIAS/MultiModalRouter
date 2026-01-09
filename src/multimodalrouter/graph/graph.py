# graph.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


from tqdm import tqdm
import zlib
import dill
import heapq
import os
import pandas as pd
from .dataclasses import Hub, EdgeMetadata, OptimizationMetric, Route, Filter, VerboseRoute, PathNode
from threading import Lock
from collections import deque
from itertools import count


class RouteGraph:

    def __init__(
        self,
        # sets the max distance to connect hubs with driving edges
        maxDistance: float,
        # dict like {hubtype -> "airport": "fly", "shippingport": "shipping"}
        transportModes: dict[str, str],
        # dict like {hubtype -> "airport": path -> "airports.csv", "shippingport": "shippingports.csv"}
        dataPaths: dict[str, str] = {},
        # if true model file will be compressed otherwise normal .dill file
        compressed: bool = False,
        # list of extra columns to add to the edge metadata (dynamically added to links when key is present in dataser)
        extraMetricsKeys: list[str] = [],
        # if true will connect hubs with driving edges
        drivingEnabled: bool = True,
        # a list of coordinate names for the source coords in the datasets (name to dataset matching is automatic)
        sourceCoordKeys: list[str] = ["source_lat", "source_lng"],
        # a list of coordinate names for the destination coords in the datasets (name to dataset matching is automatic)
        destCoordKeys: list[str] = ["destination_lat", "destination_lng"],
    ):
        self.sourceCoordKeys = set(sourceCoordKeys)
        self.destCoordKeys = set(destCoordKeys)

        self.compressed = compressed
        self.extraMetricsKeys = extraMetricsKeys
        self.drivingEnabled = drivingEnabled

        self.TransportModes = transportModes
        # hubtype -> {hubid -> Hub}
        self.Graph: dict[str, dict[str, Hub]] = {}

        # save the paths to the data in the state dict
        for key, value in dataPaths.items():
            setattr(self, key + "DataPath", value)
            self.Graph[key] = {}

        self.maxDrivingDistance = maxDistance

        self._lock = Lock()

    def __getstate__(self):
        state = self.__dict__.copy()

        # remove attributes that break pickle
        if "_lock" in state:
            del state["_lock"]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # set the lock for thread safety
        from threading import Lock
        self._lock = Lock()

    # =========== public helpers ==========

    def findClosestHub(self, allowedHubTypes: list[str], coords: list[float]) -> Hub | None:
        """
        Find the closest hub of a given type to a given location.

        Args:
            hubType: Type of hub to find
            coords: list[float] = the coordinates of the location

        Returns:
            Hub instance if found, None otherwise
        """
        potentialHubs = []
        if allowedHubTypes is not None:
            for hubType in allowedHubTypes:
                potentialHubs.extend(self.Graph.get(hubType, {}).values())

        if allowedHubTypes is None:
            potentialHubs = list(self._allHubs())

        if not potentialHubs:
            return None

        tempHub = Hub(coords=coords, hubType="temp", id="temp")
        distances = self._hubToHubDistances([tempHub], potentialHubs).flatten()  # shape (n,)
        closest_hub = potentialHubs[distances.argmin()]
        return closest_hub

    def addHub(self, hub: Hub):
        """
        Add a hub to the graph. If the hub already exists, it will not be added a second time.

        Args:
            hub: Hub instance to add to the graph

        Returns:
            None
        """
        with self._lock:
            hubType = hub.hubType
            # if the hub type doesnt exist in the graph, create it first
            if hubType not in self.Graph:
                self.Graph[hubType] = {}
            # exit if the hub already exists
            if hub.id in self.Graph[hubType]:
                return
            # add the hub
            self.Graph[hubType][hub.id] = hub

    def getHub(self, hubType: str, id: str) -> Hub | None:
        """
        Get a hub from the graph by hub type and hub id.

        Args:
            hubType: Type of hub to get
            id: ID of the hub to get

        Returns:
            Hub instance if found, None otherwise
        """
        return self.Graph.get(hubType, {}).get(id)

    def getHubById(self, id: str) -> Hub | None:
        """
        Get a hub from the graph by its ID.

        Args:
            id: ID of the hub to get

        Returns:
            Hub instance if found, None otherwise
        """
        for hubType in self.Graph:
            hub = self.Graph[hubType].get(id)
            if hub:
                return hub
        return None

    def save(
        self,
        filepath: str = os.path.join(os.getcwd(), "..", "..", "..", "data"),
        compressed: bool = False
    ):
        """
        Save the RouteGraph to a file.

        Args:
            filepath (str): Path to save the graph to.
            saveMode (str, optional): Unused. Defaults to None.
            compressed (bool, optional): Whether to compress the saved graph. Defaults to False.

        Note: The graph is saved in the following format:
            - Compressed: <filepath>.zlib
            - Uncompressed: <filepath>.dill
        """
        with self._lock:

            # ensure correct compression type is set for loading the graph
            self.compressed = compressed
            os.makedirs(filepath, exist_ok=True)
            # save the graph
            pickled = dill.dumps(self)
            if compressed:
                compressed = zlib.compress(pickled)
                with open(os.path.join(filepath, "graph.zlib"), "wb") as f:
                    f.write(compressed)
            else:
                with open(os.path.join(filepath, "graph.dill"), "wb") as f:
                    f.write(pickled)

    @staticmethod
    def load(filepath: str, compressed: bool = False) -> "RouteGraph":
        """
        Load a RouteGraph from a file.

        Args:
            filepath (str): Path to load the graph from.
            compressed (bool, optional): Whether the saved graph is compressed. Defaults to False.

        Returns:
            RouteGraph: The loaded graph.
        """
        with open(filepath, "rb") as f:
            file_data = f.read()
            if compressed:
                decompressed = zlib.decompress(file_data)
                graph = dill.loads(decompressed)
            else:
                graph = dill.loads(file_data)
        return graph

    # ============= private helpers =============

    def _allHubs(self):
        for hubType in self.Graph:
            yield from self.Graph[hubType].values()

    def _addLink(
        self,
        hub1: Hub,
        hub2: Hub,
        mode: str,
        distance: float,
        bidirectional: bool = False,
        extraData: dict | None = None
    ):
        """
        Add a connection between two hubs, dynamically storing extra metrics.
        """
        with self._lock:
            if extraData is None:
                extraData = {}
            # combine required metrics with extra
            metrics = {"distance": distance, **extraData}
            edge = EdgeMetadata(transportMode=mode, **metrics)
            hub1.addOutgoing(mode, hub2.id, edge)
            if bidirectional:
                hub2.addOutgoing(mode, hub1.id, edge.copy())

    def _loadData(self, targetHubType: str):
        dataPath = getattr(self, targetHubType + "DataPath")
        if dataPath is None:
            raise ValueError(f"Data path for {targetHubType} is not set")
        fType = os.path.splitext(dataPath)[1]
        if fType == ".csv":
            data = pd.read_csv(dataPath)
        elif fType == ".parquet":
            data = pd.read_parquet(dataPath)
        else:
            raise ValueError(f"Unsupported file type {fType}")

        if data.empty:
            raise ValueError(f"{targetHubType} data is empty load the data first")

        return data

    def _generateHubs(self):
        """
        Generate Hub instances and link them with EdgeMetadata.
        Extra columns in the data will be added to EdgeMetadata dynamically.
        """
        # no lock needed since underlying methods have locks
        for hubType in self.Graph.keys():
            data = self._loadData(hubType)
            added = set()

            thisSourceKeys = self.sourceCoordKeys & set(data.columns)
            thisDestinationKeys = self.destCoordKeys & set(data.columns)

            # get required and extra columns
            required_cols = {
                "source", "destination",
                *thisSourceKeys,
                *thisDestinationKeys,
                "distance"
            }

            # collect extra data from the dataset columns that are not required but marked as extra
            extra_metric_cols = []
            for m in self.extraMetricsKeys:
                if m not in required_cols:
                    try:
                        extra_metric_cols.append(m)
                    except KeyError:
                        continue

            for row in tqdm(data.itertuples(index=False), desc=f"Generating {hubType} Hubs", unit="hub"):
                # create hubs if they don't exist
                if row.source not in added:
                    hub = Hub(coords=[getattr(row, k) for k in thisSourceKeys], id=row.source, hubType=hubType)
                    self.addHub(hub)
                    added.add(row.source)

                if row.destination not in added:
                    hub = Hub(coords=[getattr(row, k) for k in thisDestinationKeys], id=row.destination, hubType=hubType)
                    self.addHub(hub)
                    added.add(row.destination)

                # get extra metrics
                extra_metrics = {
                    col: getattr(row, col)
                    for col in extra_metric_cols
                    if hasattr(row, col)
                }

                # link with the extra metrics
                self._addLink(
                    hub1=self.Graph[hubType][row.source],
                    hub2=self.Graph[hubType][row.destination],
                    mode=self.TransportModes[hubType],
                    distance=row.distance, # distance metric is absolutely required for all links
                    extraData=extra_metrics
                )

    def _hubToHubDistances(self, hub1: list[Hub], hub2: list[Hub]):
        """
        Compute full pairwise distance matrix between two lists of hubs using Haversine.

        Args:
            hub1: list of Hub
            hub2: list of Hub

        Returns:
            numpy.ndarray of shape (len(hub1), len(hub2))
        """
        import torch
        R = 6371.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            lat1 = torch.deg2rad(torch.tensor([h.coords[0] for h in hub1], device=device))
            lng1 = torch.deg2rad(torch.tensor([h.coords[1] for h in hub1], device=device))
            lat2 = torch.deg2rad(torch.tensor([h.coords[0] for h in hub2], device=device))
            lng2 = torch.deg2rad(torch.tensor([h.coords[1] for h in hub2], device=device))

            lat1 = lat1.unsqueeze(1)
            lng1 = lng1.unsqueeze(1)
            lat2 = lat2.unsqueeze(0)
            lng2 = lng2.unsqueeze(0)

            dlat = lat2 - lat1
            dlng = lng2 - lng1

            a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2) ** 2
            c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
            distances = R * c

        return distances.cpu().numpy()

    def _primary_metric(self, optimization_metric):
        """
        If optimization_metric is a tuple, return the last element of the tuple.
        Otherwise, return optimization_metric itself.

        Args:
            optimization_metric (OptimizationMetric | str | tuple): The optimization metric to get the primary metric from.

        Returns:
            OptimizationMetric | str: The primary optimization metric.
        """
        if isinstance(optimization_metric, tuple):
            return optimization_metric[-1]
        return optimization_metric

    def _build_priority_spec(
        self,
        optimization_metric: OptimizationMetric | str | tuple,
    ):
        """
        preprocesses the optim metric request into the expected tuple format for _dijkstra_single_source

        :param optimization_metric: the target metrics (in order of importance) or a single metric
        :type optimization_metric: OptimizationMetric | str | tuple
        """
        # ensure backwards compatibility with single metric optim
        if isinstance(optimization_metric, (OptimizationMetric, str)):
            return (optimization_metric,)

        # already correct type
        if isinstance(optimization_metric, tuple):
            return optimization_metric

        # list
        if isinstance(optimization_metric, list):
            return tuple(optimization_metric)

        raise TypeError("Invalid optimization_metric")

    def _compute_priority(
        self,
        path: PathNode,
        acc_metrics: EdgeMetadata,
        priority_spec: tuple,
    ):
        """
        collects the values from the current state,
        based on the optim metrics requested

        :param path: the last node i the current path (has prev)
        :type path: PathNode
        :param acc_metrics: the accumulated metrics
        :type acc_metrics: EdgeMetadata
        :param priority_spec: the optim metrics (in order of importance)
        :type priority_spec: tuple
        """
        values = []

        for key in priority_spec:
            if key == "hops":
                values.append(path.length)
            else:
                values.append(acc_metrics.getMetric(key))

        return tuple(values)

    def _dijkstra_single_source(
        self,
        start_id: str,
        target_ids: set[str],
        allowed_modes: list[str],
        optimization_metric: OptimizationMetric | tuple,
        max_segments: int,
        custom_filter: Filter | None,
    ):
        counter = count()

        priority_spec = self._build_priority_spec(optimization_metric)

        pq: list[tuple[tuple, int, PathNode, EdgeMetadata]] = []

        start_metrics = EdgeMetadata(
            transportMode=None,
            **{
                (m if isinstance(m, str) else m.value): 0
                for m in priority_spec
                if m != "hops"
            }
        )

        start_path = PathNode(
            hub_id=start_id,
            mode="",
            edge=EdgeMetadata(),
            prev=None,
        )

        start_priority = self._compute_priority(start_path, start_metrics, priority_spec)
        heapq.heappush(pq, (start_priority, next(counter), start_path, start_metrics))

        # best lexicographic priority seen per hub
        visited: dict[str, tuple] = {}

        # best result per target
        results: dict[str, tuple[PathNode, EdgeMetadata, tuple]] = {}

        if allowed_modes is None:
            allowed_modes = list(self.TransportModes.values())
            if self.drivingEnabled: 
                allowed_modes.append("driving")

        while pq:
            priority, _, path_node, acc_metrics = heapq.heappop(pq)
            hub_id = path_node.hub_id
            path_len = path_node.length

            prev_priority = visited.get(hub_id)
            if prev_priority is not None and prev_priority <= priority:
                continue
            visited[hub_id] = priority

            # record result if this hub is a target
            if hub_id in target_ids:
                prev = results.get(hub_id)
                if prev is None or priority < prev[2]:
                    results[hub_id] = (path_node, acc_metrics, priority)

            if path_len >= max_segments:
                continue

            current_hub = self.getHubById(hub_id)
            if current_hub is None:
                continue

            for mode in allowed_modes:
                if mode not in current_hub.outgoing:
                    continue

                for next_hub_id, conn_metrics in current_hub.outgoing[mode].items():
                    if conn_metrics is None:
                        continue

                    next_hub = self.getHubById(next_hub_id)
                    if next_hub is None:
                        continue

                    if custom_filter is not None:
                        if not custom_filter.filter(
                            current_hub,
                            next_hub,
                            conn_metrics,
                            path_node,
                        ):
                            continue

                    new_acc_metrics = EdgeMetadata(
                        transportMode=None,
                        **acc_metrics.metrics,
                    )
                    for k, v in conn_metrics.metrics.items():
                        if isinstance(v, (int, float)):
                            new_acc_metrics.metrics[k] = (
                                new_acc_metrics.metrics.get(k, 0) + v
                            )
                        else:
                            new_acc_metrics.metrics[k] = v

                    new_path_node = PathNode(
                        hub_id=next_hub_id,
                        mode=mode,
                        edge=conn_metrics,
                        prev=path_node,
                    )

                    new_priority = self._compute_priority(new_path_node, new_acc_metrics, priority_spec)

                    heapq.heappush(
                        pq,
                        (new_priority, next(counter), new_path_node, new_acc_metrics),
                    )

        # strip priority from results (external behavior unchanged)
        return {k: (v[0], v[1]) for k, v in results.items()}

    def _build_route(
        self,
        path_node: PathNode,
        acc_metrics: EdgeMetadata,
        optimization_metric: OptimizationMetric,
        verbose: bool,
    ) -> Route:
        if verbose:
            path = [
                (n.hub_id, n.mode, n.edge)
                for n in path_node
            ]
        else:
            path = [
                (n.hub_id, n.mode)
                for n in path_node
            ]

        return Route(
            path=path,
            totalMetrics=acc_metrics,
            optimizedMetric=optimization_metric,
        )

    # ============= public key functions =============

    def build(self):
        self._generateHubs()
        # exit here if not driving edges are allowed
        if not self.drivingEnabled:
            return

        # build driving edges
        hubTypes = list(self.Graph.keys())
        for i, hubType1 in enumerate(hubTypes):
            hubs1 = list(self.Graph[hubType1].values())
            for _, hubType2 in enumerate(hubTypes[i:], start=i):
                hubs2 = list(self.Graph[hubType2].values())
                distances = self._hubToHubDistances(hubs1, hubs2)

                for hi, hub1 in enumerate(hubs1):
                    for hj, hub2 in enumerate(hubs2):
                        if hub1.id == hub2.id:
                            continue

                        d = distances[hi, hj]
                        if d <= self.maxDrivingDistance:
                            self._addLink(
                                hub1=hub1,
                                hub2=hub2,
                                mode="car",  # explicitly set driving
                                distance=d,
                                bidirectional=True,
                                # no extra metrics for default drive nodes
                            )

    def find_shortest_path(
        self,
        start_id: str,
        end_id: str,
        allowed_modes: list[str] | None = None,
        optimization_metric: OptimizationMetric | str | tuple = OptimizationMetric.DISTANCE,
        max_segments: int = 10,
        verbose: bool = False,
        custom_filter: Filter | None = None,
    ) -> Route | VerboseRoute | None:
        if not isinstance(end_id, str):
            raise TypeError("end_id must be a single hub id (str)")

        if allowed_modes is None:
            allowed_modes = list(self.TransportModes.values())

        results = self._dijkstra_single_source(
            start_id=start_id,
            target_ids={end_id},
            allowed_modes=allowed_modes,
            optimization_metric=optimization_metric,
            max_segments=max_segments,
            custom_filter=custom_filter,
        )

        if end_id not in results:
            return None

        path_node, acc_metrics = results[end_id]

        return self._build_route(
            path_node,
            acc_metrics,
            self._primary_metric(optimization_metric),
            verbose,
        )

    def find_shortest_paths(
        self,
        start_id: str,
        end_ids: list[str],
        allowed_modes: list[str] | None = None,
        optimization_metric: OptimizationMetric | str | tuple = OptimizationMetric.DISTANCE,
        max_segments: int = 10,
        verbose: bool = False,
        custom_filter: Filter | None = None,
    ) -> dict[str, Route | VerboseRoute]:
        if not end_ids:
            return {}

        if allowed_modes is None:
            allowed_modes = list(self.TransportModes.values())

        target_ids = set(end_ids)

        results = self._dijkstra_single_source(
            start_id=start_id,
            target_ids=target_ids,
            allowed_modes=allowed_modes,
            optimization_metric=optimization_metric,
            max_segments=max_segments,
            custom_filter=custom_filter,
        )

        routes: dict[str, Route | VerboseRoute] = {}

        primary_metric = self._primary_metric(optimization_metric)

        for dst, (path_node, acc_metrics) in results.items():
            routes[dst] = self._build_route(
                path_node,
                acc_metrics,
                primary_metric,
                verbose,
            )

        return routes

    def fully_connect_points(self, point_ids: list[str], **kwargs):
        graph = {}
        for start in point_ids:
            targets = [p for p in point_ids if p != start]
            graph[start] = self.find_shortest_paths(start, targets, **kwargs)
        return graph

    def radial_search(
        self,
        hub_id: str,
        radius: float,
        optimization_metric: OptimizationMetric | str = OptimizationMetric.DISTANCE,
        allowed_modes: list[str] = None,
        custom_filter: Filter = None,
    ) -> list[float, Hub]:
        """
        Find all hubs within a given radius of a given hub
        (Note: distance is measured from the connecting paths not direct)

        Args:
            hub_id: ID of the center hub
            radius: maximum distance from the center hub
            optimization_metric: metric to optimize for (e.g. distance, time, cost)
            allowed_modes: list of allowed transport modes (default: None => all modes)

        Returns:
            list of tuples containing the metric value and the corresponding hub object
        """

        center = self.getHubById(hub_id)
        if center is None:
            return [center]

        if allowed_modes is None:
            allowed_modes = list(self.TransportModes.values())

        hubsToSearch = deque([center])
        queued = set([hub_id])
        reachableHubs: dict[str, tuple[float, Hub]] = {hub_id: (0.0, center)}

        while hubsToSearch:
            hub = hubsToSearch.popleft() # get the current hub to search
            currentMetricVal, _ = reachableHubs[hub.id] # get the current metric value
            for mode in allowed_modes:
                outgoing = hub.outgoing.get(mode, {}) # find all outgoing connections
                # dict like {dest_id: EdgeMetadata}
                for id, edgemetadata in outgoing.items(): # iter over outgoing connections
                    thisMetricVal = edgemetadata.getMetric(optimization_metric)
                    if thisMetricVal is None:
                        continue
                    nextMetricVal = currentMetricVal + thisMetricVal
                    if nextMetricVal > radius:
                        continue
                    knownMetric = reachableHubs.get(id, None)
                    destHub = self.getHubById(id)
                    if custom_filter is not None and not custom_filter.filter(hub, destHub, edgemetadata):
                        continue
                    # only save smaller metric values
                    if knownMetric is None or knownMetric[0] > nextMetricVal:
                        reachableHubs.update({id: (nextMetricVal, destHub)})
                    if id not in queued:
                        queued.add(id)
                        hubsToSearch.append(destHub)

        return [v for v in reachableHubs.values()]

    def compare_routes(
        self,
        start_id: str,
        end_id: str,
        allowed_modes: list[str],
        metrics_to_compare: list[OptimizationMetric] = None
    ) -> dict[OptimizationMetric, Route]:
        """
        Find optimal routes for different metrics and compare them

        Returns:
            Dictionary mapping each optimization metric to its optimal route
        """
        if metrics_to_compare is None:
            metrics_to_compare = list(OptimizationMetric)

        results = {}
        for metric in metrics_to_compare:
            route = self.find_shortest_path(start_id, end_id, optimization_metric=metric, allowed_modes=allowed_modes)
            if route:
                results[metric] = route

        return results
