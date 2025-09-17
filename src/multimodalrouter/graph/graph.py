# graph.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


from collections import defaultdict
from tqdm import tqdm
import zlib
import dill
import heapq
import os
import pandas as pd
from .dataclasses import Hub, EdgeMetadata, OptimizationMetric, Route
from threading import Lock


class RouteGraph:

    def __init__(self, 
                 maxDistance: float, # sets the max distance to connect hubs with driving edges
                 transportModes: dict[str, str], # dict like {hubtype -> "airport": "fly", "shippingport": "shipping"}
                 dataPaths: dict[str, str] = {}, # dict like {hubtype -> "airport": path -> "airports.csv", "shippingport": "shippingports.csv"}
                 compressed: bool = False, # if true model file will be compressed otherwise normal .dill file
                 extraMetricsKeys: list[str] = [], # list of extra columns to add to the edge metadata (dynamically added to links when key is present in dataser)
                 drivingEnabled: bool = True, # if true will connect hubs with driving edges
                ):

        self.compressed = compressed
        self.extraMetricsKeys = extraMetricsKeys
        self.drivingEnabled = drivingEnabled

        self.TransportModes = transportModes
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
        
        tempHub = Hub(coords = coords, hubType="temp", id="temp")
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
    

    def save(self, filepath: str = os.path.join(os.getcwd(), "..", "..", "..", "data"), compressed: bool = False):
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
        self, hub1: Hub, hub2: Hub, mode: str, distance: float,
        bidirectional: bool = False, extraData: dict | None = None
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
        dataPath =getattr(self, targetHubType + "DataPath")
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
    
    def _generateHubs(self, 
                      sourceCoordKeys: list[str] = ["source_lat", "source_lng"], 
                      destinationCoodKeys: list[str] = ['destination_lat', 'destination_lng']):
        """
        Generate Hub instances and link them with EdgeMetadata.
        Extra columns in the data will be added to EdgeMetadata dynamically.
        """
        # no lock needed since underlying methods have locks
        for hubType in self.Graph.keys():
            data = self._loadData(hubType)
            added = set()

            # get required and extra columns
            required_cols = {
                "source", "destination",
                *sourceCoordKeys, 
                *destinationCoodKeys,
                "distance"
            }

            # collect extra data from the dataset columns that are not required but marked as extra
            extra_metric_cols = []
            for m in self.extraMetricsKeys:
                if m not in required_cols:
                    try:
                        extra_metric_cols.append(m)
                    except:
                        continue

            # extra_metric_cols = [col for col in data.columns if col not in required_cols and col in extraMetricsKeys]

            for row in tqdm(data.itertuples(index=False), desc=f"Generating {hubType} Hubs", unit="hub"):
                # create hubs if they don't exist
                if row.source not in added:
                    hub = Hub(coords = [getattr(row, k) for k in sourceCoordKeys], id=row.source, hubType=hubType)
                    self.addHub(hub)
                    added.add(row.source)

                if row.destination not in added:
                    hub = Hub(coords = [getattr(row, k) for k in destinationCoodKeys], id=row.destination, hubType=hubType)
                    self.addHub(hub)
                    added.add(row.destination)

                # get extra metrics
                extra_metrics = {col: getattr(row, col) for col in extra_metric_cols}

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

    def find_shortest_path(self, start_id: str, end_id: str, 
                          allowed_modes: list[str],
                          optimization_metric: OptimizationMetric | str = OptimizationMetric.DISTANCE,
                          max_segments: int = 10,
                          verbose: bool = False
                          ) -> Route | None:
        """
        Find the optimal path between two hubs using Dijkstra
        
        Args:
            start_id: ID of the starting hub
            end_id: ID of the destination hub
            optimization_metric: Metric to optimize for (distance, time, cost, etc.) (must exist in EdgeMetadata)
            allowed_modes: List of allowed transport modes (default: all modes)
            max_segments: Maximum number of segments allowed in route
            
        Returns:
            Route object with the optimal path, or None if no path exists
        """
            
        # check if start and end hub exist
        start_hub = self.getHubById(start_id)
        end_hub = self.getHubById(end_id)
        
        if start_hub is None:
            raise ValueError(f"Start hub '{start_id}' not found in graph")
        if end_hub is None:
            raise ValueError(f"End hub '{end_id}' not found in graph")
        
        if start_id == end_id:
            # create a route with only the start hub
            # no verbose since no edges are needed
            return Route(
                path=[(start_id, "")], 
                totalMetrics=EdgeMetadata(),
                optimizedMetric=optimization_metric,
            )
        
        if verbose:
            # priority queue: (metric_value, hub_id, path_with_modes, accumulated_metrics)
            pq = [(0.0, start_id, [(start_id, "", EdgeMetadata())], EdgeMetadata())]
        else: 
            # priority queue: (metric_value, hub_id, path_with_modes, accumulated_metrics)
            pq = [(0.0, start_id, [(start_id, "")], EdgeMetadata())]

        visited = {} # dict like {hub_id : metric_value}
        
        while pq:
            # get the current path data
            # optim metric,         hub id,       path with modes, accumulated metrics (edgeMetadata object)
            current_metric_value, current_hub_id, path_with_modes, accumulated_metrics = heapq.heappop(pq)
            
            # skip this if a better path exists
            if current_hub_id in visited and visited[current_hub_id] <= current_metric_value:
                continue
            # mark as visited
            visited[current_hub_id] = current_metric_value
            
            # check if this is the end hub
            if current_hub_id == end_id:
                if verbose:
                    return Route(
                        path=path_with_modes, 
                        totalMetrics=accumulated_metrics,
                        optimizedMetric=optimization_metric,
                    )
                
                return Route(
                    path=path_with_modes, 
                    totalMetrics=accumulated_metrics,
                    optimizedMetric=optimization_metric,
                )
            
            # skip if too many segments
            if len(path_with_modes) > max_segments:
                continue
                
            # get the current hub
            current_hub = self.getHubById(current_hub_id)
            if current_hub is None:
                continue
            
            # test all outgoing connections from the current hub
            for mode in allowed_modes: # iter over the allowed transport modes
                if mode in current_hub.outgoing: # check if the mode has outgoing connections
                    # iter over all outgoing links with the selected transport type
                    for next_hub_id, connection_metrics in current_hub.outgoing[mode].items(): 
                        if connection_metrics is None: # skip if the connection has no metrics
                            continue
                        
                        # get the selected metric alue for this connection
                        connection_value = connection_metrics.getMetric(optimization_metric)
                        new_metric_value = current_metric_value + connection_value
                        
                        # skip if a better hub to get here exists
                        if next_hub_id in visited and visited[next_hub_id] <= new_metric_value:
                            continue
                        
                        # create a new edge obj for the combined metrics  |  None bc modes my change between edges (mode is not important here either)
                        new_accumulated_metrics = EdgeMetadata(transportMode=None, **accumulated_metrics.metrics)
                        # accumulate metrics
                        for metric_name, metric_value in connection_metrics.metrics.items():
                            if isinstance(metric_value, (int, float)):
                                new_accumulated_metrics.metrics[metric_name] = new_accumulated_metrics.metrics.get(metric_name, 0) + metric_value
                            else:
                                # ignore non-numeric metrics for accumulation (maybe combine strings here)
                                new_accumulated_metrics.metrics[metric_name] = metric_value

                        # combine to form a new path
                        if verbose:
                            new_path = path_with_modes + [(next_hub_id, mode, connection_metrics)]
                        else:
                            new_path = path_with_modes + [(next_hub_id, mode)]
                        # push to the priority queue for future exploration
                        heapq.heappush(pq, (new_metric_value, next_hub_id, new_path, new_accumulated_metrics))
        
        return None
    

    def compare_routes(self, start_id: str, end_id: str, allowed_modes: list[str],
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