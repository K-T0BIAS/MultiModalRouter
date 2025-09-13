from dataclasses import dataclass
from enum import Enum

class OptimizationMetric(Enum):
    DISTANCE = "distance"

class EdgeMetadata:
    __slots__ = ['transportMode', 'metrics']

    def __init__(self, transportMode: str = None, **metrics):
        self.transportMode = transportMode
        self.metrics = metrics  # e.g., {"distance": 12.3, "time": 15} NOTE distance is required by the graph

    def getMetric(self, metric: OptimizationMetric):
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
    __slots__ = ['lat', 'lng', 'id', 'outgoing', 'hubType']

    def __init__(self, lat: float, lng: float, id: str, hubType: str):
        self.lat = lat
        self.lng = lng
        self.id = id
        self.hubType = hubType
        self.outgoing = {}

    def addOutgoing(self, mode: str, dest_id: str, metrics: EdgeMetadata):
        if mode not in self.outgoing:
            self.outgoing[mode] = {}
        self.outgoing[mode][dest_id] = metrics

    def getMetrics(self, mode: str, dest_id: str) -> EdgeMetadata:
        return self.outgoing.get(mode, {}).get(dest_id, None)
    
    def getMetric(self, mode: str, dest_id: str, metric: str) -> float:
        connection = self.outgoing.get(mode, {}).get(dest_id)
        return getattr(connection, metric, None) if connection else None
    
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
    def flatPath(self):
        """Flatten the path into a list of hub IDs"""
        if not self.path:
            return []
        # get all source hubs plus the final destination
        return [start for start, _ in self.path] + ([self.path[-1][0]] if self.path else [])

