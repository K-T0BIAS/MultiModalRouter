[HOME](../README.md)

[graph](#routegraph)

[dataclasses](#dataclasses)

[Hub](#Hub)

[EdgeMetadata](#edgemetadata)

[Route](#route)

# RouteGraph

The `RouteGraph` is the central class of the package implemented as a dynamic (cyclic) directed graph. It defines all graph building and routing related functions.


##  Functionality

### initialization

This creates a new graph with no `Hubs` or `Edges`.

```python
def __init__(
    self, 
    maxDistance: float,
    transportModes: dict[str, str],
    dataPaths: dict[str, str] = {},
    compressed: bool = False,
    extraMetricsKeys: list[str] = [],
    drivingEnabled: bool = True,
):
```

#### args

Terminology:

> Hub Type: hub types are the user defined names for their hubs. e.g. when having data for flights you have `airports`, thus you may want to define the hubs for the airports as type `airport`. (Hub Types can be anything you want to name them) 

- ``maxDistance``: float = The maximum distance a driving edge is allowed to span
- ``transportModes``: dict[str, str] = a dictionary that assigns Hub Types to their mode of travel. E.g. 
```python
transportModes = {
    'airport': 'fly',# here hub type airport is assigned its primary mode of travel as fly
}
```
- ``dataPaths``: dict[str, str] = a dictionary that stores the paths to datasets realtive to their Hub Types. E.g.:
```python
dataPaths = {
    # hub type: path to dataset
    'airport': '~/MUltiModalRouter/data/AirportDataset.parquet'
}
```
- ``compressed``: bool = wheter to save this graph in compressed files or not (NOTE: this is not used at the moment so just skip)
- ``extraMetricsKeys``: list[str] = a list of metrics the graph will search for in the datasets when building edges (NOTE: default metrics must still be present)
Example:
```python
# given at least one dataset with the col 'time'

extraMetricsKeys = ['time']
```
When the graph finds this key in a dataset it will then add this metric (here `time`) to all edges that come from hubs stored inside this dataset

- ``drivingEnabled``: bool = whether the graph should connect all possible hubs that have $distance(a,b) \leq maxDistance$ (default=True)

#### example

Init a graph with Hubs: `airport`, `trainstation`

```python
from multimodalrouter import RouteGraph

graph = RouteGraph(
    maxDistance = 50,
    transportModes = {
        'airport': 'plane',
        'trainstation': 'train'
    },
    dataPaths = {
        'airport': pathToAirportData,
        'trainstation': pathToTrainData
    }
    # time and cost must each be present in at least one dataset
    extraMetricsKeys = ['time', 'cost'], 
    # default is True so this is not necessary 
    drivingEnabled = True, 
)
```

The resulting graph will be able to build `HUbs` for both `train stations` and `airports`. It will also use the extra metrics in all edges where the data is present

### build

After a graph is initialized it doesn't contain any actual nodes or edges yet. To create the nodes and edges the graph has to be build.

```python
def build(self):
```

#### example

click [here](#example) to see how to init the graph

```python
# with the graph from the previous example

graph.build()
```

After this finishes the graph is build and ready for routing

### routing / finding the shortest Path form A to B

```python
def find_shortest_path(
    self, 
    start_id: str, 
    end_id: str, 
    allowed_modes: list[str],
    optimization_metric: OptimizationMetric | str = OptimizationMetric.DISTANCE,
    max_segments: int = 10,
    verbose: bool = False
    ) -> Route | None:
```

#### args

- start_id: str = the Hub.id of the starting hub (e.g. the source field for this hub in your data -> for `airports` likely the iata code) (for coordinate searches see [here](#searching-with-coordinates))
- end_id: str = the Hub.id of the traget Hub
- allowed_modes: list[str] = a list of transport modes that are allowed in the path (all edges with different modes are excluded)(The modes are set during the graph [initailization](#args))
- optimization_metric: str = the metric by which the pathfinder will determine the length of the path (must be numeric and present in all searched edges) (default = `distance`) (metrics where also set during [initialization](#args))
- max_segments: int = the maximum number of hubs the route is allowed to include (default = 10 to avoid massive searches but should be setvrealtive to the graph size and density)
- verbose: bool = whether you want to store all edges and their data in the route or just the hub names (default=False)

**returns** : [Route](#route) or None if no route was found

### save

```python
def save(
    self, 
    filepath: str = os.path.join(os.getcwd(), "..", "..", "..", "data"), 
    compressed: bool = False):
```

The `save` method will create a save file from the last garph state. Depending on the `arguments` the file will either be stored as `.dill` or `.zlib`.
A save file contains the complete statedict of the RouteGraph instance except attributes that could break the pickling process (e.g. ``threading.Lock``).

#### args

- filepath: str = the directory that the savefile will be stored in (defaults to `MultiModalRouter/data`)
- compressed: bool = whether to compress the output into `.zlib` or store as `.dill`

#### example

Saving a graph to a custom dir, in a `.dill` file

```python
...
graph.save(filepath=customDir)
```

### load

```python
@staticmethod
def load(
    filepath: str, 
    compressed: bool = False
) -> "RouteGraph":
```

The load method is a static method that allows you to load a graph from its save file into a new graph object.
The resulting graph object is fully initialized and can be used as is.

#### args

- filepath: str = the full path to your save file 
- compressed: bool = set this to `True` if your graph was saved to a `.zlib` compressed file (default=`False`)

#### example

```python
from multimodalrouter import RouteGraph
# load a .dill file from 'pathToMyGraph'
myGraph = RouteGraph.load(filepath=pathToMyGraph) 
```

The `myGraph` object is now fully loaded and can be used to its full extend.

### searching with coordinates

Since searching by hub id is not always possible the graph has a helper that finds a hub closest to a coordinate tuple.

```python
def findClosestHub(
    self, 
    allowedHubTypes: list[str], 
    lat: float, 
    lon: float
) -> Hub | None:
```

#### args


- allowedHubTypes: list[str] = a list that defines which hubs should be searched (e.g. ['airport','trainstation'])
**NOTE:** if you set this to `None` all hubs will be included in 
the search
- lat: float = the latitude of your search point (not necessarily the latitude of a hub)
- lon: float = the longitude of your search point

> NOTE: the lat and lon must not necessarily be in degrees or any other meaningfull metric aslong as your data provides distances and you turn of enableDrive when building the graph

> NOTE: it is entirely possible to setup the graph with custom coordinate systems and distances

#### example

```python
coordinates = 100.0, 100.0
closestHub = graph.findClosestHub(
    allowedHUbTypes = None, # include all types in search
    *coordinates, 
)
```

> NOTE: you can now use `closestHub.id` in the [search](#routing--finding-the-shortest-path-form-a-to-b)

### getting hubs by id

If you want to inspect a hub and you know its `id` you can get it from the graph as follows

```python
def getHub(
    self, 
    hubType: str, 
    id: str
) -> Hub | None:
```

or 

```python
def getHubById(
    self, 
    id: str
) -> Hub | None:
```

#### args

- hubType: str = the type of the target hub
- id: str = the id of the target hub

**returns:** the Hub or if not found None

### manually adding Hubs

If you want to add a new Hub to a graph without building use this:

```python
def addHub(self, hub: Hub):
```

This will add your Hub to the garph and if its already present it will fail silently

### args

- hub: Hub = the Hub you want to add

---
---

### advanced options

#### swap distance method

When your dataset comes with neither distances nor a coordinate system in degrees you can mount your own distance function. This way you will still be able to build the default driving edges etc.

#### example

```python
from multimodalrouter import RouteGraph
import types
# define your own distance metric (NOTE the arguments must be the same as here)
def myDistancMetric(self, hub1: list[Hub], hub2: list[Hub]):
    ...
    return distances # np.array or list

# create a normal graph object 
specialGraph = RouteGraph(**kwargs)
# swap the distance method
specialGraph._hubToHubDistances = types.MethodType(myDistanceMetric, specialGraph)
# continue as you would normally
graph.build()
```

#### NOTES

- Naturally you can do the same thing for the preprocessor to calculate the transport mode based distances in the preprocessessing step.

---
---
---

## Dataclasses

### Hub

Hubs are the nodes of the [RouteGraph](#routegraph) and store all outgoing connections alongside the relevant [EdgeMetadata](#edgemetadata)

```python
def __init__(
    self, 
    lat: float, 
    lng: float, 
    id: str, 
    hubType: str
):
```

#### fields

- lat: float = the latitude coordinate of the Hub (NOTE: this can be a value in any coordinate system of your coice aslong as the graph was initialized accordingly and your data supports it)
- lng: float = the longitude of the Hub (NOTE: the same conditions as for the latitude apply here)
- id: str = a string id like iata code UNLOCODE or whatever you want (NOTE: must be unique for the hubType)
- hubType: str = the type of hub this will be (e.g. `airport`, `trainstation`,...)

#### adding edges

```python
def addOutgoing(
    self, 
    mode: str, 
    dest_id: str, 
    metrics: EdgeMetadata):
```

#### args

- mode: str = the mode of transport along this edge (e.g. `plane`, `car`,...)
- dest_id: str = the id of the destination Hub
- metrics: [EdgeMetadata](#edgemetadata) = the edge object that stores the metrics for this connection

#### getting the edge metrics 

Get the edgeMetadata from this Hub to another, with a given transport mode

```python
def getMetrics(
    self, 
    mode: str, 
    dest_id: str
)-> EdgeMetadata:
```

#### args 

- mode: str = the mode of transport along the edge
- dest_id: str = the id of the destination Hub

**returns:** the edgeMetadata or None if this edge doesn't exist

---
---
### EdgeMetadata

These objects store data about one edge such as the `transport mode` and metrics like `distance` etc.

```python
def __init__(
    self, 
    transportMode: str = None, 
    **metrics):
```

#### args

- transportMode: str = the transpot mode across this edge
- **metrics: dict = a dictionary of edge metrics like `distance`, `time` etc

#### example

create data for an edge that is traversed via `plane`, has a `distance` of `100.0` and `cost` of `250.0`

```python
edgeData = EdgeMetadata(
    transportMode = 'plane',
    **{'distance': 100.0, 'cost': '250.0'}
)
```

#### get a specific metric

```python
def getMetric(
    self,
    metric: OptimizationMetric | str
):
```

#### args

- metric: str = the name of the metric you want to retrieve

---
---

### Route

A dataclass to store all route related data; like Hubs and edges.


#### fields

```python
path: list[tuple[str, str]]
totalMetrics: EdgeMetadata
optimizedMetric: OptimizationMetric
```


#### properties
---

```
@property
    def flatPath(
        self, 
        toStr=True):
```

By calling `route.flatPath` you will get the string representation of the route 

#### example output

> NOTE: this is a verbose route from `-1.680000, 29.258334` to `3.490000, 35.840000`, connected through airports with data from [open flights](https://openflights.org/data.php)

```text
Start: GOM
        Edge: (transportMode=plane, metrics={'distance': 85.9251874180552})
-> BKY
        Edge: (transportMode=drive, metrics={'distance': np.float32(20.288797)})
-> KME
        Edge: (transportMode=plane, metrics={'distance': 147.44185301830063})
-> KGL
        Edge: (transportMode=plane, metrics={'distance': 757.9567739118678})
-> NBO
        Edge: (transportMode=plane, metrics={'distance': 515.1466233682448})
-> LOK
```




