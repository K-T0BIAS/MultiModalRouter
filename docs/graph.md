[HOME](../README.md)

# RouteGraph

The `RouteGraph` is the central class of the package. It defines all graph building and routing related functions.

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

- maxDistance: float = The maximum distance a driving edge is allowed to span
- transportModes: dict[str, str] = a dictionary that assigns Hub Types to their mode of travel. E.g. 
```python
transportModes = {
    'airport': 'fly',# here hub type airport is assigned its primary mode of travel as fly
}
```
- dataPaths: dict[str, str] = a dictionary that stores the paths to datasets realtive to their Hub Types. E.g.:
```python
dataPaths = {
    # hub type: path to dataset
    'airport': '~/MUltiModalRouter/data/AirportDataset.parquet'
}
```
- compressed: bool = wheter to save this graph in compressed files or not (NOTE: this is not used at the moment so just skip)
- extraMetricsKeys: list[str] = a list of metrics the graph will search for in the datasets when building edges (NOTE: default metrics must still be present)
Example:
```python
# given at least one dataset with the col 'time'

extraMetricsKeys = ['time']
```
When the graph finds this key in a dataset it will then add this metric (here `time`) to all edges that come from hubs stored inside this dataset

- drivingEnabled: bool = whether the graph should connect all possible hubs that have $distance(a,b) \leq maxDistance$ (default=True)

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
