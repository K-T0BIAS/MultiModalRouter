[HOME](./index.md)

# Graph Plotting

Using the build-in graph plotting tool you can [plotly](https://plotly.com/python/) plot any graph in `2D` or `3D`, while defining [transformations](#transformations) for your coordiante space or even path curvature etc.

## GraphDisplay

```python
def __init__(
    self,
    graph: RouteGraph,
    name: str = "Graph",
    iconSize: int = 10
) -> None:
```

#### args:
    - graph: RouteDisplay = the graph instance you want to plot
    - name: str = (not in use at the moment)
    - iconSize: int = the size of the nodes in the plot

#### example

```
gd = GraphDisplay(myGraphInstance)
```

[flight path CODE example on sphere](./examples/flightRouter/plot.py)


### display()

The display function will collect data from your Graph and create a [plotly](https://plotly.com/python/) plot from it.

```python
def display(
    self,
    nodeTransform=None,
    edgeTransform=None,
    displayEarth=False
):
```

#### args:

- nodeTransform: function = a [transformation](#transformations) function that transformes all node coordinates
- edgeTransform: funstion = a function that [transformes](#transformations) all your edges
- displayEarth: bool = if True -> will display a sphere that (roughly) matches earth

#### example:

this call will create the plot for your graph while mapping all coords onto the surface of the earth

```python
gd.display(
    nodeTransform = gd.degreesToCartesian3D,
    displayEarth: True
)
```

### transformations

#### base function style

IF you want to implement your own transformation function note that the call must adhere to the following parameters:

```python
def customNodeTrandsform(coords: list[list[float]]):
    return list[list[float]]

def customEdgeTransform(start: list[list[float]], end: list[list[float]]):
    return list[list[list[float]]]
```

#### args

- coords: list[list[float]] = a nested list of coordinates for all nodes
- start: list[list[float]] = a nested list of all start coordinates
- end: list[list[float]] = a nested list of all end coordinates

#### returns:

- list[list[float]] = a list of all transformed node coordinates
- list[list[list[float]]] = a list of curves whare each curve / edge can have n points defining it 

### build-in Node Transforms:

#### degreesToCartesian3D

```python
@staticmethod
    def degreesToCartesian3D(coords):
```
This function maps any valid `2D` coordinates (best if in degrees) to spherical coords on the surface of earth

### build-in Edge Transformations

```python
@staticmethod
    def curvedEdges(start, end, R=6371.0, H=0.05, n=20):
```

curves edges for coordinates on spheres (here earth) so that the edges curve along the spherical surface with a curvature that places the midpoint of the curve at $H \dot R$ above the surface. (great for displaying flights). 

If torch is installed this will use great-circle distance for the curves

> Note if torch is not installed this will fall back to using `math` with quadratic bezier curves -> some curves may end up inside the sphere to bezier inaccuracy 