from multimodalrouter import RouteGraph
from multimodalrouter.graphics import GraphDisplay
import os

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    graph = RouteGraph(
        maxDistance=50,
        transportModes={"airport": "fly", },
        dataPaths={"airport": os.path.join(path, "data", "fullDataset.csv")},
        compressed=False,
    )

    graph.build()
    display = GraphDisplay(graph)
    display.display(
        displayEarth=True,
        nodeTransform=GraphDisplay.degreesToCartesian3D,
        edgeTransform=GraphDisplay.curvedEdges
    )