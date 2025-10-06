# dataclasses.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


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
