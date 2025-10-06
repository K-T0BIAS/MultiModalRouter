# dataclasses.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


from multimodalrouter import RouteGraph
from multimodalrouter.graphics import GraphDisplay
import os


# custom transform to make lat lng to x y (-> lng lat)
def NodeTransform(coords):
    for coord in coords:
        yield list((coord[0], coord[1]))


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    # initialize the graph
    graph = RouteGraph(
        maxDistance=50,
        transportModes={"cell": "walk", },
        dataPaths={"cell": os.path.join(path, "data", "maze.csv")},
        compressed=False,
        drivingEnabled=False
    )

    graph.build()
    # init the display
    display = GraphDisplay(graph)
    # display the graph (uses the transform to swap lat lng to x y)
    display.display(nodeTransform=NodeTransform)
