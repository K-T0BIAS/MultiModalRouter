# demo.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information

from multimodalrouter import RouteGraph
import os


def main():
    path = os.path.dirname(os.path.abspath(__file__))
    # initialize the graph
    graph = RouteGraph(
        maxDistance=50,
        transportModes={"airport": "plane", },
        dataPaths={"airport": os.path.join(path, "data", "fullDataset.csv")},
        compressed=False,
    )
    # build the graph
    graph.build()
    # set start and end points
    start = [60.866699, -162.272996] # Atmautluak Airport
    end = [60.872747, -162.5247] # Kasigluk Airport

    start_hub = graph.findClosestHub(["airport"], start) # find the hubs
    end_hub = graph.findClosestHub(["airport"], end)
    # find the route
    route = graph.find_shortest_path(
        start_hub.id,
        end_hub.id,
        allowed_modes=["plane", "car"],
        verbose=True
    )
    # print the route
    print(route.flatPath if route else "No route found")


if __name__ == "__main__":
    main()
