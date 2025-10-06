# demo.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information

from multimodalrouter import RouteGraph
import os
import pandas as pd


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is not installed. Please install matplotlib to use this example.")

    path = os.path.dirname(os.path.abspath(__file__))
    # init the maze df for the plot
    mazeDf = pd.read_csv(os.path.join(path, "data", "maze.csv"))
    # init the plot
    plt.figure(figsize=(10, 10))
    # draw the maze
    # draw the maze (grid lines)
    for _, row in mazeDf.iterrows():
        plt.plot(
            [row.source_lng, row.destination_lng],   # x = "lng" column
            [row.source_lat, row.destination_lat],   # y = "lat" column
            "k-"
        )

    # initialize the graph
    graph = RouteGraph(
        maxDistance=50,
        transportModes={"cell": "walk", },
        dataPaths={"cell": os.path.join(path, "data", "maze.csv")},
        compressed=False,
        drivingEnabled=False
    )
    # build the graph
    graph.build()
    # find the shortest route
    route = graph.find_shortest_path(
        start_id="cell-(0, 0)",
        end_id="cell-(0, 9)",
        allowed_modes=["walk"],
        verbose=True,
        max_segments=100
    )
    # print the route
    print(route.flatPath if route else "No route found")
    # make the route blue in the plot
    s_prev = None
    for s, _, _ in route.path:
        if s_prev is not None:
            h1 = graph.getHubById(s_prev)
            h2 = graph.getHubById(s)
            # Swap coords so x=column, y=row
            plt.plot(
                [h1.coords[1], h2.coords[1]],  # x-axis
                [h1.coords[0], h2.coords[0]],  # y-axis
                "b-"
            )
        s_prev = s

    # display the plot
    plt.show()


if __name__ == "__main__":
    main()
