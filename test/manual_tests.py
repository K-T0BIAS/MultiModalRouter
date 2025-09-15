# manual_tests.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


from src.multimodalrouter.graph.graph import RouteGraph
import os

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    graph = RouteGraph(
        maxDistance=50,
        transportModes={"airport": "fly", },
        dataPaths={"airport": os.path.join(path, "..", "data", "fullDataset.parquet")},
        compressed=False,
        extraMetricsKeys=["type_x"]
    )

    graph.build()
    # graph.debug_hubs_and_edges()
    start_airport_id = "FAE"
    end_airport_id = "GOH"    # replace with your target airport code

    start_hub = graph.getHub("airport", start_airport_id)
    end_hub = graph.getHub("airport", end_airport_id)

    if start_hub is None or end_hub is None:
        print("One of the airports does not exist in the graph")

    def print_outgoing(hub):
        for mode, neighbors in hub.outgoing.items():
            print(f"Hub {hub.id} outgoing via {mode}: {list(neighbors.keys())}")

    print("Start airport connections:")
    print_outgoing(start_hub)

    print("End airport connections:")
    print_outgoing(end_hub)

    modes = ["fly", "drive"]

    route = graph.find_shortest_path(
        start_id=start_airport_id,
        end_id=end_airport_id,
        allowed_modes=modes,
        max_segments=10,
        verbose=True
    )

    if route:
        print(f"Route found from {start_airport_id} to {end_airport_id}:")
        # for leg_id, leg_mode in route.path:
        #     print(f" -> {leg_id} ({leg_mode})")
        print(route.flatPath)
        print("Total metrics:", route.totalMetrics)
    else:
        print(f"No route found from {start_airport_id} to {end_airport_id} using {modes}")

    start_id = "SFO"
    end_id = "OAK" 

    modes = ["drive"]

    # Find shortest path by driving
    route = graph.find_shortest_path(
        start_id=start_id,
        end_id=end_id,
        allowed_modes=modes, 
        max_segments=20,
    )

    if route:
        print(f"Route found from {start_id} to {end_id}:")
        for leg_id, leg_mode in route.path:
            print(f" -> {leg_id} ({leg_mode})")
        print("Total metrics:", route.totalMetrics)
    else:
        print(f"2 No route found from {start_id} to {end_id} using {modes}")

    # test finding the closest hub of type x to a point
    lat = 37.6
    lng = -122.
    hub_type = "airport"
    closest_hub = graph.findClosestHub(allowedHubTypes=[hub_type], lat=lat, lon=lng)
    print(f"The closest {hub_type} hub to ({lat}, {lng}) is {closest_hub.id}")


 






