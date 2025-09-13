from ..graph.graph import RouteGraph
import os

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    graph = RouteGraph(
        maxDistance=50,
        transportModes={"airport": "fly", },
        dataPaths={"airport": os.path.join(path, "..", "data", "fullDataset.parquet")},
        saveMode="light",
        compressed=False
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
        max_segments=10  # adjust if needed
    )

    if route:
        print(f"Route found from {start_airport_id} to {end_airport_id}:")
        for leg_id, leg_mode in route.path:
            print(f" -> {leg_id} ({leg_mode})")
        print("Total metrics:", route.totalMetrics)
    else:
        print(f"No route found from {start_airport_id} to {end_airport_id} using {modes}")

    start_id = "SFO"
    end_id = "OAK" 

    for hub in graph._allHubs():
        if "drive" in hub.outgoing:
            print(f"{hub.id} can drive to {list(hub.outgoing['drive'].keys())}")

    modes = ["drive"]

    # Find shortest path by driving
    route = graph.find_shortest_path(
        start_id=start_id,
        end_id=end_id,
        allowed_modes=modes,  # only consider driving
        max_segments=20           # increase if necessary
    )

    if route:
        print(f"Route found from {start_id} to {end_id}:")
        for leg_id, leg_mode in route.path:
            print(f" -> {leg_id} ({leg_mode})")
        print("Total metrics:", route.totalMetrics)
    else:
        print(f"2 No route found from {start_id} to {end_id} using {modes}")

    for hub in graph._allHubs():
        if hub.id in ["SFO", "OAK"]:
            print(hub.id, hub.lat, hub.lng)


 






