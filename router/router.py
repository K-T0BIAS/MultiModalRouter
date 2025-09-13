from graph.graph import RouteGraph
import argparse
import os

def main():
    graph = RouteGraph.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "graph.dill"), compressed=False)

    parser = argparse.ArgumentParser(
        description="parse the arguments"
    )
    parser.add_argument(
        "--start",
        nargs=2,
        type=float,
        required=True,
        help="Start coordinates"
    )
    parser.add_argument(
        "--end",
        nargs=2,
        type=float,
        required=True,
        help="End coordinates"
    )
    parser.add_argument(
        "--allowedModes",
        nargs="+",
        type=str,
        default=["drive"],
        help="Allowed transport modes"
    )
    args = parser.parse_args()

    start_lat, start_lng = args.start
    end_lat, end_lng = args.end

    start_hub = graph.findClosestHub(["airport"], start_lat, start_lng)
    end_hub = graph.findClosestHub(["airport"], end_lat, end_lng)

    if start_hub is None or end_hub is None:
        print("One of the airports does not exist in the graph")
        return

    route = graph.find_shortest_path(start_hub.id, end_hub.id, args.allowedModes)
    print(route)

if __name__ == "__main__":
    main()