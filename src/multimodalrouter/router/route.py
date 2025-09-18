# router.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


from ..graph import RouteGraph
import argparse
import os


def main():
    graph = RouteGraph.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "graph.dill"),
        compressed=False
    )

    parser = argparse.ArgumentParser(
        description="parse the arguments"
    )
    parser.add_argument(
        "--start",
        nargs="+",
        type=float,
        required=True,
        help="Start coordinates"
    )
    parser.add_argument(
        "--end",
        nargs="+",
        type=float,
        required=True,
        help="End coordinates"
    )
    parser.add_argument(
        "--allowedModes",
        nargs="+",
        type=str,
        default=["car"],
        help="Allowed transport modes"
    )
    parser.add_argument(
        "--maxSegments",
        type=int,
        default=10,
        help="Maximum number of segments in the route"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output for the paths"
    )
    args = parser.parse_args()

    start_hub = graph.findClosestHub(["airport"], args.start)
    end_hub = graph.findClosestHub(["airport"], args.end)

    if start_hub is None or end_hub is None:
        print("One of the airports does not exist in the graph")
        return

    route = graph.find_shortest_path(start_id=start_hub.id,
                                     end_id=end_hub.id,
                                     allowed_modes=args.allowedModes,
                                     max_segments=args.maxSegments,
                                     verbose=args.verbose)

    print(route.flatPath if route else "No route found")


if __name__ == "__main__":
    main()
