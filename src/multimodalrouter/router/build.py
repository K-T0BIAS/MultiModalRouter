# build.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from ..graph import RouteGraph
import argparse
import os

def main():
    print("Building graph...")
    parser = argparse.ArgumentParser(
        description="Collect key-value1-value2 triplets into two dicts"
    )
    parser.add_argument(
        "data",
        nargs="+",
        help="Arguments in groups of 3: hubType transportMode dataPath"
    )
    parser.add_argument(
        "--maxDist",
        type=int,
        default=50,
        help="Maximum distance to connect hubs with driving edges"
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Whether to compress the saved graph (default: False)"
    )

    parser.add_argument(
        "--extraMetrics",
        nargs="+",
        default=[],
        help="Extra metrics to add to the edge metadata"
    )

    parser.add_argument(
        "--drivingEnabled",
        action="store_true",
        default=True,
        help="Whether to connect hubs with driving edges (default: True)"
    )
    path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--Dir",
        type=str,
        default=os.path.join(path, "..", "..", "..", "data"),
        help="Directory to save the graph in (default: .)"
    )

    args = parser.parse_args()

    if len(args.data) % 3 != 0:
        parser.error("Arguments must be in groups of 3: hubType transportMode dataPath")

    transportModes = {}
    dataPaths = {}

    for i in range(0, len(args.data), 3):
        key, val1, val2 = args.data[i], args.data[i+1], args.data[i+2]
        transportModes[key] = val1
        dataPaths[key] = val2

    graph = RouteGraph(
        maxDistance=args.maxDist,
        transportModes=transportModes,
        dataPaths=dataPaths,
        compressed=args.compressed,
        extraMetricsKeys=args.extraMetrics,
        drivingEnabled=args.drivingEnabled  
    )

    graph.build()
    graph.save(filepath=args.Dir, compressed=args.compressed)

    print("Graph built and saved.")

if __name__ == "__main__":
    main()