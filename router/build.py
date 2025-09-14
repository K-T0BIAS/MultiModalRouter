from graph.graph import RouteGraph
import argparse

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
        saveMode="light", # unused
        compressed=args.compressed,
        extraMetricsKeys=args.extraMetrics
    )

    graph.build()
    graph.save("graph", saveMode="light", compressed=args.compressed)

    print("Graph built and saved.")

if __name__ == "__main__":
    main()