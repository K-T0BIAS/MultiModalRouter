# Multi Modal Router

The Multi Modal Router is a graph-based routing engine that allows you to build and query any hub-based network. It supports multiple transport modes like driving, flying, or shipping, and lets you optimize routes by distance, time, or custom metrics. It can be expanded to any n-dimensional space making it versatile in any coordinate space

> NOTE: This project is a work in progress and features might be added and or changed

# In depth Documentation

[installation guide](./docs/installation.md)

[graph module documentation](./docs/graph.md)

[code examples](./docs/examples/demo.py)

[command line interface documentation](./docs/cli.md)

[utilities documentation](./docs/utils.md)


# Features

## Building Freedom / Unlimited Usecases

The graph can be build from any data aslong as the required fields are present ([example](./docs/examples/demoData.csv)). Whether your data contains real world places or you are working in a more abstract spaces with special coordinates and distance metrics the graph will behave the same (with minor limitations due to dynamic distance calculation, but not a problem when distances are already precomputed. [solutions](./docs/graph.md#advanced-options)).

#### Example Usecases

- real world flight router
    - uses data with real flight data and actuall airport coordinates
    - builds a graph with `airport` [Hubs](./docs/graph.md#hub)
    - connects `airports` based on flight routes
    - `finds` the `shortest flights` or `multi leg routes` to get from `A` to `B`
    - simple example implementation [here](./docs/examples/flightRouter/main.py)

- social relation ship graph
    - uses user data like a social network where users are connected through others via a group of other users
    - builds a graph with `users` as Hubs
    - connects users based on know interactions or any other connection meric
    - `finds` users that are likely to `share`; `interests`, `friends`, `a social circle`, etc.

- coordinate based game AI and pathfinding
    - uses a predefined path network (e.g. a simple maze)
    - `builds` the garph representation of the network
    - `finds` the shortest way to get from any point `A` to any other point `B` in the network
    - you can checkout a simple example implementation for a maze pathfinder [here](./docs/examples/mazePathfinder/main.py)

![example from the maze solver](./docs/solvedMaze1.png)

## graph visualizations

Use the build-in visualization tool to plot any `2D` or `3D` Graph.

![example plot of flight paths](./docs/FlightPathPlot.png)

## Important considerations for your usecase

Depending on your usecase and datasets some features may not be usable see solutions below

### potential problems based on use case

**Please check your data for the following**

| distance present | coordinate format | unusable features | special considerations |
|------------------|-------------------|-------------------|------------------------|
|      YES         |      degrees      |      None         |        None|
|      YES|not degrees| runtime distance calculations| set [drivingEnabled = False](./docs/graph.md#args) or do [this](./docs/graph.md#swap-distance-method)|
| NO | degrees | None | distances must be calculated when [preprocessing](./src/multimodalrouter/utils/preprocessor.py) |
| NO | not degrees | **ALL** | **U cant build the graph with neither distances or supported coordinates!** [**solution**](./docs/graph.md#swap-distance-method)

[**example dataframe with the required fields**](./docs/examples/demoData.csv)

### License

[see here](./LICENSE.md)


