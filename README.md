# Multi Modal Router

The Multi Modal Router is a graph-based routing engine that allows you to build and query any hub-based network. It supports multiple transport modes like driving, flying, or shipping, and lets you optimize routes by distance, time, or custom metrics.

> NOTE: This project is a work in progress and features might be added and or changed

# Features

## Building Freedom / Unlimited Usecases

The graph can be build from any data aslong as the required fields are present. Whether your data contains real worl places or you are working in a more abstract spaces with special coordinates and distance metrics the graph will behave the same (with minor limitations due to dynamic distance calculation, but not a problem when distances are already precomputed).

#### Example Usecases

- real world flight router
    - uses data with real flight data and actuall airport coordinates
    - builds a graph with `airport` [Hubs](./docs/graph.md#hub)
    - connects `airports` based on flight routes
    - `finds` the `shortest flights` or `multi leg routes` to get from `A` to `B`

- social relation ship graph
    - uses user data like a social network where users are connected through others via a group of other users
    - builds a graph with `users` as Hubs
    - connects users based on know interactions or any other connection meric
    - `finds` users that are likely to `share`; `interests`, `friends`, `a social circle`, etc.

- coordinate based game AI and pathfinding
    - uses a predefined path network (e.g. a simple maze)
    - `builds` the garph representation of the network
    - `finds` the shortest way to get from any point `A` to any other point `B` in the network

## Important considerations for your usecase

Depending on your usecase and datasets some faetures may not be usable (NOTE: routing and building will always work)

### potential problems based on use case

**Please check your data for the following**

| distance present | coordinate format | unusable features | special considerations |
|------------------|-------------------|-------------------|------------------------|
|      YES         |      degrees      |      None         |        None|
|      YES|not degrees| runtime distance calculations| [drivingEnabled = False](./docs/graph.md#args)|
| NO | degrees | None | distances must be calculated when [preprocessing](./src/multimodalrouter/utils/preprocessor.py) |


# Documentation

[installation guide](./docs/installation.md)

[graph module documentation](./docs/graph.md)

[code examples](./docs/examples/demo.py)

[command line interface documentation](./docs/cli.md)

[utilities documentation](./docs/utils.md)

### License

[see here](./LICENSE.md)


