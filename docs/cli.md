[HOME](../README.md)

# command line interface

If you don't want to always write new scripts to build the graph or to find routes, you can simply use the command line interface.

## building a graph from the terminal

**this assumes that your data has been preprocessed into the correct format**

> from your terminal run:

```
multimodalrouter-build hubType1 transportMode1 pathToDataset1 --maxDist 123 --extraMetrics metric1 metric2 --drivingEnabled --Dir pathToSomeDir
```
### args

- `hub type`,`transport mode`,`dataset path` = the first arguments in your cli should be these three. They must follow this order to be parsed correctly
    - `hub type` = the name the hubs from this dataset will get
    - `transport mode` = the mode of transport unique to this hub
    - `dataset path` = ideally the absolute path to the dataset that contains this `hub type`
    - NOTE: you can add as many hub types and datasets at once as you want just make sure the order is always; type1 mode1 path1 type2 ...
- `--maxDist` = the float value that limits the maximum length a driving edge can have (this is irrelevant id `--enableDriving` isn't set)
- `extraMetrics` = a list of keys that the graph will scan your data for and add the values to any edge where the key exists in the data (NOTE: each key should be present in at least one dataset, the number of extra metrics is not limited)
- `--drivingEnabled` = if this flag is added the graph will try to build connections between hubs for any hub that is closer than the `--maxDist` to anoter hub (check [here](../README.md#important-considerations-for-your-usecase) to see if your data allows this)
- `--Dir` = put the target directory of the save file here. (if this is not set the graph will be saved to a default dir)

### example

> create a new dir for the garph and edit the `--Dir` if you want the graph in a easy to access directory otherwise keep as is

> NOTE: this example assumes you are still in the project dir. If not please adapt the dataset path accordingly

```text
multimodalrouter-build airport plane ./docs/examples/flightRouter/data/fullDataset.csv --maxDist 100 --drivingEnabled 
--extraMetrics source_name destination_name
```

**output:**

```text
Building graph...
Generating airport Hubs: 100hub [00:00, 186995.27hub/s]
Graph built and saved.
```

## finding a route

```text
multimodalrouter-route --start lat1 lng1 --end lat2 lng2 --allowedModes car plane --maxSegments 123 --verbose
```

> NOTE: if you have higher dimensional coordinates you can simply add them here aswell

### agrs

- `--start` = the start coordinates for the route (doesn't have to be exact to the `Hub` it will find the closest hub to the point. Also the coordinates do not have to be in dgrees, but rather fit with the coordinate system of your data. Read more [here](./graph.md#advanced-options))
- `--end` = the end coordinates of the route: (same features as for `--start` apply)
- `--allowedModes` = a filter that discards edges that use a transport mode that is not in this list. (default= ['car'])
- `--maxSegments` = an interger value of the maximum number of hubs a route can have. (e.g.: to avoid very deep searches when long routes are not logically feasable) (default = 10)
- `--verbose` = if this flag is set the output route will contain the [edgeMetadata](./graph.md#edgemetadata) for every leg of the route (default=false when flag not set)

### example

> NOTE: this example assumes you build the graph from the [previous example](#example).
> if you changed the save path this script wont find it (**this will hopefully be fixed in the future**).

```text
multimodalrouter-route --start 60.866699 -162.272996 --end 60.872747 -162.5247 --allowedModes plane car --maxSegments 10 --verbose
```

**output:**

```text
Start: ATT
        Edge: (transportMode=plane, metrics={'distance': 13.641155702721523, 'source_name': 'Atmautluak Airport', 'destination_name': 'Kasigluk Airport'})
-> KUK
```