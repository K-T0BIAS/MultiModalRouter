# Multi Modal Router

The Multi Modal Router is a graph-based routing engine that allows you to build and query any hub-based network. It supports multiple transport modes like driving, flying, or shipping, and lets you optimize routes by distance, time, or custom metrics.

> NOTE: This project is a work in progress and features might be added and or changed

---

## Hubs

Hubs are the nodes of the graph. Possible hubs include airports, ports, or train stations.

### Hub Structure

Each Hub includes:

* lat: float — latitude of the hub
* lng: float — longitude of the hub
* id: str — unique identifier (e.g., IATA codes for airports or UN/LOCODEs for ports)
* hubType: str — type of hub (e.g., "airport", "port")
* outgoing: dict[str, dict[str, EdgeMetadata]] — a dictionary of transport modes mapping to destinations and their edge metadata

Example:

outgoing = {
    "drive": {
        "SFO": EdgeMetadata(distance=50, time=60),
        "LAX": EdgeMetadata(distance=350, time=360),
    },
    "fly": {
        "YEG": EdgeMetadata(distance=1500, time=180),
    }
}

---

## EdgeMetadata

EdgeMetadata objects hold metrics related to one edge from Hub1 to Hub2.

### Attributes:

* transportMode: str — mode of transport across this edge (e.g., "drive", "fly")
* metrics: dict — dynamically created dictionary of metrics (always includes distance, may include others like time)

Example:

{
    "distance": 123.45,
    "time": 34.15
}

---

## Setup

### Create a virtual environment
```txt
python -m venv venv
```

### Activate the virtual environment

Windows:

```txt
venv\Scripts\activate
```

Linux / MacOS:
```txt
source venv/bin/activate
```

### Install dependencies
```txt
pip install -r requirements.txt
```
---

## Building a Graph

### Step 1: Data Preparation

1. Obtain datasets with the following information:
    - Source locations
    - Coordinates of source locations
    - Target locations
    - Coordinates of target locations
2. Use the utils.preprocessor to process your data
3. Repeat for all datasets you want to include

---

### Step 2: Build the Graph

Run the build script:

```txt
python -m router.build hubType1 transportMode1 pathToData1 hubType2 transportMode2 pathToData2 ... --maxDist float --compressed
```

Parameters:

* hubTypeX — type of the hub in dataset X (e.g., "airport")
* transportModeX — mode of transport in dataset X (e.g., "drive", "fly")
* pathToDataX — path to the dataset file
* --maxDist — maximum distance to connect hubs with driving edges (default 50)
* --compressed — optional flag to compress the saved graph

Example:

python -m router.build airport fly data/fullDataset.parquet port drive data/ports.csv --maxDist 100 --compressed

---

## Using the Graph

Run the routing script:

python -m router.router --start lat1 lng1 --end lat2 lng2> --allowedModes mode1 mode2 ...

Parameters:

* --start — latitude and longitude of the starting location
* --end — latitude and longitude of the destination
* --allowedModes — list of transport modes allowed for routing (e.g., fly, drive)

Example:

coming soon

---

## Notes & Tips

* Ensure your datasets include the required fields
* All hubs must have a unique id.
* When adding multiple datasets, make sure the hub types and transport modes are correctly matched to the data.
* You can pass multiple allowed modes in --allowedModes for flexible routing.
* The graph can be compressed using the --compressed flag to save space.


