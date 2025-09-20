import unittest
from unittest.mock import MagicMock
from multimodalrouter.graphics import GraphDisplay


class MockHub:
    def __init__(self, hubType, id, coords):
        self.hubType = hubType
        self.id = id
        self.coords = coords
        self.outgoing = {}


class MockRouteGraph:
    def __init__(self, hubs):
        self._hubs = hubs

    def _allHubs(self):
        return self._hubs

    def getHubById(self, hub_id):
        for hub in self._hubs:
            if hub.id == hub_id:
                return hub
        return None

class TestGraphDisplay(unittest.TestCase):

    def setUp(self):
        # create mock hubs
        self.hubs = [
            MockHub("cell", "0", [0, 0]),
            MockHub("cell", "1", [1, 1])
        ]
        # add an edge from 0 to 1
        edge = MagicMock()
        edge.allMetrics = {"distance": 1}
        self.hubs[0].outgoing = {"1": {"1": edge}}
        self.graph = MockRouteGraph(self.hubs)

        self.display = GraphDisplay(self.graph)

    def test_node_transform(self):
        # transform: add 1 to all coordinates
        def nodeTransform(coords):
            return [[x+1, y+1] for x, y, *rest in coords]

        self.display._toPlotlyFormat(nodeTransform=nodeTransform)
        self.assertEqual(self.display.nodes["cell-0"]["coords"][:2], [1, 1])
        self.assertEqual(self.display.nodes["cell-1"]["coords"][:2], [2, 2])

    def test_edge_transform(self):
        # transform edges into straight lines
        def edgeTransform(starts, ends):
            return [[[s[0], s[1]], [e[0], e[1]]] for s, e in zip(starts, ends)]

        self.display._toPlotlyFormat(edgeTransform=edgeTransform)
        self.assertIn("curve", self.display.edges[0])
        self.assertEqual(self.display.edges[0]["curve"][0], [0, 0])
        self.assertEqual(self.display.edges[0]["curve"][-1], [1, 1])

    def test_degreesToCartesian3D(self):
        coords = [[0, 0], [90, 0], [0, 90]]
        result = GraphDisplay.degreesToCartesian3D(coords)
        self.assertEqual(len(result), 3)
        # first point should be on x-axis
        self.assertAlmostEqual(result[0][0], 6371.0, places=0)
        # second point should be at north pole
        self.assertAlmostEqual(result[1][2], 6371.0, places=0)

    def test_curvedEdges(self):
        start = [[1,0,0]]
        end = [[0,1,0]]
        curve = GraphDisplay.curvedEdges(start, end, R=1.0, H=0.0, n=5)
        self.assertEqual(len(curve), 1)
        self.assertEqual(len(curve[0]), 5)  # n points
        # first and last points match start/end
        self.assertAlmostEqual(curve[0][0][0], 1.0, places=5)
        self.assertAlmostEqual(curve[0][-1][1], 1.0, places=5)

if __name__ == "__main__":
    unittest.main()
