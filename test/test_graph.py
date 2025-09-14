# test_graph.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


import unittest
import os
from graph.graph import RouteGraph
from graph.dataclasses import Hub

class TestRouteGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, "..", "data", "graph.dill")
        cls.graph_file = path 
        cls.graph = RouteGraph.load(cls.graph_file, compressed=False)

    def test_graph_loaded(self):
        # Ensure graph is loaded
        self.assertIsInstance(self.graph, RouteGraph)
        self.assertTrue(len(self.graph.Graph) > 0)

    def test_hubs_exist(self):
        # Test that hubs exist in the graph
        all_hubs = list(self.graph._allHubs())
        self.assertGreater(len(all_hubs), 0)
        # Check one hub has valid attributes
        hub = all_hubs[0]
        self.assertIsInstance(hub, Hub)
        self.assertIsInstance(hub.lat, float)
        self.assertIsInstance(hub.lng, float)

    def test_edges_exist(self):
        # Check that at least one hub has outgoing connections
        all_hubs = list(self.graph._allHubs())
        has_edges = any(hub.outgoing for hub in all_hubs)
        self.assertTrue(has_edges)

    def test_find_shortest_path(self):
        # Pick a hub that has outgoing connections
        all_hubs = list(self.graph._allHubs())
        for hub in all_hubs:
            if hub.outgoing:
                start_id = hub.id
                mode = next(iter(hub.outgoing.keys()))
                end_hub_id = next(iter(hub.outgoing[mode]))
                break
        else:
            self.skipTest("No hubs with outgoing connections found")

        # Debug prints (optional)
        print(f"Testing path from {start_id} to {end_hub_id} via mode '{mode}'")

        # Call find_shortest_path using the correct allowed mode
        route = self.graph.find_shortest_path(start_id, end_hub_id, allowed_modes=[mode])

        # Assertions
        self.assertIsNotNone(route, "No route found between connected hubs")
        self.assertEqual(route.path[0][0], start_id, "Route does not start at the expected hub")
        self.assertEqual(route.path[-1][0], end_hub_id, "Route does not end at the expected hub")

        # Optional: check that the route contains at least one edge
        self.assertGreater(len(route.path), 1, "Route should contain at least one segment")




    def test_compare_routes(self):
        hubs = list(self.graph._allHubs())
        if len(hubs) < 2:
            self.skipTest("Not enough hubs to test compare_routes")
        start_id = hubs[0].id
        end_id = hubs[1].id

        results = self.graph.compare_routes(start_id, end_id, allowed_modes=["fly"])
        self.assertIsInstance(results, dict)
        self.assertTrue(all(isinstance(r, type(self.graph.find_shortest_path(start_id, end_id, allowed_modes=["fly"]))) 
                            for r in results.values()))

if __name__ == "__main__":
    unittest.main()

