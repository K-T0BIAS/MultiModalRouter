import unittest
from unittest.mock import patch
from multimodalrouter import RouteGraph, Hub, Filter, EdgeMetadata
import os
import tempfile
import io
import contextlib
import pandas as pd
from types import MethodType


class TestRouteGraphPublicFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_file_path = os.path.join(cls.temp_dir.name, "testDataset.csv")

        testDf = pd.DataFrame(
            columns=['source', 'destination', 'distance', 'source_lat', 'source_lng', 'destination_lat', 'destination_lng'],
            data=[('A', 'B', 2, 1, 1, 1, 3),
                  ('C', 'D', 1, 2, 1, 1, 4),
                  ('B', 'D', 1, 3, 1, 1, 4)]
        )

        testDf.to_csv(cls.temp_file_path, index=False)

    @classmethod
    def tearDownClass(cls):
        # remove temp file
        cls.temp_dir.cleanup()

    def setUp(self):
        # make init use the mock lock
        patcher = patch('multimodalrouter.graph.graph.Lock')
        self.addCleanup(patcher.stop)  # ensure patch is removed after test
        mock_lock_class = patcher.start()

        self.graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'fly'},
            dataPaths={'H': 'Hs.csv'},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=True
        )

        self.mock_lock = mock_lock_class.return_value

    def test_getHUb_from_empty_graph(self):
        hub = self.graph.getHub('H', 1)
        self.assertIsNone(hub)

    def test_getHUb_from_non_empty_graph(self):
        hub = Hub(id=1, hubType='H', coords=[1, 1])
        self.graph.addHub(hub)

        retrievedHub = self.graph.getHub('H', 1)
        self.assertEqual(retrievedHub, hub)

    def test_addHub_with_no_duplicates(self):
        hub = Hub(id=1, hubType='H', coords=[1, 1])

        self.graph.addHub(hub)
        # check that the lock was used to access the graph
        self.mock_lock.__enter__.assert_called_once()
        self.mock_lock.__exit__.assert_called_once()

        retrievedHub = self.graph.getHub('H', 1)
        self.assertEqual(retrievedHub, hub)

    def test_addHub_with_duplicates(self):
        hub = Hub(id=1, hubType='H', coords=[1, 1])

        self.graph.addHub(hub)
        self.graph.addHub(hub) # try adding a duplicate

        retrievedHub = self.graph.getHub('H', 1)
        self.assertEqual(retrievedHub, hub) # check if hub is correct
        self.assertEqual(len(self.graph.Graph['H']), 1) # check if duplicates exist

    def test_getHubById(self):
        hub = Hub(id=1, hubType='H', coords=[1, 1])
        self.graph.addHub(hub)

        retrievedHub = self.graph.getHubById(1)
        self.assertEqual(retrievedHub, hub)

    def test_getHubById_not_found(self):
        retrievedHub = self.graph.getHubById(1)
        self.assertIsNone(retrievedHub)

    def test_build_without_driving(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )
        # remove the print output from build
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        self.assertEqual(len(graph.Graph['H']), 4) # check if all hubs were added (A B C D)

        for hub in graph.Graph['H'].values():
            self.assertEqual(hub.hubType, 'H')
            if hub.outgoing:
                self.assertFalse('car' in hub.outgoing)
                self.assertTrue('mv' in hub.outgoing)

    def test_build_with_driving(self):
        import types
        import numpy as np

        def distCalc(self, startHubs: list[Hub], emdHubs: list[Hub]):
            coords1 = np.array([h.coords for h in startHubs])
            coords2 = np.array([h.coords for h in emdHubs])

            diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)  # euclidean distance along last axis

            return distances

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=True
        )
        # swap out distance calculation for one that fits the grid coords from this test
        graph._hubToHubDistances = types.MethodType(distCalc, graph)

        # remove the print output from build
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        self.assertEqual(len(graph.Graph['H']), 4) # check if all hubs were added (A B C D)
        for hub in graph.Graph['H'].values():
            self.assertEqual(hub.hubType, 'H')
            if hub.id == 'D':
                self.assertTrue('car' in hub.outgoing)
                self.assertFalse('mv' in hub.outgoing)
            else:
                self.assertTrue('car' in hub.outgoing)
                self.assertTrue('mv' in hub.outgoing)

    def test_find_shortest_path_valid_route_non_verbose(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        route = graph.find_shortest_path('A', 'D', allowed_modes=['mv'])
        self.assertIsNotNone(route)
        path = route.path
        starts = [p[0] for p in path]
        modes = [p[1] for p in path]
        self.assertEqual(starts, ['A', 'B', 'D'])
        self.assertEqual(modes, ['', 'mv', 'mv'])

    def test_find_shortest_path_valid_route_verbose(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        route = graph.find_shortest_path('A', 'D', allowed_modes=['mv'], verbose=True)
        self.assertIsNotNone(route)
        path = route.path
        starts = [p[0] for p in path]
        modes = [p[1] for p in path]
        data = [p[2] for p in path]
        self.assertEqual(starts, ['A', 'B', 'D'])
        self.assertEqual(modes, ['', 'mv', 'mv'])
        for d in data:
            self.assertIsNotNone(d)

    def test_find_shortest_path_valid_route_as_graph(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        route = graph.find_shortest_path('A', 'D', allowed_modes=['mv'], verbose=True)
        self.assertIsNotNone(route)

        route_as_graph = route.asGraph(graph)
        HubA = route_as_graph.getHubById('A')
        HubB = route_as_graph.getHubById('B')
        HubD = route_as_graph.getHubById('D')

        self.assertIsNotNone(HubA)
        self.assertIsNotNone(HubB)
        self.assertIsNotNone(HubD)

        self.assertEqual(HubA.hubType, 'H')
        self.assertEqual(HubB.hubType, 'H')
        self.assertEqual(HubD.hubType, 'H')

        self.assertTrue('B' in HubA.outgoing['mv'].keys())
        self.assertTrue(len(HubA.outgoing['mv'].keys()), 1)

        self.assertTrue('D' in HubB.outgoing['mv'].keys())
        self.assertTrue(len(HubB.outgoing['mv'].keys()), 1)

        self.assertTrue(len(HubD.outgoing.keys()) == 0)

    def test_find_shortest_path_valid_graph_to_route_with_missing_hub(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        route = graph.find_shortest_path('A', 'D', allowed_modes=['mv'])

        def dist(self, hub1, hub2):
            import numpy as np

            p1 = np.array([h.coords for h in hub1], dtype=float)
            p2 = np.array([h.coords for h in hub2], dtype=float)

            diff = p1[:, None, :] - p2[None, :, :]

            distances = np.linalg.norm(diff, axis=2)

            return distances

        graph._hubToHubDistances = MethodType(dist, graph)

        self.assertIsNotNone(route)
        # collect original distances
        HubA = graph.Graph['H']['A']
        HubB = graph.Graph['H']['B']

        dAB = HubA.outgoing['mv']['B'].getMetric('distance')
        dBD = HubB.outgoing['mv']['D'].getMetric('distance')

        # drop HubB
        graph.Graph['H'].pop('B')

        route_as_graph = route.asGraph(graph)

        new_HubA = route_as_graph.getHubById('A')
        new_HubD = route_as_graph.getHubById('D')

        new_HubB = route_as_graph.getHubById('B')

        self.assertIsNone(new_HubB)

        self.assertIsNotNone(new_HubA)
        self.assertIsNotNone(new_HubD)

        self.assertEqual(new_HubA.hubType, 'H')
        self.assertEqual(new_HubD.hubType, 'H')

        self.assertTrue('D' in new_HubA.outgoing['mv'].keys())
        self.assertTrue(len(new_HubA.outgoing['mv'].keys()), 1)

        self.assertTrue(len(new_HubD.outgoing.keys()) == 0)
        # sum is allowed since A, B, D share the same lng coord (and live in 2d space)
        self.assertAlmostEqual(dAB + dBD, new_HubA.outgoing['mv']['D'].getMetric('distance'), places=5)

    def test_find_shortest_path_invalid_path(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        route = graph.find_shortest_path('D', 'A', allowed_modes=['mv'])
        self.assertIsNone(route)

    def test_shortest_path_with_custom_filter(self):
        testDf = pd.DataFrame(
            columns=['source', 'destination', 'distance', 'source_lat', 'source_lng', 'destination_lat', 'destination_lng'],
            data=[
                ('A', 'B', 2, 1, 1, 1, 3),
                ('A', 'C', 3, 1, 1, 2, 4),
                ('B', 'D', 1, 1, 3, 1, 4),
                ('B', 'E', 4, 1, 3, 2, 6),
                ('C', 'D', 2, 2, 4, 1, 4),
                ('C', 'F', 5, 2, 4, 3, 7),
                ('D', 'G', 1, 1, 4, 2, 5),
                ('E', 'G', 2, 2, 6, 2, 5),
                ('F', 'H', 3, 3, 7, 3, 9),
                ('G', 'H', 2, 2, 5, 3, 9),
                ('H', 'I', 1, 3, 9, 4, 10),
                ('E', 'I', 5, 2, 6, 4, 10),
            ]
        )

        temp_path = os.path.join(self.temp_dir.name, 'temp.csv')
        testDf.to_csv(temp_path, index=False)

        class CF(Filter):

            def __init__(self, forbiddenHubs: list[str]):
                self.forbiddenHubs = forbiddenHubs

            def filterHub(self, hub: Hub) -> bool:
                return hub.id not in self.forbiddenHubs

            def filterEdge(self, edge: EdgeMetadata) -> bool:
                return True

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': temp_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        route = graph.find_shortest_path('A', 'D', allowed_modes=['mv'], custom_filter=CF(['B']))
        self.assertIsNotNone(route)
        path = route.path
        starts = [p[0] for p in path]
        modes = [p[1] for p in path]
        self.assertEqual(starts, ['A', 'C', 'D'])
        self.assertEqual(modes, ['', 'mv', 'mv'])

    def test_save_load_compressed(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=True,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        graph.save(self.temp_dir.name, compressed=True)

        loaded = RouteGraph.load(os.path.join(self.temp_dir.name, 'graph.zlib'), compressed=True)
        self.assertEqual(graph.Graph.keys(), loaded.Graph.keys())
        self.assertEqual(graph.Graph['H'].keys(), loaded.Graph['H'].keys())
        for oldHub, loadedHub in zip(graph.Graph['H'].values(), loaded.Graph['H'].values()):
            self.assertEqual(oldHub.id, loadedHub.id)

    def test_save_load_not_compressed(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        graph.save(self.temp_dir.name, compressed=False)

        loaded = RouteGraph.load(os.path.join(self.temp_dir.name, 'graph.dill'), compressed=False)
        self.assertEqual(graph.Graph.keys(), loaded.Graph.keys())
        self.assertEqual(graph.Graph['H'].keys(), loaded.Graph['H'].keys())
        for oldHub, loadedHub in zip(graph.Graph['H'].values(), loaded.Graph['H'].values()):
            self.assertEqual(oldHub.id, loadedHub.id)

    def test_radial_search(self):
        testDf = pd.DataFrame(
            columns=['source', 'destination', 'distance', 'source_lat', 'source_lng', 'destination_lat', 'destination_lng'],
            data=[
                ('A', 'B', 2, 1, 1, 1, 3),
                ('A', 'C', 3, 1, 1, 2, 4),
                ('B', 'D', 1, 1, 3, 1, 4),
                ('B', 'E', 4, 1, 3, 2, 6),
                ('C', 'D', 2, 2, 4, 1, 4),
                ('C', 'F', 5, 2, 4, 3, 7),
                ('D', 'G', 1, 1, 4, 2, 5),
                ('E', 'G', 2, 2, 6, 2, 5),
                ('F', 'H', 3, 3, 7, 3, 9),
                ('G', 'H', 2, 2, 5, 3, 9),
                ('H', 'I', 1, 3, 9, 4, 10),
                ('E', 'I', 5, 2, 6, 4, 10),
            ]
        )

        temp_path = os.path.join(self.temp_dir.name, 'temp.csv')
        testDf.to_csv(temp_path, index=False)

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': temp_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        reachable = graph.radial_search('A', 5)
        self.assertEqual(len(reachable), 5)
        for dist, _ in reachable:
            self.assertLessEqual(dist, 5)

        reachableIds = [hub.id for _, hub in reachable]
        self.assertIn('B', reachableIds)
        self.assertIn('C', reachableIds)
        self.assertIn('D', reachableIds)
        self.assertIn('A', reachableIds)
        self.assertIn('G', reachableIds)

    def test_radial_search_with_filter(self):

        class CF(Filter):

            def __init__(self, forbiddenHubs: list[str]):
                self.forbiddenHubs = forbiddenHubs

            def filterHub(self, hub: Hub) -> bool:
                return hub.id not in self.forbiddenHubs

            def filterEdge(self, edge: EdgeMetadata) -> bool:
                return edge.getMetric('distance') < 3

        testDf = pd.DataFrame(
            columns=['source', 'destination', 'distance', 'source_lat', 'source_lng', 'destination_lat', 'destination_lng'],
            data=[
                ('A', 'B', 2, 1, 1, 1, 3),
                ('A', 'C', 3, 1, 1, 2, 4),
                ('B', 'D', 1, 1, 3, 1, 4),
                ('B', 'E', 4, 1, 3, 2, 6),
                ('C', 'D', 2, 2, 4, 1, 4),
                ('C', 'F', 5, 2, 4, 3, 7),
                ('D', 'G', 1, 1, 4, 2, 5),
                ('E', 'G', 2, 2, 6, 2, 5),
                ('F', 'H', 3, 3, 7, 3, 9),
                ('G', 'H', 2, 2, 5, 3, 9),
                ('H', 'I', 1, 3, 9, 4, 10),
                ('E', 'I', 5, 2, 6, 4, 10),
            ]
        )

        temp_path = os.path.join(self.temp_dir.name, 'temp.csv')
        testDf.to_csv(temp_path, index=False)

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': temp_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        # radial search that excludes hub 'D' and edges where distance > 3
        reachable = graph.radial_search('A', 5, custom_filter=CF(['D']))
        self.assertEqual(len(reachable), 2)
        for dist, _ in reachable:
            self.assertLessEqual(dist, 5)

        reachableIds = [hub.id for _, hub in reachable]
        self.assertIn('B', reachableIds)
        self.assertIn('A', reachableIds)

    def test_shortest_path_with_path_aware_filter(self):
        """Test that filter can access and use the current path to limit consecutive segments."""
        testDf = pd.DataFrame(
            columns=['source', 'destination', 'distance', 'source_lat', 'source_lng', 'destination_lat', 'destination_lng'],
            data=[
                ('A', 'B', 1, 1, 1, 1, 2),
                ('B', 'C', 1, 1, 2, 1, 3),
                ('C', 'D', 1, 1, 3, 1, 4),
                ('D', 'E', 1, 1, 4, 1, 5),
                ('E', 'F', 1, 1, 5, 1, 6),
                ('A', 'F', 10, 1, 1, 1, 6),  # Direct route is longer
            ]
        )

        temp_path = os.path.join(self.temp_dir.name, 'temp_path_aware.csv')
        testDf.to_csv(temp_path, index=False)

        class PathAwareFilter(Filter):
            """Filter that limits consecutive segments with the same transport mode."""

            def __init__(self, max_consecutive_segments: int = 2):
                self.max_consecutive_segments = max_consecutive_segments

            def filterHub(self, hub: Hub) -> bool:
                return True

            def filterEdge(self, edge: EdgeMetadata) -> bool:
                return True

            def filter(self, start: Hub, end: Hub, edge: EdgeMetadata, current_path: list = None) -> bool:
                if current_path is None or len(current_path) == 0:
                    return True

                mode = edge.transportMode

                # count consecutive segments with the same mode
                consecutive_count = 0
                for i in range(len(current_path) - 1, -1, -1):
                    path_mode = current_path[i][1] if len(current_path[i]) > 1 else ""
                    if path_mode == mode:
                        consecutive_count += 1
                    else:
                        break

                # if more than max consecutive segments, return False
                if consecutive_count >= self.max_consecutive_segments:
                    return False

                return True

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': temp_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        # search route that only ever has two consecutive segments of one mode
        route = graph.find_shortest_path(
            'A',
            'F',
            allowed_modes=['mv'],
            custom_filter=PathAwareFilter(max_consecutive_segments=2)
        )
        self.assertIsNotNone(route)
        path = route.path
        starts = [p[0] for p in path]

        # should take direct route A -> F (10 distance) instead of A -> B -> C (would be 3 segments)
        self.assertEqual(starts, ['A', 'F'])
        self.assertAlmostEqual(route.totalMetrics.getMetric('distance'), 10, places=5)

        # without filter, should take the shorter multi-hop route
        route_no_filter = graph.find_shortest_path('A', 'F', allowed_modes=['mv'])
        self.assertIsNotNone(route_no_filter)
        path_no_filter = route_no_filter.path
        starts_no_filter = [p[0] for p in path_no_filter]

        # should be A -> B -> C -> D -> E -> F (5 distance total)
        self.assertEqual(starts_no_filter, ['A', 'B', 'C', 'D', 'E', 'F'])
        self.assertAlmostEqual(route_no_filter.totalMetrics.getMetric('distance'), 5, places=5)

    def test_shortest_path_with_path_aware_filter_verbose(self):
        """Test path-aware filter works with verbose mode."""
        testDf = pd.DataFrame(
            columns=['source', 'destination', 'distance', 'source_lat', 'source_lng', 'destination_lat', 'destination_lng'],
            data=[
                ('A', 'B', 1, 1, 1, 1, 2),
                ('B', 'C', 1, 1, 2, 1, 3),
                ('C', 'D', 1, 1, 3, 1, 4),
                ('A', 'D', 5, 1, 1, 1, 4),
            ]
        )

        temp_path = os.path.join(self.temp_dir.name, 'temp_path_aware_verbose.csv')
        testDf.to_csv(temp_path, index=False)

        class PathAwareFilter(Filter):
            def __init__(self, max_consecutive: int = 1):
                self.max_consecutive = max_consecutive

            def filterHub(self, hub: Hub) -> bool:
                return True

            def filterEdge(self, edge: EdgeMetadata) -> bool:
                return True

            def filter(self, start: Hub, end: Hub, edge: EdgeMetadata, current_path: list = None) -> bool:
                if current_path is None or len(current_path) == 0:
                    return True

                mode = edge.transportMode
                consecutive = 0
                for i in range(len(current_path) - 1, -1, -1):
                    # Handle both verbose and non-verbose path formats
                    path_mode = current_path[i][1] if len(current_path[i]) > 1 else ""
                    if path_mode == mode:
                        consecutive += 1
                    else:
                        break

                return consecutive < self.max_consecutive

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': temp_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        route = graph.find_shortest_path(
            'A',
            'D',
            allowed_modes=['mv'],
            verbose=True,
            custom_filter=PathAwareFilter(max_consecutive=1)
        )
        self.assertIsNotNone(route)
        path = route.path
        starts = [p[0] for p in path]

        # max 1 consecutive, forced to take direct route
        self.assertEqual(starts, ['A', 'D'])
        self.assertAlmostEqual(route.totalMetrics.getMetric('distance'), 5, places=5)
