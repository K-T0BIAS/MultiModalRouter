import unittest
from unittest.mock import patch
from multimodalrouter import RouteGraph, Hub, Filter, EdgeMetadata, PathNode
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
        testDf = pd.DataFrame(
            columns=[
                'source',
                'destination',
                'distance',
                'source_lat',
                'source_lng',
                'destination_lat',
                'destination_lng',
            ],
            data=[
                ('A', 'B', 1, 1, 1, 1, 2),
                ('B', 'C', 1, 1, 2, 1, 3),
                ('C', 'D', 1, 1, 3, 1, 4),
                ('D', 'E', 1, 1, 4, 1, 5),
                ('E', 'F', 1, 1, 5, 1, 6),
                ('A', 'F', 10, 1, 1, 1, 6),
            ],
        )

        temp_path = os.path.join(self.temp_dir.name, 'temp_path_aware.csv')
        testDf.to_csv(temp_path, index=False)

        class PathAwareFilter(Filter):
            def __init__(self, max_consecutive_segments: int = 2):
                self.max_consecutive_segments = max_consecutive_segments

            def filterHub(self, hub: Hub) -> bool:
                return True

            def filterEdge(self, edge: EdgeMetadata) -> bool:
                return True

            def filter(
                self,
                start: Hub,
                end: Hub,
                edge: EdgeMetadata,
                path: PathNode | None,
            ) -> bool:
                if path is None:
                    return True

                mode = edge.transportMode
                consecutive = 0
                node = path
                while node is not None:
                    if node.mode == mode:
                        consecutive += 1
                    else:
                        break
                    node = node.prev

                return consecutive < self.max_consecutive_segments

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': temp_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False,
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        route = graph.find_shortest_path(
            'A',
            'F',
            allowed_modes=['mv'],
            custom_filter=PathAwareFilter(max_consecutive_segments=2),
        )
        self.assertIsNotNone(route)
        starts = [p[0] for p in route.path]
        self.assertEqual(starts, ['A', 'F'])
        self.assertAlmostEqual(route.totalMetrics.getMetric('distance'), 10, places=5)

        route_no_filter = graph.find_shortest_path('A', 'F', allowed_modes=['mv'])
        self.assertIsNotNone(route_no_filter)
        starts_no_filter = [p[0] for p in route_no_filter.path]
        self.assertEqual(starts_no_filter, ['A', 'B', 'C', 'D', 'E', 'F'])
        self.assertAlmostEqual(
            route_no_filter.totalMetrics.getMetric('distance'), 5, places=5
        )

    def test_shortest_path_with_path_aware_filter_verbose(self):
        testDf = pd.DataFrame(
            columns=[
                'source',
                'destination',
                'distance',
                'source_lat',
                'source_lng',
                'destination_lat',
                'destination_lng',
            ],
            data=[
                ('A', 'B', 1, 1, 1, 1, 2),
                ('B', 'C', 1, 1, 2, 1, 3),
                ('C', 'D', 1, 1, 3, 1, 4),
                ('A', 'D', 5, 1, 1, 1, 4),
            ],
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

            def filter(
                self,
                start: Hub,
                end: Hub,
                edge: EdgeMetadata,
                path: PathNode | None,
            ) -> bool:
                if path is None:
                    return True

                mode = edge.transportMode
                consecutive = 0
                node = path
                while node is not None:
                    if node.mode == mode:
                        consecutive += 1
                    else:
                        break
                    node = node.prev

                return consecutive < self.max_consecutive

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': temp_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False,
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            graph.build()

        route = graph.find_shortest_path(
            'A',
            'D',
            allowed_modes=['mv'],
            verbose=True,
            custom_filter=PathAwareFilter(max_consecutive=1),
        )
        self.assertIsNotNone(route)
        starts = [p[0] for p in route.path]
        self.assertEqual(starts, ['A', 'D'])
        self.assertAlmostEqual(route.totalMetrics.getMetric('distance'), 5, places=5)

    def test_find_shortest_paths_one_to_many(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            graph.build()

        routes = graph.find_shortest_paths(
            start_id='A',
            end_ids=['B', 'D'],
            allowed_modes=['mv'],
        )

        self.assertIsInstance(routes, dict)
        self.assertIn('B', routes)
        self.assertIn('D', routes)

        route_B = routes['B']
        route_D = routes['D']

        self.assertEqual([p[0] for p in route_B.path], ['A', 'B'])
        self.assertEqual([p[0] for p in route_D.path], ['A', 'B', 'D'])

        self.assertAlmostEqual(route_B.totalMetrics.getMetric('distance'), 2)
        self.assertAlmostEqual(route_D.totalMetrics.getMetric('distance'), 3)

    def test_fully_connect_points(self):
        testDf = pd.DataFrame(
            columns=['source', 'destination', 'distance', 'source_lat', 'source_lng', 'destination_lat', 'destination_lng'],
            data=[
                ('A', 'B', 1, 0, 0, 1, 0),
                ('B', 'C', 1, 1, 0, 2, 0),
                ('C', 'D', 1, 2, 0, 3, 0),
            ],
        )

        temp_path = os.path.join(self.temp_dir.name, 'fully_connect.csv')
        testDf.to_csv(temp_path, index=False)

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': temp_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            graph.build()

        fc = graph.fully_connect_points(
            ['A', 'B', 'C'],
            allowed_modes=['mv'],
            optimization_metric="distance",
        )

        self.assertIn('A', fc)
        self.assertIn('B', fc)
        self.assertIn('C', fc)

        self.assertIn('C', fc['A'])
        self.assertEqual([p[0] for p in fc['A']['C'].path], ['A', 'B', 'C'])

        self.assertIn('C', fc['B'])
        self.assertEqual([p[0] for p in fc['B']['C'].path], ['B', 'C'])

    def test_fully_connect_with_path_filter(self):
        class MaxDepth(Filter):
            def filterHub(self, hub):
                return True

            def filterEdge(self, edge):
                return True

            def filter(self, start, end, edge, path):
                depth = 0
                for _ in path: # use this to test __iter__
                    depth += 1
                return depth < 2

        testDf = pd.DataFrame(
            columns=['source', 'destination', 'distance', 'source_lat', 'source_lng', 'destination_lat', 'destination_lng'],
            data=[
                ('A', 'B', 1, 0, 0, 1, 0),
                ('B', 'C', 1, 1, 0, 2, 0),
            ],
        )

        temp_path = os.path.join(self.temp_dir.name, 'fc_filter.csv')
        testDf.to_csv(temp_path, index=False)

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': temp_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            graph.build()

        fc = graph.fully_connect_points(
            ['A', 'C'],
            allowed_modes=['mv'],
            custom_filter=MaxDepth(),
        )

        self.assertNotIn('C', fc['A'])

    def test_lexicographic_hops_beats_long_chain(self):
        """
        Test that minimal hops beats minimal distance
        in lexiographical order (hops, distance).
        """
        rows = []

        # A -> B -> C -> ... -> J (9 hops, distance 1 each)
        nodes = [chr(ord('A') + i) for i in range(10)]
        for i in range(len(nodes) - 1):
            rows.append((nodes[i], nodes[i + 1], 1, i, 0, i + 1, 0))

        # Direct A -> J (1 hop, distance 20)
        rows.append(('A', 'J', 20, 0, 0, 9, 0))

        testDf = pd.DataFrame(
            columns=[
                'source', 'destination', 'distance',
                'source_lat', 'source_lng',
                'destination_lat', 'destination_lng',
            ],
            data=rows,
        )

        path = os.path.join(self.temp_dir.name, 'deep_chain.csv')
        testDf.to_csv(path, index=False)

        graph = RouteGraph(
            maxDistance=100,
            transportModes={'H': 'mv'},
            dataPaths={'H': path},
            drivingEnabled=False,
        )

        graph.build()

        route = graph.find_shortest_path(
            'A',
            'J',
            allowed_modes=None, # tests default
            optimization_metric=['hops', 'distance'],
            verbose=True,
        )
        # shows that minimal hops beat minimal distance
        path_nodes = [n[0] for n in route.path]
        self.assertEqual(path_nodes, ['A', 'J'])

    def test_lexicographic_grid_prefers_straight_path(self):
        """
        Test lexicographic grid prefers least hops path
        """
        rows = []
        size = 4  # 4x4 grid

        def node(x, y):
            return f"N{x}{y}"

        for x in range(size):
            for y in range(size):
                if x + 1 < size:
                    rows.append((node(x, y), node(x + 1, y), 1, x, y, x + 1, y))
                if y + 1 < size:
                    rows.append((node(x, y), node(x, y + 1), 1, x, y, x, y + 1))

        testDf = pd.DataFrame(
            columns=[
                'source', 'destination', 'distance',
                'source_lat', 'source_lng',
                'destination_lat', 'destination_lng',
            ],
            data=rows,
        )

        path = os.path.join(self.temp_dir.name, 'grid.csv')
        testDf.to_csv(path, index=False)

        graph = RouteGraph(
            maxDistance=10,
            transportModes={'H': 'mv'},
            dataPaths={'H': path},
            drivingEnabled=False,
        )

        graph.build()

        start = node(0, 0)
        end = node(3, 3)

        route = graph.find_shortest_path(
            start,
            end,
            allowed_modes=['mv'],
            optimization_metric=['hops', 'distance'],
            verbose=True,
        )

        # Manhattan shortest hops = 6
        self.assertEqual(len(route.path) - 1, 6)

    def test_lexicographic_diamond_fanin(self):
        """
        stress test for lexicographic with many paths.
        check leats hop truth and path backtracking
        """
        rows = []

        # A -> B_i -> C
        for i in range(10):
            rows.append(('A', f'B{i}', 1, 0, 0, i, 1))
            rows.append((f'B{i}', 'C', 1, i, 1, 0, 2))

        # Direct A -> C
        rows.append(('A', 'C', 10, 0, 0, 0, 2))

        testDf = pd.DataFrame(
            columns=[
                'source', 'destination', 'distance',
                'source_lat', 'source_lng',
                'destination_lat', 'destination_lng',
            ],
            data=rows,
        )

        path = os.path.join(self.temp_dir.name, 'diamond.csv')
        testDf.to_csv(path, index=False)

        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': path},
            drivingEnabled=False,
        )

        graph.build()

        route = graph.find_shortest_path(
            'A',
            'C',
            allowed_modes=['mv'],
            optimization_metric=['hops', 'distance'],
            verbose=True,
        )

        # hops-first chooses direct edge
        self.assertEqual([n[0] for n in route.path], ['A', 'C'])

    def test_lexicographic_hops_with_depth_filter(self):
        """
        checks that filters run as expected
        (not that filters are correct since least hops will win before filter)
        """
        class MaxDepthFilter(Filter):
            def filterHub(self, hub):
                return True

            def filterEdge(self, edge):
                return True

            def filter(self, start, end, edge, path):
                # no paths longer than 3 hops
                return sum(1 for _ in path) <= 3

        rows = []

        # cheap chain A -> B -> C -> ... -> K (10 hops, distance 1)
        chain = [chr(ord('A') + i) for i in range(11)]
        for i in range(len(chain) - 1):
            rows.append((chain[i], chain[i + 1], 1, i, 0, i + 1, 0))

        # medium path A -> M -> N -> K (3 hops, distance 5)
        rows.extend([
            ('A', 'M', 2, 0, 0, 5, 1),
            ('M', 'N', 1, 5, 1, 7, 1),
            ('N', 'K', 2, 7, 1, 10, 0),
        ])

        # direct path A -> K (1 hop, distance 20)
        rows.append(('A', 'K', 20, 0, 0, 10, 0))

        testDf = pd.DataFrame(
            columns=[
                'source', 'destination', 'distance',
                'source_lat', 'source_lng',
                'destination_lat', 'destination_lng',
            ],
            data=rows,
        )

        path = os.path.join(self.temp_dir.name, 'lexi_filter_large.csv')
        testDf.to_csv(path, index=False)

        graph = RouteGraph(
            maxDistance=100,
            transportModes={'H': 'mv'},
            dataPaths={'H': path},
            drivingEnabled=False,
        )

        graph.build()

        route = graph.find_shortest_path(
            start_id='A',
            end_id='K',
            allowed_modes=['mv'],
            optimization_metric=['hops', 'distance'],
            custom_filter=MaxDepthFilter(),
            verbose=True,
        )

        # long chain is rejected by filter; direct path wins lexicographically
        path_nodes = [n[0] for n in route.path]
        self.assertEqual(path_nodes, ['A', 'K'])
