import unittest
from unittest.mock import patch
from src.multimodalrouter.graph.graph import RouteGraph
from src.multimodalrouter.graph.dataclasses import Hub
from threading import Lock
import os
import tempfile
import io
import contextlib

class TestRouteGraphInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_file_path = os.path.join(cls.temp_dir.name, "testDataset.csv")

        import pandas as pd
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
        patcher = patch('src.multimodalrouter.graph.graph.Lock')
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
        hub = Hub(id=1, hubType='H', lat=1, lng=1)
        self.graph.addHub(hub)

        retrievedHub = self.graph.getHub('H', 1)
        self.assertEqual(retrievedHub, hub)
    
    def test_addHub_with_no_duplicates(self):
        hub = Hub(id=1, hubType='H', lat=1, lng=1)

        self.graph.addHub(hub)
        # check that the lock was used to access the graph
        self.mock_lock.__enter__.assert_called_once()
        self.mock_lock.__exit__.assert_called_once()

        retrievedHub = self.graph.getHub('H', 1)
        self.assertEqual(retrievedHub, hub)
        
    def test_addHub_with_duplicates(self):
        hub = Hub(id=1, hubType='H', lat=1, lng=1)

        self.graph.addHub(hub)
        self.graph.addHub(hub) # try adding a duplicate

        retrievedHub = self.graph.getHub('H', 1)
        self.assertEqual(retrievedHub, hub) # check if hub is correct
        self.assertEqual(len(self.graph.Graph['H']), 1) # check if duplicates exist
    
    def test_getHubById(self):
        hub = Hub(id=1, hubType='H', lat=1, lng=1)
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
            coords1 = np.array([[h.lat, h.lng] for h in startHubs]) 
            coords2 = np.array([[h.lat, h.lng] for h in emdHubs]) 

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
        with contextlib.redirect_stdout(io.StringIO()):
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

        with contextlib.redirect_stdout(io.StringIO()):
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

    def test_find_shortest_path_invalid_path(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False
        )
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build()

        route = graph.find_shortest_path('D','A', allowed_modes=['mv'])
        self.assertIsNone(route)

    def test_save_load_compressed(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': self.temp_file_path},
            compressed=True,
            extraMetricsKeys=[],
            drivingEnabled=False
        )
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build()

        graph.save(self.temp_dir.name, compressed=True)

        loaded = RouteGraph.load(os.path.join(self.temp_dir.name,'graph.zlib'), compressed=True) 
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
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build()

        graph.save(self.temp_dir.name, compressed=False)

        loaded = RouteGraph.load(os.path.join(self.temp_dir.name,'graph.dill'), compressed=False)
        self.assertEqual(graph.Graph.keys(), loaded.Graph.keys())
        self.assertEqual(graph.Graph['H'].keys(), loaded.Graph['H'].keys())
        for oldHub, loadedHub in zip(graph.Graph['H'].values(), loaded.Graph['H'].values()):
            self.assertEqual(oldHub.id, loadedHub.id)
