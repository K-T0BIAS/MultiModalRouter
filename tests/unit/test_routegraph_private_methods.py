import unittest
from unittest.mock import patch
from src.multimodalrouter.graph.graph import RouteGraph
from src.multimodalrouter.graph.dataclasses import Hub
import os
import tempfile
import io
import contextlib


class TestRouteGraphPrivateMethods(unittest.TestCase):

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

        cls.mainGraph = RouteGraph(
            maxDistance=50,
            transportModes={'H': 'mv'},
            dataPaths={'H': cls.temp_file_path},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=False # set to false to speed up tests and avoid torch
        )

        # remove the print output from build
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            cls.mainGraph.build()

    @classmethod
    def tearDownClass(cls):
        # remove temp file
        cls.temp_dir.cleanup()

    def setUp(self):
        # make init use the mock lock
        patcher = patch('src.multimodalrouter.graph.graph.Lock')
        self.addCleanup(patcher.stop)  # ensure patch is removed after test
        mock_lock_class = patcher.start()
        self.mock_lock = mock_lock_class.return_value

    def test_getAllHubs(self):
        hubs = list(self.mainGraph._allHubs())
        self.assertEqual(len(hubs), 4)

    def test_addLink(self):
        hub1 = Hub([1, 1], 'A', 'H')
        hub2 = Hub([2, 2], 'B', 'H')

        testGraph = RouteGraph(maxDistance=50,
                               transportModes={'H': 'mv'},
                               dataPaths={},
                               compressed=False,
                               extraMetricsKeys=[],
                               drivingEnabled=True)

        testGraph.addHub(hub1)
        testGraph.addHub(hub2)
        self.assertEqual(len(testGraph.Graph['H']), 2)

        testGraph._addLink(hub1, hub2, 'mv', 1)

        route = testGraph.find_shortest_path('A', 'B', allowed_modes=['mv'])
        self.assertIsNotNone(route)
