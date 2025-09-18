import unittest
from unittest.mock import patch
from src.multimodalrouter.graph.graph import RouteGraph
from threading import Lock


class TestRouteGraphInit(unittest.TestCase):
    def test_init_with_no_data_paths(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=True
        )
        self.assertEqual(graph.compressed, False)
        self.assertEqual(graph.extraMetricsKeys, [])
        self.assertEqual(graph.drivingEnabled, True)
        self.assertEqual(graph.TransportModes, {})
        self.assertEqual(graph.Graph, {})
        self.assertEqual(graph.maxDrivingDistance, 50)
        self.assertIsInstance(graph._lock, Lock)

    def test_init_with_data_paths(self):
        data_paths = {
            'airport': 'airports.csv',
            'shippingport': 'shippingports.csv'
        }
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'airport': 'fly', 'shippingport': 'shipping'},
            dataPaths=data_paths,
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=True
        )
        self.assertEqual(graph.compressed, False)
        self.assertEqual(graph.extraMetricsKeys, [])
        self.assertEqual(graph.drivingEnabled, True)
        self.assertEqual(graph.TransportModes, {'airport': 'fly', 'shippingport': 'shipping'})
        self.assertEqual(graph.Graph, {'airport': {}, 'shippingport': {}})
        self.assertEqual(graph.maxDrivingDistance, 50)
        self.assertIsInstance(graph._lock, Lock)

    def test_init_with_extra_metrics_keys(self):
        extra_metrics_keys = ['time', 'cost']
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'airport': 'fly'},
            dataPaths={'airport': 'airports.csv'},
            compressed=False,
            extraMetricsKeys=extra_metrics_keys,
            drivingEnabled=True
        )
        self.assertEqual(graph.compressed, False)
        self.assertEqual(graph.extraMetricsKeys, ['time', 'cost'])
        self.assertEqual(graph.drivingEnabled, True)
        self.assertEqual(graph.TransportModes, {'airport': 'fly'})
        self.assertEqual(graph.Graph, {'airport': {}})
        self.assertEqual(graph.maxDrivingDistance, 50)
        self.assertIsInstance(graph._lock, Lock)

    @patch('src.multimodalrouter.graph.graph.Lock')
    def test_init_with_driving_enabled(self, mock_lock):
        _ = RouteGraph(
            maxDistance=50,
            transportModes={'airport': 'fly'},
            dataPaths={'airport': 'airports.csv'},
            compressed=False, extraMetricsKeys=[],
            drivingEnabled=False
        )
        mock_lock.assert_called_once()

    def test_init_with_special_keys(self):
        graph = RouteGraph(
            maxDistance=50,
            transportModes={'airport': 'fly'},
            dataPaths={'airport': 'airports.csv'},
            compressed=False,
            extraMetricsKeys=[],
            drivingEnabled=True,
            sourceCoordKeys=['a', 'b'],
            destCoordKeys=['c', 'd']
        )

        self.assertEqual(graph.sourceCoordKeys, {'a', 'b'})
        self.assertEqual(graph.destCoordKeys, {'c', 'd'})


if __name__ == '__main__':
    unittest.main()
