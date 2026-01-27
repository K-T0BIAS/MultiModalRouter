import unittest
from unittest.mock import patch
from multimodalrouter import RouteGraph
from threading import Lock
import tempfile
import pandas as pd
import os
import contextlib
import io


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
        self.assertIsInstance(graph._lock, type(Lock()))

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
        self.assertIsInstance(graph._lock, type(Lock()))

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
        self.assertIsInstance(graph._lock, type(Lock()))

    @patch('multimodalrouter.graph.graph.Lock')
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

        self.assertEqual(graph.sourceCoordKeys, ['a', 'b'])
        self.assertEqual(graph.destCoordKeys, ['c', 'd'])

    def test_multi_dataset_unique_coord_keys_are_matched_deterministically(self):
        """
        Ensure coordinate keys are matched deterministically across datasets
        where each dataset uses different coordinate column names.
        """
        with tempfile.TemporaryDirectory() as tmpdir:

            path1 = os.path.join(tmpdir, "air.csv")
            path2 = os.path.join(tmpdir, "sea.csv")

            # Dataset 1 uses *_lat / *_lng
            df1 = pd.DataFrame(
                columns=[
                    "source",
                    "destination",
                    "distance",
                    "source_lat",
                    "source_lng",
                    "destination_lat",
                    "destination_lng",
                ],
                data=[("A1", "B1", 5, 10, 20, 30, 40)],
            )

            # Dataset 2 uses *_y / *_x instead
            df2 = pd.DataFrame(
                columns=[
                    "source",
                    "destination",
                    "distance",
                    "src_y",
                    "src_x",
                    "dst_y",
                    "dst_x",
                ],
                data=[("A2", "B2", 7, 100, 200, 300, 400)],
            )

            df1.to_csv(path1, index=False)
            df2.to_csv(path2, index=False)

            graph = RouteGraph(
                maxDistance=50,
                transportModes={"AIR": "fly", "SEA": "ship"},
                dataPaths={"AIR": path1, "SEA": path2},
                compressed=False,
                extraMetricsKeys=[],
                drivingEnabled=False,
                # NOTE: superset of all possible keys — order matters
                # define source y and x sepperately
                sourceCoordKeys=["source_lat", "src_y", "source_lng", "src_x"],
                # define source coords and destination y and x separately
                destCoordKeys=["destination_lat", "destination_lng", "dst_y", "dst_x"],
            )

            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                graph.build()

            # test dataset 1
            hub_a1 = graph.getHub("AIR", "A1")
            hub_b1 = graph.getHub("AIR", "B1")

            self.assertEqual(hub_a1.coords, [10, 20])
            self.assertEqual(hub_b1.coords, [30, 40])

            # test dataset 2
            hub_a2 = graph.getHub("SEA", "A2")
            hub_b2 = graph.getHub("SEA", "B2")

            # Must preserve order from sourceCoordKeys/destCoordKeys
            self.assertEqual(hub_a2.coords, [100, 200])
            self.assertEqual(hub_b2.coords, [300, 400])


if __name__ == '__main__':
    unittest.main()
