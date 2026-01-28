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

    def test_extreme_multi_dataset_all_matching_coord_keys_are_used_and_ordered(self):
        """
        Stress test:
        - multiple datasets
        - overlapping + unique coord keys
        - multiple matching coord columns -> higher dimensional coords
        - scrambled column order
        - superset matching with deterministic ordering
        - multiple rows
        """
        with tempfile.TemporaryDirectory() as tmpdir:

            air_path = os.path.join(tmpdir, "air.csv")
            sea_path = os.path.join(tmpdir, "sea.csv")
            rail_path = os.path.join(tmpdir, "rail.csv")

            # ---------------- AIR ----------------
            air_df = pd.DataFrame(
                {
                    "source_lat": [10, 11],
                    "source_lng": [20, 21],
                    "destination_lat": [30, 31],
                    "destination_lng": [40, 41],
                    "source": ["A1", "A2"],
                    "destination": ["B1", "B2"],
                    "distance": [5, 6],
                }
            ).sample(frac=1, axis=1)

            sea_df = pd.DataFrame(
                {
                    "src_y": [100, 101],
                    "src_x": [200, 201],
                    "dst_y": [300, 301],
                    "dst_x": [400, 401],
                    "source": ["S1", "S2"],
                    "destination": ["T1", "T2"],
                    "distance": [7, 8],
                }
            ).sample(frac=1, axis=1)

            # 4D coords (using both lat/lng and y/x)
            rail_df = pd.DataFrame(
                {
                    "source_lat": [1000],
                    "source_lng": [2000],
                    "src_y": [1],
                    "src_x": [2],
                    "destination_lat": [3000],
                    "destination_lng": [4000],
                    "dst_y": [3],
                    "dst_x": [4],
                    "source": ["R1"],
                    "destination": ["R2"],
                    "distance": [20],
                }
            ).sample(frac=1, axis=1)

            print(rail_df[:])

            air_df.to_csv(air_path, index=False)
            sea_df.to_csv(sea_path, index=False)
            rail_df.to_csv(rail_path, index=False)

            graph = RouteGraph(
                maxDistance=100,
                transportModes={"AIR": "fly", "SEA": "ship", "RAIL": "rail"},
                dataPaths={"AIR": air_path, "SEA": sea_path, "RAIL": rail_path},
                compressed=False,
                extraMetricsKeys=[],
                drivingEnabled=False,
                sourceCoordKeys=[
                    "source_lat",
                    "source_lng",
                    "src_y",
                    "src_x",
                ],
                destCoordKeys=[
                    "destination_lat",
                    "dst_y",
                    "destination_lng",
                    "dst_x",
                ],
            )

            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                graph.build()

            self.assertEqual(graph.getHub("AIR", "A1").coords, [10, 20])
            self.assertEqual(graph.getHub("AIR", "A2").coords, [11, 21])
            self.assertEqual(graph.getHub("AIR", "B1").coords, [30, 40])
            self.assertEqual(graph.getHub("AIR", "B2").coords, [31, 41])

            self.assertEqual(graph.getHub("SEA", "S1").coords, [100, 200])
            self.assertEqual(graph.getHub("SEA", "S2").coords, [101, 201])
            self.assertEqual(graph.getHub("SEA", "T1").coords, [300, 400])
            self.assertEqual(graph.getHub("SEA", "T2").coords, [301, 401])

            self.assertEqual(
                graph.getHub("RAIL", "R1").coords,
                [1000, 2000, 1, 2],
            )
            self.assertEqual(
                graph.getHub("RAIL", "R2").coords,
                [3000, 3, 4000, 4],
            )


if __name__ == '__main__':
    unittest.main()
