from ..graph import RouteGraph
import plotly.graph_objects as go


class GraphDisplay():

    def __init__(self, graph: RouteGraph, name: str = "Graph", iconSize: int = 10) -> None:
        self.graph: RouteGraph = graph
        self.name: str = name
        self.iconSize: int = iconSize

    def _toPlotlyFormat(
        self,
        nodeTransform=None,
        edgeTransform=None
    ):
        """
        transform the graph data into plotly format.to use the display function

        args:
            - nodeTransform: function to transform the node coordinates (default = None)
            - edgeTransform: function to transform the edge coordinates (default = None)
        returns:
            - None (modifies self.nodes and self.edges)
        """
        self.nodes = {
            f"{hub.hubType}-{hub.id}": {
                "coords": hub.coords,
                "hubType": hub.hubType,
                "id": hub.id
            }
            for hub in self.graph._allHubs()
        }

        self.edges = [
            {
                "from": f"{hub.hubType}-{hub.id}",
                "to": f"{self.graph.getHubById(dest).hubType}-{dest}",
                **edge.allMetrics
            }
            for hub in self.graph._allHubs()
            for _, edge in hub.outgoing.items()
            for dest, edge in edge.items()
        ]
        self.dim = max(len(node.get("coords")) for node in self.nodes.values())

        if nodeTransform is not None:
            expandedCoords = [node.get("coords") + [0] * (self.dim - len(node.get("coords"))) for node in self.nodes.values()]
            transformedCoords = nodeTransform(expandedCoords)
            for node, coords in zip(self.nodes.values(), transformedCoords):
                node["coords"] = coords

            self.dim = max(len(node.get("coords")) for node in self.nodes.values())

        if edgeTransform is not None:
            starts = [edge["from"] for edge in self.edges]
            startCoords = [self.nodes[start]["coords"] for start in starts]
            ends = [edge["to"] for edge in self.edges]
            endCoords = [self.nodes[end]["coords"] for end in ends]

            transformedEdges = edgeTransform(startCoords, endCoords)
            for edge, transformedEdge in zip(self.edges, transformedEdges):
                edge["curve"] = transformedEdge

    def display(
        self,
        nodeTransform=None,
        edgeTransform=None,
        displayEarth=False
    ):
        """
        function to display any 2D or 3D RouteGraph

        args:
            - nodeTransform: function to transform the node coordinates (default = None)
            - edgeTransform: function to transform the edge coordinates (default = None)
            - displayEarth: whether to display the earth as a background (default = False, only in 3D)

        returns:
            - None (modifies self.nodes and self.edges opens the plot in a browser)

        """
        # transform the graph
        self._toPlotlyFormat(nodeTransform, edgeTransform)
        # init plotly placeholders
        node_x, node_y, node_z, text, colors = [], [], [], [], []
        edge_x, edge_y, edge_z, edge_text = [], [], [], []

        # add all the nodes
        for node_key, node_data in self.nodes.items():
            x, y, *rest = node_data["coords"]
            node_x.append(x)
            node_y.append(y)
            if self.dim == 3:
                node_z.append(node_data["coords"][2])
            text.append(f"{node_data['id']}<br>Type: {node_data['hubType']}")
            colors.append(hash(node_data['hubType']) % 10)

        # add all the edges
        for edge in self.edges:
            # check if edge has been transformed
            if "curve" in edge:
                curve = edge["curve"]
                # add all the points of the edge
                for point in curve:
                    edge_x.append(point[0])
                    edge_y.append(point[1])
                    if self.dim == 3:
                        edge_z.append(point[2])
                edge_x.append(None)
                edge_y.append(None)
                # if 3d add the extra none to close the edge
                if self.dim == 3:
                    edge_z.append(None)
            else:
                source = self.nodes[edge["from"]]["coords"]
                target = self.nodes[edge["to"]]["coords"]

                edge_x += [source[0], target[0], None]
                edge_y += [source[1], target[1], None]

                if self.dim == 3:
                    edge_z += [source[2], target[2], None]

            # add text and hover display
            hover = f"{edge['from']} → {edge['to']}"
            metrics = {k: v for k, v in edge.items() if k not in ("from", "to", "curve")}
            if metrics:
                hover += "<br>" + "<br>".join(f"{k}: {v}" for k, v in metrics.items())
            edge_text.append(hover)

        if self.dim == 2:
            # ceate the plot in 2d
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=text,
                marker=dict(
                    size=self.iconSize,
                    color=colors,
                    colorscale="Viridis",
                    showscale=True
                )
            )

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=2, color="#888"),
                hoverinfo="text",
                text=edge_text,
                mode="lines"
            )

        elif self.dim == 3:
            # create the plot in 3d
            node_trace = go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode="markers",
                hoverinfo="text",
                text=text,
                marker=dict(
                    size=self.iconSize,
                    color=colors,
                    colorscale="Viridis",
                    showscale=True
                )
            )

            edge_trace = go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                line=dict(width=1, color="#888"),
                hoverinfo="text",
                text=edge_text,
                mode="lines",
                opacity=0.6
            )

        # create the plotly figure
        fig = go.Figure(data=[edge_trace, node_trace])
        # render earth / sphere in 3d
        if self.dim == 3 and displayEarth:
            try:
                import numpy as np
                R = 6369.9  # sphere radius
                u = np.linspace(0, 2 * np.pi, 50)   # azimuthal angle
                v = np.linspace(0, np.pi, 50)       # polar angle
                u, v = np.meshgrid(u, v)

                # Cartesian coordinates
                x = R * np.cos(u) * np.sin(v)
                y = R * np.sin(u) * np.sin(v)
                z = R * np.cos(v)
            except ImportError:
                raise ImportError("numpy is required to display the earth")

            sphere_surface = go.Surface(
                x=x, y=y, z=z,
                colorscale='Blues',
                opacity=1,
                showscale=False,
                hoverinfo='skip'
            )

            fig.add_trace(sphere_surface)

        fig.update_layout(title="Interactive Graph", showlegend=False, hovermode="closest")
        fig.show()

    @staticmethod
    def degreesToCartesian3D(coords):
        try:
            import torch
            C = torch.tensor(coords)
            if C.dim() == 1:
                C = C.unsqueeze(0)
            R = 6371.0
            lat = torch.deg2rad(C[:, 0])
            lng = torch.deg2rad(C[:, 1])
            x = R * torch.cos(lat) * torch.cos(lng)
            y = R * torch.cos(lat) * torch.sin(lng)
            z = R * torch.sin(lat)
            return list(torch.stack((x, y, z), dim=1).numpy())
        except ImportError:
            import math
            R = 6371.0
            output = []
            for lat, lng in coords:
                lat = math.radians(lat)
                lng = math.radians(lng)
                x = R * math.cos(lat) * math.cos(lng)
                y = R * math.cos(lat) * math.sin(lng)
                z = R * math.sin(lat)
                output.append([x, y, z])
            return output

    @staticmethod
    def curvedEdges(start, end, R=6371.0, H=0.05, n=20):
        try:
            # if torch and np are available calc vectorized graeter circle curves
            import numpy as np
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            start_np = np.array(start, dtype=np.float32)
            end_np = np.array(end, dtype=np.float32)

            start = torch.tensor(start_np, device=device)
            end = torch.tensor(end_np, device=device)
            start = start.float()
            end = end.float()

            # normalize to sphere
            start_norm = R * start / start.norm(dim=1, keepdim=True)
            end_norm = R * end / end.norm(dim=1, keepdim=True)

            # compute angle between vectors
            dot = (start_norm * end_norm).sum(dim=1, keepdim=True) / (R**2)
            dot = torch.clamp(dot, -1.0, 1.0)
            theta = torch.acos(dot).unsqueeze(2)  # shape: (num_edges,1,1)

            # linear interpolation along great circle
            t = torch.linspace(0, 1, n, device=device).view(1, n, 1)
            one_minus_t = 1 - t
            sin_theta = torch.sin(theta)
            sin_theta[sin_theta == 0] = 1e-6

            factor_start = torch.sin(one_minus_t * theta) / sin_theta
            factor_end = torch.sin(t * theta) / sin_theta

            curve = factor_start * start_norm.unsqueeze(1) + factor_end * end_norm.unsqueeze(1)

            # normalize to radius
            curve = R * curve / curve.norm(dim=2, keepdim=True)

            # apply radial lift at curve center using sin weight
            weight = torch.sin(torch.pi * t)  # 0 at endpoints, 1 at center
            curve = curve * (1 + H * weight)

            return curve
        except ImportError:
            # fallback to calculating quadratic bezier curves with math
            import math
            curves_all = []

            for startP, endP in zip(start, end):

                mid = [(s + e) / 2 for s, e in zip(startP, endP)]
                norm = math.sqrt(sum(c ** 2 for c in mid))
                mid_proj = [R * c / norm for c in mid]
                mid_arch = [c * (1 + H) for c in mid_proj]

                curve = []
                for i in range(n):
                    t = i / (n - 1)
                    one_minus_t = 1 - t
                    point = [
                        (one_minus_t ** 2) * s + 2 * one_minus_t * t * m + (t ** 2) * e
                        for s, m, e in zip(start, mid_arch, end)
                    ]
                    curve.append(point)

                curves_all.append(curve)

            return curves_all
