"""Pandas, NetworkX and Matplotlib views of cluster hierarchies."""

from collections import defaultdict
from warnings import warn

import numpy as np

from ._native import _dbscan_clustering


class _Tree:
    def __init__(self, values):
        self._values = values

    def to_numpy(self):
        """Return a copy of the tree as a NumPy array."""
        return self._values.copy()


class SingleLinkageTree(_Tree):
    def get_clusters(self, cut_distance, min_cluster_size=5):
        """Return the cluster labels at the specified cut."""
        return np.asarray(_dbscan_clustering(self._values, cut_distance, min_cluster_size), dtype=np.intp)

    def to_pandas(self):
        from pandas import DataFrame

        table = DataFrame(self._values, columns=["left_child", "right_child", "distance", "size"])
        table.insert(0, "parent", np.arange(len(table) + 1, 2 * len(table) + 1, dtype=np.intp))
        return table

    def to_networkx(self):
        from networkx import DiGraph

        graph = DiGraph()
        for parent, (left, right, distance, size) in enumerate(self._values, start=len(self._values) + 1):
            graph.add_node(parent, size=size)
            graph.add_edge(parent, left, weight=distance)
            graph.add_edge(parent, right, weight=distance)
        return graph

    def plot(self, axis=None, truncate_mode=None, p=0, vary_line_width=True, cmap="viridis", colorbar=True):
        """Draw the single-linkage dendrogram and return its axes."""
        from matplotlib import cm, colormaps, colors, pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        axis = plt.gca() if axis is None else axis
        coords = dendrogram(self._values, truncate_mode=truncate_mode, p=p, no_plot=True)
        counts = {5.0 + 10 * i: 1 for i in range(len(coords["leaves"]))}
        segments = []
        maximum = 0.0
        for xs, ys in zip(coords["icoord"], coords["dcoord"]):
            x0, x1 = xs[0], xs[-1]
            left, right = counts[x0], counts[x1]
            counts[(x0 + x1) / 2] = left + right
            widths = np.log2([left + 1, right + 1]) if vary_line_width else (1.0, 1.0)
            log_sizes = np.log([left, right]) if vary_line_width else (0.0, 0.0)
            maximum = max(maximum, *log_sizes)
            segments.append((xs, ys, widths, log_sizes))
        normalize = colors.Normalize(0.0, maximum)
        palette = None if cmap == "none" else colormaps.get_cmap(cmap)
        for xs, ys, widths, log_sizes in segments:
            for start, width, log_size in zip((0, 2), widths, log_sizes):
                color = "k" if palette is None else palette(normalize(log_size))
                axis.plot(xs[start:start + 2], ys[start:start + 2], color=color, linewidth=width)
            axis.plot(xs[1:3], ys[1:3], color="k", linewidth=1.0)
        axis.set(xticks=[], ylabel="distance")
        if palette is not None and colorbar:
            axis.figure.colorbar(cm.ScalarMappable(norm=normalize, cmap=palette), ax=axis,
                                 label="log(Number of points)")
        return axis


class CondensedTree(_Tree):
    def __init__(self, condensed_tree_array, labels=None):
        super().__init__(condensed_tree_array)
        self._labels = labels

    def to_pandas(self):
        from pandas import DataFrame

        return DataFrame.from_records(self._values)

    def to_networkx(self):
        from networkx import DiGraph

        graph = DiGraph()
        for parent, child, level, size in self._values:
            graph.add_edge(parent, child, weight=level)
            graph.nodes[child]["size"] = size
        return graph

    def get_plot_data(self, leaf_separation=1, log_size=False, max_rectangle_per_icicle=20):
        """Return the geometry of the icicles and the links between them."""
        groups = defaultdict(list)
        children = defaultdict(list)
        births = {}
        sizes = {}
        for parent, child, level, count in self._values:
            parent, child = int(parent), int(child)
            groups[parent].append((level, count))
            if count > 1:
                children[parent].append(child)
                births[child] = level
                sizes[child] = count
        root = min(groups)
        sizes[root] = sum(count for _, count in groups[root])
        births[root] = 0.0
        scale = np.log if log_size else lambda value: value
        separation = leaf_separation * scale(sizes[root])

        order = []
        pending = [root]
        while pending:
            node = pending.pop()
            order.append(node)
            pending.extend(reversed(children[node]))
        centers = {}
        for node in order:
            if not children[node]:
                centers[node] = len(centers)
        for node in reversed(order):
            if children[node]:
                centers[node] = np.mean([centers[child] for child in children[node]])
        centers = {node: center * separation for node, center in centers.items()}

        result = {name: [] for name in ("bar_centers", "bar_tops", "bar_bottoms", "bar_widths", "line_xs", "line_ys")}
        bounds = {}

        def bar(center, width, low, high):
            if low == high:
                return
            result["bar_centers"].append(center)
            result["bar_widths"].append(width)
            result["bar_bottoms"].append(low)
            result["bar_tops"].append(high - low)

        for node in sorted(groups, reverse=True):
            center, size, birth = centers[node], sizes[node], births[node]
            departure = defaultdict(list)
            for level, count in groups[node]:
                departure[level].append(count)
            exits = sorted(departure)
            half = scale(size) / 2
            bounds[node] = [center - half, center + half, birth, exits[-1]]
            remaining = width = scale(size)
            bottom = previous = birth
            step = (width - scale(sum(departure[exits[-1]]))) / max_rectangle_per_icicle
            for level in exits[:-1]:
                if width - remaining > step:
                    bar(center, width, bottom, level)
                    bottom, width = previous, remaining
                if log_size:
                    for count in departure[level]:
                        remaining = np.log(np.exp(remaining) - count)
                else:
                    remaining -= sum(departure[level])
                previous = level
            bar(center, width, bottom, exits[-1])

        for parent, child, level, count in self._values:
            if count > 1:
                offset = scale(count) / 2
                side = centers[child] + offset if centers[child] > centers[parent] else centers[child] - offset
                result["line_xs"].append([centers[parent], side])
                result["line_ys"].append([level, level])
        result["cluster_bounds"] = bounds
        return result

    def _selected_nodes(self):
        if getattr(self, "_labels", None) is None:
            return ()
        labels = np.asarray(self._labels)
        lookup = defaultdict(set)
        child_clusters = []
        for parent, child, _, size in self._values:
            if size == 1:
                if child < len(labels) and labels[child] >= 0:
                    lookup[parent].add(int(labels[child]))
            else:
                child_clusters.append((parent, child))
        for parent, child in reversed(child_clusters):
            lookup[parent].update(lookup[child])
        selected = {}
        for node in np.unique(self._values["parent"]):
            if len(lookup[node]) == 1:
                label = next(iter(lookup[node]))
                selected.setdefault(label, node)
        return sorted(selected.items())

    def plot(self, leaf_separation=1, cmap="viridis", select_clusters=False, label_clusters=False,
             selection_palette=None, axis=None, colorbar=True, log_size=False, max_rectangles_per_icicle=20):
        """Draw the condensed hierarchy and return its axes."""
        from matplotlib import cm, colormaps, colors, pyplot as plt
        from matplotlib.patches import Ellipse

        axis = plt.gca() if axis is None else axis
        data = self.get_plot_data(leaf_separation, log_size, max_rectangles_per_icicle)
        palette = None if cmap == "none" else colormaps.get_cmap(cmap)
        widths = np.asarray(data["bar_widths"])
        norm = colors.Normalize(0.0, widths.max())
        heights = np.asarray(data["bar_tops"])
        axis.bar(data["bar_centers"], heights,
                 width=widths, bottom=data["bar_bottoms"], color="k" if palette is None else palette(norm(widths)),
                 linewidth=0)
        for xs, ys in zip(data["line_xs"], data["line_ys"]):
            axis.plot(xs, ys, color="k")
        axis.set_xlim(axis.get_xlim())
        axis.yaxis.set_inverted(True)
        axis.set_ylim(axis.get_ylim())
        if select_clusters:
            selected = self._selected_nodes()
            finite_center = finite_height = None
            if any(np.isinf(data["cluster_bounds"][node][3]) for _, node in selected):
                levels = np.concatenate((heights, data["bar_bottoms"]))
                finite = levels[np.isfinite(levels)]
                finite_center = np.mean([finite.max(), finite.min()])
                finite_height = np.diff(np.percentile(finite, [10, 90]))
                warn(
                    "Infinite lambda values encountered in chosen clusters. This might be due to duplicates in the data.",
                    UserWarning, stacklevel=2,
                )
            for label, node in selected:
                left, right, low, high = data["cluster_bounds"][node]
                center = ((left + right) / 2, (low + high) / 2)
                height = high - low
                if np.isinf(high):
                    center = (center[0], finite_center)
                    height = finite_height
                color = "r" if selection_palette is None else selection_palette[label]
                axis.add_artist(Ellipse(center, 2.0 * (right - left), 1.2 * height,
                                        fill=False, edgecolor=color, linewidth=2))
                if label_clusters:
                    axis.text(center[0] - 4 * (right - left), center[1] + .65 * height, str(label), va="bottom")
        axis.set(xticks=[], ylabel=r"$\lambda$ value")
        if palette is not None and colorbar:
            axis.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=palette), ax=axis,
                                 label="log(Number of points)" if log_size else "Number of points")
        return axis


class MinimumSpanningTree(_Tree):
    def __init__(self, mst, data):
        super().__init__(mst)
        self._data = data

    def to_pandas(self):
        from pandas import DataFrame

        return DataFrame({"from": self._values[:, 0].astype(np.intp), "to": self._values[:, 1].astype(np.intp),
                          "distance": self._values[:, 2]})

    def to_networkx(self):
        from networkx import Graph

        graph = Graph()
        for left, right, distance in self._values:
            i, j = int(left), int(right)
            graph.add_edge(i, j, weight=distance)
            graph.nodes[i]["data"] = self._data[i]
            graph.nodes[j]["data"] = self._data[j]
        return graph

    def plot(self, axis=None, node_size=40, node_color="k", node_alpha=0.8, edge_alpha=0.5,
             edge_cmap="viridis_r", edge_linewidth=2, vary_line_width=True, colorbar=True):
        """Draw the MST, projecting feature spaces over two dimensions with t-SNE."""
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection

        axis = plt.gca() if axis is None else axis
        positions = self._data
        if hasattr(positions, "toarray"):
            positions = positions.toarray()
        if positions.shape[1] > 2:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE

            if positions.shape[1] > 32:
                positions = PCA(n_components=32).fit_transform(positions)
            positions = TSNE().fit_transform(positions)
        endpoints = self._values[:, :2].astype(np.intp)
        distances = self._values[:, 2]
        widths = edge_linewidth * (np.log(distances.max() / distances) + 1.0) if vary_line_width else edge_linewidth
        edges = LineCollection(positions[endpoints], cmap=edge_cmap, alpha=edge_alpha, linewidths=widths)
        edges.set_array(distances)
        axis.add_collection(edges)
        axis.scatter(positions[:, 0], positions[:, 1], s=node_size, c=node_color, alpha=node_alpha)
        axis.set(xticks=[], yticks=[])
        if colorbar:
            axis.figure.colorbar(edges, ax=axis, label="Mutual reachability distance")
        return axis


def _graph_records(columns, names):
    records = np.rec.fromarrays(columns, names=names)
    return records.view(dtype=np.dtype(records.dtype.descr), type=np.ndarray)


class ApproximationGraph:
    """The point and edge arrays used to describe cluster connectivity."""

    def __init__(self, approximation_graphs, labels, probabilities, lens_values,
                 cluster_labels=None, cluster_probabilities=None,
                 sub_cluster_labels=None, sub_cluster_probabilities=None, *,
                 lens_name=None, sub_cluster_name=None, raw_data=None):
        self.lens_name = "lens_value" if lens_name is None else lens_name
        self.sub_cluster_name = "sub_cluster" if sub_cluster_name is None else sub_cluster_name
        self.point_mask = np.ones(len(labels), dtype=bool) if cluster_labels is None else cluster_labels >= 0
        self._data = raw_data
        index_data = np.arange(np.count_nonzero(self.point_mask), dtype=np.intp)
        fields = {
            "label": labels, "probability": probabilities, self.lens_name: lens_values,
            "cluster_label": cluster_labels, "cluster_probability": cluster_probabilities,
            f"{self.sub_cluster_name}_label": sub_cluster_labels,
            f"{self.sub_cluster_name}_probability": sub_cluster_probabilities,
        }
        fields = {name: np.asarray(value)[self.point_mask] for name, value in fields.items() if value is not None}
        self._points = _graph_records([index_data, *fields.values()], names=["id", *fields])
        graphs = tuple(approximation_graphs)
        edges = np.concatenate(graphs) if graphs else np.empty((0, 4), dtype=np.float64)
        groups = np.repeat(np.arange(len(graphs), dtype=np.intp), [len(graph) for graph in graphs])
        self._edges = _graph_records(
            [edges[:, 0].astype(np.intp), edges[:, 1].astype(np.intp), edges[:, 2], edges[:, 3], groups],
            names=["parent", "child", self.lens_name, "mutual_reachability", "component"],
        )

    def to_numpy(self):
        """Return a copy of the graph's structured point and edge arrays."""
        return self._points.copy(), self._edges.copy()

    def to_pandas(self):
        from pandas import DataFrame

        return DataFrame.from_records(self._points), DataFrame.from_records(self._edges)

    def to_networkx(self, feature_names=None):
        from networkx import Graph

        graph = Graph()
        for parent, child, lens, reachability, component in self._edges:
            graph.add_edge(
                parent, child, weight=1.0 / (1.0 + reachability), mutual_reachability=reachability,
                component=component, **{self.lens_name: lens},
            )
        rows = None if self._data is None else self._data[self.point_mask]
        if rows is not None and feature_names is None:
            feature_names = [f"feature {i}" for i in range(rows.shape[1])]
        for point in self._points:
            node = point["id"]
            if node in graph:
                graph.nodes[node].update({name: point[name] for name in self._points.dtype.names[1:]})
                if rows is not None:
                    graph.nodes[node].update(zip(feature_names, rows[node]))
        return graph

    def plot(self, positions=None, feature_names=None, node_color="label", node_vmin=None, node_vmax=None,
             node_cmap="viridis", node_alpha=1, node_size=1, node_marker="o", edge_color="k",
             edge_vmin=None, edge_vmax=None, edge_cmap="viridis", edge_alpha=1, edge_width=1):
        """Plot the approximation graph at supplied or Graphviz positions."""
        from matplotlib import colormaps, pyplot as plt
        from matplotlib.collections import LineCollection
        import networkx as nx

        axis = plt.gca()
        nodes = self._points
        endpoints = np.column_stack((self._edges["parent"], self._edges["child"]))
        if positions is None:
            layout = nx.nx_agraph.graphviz_layout(self.to_networkx(feature_names), prog="sfdp")
            positions = np.full((len(self.point_mask), 2), np.nan)
            for node, position in layout.items():
                positions[node] = position
        positions = np.asarray(positions)
        noise = np.empty((0, 2))
        if len(positions) == len(self.point_mask):
            points, noise = positions[self.point_mask], positions[~self.point_mask]
            edges = positions[endpoints]
        else:
            points = positions
            relative = np.cumsum(self.point_mask, dtype=np.intp) - 1
            edges = positions[relative[endpoints]]
        edge_colors = self._edges[edge_color] if isinstance(edge_color, str) and edge_color in self._edges.dtype.names else None
        strokes = LineCollection(edges, linewidths=edge_width, alpha=edge_alpha,
                                 colors=edge_color if edge_colors is None else None, cmap=edge_cmap)
        if edge_colors is not None:
            strokes.set_array(edge_colors)
            strokes.set_clim(edge_vmin, edge_vmax)
        axis.add_collection(strokes)
        palette = colormaps.get_cmap(node_cmap)
        numeric = isinstance(node_color, str) and node_color in nodes.dtype.names
        if numeric:
            node_color = nodes[node_color]
        elif isinstance(node_color, str) and feature_names is not None and node_color in feature_names:
            node_color = self._data[self.point_mask, list(feature_names).index(node_color)]
            numeric = True
        axis.scatter(noise[:, 0], noise[:, 1], s=node_size, c="silver", alpha=node_alpha, marker=node_marker)
        options = {"cmap": palette, "vmin": node_vmin, "vmax": node_vmax} if numeric else {}
        axis.scatter(points[:, 0], points[:, 1], s=node_size, c=node_color, alpha=node_alpha, marker=node_marker, **options)
