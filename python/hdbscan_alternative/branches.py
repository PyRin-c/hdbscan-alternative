"""Flare-sensitive branch detection and prediction within fitted clusters."""

from operator import index

import numpy as np
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin

from ._hdbscan import _CONDENSED_DTYPE
from ._native import _BranchDetectionData
from ._spatial_metrics import cache_distance
from .plots import ApproximationGraph
from .plots import CondensedTree as _CondensedTree
from .prediction import approximate_predict
from .robust_single_linkage_ import _SingleLinkageTree


class BranchDetectionData:
    """Finite training rows and neighborhoods used by the branch graphs."""

    def __init__(self, data, labels, metric, min_samples, leaf_size):
        if issparse(data):
            raise TypeError("Sparse data was passed, but dense data is required. Use '.toarray()' to convert to a dense numpy array.")
        self._raw_data = np.require(data, dtype=np.float64, requirements="A").copy()
        self._labels = labels.tolist()
        self._metric = metric
        self._min_samples = min_samples
        self._leaf_size = leaf_size
        self._build_model()
        rows, neighbors, core = self._model.metadata()
        self.all_finite = len(rows) == len(labels)
        self.finite_index = None if self.all_finite else np.asarray(rows, dtype=np.intp)
        self.internal_to_raw = None if self.all_finite else dict(enumerate(rows))
        self.neighbors = np.asarray(neighbors, dtype=np.intp).reshape(len(rows), min_samples)
        self.core_distances = np.asarray(core, dtype=np.float64)

    def _build_model(self):
        self._model = _BranchDetectionData(
            self._raw_data, self._labels, cache_distance(self._metric), self._min_samples, self._leaf_size,
        )

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != "_model"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_model()


def _threshold(value, name):
    if isinstance(value, (int, np.integer)):
        value = float(value)
    if not isinstance(value, float) or value < 0.0:
        raise ValueError(f"{name} must be a float greater or equal to 0.0, {value} given.")
    return value


def _detect(
    clusterer, cluster_labels, cluster_probabilities,
    branch_detection_method, label_sides_as_branches, min_cluster_size, max_cluster_size,
    allow_single_cluster, cluster_selection_method, cluster_selection_epsilon, cluster_selection_persistence,
):
    tree = getattr(clusterer, "_minimum_spanning_tree", None)
    if tree is None or clusterer._mst_without_features:
        raise ValueError(
            "Clusterer does not have an explicit minimum spanning tree! "
            "Try fitting with branch_detection_data=True or gen_min_span_tree=True set."
        )
    data = clusterer.branch_detection_data_
    if branch_detection_method not in ("core", "full"):
        raise ValueError(
            f'Invalid ``branch_detection_method``: {branch_detection_method}\nShould be one of: "core", "full"\n'
        )
    if min_cluster_size is None:
        min_cluster_size = clusterer.min_cluster_size
    if not isinstance(min_cluster_size, (int, np.integer)) or min_cluster_size < 2:
        raise ValueError(f"min_cluster_size must be an integer greater or equal to 2,  {min_cluster_size} given.")
    if max_cluster_size is None:
        max_cluster_size = clusterer.max_cluster_size
    if allow_single_cluster is None:
        allow_single_cluster = clusterer.allow_single_cluster
    if cluster_selection_method is None:
        cluster_selection_method = clusterer.cluster_selection_method
    if cluster_selection_method not in ("eom", "leaf"):
        raise ValueError(
            f'Invalid cluster_selection_method: {cluster_selection_method}\nShould be one of: "eom", "leaf"\n'
        )
    epsilon = _threshold(cluster_selection_epsilon, "cluster_selection_epsilon")
    persistence = _threshold(cluster_selection_persistence, "cluster_selection_persistence")
    labels_overridden = cluster_labels is not None
    if cluster_probabilities is None:
        cluster_probabilities = (
            clusterer.probabilities_ if cluster_labels is None else np.ones(len(cluster_labels))
        )
    if cluster_labels is None:
        cluster_labels = clusterer.labels_
    index(cluster_labels.max() + 1)
    size = len(clusterer.labels_)
    finite_index = data.finite_index
    if finite_index is None:
        if len(cluster_labels) != size:
            missing = min(size, len(cluster_labels))
            raise IndexError(f"index {missing} is out of bounds for axis 0 with size {missing}")
        if len(cluster_probabilities) != size:
            cluster_probabilities[cluster_labels >= 0]
    else:
        finite_labels = cluster_labels[finite_index]
        finite_probabilities = cluster_probabilities[finite_index]
    output = data._model.detect(
        cluster_labels.tolist(), cluster_probabilities.tolist(), tree._values.tolist(),
        (branch_detection_method, bool(label_sides_as_branches), labels_overridden, index(min_cluster_size),
         (cluster_selection_method, bool(allow_single_cluster)),
         (epsilon, np.inf, persistence, max_cluster_size)),
    )
    labels, probabilities, branch_labels, strengths, centrality, groups = output
    strengths = np.asarray(strengths, dtype=np.float64)
    typed = tuple(
        (
            np.asarray(points, dtype=np.intp),
            np.asarray(graph, dtype=np.float64).reshape(-1, 4),
            None if linkage is None else np.asarray(linkage, dtype=np.float64).reshape(-1, 4),
            None if condensed is None else np.asarray(condensed, dtype=_CONDENSED_DTYPE),
            tuple(0 for _ in persistences) if linkage is None else np.asarray(persistences, dtype=np.float64),
            exemplars,
        )
        for points, graph, linkage, condensed, persistences, exemplars in groups
    )
    if cluster_probabilities.dtype == np.dtype(np.float64):
        probabilities = np.asarray(probabilities, dtype=np.float64)
    else:
        # Branch strengths are accumulated in the caller's dtype before missing rows are restored.
        probabilities = cluster_probabilities.copy()
        minimum = 2 if label_sides_as_branches else 3
        for points, _, linkage, _, persistences, _ in typed:
            if linkage is None or len(persistences) >= minimum:
                updated = probabilities[points]
                np.add(updated, strengths[points], out=updated)
                updated *= 0.5
                probabilities[points] = updated
        if finite_index is not None:
            combined = np.zeros(size, dtype=np.float64)
            combined[finite_index] = probabilities[finite_index]
            probabilities = combined
    if finite_index is not None:
        cluster_labels = np.full(size, -1, dtype=np.intp)
        cluster_labels[finite_index] = finite_labels
        cluster_probabilities = np.zeros(size, dtype=np.float64)
        cluster_probabilities[finite_index] = finite_probabilities
    return (
        np.asarray(labels, dtype=np.intp), probabilities,
        cluster_labels, cluster_probabilities,
        np.asarray(branch_labels, dtype=np.intp), strengths,
        tuple(group[4] for group in typed),
        tuple(group[1] for group in typed),
        tuple(group[3] for group in typed),
        tuple(group[2] for group in typed),
        np.asarray(centrality, dtype=np.float64),
        tuple(group[0] for group in typed),
    ), tuple(group[5] for group in typed)


def detect_branches_in_clusters(
    clusterer, cluster_labels=None, cluster_probabilities=None,
    branch_detection_method="full", label_sides_as_branches=False,
    min_cluster_size=None, max_cluster_size=None, allow_single_cluster=None,
    cluster_selection_method=None, cluster_selection_epsilon=0.0, cluster_selection_persistence=0.0,
):
    """Return subgroup assignments, graphs, and local branch hierarchies."""
    result, _ = _detect(
        clusterer, cluster_labels, cluster_probabilities,
        branch_detection_method, label_sides_as_branches, min_cluster_size, max_cluster_size,
        allow_single_cluster, cluster_selection_method, cluster_selection_epsilon, cluster_selection_persistence,
    )
    return result


class BranchDetector(BaseEstimator, ClusterMixin):
    """Detect branches within the clusters of a fitted HDBSCAN model."""

    def __init__(
        self, branch_detection_method="full", label_sides_as_branches=False,
        min_cluster_size=None, max_cluster_size=None, allow_single_cluster=None,
        cluster_selection_method=None, cluster_selection_epsilon=0.0, cluster_selection_persistence=0.0,
    ):
        self.branch_detection_method = branch_detection_method
        self.label_sides_as_branches = label_sides_as_branches
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.allow_single_cluster = allow_single_cluster
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_persistence = cluster_selection_persistence

    def fit(self, clusterer, labels=None, probabilities=None):
        result, exemplars = _detect(
            clusterer, labels, probabilities,
            self.branch_detection_method, self.label_sides_as_branches, self.min_cluster_size,
            self.max_cluster_size, self.allow_single_cluster, self.cluster_selection_method,
            self.cluster_selection_epsilon, self.cluster_selection_persistence,
        )
        (
            self.labels_, self.probabilities_, self.cluster_labels_, self.cluster_probabilities_,
            self.branch_labels_, self.branch_probabilities_, self.branch_persistences_,
            self._graphs, condensed, linkage, self.centralities_, self.cluster_points_,
        ) = result
        self._clusterer = clusterer
        self.condensed_trees_ = [
            None if tree is None else _CondensedTree(tree, self.branch_labels_[points])
            for tree, points in zip(condensed, self.cluster_points_)
        ]
        self.linkage_trees_ = [None if tree is None else _SingleLinkageTree(tree) for tree in linkage]
        self._exemplar_points = exemplars
        return self

    def fit_predict(self, clusterer, labels=None, probabilities=None):
        return self.fit(clusterer, labels, probabilities).labels_

    @property
    def approximation_graph_(self):
        return ApproximationGraph(
            self._graphs, self.labels_, self.probabilities_, self.centralities_,
            self.cluster_labels_, self.cluster_probabilities_, self.branch_labels_, self.branch_probabilities_,
            lens_name="centrality", sub_cluster_name="branch", raw_data=self._clusterer._raw_data,
        )

    @property
    def exemplars_(self):
        data = self._clusterer._raw_data
        if any(tree is None for tree in self.condensed_trees_):
            raise TypeError("'NoneType' object is not subscriptable")
        return [None if rows is None else [data[branch] for branch in rows] for rows in self._exemplar_points]

    def weighted_centroid(self, label_id, data=None):
        if label_id == -1:
            raise ValueError("Cannot calculate weighted centroid for -1 cluster since it is a noise cluster")
        data = self._clusterer._raw_data if data is None else data
        mask = self.labels_ == label_id
        return np.average(data[mask], weights=self.probabilities_[mask], axis=0)

    def weighted_medoid(self, label_id, data=None):
        if label_id == -1:
            raise ValueError("Cannot calculate weighted centroid for -1 cluster since it is a noise cluster")
        data = self._clusterer._raw_data if data is None else data
        mask = self.labels_ == label_id
        members = data[mask]
        if not len(members):
            raise ValueError("Invalid shape in axis 0: 0.")
        metric, p = cache_distance(self._clusterer.branch_detection_data_._metric)
        if metric == "manhattan":
            metric = "cityblock"
        parameters = {"p": p} if metric == "minkowski" else {}
        distances = cdist(members, members, metric=metric, **parameters)
        return members[np.argmin(np.sum(distances * self.probabilities_[mask], axis=1))]


def approximate_predict_branch(branch_detector, points_to_predict):
    """Assign new points to the cluster and branch of their connecting neighbor."""
    clusters, cluster_strengths, neighbors = approximate_predict(
        branch_detector._clusterer, points_to_predict, return_connecting_points=True,
    )
    labels = clusters.astype(np.intp)
    strengths = cluster_strengths.copy()
    branches = np.zeros(len(clusters), dtype=np.intp)
    branch_strengths = np.ones(len(clusters), dtype=np.float64)
    minimum = 2 if branch_detector.label_sides_as_branches else 3
    for i, cluster in enumerate(clusters):
        if cluster < 0:
            continue
        neighbor = neighbors[i]
        if len(branch_detector.branch_persistences_[cluster]) >= minimum:
            labels[i] = branch_detector.labels_[neighbor]
            branches[i] = branch_detector.branch_labels_[neighbor]
            branch_strengths[i] = branch_detector.branch_probabilities_[neighbor]
            strengths[i] = (cluster_strengths[i] + branch_strengths[i]) * 0.5
    return labels, strengths, clusters, cluster_strengths, branches, branch_strengths
