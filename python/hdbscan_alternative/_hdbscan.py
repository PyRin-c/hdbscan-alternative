"""Estimator and NumPy adapters for the native HDBSCAN fit."""

from math import floor
from operator import index
from os import name as os_name
from sys import maxsize
from warnings import warn

import numpy as np
from joblib import Memory
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array

from ._spatial_metrics import BALL_ONLY, KD_METRICS, spatial_metric
from ._native import (
    _dbscan_clustering, _emit_fit_diagnostics, _fit_csr_with_diagnostics, _fit_dense_with_diagnostics,
    _outlier_scores, _relative_validity_components,
)
from .plots import CondensedTree, MinimumSpanningTree, SingleLinkageTree, _Tree
from .prediction import PredictionData


_CONDENSED_DTYPE = np.dtype([
    ("parent", np.intp),
    ("child", np.intp),
    ("lambda_val", np.float64),
    ("child_size", np.intp),
])
_METRICS = {
    "euclidean": "euclidean",
    "l2": "euclidean",
    "manhattan": "manhattan",
    "cityblock": "manhattan",
    "l1": "manhattan",
    "cosine": "cosine",
    "arccos": "cosine",
    "precomputed": "precomputed",
}
_FLOAT32_BUFFER_ERROR = "Buffer dtype mismatch, expected 'double_t' but got 'float'"
_PRIM_TREES = {"prims_kdtree": "KDTree", "prims_balltree": "BallTree"}
_BORUVKA_TREES = {"boruvka_kdtree": "KDTree", "boruvka_balltree": "BallTree"}
_BUFFER_TYPES = {
    "?": "bool", "b": "signed char", "h": "short", "i": "int", "l": "long", "q": "long long", "f": "float",
    "B": "unsigned char", "H": "unsigned short", "I": "unsigned int", "L": "unsigned long",
    "Q": "unsigned long long", "g": "long double",
}


def _minimum_count(value, name, minimum):
    try:
        count = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if count < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return count


def _empty_sample_error(n_features):
    return ValueError(f"Found array with 0 sample(s) (shape=(0, {n_features})) while a minimum of 1 is required.")


def _validate_matrix_shape(shape):
    n_samples, n_features = shape
    if n_samples == 0:
        raise _empty_sample_error(n_features)
    if n_features == 0:
        raise ValueError(f"Found array with 0 feature(s) (shape=({n_samples}, 0)) while a minimum of 1 is required.")


def _validate_sparse_dtype(csr, metric):
    n_samples, n_features = csr.shape
    if metric == "precomputed":
        if np.isnan(csr.data).any():
            raise ValueError("Input contains NaN.")
        if np.isinf(csr.data).any():
            raise ValueError(f"Input contains infinity or a value too large for dtype('{csr.dtype}').")
        if n_samples == 1:
            raise ValueError(
                "all the input array dimensions except for the concatenation axis must match exactly, "
                "but along dimension 1, the array at index 0 has size 0 and the array at index 2 has size 1"
            )
        return
    unsupported = csr.dtype == np.dtype(np.float32) and metric in ("euclidean", "cosine")
    wide_indices = metric == "manhattan" and csr.indices.dtype.itemsize > np.dtype(np.intc).itemsize
    if unsupported or wide_indices or n_samples == 1:
        invalid = np.flatnonzero(~np.isfinite(csr.data))
        if invalid.size and not csr.has_canonical_format:
            csr = csr.copy()
            csr.sum_duplicates()
            invalid = np.flatnonzero(~np.isfinite(csr.data))
        invalid_rows = np.searchsorted(csr.indptr[1:], invalid, side="right")
        if np.unique(invalid_rows).size == n_samples:
            raise _empty_sample_error(n_features)
        if unsupported:
            raise ValueError(_FLOAT32_BUFFER_ERROR)
        if wide_indices:
            index_type = "long" if csr.indices.dtype.char == "l" else "long long"
            raise ValueError(f"Buffer dtype mismatch, expected 'const int' but got '{index_type}'")
        raise ValueError("Invalid shape in axis 0: 0.")


def _validate_dense_dtype(data, metric, float32):
    n_samples, n_features = data.shape
    if metric == "precomputed":
        if np.isnan(data).any():
            raise ValueError("Input contains NaN.")
        if float32:
            raise ValueError(_FLOAT32_BUFFER_ERROR)
        if n_samples == 1:
            raise ValueError("Invalid shape in axis 0: 0.")
        return
    unsupported = float32 and metric in ("euclidean", "cosine")
    if unsupported or n_samples == 1:
        if not np.isfinite(data).all(axis=1).any():
            raise _empty_sample_error(n_features)
        if unsupported:
            raise ValueError(_FLOAT32_BUFFER_ERROR)
        raise ValueError("Invalid shape in axis 0: 0.")


def _dense_search(algorithm, metric, n_features, leaf_size, approximate, jobs):
    if algorithm == "generic":
        return None
    if algorithm == "best":
        if metric in ("cosine", "arccos"):
            raise ValueError(f"Unrecognized metric '{metric}'")
        if metric in KD_METRICS:
            algorithm = "boruvka_kdtree" if n_features <= 60 else "prims_kdtree"
        elif metric in BALL_ONLY:
            algorithm = "boruvka_balltree" if n_features <= 60 else "prims_balltree"
        else:
            return None
    elif metric not in KD_METRICS and (metric not in BALL_ONLY or algorithm.endswith("kdtree")):
        if algorithm in _PRIM_TREES:
            raise ValueError(f"Cannot use Prim's with {_PRIM_TREES[algorithm]} for this metric!")
        raise ValueError(f"Cannot use Boruvka with {_BORUVKA_TREES[algorithm]} for this metric!")
    leaf_size = index(max(3, leaf_size) if algorithm in _BORUVKA_TREES else leaf_size)
    if leaf_size > maxsize:
        raise OverflowError("Python int too large to convert to C ssize_t")
    if algorithm in _PRIM_TREES:
        return "prims", leaf_size, approximate, 1
    jobs = floor(max(jobs, -maxsize - 1)) if jobs < 1 else int(jobs)
    if jobs > maxsize:
        message = "int too big to convert" if os_name == "nt" else "Python int too large to convert to C long"
        raise OverflowError(message)
    return algorithm, leaf_size, approximate, jobs


def _fit_metric_distances(fit, matrix, limits, config, metric, parameters, *, functional):
    from .validity import _pairwise_distances

    if functional:
        distances = _pairwise_distances(matrix, metric, **parameters)
        # A pair with no comparable coordinates has no finite edge.
        np.copyto(distances, np.inf, where=np.isnan(distances))
        _validate_dense_dtype(distances, "precomputed", distances.dtype == np.dtype(np.float32))
        return fit(np.require(distances, dtype=np.float64, requirements="A"), limits,
                   ("precomputed", *config[1:]), None)
    if issparse(matrix):
        matrix = matrix.tocsr()
        invalid = np.flatnonzero(~np.isfinite(matrix.data))
        finite = np.ones(matrix.shape[0], dtype=bool)
        finite[np.searchsorted(matrix.indptr[1:], invalid, side="right")] = False
    else:
        finite = np.isfinite(matrix).all(axis=1)
    rows = np.count_nonzero(finite)
    if rows == 0:
        raise _empty_sample_error(matrix.shape[1])
    if rows == 1:
        raise ValueError("Invalid shape in axis 0: 0.")
    distances = _pairwise_distances(matrix if finite.all() else matrix[finite], metric, **parameters)
    if distances.dtype == np.dtype(np.float32):
        raise ValueError(_FLOAT32_BUFFER_ERROR)
    config = ("precomputed", *config[1:])
    return fit(
        np.require(distances, dtype=np.float64, requirements="A"), limits, config, None, None,
        (matrix.shape[1], finite.tolist()),
    )


def hdbscan(
    X,
    min_cluster_size=5,
    min_samples=None,
    alpha=1.0,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    max_cluster_size=0,
    metric="minkowski",
    p=2,
    leaf_size=40,
    algorithm="best",
    memory=Memory(None, verbose=0),
    approx_min_span_tree=True,
    gen_min_span_tree=False,
    core_dist_n_jobs=4,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    match_reference_implementation=False,
    cluster_selection_epsilon_max=float("inf"),
    **kwargs,
):
    """Return labels, probabilities, persistence, condensed tree, linkage and optional MST.

    Accepts finite feature vectors or precomputed distances. The last item is
    ``None`` unless ``gen_min_span_tree`` is true.
    """
    model = HDBSCAN(
        min_cluster_size, min_samples, alpha=alpha,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_persistence=cluster_selection_persistence,
        max_cluster_size=max_cluster_size, metric=metric, p=p, leaf_size=leaf_size,
        algorithm=algorithm, memory=memory, approx_min_span_tree=approx_min_span_tree,
        gen_min_span_tree=gen_min_span_tree, core_dist_n_jobs=core_dist_n_jobs,
        cluster_selection_method=cluster_selection_method, allow_single_cluster=allow_single_cluster,
        match_reference_implementation=match_reference_implementation,
        cluster_selection_epsilon_max=cluster_selection_epsilon_max,
    )
    (labels, probabilities, persistence, mst, linkage, condensed), _ = model._fit_raw(X, metric_kwargs=kwargs)
    return (
        np.asarray(labels, dtype=np.intp),
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(persistence, dtype=np.float64),
        np.asarray(condensed, dtype=_CONDENSED_DTYPE),
        np.asarray(linkage, dtype=np.float64),
        None if mst is None else np.asarray(mst, dtype=np.float64),
    )


class HDBSCAN(BaseEstimator, ClusterMixin):
    """Cluster dense or sparse features and distances with HDBSCAN."""

    match_reference_implementation = False

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=None,
        *,
        metric="euclidean",
        alpha=1.0,
        algorithm="best",
        leaf_size=40,
        approx_min_span_tree=True,
        gen_min_span_tree=False,
        core_dist_n_jobs=4,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        match_reference_implementation=False,
        prediction_data=False,
        branch_detection_data=False,
        cluster_selection_epsilon=0.0,
        cluster_selection_epsilon_max=float("inf"),
        cluster_selection_persistence=0.0,
        max_cluster_size=0,
        max_dist=0.0,
        p=None,
        memory=None,
        metric_params=None,
        **kwargs,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.alpha = alpha
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree
        self.core_dist_n_jobs = core_dist_n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.match_reference_implementation = match_reference_implementation
        self.prediction_data = prediction_data
        self.branch_detection_data = branch_detection_data
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_epsilon_max = cluster_selection_epsilon_max
        self.cluster_selection_persistence = cluster_selection_persistence
        self.max_cluster_size = max_cluster_size
        self.max_dist = max_dist
        self.p = p
        self.memory = memory
        self.metric_params = metric_params
        self._metric_kwargs = kwargs

    def _distance_parameters(self):
        return {
            **(getattr(self, "metric_params", None) or {}), **getattr(self, "_metric_kwargs", {}),
        }

    def _fit_raw(self, X, *, metric_kwargs=None):
        functional = metric_kwargs is not None
        metric = _METRICS.get(self.metric) if isinstance(self.metric, str) else None
        parameters = metric_kwargs if functional else self._distance_parameters()
        if self.metric == "minkowski":
            generic_parameters = {"p": self.p}
        elif self.metric == "arccos":
            generic_parameters = {}
        else:
            generic_parameters = parameters
        fit_dense, fit_csr = _fit_dense_with_diagnostics, _fit_csr_with_diagnostics
        memory = getattr(self, "memory", None)
        precomputed = metric == "precomputed"
        min_samples, min_cluster_size = self.min_samples, self.min_cluster_size
        if functional:
            min_samples = min_cluster_size if min_samples is None else min_samples
            if any(type(value) is not int and not isinstance(value, np.integer)
                   for value in (min_samples, min_cluster_size)):
                raise ValueError("Min samples and min cluster size must be integers!")
            if min_samples < 1 or min_cluster_size < 1:
                raise ValueError("Min samples and Min cluster size must be positive integers")
            if min_cluster_size == 1:
                raise ValueError("Min cluster size must be greater than one")
            min_samples, min_cluster_size = int(min_samples), int(min_cluster_size)
        else:
            min_cluster_size = _minimum_count(min_cluster_size, "min_cluster_size", 2)
            if min_samples is not None:
                min_samples = _minimum_count(min_samples, "min_samples", 1)
        if (functional or self.algorithm != "generic") and (not isinstance(self.alpha, float) or self.alpha <= 0.0):
            raise ValueError("Alpha must be a positive float value greater than 0!")
        if self.leaf_size < 1:
            raise ValueError("Leaf size must be greater than 0!")
        epsilon = self.cluster_selection_epsilon
        if isinstance(epsilon, (int, np.integer)) and not isinstance(epsilon, bool):
            epsilon = float(epsilon)
        if type(epsilon) is not float or epsilon < 0.0:
            raise ValueError("Epsilon must be a float value greater than or equal to 0!")
        persistence = self.cluster_selection_persistence
        if type(persistence) is not float or persistence < 0.0:
            raise ValueError("Persistence must be a float value greater than or equal to 0!")
        epsilon_max = self.cluster_selection_epsilon_max
        if epsilon_max < epsilon:
            raise ValueError("Cluster selection epsilon max must be greater than epsilon!")
        if self.metric == "minkowski":
            if self.p is None:
                raise TypeError("Minkowski metric given but no p value supplied!")
            if self.p < 0:
                raise ValueError("Minkowski metric with negative p value is not defined!")
        if self.algorithm not in ("generic", "best", *_PRIM_TREES, *_BORUVKA_TREES):
            if functional:
                raise TypeError(f"Unknown algorithm type {self.algorithm} specified")
            raise NotImplementedError("unsupported HDBSCAN distance or algorithm")
        if self.cluster_selection_method not in ("eom", "leaf"):
            if functional:
                raise ValueError('Invalid Cluster Selection Method: %s\nShould be one of: "eom", "leaf"\n')
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")
        if functional or memory is not None:
            if isinstance(memory, str):
                memory = Memory(memory, verbose=0)
            fit_dense, fit_csr = memory.cache(fit_dense), memory.cache(fit_csr)
        if functional:
            if precomputed and not issparse(X):
                checked = X.copy()
                checked[np.isinf(checked)] = 1
                check_array(checked)
            else:
                X = check_array(X, accept_sparse="csr", ensure_all_finite=False)
        max_cluster_size = self.max_cluster_size
        if max_cluster_size <= 0:
            max_cluster_size = 0
        allow_single_cluster = self.allow_single_cluster
        if functional or isinstance(allow_single_cluster, (int, np.integer)):
            allow_single_cluster = bool(allow_single_cluster)
        include_mst = bool(self.gen_min_span_tree or getattr(self, "branch_detection_data", False))
        strict = bool(getattr(self, "match_reference_implementation", False))
        if strict:
            min_samples = max(1, (min_cluster_size if min_samples is None else min_samples) - 1)
            min_cluster_size += 1
        config = (
            metric or "precomputed",
            include_mst,
            (self.cluster_selection_method, allow_single_cluster),
            (epsilon, epsilon_max, persistence, max_cluster_size),
            strict,
        )
        limits = (min_cluster_size, min_samples, self.alpha)
        if issparse(X):
            if X.ndim != 2:
                raise ValueError("expected a two-dimensional sparse matrix")
            _validate_matrix_shape(X.shape)
            if self.algorithm in _PRIM_TREES or self.algorithm in _BORUVKA_TREES:
                raise ValueError("Sparse data matrices only support algorithm 'generic'.")
            if precomputed and X.shape[0] != X.shape[1]:
                if functional:
                    raise ValueError("compressed-sparse graph must be shape (N, N)")
                raise ValueError("precomputed distances must be a square matrix")
            csr = X.tocsr()
            if len(csr.indptr) != X.shape[0] + 1:
                raise ValueError("invalid CSR row offsets for the matrix shape")
            if np.iscomplexobj(csr.data):
                raise ValueError("input values must be real")
            if functional and metric is not None:
                check_array(csr, accept_sparse="csr")
            _validate_sparse_dtype(csr, metric)
            data_dtype = np.float32 if metric == "manhattan" and csr.dtype == np.dtype(np.float32) else np.float64
            if metric is None or (functional and not precomputed and generic_parameters):
                result = _fit_metric_distances(fit_dense, csr, limits, config, self.metric,
                                               generic_parameters, functional=functional)
            else:
                max_dist = parameters.get("max_dist", 0.0) if functional else self.max_dist
                result = fit_csr(
                    np.require(csr.indptr, dtype=np.uintp, requirements="A"),
                    np.require(csr.indices, dtype=np.uintp, requirements="A"),
                    np.require(csr.data, dtype=data_dtype, requirements="A"),
                    csr.shape[1], (*limits, max_dist if precomputed else 0.0), config,
                )
        else:
            X = np.asarray(X)
            if np.iscomplexobj(X):
                raise ValueError("feature values must be real")
            if X.ndim != 2:
                raise ValueError("expected a two-dimensional feature matrix")
            _validate_matrix_shape(X.shape)
            data = np.require(X, dtype=np.float64, requirements="A")
            if not functional and self.algorithm != "generic" and not precomputed and not np.isfinite(data).all(axis=1).any():
                raise _empty_sample_error(data.shape[1])
            search = _dense_search(
                self.algorithm, self.metric, X.shape[1], self.leaf_size,
                bool(self.approx_min_span_tree) and not strict, self.core_dist_n_jobs,
            )
            if functional and not precomputed and (search is not None or metric is not None):
                check_array(X)
            if precomputed and X.shape[0] != X.shape[1]:
                if functional:
                    rows, columns = X.shape
                    if rows == 1:
                        raise ValueError(f"non-broadcastable output operand with shape (1,{columns}) "
                                         f"doesn't match the broadcast shape ({columns},{columns})")
                    if columns > 1:
                        raise ValueError(f"operands could not be broadcast together with shapes "
                                         f"({rows},{columns}) ({columns},1) ({rows},{columns}) ")
                raise ValueError("precomputed distances must be a square matrix")
            if functional and precomputed and X.dtype != np.dtype(np.float64):
                code = X.dtype.char
                if not X.dtype.isnative:
                    raise ValueError("Big-endian buffer not supported on little-endian compiler")
                if code == "e":
                    raise ValueError("Does not understand character buffer dtype format string ('e')")
                buffer_type = _BUFFER_TYPES.get(code, str(X.dtype))
                raise ValueError(f"Buffer dtype mismatch, expected 'double_t' but got '{buffer_type}'")
            if search is None and (metric is None or (functional and not precomputed and generic_parameters)):
                result = _fit_metric_distances(fit_dense, X, limits, config, self.metric,
                                               generic_parameters, functional=functional)
            else:
                distance = None
                if search is None:
                    _validate_dense_dtype(data, metric, X.dtype == np.dtype(np.float32))
                elif metric is None or parameters:
                    distance = spatial_metric(self.metric, X.shape[1], parameters)
                result = fit_dense(data, limits, config, search, distance)
        fitted, diagnostics = result
        # Apply the current NumPy error policy even when the fit came from joblib.
        if diagnostics:
            _emit_fit_diagnostics(diagnostics)
        return fitted, X

    def fit(self, X, y=None):
        """Fit a feature matrix or a square distance matrix and return the estimator."""
        result, X = self._fit_raw(X)
        precomputed = self.metric == "precomputed"
        labels, probabilities, persistence, mst, linkage, condensed = result
        labels = np.asarray(labels, dtype=np.intp)
        tree = MinimumSpanningTree(np.asarray(mst, dtype=np.float64), X) if mst is not None else None
        (
            self.labels_,
            self.probabilities_,
            self.cluster_persistence_,
            self._minimum_spanning_tree,
            self._mst_without_features,
            self.single_linkage_tree_,
            self.condensed_tree_,
        ) = (
            labels,
            np.asarray(probabilities, dtype=np.float64),
            np.asarray(persistence, dtype=np.float64),
            tree,
            precomputed and mst is not None,
            SingleLinkageTree(np.asarray(linkage, dtype=np.float64)),
            CondensedTree(np.asarray(condensed, dtype=_CONDENSED_DTYPE), labels),
        )
        self._raw_data = None if precomputed else X
        self._mst_sparse_precomputed = precomputed and issparse(X)
        if getattr(self, "prediction_data", False):
            self.generate_prediction_data()
        if getattr(self, "branch_detection_data", False):
            self.generate_branch_detection_data()
        return self

    def fit_predict(self, X, y=None):
        """Fit and return cluster labels, with `-1` for noise."""
        return self.fit(X, y).labels_

    def _postfit_distance(self, metric):
        if metric in ("euclidean", "manhattan"):
            return metric, 2.0
        parameters = self._distance_parameters()
        # The estimator's p controls generic fitting. The caches use the
        # tree metric's default unless a distance argument is supplied.
        if metric == "minkowski" and "w" in parameters:
            raise TypeError("__init__() got an unexpected keyword argument 'w'")
        name, p, _ = spatial_metric(metric, 0, parameters)
        return name, p

    def generate_prediction_data(self):
        """Cache the data needed for predictions on new samples."""
        if self.metric == "precomputed":
            warn(
                "Cannot generate prediction data for non-vector space inputs -- "
                "access to the source data rather than mere distances is required!",
                UserWarning,
                stacklevel=2,
            )
            return
        metric = KD_METRICS.get(self.metric)
        if metric is None:
            warn(f"Metric {self.metric} not supported for prediction data!", UserWarning, stacklevel=2)
            return
        tree = getattr(self, "condensed_tree_", None)
        if tree is None:
            raise AttributeError("No condensed tree was generated; try running fit first.")
        self._prediction_data = PredictionData(
            self._raw_data, tree, self.labels_, self._postfit_distance(metric),
            index(self.min_samples or self.min_cluster_size),
        )

    def generate_branch_detection_data(self):
        """Cache the finite rows and neighbors used for branch detection."""
        from .branches import BranchDetectionData

        if self.metric == "precomputed":
            warn("Branch detection for non-vector space inputs is not (yet) implemented.", UserWarning, stacklevel=2)
            return
        metric = KD_METRICS.get(self.metric)
        if metric is None:
            warn(f"Metric {self.metric} not supported for branch detection!", UserWarning, stacklevel=2)
            return
        labels = self.labels_
        self._branch_detection_data = BranchDetectionData(
            self._raw_data, labels, self._postfit_distance(metric),
            index(self.min_samples or self.min_cluster_size), 40,
        )

    @property
    def branch_detection_data_(self):
        """Training data and neighborhoods cached for branch detection."""
        data = getattr(self, "_branch_detection_data", None)
        if data is None:
            raise AttributeError("No branch detection data was generated")
        return data

    @branch_detection_data_.setter
    def branch_detection_data_(self, value):
        self._branch_detection_data = value

    @property
    def prediction_data_(self):
        """The training data and hierarchy summaries cached for prediction."""
        data = getattr(self, "_prediction_data", None)
        if data is None:
            raise AttributeError("No prediction data was generated")
        return data

    @prediction_data_.setter
    def prediction_data_(self, value):
        self._prediction_data = value

    @property
    def exemplars_(self):
        """Expose training samples representing each selected cluster."""
        if getattr(self, "_prediction_data", None) is None:
            if self.metric == "precomputed":
                raise AttributeError(
                    "Currently exemplars require the use of vector input datawith a suitable metric. "
                    "This will likely change in the future, but for now no exemplars can be provided"
                )
            self.generate_prediction_data()
        return getattr(self, "_prediction_data", None).exemplars

    @exemplars_.setter
    def exemplars_(self, value):
        self._exemplars = value

    @property
    def outlier_scores_(self):
        """Compute and cache GLOSH scores in the original sample order."""
        if getattr(self, "_outlier_scores", None) is None:
            tree = getattr(self, "condensed_tree_", None)
            if tree is None:
                raise AttributeError("No condensed tree was generated; try running fit first.")
            self._outlier_scores = np.asarray(_outlier_scores(tree._values.tolist()), dtype=np.float64)
        return self._outlier_scores

    @property
    def relative_validity_(self):
        """Compute and cache the DBCV approximation from the fitted spanning tree."""
        cached = getattr(self, "_relative_validity", None)
        if cached is not None:
            return cached
        if not self.gen_min_span_tree:
            raise AttributeError(
                "Minimum spanning tree not present. Either HDBSCAN object was created with "
                "gen_min_span_tree=False or the tree was not generated in spite of it owing to "
                "internal optimization criteria."
            )
        labels = self.labels_
        tree = getattr(self, "_minimum_spanning_tree", None)
        if (tree is None or not isinstance(labels, np.ndarray)
                or labels.ndim != 1 or labels.dtype != np.dtype(np.intp)):
            # Preserve bincount's validation for user-supplied labels.
            np.bincount(labels + 1)
        if tree is None:
            raise AttributeError("'NoneType' object has no attribute 'T'")
        n_samples = len(labels)
        sizes, sparseness, separation, fallback = _relative_validity_components(
            np.require(labels, dtype=np.int64, requirements="A"), tree._values,
            getattr(self, "_mst_sparse_precomputed", False),
        )
        sparseness = np.asarray(sparseness, dtype=np.float64)
        separation = np.asarray(separation, dtype=np.float64)
        correction = 2.0 * np.float64(fallback)
        separation[separation == np.inf] = correction
        validity = (separation - sparseness) / np.maximum(separation, sparseness)
        self._relative_validity = np.sum(sizes * validity / n_samples)
        return self._relative_validity

    @relative_validity_.setter
    def relative_validity_(self, value):
        self._relative_validity = value

    def dbscan_clustering(self, cut_distance, min_cluster_size=5):
        """Return the DBSCAN* labels at a cut of the fitted single linkage tree."""
        tree = getattr(self, "single_linkage_tree_", None)
        if tree is None:
            raise AttributeError("No single linkage tree was generated; try running fit first.")
        return np.asarray(_dbscan_clustering(tree._values, cut_distance, min_cluster_size), dtype=np.intp)

    @property
    def minimum_spanning_tree_(self):
        """The spanning tree from a fit with `gen_min_span_tree=True`."""
        if getattr(self, "_mst_without_features", False):
            warn(
                "No raw data is available; this may be due to using a precomputed metric matrix. "
                "No minimum spanning tree will be provided without raw data.",
                UserWarning,
                stacklevel=2,
            )
            return None
        tree = getattr(self, "_minimum_spanning_tree", None)
        if tree is None:
            raise AttributeError(
                "No minimum spanning tree was generated.This may be due to "
                "optimized algorithm variations that skip explicit generation of the spanning tree."
            )
        return tree
