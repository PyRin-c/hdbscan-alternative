"""Robust single linkage and its scikit-learn style estimator."""

from math import floor, sqrt
from operator import index

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClusterMixin

from ._spatial_metrics import BALL_ONLY, KD_METRICS, spatial_metric
from ._hdbscan import _validate_matrix_shape
from ._native import _robust_single_linkage
from .plots import SingleLinkageTree as _SingleLinkageTree
from .validity import _pairwise_distances


_TREE_ALGORITHMS = {"prims_kdtree", "prims_balltree", "boruvka_kdtree", "boruvka_balltree"}
_DEFAULT_MEMORY = object()


def robust_single_linkage(
    X, cut, k=5, alpha=sqrt(2.0), gamma=5, metric="euclidean", algorithm="best",
    memory=_DEFAULT_MEMORY, leaf_size=40, core_dist_n_jobs=4, **kwargs,
):
    """Return flat labels and the robust single-linkage hierarchy."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer greater than zero!")
    if not isinstance(alpha, float) or alpha < 1.0:
        raise ValueError("alpha must be a float greater than or equal to 1.0!")
    if not isinstance(gamma, int) or gamma < 1:
        raise ValueError("gamma must be an integer greater than zero!")
    if not isinstance(leaf_size, int) or leaf_size < 1:
        raise ValueError("Leaf size must be at least one!")
    if algorithm not in ("best", "generic", *_TREE_ALGORITHMS):
        raise TypeError(f"Unknown algorithm type {algorithm} specified")
    vector_metric = KD_METRICS.get(metric) if isinstance(metric, str) else None
    ball_only = isinstance(metric, str) and metric in BALL_ONLY
    if issparse(X):
        data = X.tocsr()
    else:
        data = np.asarray(X)
        if data.ndim <= 1:
            dimension = "scalar" if data.ndim == 0 else "1D"
            raise ValueError(
                f"Expected 2D array, got {dimension} array instead:\narray={data}.\n"
                "Reshape your data either using array.reshape(-1, 1) if your data has a single feature "
                "or array.reshape(1, -1) if it contains a single sample."
            )
        if data.ndim > 2:
            raise ValueError(f"Found array with dim {data.ndim}, while dim <= 2 is required.")
    _validate_matrix_shape(data.shape)
    values = data.data if issparse(data) else data
    if np.iscomplexobj(values):
        raise ValueError(f"Complex data not supported\n{data}\n")
    if values.dtype.kind not in "bifu":
        data = data.astype(np.float64)
        values = data.data if issparse(data) else data
    if np.isnan(values).any():
        raise ValueError("Input contains NaN.")
    if np.isinf(values).any():
        raise ValueError(f"Input contains infinity or a value too large for dtype('{values.dtype}').")
    if algorithm == "best":
        if issparse(data) or (vector_metric is None and not ball_only):
            algorithm = "generic"
        else:
            algorithm = "boruvka_balltree" if ball_only else "boruvka_kdtree"
    distance = None
    if algorithm == "generic":
        if k is True:
            raise ValueError("Booleans unacceptable as partition index")
        if metric == "precomputed" and issparse(data):
            try:
                from numpy.exceptions import AxisError
            except ImportError:
                from numpy import AxisError
            raise AxisError(0, ndim=0)
        if metric == "precomputed":
            rows, columns = data.shape
            if rows != columns:
                raise ValueError(
                    f"Precomputed metric requires shape (n_queries, n_indexed). Got {data.shape} for {rows} indexed."
                )
            if (data < 0).any():
                raise ValueError(
                    "Negative values in data passed to `pairwise_distances`. Precomputed distance "
                    " need to have non-negative values.."
                )
            distances = data
        else:
            distances = _pairwise_distances(data, metric, **kwargs)
        if distances.dtype == np.float32:
            raise ValueError("Buffer dtype mismatch, expected 'double_t' but got 'float'")
        data = distances
        vector_metric = "precomputed"
    elif vector_metric is None and not ball_only:
        raise ValueError(f"Unrecognized metric '{metric}'")
    elif issparse(data):
        raise TypeError("Sparse data was passed, but dense data is required. Use '.toarray()' to convert to a dense numpy array.")
    else:
        distance = spatial_metric(metric, data.shape[1], kwargs)
        if ball_only and algorithm.endswith("kdtree"):
            raise ValueError(f"metric {BALL_ONLY[metric]}Distance64 is not valid for KDTree64")
        vector_metric = "precomputed"
    if data.shape[0] == 1:
        if algorithm in ("prims_kdtree", "prims_balltree"):
            raise IndexError("index -1 is out of bounds for axis 1 with size 0")
        raise ValueError("Invalid shape in axis 0: 0.")
    if algorithm in ("boruvka_kdtree", "boruvka_balltree"):
        core_dist_n_jobs = floor(core_dist_n_jobs) if core_dist_n_jobs < 1 else int(core_dist_n_jobs)
    else:
        core_dist_n_jobs = 1
        leaf_size = 1
    fit = _fit_robust_tree
    if isinstance(memory, str):
        from joblib import Memory
        memory = Memory(memory, verbose=0)
    if memory is not _DEFAULT_MEMORY:
        fit = memory.cache(fit)
    return fit(
        np.require(data, dtype=np.float64, requirements="A"), vector_metric, algorithm,
        (cut, min(k, data.shape[0] - 1), alpha, int(gamma)), (index(leaf_size), core_dist_n_jobs), distance,
    )


def _fit_robust_tree(data, metric, algorithm, options, search, distance):
    labels, hierarchy = _robust_single_linkage(data, metric, algorithm, options, search, distance)
    return np.asarray(labels, dtype=np.intp), np.asarray(hierarchy, dtype=np.float64)


class RobustSingleLinkage(BaseEstimator, ClusterMixin):
    """Fit a robust single-linkage tree and cut it at the specified distance."""

    def __init__(
        self, cut=0.4, k=5, alpha=sqrt(2.0), gamma=5, metric="euclidean",
        algorithm="best", core_dist_n_jobs=4, metric_params=None,
    ):
        self.cut = cut
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.metric = metric
        self.algorithm = algorithm
        self.core_dist_n_jobs = core_dist_n_jobs
        self.metric_params = metric_params

    def fit(self, X, y=None):
        self.labels_, hierarchy = robust_single_linkage(
            X, self.cut, self.k, self.alpha, self.gamma, self.metric, self.algorithm,
            core_dist_n_jobs=self.core_dist_n_jobs, **(self.metric_params or {}),
        )
        self.cluster_hierarchy_ = _SingleLinkageTree(hierarchy)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_
