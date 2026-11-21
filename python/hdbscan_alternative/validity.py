"""Density-based cluster validity and its distance and tree operations."""

from math import asin, cos, sin, sqrt
from numbers import Integral
from operator import index
import os
from pathlib import Path
from warnings import warn

import numpy as np
from scipy.sparse import issparse
from scipy.spatial.distance import cdist, pdist, squareform

from ._native import _validity_internal_tree


def all_points_core_distance(distance_matrix, d=2.0):
    """Return the core distance of each row, updating the input matrix in place."""
    nonzero = distance_matrix != 0
    distance_matrix[nonzero] = (1.0 / distance_matrix[nonzero]) ** d
    mean = distance_matrix.sum(axis=1)
    mean /= distance_matrix.shape[0] - 1
    if mean.sum() == 0:
        return np.zeros(len(mean), dtype=np.float64)
    return mean ** (-1.0 / d)


def _pairwise_input(values, *, finite=True, boolean=False):
    if isinstance(values, np.matrix):
        raise TypeError(
            "np.matrix is not supported. Please convert to a numpy array with np.asarray. "
            "For more information see: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html"
        )
    sparse = issparse(values)
    original = values.data if sparse else values
    if np.iscomplexobj(original):
        raise ValueError(f"Complex data not supported\n{values}\n")
    dtype = bool if boolean else np.float32 if values.dtype == np.dtype(np.float32) else np.float64
    values = values.tocsr().astype(dtype, copy=False) if sparse else np.asarray(values, dtype=dtype)
    n_samples, n_features = values.shape
    if n_samples == 0:
        raise ValueError(
            f"Found array with 0 sample(s) (shape={values.shape}) "
            "while a minimum of 1 is required by check_pairwise_arrays."
        )
    if n_features == 0:
        raise ValueError(
            f"Found array with 0 feature(s) (shape={values.shape}) "
            "while a minimum of 1 is required by check_pairwise_arrays."
        )
    checked = values.data if sparse else values
    if finite is True and np.isnan(checked).any():
        raise ValueError("Input contains NaN.")
    if finite is not False and np.isinf(checked).any():
        raise ValueError(f"Input contains infinity or a value too large for dtype('{checked.dtype}').")
    return values


def _norms_squared(matrix):
    if not issparse(matrix):
        return np.einsum("ij,ij->i", matrix, matrix)
    squared = matrix.data * matrix.data
    result = np.zeros(matrix.shape[0], dtype=matrix.dtype)
    for row, (start, end) in enumerate(zip(matrix.indptr[:-1], matrix.indptr[1:])):
        if end > start:
            result[row] = np.add.accumulate(squared[start:end])[-1]
    return result


def _gram(left, right):
    result = left @ right.T
    return result.toarray() if issparse(result) else result


def _validate_norms(norms, data, name):
    if norms is None:
        return None
    norms = np.asarray(norms)
    if norms.ndim == 0:
        raise TypeError(f"Input should have at least 1 dimension i.e. satisfy `len(x.shape) > 0`, got scalar `{norms!r}` instead.")
    if norms.ndim > 2:
        raise ValueError(f"Found array with dim {norms.ndim}, while dim <= 2 is required.")
    if norms.dtype.kind not in "fiub":
        norms = norms.astype(np.float64)
    if np.isnan(norms).any():
        raise ValueError("Input contains NaN.")
    if np.isinf(norms).any():
        raise ValueError(f"Input contains infinity or a value too large for dtype('{norms.dtype}').")
    rows = data.shape[0]
    if norms.shape not in ((rows,), (rows, 1), (1, rows)):
        raise ValueError(
            f"Incompatible dimensions for {name} of shape {data.shape} "
            f"and {name}_norm_squared of shape {norms.shape}."
        )
    return None if norms.dtype == np.dtype(np.float32) else norms.reshape(-1)


def _euclidean(values, other=None, *, X_norm_squared=None, Y_norm_squared=None, squared=False):
    left = _pairwise_input(values)
    right = left if other is None else _pairwise_input(other)
    dtype = left.dtype
    left_norm = _validate_norms(X_norm_squared, left, "X")
    right_norm = _validate_norms(Y_norm_squared, right, "Y")
    if dtype == np.dtype(np.float32):
        left = left.astype(np.float64)
        right = left if other is None else right.astype(np.float64)
    if left_norm is None:
        left_norm = _norms_squared(left)
        if right_norm is None and other is None:
            right_norm = left_norm
    if right_norm is None:
        right_norm = _norms_squared(right)
    distances = -2.0 * _gram(left, right)
    distances += left_norm[:, None]
    distances += right_norm[None, :]
    distances = distances.astype(dtype, copy=False)
    np.maximum(distances, 0, out=distances)
    if other is None:
        np.fill_diagonal(distances, 0)
    return distances if squared else np.sqrt(distances, out=distances)


def _normalize(matrix):
    norms = np.sqrt(_norms_squared(matrix))
    if issparse(matrix):
        normalized = matrix.copy()
        for norm, start, end in zip(norms, matrix.indptr[:-1], matrix.indptr[1:]):
            if norm != 0:
                normalized.data[start:end] /= norm
    else:
        norms[norms < 10 * np.finfo(matrix.dtype).eps] = 1
        normalized = matrix / norms[:, None]
    return normalized


def _cosine(values, other=None):
    left = _normalize(_pairwise_input(values))
    right = left if other is None else _normalize(_pairwise_input(other))
    distances = 1 - _gram(left, right)
    np.clip(distances, 0, 2, out=distances)
    if other is None:
        np.fill_diagonal(distances, 0)
    return distances


def _dense(matrix):
    return matrix.toarray() if issparse(matrix) else matrix


def _manhattan(values, other=None):
    left = _dense(_pairwise_input(values))
    right = left if other is None else _dense(_pairwise_input(other))
    return cdist(left, right, metric="cityblock")


def _haversine(values, other=None):
    left = _dense(_pairwise_input(values))
    right = left if other is None else _dense(_pairwise_input(other))
    if left.shape[1] != 2:
        raise ValueError("Haversine distance only valid in 2 dimensions")
    left = left.astype(np.float64, copy=False)
    right = left if other is None else right.astype(np.float64, copy=False)
    distances = np.zeros((len(left), len(right)), dtype=np.float64)
    for i, (latitude, longitude) in enumerate(left):
        for j in range(i + 1 if other is None else 0, len(right)):
            lat, lon = right[j]
            first = sin(0.5 * (latitude - lat))
            second = sin(0.5 * (longitude - lon))
            reduced = first * first + cos(latitude) * cos(lat) * second * second
            value = 2.0 * asin(sqrt(reduced)) if 0.0 <= reduced <= 1.0 else -np.nan
            distances[i, j] = value
            if other is None:
                distances[j, i] = value
    return distances


def _mask_missing(values, missing_values, copy):
    missing_is_nan = np.isnan(missing_values)
    matrix = _pairwise_input(values, finite="allow-nan" if missing_is_nan else True)
    if issparse(matrix):
        raise TypeError("Sparse data was passed, but dense data is required. Use '.toarray()' to convert to a dense numpy array.")
    matrix = matrix.copy() if copy else matrix
    missing = np.isnan(matrix) if missing_is_nan else matrix == missing_values
    return matrix, missing


def _nan_euclidean(values, other=None, *, squared=False, missing_values=np.nan, copy=True):
    left, left_missing = _mask_missing(values, missing_values, copy)
    right, right_missing = (left, left_missing) if other is None else _mask_missing(other, missing_values, copy)
    left[left_missing] = 0
    right[right_missing] = 0
    distances = _euclidean(left, None if other is None else right, squared=True)
    distances -= (left * left) @ right_missing.T
    distances -= left_missing @ (right * right).T
    np.clip(distances, 0, None, out=distances)
    if other is None:
        np.fill_diagonal(distances, 0)
    count = (~left_missing).astype(np.intp) @ (~right_missing).astype(np.intp).T
    distances[count == 0] = np.nan
    np.maximum(count, 1, out=count)
    distances /= count
    distances *= left.shape[1]
    return distances if squared else np.sqrt(distances, out=distances)


_PAIRWISE = {
    "euclidean": (_euclidean, "euclidean_distances"),
    "l2": (_euclidean, "euclidean_distances"),
    "cosine": (_cosine, "cosine_distances"),
    "cityblock": (_manhattan, "manhattan_distances"),
    "l1": (_manhattan, "manhattan_distances"),
    "manhattan": (_manhattan, "manhattan_distances"),
    "haversine": (_haversine, "haversine_distances"),
    "nan_euclidean": (_nan_euclidean, "nan_euclidean_distances"),
}
_SCIPY_METRICS = {
    "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard",
    "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
    "sokalsneath", "sqeuclidean", "wminkowski", "yule",
}
_BOOLEAN_METRICS = {"dice", "jaccard", "rogerstanimoto", "russellrao", "sokalsneath", "yule"}


class InvalidParameterError(ValueError, TypeError):
    """Invalid argument to the array distance calculation."""


class DataConversionWarning(UserWarning):
    """Conversion of distance inputs to boolean values."""


def _distance_workers(n_jobs):
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        raise ValueError("n_jobs == 0 in Parallel has no meaning")
    if n_jobs > 0:
        return n_jobs
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    count = min(count, int(os.environ.get("LOKY_MAX_CPU_COUNT", count)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, (int(quota) + int(period) - 1) // int(period))
    except FileNotFoundError:
        pass
    return max(count + 1 + n_jobs, 1)


def _distance_blocks(matrix, workers, calculate):
    # Column slices can change the matrix product and its rounding. Preserve
    # those slices when the caller selects a worker count.
    rows = matrix.shape[0]
    distances = np.empty((rows, rows), dtype=matrix.dtype)
    width, remainder = divmod(rows, workers)
    first = 0
    for worker in range(min(workers, rows)):
        last = first + width + (worker < remainder)
        distances[:, first:last] = calculate(matrix, matrix[first:last])
        first = last
    return distances


def _pairwise_distances(matrix, metric, *, n_jobs=None, ensure_all_finite=True, **kwargs):
    if n_jobs is not None and not isinstance(n_jobs, Integral):
        raise InvalidParameterError(
            "The 'n_jobs' parameter of pairwise_distances must be an instance of 'int' or None. "
            f"Got {n_jobs!r} instead."
        )
    if not isinstance(ensure_all_finite, (bool, np.bool_)) and ensure_all_finite != "allow-nan":
        raise InvalidParameterError(
            "The 'ensure_all_finite' parameter of pairwise_distances must be an instance of "
            f"'bool' or an instance of 'numpy.bool' or a str among {{'allow-nan'}}. Got {ensure_all_finite!r} instead."
        )
    finite = bool(ensure_all_finite) if isinstance(ensure_all_finite, np.bool_) else ensure_all_finite
    if callable(metric):
        workers = _distance_workers(n_jobs)
        matrix = _pairwise_input(matrix, finite=finite)
        if workers != 1:
            def rectangular(left, right):
                return [[metric(left[i], right[j], **kwargs) for j in range(right.shape[0])]
                        for i in range(left.shape[0])]

            return _distance_blocks(matrix, workers, rectangular)
        n_samples = matrix.shape[0]
        distances = np.empty((n_samples, n_samples), dtype=matrix.dtype)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = distances[j, i] = metric(matrix[i], matrix[j], **kwargs)
        for i in range(n_samples):
            distances[i, i] = metric(matrix[i], matrix[i], **kwargs)
        return distances
    if isinstance(metric, str) and metric in _PAIRWISE:
        function, name = _PAIRWISE[metric]
        workers = _distance_workers(n_jobs)
        try:
            if workers == 1:
                return function(matrix, **kwargs)
            matrix = _pairwise_input(matrix, finite=False)
            distances = _distance_blocks(matrix, workers, lambda a, b: function(a, b, **kwargs))
            if function is _euclidean:
                np.fill_diagonal(distances, 0)
            return distances
        except TypeError as error:
            if str(error).startswith(f"{function.__name__}() got an unexpected keyword"):
                raise TypeError(str(error).replace(function.__name__, name, 1)) from None
            raise
    if not isinstance(metric, str) or metric not in _SCIPY_METRICS:
        allowed = {*_PAIRWISE, *_SCIPY_METRICS, "precomputed"}
        raise InvalidParameterError(
            f"The 'metric' parameter of pairwise_distances must be a str among {allowed} "
            f"or a callable. Got {metric!r} instead."
        )
    if issparse(matrix):
        raise TypeError("scipy distance metrics do not support sparse matrices.")
    boolean = metric in _BOOLEAN_METRICS
    if boolean and matrix.dtype != np.dtype(bool):
        warn(f"Data was converted to boolean for metric {metric}", DataConversionWarning, stacklevel=3)
    matrix = _pairwise_input(matrix, finite=finite, boolean=boolean)
    if metric == "seuclidean" and "V" not in kwargs:
        kwargs["V"] = np.var(matrix, axis=0, ddof=1)
    if metric == "mahalanobis" and "VI" not in kwargs:
        kwargs["VI"] = np.linalg.inv(np.cov(matrix.T)).T
    workers = _distance_workers(n_jobs)
    if workers != 1:
        if boolean:
            matrix = matrix.astype(np.float64)
        return _distance_blocks(matrix, workers, lambda a, b: cdist(a, b, metric=metric, **kwargs))
    return squareform(pdist(matrix, metric=metric, **kwargs))


def _reachability(raw, first, second):
    return np.dstack(np.broadcast_arrays(raw, second[None, :], first[:, None])).max(axis=-1)


def distances_between_points(
    X, labels, cluster_id, metric="euclidean", d=None, no_coredist=False,
    print_max_raw_to_coredist_ratio=False, **kwd_args,
):
    """Return the distances within one cluster and its all-points core distances."""
    if metric == "precomputed":
        if d is None:
            raise ValueError("If metric is precomputed a d value must be provided!")
        mask = labels == cluster_id
        raw = X[mask, :][:, mask]
    else:
        raw = _pairwise_distances(X[labels == cluster_id, :], metric, **kwd_args)
        d = X.shape[1]
    if no_coredist:
        return raw, None

    core = all_points_core_distance(raw.copy(), d)
    if print_max_raw_to_coredist_ratio:
        maximum = 0
        for i, j in zip(*np.nonzero(raw)):
            distance = raw[i, j]
            if maximum >= core[j] / distance:
                continue
            maximum = core[j] / distance
        print("Max raw distance to coredistance ratio:", maximum)
    return _reachability(raw, core, core), core


_BUFFER_TYPES = {
    "?": "bool", "b": "signed char", "B": "unsigned char", "h": "short", "H": "unsigned short",
    "i": "int", "I": "unsigned int", "l": "long", "L": "unsigned long",
    "q": "long long", "Q": "unsigned long long", "e": "float16_t", "f": "float",
    "g": "long double", "F": "float complex", "D": "double complex", "G": "long double complex",
}


def internal_minimum_spanning_tree(mr_distances):
    """Return the internal vertices and edges of a tree of reachability distances."""
    if not isinstance(mr_distances, np.ndarray):
        raise TypeError(
            "Argument 'distance_matrix' has incorrect type "
            f"(expected numpy.ndarray, got {type(mr_distances).__name__})"
        )
    if mr_distances.ndim != 2:
        raise ValueError(f"Buffer has wrong number of dimensions (expected 2, got {mr_distances.ndim})")
    if not mr_distances.dtype.isnative:
        raise ValueError("Big-endian buffer not supported on little-endian compiler")
    if mr_distances.dtype != np.dtype(np.float64):
        name = _BUFFER_TYPES.get(mr_distances.dtype.char, mr_distances.dtype.name)
        raise ValueError(f"Buffer dtype mismatch, expected 'double_t' but got '{name}'")
    nodes, edges, fallback = _validity_internal_tree(mr_distances)
    return (nodes if fallback else np.asarray(nodes, dtype=np.intp), np.asarray(edges, dtype=np.float64))


def density_separation(
    X, labels, cluster_id1, cluster_id2, internal_nodes1, internal_nodes2,
    core_distances1, core_distances2, metric="euclidean", no_coredist=False, **kwd_args,
):
    """Find the minimum distance between the internal vertices of two clusters."""
    left = labels == cluster_id1
    right = labels == cluster_id2
    if metric == "precomputed":
        raw = X[left, :][:, right][internal_nodes1, :][:, internal_nodes2]
    else:
        first = X[left, :][internal_nodes1, :]
        second = X[right, :][internal_nodes2, :]
        raw = cdist(first, second, metric=metric, **kwd_args)
    if no_coredist:
        return raw.min()
    first = core_distances1[internal_nodes1]
    second = core_distances2[internal_nodes2]
    return _reachability(raw, first, second).min()


def validity_index(
    X, labels, metric="euclidean", d=None, per_cluster_scores=False,
    mst_raw_dist=False, verbose=False, **kwd_args,
):
    """Compute DBCV for a clustering, optionally returning scores by cluster label."""
    cluster_count = index(labels.max() + 1)
    if cluster_count < 0:
        raise ValueError("negative dimensions are not allowed")
    scores = np.full(cluster_count, np.nan, dtype=np.float64)
    clusters = []
    for cluster in np.unique(labels[labels >= 0]):
        distances, core = distances_between_points(
            X, labels, cluster, metric, d, mst_raw_dist, verbose, **kwd_args,
        )
        nodes, edges = internal_minimum_spanning_tree(distances)
        clusters.append((cluster, nodes, core, edges[:, 2].max()))

    separation = np.full((cluster_count, cluster_count), np.inf, dtype=np.float64)
    for i, (left, left_nodes, left_core, _) in enumerate(clusters):
        for right, right_nodes, right_core, _ in clusters[i + 1:]:
            nearest = density_separation(
                X, labels, left, right, left_nodes, right_nodes, left_core, right_core,
                metric, mst_raw_dist, **kwd_args,
            )
            separation[left, right] = separation[right, left] = nearest

    overall = 0
    for cluster, _, _, sparseness in clusters:
        nearest = separation[cluster].min()
        if verbose:
            print(f"Minimum density separation: {nearest}")
            print(f"Density sparseness: {sparseness}")
        score = (nearest - sparseness) / max(nearest, sparseness)
        scores[int(cluster)] = score
        overall += (np.sum(labels == cluster) / X.shape[0]) * score
    return (overall, scores) if per_cluster_scores else overall
