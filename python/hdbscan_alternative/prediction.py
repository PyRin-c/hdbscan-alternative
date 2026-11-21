"""Predictions against a fitted HDBSCAN hierarchy."""

from operator import index
from warnings import warn

import numpy as np
from scipy.sparse import issparse

from ._native import _PredictionModel
from ._spatial_metrics import cache_distance


def _dense_values(points, requirements="AC"):
    if issparse(points):
        raise TypeError(
            "Sparse data was passed, but dense data is required. "
            "Use '.toarray()' to convert to a dense numpy array."
        )
    if np.iscomplexobj(points):
        raise ValueError(f"Complex data not supported\n{points}\n")
    values = np.require(points, dtype=np.float64, requirements=requirements)
    if values.ndim != 2:
        raise ValueError(f"Found array with dim {values.ndim}, while dim <= 2 is required.")
    if np.isnan(values).any():
        raise ValueError("Input contains NaN.")
    if not np.isfinite(values).all():
        raise ValueError("Input contains infinity or a value too large for dtype('float64').")
    if values.shape[0] == 0:
        shape = values.shape
        raise ValueError(f"Found array with 0 sample(s) (shape={shape}) while a minimum of 1 is required.")
    return values


class PredictionData:
    """Cached training features, hierarchy summaries, and exemplars for prediction."""

    def __init__(self, data, tree, labels, metric, min_samples):
        self.raw_data = _dense_values(data, requirements="A").copy(order="K")
        self._fitted_tree = tree
        self._labels = labels.tolist()
        self._metric = metric
        self._min_samples = min_samples
        self._build_model()
        core, selected, mapped, leaf_max, exemplars = self._model.metadata()
        self.core_distances = np.asarray(core, dtype=np.float64)
        self.cluster_tree = tree._values[tree._values["child_size"] > 1]
        self.cluster_map = {node: label for node, label, _ in mapped}
        self.reverse_cluster_map = dict(enumerate(selected))
        self.max_lambdas = {node: maximum for node, _, maximum in mapped}
        self.leaf_max_lambdas = dict(leaf_max)
        self.exemplars = [self.raw_data[rows] for rows in exemplars]

    def _build_model(self):
        self._model = _PredictionModel(
            self.raw_data, cache_distance(self._metric), self._min_samples, self._labels,
            self._fitted_tree._values.tolist(),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_model()


def _check_dimensions(data, points):
    if points.shape[1] != data.raw_data.shape[1]:
        raise ValueError("New points dimension does not match fit data!")


def _query_sample_count(clusterer, data):
    samples = index(clusterer.min_samples or clusterer.min_cluster_size)
    if samples < 0:
        raise ValueError("negative dimensions are not allowed")
    if samples * 2 > len(data.raw_data):
        raise ValueError("k must be less than or equal to the number of training points")
    return samples


def _query_settings(clusterer, data):
    samples = _query_sample_count(clusterer, data)
    tree = clusterer.condensed_tree_
    return samples, None if tree is data._fitted_tree else tree._values.tolist()


def approximate_predict(clusterer, points_to_predict, return_connecting_points=False):
    """Return predicted labels, strengths, and optionally the training point indices."""
    data = clusterer.prediction_data_
    points = np.asarray(points_to_predict)
    _check_dimensions(data, points)
    if not data.cluster_tree.size:
        warn(
            "Clusterer does not have any defined clusters, new data will be automatically predicted as noise.",
            UserWarning,
            stacklevel=2,
        )
        size = len(points)
        labels = np.full(size, -1, dtype=np.int32)
        result = (labels, np.zeros(size, dtype=np.float32))
        return (*result, labels.copy()) if return_connecting_points else result
    values = _dense_values(points)
    samples, tree = _query_settings(clusterer, data)
    labels, probabilities, neighbors = data._model.predict(values, samples, tree)
    result = (np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float64))
    return (*result, np.asarray(neighbors, dtype=np.int32)) if return_connecting_points else result


def approximate_predict_scores(clusterer, points_to_predict):
    """Return approximate outlier scores from the fitted condensed tree."""
    data = getattr(clusterer, "_prediction_data", None)
    if data is None:
        raise ValueError(
            "Clusterer does not have prediction data! Try fitting with prediction_data=True set, "
            "or run generate_prediction_data on the clusterer"
        )
    points = np.asarray(points_to_predict)
    _check_dimensions(data, points)
    if not data.cluster_tree.size:
        warn(
            "Clusterer does not have any defined clusters, new data will be automatically predicted as outliers.",
            UserWarning,
            stacklevel=2,
        )
        return np.ones(len(points), dtype=np.int32)
    values = _dense_values(points)
    samples, tree = _query_settings(clusterer, data)
    scores, exceptional = data._model.scores(values, samples, tree)
    scores = np.asarray(scores, dtype=np.float64)
    # NumPy controls the warning or exception for each undefined score.
    for row, maximum, density in exceptional:
        scores[row] = (np.float64(maximum) - np.float64(density)) / np.float64(maximum)
    return scores


def _membership_state(clusterer):
    tree = getattr(clusterer, "condensed_tree_", None)
    if tree is None:
        raise AttributeError("No condensed tree was generated; try running fit first.")
    data = clusterer.prediction_data_
    changed = None if tree is data._fitted_tree else (tree._values.tolist(), clusterer.labels_.tolist())
    return data, changed


def _membership_query(points, data):
    if isinstance(points, np.matrix):
        raise TypeError(
            "np.matrix is not supported. Please convert to a numpy array with np.asarray. "
            "For more information see: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html"
        )
    if not issparse(points):
        points = np.asarray(points)
    if points.ndim == 0:
        raise IndexError("tuple index out of range")
    if points.ndim == 1:
        raise ValueError(
            f"Expected 2D array, got 1D array instead:\narray={points}.\n"
            "Reshape your data either using array.reshape(-1, 1) if your data has a single feature "
            "or array.reshape(1, -1) if it contains a single sample."
        )
    values = _dense_values(points, requirements="A")
    if not values.shape[1]:
        raise ValueError(f"Found array with 0 feature(s) (shape={values.shape}) while a minimum of 1 is required.")
    if values.shape[1] != data.raw_data.shape[1]:
        raise ValueError("query data dimension must match training data dimension")
    return values


def membership_vector(clusterer, points_to_predict):
    """Return the membership probabilities of each new point in the selected clusters."""
    points = points_to_predict.astype(np.float64)
    data, changed = _membership_state(clusterer)
    values = _membership_query(points, data)
    samples = _query_sample_count(clusterer, data)
    (rows, cached_clusters, distances), (clusters, outliers), densities = (
        data._model.membership_inputs((values, samples), changed)
    )
    distances = np.asarray(distances, dtype=np.float64).reshape(rows, cached_clusters)
    outliers = np.asarray(outliers, dtype=np.float64).reshape(rows, clusters)
    result = np.empty((rows, clusters), dtype=np.float64)
    for index, (distance, outlier, density) in enumerate(zip(distances, outliers, densities, strict=True)):
        outlier = np.exp(outlier - np.nanmax(outlier))
        outlier /= outlier.sum()
        weights = np.sqrt(distance) * np.square(outlier)
        weights /= weights.sum()
        height, maximum = density
        result[index] = weights * (np.float64(height) / np.float64(maximum))
    return result


def all_points_membership_vectors(clusterer):
    """Return the selected-cluster memberships for all cached training points."""
    data, changed = _membership_state(clusterer)
    (rows, cached_clusters, distances), (clusters, outliers), densities = (
        data._model.membership_inputs(None, changed)
    )
    if clusters == 0:
        return np.zeros(rows, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64).reshape(rows, cached_clusters)
    outliers = np.asarray(outliers, dtype=np.float64).reshape(len(densities), clusters)
    outliers = np.exp(outliers - np.nanmax(outliers))
    outliers /= outliers.sum(axis=1, keepdims=True)
    probabilities = np.asarray([np.float64(height) / np.float64(maximum) for height, maximum in densities])
    combined = distances * outliers
    combined /= combined.sum(axis=1, keepdims=True)
    combined *= probabilities[:, None]
    return combined
