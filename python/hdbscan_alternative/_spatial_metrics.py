"""Parse the parameters accepted by the dense KD-tree and Ball-tree paths."""

import numpy as np


KD_METRICS = {
    "euclidean": "euclidean", "l2": "euclidean",
    "manhattan": "manhattan", "l1": "manhattan", "cityblock": "manhattan",
    "minkowski": "minkowski", "p": "minkowski",
    "chebyshev": "chebyshev", "infinity": "chebyshev",
}
BALL_ONLY = {
    "seuclidean": "SEuclidean", "mahalanobis": "Mahalanobis", "hamming": "Hamming",
    "canberra": "Canberra", "braycurtis": "BrayCurtis", "jaccard": "Jaccard",
    "dice": "Dice", "rogerstanimoto": "RogersTanimoto", "russellrao": "RussellRao",
    "sokalmichener": "SokalMichener", "sokalsneath": "SokalSneath", "haversine": "Haversine",
}


def cache_distance(metric):
    """Read the distance from a cache, including older pickles with a metric name."""
    return (metric, 2.0) if isinstance(metric, str) else metric


def spatial_metric(metric, dimensions, arguments):
    name = KD_METRICS.get(metric, metric)
    arguments = dict(arguments)
    p, parameters = 2.0, None
    if name == "minkowski":
        p = float(arguments.pop("p", p))
        if p < 1:
            raise ValueError("p must be greater than 1")
        weights = arguments.pop("w", None)
        if weights is not None:
            if np.isinf(p):
                raise ValueError("MinkowskiDistance requires finite p. For p=inf, use ChebyshevDistance.")
            raise TypeError("__init__() got an unexpected keyword argument 'w'")
    elif name == "seuclidean":
        if "V" not in arguments:
            raise TypeError("__init__() takes exactly 1 positional argument (0 given)")
        variance = np.asarray(arguments.pop("V"), dtype=np.float64)
        if variance.ndim != 1 or variance.size != dimensions:
            raise ValueError("SEuclidean dist: size of V does not match")
        parameters = variance.tolist()
    elif name == "mahalanobis":
        covariance = arguments.pop("V", None)
        inverse = arguments.pop("VI", None)
        if inverse is None:
            if covariance is None:
                raise ValueError("Must provide either V or VI for Mahalanobis distance")
            inverse = np.linalg.inv(covariance)
        if inverse.ndim != 2 or inverse.shape != (dimensions, dimensions):
            raise ValueError("Mahalanobis dist: size of V does not match")
        parameters = inverse.astype(np.float64).ravel().tolist()
    elif name == "haversine" and dimensions != 2:
        raise ValueError("Haversine distance only valid in 2 dimensions")
    if arguments:
        argument = next(iter(arguments))
        raise TypeError(f"__init__() got an unexpected keyword argument '{argument}'")
    return name, p, parameters
