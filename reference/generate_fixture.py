"""Record HDBSCAN outputs observed through the public API of its Wheel."""

import argparse
import contextlib
import io
import json
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

# Keep the Linux wheel on the same sort dispatch on x86-64-v4 hosts.
if platform.system() == "Linux" and platform.machine() == "x86_64":
    os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4,AVX512_ICL,AVX512_SPR"

import numpy as np
from hdbscan import (
    HDBSCAN,
    BranchDetector,
    RobustSingleLinkage,
    all_points_membership_vectors,
    approximate_predict,
    approximate_predict_branch,
    approximate_predict_scores,
    detect_branches_in_clusters,
    hdbscan,
    membership_vector,
    robust_single_linkage,
)
from hdbscan.validity import (
    all_points_core_distance,
    density_separation,
    distances_between_points,
    internal_minimum_spanning_tree,
    validity_index,
)
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits


def rounding_case(n_features):
    rng = np.random.default_rng(302)
    return np.concatenate([
        rng.normal(loc=mean, scale=0.95, size=(4, n_features))
        for mean in (0, 3, 8)
    ]).tolist()


CASES = {
    "two-clusters-and-noise": [
        [0.0, 0.0],
        [0.25, 0.125],
        [0.75, 0.0],
        [0.125, 0.5],
        [4.0, 1.0],
        [4.375, 1.125],
        [4.125, 1.75],
        [4.5, 1.5],
        [2.0, 8.0],
    ],
    "ties-and-duplicates": [[0.0], [0.0], [0.0], [2.0], [2.0], [2.0], [1.0]],
    "all-noise": [[0.0], [1.0], [2.0], [3.0]],
    "nested-parent-selected": [
        [0.0], [0.125], [0.25], [0.5625], [0.6875], [0.8125], [8.0], [8.125], [8.25],
    ],
    "nested-children-selected": [
        [0.0], [0.125], [0.25], [1.5], [1.625], [1.75], [8.0], [8.125], [8.25],
    ],
    "five-feature-rounding": [
        [0.31, 0.17, -0.23, 0.32, 0.09],
        [0.23, 0.28, 0.15, 0.07, 0.36],
        [0.56, -0.11, 0.21, 0.31, -0.23],
        [0.43, -0.03, 0.25, -0.07, 0.15],
        [3.26, 2.53, 3.28, 3.17, 2.99],
        [3.18, 2.89, 3.14, 2.91, 3.01],
        [2.94, 3.04, 2.91, 3.23, 3.36],
        [2.84, 2.88, 3.02, 3.09, 3.1],
        [6.2, -3.6, 9.15, 3.1, 2.3],
    ],
    "eight-feature-rounding": rounding_case(8),
    "sixteen-feature-rounding": rounding_case(16),
}

PRECOMPUTED_CASES = {
    "precomputed-finite": (
        [
            [0, 1, 2, 8, 9, 10],
            [1, 0, 1, 7, 8, 9],
            [2, 1, 0, 6, 7, 8],
            [8, 7, 6, 0, 1, 2],
            [9, 8, 7, 1, 0, 1],
            [10, 9, 8, 2, 1, 0],
        ],
        {"alpha": 0.5},
    ),
    "precomputed-connected-with-missing-distances": (
        [
            [0, 1, "inf", "inf"],
            [1, 0, 5, "inf"],
            ["inf", 5, 0, 1],
            ["inf", "inf", 1, 0],
        ],
        {"min_cluster_size": 2, "min_samples": 1},
    ),
    "precomputed-asymmetric-with-nonzero-diagonal": (
        [[3, 1, 7], [4, 2, 1], [6, 9, 8]],
        {"min_cluster_size": 2, "min_samples": 1},
    ),
}

SPARSE_CASES = {
    "csr-tied-chain": (
        csr_matrix((
            [1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
            [0, 1, 3, 5, 7, 9, 11, 12],
        ), shape=(7, 7)),
        {"min_samples": 1},
    ),
    "csr-unsorted-duplicates": (
        csr_matrix(([2, 7, 8, 2, 3, 8, 3], [1, 1, 2, 0, 3, 1, 2], [0, 2, 4, 6, 7]), shape=(4, 4)),
        {"min_samples": 1},
    ),
    "csr-fallback-for-insufficient-neighbors": (
        csr_matrix([[0, 2, 0, 0], [2, 0, 8, 0], [0, 8, 0, 3], [0, 0, 3, 0]]),
        {"min_samples": 2, "max_dist": 0.3},
    ),
    "csr-asymmetric-with-explicit-zero": (
        csr_matrix(([1, 0, 1, 2, 3, 2], [1, 2, 0, 2, 0, 1], [0, 2, 4, 6]), shape=(3, 3)),
        {"min_samples": 2},
    ),
    "csr-alpha-rounding-with-diagonal": (
        csr_matrix([[0.1, 1.1, 0, 0], [1.1, 0.1, 8, 0], [0, 8, 0.1, 3], [0, 0, 3, 0.1]]),
        {"min_samples": 2, "alpha": 0.3},
    ),
}

def sparse_rounding_case():
    rng = np.random.default_rng(9732)
    data = rng.normal(size=160)
    columns = rng.integers(0, 24, size=160)
    return csr_matrix((data, columns, np.arange(9) * 20), shape=(8, 32))


SPARSE_FEATURE_CASES = {
    "csr-features-with-empty-rows": (
        csr_matrix([
            [0.03, 0, 0.08, 0, 0], [0.1106, 0.24, 0, 0, 0], [0, 0.41, 0, 0, 0],
            [0, 0, 0, 2.13, 1.7], [0, 0, 0.13, 2.05, 1.9], [0, 0, 0.52, 0, 2.3],
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        ]),
        {"min_cluster_size": 2, "min_samples": 1},
    ),
    "csr-features-unsorted-duplicates": (
        csr_matrix((
            [0.51, 2.5, 1.25, 1.09, 2.25, 1.53, 1.75, 1.0, 2.01, 0.5, 3.0, 4.03],
            [2, 0, 0, 0, 2, 1, 2, 1, 3, 2, 1, 1],
            [0, 3, 5, 7, 10, 12],
        ), shape=(5, 4)),
        {"min_cluster_size": 2, "min_samples": 1},
    ),
    "csr-features-all-implicit-zeros": (
        csr_matrix((4, 8), dtype=np.float64),
        {},
    ),
    "csr-features-rounding": (
        sparse_rounding_case(),
        {"min_cluster_size": 2, "min_samples": 1, "alpha": 0.75},
    ),
}

ANGULAR_ROWS = [
    [1.0, 1.0, 0.0], [0.1, 0.08, 0.0], [1.0, 1.0, 0.0],
    [-1.0, -1.0, 0.0], [-1.0, -0.9, 0.0], [0.0, 0.0, 0.0],
    [2e-15, 0.0, 0.0], [2.25e-15, 0.0, 0.0],
]

METRIC_CSR = csr_matrix((
    [0.45, 0.55, 1.2, -2.5, 1.0, 2.5, 2.6, 0.77, -0.55, 0.44, 1.89],
    [1, 1, 0, 2, 0, 1, 2, 0, 2, 2, 1],
    [0, 3, 5, 7, 11],
), shape=(4, 3))


def long_duplicate_rows():
    rng = np.random.default_rng(488)
    n_rows, n_features, row_length = 7, 9, 45
    count = n_rows * row_length
    values = rng.normal(size=count)
    columns = rng.integers(0, n_features, size=count)
    return csr_matrix((values, columns, np.arange(n_rows + 1) * row_length), shape=(n_rows, n_features))

DISTANCE_METRIC_CASES = {
    "dense-manhattan": (ANGULAR_ROWS, "manhattan"),
    "csr-manhattan-duplicates": (METRIC_CSR, "manhattan"),
    "csr-manhattan-long-duplicate-rows": (long_duplicate_rows(), "manhattan"),
    "csr-float32-manhattan": (csr_matrix(ANGULAR_ROWS, dtype=np.float32), "manhattan"),
    "dense-cosine-small-and-zero-norms": (ANGULAR_ROWS, "cosine"),
    "csr-cosine-small-and-zero-norms": (csr_matrix(ANGULAR_ROWS), "cosine"),
    "csr-cosine-duplicates": (METRIC_CSR, "cosine"),
}

CLUSTER_SELECTION_CASES = {
    "leaf-selection-below-stable-parent": (
        CASES["nested-parent-selected"],
        {"cluster_selection_method": "leaf"},
    ),
    "leaf-with-root-only": (
        CASES["all-noise"],
        {"cluster_selection_method": "leaf", "allow_single_cluster": True},
    ),
    "eom-single-cluster-with-noise": (
        [[0], [0.125], [0.25], [0.5], [9]],
        {"allow_single_cluster": True},
    ),
    "eom-root-wins-equal-stability": (
        [[0], [1], [2], [4], [5], [6]],
        {"min_samples": 1, "allow_single_cluster": True},
    ),
    "eom-root-outranks-leaves-with-noise": (
        [[0], [3], [6], [10], [13], [16], [32]],
        {"min_samples": 1, "allow_single_cluster": True},
    ),
    "eom-children-outrank-root-csr": (
        csr_matrix(CASES["nested-children-selected"]),
        {"allow_single_cluster": True},
    ),
    "eom-single-cluster-identical-points": (
        [[0.0]] * 6,
        {"allow_single_cluster": True},
    ),
    "eom-single-cluster-precomputed": (
        [[0, 1, 5], [1, 0, 4], [5, 4, 0]],
        {"metric": "precomputed", "min_cluster_size": 2, "min_samples": 1, "allow_single_cluster": True},
    ),
    "leaf-sparse-precomputed": (
        SPARSE_CASES["csr-tied-chain"][0],
        {"metric": "precomputed", "min_samples": 1, "cluster_selection_method": "leaf"},
    ),
}


def tied_feature_rows(n_samples):
    rng = np.random.default_rng(20261121)
    return rng.integers(0, 30, size=(n_samples, 1)).tolist()


def tied_sparse_distances(n_samples):
    left = np.arange(n_samples - 1)
    right = left + 1
    weights = left % 7 + 1
    return csr_matrix((np.tile(weights, 2), (np.r_[left, right], np.r_[right, left])),
                      shape=(n_samples, n_samples))


LARGE_TIE_CASES = {
    "dense-256-tied-mst-edges": (tied_feature_rows(257), {}),
    "dense-257-tied-mst-edges": (tied_feature_rows(258), {}),
    "dense-259-tied-mst-edges": (tied_feature_rows(260), {"cluster_selection_method": "leaf"}),
    "csr-precomputed-321-tied-mst-edges": (
        tied_sparse_distances(322), {"metric": "precomputed", "min_samples": 1},
    ),
}

PERSISTENCE_PAIR = [
    [0.0, 0.5, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 0.125], [1.0, 1.0, 0.125, 0.0],
]
PERSISTENCE_BRANCH = [
    [0.0, 2 / 3, 1.0, 1.0, 1.0, 1.0], [2 / 3, 0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 0.2, 1 / 3, 1 / 3], [1.0, 1.0, 0.2, 0.0, 1 / 3, 1 / 3],
    [1.0, 1.0, 1 / 3, 1 / 3, 0.0, 1 / 6], [1.0, 1.0, 1 / 3, 1 / 3, 1 / 6, 0.0],
]
PERSISTENCE_PRECOMPUTED = {"metric": "precomputed", "min_cluster_size": 2, "min_samples": 1}

THRESHOLD_CASES = {
    "epsilon-keeps-branches-at-equality": (
        CASES["nested-parent-selected"], {"min_samples": 1, "cluster_selection_epsilon": 0.3125},
    ),
    "epsilon-merges-leaves-into-parent": (
        CASES["nested-parent-selected"], {"min_samples": 1, "cluster_selection_method": "leaf", "cluster_selection_epsilon": 0.5},
    ),
    "epsilon-selects-root-and-equal-child": (
        CASES["nested-parent-selected"], {"min_samples": 1, "allow_single_cluster": True, "cluster_selection_epsilon": 7.1875},
    ),
    "epsilon-root-includes-early-points": (
        [[0], [3], [6], [10], [13], [16], [32]],
        {"min_samples": 1, "cluster_selection_method": "leaf", "allow_single_cluster": True, "cluster_selection_epsilon": 16.0},
    ),
    "epsilon-root-uses-rounded-descendant-stability": (
        csr_matrix([[7.0], [3.0], [6.0], [12.0], [-3.0], [11.0]]),
        {"min_cluster_size": 2, "min_samples": 1, "allow_single_cluster": True, "cluster_selection_epsilon": 4.000000000000001},
    ),
    "epsilon-max-preserves-equal-branches": (
        CASES["nested-parent-selected"], {"min_samples": 1, "cluster_selection_epsilon_max": 0.3125},
    ),
    "epsilon-max-measures-root-from-earliest-exit": (
        [[0], [0.125], [0.25], [0.5], [9]], {"allow_single_cluster": True, "cluster_selection_epsilon_max": 8.5},
    ),
    "max-cluster-size-splits-eom-parent": (
        CASES["nested-parent-selected"], {"max_cluster_size": 3},
    ),
    "epsilon-merges-past-size-limit": (
        CASES["nested-parent-selected"], {"max_cluster_size": 3, "cluster_selection_epsilon": 0.5},
    ),
    "leaf-ignores-eom-limits": (
        SPARSE_CASES["csr-tied-chain"][0],
        {"metric": "precomputed", "min_samples": 1, "cluster_selection_method": "leaf", "max_cluster_size": 1, "cluster_selection_epsilon_max": 0.0},
    ),
    "persistence-rounds-lambda-without-removing-splits": (
        CASES["nested-parent-selected"], {"min_samples": 1, "cluster_selection_persistence": 0.01},
    ),
    "persistence-threshold-rounds-to-equal-float32": (
        PERSISTENCE_PAIR, PERSISTENCE_PRECOMPUTED | {"cluster_selection_persistence": 1.0000000000000002},
    ),
    "persistence-merges-both-leaf-siblings": (
        PERSISTENCE_PAIR, PERSISTENCE_PRECOMPUTED | {"cluster_selection_persistence": 1.00000012, "allow_single_cluster": True},
    ),
    "persistence-retains-stronger-descendant-split": (
        PERSISTENCE_BRANCH, PERSISTENCE_PRECOMPUTED | {"cluster_selection_persistence": 1.0, "cluster_selection_method": "leaf"},
    ),
    "persistence-simplifies-before-epsilon-merges": (
        PERSISTENCE_BRANCH, PERSISTENCE_PRECOMPUTED | {"cluster_selection_persistence": 1.0, "cluster_selection_epsilon": 0.5, "allow_single_cluster": True},
    ),
}

NONFINITE_FEATURES = [
    [float("nan"), 1.0], [0.0, 0.0], [1.0, 0.0], [float("inf"), 0.0],
    [5.0, 0.0], [6.0, 0.0], [-float("inf"), 2.0], [10.0, 0.0], [11.0, 0.0], [0.0, float("nan")],
]


def scattered_missing_rows():
    missing = {0, 1, 3, 5, 9, 18, 33, 34, 35}
    return [[float("nan") if row in missing else float(row)] for row in range(38)]


def sparse_duplicates_with_missing():
    return csr_matrix((
        np.r_[METRIC_CSR.data[:3], float("nan"), METRIC_CSR.data[3:]],
        np.r_[METRIC_CSR.indices[:3], 1, METRIC_CSR.indices[3:]],
        [0, 3, 4, 6, 8, 12],
    ), shape=(5, 3))


MISSING_CASES = {
    "dense-features-with-nonfinite-rows": (NONFINITE_FEATURES, {"min_cluster_size": 2, "min_samples": 1}),
    "dense-two-retained-rows-before-excluded-tail": (
        [[float("nan")], [0.0], [float("inf")], [1.0], [-float("inf")]],
        {"min_cluster_size": 2, "min_samples": 10, "allow_single_cluster": True},
    ),
    "dense-nonfinite-row-order-after-set-growth": (scattered_missing_rows(), {"min_samples": 1}),
    "dense-cosine-with-nonfinite-rows-and-epsilon": (
        [[-1, -1], [-2, -2], [float("inf"), 0], [0, 0], [0, float("nan")], [1, 1], [2, 2]],
        {"metric": "cosine", "min_cluster_size": 2, "min_samples": 1, "cluster_selection_method": "leaf", "cluster_selection_epsilon": 0.125},
    ),
    "csr-euclidean-duplicates-and-nonfinite-row": (
        sparse_duplicates_with_missing(), {"min_cluster_size": 2, "min_samples": 1},
    ),
    "csr-float32-manhattan-nonfinite-and-empty-rows": (
        csr_matrix(NONFINITE_FEATURES, dtype=np.float32), {"metric": "manhattan", "min_cluster_size": 2, "min_samples": 1},
    ),
    "dense-precomputed-negative-infinity-keeps-its-distance": (
        [[0, 1, -float("inf"), 4], [1, 0, 3, 0], [-float("inf"), 3, 0, 1], [4, 0, 1, 0]],
        {"metric": "precomputed", "min_cluster_size": 2, "min_samples": 1},
    ),
}

POSTFIT_CASES = {
    "glosh-nested-local-and-global-outliers": (
        [[7.0], [8.0], [0.0], [1.0], [3.0], [20.0], [24.0], [100.0]],
        {"min_samples": 1, "min_cluster_size": 2},
    ),
    "dbscan-ordered-components": (
        [[7.0], [0.0], [1.0], [101.0], [4.0], [8.0], [5.0], [100.0]],
        {"min_samples": 1, "min_cluster_size": 2},
    ),
    "relative-validity-zero-sparseness-and-separation": (
        [[0.0], [0.0], [0.0], [0.0]],
        {"min_samples": 1, "min_cluster_size": 2, "allow_single_cluster": True},
    ),
    "relative-validity-negative-distances": (
        np.full((4, 4), -1.0),
        {"min_samples": 1, "min_cluster_size": 2, "allow_single_cluster": True, "metric": "precomputed"},
    ),
    "relative-validity-fallback-overflow": (
        [[0.0, 1.0, 1e308, 1e308], [1.0, 0.0, 1e308, 1e308],
         [1e308, 1e308, 0.0, 1.0], [1e308, 1e308, 1.0, 0.0]],
        {"min_samples": 1, "min_cluster_size": 2, "metric": "precomputed"},
    ),
}

BEST_FALLBACK = {"algorithm": "best", "approx_min_span_tree": True, "core_dist_n_jobs": 4}
PRIM_CASES = {
    "prims-euclidean-translated-float32": (
        np.float32(CASES["five-feature-rounding"]) + np.float32(2**16),
        {"algorithm": "prims_kdtree", "min_samples": 2, "leaf_size": 2},
    ),
    "prims-manhattan-ordered-sums": (
        rounding_case(17),
        {"algorithm": "prims_balltree", "metric": "manhattan", "alpha": 1.35, "min_samples": 2},
    ),
    "prims-259-tied-mst-edges": (
        tied_feature_rows(260),
        {"algorithm": "prims_kdtree", "cluster_selection_method": "leaf"},
    ),
    "prims-missing-feature-rows": (
        [[np.nan], [0.0], [1.0], [np.inf], [5.0], [6.0], [9.0]],
        {"algorithm": "prims_balltree", "min_cluster_size": 2, "min_samples": 1},
    ),
    "prims-two-finite-feature-rows": (
        [[1.0], [np.inf], [5.0], [np.nan]],
        {"algorithm": "prims_kdtree", "min_cluster_size": 2, "min_samples": 8},
    ),
    "prims-distance-just-below-float64-limit": (
        [[0.0], [float(np.nextafter(np.finfo(np.float64).max, 0.0))]],
        {"algorithm": "prims_kdtree", "metric": "manhattan", "min_samples": 1},
    ),
    "prims-distance-at-float64-limit": (
        [[0.0], [float(np.finfo(np.float64).max)]],
        {"algorithm": "prims_kdtree", "metric": "manhattan", "min_samples": 1},
    ),
    "prims-overflow-repeated-branch": (
        [[1e200], [0.0], [1.0]],
        {"algorithm": "prims_kdtree", "min_cluster_size": 2, "min_samples": 1, "allow_single_cluster": True},
    ),
    "prims-overflow-missing-rows": (
        [[0.0], [1.0], [2.0], [1e200], [np.nan]],
        {"algorithm": "prims_balltree", "min_cluster_size": 2, "min_samples": 1},
    ),
    "best-euclidean-61-features": (rounding_case(61), BEST_FALLBACK | {"alpha": 0.7}),
    "best-manhattan-61-features": (rounding_case(61), BEST_FALLBACK | {"metric": "manhattan"}),
    "best-sparse-features": (
        SPARSE_FEATURE_CASES["csr-features-with-empty-rows"][0],
        BEST_FALLBACK | {"metric": "cosine", "min_samples": 1},
    ),
    "best-dense-precomputed": (
        PRECOMPUTED_CASES["precomputed-finite"][0], BEST_FALLBACK | {"metric": "precomputed"},
    ),
    "best-sparse-precomputed": (
        SPARSE_CASES["csr-tied-chain"][0], BEST_FALLBACK | {"metric": "precomputed", "min_samples": 1},
    ),
}

BORUVKA_APPROXIMATION = {"metric": "manhattan", "min_samples": 1, "min_cluster_size": 2, "leaf_size": 3}
BORUVKA_CASES = {
    f"{algorithm}-{'approximate' if approximate else 'exact'}": (
        np.random.default_rng(seed).normal(size=(28, 3)),
        BORUVKA_APPROXIMATION | {"algorithm": algorithm, "approx_min_span_tree": approximate},
    )
    for algorithm, seed in (("boruvka_kdtree", 4), ("boruvka_balltree", 8))
    for approximate in (False, True)
} | {
    "best-euclidean-60-features": (rounding_case(60), BEST_FALLBACK),
    "boruvka-balltree-propagates-radius-bound-in-exact-mode": (
        np.random.default_rng(302).normal(size=(20, 1)),
        {"algorithm": "boruvka_balltree", "metric": "manhattan", "min_cluster_size": 2, "min_samples": 3, "leaf_size": 3},
    ),
    "boruvka-tied-manhattan-neighbors": (
        np.array([4, 2, -1, 0, 4, 0, -2, 1, 3, 0, 1, -4, -4, -2, 4, -2, -2, 4, -3, -3, -4, 4, 4, -3, -4, 4, -4], dtype=np.float64)[:, None],
        {"algorithm": "boruvka_kdtree", "metric": "manhattan", "min_samples": 3, "leaf_size": 3},
    ),
    "boruvka-float32-missing-rows": (
        np.float32([[1, 2], [np.nan, 1], [1, 3], [4, 5], [np.inf, 5], [4, 6], [9, 8]]),
        {"algorithm": "boruvka_balltree", "min_cluster_size": 2, "min_samples": 1},
    ),
}

PARAMETERS = {
    "min_cluster_size": 3,
    "min_samples": 2,
    "metric": "euclidean",
    "alpha": 1.0,
    "algorithm": "generic",
    "approx_min_span_tree": False,
    "gen_min_span_tree": True,
    "core_dist_n_jobs": 1,
    "cluster_selection_method": "eom",
    "allow_single_cluster": False,
    "cluster_selection_epsilon": 0.0,
    "cluster_selection_persistence": 0.0,
    "max_cluster_size": 0,
}


def record_array(values):
    record = {"dtype": values.dtype.name, "shape": list(values.shape)}
    if values.dtype in (np.dtype(np.float64), np.dtype(np.float32)):
        width = values.dtype.itemsize
        record["bits"] = [f"{bits:0{width * 2}x}" for bits in values.ravel().view(f"u{width}")]
    elif values.dtype.kind in "iu":
        record["values"] = values.tolist()
    else:
        raise TypeError(f"Unexpected output dtype: {values.dtype}")
    return record


def record_dbscan_cuts(estimator, linkage):
    distances = linkage[:, 2]
    finite = distances[np.isfinite(distances)]
    boundary = finite[len(finite) // 2] if len(finite) else np.float64(0.0)
    with np.errstate(under="ignore", over="ignore"):
        above_boundary = np.nextafter(boundary, np.inf)
    cuts = [(0.0, 1), (boundary, 2), (above_boundary, 2), (np.inf, 2), (np.inf, 5), (np.nan, 1)]
    return [
        {
            "cut_distance_bits": f"{np.float64(distance).view(np.uint64):016x}",
            "min_cluster_size": minimum,
            "labels": record_array(estimator.dbscan_clustering(distance, minimum)),
        }
        for distance, minimum in cuts
    ]


def record_input_value(value):
    return value if isinstance(value, str) or np.isfinite(value) else str(value)


def record_fit(rows, overrides):
    if isinstance(rows, csr_matrix):
        dtype = np.float32 if rows.dtype == np.dtype(np.float32) else np.float64
        data = csr_matrix((rows.data.astype(dtype), rows.indices.copy(), rows.indptr.copy()), shape=rows.shape)
        # Record the input buffers before SciPy performs any internal canonicalization.
        input_record = {
            "format": "csr",
            "dtype": data.dtype.name,
            "shape": list(data.shape),
            "indptr": data.indptr.tolist(),
            "indices": data.indices.tolist(),
            "values": [record_input_value(value) for value in data.data.tolist()],
        }
    else:
        data = np.array(rows, dtype=rows.dtype if isinstance(rows, np.ndarray) else np.float64)
        input_rows = rows.tolist() if isinstance(rows, np.ndarray) else rows
        input_record = {
            "dtype": data.dtype.name,
            "shape": list(data.shape),
            "values": [[record_input_value(value) for value in row] for row in input_rows],
        }
    parameters = PARAMETERS | overrides
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        estimator = HDBSCAN(**parameters).fit(data.copy())
        predicted = HDBSCAN(**parameters).fit_predict(data.copy())
        if parameters["metric"] == "precomputed":
            # The functional API returns the raw edges without needing feature vectors.
            mst = np.asarray(hdbscan(data, **parameters)[-1])
        else:
            mst = estimator.minimum_spanning_tree_.to_numpy()
        tree = estimator.condensed_tree_.to_numpy()
        linkage = estimator.single_linkage_tree_.to_numpy()
        outputs = {
            "labels_": record_array(estimator.labels_),
            "probabilities_": record_array(estimator.probabilities_),
            "cluster_persistence_": record_array(estimator.cluster_persistence_),
            "outlier_scores_": record_array(estimator.outlier_scores_),
            "relative_validity_": observe_validity(lambda: estimator.relative_validity_),
            "dbscan_cuts": record_dbscan_cuts(estimator, linkage),
            "minimum_spanning_tree_": record_array(mst),
            "single_linkage_tree_": record_array(linkage),
            "condensed_tree_": {
                "shape": list(tree.shape),
                "fields": [{"name": name, **record_array(tree[name])} for name in tree.dtype.names],
            },
        }

    np.testing.assert_array_equal(estimator.labels_, predicted, strict=True)
    return ({"parameter_overrides": overrides} if overrides else {}) | {
        "input": input_record,
        "outputs": outputs,
        "warnings": [
            {"category": warning.category.__name__, "message": str(warning.message)}
            for warning in observed
        ],
    }


def make_fixture():
    inputs = {name: (rows, {}) for name, rows in CASES.items()} | {
        name: (rows, {"metric": "precomputed"} | overrides)
        for name, (rows, overrides) in (PRECOMPUTED_CASES | SPARSE_CASES).items()
    } | SPARSE_FEATURE_CASES | {
        name: (rows, {"metric": metric, "min_cluster_size": 2, "min_samples": 1})
        for name, (rows, metric) in DISTANCE_METRIC_CASES.items()
    } | CLUSTER_SELECTION_CASES | LARGE_TIE_CASES | THRESHOLD_CASES | MISSING_CASES | POSTFIT_CASES | PRIM_CASES | BORUVKA_CASES
    with threadpool_limits(limits=1):
        cases = {name: record_fit(*args) for name, args in inputs.items()}
        for name, args in inputs.items():
            if cases[name] != record_fit(*args):
                raise RuntimeError(f"The reference output changed on repeated execution: {name}")

    return {
        "operation": "HDBSCAN.fit / HDBSCAN.fit_predict / HDBSCAN.outlier_scores_ / HDBSCAN.relative_validity_ / HDBSCAN.dbscan_clustering",
        "environment": record_environment(),
        "parameters": PARAMETERS,
        "cases": cases,
    }


def record_environment():
    machine = platform.machine().lower()
    return {
        "platform": platform.system(),
        "machine": "x86_64" if machine == "amd64" else machine,
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "threadpool_limit": 1,
        "numpy_cpu_dispatch": "X86_V3",
        "packages": {
            name: version(name)
            for name in ("hdbscan", "numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")
        },
    }


def prediction_cases():
    points = np.array([0., 0.1, 0.3, 1., 1.2, 10., 10.2, 10.6, 12., 18.])[:, None]
    queries = np.array([-1., 0., 0.05, 0.1, 0.4, 4., 10.35, 11., 17.])[:, None]
    rng = np.random.default_rng(4721)
    lattice = rng.integers(-4, 5, size=(101, 2)).astype(np.float64)
    lattice[:47] += 20
    fortran = rng.normal(size=(45, 9))
    fortran[:12] = fortran[0]
    fortran[20:37] += 12
    many_clusters = (np.arange(13)[:, None] * 8.0 + np.array([0., 0.25, 0.75])).reshape(-1, 1)
    plane = np.column_stack((points[:, 0], [0., -0.05, 0.1, 0.125, -0.3, 1., 1.2, 0.5, 1.1, 2.]))
    return {
        "new-and-training-points": (points, queries, {}),
        "minkowski-generic-p3": (
            plane,
            np.array([[0.1, -0.125], [1.0, 0.25], [10.25, 0.75], [17.0, 3.5]]),
            {"metric": "minkowski", "p": 3.5, "algorithm": "generic"},
        ),
        "chebyshev-tied-neighbors": (
            lattice,
            np.vstack((lattice[::11], [[5, 5], [12, 12], [23, 22]])),
            {"metric": "chebyshev", "min_samples": 9},
        ),
        "leaf-selection": (points, queries, {"min_samples": 1, "cluster_selection_method": "leaf"}),
        "epsilon-at-root": (points, queries, {"allow_single_cluster": True, "cluster_selection_epsilon": 9.0}),
        "no-selected-clusters": (points, queries, {"cluster_selection_epsilon_max": 0.1}),
        "no-branches": (points[:5], queries, {}),
        "single-cluster-with-no-branches": (points[:5], queries, {"allow_single_cluster": True}),
        "zero-distance": (np.repeat(points[[0, 5]], 5, axis=0), queries, {}),
        "equal-neighbors": (
            lattice,
            np.vstack((lattice[::7], [[5, 5], [12, 12], [23, 22]])).astype(np.float32),
            {"metric": "cityblock", "min_samples": 14, "algorithm": "prims_balltree"},
        ),
        "fortran-feature-matrix": (
            np.asfortranarray(fortran),
            np.vstack((fortran[::7], fortran[::6] + 0.15)),
            {"algorithm": "generic", "allow_single_cluster": True, "min_samples": 8},
        ),
        "reversed-feature-columns": (
            fortran[:, ::-1],
            np.vstack((fortran[::7], fortran[::6] + 0.15))[:, ::-1],
            {"algorithm": "generic", "allow_single_cluster": True, "min_samples": 8},
        ),
        "mixed-infinite-densities": (
            np.concatenate((np.repeat(points[[0]], 5, axis=0), points[4:])),
            queries,
            {"min_samples": 1},
        ),
        "far-away-queries": (points, np.array([[1e200], [-1e200]]), {}),
        "column-major-queries": (
            plane,
            np.asfortranarray([[5., 0.1], [0.15, -0.12], [11., 2.], [15., -2.], [1.15, 0.3]]),
            {"algorithm": "prims_kdtree"},
        ),
        "many-clusters": (
            many_clusters,
            np.array([0., 0.4, 4., 33.7, 65.3, 100.])[:, None],
            {"algorithm": "prims_kdtree", "min_samples": 1},
        ),
    }


def record_prediction(data, queries, overrides, parameters):
    def matrix_record(values):
        if values.strides[1] < 0:
            layout = "reversed"
        elif values.flags.f_contiguous and not values.flags.c_contiguous:
            layout = "fortran"
        else:
            layout = "row-major"
        return {"layout": layout, **record_array(values)}

    model = HDBSCAN(**(parameters | overrides)).fit(data)
    cache = model.prediction_data_
    condensed = model.condensed_tree_.to_numpy()
    outputs = {}
    for name, call in (
        ("predict", lambda: approximate_predict(model, queries, return_connecting_points=True)),
        ("scores", lambda: approximate_predict_scores(model, queries)),
    ):
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            arrays = call()
        outputs[name] = {
            "arrays": [record_array(values) for values in (arrays if isinstance(arrays, tuple) else (arrays,))],
            "warnings": [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in observed
            ],
        }
    membership = {}
    for name, call in (
        ("new", lambda: membership_vector(model, queries)),
        ("training", lambda: all_points_membership_vectors(model)),
    ):
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            try:
                membership[name] = {"array": record_array(call())}
            except (KeyError, ValueError) as error:
                membership[name] = {"error": {"type": type(error).__name__, "args": list(error.args)}}
        membership[name]["warnings"] = [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in observed
        ]
    return {
        "parameter_overrides": overrides,
        "input": matrix_record(data),
        "queries": matrix_record(queries),
        "labels": record_array(model.labels_),
        "probabilities": record_array(model.probabilities_),
        "core_distances": record_array(cache.core_distances),
        "exemplars": [record_array(exemplar) for exemplar in model.exemplars_],
        "cluster_map": sorted([int(node), int(label)] for node, label in cache.cluster_map.items()),
        "condensed": {name: record_array(condensed[name]) for name in condensed.dtype.names},
        "outputs": outputs,
        "membership": membership,
    }


def make_prediction_fixture():
    parameters = {"min_cluster_size": 2, "min_samples": 2, "prediction_data": True, "core_dist_n_jobs": 1}
    with threadpool_limits(limits=1):
        cases = {
            name: record_prediction(*args, parameters) for name, args in prediction_cases().items()
        }
        for name, args in prediction_cases().items():
            if cases[name] != record_prediction(*args, parameters):
                raise RuntimeError(f"The reference prediction changed on repeated execution: {name}")
    return {
        "operation": "approximate_predict / approximate_predict_scores / membership_vector / all_points_membership_vectors",
        "environment": record_environment(),
        "parameters": parameters,
        "cases": cases,
    }


def record_major_output(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [record_major_output(item) for item in value]
    array = np.asarray(value)
    if array.dtype.names is not None:
        return {"fields": {name: record_array(array[name]) for name in array.dtype.names}}
    return record_array(array)


def observe_major(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        try:
            output = {"result": record_major_output(function(*args, **kwargs))}
        except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
            output = {"error": {"type": type(error).__name__, "message": str(error)}}
    return {
        **output,
        "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in observed],
    }


def major_branch_calls(missing):
    empty_graph = np.array([0., .25, 12.1, 12.3, 24., 24.25])[:, None]
    yield "empty-forest", empty_graph, {"algorithm": "generic"}, {}, None, None
    labels = np.array([0, 0, 0, 2, 1, 1, 1, -1], dtype=np.int8)
    yield "cluster-only-in-missing-rows", missing, {}, {}, labels, None
    yield "strengths-for-finite-rows", missing, {}, {}, None, np.ones(len(missing) + 1, dtype=np.float32)

    incomplete = np.array([
        [6.167, .584, 1.075, -.086, .170, -.632],
        [6.005, 1.028, .207, .845, .176, 1.989],
        [6.165, -.722, -.777, 1.276, 1.710, 2.069],
        [5.888, .693, -1.219, -2.007, .148, np.nan],
        [5.974, -.695, .095, 1.135, -.441, 1.274],
        [6.078, .112, -.284, .074, -.903, 1.429],
        [-.078, .575, .114, -.494, .227, -1.508],
        [-.060, -.147, .496, .510, -1.085, 3.237],
        [.043, -.029, 1.423, -2.210, -3.115, 1.460],
        [-.143, .515, .112, 1.215, .650, -1.720],
        [.106, 1.000, .052, -.571, -1.195, 3.098],
        [-.109, .075, -.084, .661, .907, -.091],
        [-.095, -.028, -2.410, -2.478, 2.065, -.772],
    ])
    minkowski = {"metric": "minkowski", "algorithm": "generic", "p": 64.0, "min_samples": 2}
    yield "minkowski-incomplete-forest", incomplete, minkowski, {}, None, None
    yield "minkowski-incomplete-forest-branch-labels", incomplete, minkowski, {
        "label_sides_as_branches": True,
    }, None, None

    rng = np.random.default_rng(3018)
    for i in range(11):
        points = rng.normal(size=(8 + i, 1 + i % 5))
        points[::2, 0] += 5
        weights = rng.uniform(.1, .95, len(points))
        if i not in (4, 10):
            continue
        points = points[:, ::-1]
        model = {"min_samples": 2, "min_cluster_size": 2 + i % 4, "algorithm": "prims_kdtree"}
        options = {"branch_detection_method": "core", "label_sides_as_branches": True}
        if i == 4:
            yield "float32-strengths", points, model, options, None, weights.astype(np.float32)
            yield "integer-strengths", points, model, options, None, (weights * 5).astype(np.uint8)
        else:
            points[2, -1] = np.nan
            yield "float32-strengths-missing", points, model, options, None, weights.astype(np.float32)


def functional_hdbscan_cases():
    line = np.array([0., 1., 2., 10., 11., 12., 30.])[:, None]
    ties = np.array([0., 0., 0., 1., 2., 2., 3., 10., 10., 10., 12., 12., 12.])[:, None]
    distances = np.abs(line - line.T)
    features = np.column_stack((line, (line % 3) / 4))
    missing = np.insert(ties, 3, np.nan, axis=0)
    yield "defaults", line, {}, False
    yield "generic-minkowski", features, {"algorithm": "generic", "p": 3.5}, False
    yield "generic-euclidean-squared", features, {"algorithm": "generic", "metric": "euclidean", "squared": True}, False
    yield "generic-unknown-metric-kwarg", features, {"algorithm": "generic", "metric": "euclidean", "unknown": 1}, False
    nan_euclidean = {"algorithm": "generic", "metric": "nan_euclidean"}
    yield "nan-euclidean-disconnected", np.insert(features[:2], 1, np.nan, axis=0), nan_euclidean, False
    yield "precomputed", distances, {"metric": "precomputed"}, False
    yield "precomputed-csr", distances, {"metric": "precomputed"}, True
    yield "precomputed-csr-without-mst", distances, {"metric": "precomputed", "gen_min_span_tree": False}, True
    yield "precomputed-infinite-edge", np.array([[0., 1., np.inf], [1., 0., np.inf], [np.inf, np.inf, 0.]]), {
        "metric": "precomputed", "min_samples": 1,
    }, False
    yield "precomputed-float32", distances.astype(np.float32), {"metric": "precomputed"}, False
    yield "euclidean-csr", features, {"metric": "euclidean"}, True
    yield "nonfinite-features", missing, {"metric": "euclidean"}, False
    yield "nonfinite-features-csr", missing, {"metric": "euclidean"}, True
    yield "one-point", line[:1], {}, False
    yield "unknown-algorithm", line, {"algorithm": "invalid"}, False
    yield "invalid-min-samples", line, {"min_samples": 0}, False
    yield "invalid-selection", line, {"cluster_selection_method": "invalid"}, False
    yield "invalid-memory", line, {"memory": None}, False
    strict = {"metric": "euclidean", "match_reference_implementation": True}
    for algorithm in ("generic", "best", "prims_kdtree", "prims_balltree", "boruvka_kdtree", "boruvka_balltree"):
        yield f"strict-{algorithm}", ties, {**strict, "algorithm": algorithm, "min_samples": None}, False
    yield "strict-leaf", ties, {**strict, "min_samples": 4, "cluster_selection_method": "leaf"}, False
    yield "strict-root", ties[:7], {**strict, "allow_single_cluster": True}, False
    yield "strict-epsilon", ties, {**strict, "cluster_selection_epsilon": 4.0}, False
    yield "strict-persistence", features, {**strict, "cluster_selection_persistence": 0.5}, False
    yield "strict-csr", ties, strict, True


def dense_precomputed_estimator_cases():
    isolated = np.array([[0., 1., np.inf], [1., 0., np.inf], [np.inf, np.inf, 0.]])
    chain = np.array([[0., 1., np.inf], [1., 0., 3.], [np.inf, 3., 0.]])
    larger = np.array([[0., 1., 4., np.inf], [1., 0., 3., np.inf],
                       [4., 3., 0., np.inf], [np.inf, np.inf, np.inf, 0.]])
    yield "isolated-point", isolated, {"min_samples": 1, "gen_min_span_tree": True}
    yield "isolated-point-default-neighborhood", isolated, {}
    yield "reachable-chain-insufficient-neighbors", chain, {"min_samples": 2}
    yield "unreachable-generic-leaf", larger, {"min_cluster_size": 3, "min_samples": 1, "algorithm": "generic",
                                              "cluster_selection_method": "leaf", "gen_min_span_tree": True}
    yield "two-unreachable-points", isolated[np.ix_([0, 2], [0, 2])], {
        "allow_single_cluster": True, "gen_min_span_tree": True,
    }


def record_dense_precomputed_estimator(data, options):
    estimator = HDBSCAN(**options)

    def fit():
        model = estimator.fit(data.copy())
        return (
            model.labels_, model.probabilities_, model.cluster_persistence_,
            model.single_linkage_tree_.to_numpy(), model.condensed_tree_.to_numpy(),
        )

    return {
        "input": record_array(data),
        "options": options,
        "fit": observe_major(fit),
        "fit_predict": observe_major(estimator.fit_predict, np.asfortranarray(data)),
        "outlier_scores": observe_major(lambda: estimator.outlier_scores_),
        "minimum_spanning_tree": observe_major(lambda: estimator.minimum_spanning_tree_),
        "relative_validity": observe_major(lambda: estimator.relative_validity_),
        "dbscan": observe_major(estimator.dbscan_clustering, np.inf, 2),
    }


def make_major_fixture():
    line = np.array([0., 1., 3., 9., 11., 13., 48.])[:, None]
    repeats = np.repeat(np.arange(7, dtype=np.float64), 3)[:, None]
    rounding = np.random.default_rng(765276).normal(size=(8, 5)) + (np.arange(8) % 3)[:, None] * 1.7
    robust_cases = {
        "generic": (line, {"algorithm": "generic", "k": 2, "cut": 6.1}),
        "precomputed": (np.abs(line - line.T), {"metric": "precomputed", "k": 2}),
        "prim-kdtree": (line, {"algorithm": "prims_kdtree", "k": 3, "alpha": 2.0}),
        "prim-balltree": (rounding, {"algorithm": "prims_balltree", "metric": "manhattan", "k": 1}),
        "boruvka-kdtree": (repeats, {"algorithm": "boruvka_kdtree", "leaf_size": 5, "k": 30}),
        "boruvka-balltree": (rounding, {"algorithm": "boruvka_balltree", "metric": "manhattan", "k": 1, "alpha": 3.5}),
        "minkowski-boruvka": (rounding, {"algorithm": "boruvka_kdtree", "metric": "minkowski", "p": 3.5, "k": 2}),
        "chebyshev-prim": (rounding, {"algorithm": "prims_balltree", "metric": "chebyshev", "k": 2}),
        "estimator": (repeats, {"k": 4, "gamma": 2, "cut": 1.01}),
    }
    rng = np.random.default_rng(753)
    t = np.linspace(.05, 3., 50)
    directions = np.array([[1., 0.], [-.5, np.sqrt(3.) / 2.], [-.5, -np.sqrt(3.) / 2.]])
    flare = np.vstack([t[:, None] * ray + rng.normal(0., .04, (len(t), 2)) for ray in directions] +
                      [rng.normal((9., 0.), .2, (35, 2))])
    missing = np.insert(line, 3, np.nan, axis=0)
    medoid = np.array([
        1.2903732423606247, -.9120432513566515, .678707246881868, 1.0047532259409178,
        -.49607149865325245, .8991277117343575, -1.057596714191959, 1.2617903504413004,
        .04839449930908703, 20.48125928444482, 20.55930299042778, 18.337764616451413,
        18.523768606263786, 19.77222698301318, 18.82780829999409, 20.637714494601184,
        18.19592591802272, 18.798403087572435,
    ])[:, None]
    metric_points = np.vstack((flare[::4], [[23., 4.], [24., 7.]]))
    metric_queries = [[0.15, 0.25], [-1., 1.73], [9., 0.], [23.5, 7.0]]
    branch_cases = {
        "full": (flare, {}, {"min_cluster_size": 6}, [[0., 0.], [-1., 1.73], [9., 0.], [14., 6.]]),
        "core": (line, {}, {"branch_detection_method": "core"}, [[1.], [11.], [48.]]),
        "minkowski-core": (
            metric_points, {"metric": "minkowski", "p": 3.5, "algorithm": "generic", "min_samples": 2},
            {"branch_detection_method": "core", "min_cluster_size": 4, "label_sides_as_branches": True},
            metric_queries,
        ),
        "chebyshev-full": (
            metric_points, {"metric": "chebyshev", "min_samples": 2},
            {"branch_detection_method": "full", "min_cluster_size": 4, "label_sides_as_branches": True},
            metric_queries,
        ),
        "missing-and-overridden": (missing, {"prediction_data": False}, {}, None),
        "medoid-tie": (medoid, {"min_samples": 2, "min_cluster_size": 4}, {"branch_detection_method": "core"}, [[20.5]]),
    }
    spatial_cases = {
        "minkowski-prim": (rounding, {"algorithm": "prims_kdtree", "metric": "minkowski", "p": 3.5}),
        "minkowski-generic": (rounding, {"algorithm": "generic", "metric": "minkowski", "p": 3.5}),
        "chebyshev-kdtree": (flare[:26], {"algorithm": "boruvka_kdtree", "metric": "chebyshev", "leaf_size": 4}),
        "seuclidean-balltree": (rounding, {"algorithm": "boruvka_balltree", "metric": "seuclidean", "V": [1., 2., 3., 4., 5.]}),
        "canberra-best": (rounding, {"metric": "canberra"}),
        "haversine-balltree": (rounding[:, :2] / 4, {"algorithm": "prims_balltree", "metric": "haversine"}),
    }
    robust, branches, branch_calls, spatial_fits, functional_fits = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for name, (data, options) in robust_cases.items():
            options = {"core_dist_n_jobs": 1, **options}
            if name == "estimator":
                def fit_estimator():
                    model = RobustSingleLinkage(**options)
                    labels = model.fit_predict(data)
                    return labels, model.cluster_hierarchy_.to_numpy(), model.cluster_hierarchy_.get_clusters(1.1, 2)
                output = observe_major(fit_estimator)
            else:
                output = observe_major(robust_single_linkage, data, **{"cut": 2.5, "gamma": 1, **options})
            robust[name] = {"input": record_array(data), "options": options, "output": output}
        for name, (data, model_options, options, queries) in branch_cases.items():
            model_options = {"min_samples": 1, "min_cluster_size": 6 if name == "full" else 2,
                             "prediction_data": True, "branch_detection_data": True, "core_dist_n_jobs": 1,
                             **model_options}
            clusterer = HDBSCAN(**model_options).fit(data)
            labels = np.zeros(len(data), dtype=np.intp) if name == "missing-and-overridden" else None
            cache = clusterer.branch_detection_data_
            output = observe_major(detect_branches_in_clusters, clusterer, cluster_labels=labels, **options)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                detector = BranchDetector(**options).fit(clusterer, labels=labels)
            record = {
                "input": record_array(data), "model": model_options, "options": options, "output": output,
                "labels": None if labels is None else labels.tolist(),
                "cache": record_major_output((cache.finite_index, cache.neighbors, cache.core_distances)),
                "graph": record_major_output(detector.approximation_graph_.to_numpy()),
            }
            if name == "full":
                plot_options = {"log_size": True, "max_rectangle_per_icicle": 3}
                geometry = clusterer.condensed_tree_.get_plot_data(**plot_options)
                record["condensed_plot"] = {
                    "options": plot_options,
                    "arrays": {key: record_array(np.asarray(values, dtype=np.float64))
                               for key, values in geometry.items() if key != "cluster_bounds"},
                    "bounds": {str(node): record_array(np.asarray(bounds, dtype=np.float64))
                               for node, bounds in geometry["cluster_bounds"].items()},
                }
            if queries is not None:
                queries = np.asarray(queries, dtype=np.float64)
                record["queries"] = record_array(queries)
                record["prediction"] = observe_major(approximate_predict_branch, detector, queries)
                record["exemplars"] = record_major_output(detector.exemplars_)
                record["representatives"] = {
                    str(label): record_major_output((detector.weighted_centroid(label), detector.weighted_medoid(label)))
                    for label in np.unique(detector.labels_[detector.labels_ >= 0])
                }
            branches[name] = record
        for name, data, model, options, labels, probabilities in major_branch_calls(missing):
            model = {"min_samples": 1, "min_cluster_size": 2, "branch_detection_data": True,
                     "core_dist_n_jobs": 1, **model}
            clusterer = HDBSCAN(**model).fit(data)
            branch_calls[name] = {
                "input": record_array(data), "model": model, "options": options,
                "labels": None if labels is None else record_array(labels),
                "probabilities": None if probabilities is None else record_array(probabilities),
                "output": observe_major(detect_branches_in_clusters, clusterer, labels, probabilities, **options),
            }
        for name, (data, options) in spatial_cases.items():
            options = {"min_samples": 2, "min_cluster_size": 2, "gen_min_span_tree": True,
                       "core_dist_n_jobs": 1, **options}
            def fit_spatial():
                model = HDBSCAN(**options).fit(data)
                return (
                    model.labels_, model.probabilities_, model.cluster_persistence_,
                    model.minimum_spanning_tree_.to_numpy(), model.single_linkage_tree_.to_numpy(),
                    model.condensed_tree_.to_numpy(),
                )
            spatial_fits[name] = {"input": record_array(data), "options": options, "output": observe_major(fit_spatial)}
        for name, points, options, sparse in functional_hdbscan_cases():
            options = {"min_samples": 1, "min_cluster_size": 2, "gen_min_span_tree": True,
                       "core_dist_n_jobs": 1, **options}
            output = observe_major(hdbscan, csr_matrix(points) if sparse else points, **options)
            functional_fits[name] = {"input": record_array(points), "options": options,
                                     "sparse": sparse, "output": output}
        precomputed_estimators = {
            name: record_dense_precomputed_estimator(data, {"metric": "precomputed", "min_cluster_size": 2, **options})
            for name, data, options in dense_precomputed_estimator_cases()
        }
    return {"operation": "robust single linkage, spatial distances, branch detection and functional HDBSCAN",
            "environment": record_environment(), "robust": robust, "branches": branches,
            "branch_calls": branch_calls, "spatial_fits": spatial_fits, "functional_fits": functional_fits,
            "precomputed_estimators": precomputed_estimators}


def observe_validity(function, *args, **kwargs):
    def output(value):
        if isinstance(value, tuple):
            return [output(item) for item in value]
        if isinstance(value, list):
            return {"list": value}
        return None if value is None else record_array(np.asarray(value))

    log = io.StringIO()
    with warnings.catch_warnings(record=True) as observed, contextlib.redirect_stdout(log):
        warnings.simplefilter("always")
        try:
            record = {"result": output(function(*args, **kwargs))}
        except (AssertionError, AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
            record = {"error": {"type": type(error).__name__, "message": str(error)}}
    return {
        **record,
        "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in observed],
        "stdout": log.getvalue(),
    }


def validity_cases():
    labels = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3, -1], dtype=np.int64)
    line = np.array([0, 1, 2, 5, 6, 8, 12, 15, 17, 27], dtype=np.float64)[:, None]
    roundoff = np.array(rounding_case(8))
    wide_labels = np.repeat([0, 1, 2], 4).astype(np.int64)
    precomputed = np.abs(line - line.T)
    duplicates = np.array([[0., 0.], [0., 0.], [0., 0.], [1., 1.], [1., 1.], [1., 1.]])
    cosine_zero = np.array([[0., 0.], [2., 1.], [0., 0.], [-5., 3.], [-3., 6.], [-4., 9.]])
    infinite = precomputed.copy()
    infinite[0, 1] = infinite[1, 0] = np.inf
    near_nan = precomputed.copy()
    near_nan[0, 2] = near_nan[2, 0] = np.nan
    two_groups = np.repeat([0, 1], 3).astype(np.int64)
    fraction = {"metric": "precomputed", "d": 1.5}
    return {
        "gapped-labels-and-noise": (line, labels, {"verbose": True}),
        "raw-tree": (line, labels, {"mst_raw_dist": True}),
        "duplicates": (duplicates, two_groups, {"verbose": True}),
        "all-noise": (line, np.full(len(line), -1, dtype=np.int64), {}),
        "single-cluster": (line, np.zeros(len(line), dtype=np.int64), {}),
        "singleton": (line, np.array([0] * 9 + [1], dtype=np.int64), {}),
        "empty": (line[:0], labels[:0], {}),
        "cityblock": (roundoff, wide_labels, {"metric": "cityblock"}),
        "cosine": (roundoff, wide_labels, {"metric": "cosine"}),
        "zero-vector-cosine": (cosine_zero, two_groups, {"metric": "cosine"}),
        "zero-vector-raw-cosine": (cosine_zero, two_groups, {"metric": "cosine", "mst_raw_dist": True}),
        "precomputed": (precomputed, labels, {"metric": "precomputed", "d": 1}),
        "fractional-dimension": (precomputed, labels, fraction),
        "negative-dimension": (precomputed, labels, {**fraction, "d": -2.5}),
        "zero-dimension": (precomputed, labels, {**fraction, "d": 0}),
        "precomputed-raw-layout-error": (precomputed, labels, {**fraction, "mst_raw_dist": True}),
        "precomputed-missing-dimension": (precomputed, labels, {"metric": "precomputed"}),
        "precomputed-negative-distances": (-precomputed, labels, {**fraction, "d": 2}),
        "precomputed-asymmetric": (precomputed + np.triu(precomputed), labels, fraction),
        "verbose-infinite": (infinite, labels, {**fraction, "d": -1.5, "verbose": True}),
        "verbose-partial-nan": (near_nan, labels, {**fraction, "verbose": True}),
        "column-major": (np.asfortranarray(roundoff), wide_labels, {}),
        "reversed-features": (roundoff[:, ::-1], wide_labels, {}),
        "float32": (roundoff.astype(np.float32), wide_labels, {}),
        "float32-cityblock": (roundoff.astype(np.float32), wide_labels, {"metric": "cityblock"}),
        "float32-seuclidean": (roundoff.astype(np.float32), wide_labels, {"metric": "seuclidean"}),
        "weighted-minkowski": (roundoff, wide_labels, {"metric": "minkowski", "p": 1.8, "w": list(range(1, 9))}),
        "worker-layout": (roundoff, wide_labels, {"metric": "cosine", "n_jobs": 2, "mst_raw_dist": True}),
        "boolean-conversion": (duplicates, two_groups, {"metric": "jaccard", "mst_raw_dist": True}),
        "nan-in-noise": (np.where(labels[:, None] < 0, np.nan, line), labels, {}),
    }


def make_validity_fixture():
    def record_case(X, labels, options):
        layout = "reversed" if X.strides[1] < 0 else "fortran" if not X.flags.c_contiguous else "row-major"
        observation = observe_validity(validity_index, X, labels, per_cluster_scores=True, **options)
        missing = []
        if "result" in observation:
            # Slots for unused labels vary across calls to the public API.
            scores = observation["result"][1]
            missing = sorted(set(range(scores["shape"][0])) - set(labels.tolist()))
            for label in missing:
                scores["bits"][label] = None
        return {
            "input": {**record_array(X), "layout": layout},
            "labels": record_array(labels),
            "options": options,
            "missing_labels": missing,
            **observation,
        }

    line = np.array([[0.], [1.], [2.], [5.], [6.], [8.]])
    labels = np.repeat([0, 1], 3)
    symmetric = cdist(line, line)
    path = symmetric[:4, :4].copy()
    path[0, 2] = path[2, 0] = 1.000001
    tree = np.array([[0., 6., 4., 5.], [6., 0., 2., 9.], [4., 2., 0., 3.], [5., 9., 3., 0.]])
    asymmetric = np.array([[0., 1., 9.], [2., 0., 1.], [3., 6., 0.]])
    all_core = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
    coordinates = np.array([[0.831, -0.24], [-1.01, 0.725], [1.14, 2.5]], dtype=np.float32)
    supplied_norms = np.zeros(3, dtype=np.float32)
    with threadpool_limits(limits=1):
        cases = {name: record_case(*args) for name, args in validity_cases().items()}
        for name, args in validity_cases().items():
            if cases[name] != record_case(*args):
                raise RuntimeError(f"The reference validity score changed on repeated execution: {name}")
        helpers = {
            "core": observe_validity(all_points_core_distance, all_core.copy(), 1.25),
            "cluster-matrix": observe_validity(distances_between_points, line, labels, 1),
            "precomputed-matrix": observe_validity(distances_between_points, symmetric, labels, 1, "precomputed", 1.0),
            "float32-haversine": observe_validity(distances_between_points, coordinates, labels[:3], 0, "haversine"),
            "float32-callable": observe_validity(distances_between_points, coordinates, labels[:3], 0,
                                                 lambda left, right: np.abs(left - right).sum()),
            "float32-euclidean-norms": observe_validity(distances_between_points, coordinates, labels[:3], 0,
                                                        X_norm_squared=supplied_norms, Y_norm_squared=supplied_norms),
            "worker-cosine": observe_validity(distances_between_points, coordinates, labels[:3], 0, "cosine", n_jobs=2),
            "tree-tie": observe_validity(internal_minimum_spanning_tree, path),
            "tree-internal-edge": observe_validity(internal_minimum_spanning_tree, tree),
            "tree-two-vertices": observe_validity(internal_minimum_spanning_tree, symmetric[:2, :2].copy()),
            "tree-asymmetric": observe_validity(internal_minimum_spanning_tree, asymmetric),
            "tree-fortran": observe_validity(internal_minimum_spanning_tree, np.asfortranarray(tree)),
            "separation": observe_validity(density_separation, line, labels, 0, 1, [1, 2], [0, 1],
                                           np.array([1., 2., 5.]), np.array([9., 3., 4.])),
        }
    return {"operation": "validity_index and density validity helpers", "environment": record_environment(),
            "cases": cases, "helpers": helpers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--output", type=Path)
    action.add_argument("--check", type=Path)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--prediction", action="store_true", help="record predictions on unseen points")
    operation.add_argument("--validity", action="store_true", help="record density validity scores")
    operation.add_argument("--major", action="store_true", help="record robust linkage and cluster branches")
    args = parser.parse_args()

    if args.major:
        fixture = make_major_fixture()
    elif args.validity:
        fixture = make_validity_fixture()
    elif args.prediction:
        fixture = make_prediction_fixture()
    else:
        fixture = make_fixture()
    if args.check:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        if fixture != expected:
            parser.exit(1, f"Reference output differs from {args.check}\n")
        print(f"Reference output matches {args.check}")
        return

    document = json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8", newline="\n")
    else:
        print(document, end="")


if __name__ == "__main__":
    main()
