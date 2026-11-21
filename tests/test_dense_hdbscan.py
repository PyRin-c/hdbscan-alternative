import json
import pickle
import platform
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array, csr_matrix

from hdbscan_alternative import (
    BranchDetector,
    HDBSCAN,
    RobustSingleLinkage,
    all_points_membership_vectors,
    approximate_predict,
    approximate_predict_branch,
    approximate_predict_scores,
    detect_branches_in_clusters,
    hdbscan,
    membership_vector,
    robust_single_linkage,
    validity_index,
)
from hdbscan_alternative.validity import (
    all_points_core_distance,
    density_separation,
    distances_between_points,
    internal_minimum_spanning_tree,
)


def _load_reference(prefix="hdbscan"):
    os_name = "windows" if platform.system() == "Windows" else "linux"
    fixtures = Path(__file__).resolve().parents[1] / "reference" / "fixtures"
    return json.loads((fixtures / f"{prefix}-0.8.44-{os_name}-x86_64.json").read_text())


REFERENCE = _load_reference()
PREDICTION_REFERENCE = _load_reference("prediction")
VALIDITY_REFERENCE = _load_reference("validity")
MAJOR_REFERENCE = _load_reference("major")


def _assert_array(actual, record):
    assert actual.shape == tuple(record["shape"])
    assert actual.dtype.name == record["dtype"]
    if "bits" in record:
        encoding = np.dtype(f"u{actual.dtype.itemsize}")
        bits = np.array([int(value, 16) for value in record["bits"]], dtype=encoding)
        np.testing.assert_array_equal(actual.view(encoding).ravel(), bits)
    else:
        np.testing.assert_array_equal(actual, record["values"])


def _assert_major_output(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, list):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected)
        for item, record in zip(actual, expected):
            _assert_major_output(item, record)
    elif "fields" in expected:
        assert actual.dtype.names == tuple(expected["fields"])
        for name, record in expected["fields"].items():
            _assert_array(actual[name], record)
    else:
        _assert_array(np.asarray(actual), expected)


def _check_major_output(call, expected):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(Exception) as caught:
                call()
            assert type(caught.value).__name__ == expected["error"]["type"]
            assert str(caught.value) == expected["error"]["message"]
        else:
            _assert_major_output(call(), expected["result"])
    assert [{"category": w.category.__name__, "message": str(w.message)} for w in observed] == expected["warnings"]


@pytest.mark.parametrize("name", MAJOR_REFERENCE["functional_fits"])
def test_hdbscan_function_matches_reference(name):
    case = MAJOR_REFERENCE["functional_fits"][name]
    points = _matrix_from_bits({"layout": "row-major", **case["input"]})
    if case["sparse"]:
        points = csr_matrix(points)
    options = case["options"].copy()
    min_cluster_size, min_samples = options.pop("min_cluster_size"), options.pop("min_samples")
    _check_major_output(lambda: hdbscan(points, min_cluster_size, min_samples, **options), case["output"])


def test_reference_implementation_flag_in_estimator(tmp_path):
    from sklearn.base import clone

    case = MAJOR_REFERENCE["functional_fits"]["strict-generic"]
    points = _matrix_from_bits({"layout": "row-major", **case["input"]})
    expected = case["output"]["result"]
    model = clone(HDBSCAN(memory=str(tmp_path), **case["options"]))
    assert model.get_params()["match_reference_implementation"] is True
    for _ in range(2):
        model.fit(points)
        _assert_major_output((model.labels_, model.probabilities_, model.cluster_persistence_,
                              model.condensed_tree_.to_numpy(), model.single_linkage_tree_.to_numpy(),
                              model.minimum_spanning_tree_.to_numpy()), expected)


def test_function_checks_disconnected_condensed_tree_capacity():
    points = [[0., 0.], [1., 2.], [np.nan, np.nan], [2., 1.], [5., 1.], [5., 2.], [6., 3.], [9., 9.]]
    with pytest.raises(ValueError, match=r"could not broadcast input array from shape \(16,\) into shape \(20,\)"):
        hdbscan(points, 2, 1, algorithm="generic", metric="nan_euclidean")


@pytest.mark.parametrize("name", MAJOR_REFERENCE["precomputed_estimators"])
def test_dense_precomputed_missing_distances_match_reference(name):
    case = MAJOR_REFERENCE["precomputed_estimators"][name]
    distances = _matrix_from_bits({"layout": "row-major", **case["input"]})
    original = distances.copy()
    model = HDBSCAN(**case["options"])

    def fit():
        assert model.fit(distances) is model
        return (
            model.labels_, model.probabilities_, model.cluster_persistence_,
            model.single_linkage_tree_.to_numpy(), model.condensed_tree_.to_numpy(),
        )

    _check_major_output(fit, case["fit"])
    _check_major_output(lambda: model.fit_predict(np.asfortranarray(distances)), case["fit_predict"])
    _check_major_output(lambda: model.outlier_scores_, case["outlier_scores"])
    _check_major_output(lambda: model.minimum_spanning_tree_, case["minimum_spanning_tree"])
    _check_major_output(lambda: model.relative_validity_, case["relative_validity"])
    _check_major_output(lambda: model.dbscan_clustering(np.inf, 2), case["dbscan"])
    np.testing.assert_array_equal(distances, original, strict=True)


@pytest.mark.parametrize("functional", [False, True], ids=["estimator", "function"])
def test_cached_fit_uses_current_numpy_error_policy(tmp_path, functional):
    case = MAJOR_REFERENCE["precomputed_estimators"]["isolated-point"]
    distances = _matrix_from_bits({"layout": "row-major", **case["input"]})
    options = {**case["options"], "memory": str(tmp_path)}
    model = HDBSCAN(**options)

    def fit():
        if functional:
            return hdbscan(distances, **options)[0]
        return model.fit_predict(distances)

    with np.errstate(over="ignore"):
        _check_major_output(fit, {"result": case["fit"]["result"][0], "warnings": []})
    with np.errstate(over="warn"):
        _check_major_output(
            fit, {"result": case["fit"]["result"][0], "warnings": case["fit"]["warnings"]},
        )
    old_labels = getattr(model, "labels_", None)
    with np.errstate(over="raise"), pytest.raises(FloatingPointError, match="overflow encountered in scalar divide"):
        fit()
    assert getattr(model, "labels_", None) is old_labels


@pytest.mark.parametrize("name", MAJOR_REFERENCE["robust"])
def test_robust_single_linkage_reference(name):
    case = MAJOR_REFERENCE["robust"][name]
    points = _matrix_from_bits({"layout": "row-major", **case["input"]})
    if name == "estimator":
        def fit():
            model = RobustSingleLinkage(**case["options"])
            labels = model.fit_predict(points)
            return labels, model.cluster_hierarchy_.to_numpy(), model.cluster_hierarchy_.get_clusters(1.1, 2)
    else:
        def fit():
            return robust_single_linkage(points, **{"cut": 2.5, "gamma": 1, **case["options"]})
    _check_major_output(fit, case["output"])


@pytest.mark.parametrize("name", MAJOR_REFERENCE["spatial_fits"])
def test_spatial_distance_fit_reference(name):
    case = MAJOR_REFERENCE["spatial_fits"][name]
    points = _matrix_from_bits({"layout": "row-major", **case["input"]})

    def fit():
        model = HDBSCAN(**case["options"]).fit(points)
        return (
            model.labels_, model.probabilities_, model.cluster_persistence_,
            model.minimum_spanning_tree_.to_numpy(), model.single_linkage_tree_.to_numpy(),
            model.condensed_tree_.to_numpy(),
        )

    _check_major_output(fit, case["output"])


@pytest.mark.parametrize("case", MAJOR_REFERENCE["branches"].values(), ids=MAJOR_REFERENCE["branches"])
def test_branch_detection_reference(case):
    points = _matrix_from_bits({"layout": "row-major", **case["input"]})
    labels = None if case["labels"] is None else np.asarray(case["labels"], dtype=np.intp)
    model = HDBSCAN(**case["model"]).fit(points)
    cache = pickle.loads(pickle.dumps(model.branch_detection_data_))
    _assert_major_output((cache.finite_index, cache.neighbors, cache.core_distances), case["cache"])
    _check_major_output(lambda: detect_branches_in_clusters(model, labels, **case["options"]), case["output"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        detector = BranchDetector(**case["options"]).fit(model, labels=labels)
    _assert_major_output(detector.labels_, case["output"]["result"][0])
    graph = detector.approximation_graph_.to_numpy()
    assert all(part.dtype.type is np.void for part in graph)
    _assert_major_output(graph, case["graph"])
    if "condensed_plot" in case:
        expected = case["condensed_plot"]
        geometry = model.condensed_tree_.get_plot_data(**expected["options"])
        for key, values in expected["arrays"].items():
            _assert_array(np.asarray(geometry[key], dtype=np.float64), values)
        assert set(map(str, geometry["cluster_bounds"])) == set(expected["bounds"])
        for node, bounds in geometry["cluster_bounds"].items():
            _assert_array(np.asarray(bounds, dtype=np.float64), expected["bounds"][str(node)])
    if "queries" in case:
        queries = _matrix_from_bits({"layout": "row-major", **case["queries"]})
        _check_major_output(lambda: approximate_predict_branch(detector, queries), case["prediction"])
        _assert_major_output(detector.exemplars_, case["exemplars"])
        for label, record in case["representatives"].items():
            label = int(label)
            _assert_major_output((detector.weighted_centroid(label), detector.weighted_medoid(label)), record)


@pytest.mark.parametrize("case", MAJOR_REFERENCE["branch_calls"].values(), ids=MAJOR_REFERENCE["branch_calls"])
def test_branch_detection_call_reference(case):
    points = _matrix_from_bits({"layout": "row-major", **case["input"]})
    model = HDBSCAN(**case["model"]).fit(points)
    labels = case["labels"]
    if labels is not None:
        labels = np.asarray(labels["values"], dtype=labels["dtype"])
    probabilities = case["probabilities"]
    if probabilities is not None:
        if "bits" in probabilities:
            probabilities = _matrix_from_bits({"layout": "row-major", **probabilities})
        else:
            probabilities = np.asarray(probabilities["values"], dtype=probabilities["dtype"])
    _check_major_output(
        lambda: detect_branches_in_clusters(model, labels, probabilities, **case["options"]), case["output"],
    )


@pytest.mark.parametrize(("sample", "message"), [
    (20, "could not broadcast input array from shape (34,) into shape (39,)"),
    (60, "shape mismatch: value array of shape (31,) could not be broadcast to indexing result of shape (32,)"),
])
def test_branch_detection_incomplete_hierarchy_error_order(sample, message):
    # The public Wheel can terminate after returning the first error, so run
    # this pair only against the independent implementation in the test process.
    rng = np.random.default_rng(753)
    for n in (50, 50, 50, 35):
        rng.normal(size=(n, 2))
    for i in range(12, sample + 1):
        points = rng.normal(size=(i % 77 + 36, 1 + i % 7))
    points[:, 0] += np.arange(len(points)) % 3 * 12.0
    points = np.round(points, 1)
    points[1, 0] = np.nan
    model = HDBSCAN(min_cluster_size=2 + sample % 6, min_samples=1, leaf_size=4, algorithm="generic",
                    branch_detection_data=True, core_dist_n_jobs=1).fit(points)
    with pytest.warns(DeprecationWarning) as observed, pytest.raises(ValueError) as error:
        detect_branches_in_clusters(model, min_cluster_size=2, cluster_selection_method="leaf")
    assert len(observed) == 6
    assert str(error.value) == message


def _assert_validity_output(actual, expected):
    if isinstance(expected, list):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for value, record in zip(actual, expected):
            _assert_validity_output(value, record)
    elif expected is None:
        assert actual is None
    elif "list" in expected:
        assert isinstance(actual, list) and actual == expected["list"]
    else:
        _assert_array(np.asarray(actual), expected)


def _check_validity(case, call, capsys):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        if "error" in case:
            with pytest.raises(Exception) as caught:
                call()
            assert type(caught.value).__name__ == case["error"]["type"]
            assert str(caught.value) == case["error"]["message"]
        else:
            actual = call()
            expected = case["result"]
            if missing := case.get("missing_labels"):
                assert np.isnan(actual[1][missing]).all()
                expected = [expected[0], {**expected[1], "bits": [
                    bits if bits is not None else "7ff8000000000000" for bits in expected[1]["bits"]
                ]}]
            _assert_validity_output(actual, expected)
    assert [{"category": item.category.__name__, "message": str(item.message)}
            for item in observed] == case["warnings"]
    assert capsys.readouterr().out == case["stdout"]


@pytest.mark.parametrize("case", VALIDITY_REFERENCE["cases"].values(), ids=VALIDITY_REFERENCE["cases"].keys())
def test_validity_index_matches_reference(case, capsys):
    data = _matrix_from_bits(case["input"])
    labels = np.asarray(case["labels"]["values"], dtype=case["labels"]["dtype"])
    _check_validity(case, lambda: validity_index(data, labels, per_cluster_scores=True, **case["options"]), capsys)


@pytest.mark.parametrize("name", VALIDITY_REFERENCE["helpers"])
def test_validity_helpers_match_reference(name, capsys):
    line = np.array([[0.], [1.], [2.], [5.], [6.], [8.]])
    labels = np.repeat([0, 1], 3)
    symmetric = np.abs(line - line.T)
    path = symmetric[:4, :4].copy()
    path[0, 2] = path[2, 0] = 1.000001
    tree = np.array([[0., 6., 4., 5.], [6., 0., 2., 9.], [4., 2., 0., 3.], [5., 9., 3., 0.]])
    asymmetric = np.array([[0., 1., 9.], [2., 0., 1.], [3., 6., 0.]])
    all_core = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
    coordinates = np.array([[0.831, -0.24], [-1.01, 0.725], [1.14, 2.5]], dtype=np.float32)
    supplied_norms = np.zeros(3, dtype=np.float32)
    calls = {
        "core": (all_points_core_distance, all_core, 1.25),
        "cluster-matrix": (distances_between_points, line, labels, 1),
        "precomputed-matrix": (distances_between_points, symmetric, labels, 1, "precomputed", 1.0),
        "float32-haversine": (distances_between_points, coordinates, labels[:3], 0, "haversine"),
        "float32-callable": (distances_between_points, coordinates, labels[:3], 0,
                             lambda left, right: np.abs(left - right).sum()),
        "float32-euclidean-norms": (lambda: distances_between_points(
            coordinates, labels[:3], 0, X_norm_squared=supplied_norms, Y_norm_squared=supplied_norms,
        ),),
        "worker-cosine": (lambda: distances_between_points(coordinates, labels[:3], 0, "cosine", n_jobs=2),),
        "tree-tie": (internal_minimum_spanning_tree, path),
        "tree-internal-edge": (internal_minimum_spanning_tree, tree),
        "tree-two-vertices": (internal_minimum_spanning_tree, symmetric[:2, :2].copy()),
        "tree-asymmetric": (internal_minimum_spanning_tree, asymmetric),
        "tree-fortran": (internal_minimum_spanning_tree, np.asfortranarray(tree)),
        "separation": (density_separation, line, labels, 0, 1, [1, 2], [0, 1],
                       np.array([1., 2., 5.]), np.array([9., 3., 4.])),
    }
    function, *args = calls[name]
    _check_validity(VALIDITY_REFERENCE["helpers"][name], lambda: function(*args), capsys)
    if name == "core":
        assert all_core[0, 1] == 1.0
        assert all_core[0, 2] == (1.0 / 3.0) ** 1.25


@pytest.mark.parametrize("case", REFERENCE["cases"].values(), ids=REFERENCE["cases"].keys())
def test_public_fit_matches_reference(case, capsys):
    params = REFERENCE["parameters"] | case.get("parameter_overrides", {})
    estimator = HDBSCAN(**params)
    source = case["input"]
    if source.get("format") == "csr":
        values = np.array(source["values"], dtype=source["dtype"])
        data = csr_matrix((values.copy(), source["indices"], source["indptr"]), shape=source["shape"])
        repeat_input = csr_array(data.copy())
    else:
        data = np.array(source["values"], dtype=source["dtype"])
        repeat_input = np.asfortranarray(data.copy())
    assert estimator.fit(data) is estimator
    outputs = case["outputs"]
    for name in ("labels_", "probabilities_", "cluster_persistence_", "outlier_scores_"):
        _assert_array(getattr(estimator, name), outputs[name])
    _check_validity(outputs["relative_validity_"], lambda: estimator.relative_validity_, capsys)
    for cut in outputs["dbscan_cuts"]:
        distance = np.uint64(int(cut["cut_distance_bits"], 16)).view(np.float64)
        _assert_array(estimator.dbscan_clustering(distance, cut["min_cluster_size"]), cut["labels"])
    if params["metric"] == "precomputed":
        with pytest.warns(UserWarning, match="precomputed metric"):
            assert estimator.minimum_spanning_tree_ is None
    else:
        _assert_array(estimator.minimum_spanning_tree_.to_numpy(), outputs["minimum_spanning_tree_"])
    _assert_array(estimator.single_linkage_tree_.to_numpy(), outputs["single_linkage_tree_"])
    condensed = estimator.condensed_tree_.to_numpy()
    expected = outputs["condensed_tree_"]
    assert condensed.shape == tuple(expected["shape"])
    assert condensed.dtype.names == tuple(field["name"] for field in expected["fields"])
    for field in expected["fields"]:
        _assert_array(condensed[field["name"]], field)
    _assert_array(estimator.fit_predict(repeat_input), outputs["labels_"])
    if source.get("format") == "csr":
        for name in ("indices", "indptr"):
            np.testing.assert_array_equal(getattr(data, name), source[name])
        np.testing.assert_array_equal(data.data, values)
    else:
        np.testing.assert_array_equal(data, np.array(source["values"], dtype=source["dtype"]))


def test_relative_validity_cache_and_tree_option():
    points = [[0.0], [1.0], [6.0], [7.0]]
    model = HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=0)
    with pytest.raises(AttributeError, match="Minimum spanning tree not present"):
        _ = model.relative_validity_
    model.fit(points)
    model.gen_min_span_tree = 1
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'T'"):
        _ = model.relative_validity_
    model.labels_[0] = -2
    with pytest.raises(ValueError, match="must have no negative elements"):
        _ = model.relative_validity_
    unfitted = HDBSCAN(gen_min_span_tree=True)
    with pytest.raises(AttributeError, match="has no attribute 'labels_'"):
        _ = unfitted.relative_validity_

    model.fit(points)
    score = model.relative_validity_
    assert type(score) is np.float64 and score == 0.8
    model.fit([[0.0], [1.0], [3.0], [4.0]])
    assert model.relative_validity_ is score
    model.relative_validity_ = None
    assert model.relative_validity_ == 0.5
    model.gen_min_span_tree = False
    assert model.relative_validity_ == 0.5
    model.relative_validity_ = 0
    assert type(model.relative_validity_) is int and model.relative_validity_ == 0


def _matrix_from_bits(record):
    dtype = np.dtype(record["dtype"])
    bits = np.array([int(value, 16) for value in record["bits"]], dtype=f"u{dtype.itemsize}")
    matrix = bits.view(dtype).reshape(record["shape"])
    if record["layout"] == "fortran":
        return np.asfortranarray(matrix)
    if record["layout"] == "reversed":
        return matrix[:, ::-1].copy()[:, ::-1]
    return matrix


@pytest.mark.parametrize("case", PREDICTION_REFERENCE["cases"].values(), ids=PREDICTION_REFERENCE["cases"])
def test_prediction_matches_reference(case):
    params = PREDICTION_REFERENCE["parameters"] | case["parameter_overrides"]
    estimator = HDBSCAN(**params).fit(_matrix_from_bits(case["input"]))
    query = _matrix_from_bits(case["queries"])
    data = estimator.prediction_data_
    _assert_array(estimator.labels_, case["labels"])
    _assert_array(estimator.probabilities_, case["probabilities"])
    _assert_array(data.core_distances, case["core_distances"])
    assert sorted([node, label] for node, label in data.cluster_map.items()) == case["cluster_map"]
    assert len(estimator.exemplars_) == len(case["exemplars"])
    for actual, expected in zip(estimator.exemplars_, case["exemplars"], strict=True):
        _assert_array(actual, expected)

    restored = pickle.loads(pickle.dumps(estimator))
    for model in (estimator, restored):
        for key, call in (
            ("predict", lambda: approximate_predict(model, query, return_connecting_points=True)),
            ("scores", lambda: (approximate_predict_scores(model, query),)),
        ):
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                outputs = call()
            for output, expected in zip(outputs, case["outputs"][key]["arrays"], strict=True):
                _assert_array(output, expected)
            assert [
                {"category": item.category.__name__, "message": str(item.message)} for item in observed
            ] == case["outputs"][key]["warnings"]


@pytest.mark.parametrize("case", PREDICTION_REFERENCE["cases"].values(), ids=PREDICTION_REFERENCE["cases"])
def test_membership_matches_reference(case):
    options = PREDICTION_REFERENCE["parameters"] | case["parameter_overrides"]
    estimator = HDBSCAN(**options).fit(_matrix_from_bits(case["input"]))
    estimator = pickle.loads(pickle.dumps(estimator))
    queries = _matrix_from_bits(case["queries"])
    for name, call in (
        ("new", lambda: membership_vector(estimator, queries)),
        ("training", lambda: all_points_membership_vectors(estimator)),
    ):
        expected = case["membership"][name]
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            if "error" in expected:
                error = expected["error"]
                error_type = {"KeyError": KeyError, "ValueError": ValueError}[error["type"]]
                with pytest.raises(error_type) as captured:
                    call()
                assert list(captured.value.args) == error["args"]
            else:
                _assert_array(call(), expected["array"])
        assert [
            {"category": item.category.__name__, "message": str(item.message)} for item in observed
        ] == expected["warnings"]


def test_membership_validation_and_numpy_error_state():
    data = np.repeat([[0.], [10.]], 5, axis=0)
    estimator = HDBSCAN(min_samples=2, min_cluster_size=2)
    with pytest.raises(AttributeError, match="No condensed tree"):
        all_points_membership_vectors(estimator)
    estimator.fit(data)
    with pytest.raises(AttributeError, match="No prediction data"):
        all_points_membership_vectors(estimator)
    estimator.generate_prediction_data()
    with pytest.raises(ValueError, match="query data dimension must match training data dimension"):
        membership_vector(estimator, np.array([[0., 1.]]))
    with pytest.raises(ValueError, match="Expected 2D array, got 1D array instead"):
        membership_vector(estimator, np.array([1.]))
    with pytest.raises(IndexError, match="tuple index out of range"):
        membership_vector(estimator, np.array(1.))
    with pytest.raises(AttributeError, match="no attribute 'astype'"):
        membership_vector(estimator, [[1.]])
    with np.errstate(invalid="raise"):
        with pytest.raises(FloatingPointError, match="invalid value encountered in scalar divide"):
            all_points_membership_vectors(estimator)
        with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
            assert np.isnan(membership_vector(estimator, np.ma.array([[1.]], mask=True))).all()


def test_membership_with_retained_prediction_data():
    train = np.r_[np.arange(5.), np.arange(10., 15.), np.arange(22., 27.)][:, None]
    queries = np.array([[1.], [5.], [12.], [25.]])
    estimator = HDBSCAN(min_samples=2, min_cluster_size=2, prediction_data=True).fit(train)
    expected = membership_vector(estimator, queries)
    expected_training = all_points_membership_vectors(estimator)
    estimator.prediction_data = False
    estimator.fit(train + 100.)
    np.testing.assert_array_equal(membership_vector(estimator, queries), expected)
    np.testing.assert_array_equal(all_points_membership_vectors(estimator), expected_training)
    estimator.fit(np.arange(len(train), dtype=np.float64)[:, None])
    with pytest.raises(ValueError, match="shapes \\(3,\\) \\(2,\\)"):
        membership_vector(estimator, queries)
    with pytest.raises(ValueError, match="shapes \\(15,3\\) \\(15,2\\)"):
        all_points_membership_vectors(estimator)
    estimator.fit(train[:11])
    with pytest.raises(KeyError) as captured:
        all_points_membership_vectors(estimator)
    assert captured.value.args == (11,)


def test_prediction_late_cache_and_refit():
    train = np.r_[np.arange(5.), np.arange(10., 15.)][:, None]
    query = [[0.], [1.1], [5.], [12.], [24.]]
    estimator = HDBSCAN(min_cluster_size=2, min_samples=2)
    with pytest.raises(AttributeError, match="No condensed tree"):
        estimator.generate_prediction_data()
    with pytest.raises(AttributeError, match="No prediction data was generated"):
        approximate_predict(estimator, query)
    with pytest.raises(ValueError, match="Clusterer does not have prediction data"):
        approximate_predict_scores(estimator, query)

    estimator.fit(train)
    assert not hasattr(estimator, "prediction_data_")
    assert len(estimator.exemplars_) == 2
    old_cache = estimator.prediction_data_
    assert approximate_predict(estimator, query)[0].tolist() == [0, 0, 0, 1, -1]
    estimator.fit(train[[0, 5, 6, 1, 2, 3, 4, 7, 8, 9]])
    assert estimator.prediction_data_ is old_cache
    assert approximate_predict(estimator, query)[0].tolist() == [0, 1, 0, 1, -1]
    estimator.generate_prediction_data()
    assert estimator.prediction_data_ is not old_cache


@pytest.mark.parametrize(
    ("refit", "query", "label", "strength", "neighbor", "score"),
    [
        ([0, 3, 4, 6, 9, 10, 11, 13, 14, 20, 21, 24, 26], 6., 0,
         "0x1.5555555555555p-2", 5, "0x1.5555555555556p-2"),
        ([1, 2, 3, 11, 13, 14, 16, 21, 22, 23], 20.5, -1,
         "0x0.0p+0", 13, "-0x1.5555555555554p-2"),
    ],
)
def test_prediction_after_refit_with_fewer_training_rows(refit, query, label, strength, neighbor, score):
    original = np.r_[np.arange(6.), np.arange(10., 16.), np.arange(20., 24.)][:, None]
    estimator = HDBSCAN(min_cluster_size=2, min_samples=2, prediction_data=True).fit(original)
    estimator.prediction_data = False
    estimator.fit(np.asarray(refit, dtype=np.float64)[:, None])
    labels, strengths, neighbors = approximate_predict(estimator, [[query]], return_connecting_points=True)
    assert (labels[0], strengths[0], neighbors[0]) == (label, float.fromhex(strength), neighbor)
    assert approximate_predict_scores(estimator, [[query]])[0] == float.fromhex(score)


def test_prediction_scores_after_changing_min_samples():
    train = np.array([0., 0.1, 0.3, 1., 1.2, 10., 10.2, 10.6, 12., 18.])[:, None]
    estimator = HDBSCAN(min_cluster_size=2, min_samples=3, prediction_data=True).fit(train)
    estimator.min_samples = 1
    labels, strengths, neighbors = approximate_predict(estimator, train, return_connecting_points=True)
    assert neighbors[-1] == 8
    assert labels[-1] == 1
    assert strengths[-1] == float.fromhex("0x1.3333333333335p-2")
    scores = approximate_predict_scores(estimator, train)
    np.testing.assert_array_equal(scores[:-1], np.zeros(9))
    assert scores[-1] == float.fromhex("0x1.9999999999983p-4")


def test_prediction_validation_for_unsupported_inputs():
    train = np.r_[np.arange(5.), np.arange(10., 15.)][:, None]
    options = {"min_cluster_size": 2, "min_samples": 2, "prediction_data": True}
    with pytest.raises(TypeError, match="Sparse data was passed, but dense data is required"):
        HDBSCAN(**options).fit(csr_matrix(train))
    with pytest.raises(ValueError, match="Input contains NaN"):
        HDBSCAN(**options).fit(np.vstack((train, [[np.nan]])))
    with pytest.warns(UserWarning, match="Cannot generate prediction data for non-vector space inputs"):
        distance_model = HDBSCAN(metric="precomputed", **options).fit(np.abs(train - train.T))
    assert not hasattr(distance_model, "prediction_data_")
    with pytest.warns(UserWarning, match="Metric cosine not supported for prediction data"):
        cosine_model = HDBSCAN(metric="cosine", algorithm="generic", **options).fit(np.c_[train, train + 1])
    assert not hasattr(cosine_model, "prediction_data_")

    estimator = HDBSCAN(**options).fit(train)
    for predict in (approximate_predict, approximate_predict_scores):
        with pytest.raises(ValueError, match="New points dimension does not match fit data"):
            predict(estimator, [[1., 2.]])
        with pytest.raises(ValueError, match="Found array with 0 sample"):
            predict(estimator, np.empty((0, 1)))
        with pytest.raises(ValueError, match="Input contains NaN"):
            predict(estimator, [[np.nan]])
        with pytest.raises(IndexError, match="tuple index out of range"):
            predict(estimator, [1.])
    estimator.min_samples = 8
    with pytest.raises(ValueError, match="k must be less than or equal to"):
        approximate_predict(estimator, [[1.]])
    with pytest.raises(ValueError, match="Input contains NaN"):
        approximate_predict(estimator, [[np.nan]])


@pytest.mark.parametrize(("metric", "alias", "options"), [
    ("minkowski", "p", {"p": 3.5}),
    ("chebyshev", "infinity", {}),
])
def test_spatial_distance_aliases_support_late_cache_generation(metric, alias, options):
    case = MAJOR_REFERENCE["branches"]["minkowski-core"]
    points = _matrix_from_bits({"layout": "row-major", **case["input"]})
    common = {"min_cluster_size": 2, "min_samples": 2, "core_dist_n_jobs": 1, "gen_min_span_tree": True}
    expected = HDBSCAN(metric=metric, prediction_data=True, branch_detection_data=True, **common, **options).fit(points)
    model = HDBSCAN(metric=alias, **common).fit(points)
    model.generate_prediction_data()
    model.generate_branch_detection_data()
    model = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.prediction_data_.core_distances, expected.prediction_data_.core_distances)
    np.testing.assert_array_equal(model.branch_detection_data_.neighbors, expected.branch_detection_data_.neighbors)
    np.testing.assert_array_equal(model.branch_detection_data_.core_distances,
                                  expected.branch_detection_data_.core_distances)


def test_minkowski_cache_rejects_weight_argument():
    case = PREDICTION_REFERENCE["cases"]["minkowski-generic-p3"]
    model = HDBSCAN(**case["parameter_overrides"], min_cluster_size=2, w=None).fit(_matrix_from_bits(case["input"]))
    for generate in (model.generate_prediction_data, model.generate_branch_detection_data):
        with pytest.raises(TypeError, match="unexpected keyword argument 'w'"):
            generate()


def test_prediction_scores_respect_numpy_invalid_handling():
    train = np.repeat([[0.], [10.]], 5, axis=0)
    estimator = HDBSCAN(min_cluster_size=2, min_samples=2, prediction_data=True).fit(train)
    with np.errstate(invalid="raise"):
        with pytest.raises(FloatingPointError, match="invalid value encountered in scalar subtract"):
            approximate_predict_scores(estimator, [[0.]])
        with pytest.raises(FloatingPointError, match="invalid value encountered in scalar divide"):
            approximate_predict_scores(estimator, [[0.25]])


@pytest.mark.parametrize(("algorithm", "tree"), [("prims_kdtree", "KDTree"), ("prims_balltree", "BallTree")])
def test_prims_validates_inputs(algorithm, tree):
    estimator = HDBSCAN(algorithm=algorithm, min_cluster_size=2)
    with pytest.raises(ValueError, match="Sparse data matrices only support algorithm 'generic'[.]$"):
        estimator.fit(csr_matrix([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="k must be less than or equal to the number of training points$"):
        estimator.fit([[np.inf], [1.0]])
    estimator.metric = "cosine"
    with pytest.raises(ValueError, match=f"Cannot use Prim's with {tree} for this metric!$"):
        estimator.fit([[0.0], [1.0]])


@pytest.mark.parametrize("metric", ["cosine", "arccos"])
def test_best_rejects_dense_cosine(metric):
    with pytest.raises(ValueError, match=f"Unrecognized metric '{metric}'$"):
        HDBSCAN(algorithm="best", metric=metric).fit([[0.0], [1.0]])


def test_default_fit_uses_upstream_best():
    case = REFERENCE["cases"]["best-euclidean-60-features"]
    estimator = HDBSCAN(min_cluster_size=3, min_samples=2, gen_min_span_tree=True)
    data = np.asarray(case["input"]["values"])
    assert estimator.algorithm == "best"
    assert estimator.approx_min_span_tree is True
    assert estimator.core_dist_n_jobs == 4
    assert estimator.fit(data) is estimator
    _assert_array(estimator.labels_, case["outputs"]["labels_"])
    _assert_array(estimator.minimum_spanning_tree_.to_numpy(), case["outputs"]["minimum_spanning_tree_"])


def test_prims_repeated_hierarchy_reports_reference_capacity_error():
    estimator = HDBSCAN(algorithm="prims_kdtree", min_cluster_size=2, min_samples=1)
    with pytest.raises(ValueError, match=r"could not broadcast input array from shape \(10,\) into shape \(30,\)$"):
        estimator.fit(np.array([[-1e200], [-9e199], [0.0], [9e199], [1e200]]))


def test_postfit_queries_require_fit():
    estimator = HDBSCAN()
    with pytest.raises(AttributeError, match="No condensed tree was generated; try running fit first[.]$"):
        _ = estimator.outlier_scores_
    with pytest.raises(AttributeError, match="No single linkage tree was generated; try running fit first[.]$"):
        estimator.dbscan_clustering(1.0)


def test_postfit_queries_after_refit():
    estimator = HDBSCAN(min_cluster_size=2, min_samples=1)
    estimator.fit([[0.0], [1.0], [5.0], [6.0], [8.0]])
    scores = estimator.outlier_scores_
    np.testing.assert_array_equal(scores, [0.0, 0.0, 0.0, 0.0, 0.5])
    assert estimator.outlier_scores_ is scores

    estimator.fit([[0.0], [1.0], [5.0], [6.0], [14.0], [17.0]])
    assert estimator.outlier_scores_ is scores
    np.testing.assert_array_equal(estimator.dbscan_clustering(2.0, 2), [0, 0, 1, 1, -1, -1])
    np.testing.assert_array_equal(estimator.dbscan_clustering(np.inf), [0, 0, 0, 0, 0, 0])


def test_dbscan_minimum_accepts_int_subclasses_and_platform_overflow():
    class Minimum(int):
        def __int__(self):
            raise AssertionError("integer subclasses use the stored integer")

    estimator = HDBSCAN(min_samples=1, min_cluster_size=2).fit([[0.0], [1.0], [5.0]])
    np.testing.assert_array_equal(estimator.dbscan_clustering(2.0, Minimum(2)), [0, 0, -1])
    message = "int too big to convert" if platform.system() == "Windows" else "Python int too large to convert to C long"
    with pytest.raises(OverflowError, match=message):
        estimator.dbscan_clustering(2.0, 2**63)


def test_refit_updates_tree_availability_after_success_only():
    estimator = HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
    data = [[0.0], [1.0]]
    labels = estimator.fit_predict(data)
    estimator.gen_min_span_tree = False
    with pytest.raises(ValueError, match="k must be less than or equal to the number of training points"):
        estimator.fit([[0.0], [np.nan]])
    assert estimator.labels_ is labels
    assert estimator.minimum_spanning_tree_.to_numpy().shape == (1, 3)

    estimator.fit(data)
    assert not hasattr(estimator, "minimum_spanning_tree_")


@pytest.mark.parametrize("allow_single_cluster", [True, 1, np.int64(1)])
def test_invalid_selection_preserves_previous_fit(allow_single_cluster):
    model = HDBSCAN(min_cluster_size=2, min_samples=1, allow_single_cluster=allow_single_cluster)
    data = [[0.0], [1.0], [5.0]]
    old_labels = model.fit_predict(data)
    np.testing.assert_array_equal(old_labels, [0, 0, -1])
    model.cluster_selection_method = "unknown"
    with pytest.raises(ValueError, match="cluster_selection_method"):
        model.fit(data)
    assert model.labels_ is old_labels
    model.cluster_selection_method = "leaf"
    np.testing.assert_array_equal(model.fit_predict(data), [-1, -1, -1])


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"cluster_selection_epsilon": -0.5}, ValueError, "Epsilon must be a float"),
        ({"cluster_selection_epsilon": np.float64(0.5)}, ValueError, "Epsilon must be a float"),
        ({"cluster_selection_epsilon": 2, "cluster_selection_epsilon_max": 1.0}, ValueError, "epsilon max"),
        ({"cluster_selection_persistence": 0}, ValueError, "Persistence must be a float"),
        ({"cluster_selection_persistence": 0.5}, IndexError, "index -1 is out of bounds"),
    ],
)
def test_invalid_thresholds_preserve_previous_fit(overrides, error, message):
    model = HDBSCAN(min_cluster_size=2, min_samples=1, allow_single_cluster=True)
    data = [[0.0], [1.0], [5.0]]
    old_labels = model.fit_predict(data)
    for key, value in overrides.items():
        setattr(model, key, value)
    with pytest.raises(error, match=message):
        model.fit(data)
    assert model.labels_ is old_labels


@pytest.mark.parametrize(("epsilon", "labels"), [(np.int32(4), [0, 0, 0]), (float("inf"), [0, 0, 0]), (float("nan"), [-1, -1, -1])])
def test_root_cutoff_with_special_epsilon(epsilon, labels):
    model = HDBSCAN(min_cluster_size=2, min_samples=1, allow_single_cluster=True, cluster_selection_epsilon=epsilon)
    np.testing.assert_array_equal(model.fit_predict([[0.0], [1.0], [5.0]]), labels)


@pytest.mark.parametrize(
    ("scale", "persistence", "messages"),
    [
        (1.0, 1e39, ["overflow encountered in cast"]),
        (1e-39, 0.1, ["overflow encountered in cast", "invalid value encountered in scalar subtract"]),
        (1 / (float(np.finfo(np.float32).tiny) * (1 - 2**-24)), 0.1, ["underflow encountered in cast"]),
    ],
)
def test_persistence_reports_numpy_cast_and_subtraction(scale, persistence, messages):
    pair = REFERENCE["cases"]["persistence-threshold-rounds-to-equal-float32"]["input"]["values"]
    data = np.asarray(pair) * scale
    model = HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1, cluster_selection_persistence=persistence)
    with np.errstate(all="warn"), pytest.warns(RuntimeWarning) as recorded:
        np.testing.assert_array_equal(model.fit_predict(data), [-1, -1, -1, -1])
    assert [str(warning.message) for warning in recorded] == messages
    with np.errstate(all="raise"), pytest.raises(FloatingPointError, match=messages[0]):
        model.fit(data)


def test_epsilon_search_reports_zero_lambda_after_persistence_rounding():
    pair = REFERENCE["cases"]["persistence-threshold-rounds-to-equal-float32"]["input"]["values"]
    model = HDBSCAN(
        metric="precomputed", min_cluster_size=2, min_samples=1,
        cluster_selection_persistence=5e-324, cluster_selection_epsilon=0.5,
    )
    with np.errstate(all="warn"), pytest.warns(RuntimeWarning) as recorded:
        with pytest.raises(ZeroDivisionError, match="float division"):
            model.fit(np.asarray(pair) * 1e200)
    assert [str(warning.message) for warning in recorded] == [
        "underflow encountered in cast",
        "divide by zero encountered in scalar divide",
        "divide by zero encountered in scalar divide",
    ]


@pytest.mark.parametrize("base, alias", [("manhattan", "cityblock"), ("manhattan", "l1"), ("cosine", "arccos")])
@pytest.mark.parametrize("sparse", [False, True])
def test_feature_metric_aliases_preserve_distances(base, alias, sparse):
    data = np.array([[1.0, 1.0], [0.1, 0.08], [-1.0, -1.0], [-1.0, -0.9], [0.0, 0.0]])
    if sparse:
        data = csr_matrix(data)
    opts = dict(algorithm="generic", min_cluster_size=2, min_samples=1, gen_min_span_tree=True)
    ref = HDBSCAN(**opts, metric=base).fit(data)
    alt = HDBSCAN(**opts, metric=alias).fit(data)
    np.testing.assert_array_equal(alt.labels_, ref.labels_)
    np.testing.assert_array_equal(alt.minimum_spanning_tree_.to_numpy(), ref.minimum_spanning_tree_.to_numpy())


@pytest.mark.parametrize("sparse", [False, True])
def test_cosine_float32_requires_conversion(sparse):
    data = np.array([[1, 1], [1, 2], [-1, -1], [-1, -2]], dtype=np.float32)
    if sparse:
        data = csr_matrix(data)
    model = HDBSCAN(algorithm="generic", metric="cosine", min_cluster_size=2, min_samples=1)
    with pytest.raises(ValueError, match="Buffer dtype mismatch"):
        model.fit(data)
    np.testing.assert_array_equal(model.fit_predict(data.astype(np.float64)), [0, 0, 1, 1])


@pytest.mark.parametrize("sparse", [False, True])
def test_missing_rows_and_float32_dtype_error_order(sparse):
    convert = csr_matrix if sparse else np.asarray
    model = HDBSCAN(algorithm="generic", min_cluster_size=2, min_samples=1)
    with pytest.raises(ValueError, match=r"0 sample\(s\) \(shape=\(0, 2\)\)"):
        model.fit(convert([[np.nan, 0.0], [0.0, np.inf]], dtype=np.float32))
    with pytest.raises(ValueError, match="Buffer dtype mismatch"):
        model.fit(convert([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="Invalid shape in axis 0: 0"):
        model.fit(convert([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float64))


def test_missing_csr_manhattan_checks_its_index_width_after_empty_rows():
    rows = csr_array(
        ([np.nan, 1.0, 2.0], np.zeros(3, dtype=np.int64), np.arange(4, dtype=np.int64)),
        shape=(3, 2),
    )
    model = HDBSCAN(metric="manhattan", min_cluster_size=2, min_samples=1)
    with pytest.raises(ValueError, match="Buffer dtype mismatch, expected 'const int'"):
        model.fit(rows)
    rows.data[:] = np.nan
    with pytest.raises(ValueError, match=r"0 sample\(s\)"):
        model.fit(rows)


@pytest.mark.parametrize("data", [[], [[1.0]], [[], []], [0.0, 1.0], [[1j], [0.0]]])
def test_fit_rejects_invalid_matrices(data):
    with pytest.raises(ValueError):
        HDBSCAN().fit(data)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[0, 1, 2], [1, 0, 3]], "square"),
        ([[0.0]], "Invalid shape in axis 0: 0"),
        ([[0, np.nan], [np.nan, 0]], "NaN"),
        (np.where(np.eye(4, dtype=bool), 0., np.inf), "could not broadcast input array"),
    ],
)
def test_invalid_precomputed_fit_preserves_previous_result(matrix, message):
    model = HDBSCAN(metric="precomputed", min_cluster_size=2, gen_min_span_tree=False)
    valid = model.fit_predict([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match=message):
        model.fit(matrix)
    assert model.labels_ is valid
    assert not hasattr(model, "minimum_spanning_tree_")


@pytest.mark.parametrize("form", ["coo", "csc"])
def test_sparse_precomputed_accepts_other_scipy_formats(form):
    data = csr_matrix([[0, 2, 0, 0], [2, 0, 8, 0], [0, 8, 0, 3], [0, 0, 3, 0]])
    model = HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1)
    np.testing.assert_array_equal(model.fit_predict(data.asformat(form)), [0, 0, 1, 1])


@pytest.mark.parametrize("form", ["coo", "csc"])
def test_sparse_euclidean_accepts_other_scipy_formats_and_integer_features(form):
    data = csr_matrix([[1, 0], [2, 0], [0, 10], [0, 11]])
    model = HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=True)
    np.testing.assert_array_equal(model.fit_predict(data.asformat(form)), [0, 0, 1, 1])
    assert model.minimum_spanning_tree_.to_numpy().shape == (3, 3)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (csr_array([1, 0]), "two-dimensional"),
        (csr_matrix((3, 0)), r"0 feature\(s\)"),
        (csr_matrix([[1], [2]], dtype=np.float32), "Buffer dtype mismatch"),
        (csr_matrix([[0], [np.inf]]), "Invalid shape in axis 0: 0"),
        (csr_matrix([[0], [np.nan]]), "Invalid shape in axis 0: 0"),
        (csr_matrix([[0], [1j]]), "real"),
    ],
)
def test_sparse_euclidean_errors_preserve_previous_fit(data, message):
    model = HDBSCAN(min_cluster_size=2, min_samples=1)
    labels = model.fit_predict(csr_matrix([[0.0], [1.0]]))
    with pytest.raises(ValueError, match=message):
        model.fit(data)
    assert model.labels_ is labels


def test_sparse_fallback_does_not_add_unstored_edges():
    data = csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    model = HDBSCAN(metric="precomputed", min_cluster_size=2, max_dist=3.0)
    with pytest.raises(ValueError, match="multiple connected components"):
        model.fit(data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (csr_matrix((3, 4)), "square"),
        (csr_matrix((1, 1)), "array at index 0 has size 0"),
        (csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), "multiple connected components"),
        (csr_matrix([[0, 1, 0], [1, 0, 2], [0, 2, 0]]), "less than 2 neighbors"),
        (csr_matrix([[0, np.inf], [1, 0]]), r"Input contains infinity.*float64"),
        (csr_matrix([[0, -np.inf], [1, 0]], dtype=np.float32), r"Input contains infinity.*float32"),
        (csr_matrix([[0, np.nan], [1, 0]]), "Input contains NaN"),
        (csr_matrix(([1.0, 0.0, 0.0, 3.0], [1, 2, 0, 1], [0, 2, 3, 4]), shape=(3, 3)), "less than 2 neighbors"),
    ],
)
def test_sparse_precomputed_errors_preserve_previous_dense_fit(data, message):
    model = HDBSCAN(metric="precomputed", min_cluster_size=2)
    labels = model.fit_predict([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match=message):
        model.fit(data)
    assert model.labels_ is labels
