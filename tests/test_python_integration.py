import warnings

import numpy as np
import pytest
from joblib import Memory
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from hdbscan_alternative import BranchDetector, HDBSCAN, RobustSingleLinkage


POINTS = np.array([[0, 0], [.1, .1], [.2, 0], [1, .2], [5, 0], [5.3, .4], [5.5, .5],
                   [6.3, .5], [13, 0], [13.2, .4], [13.4, .4], [19, 0]])


def test_estimators_with_sklearn_tools(tmp_path):
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, metric="minkowski", p=3.0,
                        memory=Memory(tmp_path, verbose=0))
    estimator = clone(clusterer).set_params(cluster_selection_method="leaf")
    assert isinstance(estimator, BaseEstimator)
    pipeline = make_pipeline(StandardScaler(), estimator)
    labels = pipeline.fit_predict(POINTS)
    check_is_fitted(pipeline)
    np.testing.assert_array_equal(labels, pipeline.fit_predict(POINTS))

    robust = clone(RobustSingleLinkage(metric="minkowski", metric_params={"p": 3.0}, k=1, gamma=2))
    check_is_fitted(robust.fit(POINTS))
    detector = clone(BranchDetector(branch_detection_method="core"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = HDBSCAN(min_samples=1, min_cluster_size=2, branch_detection_data=True).fit(POINTS)
        check_is_fitted(detector.fit(fitted))
    np.testing.assert_array_equal(detector.cluster_labels_, fitted.labels_)

    def count_clusters(model, data):
        return len(set(model.labels_)) - int(-1 in model.labels_)

    search = GridSearchCV(HDBSCAN(min_samples=1), {"min_cluster_size": [2, 3]}, scoring=count_clusters, cv=2)
    assert search.fit(POINTS).best_estimator_.min_cluster_size in (2, 3)


def test_tree_and_branch_plots():
    mpl = pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")
    pytest.importorskip("pandas")
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = HDBSCAN(min_samples=1, min_cluster_size=2, branch_detection_data=True).fit(POINTS)
        branch = BranchDetector(branch_detection_method="core").fit(model)
    fig, axes = plt.subplots(2, 2)
    single, condensed, mst, graph = axes.ravel()
    try:
        assert model.single_linkage_tree_.plot(axis=single, colorbar=False) is single
        assert len(single.lines) == 3 * (len(POINTS) - 1)
        assert model.condensed_tree_.plot(axis=condensed, select_clusters=True,
                                          label_clusters=True, colorbar=False) is condensed
        icicles = model.condensed_tree_.get_plot_data()
        assert len(condensed.lines) == len(icicles["line_xs"]) == 4
        assert len(condensed.texts) == 3
        assert condensed.yaxis_inverted()
        assert model.minimum_spanning_tree_.plot(axis=mst, colorbar=False) is mst
        np.testing.assert_array_equal(mst.collections[0].get_array(), model.minimum_spanning_tree_.to_numpy()[:, 2])
        plt.sca(graph)
        assert branch.approximation_graph_.plot(positions=POINTS, edge_color="component") is None
        assert len(graph.collections) == 3
        assert len(graph.collections[0].get_segments()) == len(branch.approximation_graph_.to_numpy()[1])
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "points", [POINTS, np.insert(POINTS, 3, [80., 26.], axis=0), np.insert(POINTS, 3, np.nan, axis=0), POINTS[:3]],
    ids=["clusters", "interleaved-noise", "missing-row", "all-noise"],
)
@pytest.mark.filterwarnings("ignore:Passing more than 2 positional arguments:DeprecationWarning")
def test_branch_plot_uses_graphviz_coordinates(points):
    mpl = pytest.importorskip("matplotlib")
    nx = pytest.importorskip("networkx")
    pytest.importorskip("pygraphviz")
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba

    model = HDBSCAN(min_samples=1, min_cluster_size=2, branch_detection_data=True).fit(points)
    graph = BranchDetector(branch_detection_method="core").fit(model).approximation_graph_
    layout = nx.nx_agraph.graphviz_layout(graph.to_networkx(), prog="sfdp")
    nodes, edges = graph.to_numpy()
    clustered = model.labels_ >= 0
    fig, axis = plt.subplots()
    try:
        assert graph.plot(node_color="probability", edge_color="component") is None
        fig.canvas.draw()
        lines, noise, members = axis.collections
        expected_lines = [[layout[parent], layout[child]] for parent, child in zip(edges["parent"], edges["child"])]
        np.testing.assert_array_equal(lines.get_segments(), expected_lines)
        if len(edges):
            np.testing.assert_array_equal(lines.get_array(), edges["component"])
        assert noise.get_offsets().shape == (np.count_nonzero(~clustered), 2)
        assert np.ma.getmaskarray(noise.get_offsets()).all()
        np.testing.assert_array_equal(noise.get_facecolors(), [to_rgba("silver")])
        expected_nodes = np.asarray([layout[node] for node in np.flatnonzero(clustered)]).reshape(-1, 2)
        np.testing.assert_array_equal(members.get_offsets(), expected_nodes)
        if len(nodes):
            np.testing.assert_array_equal(members.get_array(), nodes["probability"])
    finally:
        plt.close(fig)
