from importlib.metadata import distribution
from hdbscan_alternative import BranchDetector, HDBSCAN, RobustSingleLinkage, __version__, approximate_predict_branch, hdbscan
from hdbscan_alternative.validity import validity_index
import numpy as np
from scipy.sparse import csr_matrix

package = distribution("hdbscan-alternative")
assert __version__ == package.version
assert HDBSCAN(min_cluster_size=2).fit_predict([[0.0], [1.0]]).tolist() == [-1, -1]
assert HDBSCAN(min_cluster_size=2, metric="precomputed").fit_predict([[0.0, 1.0], [1.0, 0.0]]).tolist() == [-1, -1]
disconnected = np.array([[0.0, 1.0, np.inf], [1.0, 0.0, np.inf], [np.inf, np.inf, 0.0]])
with np.errstate(over="ignore"):
    fitted = HDBSCAN(min_cluster_size=2, min_samples=1, metric="precomputed").fit(disconnected)
assert fitted.labels_.tolist() == [1, 1, -1]
assert fitted.single_linkage_tree_.to_numpy()[-1, 2] == np.finfo(float).max
distances = csr_matrix([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 5.0, 0.0], [0.0, 5.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
assert HDBSCAN(min_cluster_size=2, min_samples=1, metric="precomputed").fit_predict(distances).tolist() == [0, 0, 1, 1]
features = csr_matrix([[1, 0], [2, 0], [0, 10], [0, 11]])
assert HDBSCAN(min_cluster_size=2, min_samples=1).fit_predict(features).tolist() == [0, 0, 1, 1]
assert HDBSCAN(min_cluster_size=2, min_samples=1, metric="manhattan").fit_predict(features).tolist() == [0, 0, 1, 1]
signed = [[1, 0], [2, 0], [-1, 0], [-2, 0]]
assert HDBSCAN(min_cluster_size=2, min_samples=1, metric="cosine", algorithm="generic").fit_predict(signed).tolist() == [0, 0, 1, 1]
root = HDBSCAN(min_cluster_size=2, min_samples=1, allow_single_cluster=True)
assert root.fit_predict([[0.0], [1.0], [5.0]]).tolist() == [0, 0, -1]
root.cluster_selection_epsilon = 4.0
assert root.fit_predict([[0.0], [1.0], [5.0]]).tolist() == [0, 0, 0]
root.cluster_selection_method = "leaf"
assert root.fit_predict([[0.0], [1.0], [5.0]]).tolist() == [-1, -1, -1]
missing = HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=True)
assert missing.fit_predict([[float('nan')], [0.0], [1.0], [5.0], [6.0]]).tolist() == [-1, 0, 0, 1, 1]
assert missing.minimum_spanning_tree_.to_numpy().shape == (3, 3)
assert missing.outlier_scores_.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
assert missing.dbscan_clustering(1.25, 2).tolist() == [-1, 0, 0, 1, 1]
wide = [[value] + [0.0] * 60 for value in (0.0, 1.0, 5.0, 6.0)]
best = HDBSCAN(min_cluster_size=2, min_samples=1, algorithm="best")
assert best.fit_predict(wide).tolist() == [0, 0, 1, 1]
points = np.array([[0.0], [1.0], [5.0], [6.0]])
labels, _, _, condensed, linkage, mst = hdbscan(points, 2, 1, gen_min_span_tree=True)
assert labels.tolist() == [0, 0, 1, 1]
assert condensed.dtype.names == ('parent', 'child', 'lambda_val', 'child_size')
assert linkage.shape == (3, 4) and mst.shape == (3, 3)
assert validity_index(points, np.array([0, 0, 1, 1])) == 0.8
relative = HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=True).fit(points)
assert relative.relative_validity_ == 0.75
robust = RobustSingleLinkage(cut=1.25, k=1, gamma=2).fit(points)
assert robust.labels_.tolist() == [0, 0, 1, 1]
assert robust.cluster_hierarchy_.to_numpy().shape == (3, 4)
clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, branch_detection_data=True, prediction_data=True)
detector = BranchDetector().fit(clusterer.fit(points))
assert detector.labels_.tolist() == [0, 0, 1, 1]
assert approximate_predict_branch(detector, points)[0].tolist() == [0, 0, 1, 1]
for metric, distance_options in (("minkowski", {"p": 3.5}), ("chebyshev", {})):
    clusterer = HDBSCAN(
        min_cluster_size=2, min_samples=1, metric=metric,
        prediction_data=True, branch_detection_data=True, **distance_options,
    ).fit(points)
    assert clusterer.prediction_data_.core_distances.shape == (4,)
    assert clusterer.branch_detection_data_.neighbors.shape == (4, 1)
    detector = BranchDetector(branch_detection_method="core").fit(clusterer)
    assert approximate_predict_branch(detector, points)[0].tolist() == [0, 0, 1, 1]
assert package.metadata["License-Expression"] == "MIT"
assert "LICENSE" in package.metadata.get_all("License-File", [])
assert any(
    entry.as_posix().endswith(".dist-info/licenses/LICENSE")
    and entry.locate().is_file()
    for entry in package.files
)
