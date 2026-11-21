use hdbscan_core::branches::{BranchDetectionData, BranchDetectionMethod, BranchOptions};
use hdbscan_core::robust::{
    RobustLinkageAlgorithm, RobustLinkageOptions, robust_single_linkage_dense,
    robust_single_linkage_precomputed, robust_single_linkage_spatial,
};
use hdbscan_core::validity::{
    RelativeValidityError, RelativeValidityTree, ValidityError, internal_minimum_spanning_tree,
    relative_validity_components,
};
use hdbscan_core::{
    BoruvkaOptions, BoruvkaTree, ClusterSelection, ClusterSelectionMethod,
    ClusterSelectionThresholds, CondensedEdge, FeatureMetric, FitError, FitOptions, FitResult,
    LinkageMerge, MembershipError, MembershipQuery, PredictionModel, SelectionDiagnostic,
    SpanningEdge, TreeDistance, capture_selection_diagnostics, dbscan_clustering,
    fit_csr_manhattan_f32_with_missing, fit_csr_precomputed_configured, fit_csr_with_missing,
    fit_dense_boruvka, fit_dense_euclidean_strided_columns, fit_dense_precomputed_bounded,
    fit_dense_prims, fit_dense_tree_boruvka, fit_dense_tree_prims, fit_dense_with_missing,
    fit_pairwise_feature_distances, outlier_scores,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{
    PyAssertionError, PyIndexError, PyKeyError, PyOverflowError, PyTypeError, PyValueError,
    PyZeroDivisionError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyInt;

type NativeFit = (
    Vec<i64>,
    Vec<f64>,
    Vec<f64>,
    Option<Vec<[f64; 3]>>,
    Vec<[f64; 4]>,
    Vec<(usize, usize, f64, usize)>,
);
type NativeFitWithDiagnostics = (NativeFit, Vec<(&'static str, [f64; 3])>);

type NativeConfiguration<'a> = (&'a str, bool, (&'a str, bool), (f64, f64, f64, f64), bool);
type SpatialDistance<'a> = (&'a str, f64, Option<Vec<f64>>);

fn cached_tree_distance((metric, p): (&str, f64)) -> PyResult<TreeDistance<'static>> {
    match metric {
        "euclidean" => Ok(TreeDistance::Euclidean),
        "manhattan" => Ok(TreeDistance::Manhattan),
        "chebyshev" => Ok(TreeDistance::Chebyshev),
        "minkowski" => Ok(TreeDistance::Minkowski { p, weights: None }),
        _ => Err(PyValueError::new_err("unsupported spatial cache distance")),
    }
}

fn tree_distance<'a>(metric: &'a SpatialDistance<'_>) -> PyResult<TreeDistance<'a>> {
    let (name, p, data) = metric;
    let supplied = data.as_deref();
    let required = || supplied.ok_or_else(|| PyValueError::new_err("missing distance parameters"));
    Ok(match *name {
        "euclidean" => TreeDistance::Euclidean,
        "manhattan" => TreeDistance::Manhattan,
        "chebyshev" => TreeDistance::Chebyshev,
        "minkowski" => TreeDistance::Minkowski {
            p: *p,
            weights: supplied,
        },
        "seuclidean" => TreeDistance::StandardizedEuclidean(required()?),
        "mahalanobis" => TreeDistance::Mahalanobis(required()?),
        "hamming" => TreeDistance::Hamming,
        "canberra" => TreeDistance::Canberra,
        "braycurtis" => TreeDistance::BrayCurtis,
        "jaccard" => TreeDistance::Jaccard,
        "dice" => TreeDistance::Dice,
        "rogerstanimoto" => TreeDistance::RogersTanimoto,
        "russellrao" => TreeDistance::RussellRao,
        "sokalmichener" => TreeDistance::SokalMichener,
        "sokalsneath" => TreeDistance::SokalSneath,
        "haversine" => TreeDistance::Haversine,
        _ => return Err(PyValueError::new_err("unsupported spatial distance")),
    })
}
type TreeRows = Vec<(usize, usize, f64, usize)>;

fn thresholds(
    (epsilon, epsilon_max, persistence, max_cluster_size): (f64, f64, f64, f64),
) -> ClusterSelectionThresholds {
    ClusterSelectionThresholds {
        epsilon,
        epsilon_max,
        persistence,
        max_cluster_size,
    }
}

fn fit_error(error: FitError) -> PyErr {
    match error {
        FitError::EmptyPersistenceHierarchy => PyIndexError::new_err(error.to_string()),
        FitError::ZeroLambdaEpsilon | FitError::ZeroBranchWeight => {
            PyZeroDivisionError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn run_fit(
    py: Python<'_>,
    include_mst: bool,
    fit: impl FnOnce() -> Result<FitResult, FitError> + Send,
) -> PyResult<NativeFitWithDiagnostics> {
    let (result, diagnostics) = py.detach(move || capture_selection_diagnostics(fit));
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            if !diagnostics.is_empty() {
                report_selection_diagnostics(py, diagnostics)?;
            }
            return Err(fit_error(error));
        }
    };
    let diagnostics = diagnostics
        .into_iter()
        .map(|event| match event {
            SelectionDiagnostic::BranchGraphExtrema { core } => {
                ("graph_extrema", [f64::from(core), 0.0, 0.0])
            }
            SelectionDiagnostic::LambdaCast([first, second]) => {
                ("lambda_cast", [first, second, 0.0])
            }
            SelectionDiagnostic::InvalidLifetime => ("invalid_lifetime", [0.0; 3]),
            SelectionDiagnostic::ThresholdCastOverflow => ("threshold_overflow", [0.0; 3]),
            SelectionDiagnostic::BranchLambdaReciprocal(lambda) => {
                ("branch_reciprocal", [lambda, 0.0, 0.0])
            }
            SelectionDiagnostic::RootLambdaReciprocal(lambdas) => ("root_reciprocal", lambdas),
        })
        .collect();
    Ok((pack_fit(result, include_mst), diagnostics))
}

#[pyfunction]
fn _emit_fit_diagnostics(py: Python<'_>, events: Vec<(String, [f64; 3])>) -> PyResult<()> {
    let mut diagnostics = Vec::with_capacity(events.len());
    for (kind, values) in events {
        let [first, second, _] = values;
        let event = match kind.as_str() {
            "graph_extrema" => SelectionDiagnostic::BranchGraphExtrema { core: first != 0.0 },
            "lambda_cast" => SelectionDiagnostic::LambdaCast([first, second]),
            "invalid_lifetime" => SelectionDiagnostic::InvalidLifetime,
            "threshold_overflow" => SelectionDiagnostic::ThresholdCastOverflow,
            "branch_reciprocal" => SelectionDiagnostic::BranchLambdaReciprocal(first),
            "root_reciprocal" => SelectionDiagnostic::RootLambdaReciprocal(values),
            _ => return Err(PyValueError::new_err("unknown fit diagnostic")),
        };
        diagnostics.push(event);
    }
    report_selection_diagnostics(py, diagnostics)
}

fn report_selection_diagnostics(
    py: Python<'_>,
    diagnostics: Vec<SelectionDiagnostic>,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;
    let float32 = numpy.getattr("float32")?;
    // Replay the floating-point events under the caller's NumPy error settings.
    for diagnostic in diagnostics {
        match diagnostic {
            SelectionDiagnostic::BranchGraphExtrema { core } => {
                // The graph uses NumPy's positional output argument for its extrema.
                let empty = numpy.getattr("empty")?.call1((0,))?;
                let maximum = numpy.getattr("maximum")?;
                for _ in 0..if core { 6 } else { 2 } {
                    maximum.call1((&empty, &empty, &empty))?;
                }
            }
            SelectionDiagnostic::LambdaCast(lambdas) => {
                numpy
                    .getattr("array")?
                    .call1((lambdas.to_vec(),))?
                    .call_method1("astype", (&float32,))?;
            }
            SelectionDiagnostic::InvalidLifetime => {
                let infinity = float32.call1((f64::INFINITY,))?;
                infinity.call_method1("__sub__", (&infinity,))?;
            }
            SelectionDiagnostic::ThresholdCastOverflow => {
                float32.call1((0.0,))?.call_method1("__ge__", (1e39,))?;
            }
            SelectionDiagnostic::BranchLambdaReciprocal(lambda) => {
                numpy
                    .getattr("float64")?
                    .call1((lambda,))?
                    .call_method1("__rtruediv__", (1.0,))?;
            }
            SelectionDiagnostic::RootLambdaReciprocal(lambdas) => {
                numpy.getattr("divide")?.call1((1.0, lambdas.to_vec()))?;
            }
        }
    }
    Ok(())
}

fn feature_metric(metric: &str) -> PyResult<Option<FeatureMetric>> {
    match metric {
        "euclidean" => Ok(Some(FeatureMetric::Euclidean)),
        "manhattan" => Ok(Some(FeatureMetric::Manhattan)),
        "cosine" => Ok(Some(FeatureMetric::Cosine)),
        "precomputed" => Ok(None),
        _ => Err(PyValueError::new_err("unsupported distance metric")),
    }
}

fn cluster_selection(method: &str, allow_single_cluster: bool) -> PyResult<ClusterSelection> {
    let method = match method {
        "eom" => ClusterSelectionMethod::Eom,
        "leaf" => ClusterSelectionMethod::Leaf,
        _ => {
            return Err(PyValueError::new_err(
                "cluster_selection_method must be 'eom' or 'leaf'",
            ));
        }
    };
    Ok(ClusterSelection {
        method,
        allow_single_cluster,
    })
}

#[pyfunction]
#[pyo3(signature = (data, fit, config, search, distance=None, finite_rows=None))]
fn _fit_dense_with_diagnostics(
    py: Python<'_>,
    data: PyBuffer<f64>,
    fit: (usize, Option<usize>, f64),
    config: NativeConfiguration<'_>,
    search: Option<(&str, usize, bool, i64)>,
    distance: Option<SpatialDistance<'_>>,
    finite_rows: Option<(usize, Vec<bool>)>,
) -> PyResult<NativeFitWithDiagnostics> {
    let (min_cluster_size, min_samples, alpha) = fit;
    let (metric, gen_min_span_tree, (method, allow_single_cluster), limits, strict) = config;
    let metric = if distance.is_some() {
        None
    } else {
        feature_metric(metric)?
    };
    let (use_prims, boruvka) = match search {
        None => (false, None),
        Some(("prims", _, _, _)) => (true, None),
        Some((tree, leaf_size, approximate, core_dist_n_jobs)) => {
            let tree = match tree {
                "boruvka_kdtree" => BoruvkaTree::KdTree,
                "boruvka_balltree" => BoruvkaTree::BallTree,
                _ => return Err(PyValueError::new_err("unsupported spatial search")),
            };
            (
                false,
                Some(BoruvkaOptions {
                    tree,
                    leaf_size,
                    approximate,
                    core_dist_n_jobs,
                }),
            )
        }
    };
    let &[n_samples, n_features] = data.shape() else {
        return Err(PyValueError::new_err(
            "expected a two-dimensional feature matrix",
        ));
    };
    if distance.is_none() && metric.is_none() && n_features != n_samples {
        return Err(PyValueError::new_err(
            "precomputed distances must be a square matrix",
        ));
    }
    let strided_columns = data.strides()[1] != size_of::<f64>() as isize;
    let values = data.to_vec(py)?;
    drop(data);
    let mut options = FitOptions {
        min_cluster_size,
        min_samples,
        alpha,
    }
    .with_cluster_selection(cluster_selection(method, allow_single_cluster)?)
    .with_cluster_selection_thresholds(thresholds(limits));
    options.strict_cluster_membership = strict;
    if let Some(distance) = distance.as_ref() {
        let distance = tree_distance(distance)?;
        return run_fit(py, gen_min_span_tree, move || match boruvka {
            Some(search) => {
                fit_dense_tree_boruvka(&values, n_samples, n_features, distance, options, search)
            }
            None => fit_dense_tree_prims(&values, n_samples, n_features, distance, options),
        });
    }
    if let Some((dimensions, finite)) = finite_rows {
        return run_fit(py, gen_min_span_tree, move || {
            fit_pairwise_feature_distances(&values, &finite, dimensions, options)
        });
    }
    run_fit(py, gen_min_span_tree, move || match metric {
        Some(metric) => match boruvka {
            Some(search) => {
                fit_dense_boruvka(&values, n_samples, n_features, metric, options, search)
            }
            None if use_prims => fit_dense_prims(&values, n_samples, n_features, metric, options),
            None if metric == FeatureMetric::Euclidean && strided_columns => {
                fit_dense_euclidean_strided_columns(&values, n_samples, n_features, options)
            }
            None => fit_dense_with_missing(&values, n_samples, n_features, metric, options),
        },
        None => fit_dense_precomputed_bounded(&values, n_samples, options),
    })
}

#[pyfunction]
fn _fit_csr_with_diagnostics(
    py: Python<'_>,
    indptr: PyBuffer<usize>,
    indices: PyBuffer<usize>,
    data: &Bound<'_, PyAny>,
    n_features: usize,
    options: (usize, Option<usize>, f64, f64),
    config: NativeConfiguration<'_>,
) -> PyResult<NativeFitWithDiagnostics> {
    if indptr.dimensions() != 1 || indices.dimensions() != 1 {
        return Err(PyValueError::new_err("CSR buffers must be one-dimensional"));
    }
    let row_offsets = indptr.to_vec(py)?;
    let columns = indices.to_vec(py)?;
    let (min_cluster_size, min_samples, alpha, max_dist) = options;
    let (metric, gen_min_span_tree, (method, allow_single_cluster), limits, strict) = config;
    let metric = feature_metric(metric)?;
    let mut options = FitOptions {
        min_cluster_size,
        min_samples,
        alpha,
    }
    .with_cluster_selection(cluster_selection(method, allow_single_cluster)?)
    .with_cluster_selection_thresholds(thresholds(limits));
    options.strict_cluster_membership = strict;
    let include_mst = gen_min_span_tree;
    if metric == Some(FeatureMetric::Manhattan)
        && let Ok(data) = PyBuffer::<f32>::get(data)
    {
        let values = copy_csr_values(data, py)?;
        return run_fit(py, include_mst, move || {
            fit_csr_manhattan_f32_with_missing(&row_offsets, &columns, &values, n_features, options)
        });
    }
    let values = copy_csr_values(PyBuffer::<f64>::get(data)?, py)?;
    run_fit(py, include_mst, move || match metric {
        Some(metric) => {
            fit_csr_with_missing(&row_offsets, &columns, &values, n_features, metric, options)
        }
        None => fit_csr_precomputed_configured(&row_offsets, &columns, &values, options, max_dist),
    })
}

fn copy_csr_values<T: pyo3::buffer::Element>(
    data: PyBuffer<T>,
    py: Python<'_>,
) -> PyResult<Vec<T>> {
    if data.dimensions() != 1 {
        return Err(PyValueError::new_err("CSR buffers must be one-dimensional"));
    }
    data.to_vec(py)
}

fn pack_fit(result: FitResult, include_mst: bool) -> NativeFit {
    let mst = include_mst.then(|| {
        result
            .minimum_spanning_tree
            .into_iter()
            .map(|edge| [edge.source as f64, edge.target as f64, edge.distance])
            .collect()
    });
    let linkage = result
        .single_linkage_tree
        .into_iter()
        .map(|edge| {
            [
                edge.left as f64,
                edge.right as f64,
                edge.distance,
                edge.size as f64,
            ]
        })
        .collect();
    let condensed = result
        .condensed_tree
        .into_iter()
        .map(|edge| (edge.parent, edge.child, edge.lambda, edge.child_size))
        .collect();
    (
        result.labels,
        result.probabilities,
        result.cluster_persistence,
        mst,
        linkage,
        condensed,
    )
}

#[pyfunction]
fn _outlier_scores(py: Python<'_>, tree: TreeRows) -> PyResult<Vec<f64>> {
    py.detach(move || outlier_scores(&condensed_edges(tree)))
        .map_err(fit_error)
}

#[pyfunction]
#[pyo3(signature = (data, metric, algorithm, linkage, spatial, distance=None))]
fn _robust_single_linkage(
    py: Python<'_>,
    data: PyBuffer<f64>,
    metric: &str,
    algorithm: &str,
    linkage: (f64, usize, f64, i64),
    spatial: (usize, i64),
    distance: Option<SpatialDistance<'_>>,
) -> PyResult<(Vec<i64>, Vec<[f64; 4]>)> {
    let (cut, k, alpha, gamma) = linkage;
    let (leaf_size, core_dist_n_jobs) = spatial;
    let &[rows, columns] = data.shape() else {
        return Err(PyValueError::new_err(
            "expected a two-dimensional feature matrix",
        ));
    };
    let metric = if distance.is_some() {
        None
    } else {
        feature_metric(metric)?
    };
    if distance.is_none() && metric.is_none() && rows != columns {
        return Err(PyValueError::new_err(
            "precomputed distances must be a square matrix",
        ));
    }
    let search = match algorithm {
        "generic" => RobustLinkageAlgorithm::Generic,
        "prims_kdtree" | "prims_balltree" => RobustLinkageAlgorithm::Prim,
        "boruvka_kdtree" | "boruvka_balltree" => RobustLinkageAlgorithm::Boruvka(BoruvkaOptions {
            tree: if algorithm == "boruvka_kdtree" {
                BoruvkaTree::KdTree
            } else {
                BoruvkaTree::BallTree
            },
            leaf_size,
            approximate: false,
            core_dist_n_jobs,
        }),
        _ => {
            return Err(PyValueError::new_err(
                "unsupported robust linkage algorithm",
            ));
        }
    };
    let values = data.to_vec(py)?;
    drop(data);
    let options = RobustLinkageOptions {
        cut,
        k,
        alpha,
        gamma,
    };
    let spatial_distance = distance.as_ref().map(tree_distance).transpose()?;
    py.detach(|| {
        if let Some(distance) = spatial_distance {
            return robust_single_linkage_spatial(
                &values, rows, columns, distance, search, options,
            );
        }
        match metric {
            Some(metric) => {
                robust_single_linkage_dense(&values, rows, columns, metric, search, options)
            }
            None => robust_single_linkage_precomputed(&values, rows, options),
        }
    })
    .map(|result| {
        let hierarchy = result
            .hierarchy
            .into_iter()
            .map(|edge| {
                [
                    edge.left as f64,
                    edge.right as f64,
                    edge.distance,
                    edge.size as f64,
                ]
            })
            .collect();
        (result.labels, hierarchy)
    })
    .map_err(fit_error)
}

type RelativeValidityParts = (Vec<usize>, Vec<f64>, Vec<f64>, f64);

type NativeBranchCluster = (
    Vec<usize>,
    Vec<[f64; 4]>,
    Option<Vec<[f64; 4]>>,
    Option<TreeRows>,
    Vec<f64>,
    Option<Vec<Vec<usize>>>,
);
type NativeBranches = (
    Vec<i64>,
    Vec<f64>,
    Vec<i64>,
    Vec<f64>,
    Vec<f64>,
    Vec<NativeBranchCluster>,
);
type BranchConfiguration<'a> = (
    &'a str,
    bool,
    bool,
    usize,
    (&'a str, bool),
    (f64, f64, f64, f64),
);

#[pyclass(frozen, module = "hdbscan_alternative._native")]
struct _BranchDetectionData(BranchDetectionData);

#[pymethods]
impl _BranchDetectionData {
    #[new]
    fn new(
        py: Python<'_>,
        data: PyBuffer<f64>,
        labels: Vec<i64>,
        metric: (&str, f64),
        min_samples: usize,
        leaf_size: usize,
    ) -> PyResult<Self> {
        let &[_, n_features] = data.shape() else {
            return Err(PyValueError::new_err(
                "expected a two-dimensional feature matrix",
            ));
        };
        let metric = cached_tree_distance(metric)?;
        let values = data.to_vec(py)?;
        drop(data);
        py.detach(|| {
            BranchDetectionData::new_with_distance(
                &values,
                n_features,
                &labels,
                metric,
                min_samples,
                leaf_size,
            )
        })
        .map(Self)
        .map_err(|error| match error {
            FitError::InvalidMinSamples => PyValueError::new_err(
                "k must be less than or equal to the number of training points",
            ),
            _ => fit_error(error),
        })
    }

    fn metadata(&self) -> (Vec<usize>, Vec<i64>, Vec<f64>) {
        (
            self.0.finite_rows().to_vec(),
            self.0.neighbors().to_vec(),
            self.0.core_distances().to_vec(),
        )
    }

    fn detect(
        &self,
        py: Python<'_>,
        labels: Vec<i64>,
        probabilities: Vec<f64>,
        mst: Vec<[f64; 3]>,
        config: BranchConfiguration<'_>,
    ) -> PyResult<NativeBranches> {
        let (
            method,
            label_sides_as_branches,
            cluster_labels_overridden,
            min_cluster_size,
            (selection, allow_single_cluster),
            limits,
        ) = config;
        let method = match method {
            "full" => BranchDetectionMethod::Full,
            "core" => BranchDetectionMethod::Core,
            _ => return Err(PyValueError::new_err("invalid branch detection method")),
        };
        let options = BranchOptions {
            method,
            label_sides_as_branches,
            cluster_labels_overridden,
            min_cluster_size,
            selection: cluster_selection(selection, allow_single_cluster)?,
            thresholds: thresholds(limits),
        };
        let edges: Vec<_> = mst
            .into_iter()
            .map(|[source, target, distance]| SpanningEdge {
                source: source as usize,
                target: target as usize,
                distance,
            })
            .collect();
        let (result, diagnostics) = py.detach(|| {
            capture_selection_diagnostics(|| {
                self.0.detect(&labels, &probabilities, &edges, options)
            })
        });
        if !diagnostics.is_empty() {
            report_selection_diagnostics(py, diagnostics)?;
        }
        let result = result.map_err(|error| match error {
            FitError::InsufficientTreePoints => {
                PyValueError::new_err("Invalid shape in axis 0: 0.")
            }
            _ => fit_error(error),
        })?;
        let clusters = result
            .clusters
            .into_iter()
            .map(|group| {
                let graph = group
                    .graph
                    .into_iter()
                    .map(|edge| {
                        [
                            edge.source as f64,
                            edge.target as f64,
                            edge.centrality,
                            edge.reachability,
                        ]
                    })
                    .collect();
                let linkage = group.linkage.map(|linkage| {
                    linkage
                        .into_iter()
                        .map(|edge| {
                            [
                                edge.left as f64,
                                edge.right as f64,
                                edge.distance,
                                edge.size as f64,
                            ]
                        })
                        .collect()
                });
                let condensed = group.condensed.map(|condensed| {
                    condensed
                        .into_iter()
                        .map(|edge| (edge.parent, edge.child, edge.lambda, edge.child_size))
                        .collect()
                });
                (
                    group.points,
                    graph,
                    linkage,
                    condensed,
                    group.persistences,
                    group.exemplars,
                )
            })
            .collect();
        Ok((
            result.labels,
            result.probabilities,
            result.branch_labels,
            result.branch_probabilities,
            result.centralities,
            clusters,
        ))
    }
}

#[pyfunction]
fn _relative_validity_components(
    py: Python<'_>,
    labels: PyBuffer<i64>,
    tree: PyBuffer<f64>,
    sparse_precomputed: bool,
) -> PyResult<RelativeValidityParts> {
    if labels.dimensions() != 1 || !matches!(tree.shape(), [_, 3]) {
        return Err(PyValueError::new_err(
            "expected labels and a tree with three columns",
        ));
    }
    let labels = labels.to_vec(py)?;
    let values = tree.to_vec(py)?;
    let n_samples = labels.len();
    let tree = if sparse_precomputed {
        RelativeValidityTree::CsrPrecomputed
    } else {
        RelativeValidityTree::EdgeList
    };
    py.detach(move || {
        let edges: Vec<_> = values
            .chunks_exact(3)
            .map(|row| SpanningEdge {
                source: row[0] as usize,
                target: row[1] as usize,
                distance: row[2],
            })
            .collect();
        relative_validity_components(&labels, &edges, tree)
    })
    .map(|parts| {
        (
            parts.counts,
            parts.sparseness,
            parts.separation,
            parts.fallback,
        )
    })
    .map_err(|error| match error {
        RelativeValidityError::MissingEndpoint(point) => PyIndexError::new_err(format!(
            "index {point} is out of bounds for axis 0 with size {n_samples}"
        )),
        _ => PyValueError::new_err(error.to_string()),
    })
}

#[pyfunction]
fn _validity_internal_tree(
    py: Python<'_>,
    distances: PyBuffer<f64>,
) -> PyResult<(Vec<usize>, Vec<[f64; 3]>, bool)> {
    let &[rows, columns] = distances.shape() else {
        return Err(PyValueError::new_err(format!(
            "Buffer has wrong number of dimensions (expected 2, got {})",
            distances.dimensions(),
        )));
    };
    if !distances.is_c_contiguous() {
        return Err(PyValueError::new_err("ndarray is not C-contiguous"));
    }
    let values = distances.to_vec(py)?;
    drop(distances);
    let result = py
        .detach(|| internal_minimum_spanning_tree(&values, rows, columns))
        .map_err(|error| match error {
            ValidityError::MissingTreeNeighbor => PyAssertionError::new_err(()),
            ValidityError::InvalidTreeMask => PyIndexError::new_err(error.to_string()),
            ValidityError::Fit(error) => fit_error(error),
            _ => PyValueError::new_err(error.to_string()),
        })?;
    let edges = result
        .edges
        .iter()
        .map(|edge| [edge.source as f64, edge.target as f64, edge.distance])
        .collect();
    Ok((result.nodes, edges, result.fallback_node))
}

fn condensed_edges(tree: TreeRows) -> Vec<CondensedEdge> {
    tree.into_iter()
        .map(|(parent, child, lambda, child_size)| CondensedEdge {
            parent,
            child,
            lambda,
            child_size,
        })
        .collect()
}

#[pyclass(frozen, module = "hdbscan_alternative._native")]
struct _PredictionModel(PredictionModel);

type PredictionMetadata = (
    Vec<f64>,
    Vec<usize>,
    Vec<(usize, i64, f64)>,
    Vec<(usize, f64)>,
    Vec<Vec<usize>>,
);
type PredictionScores = (Vec<f64>, Vec<(usize, f64, f64)>);
type MembershipInputs = ((usize, usize, Vec<f64>), (usize, Vec<f64>), Vec<(f64, f64)>);

#[pymethods]
impl _PredictionModel {
    #[new]
    fn new(
        py: Python<'_>,
        data: PyBuffer<f64>,
        metric: (&str, f64),
        min_samples: usize,
        labels: Vec<i64>,
        tree: TreeRows,
    ) -> PyResult<Self> {
        let metric = cached_tree_distance(metric)?;
        let &[_, n_features] = data.shape() else {
            return Err(PyValueError::new_err(
                "expected a two-dimensional feature matrix",
            ));
        };
        let column_major = data.is_fortran_contiguous() && !data.is_c_contiguous();
        let values = data.to_vec(py)?;
        drop(data);
        py.detach(move || {
            let model = PredictionModel::new_with_distance(
                values,
                n_features,
                metric,
                min_samples,
                &labels,
                &condensed_edges(tree),
            )?;
            Ok(if column_major {
                model.with_column_major_training()
            } else {
                model
            })
        })
        .map(Self)
        .map_err(fit_error)
    }

    fn metadata(&self) -> PredictionMetadata {
        (
            self.0.core_distances().to_vec(),
            self.0.selected_clusters().to_vec(),
            self.0.cluster_mapping().collect(),
            self.0.leaf_max_lambdas().collect(),
            self.0.exemplars().to_vec(),
        )
    }

    fn predict(
        &self,
        py: Python<'_>,
        data: PyBuffer<f64>,
        min_samples: usize,
        tree: Option<TreeRows>,
    ) -> PyResult<(Vec<i64>, Vec<f64>, Vec<i64>)> {
        let values = data.to_vec(py)?;
        drop(data);
        py.detach(move || {
            let tree = tree.map(condensed_edges);
            let rows = self.0.predict(&values, min_samples, tree.as_deref())?;
            let mut labels = Vec::with_capacity(rows.len());
            let mut probabilities = Vec::with_capacity(rows.len());
            let mut points = Vec::with_capacity(rows.len());
            for row in rows {
                labels.push(row.label);
                probabilities.push(row.probability);
                points.push(row.connecting_point);
            }
            Ok((labels, probabilities, points))
        })
        .map_err(fit_error)
    }

    fn scores(
        &self,
        py: Python<'_>,
        data: PyBuffer<f64>,
        min_samples: usize,
        tree: Option<TreeRows>,
    ) -> PyResult<PredictionScores> {
        let values = data.to_vec(py)?;
        drop(data);
        py.detach(move || {
            let tree = tree.map(condensed_edges);
            let densities = self
                .0
                .predict_densities(&values, min_samples, tree.as_deref())?;
            let mut scores = Vec::with_capacity(densities.len());
            let mut exceptional = Vec::new();
            for (index, row) in densities.into_iter().enumerate() {
                let score = row.score();
                if !score.is_finite() {
                    exceptional.push((index, row.maximum, row.lambda));
                }
                scores.push(score);
            }
            Ok((scores, exceptional))
        })
        .map_err(fit_error)
    }

    fn membership_inputs(
        &self,
        py: Python<'_>,
        query: Option<(PyBuffer<f64>, usize)>,
        current_fit: Option<(TreeRows, Vec<i64>)>,
    ) -> PyResult<MembershipInputs> {
        let query = query
            .map(|(data, min_samples)| {
                let column_major = data.is_fortran_contiguous() && !data.is_c_contiguous();
                data.to_vec(py)
                    .map(|values| (values, min_samples, column_major))
            })
            .transpose()?;
        py.detach(move || {
            let current_fit = current_fit.map(|(tree, labels)| (condensed_edges(tree), labels));
            let components = self.0.membership_components(
                query
                    .as_ref()
                    .map(|(values, min_samples, column_major)| MembershipQuery {
                        values,
                        min_samples: *min_samples,
                        column_major: *column_major,
                    }),
                current_fit
                    .as_ref()
                    .map(|(tree, labels)| (tree.as_slice(), labels.as_slice())),
            )?;
            Ok((
                (
                    components.distance_rows,
                    components.distance_clusters,
                    components.distances,
                ),
                (components.clusters, components.outliers),
                components.densities,
            ))
        })
        .map_err(|error| match error {
            MembershipError::Fit(error) => fit_error(error),
            MembershipError::MissingDensity(node) => PyKeyError::new_err(node),
            _ => PyValueError::new_err(error.to_string()),
        })
    }
}

#[pyfunction]
fn _dbscan_clustering(
    py: Python<'_>,
    tree: PyBuffer<f64>,
    cut_distance: f64,
    min_cluster_size: &Bound<'_, PyAny>,
) -> PyResult<Vec<i64>> {
    let min_cluster_size = cut_minimum(min_cluster_size)?;
    if !matches!(tree.shape(), [_, 4]) {
        return Err(PyValueError::new_err(
            "expected four columns in a single linkage tree",
        ));
    }
    let values = tree.to_vec(py)?;
    py.detach(move || {
        let merges: Vec<_> = values
            .chunks_exact(4)
            .map(|row| LinkageMerge {
                left: row[0] as usize,
                right: row[1] as usize,
                distance: row[2],
                size: row[3] as usize,
            })
            .collect();
        dbscan_clustering(&merges, cut_distance, min_cluster_size)
    })
    .map_err(fit_error)
}

fn cut_minimum(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = value.py();
    let integer = if value.is_instance_of::<PyInt>() {
        value.clone()
    } else {
        let value_type = value.get_type();
        // The reference accepts the integer conversion slot and ignores __index__ alone.
        // SAFETY: value_type owns a reference to a Python type for the duration of the call.
        let slot = unsafe { ffi::PyType_GetSlot(value_type.as_type_ptr(), ffi::Py_nb_int) };
        if slot.is_null() {
            return Err(PyTypeError::new_err("an integer is required"));
        }
        py.get_type::<PyInt>().call1((value,))?
    };
    integer.extract::<i64>().map_err(|error| {
        if error.is_instance_of::<PyOverflowError>(py) {
            let message = if size_of::<std::ffi::c_long>() == size_of::<i64>() {
                "Python int too large to convert to C long"
            } else {
                "int too big to convert"
            };
            PyOverflowError::new_err(message)
        } else {
            error
        }
    })
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", hdbscan_core::VERSION)?;
    module.add_function(wrap_pyfunction!(_fit_dense_with_diagnostics, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_csr_with_diagnostics, module)?)?;
    module.add_function(wrap_pyfunction!(_emit_fit_diagnostics, module)?)?;
    module.add_function(wrap_pyfunction!(_outlier_scores, module)?)?;
    module.add_function(wrap_pyfunction!(_robust_single_linkage, module)?)?;
    module.add_class::<_BranchDetectionData>()?;
    module.add_function(wrap_pyfunction!(_relative_validity_components, module)?)?;
    module.add_function(wrap_pyfunction!(_validity_internal_tree, module)?)?;
    module.add_class::<_PredictionModel>()?;
    module.add_function(wrap_pyfunction!(_dbscan_clustering, module)?)
}
