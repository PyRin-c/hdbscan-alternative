//! Core crate for the independent Rust implementation of HDBSCAN.

mod boruvka;
pub mod branches;
mod cluster;
mod distance;
mod hierarchy;
mod missing;
mod ordering;
mod postfit;
mod prediction;
pub mod robust;
mod sparse;
mod spatial;
pub mod validity;

use std::fmt;

pub use boruvka::{BoruvkaOptions, BoruvkaTree, fit_dense_boruvka, fit_dense_tree_boruvka};
#[doc(hidden)]
pub use cluster::{SelectionDiagnostic, capture_selection_diagnostics};
pub use missing::{
    fit_csr_manhattan_f32_with_missing, fit_csr_with_missing, fit_dense_euclidean_strided_columns,
    fit_dense_with_missing,
};
pub use postfit::{dbscan_clustering, outlier_scores};
pub use prediction::{
    MembershipComponents, MembershipError, MembershipMatrix, MembershipQuery, Prediction,
    PredictionDensity, PredictionModel,
};
pub use sparse::{fit_csr_precomputed, fit_csr_precomputed_configured};
pub use spatial::TreeDistance;

/// Version of the Rust implementation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Distance used to fit a model from features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureMetric {
    Euclidean,
    Manhattan,
    Cosine,
}

/// Settings that determine the condensed hierarchy.
#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    pub min_cluster_size: usize,
    /// Defaults to `min_cluster_size`. Values above the number of other points are clipped.
    pub min_samples: Option<usize>,
    /// Divides pair distances before the mutual reachability maximum.
    pub alpha: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: None,
            alpha: 1.0,
        }
    }
}

/// How to select flat clusters from the condensed hierarchy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClusterSelectionMethod {
    #[default]
    Eom,
    Leaf,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ClusterSelection {
    pub method: ClusterSelectionMethod,
    /// Includes the root in EOM and lets epsilon merging reach it.
    pub allow_single_cluster: bool,
}

/// Thresholds applied to the condensed hierarchy before or during cluster selection.
#[derive(Clone, Copy, Debug)]
pub struct ClusterSelectionThresholds {
    /// Merge clusters below this distance threshold. Defaults to zero.
    pub epsilon: f64,
    /// Reject EOM candidates above this distance threshold. Defaults to infinity.
    pub epsilon_max: f64,
    /// Merge short-lived branches before selecting clusters. Defaults to zero.
    pub persistence: f64,
    /// Reject EOM candidates above this size. Nonpositive values disable the limit.
    pub max_cluster_size: f64,
}

impl Default for ClusterSelectionThresholds {
    fn default() -> Self {
        Self {
            epsilon: 0.0,
            epsilon_max: f64::INFINITY,
            persistence: 0.0,
            max_cluster_size: 0.0,
        }
    }
}

impl ClusterSelectionThresholds {
    fn validate(self) -> Result<(), FitError> {
        if self.epsilon < 0.0 {
            return Err(FitError::InvalidClusterSelectionEpsilon);
        }
        if self.persistence < 0.0 {
            return Err(FitError::InvalidClusterSelectionPersistence);
        }
        if self.epsilon_max < self.epsilon {
            return Err(FitError::InvalidClusterSelectionEpsilonMax);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FitConfiguration {
    pub options: FitOptions,
    pub cluster_selection: ClusterSelection,
    pub cluster_selection_thresholds: ClusterSelectionThresholds,
    /// Require a point to leave a non-root cluster after the cluster's birth.
    pub strict_cluster_membership: bool,
}

impl FitConfiguration {
    pub fn with_cluster_selection_thresholds(
        mut self,
        thresholds: ClusterSelectionThresholds,
    ) -> Self {
        self.cluster_selection_thresholds = thresholds;
        self
    }
}

impl FitOptions {
    pub fn with_cluster_selection(self, cluster_selection: ClusterSelection) -> FitConfiguration {
        FitConfiguration {
            options: self,
            cluster_selection,
            cluster_selection_thresholds: ClusterSelectionThresholds::default(),
            strict_cluster_membership: false,
        }
    }
}

impl From<FitOptions> for FitConfiguration {
    fn from(options: FitOptions) -> Self {
        options.with_cluster_selection(ClusterSelection::default())
    }
}

/// An edge in the mutual reachability minimum spanning tree.
#[derive(Clone, Copy, Debug)]
pub struct SpanningEdge {
    pub source: usize,
    pub target: usize,
    pub distance: f64,
}

/// A single linkage merge. Leaves are numbered `0..n_samples`, then merges in row order.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LinkageMerge {
    pub left: usize,
    pub right: usize,
    pub distance: f64,
    pub size: usize,
}

/// An edge of the condensed hierarchy. The root has ID `n_samples`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CondensedEdge {
    pub parent: usize,
    pub child: usize,
    pub lambda: f64,
    pub child_size: usize,
}

/// Output of fitting an exact HDBSCAN model. Noise has label `-1`.
#[derive(Debug)]
pub struct FitResult {
    pub labels: Vec<i64>,
    pub probabilities: Vec<f64>,
    pub cluster_persistence: Vec<f64>,
    pub minimum_spanning_tree: Vec<SpanningEdge>,
    pub single_linkage_tree: Vec<LinkageMerge>,
    pub condensed_tree: Vec<CondensedEdge>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitError {
    InvalidShape,
    InvalidHierarchy,
    LengthMismatch,
    InvalidMinClusterSize,
    InvalidMinSamples,
    InvalidLeafSize,
    NeighborWorkerUnavailable,
    InvalidAlpha,
    InvalidClusterSelectionEpsilon,
    InvalidClusterSelectionEpsilonMax,
    InvalidClusterSelectionPersistence,
    EmptyPersistenceHierarchy,
    EmptyBranchTree,
    ZeroBranchWeight,
    BranchLabelShape { values: usize, rows: usize },
    ZeroLambdaEpsilon,
    NonFiniteInput,
    NoFiniteFeatureRows { n_features: usize },
    OneFiniteFeatureRow,
    InsufficientTreePoints,
    UnsupportedMetric,
    InvalidDistanceOptions,
    CondensedTreeCapacity { capacity: usize, required: u128 },
    DistanceOverflow,
    NaNDistance,
    DisconnectedDistances,
    InvalidCsr,
    DisconnectedSparseGraph,
    InsufficientSparseNeighbors { min_samples: usize },
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidShape => {
                "expected at least two samples and one feature with representable dimensions"
            }
            Self::InvalidHierarchy => "invalid hierarchy node IDs",
            Self::LengthMismatch => "input length does not match the matrix shape",
            Self::InvalidMinClusterSize => "min_cluster_size must be at least 2",
            Self::InvalidMinSamples => "min_samples must be positive",
            Self::InvalidLeafSize => "Leaf size must be greater than 0!",
            Self::NeighborWorkerUnavailable => {
                "could not run the worker for the nearest-neighbor query"
            }
            Self::InvalidAlpha => "alpha must be greater than zero",
            Self::InvalidClusterSelectionEpsilon => {
                "Epsilon must be a float value greater than or equal to 0!"
            }
            Self::InvalidClusterSelectionEpsilonMax => {
                "Cluster selection epsilon max must be greater than epsilon!"
            }
            Self::InvalidClusterSelectionPersistence => {
                "Persistence must be a float value greater than or equal to 0!"
            }
            Self::EmptyPersistenceHierarchy => "index -1 is out of bounds for axis 0 with size 0",
            Self::EmptyBranchTree => {
                "zero-size array to reduction operation maximum which has no identity"
            }
            Self::ZeroBranchWeight => "Weights sum to zero, can't be normalized",
            Self::BranchLabelShape { values, rows } => {
                return write!(
                    f,
                    "shape mismatch: value array of shape ({values},) could not be broadcast to indexing result of shape ({rows},)"
                );
            }
            Self::ZeroLambdaEpsilon => "float division",
            Self::NonFiniteInput => "all input values must be finite",
            Self::NoFiniteFeatureRows { n_features } => {
                return write!(
                    f,
                    "Found array with 0 sample(s) (shape=(0, {n_features})) while a minimum of 1 is required."
                );
            }
            Self::OneFiniteFeatureRow => "Invalid shape in axis 0: 0.",
            Self::InsufficientTreePoints => {
                "k must be less than or equal to the number of training points"
            }
            Self::UnsupportedMetric => "unsupported distance metric for this algorithm",
            Self::InvalidDistanceOptions => {
                "invalid distance parameters for the feature dimensions"
            }
            Self::CondensedTreeCapacity { capacity, required } => {
                return write!(
                    f,
                    "could not broadcast input array from shape ({capacity},) into shape ({required},)"
                );
            }
            Self::DistanceOverflow => "distance calculation exceeded float64 range",
            Self::NaNDistance => "precomputed distance matrix contains NaN",
            Self::DisconnectedDistances => {
                "finite mutual reachability distances do not connect all samples; provide more distances or reduce min_samples"
            }
            Self::InvalidCsr => "invalid CSR row offsets, column indices, or data lengths",
            Self::DisconnectedSparseGraph => {
                "Sparse distance matrix has multiple connected components!\n\
                 That is, there exist groups of points that are completely disjoint -- there are no distance relations connecting them\n\
                 Run hdbscan on each component."
            }
            Self::InsufficientSparseNeighbors { min_samples } => {
                return write!(
                    f,
                    "There exists points with less than {min_samples} neighbors. Ensure your distance matrix has non zeros values for at least `min_sample`={min_samples} neighbors for each points (i.e. K-nn graph), or specify a `max_dist` to use when distances are missing."
                );
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for FitError {}

/// Fits HDBSCAN to a row-major matrix of finite Euclidean features.
///
/// Computes an exact minimum spanning tree. `FitOptions` alone selects clusters
/// by Excess of Mass below the root.
pub fn fit_dense_euclidean(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    options: FitOptions,
) -> Result<FitResult, FitError> {
    fit_dense(
        values,
        n_samples,
        n_features,
        FeatureMetric::Euclidean,
        options,
    )
}

/// Fits exact HDBSCAN to a row-major matrix of finite features.
pub fn fit_dense(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    metric: FeatureMetric,
    options: FitOptions,
) -> Result<FitResult, FitError> {
    fit_dense_configured(values, n_samples, n_features, metric, options.into())
}

/// Fits dense features with the specified distance and cluster selection.
pub fn fit_dense_configured(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    metric: FeatureMetric,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    let min_samples = validate_fit(values.len(), n_samples, n_features, configuration.options)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FitError::NonFiniteInput);
    }

    match metric {
        FeatureMetric::Euclidean => {
            let distances = distance::DenseDistances::new(values, n_features)?;
            fit_euclidean_distances(distances, n_samples, min_samples, configuration)
        }
        FeatureMetric::Manhattan => {
            let row = |sample| &values[sample * n_features..(sample + 1) * n_features];
            fit_distances(
                n_samples,
                min_samples,
                configuration,
                DistanceBounds::Unbounded,
                |left, right| distance::dense_manhattan(row(left), row(right)),
            )
        }
        FeatureMetric::Cosine => {
            let normalized = distance::normalize_dense(values, n_features);
            let row = |sample| &normalized[sample * n_features..(sample + 1) * n_features];
            fit_distances(
                n_samples,
                min_samples,
                configuration,
                DistanceBounds::Unbounded,
                |left, right| {
                    Ok(if left == right {
                        0.0
                    } else {
                        distance::cosine_from_dot(distance::dense_dot(row(left), row(right)))
                    })
                },
            )
        }
    }
}

fn fit_euclidean_distances(
    distances: distance::DenseDistances<'_>,
    n_samples: usize,
    min_samples: usize,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    fit_distances(
        n_samples,
        min_samples,
        configuration,
        DistanceBounds::Unbounded,
        |left, right| distances.between(left, right),
    )
}

/// Fits the exact Prim vector path for Euclidean or Manhattan features.
///
/// Non-finite rows are excluded and restored in the output. Computes neighbors
/// by direct distance evaluation and returns the MST in single-linkage order.
pub fn fit_dense_prims(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    metric: FeatureMetric,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    fit_dense_tree_prims(
        values,
        n_samples,
        n_features,
        metric.try_into()?,
        configuration,
    )
}

/// Fits the exact Prim vector path with the supplied distance.
pub fn fit_dense_tree_prims(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    metric: TreeDistance<'_>,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    let metric = metric.validated(n_features)?;
    fit_vector_with_missing(
        values,
        n_samples,
        n_features,
        configuration,
        |values, n_samples, min_samples| {
            let row = |sample| &values[sample * n_features..(sample + 1) * n_features];
            let between = |left, right| Ok(metric.distance(metric.reduced(row(left), row(right))));
            let core = distance::core_distances(n_samples, min_samples, between)?;
            hierarchy::prim_tree(&core, configuration.options.alpha, between, f64::MAX)
        },
    )
}

fn fit_vector_with_missing(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    configuration: FitConfiguration,
    spanning_tree: impl Fn(&[f64], usize, usize) -> Result<Vec<SpanningEdge>, FitError>,
) -> Result<FitResult, FitError> {
    let mut validation = configuration.options;
    if validation.alpha.is_nan() {
        validation.alpha = 1.0;
    }
    let fit =
        missing::fit_dense_rows_with_missing(values, n_samples, n_features, |values, n_samples| {
            if n_samples == 1 && n_features > 0 && values.len() == n_features {
                return if values.iter().all(|value| value.is_finite()) {
                    Err(FitError::InsufficientTreePoints)
                } else {
                    Err(FitError::NonFiniteInput)
                };
            }
            let min_samples = validate_fit(values.len(), n_samples, n_features, validation)?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(FitError::NonFiniteInput);
            }
            configuration.cluster_selection_thresholds.validate()?;
            let edges = spanning_tree(values, n_samples, min_samples)?;
            let ordered: Vec<_> = hierarchy::edges_in_linkage_order(&edges).collect();
            let linkage = hierarchy::single_linkage(ordered.iter().copied(), n_samples);
            finish_fit(n_samples, configuration, ordered, linkage)
        });
    match fit {
        Err(FitError::OneFiniteFeatureRow) => Err(FitError::InsufficientTreePoints),
        _ => fit,
    }
}

/// Fits exact HDBSCAN to a CSR matrix of finite Euclidean features.
///
/// The sample count is `indptr.len() - 1`. Each row retains its stored order
/// and duplicate entries for the pairwise distance calculation.
pub fn fit_csr_euclidean(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    n_features: usize,
    options: FitOptions,
) -> Result<FitResult, FitError> {
    fit_csr(
        indptr,
        indices,
        data,
        n_features,
        FeatureMetric::Euclidean,
        options,
    )
}

/// Fits exact HDBSCAN to a CSR matrix of finite features.
///
/// Manhattan combines duplicate columns. Euclidean and cosine retain the stored entries.
pub fn fit_csr(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    n_features: usize,
    metric: FeatureMetric,
    options: FitOptions,
) -> Result<FitResult, FitError> {
    fit_csr_configured(indptr, indices, data, n_features, metric, options.into())
}

/// Fits CSR features with the specified distance and cluster selection.
pub fn fit_csr_configured(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    n_features: usize,
    metric: FeatureMetric,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    let n_samples = sparse::validate_csr(indptr, indices, data.len(), n_features)?;
    let min_samples = validate_options(n_samples, configuration.options)?;
    if data.iter().any(|value| !value.is_finite()) {
        return Err(FitError::NonFiniteInput);
    }
    match metric {
        FeatureMetric::Euclidean => {
            let distances = distance::CsrDistances::new(indptr, indices, data)?;
            fit_distances(
                n_samples,
                min_samples,
                configuration,
                DistanceBounds::Unbounded,
                |left, right| distances.between(left, right),
            )
        }
        FeatureMetric::Manhattan => {
            let distances = distance::CanonicalCsr::new(indptr, indices, data);
            fit_distances(
                n_samples,
                min_samples,
                configuration,
                DistanceBounds::Unbounded,
                |left, right| distances.manhattan_between(left, right),
            )
        }
        FeatureMetric::Cosine => {
            let normalized = distance::normalize_csr(indptr, data);
            let dots = distance::CsrDotProducts::new(indptr, indices, &normalized);
            fit_distances(
                n_samples,
                min_samples,
                configuration,
                DistanceBounds::Unbounded,
                |left, right| {
                    Ok(if left == right {
                        0.0
                    } else {
                        distance::cosine_from_dot(dots.between(left, right))
                    })
                },
            )
        }
    }
}

/// Fits a float32 CSR matrix with rounded float32 differences and a float64 sum.
pub fn fit_csr_manhattan_f32(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    n_features: usize,
    options: FitOptions,
) -> Result<FitResult, FitError> {
    fit_csr_manhattan_f32_configured(indptr, indices, data, n_features, options.into())
}

/// Fits a float32 CSR matrix with Manhattan distance and cluster selection options.
pub fn fit_csr_manhattan_f32_configured(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    n_features: usize,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    let n_samples = sparse::validate_csr(indptr, indices, data.len(), n_features)?;
    let min_samples = validate_options(n_samples, configuration.options)?;
    if data.iter().any(|value| !value.is_finite()) {
        return Err(FitError::NonFiniteInput);
    }
    let distances = distance::CanonicalCsr::new(indptr, indices, data);
    fit_distances(
        n_samples,
        min_samples,
        configuration,
        DistanceBounds::Unbounded,
        |left, right| distances.manhattan_between(left, right),
    )
}

/// Fits exact HDBSCAN to a square, row-major matrix of precomputed distances.
///
/// The entries are used as stored, including the diagonal and any asymmetry.
/// Positive infinity represents a missing distance. Returns an error if the
/// finite mutual reachability distances cannot connect all samples.
pub fn fit_dense_precomputed(
    values: &[f64],
    n_samples: usize,
    options: FitOptions,
) -> Result<FitResult, FitError> {
    fit_dense_precomputed_configured(values, n_samples, options.into())
}

/// Fits a square, row-major distance matrix with cluster selection options.
pub fn fit_dense_precomputed_configured(
    values: &[f64],
    n_samples: usize,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    fit_precomputed(values, n_samples, configuration, DistanceBounds::Finite)
}

/// Fits a square matrix with the Python API's bounded MST construction.
/// It retains the initial edge rows when no unused point can be reached.
pub fn fit_dense_precomputed_bounded(
    values: &[f64],
    n_samples: usize,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    fit_precomputed(values, n_samples, configuration, DistanceBounds::Repeat)
}

fn fit_precomputed(
    values: &[f64],
    n_samples: usize,
    configuration: FitConfiguration,
    bounds: DistanceBounds,
) -> Result<FitResult, FitError> {
    let min_samples = validate_fit(values.len(), n_samples, n_samples, configuration.options)?;
    if values.iter().any(|value| value.is_nan()) {
        return Err(FitError::NaNDistance);
    }
    fit_distances(
        n_samples,
        min_samples,
        configuration,
        bounds,
        |left, right| Ok(values[left * n_samples + right]),
    )
}

/// Fits the generic feature path using a distance matrix of the finite rows.
pub fn fit_pairwise_feature_distances(
    values: &[f64],
    finite_rows: &[bool],
    n_features: usize,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    let n_samples = finite_rows.iter().filter(|&&finite| finite).count();
    let min_samples = validate_fit(values.len(), n_samples, n_samples, configuration.options)?;
    let fitted = fit_distances(
        n_samples,
        min_samples,
        configuration,
        DistanceBounds::Unbounded,
        |left, right| Ok(values[left * n_samples + right]),
    )?;
    if n_samples == finite_rows.len() {
        Ok(fitted)
    } else {
        Ok(missing::Rows::new(finite_rows.iter().copied(), n_features)?.restore(fitted))
    }
}

fn validate_fit(
    len: usize,
    n_samples: usize,
    n_features: usize,
    options: FitOptions,
) -> Result<usize, FitError> {
    if !valid_sample_count(n_samples) || n_features == 0 {
        return Err(FitError::InvalidShape);
    }
    if n_samples.checked_mul(n_features) != Some(len) {
        return Err(FitError::LengthMismatch);
    }
    validate_options(n_samples, options)
}

fn valid_sample_count(n_samples: usize) -> bool {
    (2..=isize::MAX as usize / 2).contains(&n_samples)
}

fn validate_options(n_samples: usize, options: FitOptions) -> Result<usize, FitError> {
    if options.min_cluster_size < 2 {
        return Err(FitError::InvalidMinClusterSize);
    }
    let min_samples = options.min_samples.unwrap_or(options.min_cluster_size);
    if min_samples == 0 {
        return Err(FitError::InvalidMinSamples);
    }
    if options.alpha.is_nan() || options.alpha <= 0.0 {
        return Err(FitError::InvalidAlpha);
    }
    Ok(min_samples.min(n_samples - 1))
}

enum DistanceBounds {
    Unbounded,
    Finite,
    Repeat,
}

fn fit_distances(
    n_samples: usize,
    min_samples: usize,
    configuration: FitConfiguration,
    bounds: DistanceBounds,
    distance: impl Fn(usize, usize) -> Result<f64, FitError>,
) -> Result<FitResult, FitError> {
    configuration.cluster_selection_thresholds.validate()?;
    let core_distances = distance::core_distances(n_samples, min_samples, &distance)?;
    let alpha = configuration.options.alpha;
    let minimum_spanning_tree = match bounds {
        DistanceBounds::Repeat => hierarchy::prim_tree(&core_distances, alpha, distance, f64::MAX),
        _ => hierarchy::minimum_spanning_tree(&core_distances, alpha, distance),
    }?;
    if matches!(bounds, DistanceBounds::Finite)
        && minimum_spanning_tree
            .iter()
            .any(|edge| edge.distance == f64::INFINITY)
    {
        return Err(FitError::DisconnectedDistances);
    }
    let single_linkage_tree = hierarchy::single_linkage(
        hierarchy::edges_in_linkage_order(&minimum_spanning_tree),
        n_samples,
    );
    finish_fit(
        n_samples,
        configuration,
        minimum_spanning_tree,
        single_linkage_tree,
    )
}

fn finish_fit(
    n_samples: usize,
    configuration: FitConfiguration,
    minimum_spanning_tree: Vec<SpanningEdge>,
    single_linkage_tree: Vec<LinkageMerge>,
) -> Result<FitResult, FitError> {
    let thresholds = configuration.cluster_selection_thresholds;
    let min_cluster_size = configuration.options.min_cluster_size;
    let mut condensed_tree = if minimum_spanning_tree
        .iter()
        .any(|edge| edge.source == edge.target)
    {
        cluster::condense_repeated(&single_linkage_tree, n_samples, min_cluster_size)?
    } else {
        cluster::condense(&single_linkage_tree, n_samples, min_cluster_size)
    };
    if thresholds.persistence > 0.0 {
        cluster::simplify(&mut condensed_tree, n_samples, thresholds.persistence)?;
    }
    let flat = cluster::select_clusters(
        &condensed_tree,
        n_samples,
        configuration.cluster_selection,
        thresholds,
        configuration.strict_cluster_membership,
    )?;

    Ok(FitResult {
        labels: flat.labels,
        probabilities: flat.probabilities,
        cluster_persistence: flat.persistence,
        minimum_spanning_tree,
        single_linkage_tree,
        condensed_tree,
    })
}
