//! C ABI for the independent Rust implementation of HDBSCAN.

use std::ffi::c_char;
use std::panic::catch_unwind;
use std::{ptr, slice};

use hdbscan_core::{
    BoruvkaOptions, BoruvkaTree, ClusterSelection, ClusterSelectionMethod,
    ClusterSelectionThresholds, CondensedEdge, FeatureMetric, FitConfiguration, FitError,
    FitOptions, FitResult, LinkageMerge, MembershipError, Prediction, PredictionModel,
    SpanningEdge, TreeDistance,
    branches::{
        ApproximationEdge, BranchDetectionData, BranchDetectionMethod, BranchOptions,
        BranchPrediction, BranchResult,
    },
    dbscan_clustering, fit_csr as fit_csr_features, fit_csr_configured, fit_csr_precomputed,
    fit_csr_precomputed_configured, fit_csr_with_missing, fit_dense, fit_dense_boruvka,
    fit_dense_configured, fit_dense_precomputed, fit_dense_precomputed_configured, fit_dense_prims,
    fit_dense_tree_boruvka, fit_dense_tree_prims, fit_dense_with_missing, outlier_scores,
    robust::{
        RobustLinkageAlgorithm, RobustLinkageOptions, RobustLinkageResult,
        robust_single_linkage_dense, robust_single_linkage_precomputed,
        robust_single_linkage_spatial,
    },
    validity::{
        RelativeValidityTree, ValidityError, relative_validity, validity_index_dense,
        validity_index_precomputed,
    },
};

const PRECOMPUTED_METRIC: u32 = 3;

#[repr(u32)]
enum Status {
    Ok = 0,
    NullArgument = 1,
    InvalidPointer = 2,
    InvalidLength = 3,
    InvalidShape = 4,
    InvalidOptions = 5,
    NonFiniteInput = 6,
    DistanceOverflow = 7,
    Panic = 8,
    DisconnectedDistances = 9,
    InvalidCsr = 10,
    InsufficientSparseNeighbors = 11,
    InvalidMetric = 12,
}

fn feature_metric(metric: u32) -> Result<FeatureMetric, Status> {
    match metric {
        0 => Ok(FeatureMetric::Euclidean),
        1 => Ok(FeatureMetric::Manhattan),
        2 => Ok(FeatureMetric::Cosine),
        _ => Err(Status::InvalidMetric),
    }
}

impl From<FitError> for Status {
    fn from(error: FitError) -> Self {
        match error {
            FitError::InvalidShape
            | FitError::InvalidHierarchy
            | FitError::EmptyBranchTree
            | FitError::BranchLabelShape { .. }
            | FitError::CondensedTreeCapacity { .. }
            | FitError::NoFiniteFeatureRows { .. }
            | FitError::OneFiniteFeatureRow
            | FitError::InsufficientTreePoints => Self::InvalidShape,
            FitError::UnsupportedMetric => Self::InvalidMetric,
            FitError::LengthMismatch => Self::InvalidLength,
            FitError::InvalidMinClusterSize
            | FitError::InvalidMinSamples
            | FitError::InvalidDistanceOptions
            | FitError::InvalidLeafSize
            | FitError::NeighborWorkerUnavailable
            | FitError::InvalidAlpha
            | FitError::InvalidClusterSelectionEpsilon
            | FitError::InvalidClusterSelectionEpsilonMax
            | FitError::InvalidClusterSelectionPersistence
            | FitError::EmptyPersistenceHierarchy
            | FitError::ZeroBranchWeight
            | FitError::ZeroLambdaEpsilon => Self::InvalidOptions,
            FitError::NonFiniteInput | FitError::NaNDistance => Self::NonFiniteInput,
            FitError::DistanceOverflow => Self::DistanceOverflow,
            FitError::DisconnectedDistances | FitError::DisconnectedSparseGraph => {
                Self::DisconnectedDistances
            }
            FitError::InvalidCsr => Self::InvalidCsr,
            FitError::InsufficientSparseNeighbors { .. } => Self::InsufficientSparseNeighbors,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HdbscanCapiOptions {
    min_cluster_size: usize,
    min_samples: usize,
    alpha: f64,
}

impl From<HdbscanCapiOptions> for FitOptions {
    fn from(options: HdbscanCapiOptions) -> Self {
        Self {
            min_cluster_size: options.min_cluster_size,
            min_samples: (options.min_samples != 0).then_some(options.min_samples),
            alpha: options.alpha,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HdbscanCapiFitConfiguration {
    options: HdbscanCapiOptions,
    cluster_selection_method: u32,
    allow_single_cluster: u32,
    max_dist: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HdbscanCapiClusterSelectionThresholds {
    epsilon: f64,
    epsilon_max: f64,
    persistence: f64,
    max_cluster_size: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HdbscanCapiBoruvkaOptions {
    tree: u32,
    approximate: u32,
    leaf_size: usize,
    core_dist_n_jobs: i64,
}

#[repr(C)]
pub struct HdbscanCapiTreeDistance {
    kind: u32,
    p: f64,
    parameters: *const f64,
    parameter_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HdbscanCapiCsr {
    indptr: *const usize,
    indices: *const usize,
    data: *const f64,
    n_samples: usize,
    nnz: usize,
}

#[repr(C)]
pub struct HdbscanCapiResultView {
    labels: *const i64,
    probabilities: *const f64,
    n_samples: usize,
    cluster_persistence: *const f64,
    n_clusters: usize,
}

pub struct HdbscanCapiResult {
    labels: Vec<i64>,
    probabilities: Vec<f64>,
    cluster_persistence: Vec<f64>,
    condensed_tree: Vec<CondensedEdge>,
    single_linkage_tree: Vec<LinkageMerge>,
    minimum_spanning_tree: Vec<SpanningEdge>,
    relative_validity_tree: RelativeValidityTree,
}

pub struct HdbscanCapiPredictionModel {
    model: PredictionModel,
    n_features: usize,
    min_samples: usize,
}

#[repr(C)]
pub struct HdbscanCapiMembershipShape {
    n_samples: usize,
    n_clusters: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HdbscanCapiRobustOptions {
    cut: f64,
    k: usize,
    alpha: f64,
    gamma: i64,
    algorithm: u32,
    leaf_size: usize,
    core_dist_n_jobs: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HdbscanCapiBranchOptions {
    min_samples: usize,
    min_cluster_size: usize,
    method: u32,
    label_sides_as_branches: u32,
    cluster_selection_method: u32,
    allow_single_cluster: u32,
}

#[repr(C)]
pub struct HdbscanCapiBranchResultView {
    n_samples: usize,
    n_clusters: usize,
    labels: *const i64,
    probabilities: *const f64,
    cluster_labels: *const i64,
    cluster_probabilities: *const f64,
    branch_labels: *const i64,
    branch_probabilities: *const f64,
    centralities: *const f64,
}

#[repr(C)]
pub struct HdbscanCapiBranchClusterView {
    points: *const usize,
    n_points: usize,
    graph: *const ApproximationEdge,
    n_edges: usize,
    linkage: *const LinkageMerge,
    n_merges: usize,
    condensed: *const CondensedEdge,
    n_condensed: usize,
    persistences: *const f64,
    n_persistences: usize,
}

/// Returns the library version as a borrowed NUL-terminated string.
#[unsafe(no_mangle)]
pub extern "C" fn hdbscan_capi_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Fits an exact dense Euclidean model and returns an owning result handle.
///
/// # Safety
/// A non-null `data` must be valid for reads of `data_len` initialized doubles
/// in one allocation for the duration of the call. A non-null `options` must
/// point to a readable options struct. A non-null `out` must be valid for
/// writing a pointer and must not alias the inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_dense_euclidean(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    options: *const HdbscanCapiOptions,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        hdbscan_capi_fit_dense_with_metric(data, data_len, n_samples, n_features, 0, options, out)
    }
}

/// Fits an exact model from dense features using the selected distance metric.
///
/// # Safety
/// The input, options, and output pointers follow the same contract as
/// [`hdbscan_capi_fit_dense_euclidean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_dense_with_metric(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    metric: u32,
    options: *const HdbscanCapiOptions,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit(options, out, false, |options| {
            let values = input_slice(data, data_len)?;
            Ok(fit_dense(
                values,
                n_samples,
                n_features,
                feature_metric(metric)?,
                options,
            )?)
        })
    }
}

/// Fits an exact model from a square, row-major distance matrix.
///
/// # Safety
/// The input, options, and output pointers follow the same contract as
/// [`hdbscan_capi_fit_dense_euclidean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_dense_precomputed(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    options: *const HdbscanCapiOptions,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit(options, out, false, |options| {
            let values = input_slice(data, data_len)?;
            Ok(fit_dense_precomputed(values, n_samples, options)?)
        })
    }
}

/// Fits a square precomputed distance matrix in CSR format.
///
/// # Safety
/// A non-null `matrix` must point to a readable CSR view. Its buffers must be
/// readable for `n_samples + 1` row offsets and `nnz` columns and values. Each
/// buffer must be in one allocation and valid for the duration of the call.
/// `options` and `out` follow the contract of [`hdbscan_capi_fit_dense_euclidean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_csr_precomputed(
    matrix: *const HdbscanCapiCsr,
    options: *const HdbscanCapiOptions,
    max_dist: f64,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit_csr(
            matrix,
            options,
            out,
            true,
            |indptr, indices, data, options| {
                fit_csr_precomputed(indptr, indices, data, options, max_dist).map_err(Status::from)
            },
        )
    }
}

/// Fits an exact Euclidean model from a CSR feature matrix.
///
/// # Safety
/// The CSR view, options, and output pointers follow the contract of
/// [`hdbscan_capi_fit_csr_precomputed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_csr_euclidean(
    matrix: *const HdbscanCapiCsr,
    n_features: usize,
    options: *const HdbscanCapiOptions,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe { hdbscan_capi_fit_csr_with_metric(matrix, n_features, 0, options, out) }
}

/// Fits an exact model from CSR features using the selected distance metric.
///
/// # Safety
/// The CSR view, options, and output pointers follow the contract of
/// [`hdbscan_capi_fit_csr_precomputed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_csr_with_metric(
    matrix: *const HdbscanCapiCsr,
    n_features: usize,
    metric: u32,
    options: *const HdbscanCapiOptions,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit_csr(
            matrix,
            options,
            out,
            false,
            |indptr, indices, data, options| {
                Ok(fit_csr_features(
                    indptr,
                    indices,
                    data,
                    n_features,
                    feature_metric(metric)?,
                    options,
                )?)
            },
        )
    }
}

/// Fits a dense feature or distance matrix with cluster selection options.
///
/// # Safety
/// `data` and `out` follow the contract of [`hdbscan_capi_fit_dense_euclidean`].
/// A non-null `configuration` must point to a readable configuration for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_dense_configured(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    metric: u32,
    configuration: *const HdbscanCapiFitConfiguration,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        hdbscan_capi_fit_dense_with_thresholds(
            data,
            data_len,
            n_samples,
            n_features,
            metric,
            configuration,
            ptr::null(),
            out,
        )
    }
}

/// Fits a dense feature or distance matrix with cluster selection thresholds.
///
/// # Safety
/// The pointers follow the contract of [`hdbscan_capi_fit_dense_configured`].
/// A non-null `thresholds` must point to readable thresholds for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_dense_with_thresholds(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    metric: u32,
    configuration: *const HdbscanCapiFitConfiguration,
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit_result(out, false, || {
            let (configuration, _) = read_fit_configuration(configuration, thresholds)?;
            let values = input_slice(data, data_len)?;
            if metric == PRECOMPUTED_METRIC {
                if n_samples != n_features {
                    return Err(Status::InvalidShape);
                }
                Ok(fit_dense_precomputed_configured(
                    values,
                    n_samples,
                    configuration,
                )?)
            } else {
                Ok(fit_dense_configured(
                    values,
                    n_samples,
                    n_features,
                    feature_metric(metric)?,
                    configuration,
                )?)
            }
        })
    }
}

/// Fits a CSR feature or distance matrix with cluster selection options.
///
/// # Safety
/// `matrix` and `out` follow the contract of [`hdbscan_capi_fit_csr_precomputed`].
/// A non-null `configuration` must point to a readable configuration for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_csr_configured(
    matrix: *const HdbscanCapiCsr,
    n_features: usize,
    metric: u32,
    configuration: *const HdbscanCapiFitConfiguration,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        hdbscan_capi_fit_csr_with_thresholds(
            matrix,
            n_features,
            metric,
            configuration,
            ptr::null(),
            out,
        )
    }
}

/// Fits a CSR feature or distance matrix with cluster selection thresholds.
///
/// # Safety
/// The pointers follow the contract of [`hdbscan_capi_fit_csr_configured`].
/// A non-null `thresholds` must point to readable thresholds for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_csr_with_thresholds(
    matrix: *const HdbscanCapiCsr,
    n_features: usize,
    metric: u32,
    configuration: *const HdbscanCapiFitConfiguration,
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit_result(out, metric == PRECOMPUTED_METRIC, || {
            let (configuration, max_dist) = read_fit_configuration(configuration, thresholds)?;
            let csr = read_csr(matrix)?;
            if metric == PRECOMPUTED_METRIC {
                if csr.indptr.len() - 1 != n_features {
                    return Err(Status::InvalidShape);
                }
                Ok(fit_csr_precomputed_configured(
                    csr.indptr,
                    csr.indices,
                    csr.data,
                    configuration,
                    max_dist,
                )?)
            } else {
                Ok(fit_csr_configured(
                    csr.indptr,
                    csr.indices,
                    csr.data,
                    n_features,
                    feature_metric(metric)?,
                    configuration,
                )?)
            }
        })
    }
}

/// Fits dense features and reports rows containing nonfinite values as noise.
///
/// # Safety
/// The pointers and optional configurations follow the contract of
/// [`hdbscan_capi_fit_dense_with_thresholds`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_dense_with_missing(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    metric: u32,
    configuration: *const HdbscanCapiFitConfiguration,
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit_result(out, false, || {
            let (configuration, _) = read_fit_configuration(configuration, thresholds)?;
            Ok(fit_dense_with_missing(
                input_slice(data, data_len)?,
                n_samples,
                n_features,
                feature_metric(metric)?,
                configuration,
            )?)
        })
    }
}

/// Fits Euclidean or Manhattan features using the exact Prim vector path.
///
/// # Safety
/// The pointers and optional configurations follow the contract of
/// [`hdbscan_capi_fit_dense_with_missing`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_dense_prims(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    metric: u32,
    configuration: *const HdbscanCapiFitConfiguration,
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit_result(out, false, || {
            let (configuration, _) = read_fit_configuration(configuration, thresholds)?;
            Ok(fit_dense_prims(
                input_slice(data, data_len)?,
                n_samples,
                n_features,
                feature_metric(metric)?,
                configuration,
            )?)
        })
    }
}

/// Fits Euclidean or Manhattan features using the requested spatial search.
///
/// # Safety
/// The fit arguments follow the contract of [`hdbscan_capi_fit_dense_with_missing`].
/// `search` must be null or a readable pointer to Borůvka settings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_dense_boruvka(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    metric: u32,
    configuration: *const HdbscanCapiFitConfiguration,
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
    search: *const HdbscanCapiBoruvkaOptions,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit_result(out, false, || {
            let (configuration, _) = read_fit_configuration(configuration, thresholds)?;
            let search = read_boruvka_options(search)?;
            Ok(fit_dense_boruvka(
                input_slice(data, data_len)?,
                n_samples,
                n_features,
                feature_metric(metric)?,
                configuration,
                search,
            )?)
        })
    }
}

unsafe fn read_boruvka_options(
    search: *const HdbscanCapiBoruvkaOptions,
) -> Result<BoruvkaOptions, Status> {
    if search.is_null() {
        return Ok(BoruvkaOptions::default());
    }
    if !search.is_aligned() {
        return Err(Status::InvalidPointer);
    }
    let search = unsafe { search.read() };
    let tree = match search.tree {
        0 => BoruvkaTree::KdTree,
        1 => BoruvkaTree::BallTree,
        _ => return Err(Status::InvalidOptions),
    };
    if search.approximate > 1 {
        return Err(Status::InvalidOptions);
    }
    Ok(BoruvkaOptions {
        tree,
        approximate: search.approximate == 1,
        leaf_size: search.leaf_size,
        core_dist_n_jobs: search.core_dist_n_jobs,
    })
}

unsafe fn read_tree_distance<'a>(
    distance: *const HdbscanCapiTreeDistance,
) -> Result<TreeDistance<'a>, Status> {
    if distance.is_null() {
        return Err(Status::NullArgument);
    }
    if !distance.is_aligned() {
        return Err(Status::InvalidPointer);
    }
    let distance = unsafe { &*distance };
    if distance.parameter_count != 0 && !matches!(distance.kind, 3..=5) {
        return Err(Status::InvalidOptions);
    }
    let parameters = unsafe { input_slice(distance.parameters, distance.parameter_count)? };
    Ok(match distance.kind {
        0 => TreeDistance::Euclidean,
        1 => TreeDistance::Manhattan,
        2 => TreeDistance::Chebyshev,
        3 => TreeDistance::Minkowski {
            p: distance.p,
            weights: (!parameters.is_empty()).then_some(parameters),
        },
        4 => TreeDistance::StandardizedEuclidean(parameters),
        5 => TreeDistance::Mahalanobis(parameters),
        6 => TreeDistance::Hamming,
        7 => TreeDistance::Canberra,
        8 => TreeDistance::BrayCurtis,
        9 => TreeDistance::Jaccard,
        10 => TreeDistance::Dice,
        11 => TreeDistance::RogersTanimoto,
        12 => TreeDistance::RussellRao,
        13 => TreeDistance::SokalMichener,
        14 => TreeDistance::SokalSneath,
        15 => TreeDistance::Haversine,
        _ => return Err(Status::InvalidMetric),
    })
}

/// Fits dense features with a configurable spatial distance.
///
/// # Safety
/// The data, output and settings pointers must follow the existing dense
/// fit contract. The distance and indicated parameters must be readable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_dense_tree(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    distance: *const HdbscanCapiTreeDistance,
    configuration: *const HdbscanCapiFitConfiguration,
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
    search: *const HdbscanCapiBoruvkaOptions,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit_result(out, false, || {
            let (configuration, _) = read_fit_configuration(configuration, thresholds)?;
            let distance = read_tree_distance(distance)?;
            let values = input_slice(data, data_len)?;
            if search.is_null() {
                Ok(fit_dense_tree_prims(
                    values,
                    n_samples,
                    n_features,
                    distance,
                    configuration,
                )?)
            } else {
                Ok(fit_dense_tree_boruvka(
                    values,
                    n_samples,
                    n_features,
                    distance,
                    configuration,
                    read_boruvka_options(search)?,
                )?)
            }
        })
    }
}

/// Fits CSR features and reports rows containing nonfinite stored values as noise.
///
/// # Safety
/// The pointers and optional configurations follow the contract of
/// [`hdbscan_capi_fit_csr_with_thresholds`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_fit_csr_with_missing(
    matrix: *const HdbscanCapiCsr,
    n_features: usize,
    metric: u32,
    configuration: *const HdbscanCapiFitConfiguration,
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
    out: *mut *mut HdbscanCapiResult,
) -> u32 {
    unsafe {
        fit_result(out, false, || {
            let (configuration, _) = read_fit_configuration(configuration, thresholds)?;
            let csr = read_csr(matrix)?;
            Ok(fit_csr_with_missing(
                csr.indptr,
                csr.indices,
                csr.data,
                n_features,
                feature_metric(metric)?,
                configuration,
            )?)
        })
    }
}

unsafe fn read_fit_configuration(
    configuration: *const HdbscanCapiFitConfiguration,
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
) -> Result<(FitConfiguration, f64), Status> {
    let (configuration, max_dist) = unsafe { read_configuration(configuration)? };
    let thresholds = unsafe { read_thresholds(thresholds)? };
    Ok((
        configuration.with_cluster_selection_thresholds(thresholds),
        max_dist,
    ))
}

unsafe fn read_thresholds(
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
) -> Result<ClusterSelectionThresholds, Status> {
    if thresholds.is_null() {
        return Ok(ClusterSelectionThresholds::default());
    }
    if !thresholds.is_aligned() {
        return Err(Status::InvalidPointer);
    }
    let thresholds = unsafe { thresholds.read() };
    Ok(ClusterSelectionThresholds {
        epsilon: thresholds.epsilon,
        epsilon_max: thresholds.epsilon_max,
        persistence: thresholds.persistence,
        max_cluster_size: thresholds.max_cluster_size,
    })
}

unsafe fn read_configuration(
    configuration: *const HdbscanCapiFitConfiguration,
) -> Result<(FitConfiguration, f64), Status> {
    if configuration.is_null() {
        return Ok((FitConfiguration::default(), 0.0));
    }
    if !configuration.is_aligned() {
        return Err(Status::InvalidPointer);
    }
    let configuration = unsafe { configuration.read() };
    let method = match configuration.cluster_selection_method {
        0 => ClusterSelectionMethod::Eom,
        1 => ClusterSelectionMethod::Leaf,
        _ => return Err(Status::InvalidOptions),
    };
    if configuration.allow_single_cluster > 1 {
        return Err(Status::InvalidOptions);
    }
    let options =
        FitOptions::from(configuration.options).with_cluster_selection(ClusterSelection {
            method,
            allow_single_cluster: configuration.allow_single_cluster == 1,
        });
    Ok((options, configuration.max_dist))
}

unsafe fn fit_csr(
    matrix: *const HdbscanCapiCsr,
    options: *const HdbscanCapiOptions,
    out: *mut *mut HdbscanCapiResult,
    sparse_precomputed: bool,
    run: impl FnOnce(&[usize], &[usize], &[f64], FitOptions) -> Result<FitResult, Status>
    + std::panic::UnwindSafe,
) -> u32 {
    unsafe {
        fit(options, out, sparse_precomputed, |options| {
            let csr = read_csr(matrix)?;
            run(csr.indptr, csr.indices, csr.data, options)
        })
    }
}

struct CsrInput<'a> {
    indptr: &'a [usize],
    indices: &'a [usize],
    data: &'a [f64],
}

unsafe fn read_csr<'a>(matrix: *const HdbscanCapiCsr) -> Result<CsrInput<'a>, Status> {
    if matrix.is_null() {
        return Err(Status::NullArgument);
    }
    if !matrix.is_aligned() {
        return Err(Status::InvalidPointer);
    }
    let matrix = unsafe { matrix.read() };
    let offset_count = matrix
        .n_samples
        .checked_add(1)
        .ok_or(Status::InvalidLength)?;
    Ok(unsafe {
        CsrInput {
            indptr: input_slice(matrix.indptr, offset_count)?,
            indices: input_slice(matrix.indices, matrix.nnz)?,
            data: input_slice(matrix.data, matrix.nnz)?,
        }
    })
}

unsafe fn input_slice<'a, T>(data: *const T, len: usize) -> Result<&'a [T], Status> {
    if len != 0 && data.is_null() {
        return Err(Status::NullArgument);
    }
    if !data.is_aligned() {
        return Err(Status::InvalidPointer);
    }
    if len > isize::MAX as usize / size_of::<T>() {
        return Err(Status::InvalidLength);
    }
    if len == 0 {
        Ok(&[])
    } else {
        Ok(unsafe { slice::from_raw_parts(data, len) })
    }
}

unsafe fn fit(
    options: *const HdbscanCapiOptions,
    out: *mut *mut HdbscanCapiResult,
    sparse_precomputed: bool,
    fit: impl FnOnce(FitOptions) -> Result<FitResult, Status> + std::panic::UnwindSafe,
) -> u32 {
    unsafe {
        fit_result(out, sparse_precomputed, || {
            if !options.is_aligned() {
                return Err(Status::InvalidPointer);
            }
            let options = if options.is_null() {
                FitOptions::default()
            } else {
                options.read().into()
            };
            fit(options)
        })
    }
}

unsafe fn fit_result(
    out: *mut *mut HdbscanCapiResult,
    sparse_precomputed: bool,
    fit: impl FnOnce() -> Result<FitResult, Status> + std::panic::UnwindSafe,
) -> u32 {
    if out.is_null() {
        return Status::NullArgument as u32;
    }
    if !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    unsafe { out.write(ptr::null_mut()) };
    let result = catch_unwind(|| {
        fit().map(|fit| {
            Box::new(HdbscanCapiResult {
                labels: fit.labels,
                probabilities: fit.probabilities,
                cluster_persistence: fit.cluster_persistence,
                condensed_tree: fit.condensed_tree,
                single_linkage_tree: fit.single_linkage_tree,
                minimum_spanning_tree: fit.minimum_spanning_tree,
                relative_validity_tree: if sparse_precomputed {
                    RelativeValidityTree::CsrPrecomputed
                } else {
                    RelativeValidityTree::EdgeList
                },
            })
        })
    });
    match result {
        Ok(Ok(result)) => {
            unsafe { out.write(Box::into_raw(result)) };
            Status::Ok as u32
        }
        Ok(Err(status)) => status as u32,
        Err(_) => Status::Panic as u32,
    }
}

/// Returns read-only pointers into a result. They are valid until it is freed.
///
/// # Safety
/// `result` must be null or a live handle returned by the fit function. `out`
/// must be null or valid for writing a view and must not alias the result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_result_view(
    result: *const HdbscanCapiResult,
    out: *mut HdbscanCapiResultView,
) -> u32 {
    if result.is_null() || out.is_null() {
        return Status::NullArgument as u32;
    }
    if !result.is_aligned() || !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    let result = unsafe { &*result };
    let n_clusters = result.cluster_persistence.len();
    let view = HdbscanCapiResultView {
        labels: result.labels.as_ptr(),
        probabilities: result.probabilities.as_ptr(),
        n_samples: result.labels.len(),
        cluster_persistence: if n_clusters == 0 {
            ptr::null()
        } else {
            result.cluster_persistence.as_ptr()
        },
        n_clusters,
    };
    unsafe { out.write(view) };
    Status::Ok as u32
}

/// Computes one GLOSH score per original sample.
///
/// # Safety
/// `result` must be null or a live fit handle. `out` must be null or writable
/// for `out_len` elements and must not alias any borrowed part of `result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_result_outlier_scores(
    result: *const HdbscanCapiResult,
    out: *mut f64,
    out_len: usize,
) -> u32 {
    unsafe {
        copy_result_array(result, out, out_len, |result| {
            outlier_scores(&result.condensed_tree)
        })
    }
}

/// Computes relative validity using the fitted minimum spanning tree.
///
/// # Safety
/// `result` must be null or a live fit handle. `out` must be null or writable
/// for one double and must not alias any part of the result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_result_relative_validity(
    result: *const HdbscanCapiResult,
    out: *mut f64,
) -> u32 {
    if result.is_null() || out.is_null() {
        return Status::NullArgument as u32;
    }
    if !result.is_aligned() || !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    let result = unsafe { &*result };
    match catch_unwind(|| {
        relative_validity(
            &result.labels,
            &result.minimum_spanning_tree,
            result.relative_validity_tree,
        )
    }) {
        Ok(Ok(value)) => {
            unsafe { out.write(value) };
            Status::Ok as u32
        }
        Ok(Err(_)) => Status::InvalidShape as u32,
        Err(_) => Status::Panic as u32,
    }
}

/// Computes DBSCAN* labels at a mutual-reachability cut without refitting.
///
/// # Safety
/// The pointers follow the same contract as [`hdbscan_capi_result_outlier_scores`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_result_dbscan_clustering(
    result: *const HdbscanCapiResult,
    cut_distance: f64,
    min_cluster_size: i64,
    out: *mut i64,
    out_len: usize,
) -> u32 {
    unsafe {
        copy_result_array(result, out, out_len, |result| {
            dbscan_clustering(&result.single_linkage_tree, cut_distance, min_cluster_size)
        })
    }
}

/// Builds a prediction cache from a dense training matrix and its fitted result.
///
/// # Safety
/// The result must be null or a live fit handle. The data pointer must be
/// readable for data_len doubles. The output must be null or writable for
/// one handle and must not alias any input. The cache owns its training data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_prediction_create(
    result: *const HdbscanCapiResult,
    data: *const f64,
    data_len: usize,
    n_features: usize,
    metric: u32,
    min_samples: usize,
    out: *mut *mut HdbscanCapiPredictionModel,
) -> u32 {
    if out.is_null() {
        return Status::NullArgument as u32;
    }
    if !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    unsafe { out.write(ptr::null_mut()) };
    if result.is_null() {
        return Status::NullArgument as u32;
    }
    if !result.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    match catch_unwind(|| -> Result<_, Status> {
        let result = unsafe { &*result };
        let data = unsafe { input_slice(data, data_len)? };
        let model = PredictionModel::new(
            data.to_vec(),
            n_features,
            feature_metric(metric)?,
            min_samples,
            &result.labels,
            &result.condensed_tree,
        )
        .map_err(Status::from)?;
        Ok(Box::new(HdbscanCapiPredictionModel {
            model,
            n_features,
            min_samples,
        }))
    }) {
        Ok(Ok(model)) => {
            unsafe { out.write(Box::into_raw(model)) };
            Status::Ok as u32
        }
        Ok(Err(status)) => status as u32,
        Err(_) => Status::Panic as u32,
    }
}

/// Writes the predicted label, membership strength, and connecting index for each row.
///
/// # Safety
/// The model must be null or a live prediction handle. The data pointer must
/// be readable for data_len doubles. The output must be writable for out_len
/// rows and must not alias other arguments. Either buffer may be null when empty.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_approximate_predict(
    model: *const HdbscanCapiPredictionModel,
    data: *const f64,
    data_len: usize,
    out: *mut Prediction,
    out_len: usize,
) -> u32 {
    unsafe {
        write_predictions(model, data, data_len, out, out_len, |cache, values| {
            cache.model.predict(values, cache.min_samples, None)
        })
    }
}

/// Writes the approximate outlier score for each row.
///
/// # Safety
/// The arguments follow the same contract as hdbscan_capi_approximate_predict.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_approximate_predict_scores(
    model: *const HdbscanCapiPredictionModel,
    data: *const f64,
    data_len: usize,
    out: *mut f64,
    out_len: usize,
) -> u32 {
    unsafe {
        write_predictions(model, data, data_len, out, out_len, |cache, values| {
            cache
                .model
                .predict_densities(values, cache.min_samples, None)
                .map(|rows| rows.into_iter().map(|row| row.score()).collect())
        })
    }
}

/// Returns the shape of the training membership matrix.
///
/// # Safety
/// The model must be null or a live prediction handle. The output must be
/// null or writable for one shape and must not alias the model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_membership_shape(
    model: *const HdbscanCapiPredictionModel,
    out: *mut HdbscanCapiMembershipShape,
) -> u32 {
    if model.is_null() || out.is_null() {
        return Status::NullArgument as u32;
    }
    if !model.is_aligned() || !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    let cache = unsafe { &*model };
    unsafe {
        out.write(HdbscanCapiMembershipShape {
            n_samples: cache.model.core_distances().len(),
            n_clusters: cache.model.selected_clusters().len(),
        });
    }
    Status::Ok as u32
}

/// Writes a row-major matrix of cluster memberships for new points.
///
/// # Safety
/// The model must be null or a live prediction handle. The input and output
/// buffers must be valid for their given lengths and must not alias each other
/// or the model. A buffer can be null when it has zero elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_membership_vector(
    model: *const HdbscanCapiPredictionModel,
    data: *const f64,
    data_len: usize,
    out: *mut f64,
    out_len: usize,
) -> u32 {
    unsafe { write_memberships(model, Some((data, data_len)), out, out_len) }
}

/// Writes the row-major membership matrix of the training points.
///
/// # Safety
/// The model and output pointers follow the contract of hdbscan_capi_membership_vector.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_all_points_membership_vectors(
    model: *const HdbscanCapiPredictionModel,
    out: *mut f64,
    out_len: usize,
) -> u32 {
    unsafe { write_memberships(model, None, out, out_len) }
}

unsafe fn write_memberships(
    model: *const HdbscanCapiPredictionModel,
    query: Option<(*const f64, usize)>,
    out: *mut f64,
    out_len: usize,
) -> u32 {
    if model.is_null() || (out_len != 0 && out.is_null()) {
        return Status::NullArgument as u32;
    }
    if !model.is_aligned() || !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    let cache = unsafe { &*model };
    let n_samples = match query {
        Some((_, len)) if len.is_multiple_of(cache.n_features) => len / cache.n_features,
        Some(_) => return Status::InvalidLength as u32,
        None => cache.model.core_distances().len(),
    };
    if n_samples.checked_mul(cache.model.selected_clusters().len()) != Some(out_len)
        || out_len > isize::MAX as usize / size_of::<f64>()
    {
        return Status::InvalidLength as u32;
    }
    let query = match query
        .map(|(data, len)| unsafe { input_slice(data, len) })
        .transpose()
    {
        Ok(query) => query,
        Err(status) => return status as u32,
    };
    if n_samples == 0 {
        return Status::Ok as u32;
    }
    match catch_unwind(|| match query {
        Some(values) => cache.model.membership_vector(values, cache.min_samples),
        None => cache.model.all_points_membership_vectors(),
    }) {
        Ok(Ok(matrix)) if matrix.values.len() == out_len => {
            if out_len != 0 {
                unsafe { ptr::copy_nonoverlapping(matrix.values.as_ptr(), out, out_len) };
            }
            Status::Ok as u32
        }
        Ok(Err(MembershipError::Fit(error))) => Status::from(error) as u32,
        Ok(_) => Status::InvalidShape as u32,
        Err(_) => Status::Panic as u32,
    }
}

/// Evaluates externally supplied cluster labels with DBCV.
///
/// # Safety
/// Nonempty inputs must be valid for their full lengths. The overall output
/// must point to a writable double. A non-null cluster output must be writable
/// for the given length. Output buffers must be disjoint from each other and
/// from both inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_validity_index(
    data: *const f64,
    data_len: usize,
    labels: *const i64,
    n_samples: usize,
    n_features: usize,
    metric: u32,
    dimension: f64,
    mst_raw_dist: u32,
    overall: *mut f64,
    per_cluster: *mut f64,
    per_cluster_len: usize,
) -> u32 {
    if overall.is_null() || (per_cluster.is_null() && per_cluster_len != 0) {
        return Status::NullArgument as u32;
    }
    if !overall.is_aligned() || !per_cluster.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    if per_cluster_len > isize::MAX as usize / size_of::<f64>() {
        return Status::InvalidLength as u32;
    }
    match catch_unwind(|| {
        let values = unsafe { input_slice(data, data_len)? };
        let labels = unsafe { input_slice(labels, n_samples)? };
        if mst_raw_dist > 1 {
            return Err(Status::InvalidOptions);
        }
        let raw = mst_raw_dist != 0;
        let scores = if metric == PRECOMPUTED_METRIC {
            if n_features != n_samples {
                return Err(Status::InvalidShape);
            }
            validity_index_precomputed(values, labels, dimension, raw)
        } else {
            validity_index_dense(values, n_features, labels, feature_metric(metric)?, raw)
        }
        .map_err(|error| match error {
            ValidityError::Fit(error) => Status::from(error),
            ValidityError::ZeroDimension => Status::InvalidOptions,
            _ => Status::InvalidShape,
        })?;
        if !per_cluster.is_null() && per_cluster_len != scores.per_cluster.len() {
            return Err(Status::InvalidLength);
        }
        unsafe {
            if per_cluster_len != 0 {
                ptr::copy_nonoverlapping(scores.per_cluster.as_ptr(), per_cluster, per_cluster_len);
            }
            overall.write(scores.overall);
        }
        Ok(())
    }) {
        Ok(Ok(())) => Status::Ok as u32,
        Ok(Err(status)) => status as u32,
        Err(_) => Status::Panic as u32,
    }
}

/// Fits robust single linkage and writes its labels and hierarchy.
///
/// # Safety
/// Non-null buffers must be valid for their indicated lengths. `options` must
/// be null or readable, and outputs must be disjoint from each other and inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_robust_single_linkage(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    metric: u32,
    options: *const HdbscanCapiRobustOptions,
    labels: *mut i64,
    labels_len: usize,
    hierarchy: *mut LinkageMerge,
    hierarchy_len: usize,
) -> u32 {
    unsafe {
        write_robust_linkage(
            n_samples,
            (labels, labels_len),
            (hierarchy, hierarchy_len),
            || {
                let default = (metric < 2).then_some(BoruvkaTree::KdTree);
                let (config, algorithm) = read_robust_options(options, default)?;
                let values = input_slice(data, data_len)?;
                if metric == PRECOMPUTED_METRIC {
                    if n_samples != n_features {
                        return Err(Status::InvalidShape);
                    }
                    if !matches!(
                        algorithm,
                        RobustLinkageAlgorithm::Best | RobustLinkageAlgorithm::Generic
                    ) {
                        return Err(Status::InvalidMetric);
                    }
                    Ok(robust_single_linkage_precomputed(
                        values, n_samples, config,
                    )?)
                } else {
                    Ok(robust_single_linkage_dense(
                        values,
                        n_samples,
                        n_features,
                        feature_metric(metric)?,
                        algorithm,
                        config,
                    )?)
                }
            },
        )
    }
}

/// Fits robust single linkage with a configurable spatial distance.
///
/// # Safety
/// All buffers must be valid for their indicated lengths and disjoint from
/// the mutable output buffers. The distance options must be readable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_robust_single_linkage_tree(
    data: *const f64,
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    distance: *const HdbscanCapiTreeDistance,
    options: *const HdbscanCapiRobustOptions,
    labels: *mut i64,
    labels_len: usize,
    hierarchy: *mut LinkageMerge,
    hierarchy_len: usize,
) -> u32 {
    unsafe {
        write_robust_linkage(
            n_samples,
            (labels, labels_len),
            (hierarchy, hierarchy_len),
            || {
                let distance = read_tree_distance(distance)?;
                let default = if distance.supports_kd() {
                    BoruvkaTree::KdTree
                } else {
                    BoruvkaTree::BallTree
                };
                let (config, algorithm) = read_robust_options(options, Some(default))?;
                Ok(robust_single_linkage_spatial(
                    input_slice(data, data_len)?,
                    n_samples,
                    n_features,
                    distance,
                    algorithm,
                    config,
                )?)
            },
        )
    }
}

unsafe fn read_robust_options(
    options: *const HdbscanCapiRobustOptions,
    default_tree: Option<BoruvkaTree>,
) -> Result<(RobustLinkageOptions, RobustLinkageAlgorithm), Status> {
    if options.is_null() {
        return Ok((
            RobustLinkageOptions::default(),
            RobustLinkageAlgorithm::Best,
        ));
    }
    if !options.is_aligned() {
        return Err(Status::InvalidPointer);
    }
    let options = unsafe { options.read() };
    let config = RobustLinkageOptions {
        cut: options.cut,
        k: options.k,
        alpha: options.alpha,
        gamma: options.gamma,
    };
    let boruvka = |tree| {
        RobustLinkageAlgorithm::Boruvka(BoruvkaOptions {
            tree,
            leaf_size: options.leaf_size,
            approximate: false,
            core_dist_n_jobs: options.core_dist_n_jobs,
        })
    };
    let algorithm = match options.algorithm {
        0 => default_tree.map_or(RobustLinkageAlgorithm::Best, boruvka),
        1 => RobustLinkageAlgorithm::Generic,
        2 => RobustLinkageAlgorithm::Prim,
        3 => boruvka(BoruvkaTree::KdTree),
        4 => boruvka(BoruvkaTree::BallTree),
        _ => return Err(Status::InvalidOptions),
    };
    Ok((config, algorithm))
}

unsafe fn write_robust_linkage(
    n_samples: usize,
    (labels, labels_len): (*mut i64, usize),
    (hierarchy, hierarchy_len): (*mut LinkageMerge, usize),
    fit: impl FnOnce() -> Result<RobustLinkageResult, Status> + std::panic::UnwindSafe,
) -> u32 {
    if labels.is_null() || hierarchy.is_null() {
        return Status::NullArgument as u32;
    }
    if !labels.is_aligned() || !hierarchy.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    if labels_len != n_samples || n_samples.checked_sub(1) != Some(hierarchy_len) {
        return Status::InvalidLength as u32;
    }
    match catch_unwind(fit) {
        Ok(Ok(result)) => {
            unsafe {
                ptr::copy_nonoverlapping(result.labels.as_ptr(), labels, labels_len);
                ptr::copy_nonoverlapping(result.hierarchy.as_ptr(), hierarchy, hierarchy_len);
            }
            Status::Ok as u32
        }
        Ok(Err(status)) => status as u32,
        Err(_) => Status::Panic as u32,
    }
}

/// Detects branches from a fitted result and its original dense training data.
///
/// # Safety
/// The fit must be null or a live handle. Non-null data and options must be
/// readable, and `out` must be writable and disjoint from the inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_detect_branches(
    result: *const HdbscanCapiResult,
    data: *const f64,
    data_len: usize,
    n_features: usize,
    metric: u32,
    options: *const HdbscanCapiBranchOptions,
    thresholds: *const HdbscanCapiClusterSelectionThresholds,
    out: *mut *mut BranchResult,
) -> u32 {
    if out.is_null() {
        return Status::NullArgument as u32;
    }
    if !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    unsafe { out.write(ptr::null_mut()) };
    if result.is_null() || options.is_null() {
        return Status::NullArgument as u32;
    }
    if !result.is_aligned() || !options.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    match catch_unwind(|| {
        let result = unsafe { &*result };
        let options = unsafe { options.read() };
        let values = unsafe { input_slice(data, data_len)? };
        let method = match options.method {
            0 => BranchDetectionMethod::Full,
            1 => BranchDetectionMethod::Core,
            _ => return Err(Status::InvalidOptions),
        };
        let selection = match (
            options.cluster_selection_method,
            options.allow_single_cluster,
        ) {
            (0, 0 | 1) => ClusterSelectionMethod::Eom,
            (1, 0 | 1) => ClusterSelectionMethod::Leaf,
            _ => return Err(Status::InvalidOptions),
        };
        if options.label_sides_as_branches > 1 {
            return Err(Status::InvalidOptions);
        }
        let cache = BranchDetectionData::new(
            values,
            n_features,
            &result.labels,
            feature_metric(metric)?,
            options.min_samples,
            40,
        )?;
        let scored = cache.detect(
            &result.labels,
            &result.probabilities,
            &result.minimum_spanning_tree,
            BranchOptions {
                method,
                cluster_labels_overridden: false,
                label_sides_as_branches: options.label_sides_as_branches == 1,
                min_cluster_size: options.min_cluster_size,
                selection: ClusterSelection {
                    method: selection,
                    allow_single_cluster: options.allow_single_cluster == 1,
                },
                thresholds: unsafe { read_thresholds(thresholds)? },
            },
        )?;
        Ok(Box::new(scored))
    }) {
        Ok(Ok(result)) => {
            unsafe { out.write(Box::into_raw(result)) };
            Status::Ok as u32
        }
        Ok(Err(status)) => status as u32,
        Err(_) => Status::Panic as u32,
    }
}

/// Borrows the fixed-length arrays from a live branch result.
///
/// # Safety
/// The input must be null or a live handle. The output must be null or
/// writable and disjoint from the handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_branch_result_view(
    result: *const BranchResult,
    out: *mut HdbscanCapiBranchResultView,
) -> u32 {
    if result.is_null() || out.is_null() {
        return Status::NullArgument as u32;
    }
    if !result.is_aligned() || !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    let r = unsafe { &*result };
    unsafe {
        out.write(HdbscanCapiBranchResultView {
            n_samples: r.labels.len(),
            n_clusters: r.clusters.len(),
            labels: r.labels.as_ptr(),
            probabilities: r.probabilities.as_ptr(),
            cluster_labels: r.cluster_labels.as_ptr(),
            cluster_probabilities: r.cluster_probabilities.as_ptr(),
            branch_labels: r.branch_labels.as_ptr(),
            branch_probabilities: r.branch_probabilities.as_ptr(),
            centralities: r.centralities.as_ptr(),
        })
    };
    Status::Ok as u32
}

/// Borrows one cluster's graph, hierarchy and persistence arrays.
///
/// # Safety
/// The pointers follow the contract of `hdbscan_capi_branch_result_view`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_branch_cluster_view(
    result: *const BranchResult,
    cluster: usize,
    out: *mut HdbscanCapiBranchClusterView,
) -> u32 {
    if result.is_null() || out.is_null() {
        return Status::NullArgument as u32;
    }
    if !result.is_aligned() || !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    let Some(r) = (unsafe { &*result }).clusters.get(cluster) else {
        return Status::InvalidOptions as u32;
    };
    let linkage = r.linkage.as_deref().unwrap_or_default();
    let condensed = r.condensed.as_deref().unwrap_or_default();
    unsafe {
        out.write(HdbscanCapiBranchClusterView {
            points: r.points.as_ptr(),
            n_points: r.points.len(),
            graph: r.graph.as_ptr(),
            n_edges: r.graph.len(),
            linkage: linkage.as_ptr(),
            n_merges: linkage.len(),
            condensed: condensed.as_ptr(),
            n_condensed: condensed.len(),
            persistences: r.persistences.as_ptr(),
            n_persistences: r.persistences.len(),
        })
    };
    Status::Ok as u32
}

/// Predicts a cluster and a branch for each new point.
///
/// # Safety
/// The branch result and prediction cache must be null or live handles from
/// the same training fit. Buffers must be valid and nonoverlapping.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_approximate_predict_branch(
    branches: *const BranchResult,
    prediction: *const HdbscanCapiPredictionModel,
    data: *const f64,
    data_len: usize,
    out: *mut BranchPrediction,
    out_len: usize,
) -> u32 {
    if branches.is_null() {
        return Status::NullArgument as u32;
    }
    if !branches.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    let branches = unsafe { &*branches };
    unsafe {
        write_predictions(prediction, data, data_len, out, out_len, |cache, values| {
            if cache.model.core_distances().len() != branches.labels.len() {
                return Err(FitError::LengthMismatch);
            }
            cache
                .model
                .predict(values, cache.min_samples, None)?
                .into_iter()
                .map(|row| branches.classify(row))
                .collect()
        })
    }
}

/// Frees an owning branch result.
///
/// # Safety
/// Non-null handles must be live and have no concurrent readers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_branch_result_free(result: *mut BranchResult) {
    if !result.is_null() {
        drop(unsafe { Box::from_raw(result) });
    }
}

unsafe fn write_predictions<T>(
    cache: *const HdbscanCapiPredictionModel,
    data: *const f64,
    data_len: usize,
    out: *mut T,
    out_len: usize,
    predict: impl FnOnce(&HdbscanCapiPredictionModel, &[f64]) -> Result<Vec<T>, FitError>
    + std::panic::UnwindSafe,
) -> u32 {
    if cache.is_null() || out_len != 0 && out.is_null() {
        return Status::NullArgument as u32;
    }
    if !cache.is_aligned() || !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    let cache = unsafe { &*cache };
    if Some(data_len) != out_len.checked_mul(cache.n_features)
        || out_len > isize::MAX as usize / size_of::<T>()
    {
        return Status::InvalidLength as u32;
    }
    let values = match unsafe { input_slice(data, data_len) } {
        Ok(values) => values,
        Err(status) => return status as u32,
    };
    if out_len == 0 {
        return Status::Ok as u32;
    }
    match catch_unwind(|| predict(cache, values)) {
        Ok(Ok(rows)) if rows.len() == out_len => {
            for (index, row) in rows.into_iter().enumerate() {
                unsafe { out.add(index).write(row) };
            }
            Status::Ok as u32
        }
        Ok(Ok(_)) => Status::InvalidShape as u32,
        Ok(Err(error)) => Status::from(error) as u32,
        Err(_) => Status::Panic as u32,
    }
}

/// Frees the prediction cache. A null pointer has no effect.
///
/// # Safety
/// A non-null handle must be live and have no concurrent readers. Free it once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_prediction_free(cache: *mut HdbscanCapiPredictionModel) {
    if !cache.is_null() {
        drop(unsafe { Box::from_raw(cache) });
    }
}

unsafe fn copy_result_array<T: Copy>(
    result: *const HdbscanCapiResult,
    out: *mut T,
    out_len: usize,
    compute: impl FnOnce(&HdbscanCapiResult) -> Result<Vec<T>, FitError> + std::panic::UnwindSafe,
) -> u32 {
    if result.is_null() || out.is_null() {
        return Status::NullArgument as u32;
    }
    if !result.is_aligned() || !out.is_aligned() {
        return Status::InvalidPointer as u32;
    }
    let result = unsafe { &*result };
    if out_len != result.labels.len() {
        return Status::InvalidLength as u32;
    }
    match catch_unwind(|| compute(result)) {
        Ok(Ok(values)) if values.len() == out_len => {
            unsafe { ptr::copy_nonoverlapping(values.as_ptr(), out, out_len) };
            Status::Ok as u32
        }
        Ok(Ok(_)) => Status::InvalidShape as u32,
        Ok(Err(error)) => Status::from(error) as u32,
        Err(_) => Status::Panic as u32,
    }
}

/// Releases an owning result. A null pointer has no effect.
///
/// # Safety
/// A non-null `result` must be a live handle from the fit function, with no
/// concurrent readers. It may be freed only once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hdbscan_capi_result_free(result: *mut HdbscanCapiResult) {
    if !result.is_null() {
        drop(unsafe { Box::from_raw(result) });
    }
}
