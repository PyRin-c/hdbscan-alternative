//! Robust single linkage and flat cuts of its hierarchy.

use crate::{
    BoruvkaOptions, BoruvkaTree, FeatureMetric, FitError, LinkageMerge, TreeDistance,
    boruvka::boruvka_tree,
    dbscan_clustering,
    distance::{
        DenseDistances, core_distances, cosine_from_dot, dense_dot, dense_l1, normalize_dense,
    },
    hierarchy::{edges_in_linkage_order, minimum_spanning_tree, prim_tree, single_linkage},
    spatial::Points,
    valid_sample_count,
};

#[derive(Clone, Copy, Debug, Default)]
pub enum RobustLinkageAlgorithm {
    #[default]
    Best,
    Generic,
    Prim,
    Boruvka(BoruvkaOptions),
}

#[derive(Clone, Copy, Debug)]
pub struct RobustLinkageOptions {
    pub cut: f64,
    pub k: usize,
    pub alpha: f64,
    pub gamma: i64,
}

impl Default for RobustLinkageOptions {
    fn default() -> Self {
        Self {
            cut: 0.4,
            k: 5,
            alpha: std::f64::consts::SQRT_2,
            gamma: 5,
        }
    }
}

#[derive(Debug)]
pub struct RobustLinkageResult {
    pub labels: Vec<i64>,
    pub hierarchy: Vec<LinkageMerge>,
}

/// Fits robust single linkage to row-major features.
pub fn robust_single_linkage_dense(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    metric: FeatureMetric,
    algorithm: RobustLinkageAlgorithm,
    options: RobustLinkageOptions,
) -> Result<RobustLinkageResult, FitError> {
    validate(values.len(), n_samples, n_features, options)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FitError::NonFiniteInput);
    }
    if !(matches!(algorithm, RobustLinkageAlgorithm::Generic)
        || matches!(algorithm, RobustLinkageAlgorithm::Best) && metric == FeatureMetric::Cosine)
    {
        return robust_single_linkage_spatial(
            values,
            n_samples,
            n_features,
            metric.try_into()?,
            algorithm,
            options,
        );
    }
    let points = Points::new(values, n_features);
    match metric {
        FeatureMetric::Euclidean => {
            let distances = DenseDistances::new(values, n_features)?;
            generic_linkage(n_samples, options, |i, j| distances.between(i, j))
        }
        FeatureMetric::Manhattan => generic_linkage(n_samples, options, |i, j| {
            Ok(dense_l1(points.row(i), points.row(j)))
        }),
        FeatureMetric::Cosine => {
            let normalized = normalize_dense(values, n_features);
            let points = Points::new(&normalized, n_features);
            generic_linkage(n_samples, options, |i, j| {
                Ok(if i == j {
                    0.0
                } else {
                    cosine_from_dot(dense_dot(points.row(i), points.row(j)))
                })
            })
        }
    }
}

/// Fits robust single linkage with a distance supported by a spatial index.
pub fn robust_single_linkage_spatial(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    metric: TreeDistance<'_>,
    algorithm: RobustLinkageAlgorithm,
    options: RobustLinkageOptions,
) -> Result<RobustLinkageResult, FitError> {
    validate(values.len(), n_samples, n_features, options)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FitError::NonFiniteInput);
    }
    let metric = metric.validated(n_features)?;
    let points = Points::new(values, n_features);
    let distance = |i, j| Ok(metric.distance(metric.reduced(points.row(i), points.row(j))));
    let algorithm = match algorithm {
        RobustLinkageAlgorithm::Best => RobustLinkageAlgorithm::Boruvka(BoruvkaOptions {
            tree: if metric.supports_kd() {
                BoruvkaTree::KdTree
            } else {
                BoruvkaTree::BallTree
            },
            approximate: false,
            ..BoruvkaOptions::default()
        }),
        algorithm => algorithm,
    };
    match algorithm {
        RobustLinkageAlgorithm::Generic => generic_linkage(n_samples, options, distance),
        RobustLinkageAlgorithm::Prim => {
            let core = core_distances(n_samples, options.k.min(n_samples - 1) - 1, distance)?;
            let edges = prim_tree(&core, options.alpha, distance, f64::MAX)?;
            finish(single_linkage(edges, n_samples), options)
        }
        RobustLinkageAlgorithm::Boruvka(search) => {
            if search.leaf_size == 0 {
                return Err(FitError::InvalidLeafSize);
            }
            let edges = boruvka_tree(
                points,
                options.k.min(n_samples - 1),
                metric,
                search,
                options.alpha,
                true,
            )?;
            finish(single_linkage(edges, n_samples), options)
        }
        RobustLinkageAlgorithm::Best => unreachable!(),
    }
}

/// Fits the generic robust single-linkage path to a square, row-major distance matrix.
pub fn robust_single_linkage_precomputed(
    distances: &[f64],
    n_samples: usize,
    options: RobustLinkageOptions,
) -> Result<RobustLinkageResult, FitError> {
    validate(distances.len(), n_samples, n_samples, options)?;
    if distances.iter().any(|value| value.is_nan()) {
        return Err(FitError::NaNDistance);
    }
    generic_linkage(n_samples, options, |i, j| Ok(distances[i * n_samples + j]))
}

fn validate(
    len: usize,
    n_samples: usize,
    n_features: usize,
    options: RobustLinkageOptions,
) -> Result<(), FitError> {
    if !valid_sample_count(n_samples) || n_features == 0 {
        return Err(FitError::InvalidShape);
    }
    if n_samples.checked_mul(n_features) != Some(len) {
        return Err(FitError::LengthMismatch);
    }
    if options.k == 0 {
        return Err(FitError::InvalidMinSamples);
    }
    if options.gamma <= 0 {
        return Err(FitError::InvalidMinClusterSize);
    }
    if options.alpha < 1.0 {
        return Err(FitError::InvalidAlpha);
    }
    Ok(())
}

fn generic_linkage(
    n_samples: usize,
    options: RobustLinkageOptions,
    distance: impl Fn(usize, usize) -> Result<f64, FitError>,
) -> Result<RobustLinkageResult, FitError> {
    let core = core_distances(n_samples, options.k.min(n_samples - 1), &distance)?;
    let edges = minimum_spanning_tree(&core, 1.0, distance)?;
    finish(
        single_linkage(edges_in_linkage_order(&edges), n_samples),
        options,
    )
}

fn finish(
    hierarchy: Vec<LinkageMerge>,
    options: RobustLinkageOptions,
) -> Result<RobustLinkageResult, FitError> {
    let labels = dbscan_clustering(&hierarchy, options.cut, options.gamma)?;
    Ok(RobustLinkageResult { labels, hierarchy })
}
