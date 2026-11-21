//! Density-Based Clustering Validation (DBCV).

use std::{collections::BTreeMap, fmt};

use crate::{
    FeatureMetric, FitError, SpanningEdge,
    distance::{DenseDistances, cosine_from_dot, dense_dot, dense_l1, dense_l2, normalize_dense},
    hierarchy::prim_tree,
    ordering::pairwise_sum,
};

mod relative;
#[doc(hidden)]
pub use relative::{RelativeValidityComponents, relative_validity_components};
pub use relative::{RelativeValidityError, RelativeValidityTree, relative_validity};

const NEGATIVE_NAN: f64 = f64::from_bits(0xfff8_0000_0000_0000);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidityError {
    Fit(FitError),
    EmptyLabels,
    NegativeClusterCount,
    ClusterCountOverflow,
    ZeroDimension,
    EmptyTree,
    SinglePointTree,
    MissingTreeNeighbor,
    InvalidTreeMask,
}

impl From<FitError> for ValidityError {
    fn from(error: FitError) -> Self {
        Self::Fit(error)
    }
}

impl fmt::Display for ValidityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Fit(error) => return error.fmt(f),
            Self::EmptyLabels => {
                "zero-size array to reduction operation maximum which has no identity"
            }
            Self::NegativeClusterCount | Self::EmptyTree => "negative dimensions are not allowed",
            Self::ClusterCountOverflow => "cluster score array is too large",
            Self::ZeroDimension => "float division by zero",
            Self::SinglePointTree => "Invalid shape in axis 0: 0.",
            Self::MissingTreeNeighbor => "minimum spanning tree has no matching previous neighbor",
            Self::InvalidTreeMask => {
                "boolean index did not match indexed array along axis 0; size of axis is 2 but size of corresponding boolean axis is 1"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for ValidityError {}

/// Global score and scores indexed by cluster label. Unused labels contain NaN.
#[derive(Debug)]
pub struct ValidityScores {
    pub overall: f64,
    pub per_cluster: Vec<f64>,
}

/// The internal vertices and edges used for density separation and sparseness.
#[derive(Debug)]
pub struct InternalTree {
    pub nodes: Vec<usize>,
    pub edges: Vec<SpanningEdge>,
    pub fallback_node: bool,
}

/// Computes the all-points core distance for each row of a distance matrix.
/// The matrix is updated in place with the inverse-distance powers.
pub fn all_points_core_distance(
    distances: &mut [f64],
    rows: usize,
    columns: usize,
    dimension: f64,
) -> Result<Vec<f64>, ValidityError> {
    if rows.checked_mul(columns) != Some(distances.len()) {
        return Err(FitError::LengthMismatch.into());
    }
    for distance in distances.iter_mut().filter(|value| **value != 0.0) {
        *distance = distance_power(1.0 / *distance, dimension);
    }
    let mut result = if columns == 0 {
        vec![0.0; rows]
    } else {
        distances.chunks_exact(columns).map(pairwise_sum).collect()
    };
    for value in &mut result {
        *value /= rows as f64 - 1.0;
        if value.is_nan() && rows == 1 {
            *value = NEGATIVE_NAN;
        }
    }
    if pairwise_sum(&result) == 0.0 {
        result.fill(0.0);
    } else {
        if dimension == 0.0 {
            return Err(ValidityError::ZeroDimension);
        }
        let exponent = -1.0 / dimension;
        for value in &mut result {
            *value = distance_power(*value, exponent);
        }
    }
    Ok(result)
}

fn distance_power(value: f64, exponent: f64) -> f64 {
    match exponent {
        2.0 => value * value,
        0.5 => value.sqrt(),
        -1.0 => 1.0 / value,
        _ => value.powf(exponent),
    }
}

/// Builds the reference Prim tree and takes the subgraph induced by its non-leaf vertices.
pub fn internal_minimum_spanning_tree(
    distances: &[f64],
    rows: usize,
    columns: usize,
) -> Result<InternalTree, ValidityError> {
    if rows.checked_mul(columns) != Some(distances.len()) {
        return Err(FitError::LengthMismatch.into());
    }
    if rows == 0 {
        return Err(ValidityError::EmptyTree);
    }
    if rows == 1 {
        return Err(ValidityError::SinglePointTree);
    }
    if columns < rows {
        return Err(ValidityError::MissingTreeNeighbor);
    }
    let at = |left: usize, right: usize| distances[left * columns + right];
    let mut edges = prim_tree(
        &vec![f64::NEG_INFINITY; rows],
        1.0,
        |left, right| Ok(at(left, right)),
        f64::MAX,
    )?;
    let first = edges[0];
    let mut included = vec![false; rows];
    included[first.source] = true;
    included[first.target] = true;
    // Earlier rows determine which endpoints are already in the tree. The
    // smallest matching ID wins, including distances within the tolerance.
    for edge in &mut edges[1..] {
        let target = edge.target;
        edge.source = (0..rows)
            .find(|&point| {
                included[point] && point != target && close(at(target, point), edge.distance)
            })
            .ok_or(ValidityError::MissingTreeNeighbor)?;
        included[target] = true;
    }
    if rows == 2 && first.source == first.target {
        return Err(ValidityError::InvalidTreeMask);
    }
    let mut degree = vec![0_usize; rows];
    for edge in &edges {
        degree[edge.source] += 1;
        degree[edge.target] += 1;
    }
    let mut nodes: Vec<_> = (0..rows).filter(|&point| degree[point] > 1).collect();
    let fallback_node = nodes.is_empty();
    if fallback_node {
        nodes.push(0);
    }
    let internal: Vec<_> = edges
        .iter()
        .filter(|edge| degree[edge.source] > 1 && degree[edge.target] > 1)
        .copied()
        .collect();
    if !internal.is_empty() {
        edges = internal;
    }
    Ok(InternalTree {
        nodes,
        edges,
        fallback_node,
    })
}

fn close(left: f64, right: f64) -> bool {
    left == right
        || (left.is_finite()
            && right.is_finite()
            && (left - right).abs() <= 1e-8 + 1e-5 * right.abs())
}

/// Evaluates externally supplied cluster labels on a dense feature matrix.
/// The matrix contains one row per label, with features packed in row-major order.
pub fn validity_index_dense(
    values: &[f64],
    n_features: usize,
    labels: &[i64],
    metric: FeatureMetric,
    mst_raw_dist: bool,
) -> Result<ValidityScores, ValidityError> {
    if n_features == 0 {
        return Err(FitError::InvalidShape.into());
    }
    if labels.len().checked_mul(n_features) != Some(values.len()) {
        return Err(FitError::LengthMismatch.into());
    }
    let row = |sample: usize| &values[sample * n_features..(sample + 1) * n_features];
    validity_index(
        labels,
        n_features as f64,
        mst_raw_dist,
        |points| {
            let packed: Vec<_> = points
                .iter()
                .flat_map(|&sample| row(sample).iter().copied())
                .collect();
            within_distances(&packed, n_features, metric)
        },
        |left, right| match metric {
            FeatureMetric::Euclidean => dense_l2(row(left), row(right)),
            FeatureMetric::Manhattan => dense_l1(row(left), row(right)),
            FeatureMetric::Cosine => cross_cosine(row(left), row(right)),
        },
    )
}

/// Evaluates labels using a square row-major matrix of precomputed distances.
/// The dimension is the number of features in the space that defined the distances.
pub fn validity_index_precomputed(
    distances: &[f64],
    labels: &[i64],
    dimension: f64,
    mst_raw_dist: bool,
) -> Result<ValidityScores, ValidityError> {
    let rows = labels.len();
    if rows.checked_mul(rows) != Some(distances.len()) {
        return Err(FitError::LengthMismatch.into());
    }
    let between = |left: usize, right: usize| distances[left * rows + right];
    validity_index(
        labels,
        dimension,
        mst_raw_dist,
        |points| {
            Ok(points
                .iter()
                .flat_map(|&left| points.iter().map(move |&right| between(left, right)))
                .collect())
        },
        between,
    )
}

fn within_distances(
    values: &[f64],
    n_features: usize,
    metric: FeatureMetric,
) -> Result<Vec<f64>, FitError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FitError::NonFiniteInput);
    }
    let count = values.len() / n_features;
    let mut distances = Vec::with_capacity(count * count);
    match metric {
        FeatureMetric::Euclidean => {
            let matrix = DenseDistances::new(values, n_features)?;
            for left in 0..count {
                for right in 0..count {
                    distances.push(matrix.between(left, right)?);
                }
            }
        }
        FeatureMetric::Manhattan => {
            for left in values.chunks_exact(n_features) {
                distances.extend(
                    values
                        .chunks_exact(n_features)
                        .map(|right| dense_l1(left, right)),
                );
            }
        }
        FeatureMetric::Cosine => {
            let normalized = normalize_dense(values, n_features);
            for (i, left) in normalized.chunks_exact(n_features).enumerate() {
                for (j, right) in normalized.chunks_exact(n_features).enumerate() {
                    distances.push(if i == j {
                        0.0
                    } else {
                        cosine_from_dot(dense_dot(left, right))
                    });
                }
            }
        }
    }
    Ok(distances)
}

fn cross_cosine(left: &[f64], right: &[f64]) -> f64 {
    fn dot(left: &[f64], right: &[f64]) -> f64 {
        let mut zipped = left.iter().zip(right);
        if left.len() < 4 {
            return zipped.map(|(&x, &y)| x * y).sum();
        }
        let mut lanes = [0.0; 2];
        for _ in 0..left.len() / 2 {
            for lane in &mut lanes {
                let (&x, &y) = zipped.next().unwrap();
                *lane += x * y;
            }
        }
        zipped.fold(lanes[0] + lanes[1], |sum, (&x, &y)| sum + x * y)
    }
    let similarity = dot(left, right) / (dot(left, left).sqrt() * dot(right, right).sqrt());
    if similarity.is_nan() {
        NEGATIVE_NAN
    } else {
        1.0 - similarity.clamp(-1.0, 1.0)
    }
}

fn validity_index(
    labels: &[i64],
    dimension: f64,
    mst_raw_dist: bool,
    within: impl Fn(&[usize]) -> Result<Vec<f64>, FitError>,
    between: impl Fn(usize, usize) -> f64,
) -> Result<ValidityScores, ValidityError> {
    let last = *labels.iter().max().ok_or(ValidityError::EmptyLabels)?;
    if last < -1 {
        return Err(ValidityError::NegativeClusterCount);
    }
    let score_count =
        usize::try_from(i128::from(last) + 1).map_err(|_| ValidityError::ClusterCountOverflow)?;
    let mut scores = Vec::new();
    scores
        .try_reserve_exact(score_count)
        .map_err(|_| ValidityError::ClusterCountOverflow)?;
    scores.resize(score_count, f64::NAN);

    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (sample, &label) in labels.iter().enumerate() {
        if label >= 0 {
            groups.entry(label as usize).or_default().push(sample);
        }
    }

    let mut clusters = Vec::with_capacity(groups.len());
    for (label, samples) in groups {
        let count = samples.len();
        let mut distances = within(&samples)?;
        let core = if mst_raw_dist {
            None
        } else {
            let core = all_points_core_distance(&mut distances.clone(), count, count, dimension)?;
            for (left, row) in distances.chunks_exact_mut(count).enumerate() {
                for (right, distance) in row.iter_mut().enumerate() {
                    *distance = reachability(*distance, core[left], core[right]);
                }
            }
            Some(core)
        };
        let tree = internal_minimum_spanning_tree(&distances, count, count)?;
        let sparseness = tree
            .edges
            .iter()
            .map(|edge| edge.distance)
            .fold(f64::NEG_INFINITY, maximum);
        let internal_points: Vec<_> = tree
            .nodes
            .iter()
            .map(|&point| {
                (
                    samples[point],
                    core.as_ref().map_or(f64::NEG_INFINITY, |core| core[point]),
                )
            })
            .collect();
        clusters.push((label, count, internal_points, sparseness));
    }

    let mut separations = vec![Vec::new(); clusters.len()];
    let between = &between;
    for (left, (_, _, left_points, _)) in clusters.iter().enumerate() {
        for (right, (_, _, right_points, _)) in clusters.iter().enumerate().skip(left + 1) {
            let count = left_points.len() * right_points.len();
            let nearest = minimum_distances(
                count,
                left_points
                    .iter()
                    .flat_map(|&(a, a_core)| {
                        right_points.iter().map(move |&(b, b_core)| {
                            let raw = between(a, b);
                            if mst_raw_dist {
                                raw
                            } else {
                                reachability(raw, a_core, b_core)
                            }
                        })
                    })
                    .enumerate(),
            );
            separations[left].push((clusters[right].0, nearest));
            separations[right].push((clusters[left].0, nearest));
        }
    }
    let mut overall = 0.0_f64;
    for ((label, count, _, sparseness), neighbors) in clusters.into_iter().zip(separations) {
        let separation = minimum_distances(score_count, neighbors);
        let score = if separation.is_nan() {
            separation
        } else {
            let score = (separation - sparseness) / maximum(separation, sparseness);
            if score.is_nan() { NEGATIVE_NAN } else { score }
        };
        scores[label] = score;
        // The Linux and Windows NumPy scalar additions retain different
        // operands when both scores are NaN.
        if score.is_nan() {
            if !overall.is_nan() || cfg!(target_os = "linux") {
                overall = score;
            }
        } else {
            overall += (count as f64 / labels.len() as f64) * score;
        }
    }
    Ok(ValidityScores {
        overall,
        per_cluster: scores,
    })
}

fn maximum(left: f64, right: f64) -> f64 {
    if left.is_nan() || left > right {
        left
    } else {
        right
    }
}

fn reachability(raw: f64, left: f64, right: f64) -> f64 {
    // The pinned NumPy reduction canonicalizes a NaN in its initial slot.
    maximum(
        maximum(if raw.is_nan() { f64::NAN } else { raw }, right),
        left,
    )
}

fn minimum_distances(count: usize, distances: impl IntoIterator<Item = (usize, f64)>) -> f64 {
    // NumPy's contiguous reduction canonicalizes a NaN from the first slot
    // or from a four-value vector group. A NaN in the scalar tail retains its sign.
    let tail_start = if count <= 1 {
        0
    } else {
        1 + (count - 1) / 4 * 4
    };
    let mut minimum = f64::INFINITY;
    for (index, distance) in distances {
        if distance.is_nan() {
            return if index < tail_start {
                f64::NAN
            } else {
                distance
            };
        }
        if distance <= minimum {
            minimum = distance;
        }
    }
    minimum
}
