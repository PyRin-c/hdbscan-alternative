use std::fmt;

use crate::{SpanningEdge, ordering::pairwise_sum};

use super::{NEGATIVE_NAN, maximum};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelativeValidityError {
    MissingTree,
    InvalidLabel,
    ClusterCountOverflow,
    MissingEndpoint(usize),
    SparseTreeShape,
}

impl fmt::Display for RelativeValidityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTree => f.write_str("minimum spanning tree has no edges"),
            Self::InvalidLabel => f.write_str("'list' argument must have no negative elements"),
            Self::ClusterCountOverflow => f.write_str("cluster score array is too large"),
            Self::MissingEndpoint(point) => write!(f, "tree endpoint {point} has no label"),
            Self::SparseTreeShape => f.write_str("array is not broadcastable to correct shape"),
        }
    }
}

impl std::error::Error for RelativeValidityError {}

/// How the fitted minimum spanning tree was produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelativeValidityTree {
    /// A dense feature, sparse feature, or dense precomputed fit.
    EdgeList,
    /// A CSR precomputed fit. Preserves the reference error for labeled edges.
    CsrPrecomputed,
}

/// Approximates DBCV using the fitted mutual-reachability tree.
///
/// Edge endpoints use the same indices that were returned by fitting, and
/// labels include all original points, including noise.
pub fn relative_validity(
    labels: &[i64],
    edges: &[SpanningEdge],
    tree: RelativeValidityTree,
) -> Result<f64, RelativeValidityError> {
    Ok(relative_validity_components(labels, edges, tree)?.score(labels.len()))
}

#[doc(hidden)]
pub struct RelativeValidityComponents {
    pub counts: Vec<usize>,
    pub sparseness: Vec<f64>,
    pub separation: Vec<f64>,
    pub fallback: f64,
}

impl RelativeValidityComponents {
    pub fn score(&self, n_samples: usize) -> f64 {
        let scores: Vec<_> = self
            .counts
            .iter()
            .zip(&self.sparseness)
            .zip(&self.separation)
            .map(|((&count, &sparseness), &separation)| {
                let separation = if separation == f64::INFINITY {
                    2.0 * self.fallback
                } else {
                    separation
                };
                let validity = (separation - sparseness) / maximum(separation, sparseness);
                let value = count as f64 * validity / n_samples as f64;
                if value.is_nan() && !sparseness.is_nan() && !separation.is_nan() {
                    NEGATIVE_NAN
                } else {
                    value
                }
            })
            .collect();
        // NumPy starts the final reduction at positive zero.
        0.0 + pairwise_sum(&scores)
    }
}

#[doc(hidden)]
pub fn relative_validity_components(
    labels: &[i64],
    edges: &[SpanningEdge],
    tree: RelativeValidityTree,
) -> Result<RelativeValidityComponents, RelativeValidityError> {
    if labels.iter().any(|&label| label < -1 || label == i64::MAX) {
        return Err(RelativeValidityError::InvalidLabel);
    }
    if let Some(point) = edges
        .iter()
        .map(|edge| edge.source)
        .chain(edges.iter().map(|edge| edge.target))
        .find(|&point| point >= labels.len())
    {
        return Err(RelativeValidityError::MissingEndpoint(point));
    }
    let count = usize::try_from(i128::from(labels.iter().max().copied().unwrap_or(-1)) + 1)
        .map_err(|_| RelativeValidityError::ClusterCountOverflow)?;
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(count)
        .map_err(|_| RelativeValidityError::ClusterCountOverflow)?;
    counts.resize(count, 0_usize);
    for &label in labels.iter().filter(|&&label| label >= 0) {
        counts[label as usize] += 1;
    }
    if edges.is_empty() {
        return Err(RelativeValidityError::MissingTree);
    }

    let mut sparseness = vec![0.0; count];
    let mut separation = vec![f64::INFINITY; count];
    let mut greatest = f64::NEG_INFINITY;
    let mut noise_separation = f64::INFINITY;
    for edge in edges {
        let a = labels[edge.source];
        let b = labels[edge.target];
        greatest = maximum(greatest, edge.distance);
        match (a, b) {
            (-1, -1) => {}
            (-1, _) | (_, -1) => noise_separation = minimum(noise_separation, edge.distance),
            // The sparse precomputed reference path supplies matrix-shaped
            // edge weights to a vector update for edges between labeled points.
            _ if tree == RelativeValidityTree::CsrPrecomputed => {
                return Err(RelativeValidityError::SparseTreeShape);
            }
            (a, b) if a == b => {
                let a = a as usize;
                sparseness[a] = maximum(sparseness[a], edge.distance);
            }
            (a, b) => {
                let (a, b) = (a as usize, b as usize);
                separation[a] = minimum(separation[a], edge.distance);
                separation[b] = minimum(separation[b], edge.distance);
            }
        }
    }
    let fallback = if count == 1 && noise_separation != f64::INFINITY {
        noise_separation
    } else {
        greatest
    };
    Ok(RelativeValidityComponents {
        counts,
        sparseness,
        separation,
        fallback,
    })
}

fn minimum(left: f64, right: f64) -> f64 {
    if left.is_nan() || left < right {
        left
    } else {
        right
    }
}
