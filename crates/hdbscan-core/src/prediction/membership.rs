use std::fmt;

use super::{Hierarchy, PredictionModel};
use crate::{CondensedEdge, FitError, ordering::pairwise_sum, spatial::Points};

const DENSITY_EPSILON: f64 = 1e-8;
const NEGATIVE_NAN: f64 = f64::from_bits(0xfff8_0000_0000_0000);

#[derive(Debug)]
pub enum MembershipError {
    Fit(FitError),
    MissingDensity(usize),
    NoClusters,
    QueryShape {
        distance_clusters: usize,
        clusters: usize,
    },
    IncompatibleShapes,
}

impl From<FitError> for MembershipError {
    fn from(error: FitError) -> Self {
        Self::Fit(error)
    }
}

impl fmt::Display for MembershipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Fit(error) => error.fmt(f),
            Self::MissingDensity(node) => write!(f, "missing density for cluster {node}"),
            Self::NoClusters => write!(
                f,
                "zero-size array to reduction operation fmax which has no identity"
            ),
            Self::QueryShape {
                distance_clusters,
                clusters,
            } => write!(
                f,
                "operands could not be broadcast together with shapes ({distance_clusters},) ({clusters},) "
            ),
            Self::IncompatibleShapes => {
                write!(f, "incompatible distance and hierarchy membership shapes")
            }
        }
    }
}

impl std::error::Error for MembershipError {}

/// Row-major membership probabilities, one column per selected cluster in ID order.
#[derive(Debug)]
pub struct MembershipMatrix {
    pub rows: usize,
    pub clusters: usize,
    pub values: Vec<f64>,
}

/// Packed feature rows and their source layout for exemplar distances.
#[derive(Clone, Copy)]
pub struct MembershipQuery<'a> {
    pub values: &'a [f64],
    pub min_samples: usize,
    pub column_major: bool,
}

/// Inputs to membership normalization. Language bindings can use these to
/// reproduce floating-point warnings while keeping distance and tree work native.
pub struct MembershipComponents {
    pub distance_rows: usize,
    pub distance_clusters: usize,
    pub clusters: usize,
    pub distances: Vec<f64>,
    pub outliers: Vec<f64>,
    /// Numerator and denominator of the probability of belonging to any cluster.
    pub densities: Vec<(f64, f64)>,
}

impl PredictionModel {
    /// Predicts a probability for every selected cluster for each new feature row.
    pub fn membership_vector(
        &self,
        values: &[f64],
        min_samples: usize,
    ) -> Result<MembershipMatrix, MembershipError> {
        self.membership_vector_for_layout(MembershipQuery {
            values,
            min_samples,
            column_major: false,
        })
    }

    /// Predicts memberships with the original layout of the supplied query rows.
    pub fn membership_vector_for_layout(
        &self,
        query: MembershipQuery<'_>,
    ) -> Result<MembershipMatrix, MembershipError> {
        self.membership_components(Some(query), None)?.finish(false)
    }

    /// Computes memberships of the cached training rows from their fitted hierarchy.
    pub fn all_points_membership_vectors(&self) -> Result<MembershipMatrix, MembershipError> {
        self.membership_components(None, None)?.finish(true)
    }

    /// Computes distance weights and cluster scores for training rows or a query.
    /// A fit override pairs an estimator's current tree and labels with its retained cache.
    pub fn membership_components(
        &self,
        query: Option<MembershipQuery<'_>>,
        current_fit: Option<(&[CondensedEdge], &[i64])>,
    ) -> Result<MembershipComponents, MembershipError> {
        let current = current_fit
            .map(|(tree, labels)| {
                let hierarchy = Hierarchy::new(tree, labels.len())?;
                let selected = hierarchy.selected_clusters(labels)?;
                Ok::<_, FitError>((hierarchy, selected))
            })
            .transpose()?;
        let (current, selected) = current.as_ref().map_or(
            (&self.hierarchy, self.selected.as_slice()),
            |(tree, selected)| (tree, selected.as_slice()),
        );
        let old = Points::new(&self.values, self.dimensions);
        let mut components = MembershipComponents {
            distance_rows: query.map_or(old.len(), |query| query.values.len() / self.dimensions),
            distance_clusters: self.exemplars.len(),
            clusters: selected.len(),
            distances: Vec::new(),
            outliers: Vec::new(),
            densities: Vec::new(),
        };
        if query.is_none() && selected.is_empty() {
            return Ok(components);
        }

        let mut visit = |edge: CondensedEdge, lambda: f64, query: bool| {
            let maximum = self.hierarchy.membership_maximum(edge.parent)?;
            if selected.is_empty() {
                return Err(MembershipError::NoClusters);
            }
            if query
                && self.exemplars.len() != selected.len()
                && self.exemplars.len() != 1
                && selected.len() != 1
            {
                return Err(MembershipError::QueryShape {
                    distance_clusters: self.exemplars.len(),
                    clusters: selected.len(),
                });
            }
            let lambda = lambda.min(edge.lambda);
            let mut nearest = selected[0];
            let mut highest = f64::NEG_INFINITY;
            for &cluster in selected {
                let branch = self.hierarchy.merge_height(edge.parent, cluster)?;
                let height = branch.unwrap_or(lambda);
                if height > highest {
                    highest = height;
                    nearest = cluster;
                }
                let score = if query {
                    // The point density entering the outlier score is single
                    // precision. The branch densities and probability stay double.
                    let height = branch.unwrap_or(f64::from(lambda as f32));
                    let delta = maximum - height;
                    if maximum.is_infinite() {
                        NEGATIVE_NAN
                    } else {
                        maximum / if delta <= 0.0 { DENSITY_EPSILON } else { delta }
                    }
                } else {
                    training_outlier_score(maximum, height)
                };
                components.outliers.push(score);
            }
            let denominator = self.hierarchy.membership_maximum(nearest)?.max(lambda);
            components.densities.push((
                highest,
                if query {
                    denominator + DENSITY_EPSILON
                } else {
                    denominator
                },
            ));
            Ok::<_, MembershipError>(())
        };

        if let Some(query) = query {
            let rows = self.point_queries(query.values, query.min_samples)?;
            let column_major = query.column_major.then(|| {
                column_major_distance_rows(
                    Points::new(query.values, self.dimensions),
                    self.dimensions,
                )
            });
            for (index, point) in rows.enumerate() {
                let edge = current
                    .departure(point.neighbor)
                    .ok_or(FitError::InvalidHierarchy)?;
                visit(edge, point.lambda, true)?;
                let row = match &column_major {
                    Some(values) => &values[index..index + self.dimensions],
                    None => point.row,
                };
                self.exemplar_weights(old, row, &mut components.distances);
            }
        } else {
            if let Some((tree, _)) = current_fit {
                // For a retained cache the bulk calculation checks the current
                // point departures in tree order before writing them by sample ID.
                for edge in tree.iter().filter(|edge| edge.child_size == 1) {
                    self.hierarchy.membership_maximum(edge.parent)?;
                }
            }
            for sample in 0..current.root {
                let edge = current
                    .departure(sample)
                    .ok_or(FitError::InvalidHierarchy)?;
                visit(edge, f64::INFINITY, false)?;
            }
            let column_major = self
                .column_major_training
                .then(|| column_major_distance_rows(old, self.dimensions));
            for index in 0..old.len() {
                let row = match &column_major {
                    Some(values) => &values[index..index + self.dimensions],
                    None => old.row(index),
                };
                self.exemplar_weights(old, row, &mut components.distances);
            }
        }
        Ok(components)
    }

    fn exemplar_weights(&self, data: Points<'_>, row: &[f64], weights: &mut Vec<f64>) {
        let start = weights.len();
        let exact_match_weight = f64::MAX / self.exemplars.len() as f64;
        let mut total = 0.0;
        for exemplars in &self.exemplars {
            let distance = exemplars.iter().fold(f64::MAX, |nearest, &point| {
                nearest.min(
                    self.metric
                        .distance(self.metric.reduced(row, data.row(point))),
                )
            });
            let weight = if distance == 0.0 {
                exact_match_weight
            } else {
                1.0 / distance
            };
            total += weight;
            weights.push(weight);
        }
        for weight in &mut weights[start..] {
            *weight /= total;
        }
    }
}

// The reference reads adjacent doubles from each row's first feature for
// exemplar distances, including when the source layout is column-major.
fn column_major_distance_rows(points: Points<'_>, dimensions: usize) -> Vec<f64> {
    let rows = points.len();
    if rows == 0 {
        return Vec::new();
    }
    (0..rows - 1 + dimensions)
        .map(|offset| points.row(offset % rows)[offset / rows])
        .collect()
}

impl Hierarchy {
    fn membership_maximum(&self, node: usize) -> Result<f64, MembershipError> {
        if self.has_branches
            && let Some(offset) = node.checked_sub(self.root)
            && let Some(edges) = self.children.get(offset)
            && !edges.is_empty()
        {
            Ok(self.maximum[offset])
        } else {
            Err(MembershipError::MissingDensity(node))
        }
    }

    // An ancestor and its descendant use the query density. Separate branches
    // use the density of the edge at which the two walks meet.
    fn merge_height(
        &self,
        mut left: usize,
        mut right: usize,
    ) -> Result<Option<f64>, MembershipError> {
        let (parent, cluster) = (left, right);
        let mut height = 0.0;
        while left != right {
            let descend = if left > right { &mut left } else { &mut right };
            let Some(Some(edge)) = self.incoming.get(*descend) else {
                return Err(MembershipError::MissingDensity(*descend));
            };
            *descend = edge.parent;
            height = edge.lambda;
        }
        Ok((left != parent && left != cluster).then_some(height))
    }
}

fn training_outlier_score(maximum: f64, height: f64) -> f64 {
    if maximum.is_infinite() && height.is_infinite() {
        // The reference's inner exponential receives NaNs of different signs
        // from the Windows and Linux Wheels.
        if cfg!(target_os = "linux") {
            f64::NAN
        } else {
            NEGATIVE_NAN
        }
    } else {
        (-(maximum + DENSITY_EPSILON) / height).exp()
    }
}

impl MembershipComponents {
    fn finish(mut self, training: bool) -> Result<MembershipMatrix, MembershipError> {
        if self.clusters == 0 {
            return Ok(MembershipMatrix {
                rows: self.distance_rows,
                clusters: 0,
                values: Vec::new(),
            });
        }
        // Native calls address the hierarchy that created the cache.
        if self.clusters != self.distance_clusters || self.distance_rows != self.densities.len() {
            return Err(MembershipError::IncompatibleShapes);
        }
        let global_maximum = row_maximum(&self.outliers);
        let size = self.distances.len();
        for (row, ((weights, scores), (height, maximum))) in self
            .distances
            .chunks_exact_mut(self.clusters)
            .zip(self.outliers.chunks_exact_mut(self.clusters))
            .zip(self.densities)
            .enumerate()
        {
            let peak = if training {
                global_maximum
            } else {
                row_maximum(scores)
            };
            let score_nan = scores.iter().copied().find(|value| value.is_nan());
            if let Some(nan) = score_nan {
                for (column, weight) in weights.iter_mut().enumerate() {
                    let flat_index = row * self.clusters + column;
                    // In the Linux X86_V3 NumPy wheel, the scalar tail of the
                    // broadcast multiply retains the density NaN on ties.
                    *weight = if training
                        && cfg!(all(target_os = "linux", target_arch = "x86_64"))
                        && height.is_infinite()
                        && maximum.is_infinite()
                        && size >= 8
                        && flat_index >= size - size % 8
                    {
                        NEGATIVE_NAN
                    } else {
                        nan
                    };
                }
                continue;
            }
            for score in scores.iter_mut() {
                *score = (*score - peak).exp();
            }
            let sum = pairwise_sum(scores);
            for (weight, &score) in weights.iter_mut().zip(scores.iter()) {
                let normalized = score / sum;
                if training {
                    *weight *= normalized;
                } else {
                    *weight = weight.sqrt() * (normalized * normalized);
                }
            }
            let sum = pairwise_sum(weights);
            let probability = height / maximum;
            for weight in weights {
                *weight = (*weight / sum) * probability;
                if weight.is_nan() {
                    *weight = NEGATIVE_NAN;
                }
            }
        }
        Ok(MembershipMatrix {
            rows: self.distance_rows,
            clusters: self.clusters,
            values: self.distances,
        })
    }
}

fn row_maximum(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NAN, f64::max)
}
