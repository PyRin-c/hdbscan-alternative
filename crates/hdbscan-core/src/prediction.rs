use crate::{
    CondensedEdge, FeatureMetric, FitError,
    spatial::{KdIndex, Metric, Neighbor, Points},
};

mod membership;
pub use membership::{MembershipComponents, MembershipError, MembershipMatrix, MembershipQuery};

const PREDICTION_LEAF_SIZE: usize = 40;

/// One assignment to the fitted clustering. Both IDs are -1 when no hierarchy branches exist.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Prediction {
    pub label: i64,
    pub probability: f64,
    pub connecting_point: i64,
}

/// The departure density and the maximum density used to score a predicted point.
#[derive(Clone, Copy, Debug)]
pub struct PredictionDensity {
    pub lambda: f64,
    pub maximum: f64,
}

impl PredictionDensity {
    pub fn score(self) -> f64 {
        if self.maximum == 0.0 {
            0.0
        } else if self.maximum.is_infinite() {
            f64::from_bits(0xfff8_0000_0000_0000)
        } else {
            (self.maximum - self.lambda) / self.maximum
        }
    }
}

/// Cached spatial and hierarchy data for predicting on dense features.
pub struct PredictionModel {
    values: Vec<f64>,
    dimensions: usize,
    metric: Metric<'static>,
    index: KdIndex,
    hierarchy: Hierarchy,
    core_distances: Vec<f64>,
    selected: Vec<usize>,
    cluster_labels: Vec<i64>,
    max_lambdas: Vec<f64>,
    exemplars: Vec<Vec<usize>>,
    column_major_training: bool,
}

impl PredictionModel {
    pub fn new(
        values: Vec<f64>,
        dimensions: usize,
        metric: FeatureMetric,
        min_samples: usize,
        labels: &[i64],
        condensed_tree: &[CondensedEdge],
    ) -> Result<Self, FitError> {
        Self::new_with_distance(
            values,
            dimensions,
            metric.try_into()?,
            min_samples,
            labels,
            condensed_tree,
        )
    }

    /// Prepares prediction data with a distance supported by the KD-tree.
    pub fn new_with_distance(
        values: Vec<f64>,
        dimensions: usize,
        metric: Metric<'static>,
        min_samples: usize,
        labels: &[i64],
        condensed_tree: &[CondensedEdge],
    ) -> Result<Self, FitError> {
        if labels.is_empty() || dimensions == 0 {
            return Err(FitError::InvalidShape);
        }
        if labels.len().checked_mul(dimensions) != Some(values.len()) {
            return Err(FitError::LengthMismatch);
        }
        if min_samples == 0 {
            return Err(FitError::InvalidMinSamples);
        }
        if min_samples > labels.len() {
            return Err(FitError::InsufficientTreePoints);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FitError::NonFiniteInput);
        }
        if !metric.supports_kd() {
            return Err(FitError::UnsupportedMetric);
        }
        let metric = metric.validated(dimensions)?;
        let hierarchy = Hierarchy::new(condensed_tree, labels.len())?;
        let points = Points::new(&values, dimensions);
        let index = KdIndex::new(points, metric, PREDICTION_LEAF_SIZE);
        let spatial_tree = index.tree(points, metric);
        let mut neighbors = neighbor_buffer(min_samples);
        let core_distances = (0..points.len())
            .map(|sample| {
                spatial_tree.query(points.row(sample), &mut neighbors);
                neighbors[min_samples - 1].distance
            })
            .collect();
        let selected = hierarchy.selected_clusters(labels)?;
        let mut cluster_labels = vec![-1; hierarchy.children.len()];
        let mut max_lambdas = vec![0.0; hierarchy.children.len()];
        let mut exemplars = Vec::with_capacity(selected.len());
        for (label, &node) in selected.iter().enumerate() {
            cluster_labels[node - labels.len()] = label as i64;
        }
        for &node in &selected {
            let label = cluster_labels[node - labels.len()];
            let maximum = hierarchy.maximum[node - labels.len()];
            let mut examples = Vec::new();
            let mut stack = vec![node];
            while let Some(node) = stack.pop() {
                let offset = node - labels.len();
                cluster_labels[offset] = label;
                max_lambdas[offset] = maximum;
                let children = &hierarchy.children[offset];
                let branches = children.iter().rev().filter(|edge| edge.child_size > 1);
                stack.extend(branches.map(|edge| edge.child));
                if !children.iter().any(|edge| edge.child_size > 1) {
                    examples.extend(
                        children
                            .iter()
                            .filter(|edge| edge.lambda == hierarchy.maximum[offset])
                            .map(|edge| edge.child),
                    );
                }
            }
            exemplars.push(examples);
        }
        Ok(Self {
            values,
            dimensions,
            metric,
            index,
            hierarchy,
            core_distances,
            selected,
            cluster_labels,
            max_lambdas,
            exemplars,
            column_major_training: false,
        })
    }

    /// Records that the packed input rows came from a column-major matrix.
    /// Training memberships use the source layout for exemplar distances.
    pub fn with_column_major_training(mut self) -> Self {
        self.column_major_training = true;
        self
    }

    pub fn has_branches(&self) -> bool {
        self.hierarchy.has_branches
    }

    pub fn core_distances(&self) -> &[f64] {
        &self.core_distances
    }

    pub fn selected_clusters(&self) -> &[usize] {
        &self.selected
    }

    pub fn exemplars(&self) -> &[Vec<usize>] {
        &self.exemplars
    }

    /// Returns the cluster ID, label, and maximum density for every mapped node.
    pub fn cluster_mapping(&self) -> impl Iterator<Item = (usize, i64, f64)> + '_ {
        self.cluster_labels
            .iter()
            .zip(&self.max_lambdas)
            .enumerate()
            .filter(|(_, (label, _))| **label >= 0)
            .map(|(offset, (&label, &maximum))| (self.hierarchy.root + offset, label, maximum))
    }

    pub fn leaf_max_lambdas(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.hierarchy
            .maximum
            .iter()
            .enumerate()
            .filter_map(|(offset, &lambda)| {
                (self.has_branches() && !self.hierarchy.children[offset].is_empty())
                    .then_some((self.hierarchy.root + offset, lambda))
            })
    }

    /// Assigns each row to the hierarchy. An override supplies the current hierarchy
    /// for callers that retain prediction data when fitting again.
    pub fn predict(
        &self,
        values: &[f64],
        min_samples: usize,
        tree_override: Option<&[CondensedEdge]>,
    ) -> Result<Vec<Prediction>, FitError> {
        self.with_neighbors(
            values,
            min_samples,
            tree_override,
            Prediction {
                label: -1,
                probability: 0.0,
                connecting_point: -1,
            },
            |hist, neighbor, lambda, _| {
                let mut result = Prediction {
                    label: -1,
                    probability: 0.0,
                    connecting_point: neighbor as i64,
                };
                let Some(edge) = hist.departure(neighbor) else {
                    return result;
                };
                let mut node = edge.parent;
                if edge.lambda > lambda {
                    // Refitting without rebuilding the prediction cache still
                    // uses the cached cluster branches for the walk upward.
                    while node > self.hierarchy.root
                        && let Some(Some(birth)) = self.hierarchy.incoming.get(node)
                    {
                        if birth.lambda < lambda {
                            break;
                        }
                        node = birth.parent;
                    }
                }
                if let Some(offset) = node.checked_sub(self.hierarchy.root)
                    && let Some(&label) = self.cluster_labels.get(offset)
                    && label >= 0
                {
                    let maximum = self.max_lambdas[offset];
                    result.label = label;
                    result.probability = if maximum > 0.0 {
                        maximum.min(lambda) / maximum
                    } else {
                        1.0
                    };
                }
                result
            },
        )
    }

    /// Returns the density pairs used by approximate GLOSH scoring.
    pub fn predict_densities(
        &self,
        values: &[f64],
        min_samples: usize,
        tree_override: Option<&[CondensedEdge]>,
    ) -> Result<Vec<PredictionDensity>, FitError> {
        self.with_neighbors(
            values,
            min_samples,
            tree_override,
            PredictionDensity {
                lambda: 0.0,
                maximum: 1.0,
            },
            |hist, neighbor, lambda, coincides_with_training_point| {
                let Some(edge) = hist.departure(neighbor) else {
                    return PredictionDensity {
                        lambda: 0.0,
                        maximum: 0.0,
                    };
                };
                PredictionDensity {
                    lambda: if coincides_with_training_point {
                        edge.lambda
                    } else {
                        lambda
                    },
                    maximum: hist.propagated_maximum[edge.parent - hist.root],
                }
            },
        )
    }

    fn with_neighbors<T: Clone>(
        &self,
        values: &[f64],
        min_samples: usize,
        tree_override: Option<&[CondensedEdge]>,
        unclustered: T,
        mut classify: impl FnMut(&Hierarchy, usize, f64, bool) -> T,
    ) -> Result<Vec<T>, FitError> {
        if !values.len().is_multiple_of(self.dimensions) {
            return Err(FitError::LengthMismatch);
        }
        if !self.has_branches() {
            return Ok(vec![unclustered; values.len() / self.dimensions]);
        }
        let override_hierarchy = tree_override
            .map(|tree| {
                let root = tree
                    .first()
                    .map_or(self.core_distances.len(), |edge| edge.parent);
                Hierarchy::new(tree, root)
            })
            .transpose()?;
        let hierarchy = override_hierarchy.as_ref().unwrap_or(&self.hierarchy);
        Ok(self
            .point_queries(values, min_samples)?
            .map(|query| {
                classify(
                    hierarchy,
                    query.neighbor,
                    query.lambda,
                    query.training_distance_zero,
                )
            })
            .collect())
    }

    fn point_queries<'a>(
        &'a self,
        values: &'a [f64],
        min_samples: usize,
    ) -> Result<impl Iterator<Item = PointQuery<'a>> + 'a, FitError> {
        if !values.len().is_multiple_of(self.dimensions) {
            return Err(FitError::LengthMismatch);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FitError::NonFiniteInput);
        }
        if min_samples == 0 {
            return Err(FitError::InvalidMinSamples);
        }
        let count = min_samples
            .checked_mul(2)
            .ok_or(FitError::InsufficientTreePoints)?;
        let points = Points::new(&self.values, self.dimensions);
        if count > points.len() {
            return Err(FitError::InsufficientTreePoints);
        }
        let tree = self.index.tree(points, self.metric);
        let mut neighbors = neighbor_buffer(count);
        Ok(values.chunks_exact(self.dimensions).map(move |row| {
            tree.query(row, &mut neighbors);
            let point_core = neighbors[min_samples].distance;
            let mut nearest = 0;
            let mut reachability = f64::INFINITY;
            for (index, entry) in neighbors.iter().enumerate() {
                let reach = entry
                    .distance
                    .max(point_core)
                    .max(self.core_distances[entry.point]);
                if reach < reachability {
                    nearest = index;
                    reachability = reach;
                }
            }
            let lambda = if reachability > 0.0 {
                1.0 / reachability
            } else {
                f64::MAX
            };
            PointQuery {
                row,
                neighbor: neighbors[nearest].point,
                lambda,
                training_distance_zero: neighbors[0].distance == 0.0,
            }
        }))
    }
}

struct PointQuery<'a> {
    row: &'a [f64],
    neighbor: usize,
    lambda: f64,
    training_distance_zero: bool,
}

struct Hierarchy {
    root: usize,
    has_branches: bool,
    first_edge: Option<CondensedEdge>,
    incoming: Vec<Option<CondensedEdge>>,
    children: Vec<Vec<CondensedEdge>>,
    maximum: Vec<f64>,
    propagated_maximum: Vec<f64>,
}

impl Hierarchy {
    fn new(tree: &[CondensedEdge], root: usize) -> Result<Self, FitError> {
        if !tree.is_empty() && root > tree.len() {
            return Err(FitError::InvalidHierarchy);
        }
        let cluster_count = tree.len().saturating_sub(root) + 1;
        let mut incoming = vec![None; root + cluster_count];
        let mut children = vec![Vec::new(); cluster_count];
        let mut maximum = vec![0.0_f64; cluster_count];
        for &edge in tree {
            if edge.parent < root
                || edge.parent >= incoming.len()
                || edge.child >= incoming.len()
                || (edge.child >= root && edge.child <= edge.parent)
                || (edge.child >= root) != (edge.child_size > 1)
            {
                return Err(FitError::InvalidHierarchy);
            }
            incoming[edge.child] = Some(edge);
            let offset = edge.parent - root;
            children[offset].push(edge);
            maximum[offset] = maximum[offset].max(edge.lambda);
        }
        let mut propagated_maximum = maximum.clone();
        for child in (root + 1..incoming.len()).rev() {
            if let Some(edge) = incoming[child] {
                let parent = edge.parent - root;
                propagated_maximum[parent] =
                    propagated_maximum[parent].max(propagated_maximum[child - root]);
            }
        }
        Ok(Self {
            root,
            has_branches: tree.iter().any(|edge| edge.child_size > 1),
            first_edge: tree.first().copied(),
            incoming,
            children,
            maximum,
            propagated_maximum,
        })
    }

    fn departure(&self, sample: usize) -> Option<CondensedEdge> {
        // A retained cache can refer to a row ID that now belongs to a cluster
        // in a shorter fit. The current tree is searched by ID in either case.
        self.incoming
            .get(sample)
            .copied()
            .flatten()
            .or(self.first_edge)
    }

    fn selected_clusters(&self, labels: &[i64]) -> Result<Vec<usize>, FitError> {
        let mut selected = vec![usize::MAX; labels.len()];
        for (sample, &label) in labels.iter().enumerate() {
            if label >= 0 {
                let label = selected
                    .get_mut(label as usize)
                    .ok_or(FitError::InvalidHierarchy)?;
                let edge = self.incoming[sample].ok_or(FitError::InvalidHierarchy)?;
                *label = (*label).min(edge.parent);
            }
        }
        selected.retain(|&node| node != usize::MAX);
        selected.sort_unstable();
        selected.dedup();
        Ok(selected)
    }
}

fn neighbor_buffer(size: usize) -> Vec<Neighbor> {
    vec![
        Neighbor {
            point: 0,
            distance: f64::INFINITY
        };
        size
    ]
}
