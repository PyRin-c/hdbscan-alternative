use crate::hierarchy::{edges_in_linkage_order, root, single_linkage};
use crate::{
    FitConfiguration, FitError, FitOptions, FitResult, SpanningEdge, finish_fit,
    valid_sample_count, validate_options,
};

pub(crate) fn validate_csr(
    indptr: &[usize],
    indices: &[usize],
    data_len: usize,
    n_features: usize,
) -> Result<usize, FitError> {
    let n_samples = indptr.len().saturating_sub(1);
    if !valid_sample_count(n_samples) || n_features == 0 {
        return Err(FitError::InvalidShape);
    }
    if indices.len() != data_len
        || indptr[0] != 0
        || indptr[n_samples] != data_len
        || indptr.windows(2).any(|pair| pair[0] > pair[1])
        || indices.iter().any(|&column| column >= n_features)
    {
        return Err(FitError::InvalidCsr);
    }
    Ok(n_samples)
}

/// Fits HDBSCAN to a square precomputed distance matrix in CSR format.
///
/// The matrix size is `indptr.len() - 1`. Duplicate columns in a row are summed.
/// The stored distances, including explicit zeros, contribute to the row's core
/// distance. Only nonzero scaled entries can become reachability edges. A positive
/// `max_dist` replaces a nonfinite weight on a stored edge. Missing entries are
/// absent edges. The MST is returned in linkage order.
pub fn fit_csr_precomputed(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    options: FitOptions,
    max_dist: f64,
) -> Result<FitResult, FitError> {
    fit_csr_precomputed_configured(indptr, indices, data, options.into(), max_dist)
}

/// Fits a square CSR distance matrix with cluster selection options.
pub fn fit_csr_precomputed_configured(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    configuration: FitConfiguration,
    max_dist: f64,
) -> Result<FitResult, FitError> {
    let options = configuration.options;
    let n_samples = validate_csr(indptr, indices, data.len(), indptr.len().saturating_sub(1))?;
    let min_samples = validate_options(n_samples, options)?;
    configuration.cluster_selection_thresholds.validate()?;
    if data.iter().any(|distance| !distance.is_finite()) {
        return Err(FitError::NonFiniteInput);
    }

    let mut components = Components::new(n_samples);
    let mut entries = Vec::with_capacity(data.len());
    let mut core_distances = Vec::with_capacity(n_samples);
    let mut row = Vec::new();
    let mut distances = Vec::new();
    // The reference sparse path rounds these options to float32 before using them.
    let alpha = f64::from(options.alpha as f32);
    let max_dist = f64::from(max_dist as f32);

    for (source, offsets) in indptr.windows(2).enumerate() {
        let range = offsets[0]..offsets[1];
        row.clear();
        row.extend(
            indices[range.clone()]
                .iter()
                .copied()
                .zip(data[range].iter().copied()),
        );
        row.sort_by_key(|&(column, _)| column);

        distances.clear();
        let mut remaining = row.as_slice();
        while let Some((&(target, mut distance), rest)) = remaining.split_first() {
            remaining = rest;
            while let Some((&(column, duplicate), rest)) = remaining.split_first() {
                if column != target {
                    break;
                }
                distance += duplicate;
                remaining = rest;
            }
            if !distance.is_finite() {
                return Err(FitError::DistanceOverflow);
            }
            components.join(source, target);
            distances.push(distance);

            let scaled = distance / alpha;
            if source != target && scaled != 0.0 && distance != 0.0 {
                entries.push(SpanningEdge {
                    source,
                    target,
                    distance: scaled,
                });
            }
        }
        let core = if distances.len() < min_samples {
            f64::INFINITY
        } else {
            *distances
                .select_nth_unstable_by(min_samples - 1, f64::total_cmp)
                .1
        };
        core_distances.push(core);
    }

    if components.count != 1 {
        return Err(FitError::DisconnectedSparseGraph);
    }

    entries.retain_mut(|edge| {
        let distance = edge
            .distance
            .max(core_distances[edge.source])
            .max(core_distances[edge.target]);
        edge.distance = if !distance.is_finite() && max_dist > 0.0 {
            max_dist
        } else if distance.is_finite() {
            distance
        } else {
            return false;
        };
        edge.distance != 0.0
    });

    let tree = spanning_forest(n_samples, entries);
    if tree.len() != n_samples - 1 {
        return Err(FitError::InsufficientSparseNeighbors { min_samples });
    }

    let sorted_tree: Vec<_> = edges_in_linkage_order(&tree).collect();
    let linkage = single_linkage(sorted_tree.iter().copied(), n_samples);
    finish_fit(n_samples, configuration, sorted_tree, linkage)
}

pub(crate) fn spanning_forest(
    n_samples: usize,
    mut entries: Vec<SpanningEdge>,
) -> Vec<SpanningEdge> {
    // Stable ties retain the canonical CSR order when Kruskal selects an edge.
    entries.sort_by(|left, right| left.distance.total_cmp(&right.distance));
    let mut components = Components::new(n_samples);
    let mut tree = Vec::with_capacity(n_samples.saturating_sub(1));
    for edge in entries {
        if components.join(edge.source, edge.target) {
            tree.push(edge);
            if components.count == 1 {
                break;
            }
        }
    }
    // The reference visits the MST in CSR order before its linkage sort.
    tree.sort_unstable_by_key(|edge| (edge.source, edge.target));
    tree
}

struct Components {
    parents: Vec<usize>,
    sizes: Vec<usize>,
    count: usize,
}

impl Components {
    fn new(count: usize) -> Self {
        Self {
            parents: (0..count).collect(),
            sizes: vec![1; count],
            count,
        }
    }

    fn join(&mut self, left: usize, right: usize) -> bool {
        let mut left = root(&mut self.parents, left);
        let mut right = root(&mut self.parents, right);
        if left == right {
            return false;
        }
        if self.sizes[left] < self.sizes[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parents[right] = left;
        self.sizes[left] += self.sizes[right];
        self.count -= 1;
        true
    }
}
