use std::ops::AddAssign;

use crate::distance::{CanonicalCsr, DenseDistances};
use crate::{
    CondensedEdge, FeatureMetric, FitConfiguration, FitError, FitResult, LinkageMerge,
    fit_csr_configured, fit_csr_manhattan_f32_configured, fit_dense_configured,
    fit_euclidean_distances, validate_fit,
};

/// Fits dense features, reporting rows with any non-finite value as noise.
///
/// Linkage and condensed tree IDs refer to the original rows. MST IDs refer
/// to the retained rows in their original order, as in the Python estimator.
pub fn fit_dense_with_missing(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    metric: FeatureMetric,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    fit_dense_rows_with_missing(values, n_samples, n_features, |values, n_samples| {
        fit_dense_configured(values, n_samples, n_features, metric, configuration)
    })
}

/// Fits row-major Euclidean features copied from a matrix with non-unit column stride.
/// The input layout before packing determines the reference row-norm accumulation order.
pub fn fit_dense_euclidean_strided_columns(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    fit_dense_rows_with_missing(values, n_samples, n_features, |values, rows| {
        let min_samples = validate_fit(values.len(), rows, n_features, configuration.options)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FitError::NonFiniteInput);
        }
        let distances = if rows == n_samples {
            DenseDistances::strided_columns(values, n_features)?
        } else {
            // Compacting the finite rows produces packed columns before fitting.
            DenseDistances::new(values, n_features)?
        };
        fit_euclidean_distances(distances, rows, min_samples, configuration)
    })
}

pub(crate) fn fit_dense_rows_with_missing(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    run: impl Fn(&[f64], usize) -> Result<FitResult, FitError>,
) -> Result<FitResult, FitError> {
    match run(values, n_samples) {
        Err(FitError::NonFiniteInput) => {
            let rows = Rows::new(
                values
                    .chunks_exact(n_features)
                    .map(|row| row.iter().all(|value| value.is_finite())),
                n_features,
            )?;
            let mut clean = Vec::with_capacity(rows.finite.len() * n_features);
            for &row in &rows.finite {
                clean.extend_from_slice(&values[row * n_features..(row + 1) * n_features]);
            }
            Ok(rows.restore(run(&clean, rows.finite.len())?))
        }
        result => result,
    }
}

/// Fits CSR features, reporting rows with any non-finite stored value as noise.
/// Tree IDs follow [`fit_dense_with_missing`].
pub fn fit_csr_with_missing(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    n_features: usize,
    metric: FeatureMetric,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    fit_sparse_with_missing(
        indptr,
        indices,
        data,
        n_features,
        |x| x.is_finite(),
        |indptr, indices, data| {
            fit_csr_configured(indptr, indices, data, n_features, metric, configuration)
        },
    )
}

/// Fits float32 CSR Manhattan features and reports non-finite rows as noise.
pub fn fit_csr_manhattan_f32_with_missing(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    n_features: usize,
    configuration: FitConfiguration,
) -> Result<FitResult, FitError> {
    fit_sparse_with_missing(
        indptr,
        indices,
        data,
        n_features,
        |x| x.is_finite(),
        |indptr, indices, data| {
            fit_csr_manhattan_f32_configured(indptr, indices, data, n_features, configuration)
        },
    )
}

fn fit_sparse_with_missing<T: Copy + AddAssign>(
    indptr: &[usize],
    indices: &[usize],
    data: &[T],
    n_features: usize,
    finite: impl Fn(T) -> bool,
    run: impl Fn(&[usize], &[usize], &[T]) -> Result<FitResult, FitError>,
) -> Result<FitResult, FitError> {
    match run(indptr, indices, data) {
        Err(FitError::NonFiniteInput) => {
            // Once a row contains nonfinite values, the reference combines
            // duplicate columns across the CSR matrix before testing rows.
            let canonical = CanonicalCsr::new(indptr, indices, data);
            let rows = Rows::new(
                (0..indptr.len() - 1)
                    .map(|index| canonical.row(index).iter().all(|&(_, value)| finite(value))),
                n_features,
            )?;
            let mut clean_indptr = Vec::with_capacity(rows.finite.len() + 1);
            let mut clean_indices = Vec::new();
            let mut clean_data = Vec::new();
            clean_indptr.push(0);
            for &row in &rows.finite {
                for &(column, value) in canonical.row(row) {
                    clean_indices.push(column);
                    clean_data.push(value);
                }
                clean_indptr.push(clean_data.len());
            }
            Ok(rows.restore(run(&clean_indptr, &clean_indices, &clean_data)?))
        }
        result => result,
    }
}

pub(crate) struct Rows {
    finite: Vec<usize>,
    excluded: Vec<usize>,
}

impl Rows {
    pub(crate) fn new(
        finite_rows: impl Iterator<Item = bool>,
        n_features: usize,
    ) -> Result<Self, FitError> {
        let mut rows = Self {
            finite: Vec::new(),
            excluded: Vec::new(),
        };
        for (index, finite) in finite_rows.enumerate() {
            if finite {
                rows.finite.push(index);
            } else {
                rows.excluded.push(index);
            }
        }
        match rows.finite.len() {
            0 => Err(FitError::NoFiniteFeatureRows { n_features }),
            1 => Err(FitError::OneFiniteFeatureRow),
            _ => Ok(rows),
        }
    }

    pub(crate) fn restore(&self, mut fit: FitResult) -> FitResult {
        let n_finite = self.finite.len();
        let n_samples = n_finite + self.excluded.len();
        let offset = self.excluded.len();
        let original_id = |id: usize| self.finite.get(id).copied().unwrap_or(id + offset);
        let missing = wheel_set_order(&self.excluded);

        let mut labels = vec![-1; n_samples];
        let mut probabilities = vec![0.0; n_samples];
        for (compact, &original) in self.finite.iter().enumerate() {
            labels[original] = fit.labels[compact];
            probabilities[original] = fit.probabilities[compact];
        }
        fit.labels = labels;
        fit.probabilities = probabilities;

        let mut largest_endpoint = 0;
        for merge in &mut fit.single_linkage_tree {
            merge.left = original_id(merge.left);
            merge.right = original_id(merge.right);
            largest_endpoint = largest_endpoint.max(merge.left).max(merge.right);
        }
        // The reference starts at one past the largest endpoint, including
        // when a two-point fit has only original row IDs in its endpoints.
        let root = largest_endpoint + 1;
        let root_size = fit
            .single_linkage_tree
            .last()
            .map_or(n_finite, |edge| edge.size);
        fit.single_linkage_tree
            .extend(
                missing
                    .iter()
                    .enumerate()
                    .map(|(index, &sample)| LinkageMerge {
                        left: sample,
                        right: root + index,
                        distance: f64::INFINITY,
                        size: root_size.saturating_add(index + 1),
                    }),
            );

        let mut condensed = Vec::with_capacity(fit.condensed_tree.len() + offset);
        condensed.extend(missing.into_iter().map(|sample| CondensedEdge {
            parent: n_samples,
            child: sample,
            lambda: 0.0,
            child_size: 1,
        }));
        condensed.extend(fit.condensed_tree.into_iter().map(|edge| CondensedEdge {
            parent: edge.parent + offset,
            child: original_id(edge.child),
            ..edge
        }));
        fit.condensed_tree = condensed;
        fit
    }
}

const EMPTY: usize = usize::MAX;

// The pinned Wheel exposes the order of a CPython 3.13 set of row IDs.
fn wheel_set_order(rows: &[usize]) -> Vec<usize> {
    let mut table = vec![EMPTY; 8];
    for (index, &row) in rows.iter().enumerate() {
        insert(&mut table, row);
        let used = index + 1;
        if used * 5 >= (table.len() - 1) * 3 {
            let minimum = used * if used <= 50_000 { 4 } else { 2 };
            let mut grown = vec![EMPTY; (minimum + 1).next_power_of_two()];
            for id in table.into_iter().filter(|&id| id != EMPTY) {
                insert(&mut grown, id);
            }
            table = grown;
        }
    }
    table.into_iter().filter(|&id| id != EMPTY).collect()
}

fn insert(table: &mut [usize], row: usize) {
    let mask = table.len() - 1;
    let mut perturb = row;
    let mut slot = row & mask;
    loop {
        let probes = if slot + 9 <= mask { 10 } else { 1 };
        if let Some(empty) = table[slot..slot + probes]
            .iter_mut()
            .find(|entry| **entry == EMPTY)
        {
            *empty = row;
            return;
        }
        perturb >>= 5;
        slot = slot.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
    }
}
