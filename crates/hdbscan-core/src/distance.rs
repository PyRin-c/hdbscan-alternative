use crate::FitError;

mod manhattan;
pub(crate) use manhattan::{CanonicalCsr, dense_l1, dense_manhattan};

pub(crate) fn dense_l2(left: &[f64], right: &[f64]) -> f64 {
    dense_squared_l2(left, right).sqrt()
}

pub(crate) fn dense_squared_l2(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (&x, &y)| {
        let difference = x - y;
        sum + difference * difference
    })
}

pub(crate) struct DenseDistances<'a> {
    values: &'a [f64],
    n_features: usize,
    norms: Vec<f64>,
}

impl<'a> DenseDistances<'a> {
    pub(crate) fn new(values: &'a [f64], n_features: usize) -> Result<Self, FitError> {
        Self::with_norm(values, n_features, dense_squared_norm)
    }

    pub(crate) fn strided_columns(values: &'a [f64], n_features: usize) -> Result<Self, FitError> {
        Self::with_norm(values, n_features, sequential_squared_norm)
    }

    fn with_norm(
        values: &'a [f64],
        n_features: usize,
        squared_norm: fn(&[f64]) -> f64,
    ) -> Result<Self, FitError> {
        let norms: Vec<_> = values.chunks_exact(n_features).map(squared_norm).collect();
        if norms.iter().any(|norm| !norm.is_finite()) {
            return Err(FitError::DistanceOverflow);
        }
        Ok(Self {
            values,
            n_features,
            norms,
        })
    }

    fn row(&self, sample: usize) -> &[f64] {
        let start = sample * self.n_features;
        &self.values[start..start + self.n_features]
    }

    pub(crate) fn between(&self, left: usize, right: usize) -> Result<f64, FitError> {
        if left == right {
            return Ok(0.0);
        }

        // Apply the row norms in the order observed through the reference distance API.
        let dot = dense_dot(self.row(left), self.row(right));
        euclidean_from_dot(dot, self.norms[left], self.norms[right])
    }
}

pub(crate) struct CsrDistances<'a> {
    dots: CsrDotProducts<'a>,
    norms: Vec<f64>,
}

impl<'a> CsrDistances<'a> {
    pub(crate) fn new(
        indptr: &'a [usize],
        indices: &'a [usize],
        data: &'a [f64],
    ) -> Result<Self, FitError> {
        let norms: Vec<_> = indptr
            .windows(2)
            .map(|offsets| sequential_squared_norm(&data[offsets[0]..offsets[1]]))
            .collect();
        if norms.iter().any(|norm| !norm.is_finite()) {
            return Err(FitError::DistanceOverflow);
        }
        Ok(Self {
            dots: CsrDotProducts::new(indptr, indices, data),
            norms,
        })
    }

    pub(crate) fn between(&self, left: usize, right: usize) -> Result<f64, FitError> {
        if left == right {
            return Ok(0.0);
        }
        euclidean_from_dot(
            self.dots.between(left, right),
            self.norms[left],
            self.norms[right],
        )
    }
}

pub(crate) struct CsrDotProducts<'a> {
    indptr: &'a [usize],
    indices: &'a [usize],
    data: &'a [f64],
    sorted_entries: Vec<(usize, f64)>,
    sorted_rows: Vec<bool>,
}

impl<'a> CsrDotProducts<'a> {
    pub(crate) fn new(indptr: &'a [usize], indices: &'a [usize], data: &'a [f64]) -> Self {
        let mut sorted_entries: Vec<_> =
            indices.iter().copied().zip(data.iter().copied()).collect();
        let mut sorted_rows = Vec::with_capacity(indptr.len() - 1);
        for offsets in indptr.windows(2) {
            let range = offsets[0]..offsets[1];
            let is_sorted = indices[range.clone()].is_sorted();
            if !is_sorted {
                // Sort only the right-hand lookup and retain the order within duplicate columns.
                sorted_entries[range].sort_by_key(|&(column, _)| column);
            }
            sorted_rows.push(is_sorted);
        }
        Self {
            indptr,
            indices,
            data,
            sorted_entries,
            sorted_rows,
        }
    }

    pub(crate) fn between(&self, left: usize, right: usize) -> f64 {
        let left_range = self.indptr[left]..self.indptr[left + 1];
        let right_row = &self.sorted_entries[self.indptr[right]..self.indptr[right + 1]];
        let mut right_tail = right_row;
        let mut dot = 0.0;
        for (&column, &value) in self.indices[left_range.clone()]
            .iter()
            .zip(&self.data[left_range])
        {
            let matches = if self.sorted_rows[left] {
                while right_tail.first().is_some_and(|&(index, _)| index < column) {
                    right_tail = &right_tail[1..];
                }
                right_tail
            } else {
                &right_row[right_row.partition_point(|&(index, _)| index < column)..]
            };
            // Each matching pair in storage order contributes a separate rounded product.
            for &(index, other) in matches {
                if index != column {
                    break;
                }
                dot += value * other;
            }
        }
        dot
    }
}

fn dense_squared_norm(row: &[f64]) -> f64 {
    let mut squares = [0.0; 2];
    let mut blocks = row.chunks_exact(8);
    // The reference adds products in each eight-value block back to front in two lanes.
    for block in blocks.by_ref() {
        for (lane, sum) in squares.iter_mut().enumerate() {
            for index in (lane..block.len()).step_by(2).rev() {
                *sum += block[index] * block[index];
            }
        }
    }
    for (index, value) in blocks.remainder().iter().enumerate() {
        squares[index % 2] += value * value;
    }
    squares[0] + squares[1]
}

fn sequential_squared_norm(row: &[f64]) -> f64 {
    row.iter().fold(0.0, |sum, &value| sum + value * value)
}

pub(crate) fn dense_dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0, |sum, (&x, &y)| x.mul_add(y, sum))
}

pub(crate) fn normalize_dense(values: &[f64], n_features: usize) -> Vec<f64> {
    let mut normalized = values.to_vec();
    for row in normalized.chunks_exact_mut(n_features) {
        let norm = dense_squared_norm(row).sqrt();
        if norm >= 10.0 * f64::EPSILON {
            for value in row {
                *value /= norm;
            }
        }
    }
    normalized
}

pub(crate) fn normalize_csr(indptr: &[usize], values: &[f64]) -> Vec<f64> {
    let mut normalized = values.to_vec();
    for offsets in indptr.windows(2) {
        let row = &mut normalized[offsets[0]..offsets[1]];
        let norm = sequential_squared_norm(row).sqrt();
        if norm != 0.0 {
            for value in row {
                *value /= norm;
            }
        }
    }
    normalized
}

pub(crate) fn cosine_from_dot(dot: f64) -> f64 {
    (1.0 - dot).clamp(0.0, 2.0)
}

fn euclidean_from_dot(dot: f64, left_norm: f64, right_norm: f64) -> Result<f64, FitError> {
    let squared = (-2.0 * dot + left_norm) + right_norm;
    if !squared.is_finite() {
        return Err(FitError::DistanceOverflow);
    }
    Ok(squared.max(0.0).sqrt())
}

pub(crate) fn core_distances(
    n_samples: usize,
    min_samples: usize,
    distance: impl Fn(usize, usize) -> Result<f64, FitError>,
) -> Result<Vec<f64>, FitError> {
    let mut column = vec![0.0; n_samples];
    let mut core_distances = Vec::with_capacity(n_samples);
    for target in 0..n_samples {
        for (source, entry) in column.iter_mut().enumerate() {
            *entry = distance(source, target)?;
        }
        let (_, kth_distance, _) = column.select_nth_unstable_by(min_samples, f64::total_cmp);
        core_distances.push(*kth_distance);
    }
    Ok(core_distances)
}

#[cfg(test)]
mod tests {
    use super::{DenseDistances, core_distances};
    use crate::hierarchy::minimum_spanning_tree;

    #[test]
    fn core_distance_counts_other_points_and_alpha_scales_the_pair_distance() {
        let data = [0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        let distances = DenseDistances::new(&data, 2).unwrap();
        let distance = |left, right| distances.between(left, right);
        let core = core_distances(4, 2, distance).unwrap();
        assert_eq!(core, [3.0, 3.0, 4.0, 3.0]);

        let tree = minimum_spanning_tree(&core, 0.5, distance).unwrap();
        let edges: Vec<_> = tree
            .iter()
            .map(|edge| (edge.source, edge.target, edge.distance))
            .collect();
        assert_eq!(edges, [(0, 3, 3.0), (0, 1, 6.0), (0, 2, 8.0)]);
    }
}
