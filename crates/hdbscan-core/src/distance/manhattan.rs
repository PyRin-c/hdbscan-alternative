use std::cmp::Ordering;
use std::ops::{AddAssign, Sub};

use crate::FitError;
use crate::ordering::partition_gnu_order;

fn check_distance(distance: f64) -> Result<f64, FitError> {
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(FitError::DistanceOverflow)
    }
}

pub(crate) fn dense_manhattan(left: &[f64], right: &[f64]) -> Result<f64, FitError> {
    check_distance(dense_l1(left, right))
}

pub(crate) fn dense_l1(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0, |sum, (&x, &y)| sum + (x - y).abs())
}

pub(crate) struct CanonicalCsr<T> {
    indptr: Vec<usize>,
    entries: Vec<(usize, T)>,
}

impl<T: Copy + AddAssign> CanonicalCsr<T> {
    pub(crate) fn new(indptr: &[usize], indices: &[usize], data: &[T]) -> Self {
        let mut row = Vec::new();
        let mut entries: Vec<(usize, T)> = Vec::with_capacity(data.len());
        let mut offsets = Vec::with_capacity(indptr.len());
        let sort_indices = indptr
            .windows(2)
            .any(|bounds| !indices[bounds[0]..bounds[1]].is_sorted());
        offsets.push(0);
        for bounds in indptr.windows(2) {
            let range = bounds[0]..bounds[1];
            row.clear();
            row.extend(
                indices[range.clone()]
                    .iter()
                    .copied()
                    .zip(data[range].iter().copied()),
            );
            if sort_indices {
                order_csr_row(&mut row);
            }
            let row_start = entries.len();
            for &(column, value) in &row {
                if entries.len() > row_start
                    && let Some((last_column, total)) = entries.last_mut()
                    && *last_column == column
                {
                    *total += value;
                } else {
                    entries.push((column, value));
                }
            }
            offsets.push(entries.len());
        }
        Self {
            indptr: offsets,
            entries,
        }
    }

    pub(crate) fn row(&self, row: usize) -> &[(usize, T)] {
        &self.entries[self.indptr[row]..self.indptr[row + 1]]
    }
}

impl<T: Copy + Into<f64> + AddAssign + Sub<Output = T>> CanonicalCsr<T> {
    pub(crate) fn manhattan_between(&self, left: usize, right: usize) -> Result<f64, FitError> {
        let mut left = self.row(left);
        let mut right = self.row(right);
        let mut sum = 0.0;
        while let (Some((&(i, x), next_left)), Some((&(j, y), next_right))) =
            (left.split_first(), right.split_first())
        {
            match i.cmp(&j) {
                Ordering::Less => {
                    sum += x.into().abs();
                    left = next_left;
                }
                Ordering::Greater => {
                    sum += y.into().abs();
                    right = next_right;
                }
                Ordering::Equal => {
                    // float32 inputs round the subtraction before adding to the float64 total.
                    sum += (x - y).into().abs();
                    left = next_left;
                    right = next_right;
                }
            }
        }
        for &(_, value) in left.iter().chain(right) {
            sum += value.into().abs();
        }
        check_distance(sum)
    }
}

fn order_csr_row<T>(row: &mut [(usize, T)]) {
    partition_gnu_order(row, |&(column, _)| column);
    row.sort_by_key(|&(column, _)| column);
}

#[cfg(test)]
mod tests {
    use super::order_csr_row;

    #[test]
    fn duplicate_column_order_matches_reference_after_heap_fallback() {
        let len = 1024_usize;
        let mut row: Vec<_> = (0..len).map(|i| (i.min(len - i) / 2, i as u64)).collect();
        order_csr_row(&mut row);
        assert!(row.iter().map(|&(column, _)| column).is_sorted());
        // Fingerprint the full permutation from SciPy's public csr_matrix.sort_indices().
        let order = row
            .iter()
            .fold(14_695_981_039_346_656_037_u64, |hash, &(_, index)| {
                (hash ^ index).wrapping_mul(1_099_511_628_211)
            });
        assert_eq!(order, 0x97d8_c19c_250d_494b);
    }
}
