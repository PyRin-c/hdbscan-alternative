// The reference's depth-exhausted partitions have platform-specific tie order.
// Reproduce those permutations with Rust comparisons on the original weights.

use crate::ordering;

#[cfg(target_env = "msvc")]
pub(super) fn sort(order: &mut [usize], weight: impl Fn(usize) -> f64 + Copy) {
    let ideal = order.len();
    sort_msvc(order, weight, ideal);
}

#[cfg(not(target_env = "msvc"))]
pub(super) fn sort(order: &mut [usize], weight: impl Fn(usize) -> f64 + Copy) {
    ordering::partition_gnu_order(order, |&index| weight(index));
    insertion_sort(order, weight);
}

#[cfg(target_env = "msvc")]
fn sort_msvc(mut order: &mut [usize], weight: impl Fn(usize) -> f64 + Copy, mut ideal: usize) {
    while order.len() > 32 {
        if ideal == 0 {
            ordering::heap_sort(order, |&index| weight(index));
            return;
        }
        ideal = ideal / 2 + ideal / 4;
        let middle = order.len() / 2;
        let (lower_end, upper_start) = partition_msvc_at_median(order, middle, weight);
        let (lower, rest) = order.split_at_mut(lower_end);
        let upper = &mut rest[upper_start - lower_end..];
        if lower.len() < upper.len() {
            sort_msvc(lower, weight, ideal);
            order = upper;
        } else {
            sort_msvc(upper, weight, ideal);
            order = lower;
        }
    }
    insertion_sort(order, weight);
}

#[cfg(target_env = "msvc")]
pub(crate) fn partition_msvc_at_median<K: PartialOrd>(
    order: &mut [usize],
    middle: usize,
    weight: impl Fn(usize) -> K + Copy,
) -> (usize, usize) {
    let last = order.len() - 1;
    if last > 40 {
        let stride = order.len() / 8;
        order_three(order, [0, stride, 2 * stride], weight);
        order_three(order, [middle - stride, middle, middle + stride], weight);
        order_three(order, [last - 2 * stride, last - stride, last], weight);
        order_three(order, [stride, middle, last - stride], weight);
    } else {
        order_three(order, [0, middle, last], weight);
    }
    partition_msvc(order, middle, weight)
}

#[cfg(target_env = "msvc")]
fn order_three<K: PartialOrd>(
    order: &mut [usize],
    [first, middle, last]: [usize; 3],
    weight: impl Fn(usize) -> K,
) {
    if weight(order[middle]) < weight(order[first]) {
        order.swap(first, middle);
    }
    if weight(order[last]) < weight(order[middle]) {
        order.swap(middle, last);
        if weight(order[middle]) < weight(order[first]) {
            order.swap(first, middle);
        }
    }
}

#[cfg(target_env = "msvc")]
fn partition_msvc<K: PartialOrd>(
    order: &mut [usize],
    middle: usize,
    weight: impl Fn(usize) -> K,
) -> (usize, usize) {
    let pivot = weight(order[middle]);
    let (mut equal_start, mut equal_end) = (middle, middle + 1);
    while equal_start > 0 && weight(order[equal_start - 1]) == pivot {
        equal_start -= 1;
    }
    while equal_end < order.len() && weight(order[equal_end]) == pivot {
        equal_end += 1;
    }

    // Gather equals in the center as the two scans move outwards.
    let (mut left, mut right) = (equal_start, equal_end);
    loop {
        while right < order.len() && weight(order[right]) >= pivot {
            if weight(order[right]) == pivot {
                order.swap(equal_end, right);
                equal_end += 1;
            }
            right += 1;
        }
        while left > 0 && weight(order[left - 1]) <= pivot {
            left -= 1;
            if weight(order[left]) == pivot {
                equal_start -= 1;
                order.swap(equal_start, left);
            }
        }
        if left == 0 && right == order.len() {
            return (equal_start, equal_end);
        }
        if left == 0 {
            if right != equal_end {
                order.swap(equal_start, equal_end);
            }
            order.swap(equal_start, right);
            equal_start += 1;
            equal_end += 1;
            right += 1;
        } else if right == order.len() {
            left -= 1;
            equal_start -= 1;
            equal_end -= 1;
            order.swap(left, equal_start);
            order.swap(equal_start, equal_end);
        } else {
            left -= 1;
            order.swap(left, right);
            right += 1;
        }
    }
}

fn insertion_sort(order: &mut [usize], weight: impl Fn(usize) -> f64) {
    for offset in 1..order.len() {
        let index = order[offset];
        let key = weight(index);
        let mut hole = offset;
        while hole > 0 && key < weight(order[hole - 1]) {
            order[hole] = order[hole - 1];
            hole -= 1;
        }
        order[hole] = index;
    }
}
