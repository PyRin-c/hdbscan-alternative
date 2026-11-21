/// Reproduces the partition order before the final stable sort of the short runs.
pub(crate) fn partition_gnu_order<T, K: PartialOrd>(items: &mut [T], key: impl Fn(&T) -> K + Copy) {
    if items.len() > 16 {
        partition_runs(items, 2 * items.len().ilog2(), key);
    }
}

fn partition_runs<T, K: PartialOrd>(
    mut items: &mut [T],
    mut budget: u32,
    key: impl Fn(&T) -> K + Copy,
) {
    while items.len() > 16 {
        if budget == 0 {
            heap_sort(items, key);
            return;
        }
        budget -= 1;
        let split = partition_at_median(items, key);
        let (left, right) = items.split_at_mut(split);
        if left.len() < right.len() {
            partition_runs(left, budget, key);
            items = right;
        } else {
            partition_runs(right, budget, key);
            items = left;
        }
    }
}

fn partition_at_median<T, K: PartialOrd>(items: &mut [T], key: impl Fn(&T) -> K) -> usize {
    let middle = items.len() / 2;
    let last = items.len() - 1;
    let (a, b, c) = (key(&items[1]), key(&items[middle]), key(&items[last]));
    let median = if a < b {
        if b < c {
            middle
        } else if a < c {
            last
        } else {
            1
        }
    } else if a < c {
        1
    } else if b < c {
        last
    } else {
        middle
    };
    items.swap(0, median);
    let pivot = key(&items[0]);
    let mut left = 1;
    let mut right = items.len();
    loop {
        while key(&items[left]) < pivot {
            left += 1;
        }
        right -= 1;
        while pivot < key(&items[right]) {
            right -= 1;
        }
        if left >= right {
            return left;
        }
        items.swap(left, right);
        left += 1;
    }
}

/// Produces the median partition and the point order exposed by spatial trees.
/// The secondary row number makes the coordinate key unique.
pub(crate) fn select_spatial_median(items: &mut [usize], coordinate: impl Fn(usize) -> f64 + Copy) {
    let key = |index| (coordinate(index), index);
    select_by_key(items, items.len() / 2, key);
}

#[cfg(target_env = "msvc")]
fn select_by_key<K: PartialOrd>(
    mut items: &mut [usize],
    mut target: usize,
    key: impl Fn(usize) -> K + Copy,
) {
    while items.len() > 32 {
        let middle = items.len() / 2;
        let (lower, upper) = crate::hierarchy::partition_msvc_at_median(items, middle, key);
        if target >= upper {
            items = &mut items[upper..];
            target -= upper;
        } else if target < lower {
            items = &mut items[..lower];
        } else {
            return;
        }
    }
    items.sort_unstable_by(|&left, &right| key(left).partial_cmp(&key(right)).unwrap());
}

#[cfg(not(target_env = "msvc"))]
fn select_by_key<K: PartialOrd>(
    mut items: &mut [usize],
    mut target: usize,
    key: impl Fn(usize) -> K + Copy,
) {
    let mut budget = 2 * items.len().ilog2();
    while items.len() > 3 {
        if budget == 0 {
            let (heap, tail) = items.split_at_mut(target + 1);
            for parent in (0..heap.len() / 2).rev() {
                sift(heap, parent, |&index| key(index));
            }
            for item in tail {
                if key(*item) < key(heap[0]) {
                    std::mem::swap(item, &mut heap[0]);
                    sift(heap, 0, |&index| key(index));
                }
            }
            heap.swap(0, target);
            return;
        }
        budget -= 1;
        let split = partition_at_median(items, |&index| key(index));
        if target >= split {
            items = &mut items[split..];
            target -= split;
        } else {
            items = &mut items[..split];
        }
    }
    items.sort_unstable_by(|&left, &right| key(left).partial_cmp(&key(right)).unwrap());
}

pub(crate) fn heap_sort<T, K: PartialOrd>(items: &mut [T], key: impl Fn(&T) -> K + Copy) {
    for parent in (0..items.len() / 2).rev() {
        sift(items, parent, key);
    }
    for end in (1..items.len()).rev() {
        items.swap(0, end);
        sift(&mut items[..end], 0, key);
    }
}

// The reference packed-axis reduction accumulates eight lanes in blocks of 128.
pub(crate) fn pairwise_sum(values: &[f64]) -> f64 {
    match values.len() {
        0..8 => values.iter().sum(),
        8..=128 => {
            let mut blocks = values.chunks_exact(8);
            let mut lanes: [f64; 8] = blocks.next().unwrap().try_into().unwrap();
            for block in blocks.by_ref() {
                for (lane, value) in lanes.iter_mut().zip(block) {
                    *lane += value;
                }
            }
            let [a, b, c, d, e, f, g, h] = lanes;
            let base = ((a + b) + (c + d)) + ((e + f) + (g + h));
            blocks
                .remainder()
                .iter()
                .fold(base, |sum, &value| sum + value)
        }
        _ => {
            let (left, right) = values.split_at(values.len() / 2 / 8 * 8);
            pairwise_sum(left) + pairwise_sum(right)
        }
    }
}

fn sift<T, K: PartialOrd>(items: &mut [T], start: usize, key: impl Fn(&T) -> K) {
    // Promote the larger child, taking the right child on equal keys, then sift up.
    let mut hole = start;
    while hole < items.len() / 2 {
        let left = 2 * hole + 1;
        let right = left + 1;
        let child = if right < items.len() && key(&items[left]) <= key(&items[right]) {
            right
        } else {
            left
        };
        items.swap(hole, child);
        hole = child;
    }
    while hole > start {
        let parent = (hole - 1) / 2;
        if key(&items[parent]) >= key(&items[hole]) {
            break;
        }
        items.swap(parent, hole);
        hole = parent;
    }
}
