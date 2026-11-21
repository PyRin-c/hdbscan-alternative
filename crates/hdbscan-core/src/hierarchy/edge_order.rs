use crate::SpanningEdge;

mod fallback;
#[cfg(target_env = "msvc")]
pub(crate) use fallback::partition_msvc_at_median;

const NETWORK_CAPACITY: usize = 256;
const PARTITION_BLOCK: usize = 16;
const PADDING: usize = usize::MAX;

pub(super) fn sort_indices(edges: &[SpanningEdge]) -> Vec<usize> {
    let mut order: Vec<_> = (0..edges.len()).collect();
    if edges
        .windows(2)
        .any(|pair| pair[0].distance > pair[1].distance)
    {
        let budget = 2 * order.len().ilog2();
        let weight = |index| {
            edges
                .get(index)
                .map_or(f64::INFINITY, |edge: &SpanningEdge| edge.distance)
        };
        sort_partition(&mut order, weight, budget);
    }
    order
}

fn sort_partition(order: &mut [usize], weight: impl Fn(usize) -> f64 + Copy, budget: u32) {
    if order.len() < 2 {
        return;
    }
    if order.len() <= NETWORK_CAPACITY {
        sort_network(order, weight);
        return;
    }
    if budget == 0 {
        fallback::sort(order, weight);
        return;
    }

    let stride = (order.len() - 1) / 4;
    let mut samples = [1, 2, 3, 4].map(|sample| weight(order[sample * stride]));
    samples.sort_unstable_by(f64::total_cmp);
    let pivot = samples[2];
    let middle = partition(order, |index| weight(index) < pivot);
    let (lower, upper) = order.split_at_mut(middle);
    sort_partition(lower, weight, budget - 1);
    if upper.iter().any(|&index| weight(index) > pivot) {
        sort_partition(upper, weight, budget - 1);
    }
}

fn sort_network(order: &mut [usize], weight: impl Fn(usize) -> f64) {
    let width = order.len().next_power_of_two();
    let mut indices = [PADDING; NETWORK_CAPACITY];
    indices[..order.len()].copy_from_slice(order);
    // Mirrored compare-exchanges leave equal keys in their current lanes.
    for power in 1..=width.ilog2() {
        let run = 1 << power;
        for mask in std::iter::once(run - 1).chain((0..power - 1).rev().map(|bit| 1 << bit)) {
            for left in 0..width {
                let right = left ^ mask;
                if left < right && weight(indices[left]) > weight(indices[right]) {
                    indices.swap(left, right);
                }
            }
        }
    }
    for (output, index) in order
        .iter_mut()
        .zip(indices.into_iter().filter(|&index| index != PADDING))
    {
        *output = index;
    }
}

fn partition(order: &mut [usize], is_lower: impl Fn(usize) -> bool) -> usize {
    let copy_block = |order: &[usize], start| -> [usize; PARTITION_BLOCK] {
        order[start..start + PARTITION_BLOCK].try_into().unwrap()
    };
    let mut left_write = 0;
    let mut right_write = order.len();

    // Align the unread region to full blocks before saving the two end blocks.
    for _ in 0..order.len() % PARTITION_BLOCK {
        if is_lower(order[left_write]) {
            left_write += 1;
        } else {
            right_write -= 1;
            order.swap(left_write, right_write);
        }
    }
    let mut left_read = left_write + PARTITION_BLOCK;
    let mut right_read = right_write - PARTITION_BLOCK;
    let left_end = copy_block(order, left_write);
    let right_end = copy_block(order, right_read);
    let write_block = |order: &mut [usize],
                       block: [usize; PARTITION_BLOCK],
                       left: &mut usize,
                       right: &mut usize| {
        for index in block {
            if is_lower(index) {
                order[*left] = index;
                *left += 1;
            } else {
                *right -= 1;
                order[*right] = index;
            }
        }
    };

    // Load from the side with less space so a full block can go to either side.
    while left_read < right_read {
        let block = if left_read - left_write <= right_write - right_read {
            let block = copy_block(order, left_read);
            left_read += PARTITION_BLOCK;
            block
        } else {
            right_read -= PARTITION_BLOCK;
            copy_block(order, right_read)
        };
        write_block(order, block, &mut left_write, &mut right_write);
    }
    write_block(order, left_end, &mut left_write, &mut right_write);
    write_block(order, right_end, &mut left_write, &mut right_write);
    left_write
}
