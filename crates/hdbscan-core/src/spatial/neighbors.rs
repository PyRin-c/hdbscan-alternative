use super::{Geometry, SpaceTree};

#[derive(Clone, Copy)]
pub(crate) struct Neighbor {
    pub(crate) distance: f64,
    pub(crate) point: usize,
}

#[derive(Clone, Copy)]
struct NodePair {
    bound: f64,
    query: usize,
    reference: usize,
}

impl SpaceTree<'_> {
    /// Finds neighbors in the depth-first traversal used for a prediction query.
    pub(crate) fn query(&self, row: &[f64], heap: &mut [Neighbor]) {
        heap.fill(Neighbor {
            point: 0,
            distance: f64::INFINITY,
        });
        self.query_node(row, 0, self.row_bound(0, row), heap);
        order_query_neighbors(heap);
        for neighbor in heap {
            neighbor.distance = self.metric.distance(neighbor.distance);
        }
    }

    fn query_node(&self, row: &[f64], node: usize, bound: f64, heap: &mut [Neighbor]) {
        if bound > heap[0].distance {
            return;
        }
        if self.is_leaf(node) {
            for &point in self.members(node) {
                push_neighbor(
                    heap,
                    Neighbor {
                        point,
                        distance: self.metric.reduced(row, self.points.row(point)),
                    },
                );
            }
            return;
        }
        let left = 2 * node + 1;
        let right = left + 1;
        let left_bound = self.row_bound(left, row);
        let right_bound = self.row_bound(right, row);
        if left_bound <= right_bound {
            self.query_node(row, left, left_bound, heap);
            self.query_node(row, right, right_bound, heap);
        } else {
            self.query_node(row, right, right_bound, heap);
            self.query_node(row, left, left_bound, heap);
        }
    }

    fn row_bound(&self, node: usize, row: &[f64]) -> f64 {
        let node = &self.nodes[node];
        self.metric.box_distance(
            row.iter()
                .zip(node.lower.iter().zip(&node.upper))
                .map(|(&value, (&lower, &upper))| (lower - value).max(value - upper).max(0.0)),
        )
    }

    /// Finds training neighbors using the same breadth-first spatial queries
    /// for every row. Equal distances retain the order from the search heap.
    pub(crate) fn neighbors(&self, queries: &Self, count: usize) -> Vec<Neighbor> {
        let mut neighbors = vec![
            Neighbor {
                point: 0,
                distance: f64::INFINITY
            };
            queries.points.len() * count
        ];
        let mut bounds = vec![f64::INFINITY; queries.nodes.len()];
        let mut pending = vec![NodePair {
            query: 0,
            reference: 0,
            bound: 0.0,
        }];
        while let Some(pair) = pop_pair(&mut pending) {
            let (query, reference) = (pair.query, pair.reference);
            if pair.bound > bounds[query] {
                continue;
            }
            if queries.is_leaf(query) && self.is_leaf(reference) {
                // A negative initial bound prunes the remaining node pairs
                // when none of the point heaps can still be improved.
                let mut largest = -1.0_f64;
                for &point in queries.members(query) {
                    let heap = &mut neighbors[point * count..(point + 1) * count];
                    if pair.bound >= heap[0].distance {
                        continue;
                    }
                    let row = queries.points.row(point);
                    for &other in self.members(reference) {
                        push_neighbor(
                            heap,
                            Neighbor {
                                point: other,
                                distance: self.metric.reduced(row, self.points.row(other)),
                            },
                        );
                    }
                    largest = largest.max(heap[0].distance);
                }
                bounds[query] = largest;
            } else {
                for (query, reference) in queries.split_pair(query, self, reference, false) {
                    let bound = self.between(reference, queries, query);
                    let bound = if self.geometry == Geometry::Ball {
                        self.metric.to_reduced(bound)
                    } else {
                        bound
                    };
                    push_pair(
                        &mut pending,
                        NodePair {
                            query,
                            reference,
                            bound,
                        },
                    );
                }
            }
        }
        for row in neighbors.chunks_exact_mut(count) {
            order_query_neighbors(row);
        }
        neighbors
    }
}

// Tied neighbors preserve the order left by median partitioning from 16
// neighbors onward and a stable sort within the remaining short runs.
fn order_query_neighbors(mut neighbors: &mut [Neighbor]) {
    while neighbors.len() >= 16 {
        let last = neighbors.len() - 1;
        let middle = neighbors.len() / 2;
        for (left, right) in [(0, last), (last, middle), (0, last)] {
            if neighbors[left].distance > neighbors[right].distance {
                neighbors.swap(left, right);
            }
        }
        let pivot = neighbors[last].distance;
        let (mut left, mut right) = (0, last);
        loop {
            left += 1;
            while neighbors[left].distance < pivot {
                left += 1;
            }
            right -= 1;
            while pivot < neighbors[right].distance {
                right -= 1;
            }
            if left >= right {
                break;
            }
            neighbors.swap(left, right);
        }
        neighbors.swap(left, last);
        let (left, tail) = neighbors.split_at_mut(left);
        let right = &mut tail[1..];
        if left.len() < right.len() {
            order_query_neighbors(left);
            neighbors = right;
        } else {
            order_query_neighbors(right);
            neighbors = left;
        }
    }
    neighbors.sort_by(|left, right| left.distance.total_cmp(&right.distance));
}

fn push_neighbor(heap: &mut [Neighbor], entry: Neighbor) {
    if entry.distance >= heap[0].distance {
        return;
    }
    let mut index = 0;
    while index < heap.len() / 2 {
        let (left, right) = (2 * index + 1, 2 * index + 2);
        let child = if right < heap.len() && heap[right].distance > heap[left].distance {
            right
        } else {
            left
        };
        if entry.distance >= heap[child].distance {
            break;
        }
        heap[index] = heap[child];
        index = child;
    }
    heap[index] = entry;
}

fn push_pair(heap: &mut Vec<NodePair>, pair: NodePair) {
    let mut index = heap.len();
    heap.push(pair);
    while index > 0 {
        let parent = (index - 1) / 2;
        if pair.bound >= heap[parent].bound {
            break;
        }
        heap[index] = heap[parent];
        index = parent;
    }
    heap[index] = pair;
}

fn pop_pair(heap: &mut Vec<NodePair>) -> Option<NodePair> {
    let first = *heap.first()?;
    let replacement = heap.pop().unwrap();
    if !heap.is_empty() {
        let mut index = 0;
        while index < heap.len() / 2 {
            let (left, right) = (2 * index + 1, 2 * index + 2);
            let child = if right < heap.len() && heap[right].bound < heap[left].bound {
                right
            } else {
                left
            };
            if replacement.bound < heap[child].bound {
                break;
            }
            heap[index] = heap[child];
            index = child;
        }
        heap[index] = replacement;
    }
    Some(first)
}
