use std::borrow::Cow;

use crate::ordering::select_spatial_median;

mod metric;
mod neighbors;
pub use metric::TreeDistance;
pub(super) use metric::TreeDistance as Metric;
pub(super) use neighbors::Neighbor;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Geometry {
    Kd,
    Ball,
}

#[derive(Clone, Copy)]
pub(super) struct Points<'a> {
    values: &'a [f64],
    dimensions: usize,
}

impl<'a> Points<'a> {
    pub(super) fn new(values: &'a [f64], dimensions: usize) -> Self {
        Self { values, dimensions }
    }

    pub(super) fn len(self) -> usize {
        self.values.len() / self.dimensions
    }

    pub(super) fn row(self, index: usize) -> &'a [f64] {
        let start = index * self.dimensions;
        &self.values[start..start + self.dimensions]
    }

    pub(super) fn slice(self, start: usize, end: usize) -> Self {
        Self::new(
            &self.values[start * self.dimensions..end * self.dimensions],
            self.dimensions,
        )
    }
}

pub(super) struct SpaceTree<'a> {
    pub(super) points: Points<'a>,
    pub(super) geometry: Geometry,
    pub(super) metric: Metric<'a>,
    pub(super) nodes: Cow<'a, [Node]>,
    order: Cow<'a, [usize]>,
}

pub(super) struct KdIndex {
    nodes: Vec<Node>,
    order: Vec<usize>,
}

impl KdIndex {
    pub(super) fn new(points: Points<'_>, metric: Metric<'_>, leaf_size: usize) -> Self {
        let tree = SpaceTree::new(points, metric, Geometry::Kd, leaf_size);
        Self {
            nodes: tree.nodes.into_owned(),
            order: tree.order.into_owned(),
        }
    }

    pub(super) fn tree<'a>(&'a self, points: Points<'a>, metric: Metric<'a>) -> SpaceTree<'a> {
        SpaceTree {
            points,
            geometry: Geometry::Kd,
            metric,
            nodes: Cow::Borrowed(&self.nodes),
            order: Cow::Borrowed(&self.order),
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct Node {
    start: usize,
    end: usize,
    pub(super) radius: f64,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl<'a> SpaceTree<'a> {
    pub(crate) fn traversal_order(&self) -> &[usize] {
        &self.order
    }

    pub(super) fn new(
        points: Points<'a>,
        metric: Metric<'a>,
        geometry: Geometry,
        leaf_size: usize,
    ) -> Self {
        let levels = ((points.len() - 1) / leaf_size).max(1).ilog2() + 1;
        let mut tree = Self {
            points,
            geometry,
            metric,
            nodes: Cow::Owned(vec![Node::default(); (1 << levels) - 1]),
            order: Cow::Owned((0..points.len()).collect()),
        };
        tree.build_node(0, 0, points.len());
        tree
    }

    fn build_node(&mut self, id: usize, start: usize, end: usize) {
        let mut lower = vec![f64::INFINITY; self.points.dimensions];
        let mut upper = vec![f64::NEG_INFINITY; self.points.dimensions];
        for &index in &self.order[start..end] {
            for (column, &value) in self.points.row(index).iter().enumerate() {
                lower[column] = lower[column].min(value);
                upper[column] = upper[column].max(value);
            }
        }
        let split_dimension = lower
            .iter()
            .zip(&upper)
            .enumerate()
            .fold((0, -1.0), |best, (column, (&low, &high))| {
                let spread = high - low;
                if spread > best.1 {
                    (column, spread)
                } else {
                    best
                }
            })
            .0;

        let radius = if self.geometry == Geometry::Ball {
            lower.fill(0.0);
            for &index in &self.order[start..end] {
                for (sum, &value) in lower.iter_mut().zip(self.points.row(index)) {
                    *sum += value;
                }
            }
            for sum in &mut lower {
                *sum /= (end - start) as f64;
            }
            upper.clear();
            let reduced = self.order[start..end]
                .iter()
                .map(|&index| self.metric.reduced(&lower, self.points.row(index)))
                .fold(0.0, f64::max);
            self.metric.distance(reduced)
        } else if self.metric == Metric::Chebyshev {
            // The KD-tree's radius evaluates the Minkowski formula at infinite p.
            1.0
        } else {
            let widths = lower
                .iter()
                .zip(&upper)
                .map(|(&low, &high)| (high - low) * 0.5);
            let reduced = self.metric.box_distance(widths);
            self.metric.distance(reduced)
        };
        self.nodes.to_mut()[id] = Node {
            start,
            end,
            radius,
            lower,
            upper,
        };

        if !self.is_leaf(id) {
            select_spatial_median(&mut self.order.to_mut()[start..end], |index| {
                self.points.row(index)[split_dimension]
            });
            let middle = start + (end - start) / 2;
            self.build_node(id * 2 + 1, start, middle);
            self.build_node(id * 2 + 2, middle, end);
        }
    }

    pub(super) fn is_leaf(&self, node: usize) -> bool {
        node >= self.nodes.len() / 2
    }

    pub(super) fn split_pair(
        &self,
        query: usize,
        other: &Self,
        reference: usize,
        split_query_on_equal: bool,
    ) -> [(usize, usize); 2] {
        let left = self.nodes[query].radius;
        let right = other.nodes[reference].radius;
        if other.is_leaf(reference)
            || (!self.is_leaf(query) && (left > right || split_query_on_equal && left == right))
        {
            [(2 * query + 1, reference), (2 * query + 2, reference)]
        } else {
            [(query, 2 * reference + 1), (query, 2 * reference + 2)]
        }
    }

    pub(super) fn members(&self, node: usize) -> &[usize] {
        let node = &self.nodes[node];
        &self.order[node.start..node.end]
    }

    pub(super) fn between(&self, left: usize, other: &Self, right: usize) -> f64 {
        let (left, right) = (&self.nodes[left], &other.nodes[right]);
        match self.geometry {
            Geometry::Ball => (self
                .metric
                .distance(self.metric.reduced(&left.lower, &right.lower))
                - left.radius
                - right.radius)
                .max(0.0),
            Geometry::Kd => self
                .metric
                .box_distance((0..self.points.dimensions).map(|column| {
                    (left.lower[column] - right.upper[column])
                        .max(right.lower[column] - left.upper[column])
                        .max(0.0)
                })),
        }
    }
}
