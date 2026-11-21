use crate::hierarchy::root;
use crate::spatial::{Geometry, Metric, Neighbor, Points, SpaceTree};
use crate::{
    FeatureMetric, FitConfiguration, FitError, FitResult, SpanningEdge, fit_vector_with_missing,
};

/// Spatial index used for mutual reachability search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoruvkaTree {
    #[default]
    KdTree,
    BallTree,
}

#[derive(Clone, Copy, Debug)]
pub struct BoruvkaOptions {
    pub tree: BoruvkaTree,
    pub leaf_size: usize,
    pub approximate: bool,
    pub core_dist_n_jobs: i64,
}

impl Default for BoruvkaOptions {
    fn default() -> Self {
        Self {
            tree: BoruvkaTree::KdTree,
            leaf_size: 40,
            approximate: true,
            core_dist_n_jobs: 4,
        }
    }
}

/// Fits dense Euclidean or Manhattan features using a spatial Borůvka search.
pub fn fit_dense_boruvka(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    metric: FeatureMetric,
    configuration: FitConfiguration,
    options: BoruvkaOptions,
) -> Result<FitResult, FitError> {
    fit_dense_tree_boruvka(
        values,
        n_samples,
        n_features,
        metric.try_into()?,
        configuration,
        options,
    )
}

/// Fits dense features using a KD-tree or Ball-tree with the supplied distance.
pub fn fit_dense_tree_boruvka(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    metric: Metric<'_>,
    configuration: FitConfiguration,
    options: BoruvkaOptions,
) -> Result<FitResult, FitError> {
    let metric = metric.validated(n_features)?;
    if options.leaf_size == 0 {
        return Err(FitError::InvalidLeafSize);
    }
    fit_vector_with_missing(
        values,
        n_samples,
        n_features,
        configuration,
        |values, _, min_samples| {
            boruvka_tree(
                Points::new(values, n_features),
                min_samples,
                metric,
                options,
                1.0,
                false,
            )
        },
    )
}

pub(super) fn boruvka_tree(
    data: Points<'_>,
    min_samples: usize,
    metric: Metric<'_>,
    options: BoruvkaOptions,
    alpha: f64,
    robust_linkage: bool,
) -> Result<Vec<SpanningEdge>, FitError> {
    let geometry = match options.tree {
        BoruvkaTree::KdTree => Geometry::Kd,
        BoruvkaTree::BallTree => Geometry::Ball,
    };
    if geometry == Geometry::Kd && !metric.supports_kd() {
        return Err(FitError::UnsupportedMetric);
    }
    let leaf_size = if robust_linkage {
        options.leaf_size
    } else {
        options.leaf_size.max(3)
    };
    let count = min_samples + 1;
    let outer = SpaceTree::new(data, metric, geometry, leaf_size);
    let neighbors = training_neighbors(&outer, leaf_size, count, options.core_dist_n_jobs)?;
    let reduced = geometry == Geometry::Kd;
    let core: Vec<_> = neighbors
        .chunks_exact(count)
        .map(|row| {
            let distance = metric.distance(row[count - 1].distance);
            if reduced {
                metric.to_reduced(distance)
            } else {
                distance
            }
        })
        .collect();
    let mut forest = Forest::new(data.len(), metric, reduced);
    for (point, row) in neighbors.chunks_exact(count).enumerate() {
        if let Some(neighbor) = row
            .iter()
            .find(|neighbor| neighbor.point != point && core[neighbor.point] <= core[point])
        {
            forest.add(SpanningEdge {
                source: point,
                target: neighbor.point,
                distance: core[point],
            });
        }
    }

    let search_leaf_size = if robust_linkage {
        leaf_size
    } else {
        leaf_size / 3
    };
    let tree = SpaceTree::new(data, metric, geometry, search_leaf_size);
    let mut bounds = vec![f64::INFINITY; tree.nodes.len()];
    while forest.edges.len() < data.len() - 1 {
        let fresh = bounds.iter().all(|bound| *bound == f64::INFINITY);
        let membership = (0..data.len())
            .map(|i| root(&mut forest.parents, i))
            .collect::<Vec<_>>();
        let mut search = Search {
            tree: &tree,
            core: &core,
            membership: &membership,
            bounds: &mut bounds,
            candidates: vec![None; data.len()],
            alpha,
        };
        search.visit(0, 0, 0.0);
        let before = forest.edges.len();
        for edge in search.candidates.into_iter().flatten() {
            forest.add(edge);
        }
        if forest.edges.len() == before && fresh {
            return Err(FitError::DisconnectedDistances);
        }
        if !options.approximate || forest.edges.len() == before {
            bounds.fill(f64::INFINITY);
        }
    }
    Ok(forest.edges)
}

fn training_neighbors(
    tree: &SpaceTree<'_>,
    leaf_size: usize,
    count: usize,
    jobs: i64,
) -> Result<Vec<Neighbor>, FitError> {
    let n_samples = tree.points.len();
    if n_samples <= 16_384 || jobs == 1 {
        return Ok(tree.neighbors(tree, count));
    }
    let jobs = if jobs < 1 {
        let cpus = std::thread::available_parallelism().map_or(1, |count| count.get());
        jobs.saturating_add(cpus as i64 + 1).max(1)
    } else {
        jobs
    };
    let jobs = usize::try_from(jobs).unwrap_or(usize::MAX).min(n_samples);
    if jobs == 1 {
        return Ok(tree.neighbors(tree, count));
    }
    let chunk = n_samples / jobs;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(jobs);
        for index in 0..jobs {
            let start = index * chunk;
            let end = if index + 1 == jobs {
                n_samples
            } else {
                start + chunk
            };
            handles.push(
                std::thread::Builder::new()
                    .spawn_scoped(scope, move || {
                        let queries = SpaceTree::new(
                            tree.points.slice(start, end),
                            tree.metric,
                            tree.geometry,
                            leaf_size,
                        );
                        tree.neighbors(&queries, count)
                    })
                    .map_err(|_| FitError::NeighborWorkerUnavailable)?,
            );
        }
        let mut neighbors = Vec::with_capacity(n_samples * count);
        for thread in handles {
            neighbors.extend(
                thread
                    .join()
                    .map_err(|_| FitError::NeighborWorkerUnavailable)?,
            );
        }
        Ok(neighbors)
    })
}

struct Forest<'a> {
    parents: Vec<usize>,
    ranks: Vec<u8>,
    edges: Vec<SpanningEdge>,
    metric: Metric<'a>,
    reduced: bool,
}

impl<'a> Forest<'a> {
    fn new(n: usize, metric: Metric<'a>, reduced: bool) -> Self {
        Self {
            parents: (0..n).collect(),
            ranks: vec![0; n],
            edges: Vec::with_capacity(n - 1),
            metric,
            reduced,
        }
    }

    fn add(&mut self, mut edge: SpanningEdge) {
        let mut left = root(&mut self.parents, edge.source);
        let mut right = root(&mut self.parents, edge.target);
        if left == right {
            return;
        }
        if self.ranks[left] < self.ranks[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parents[right] = left;
        self.ranks[left] += u8::from(self.ranks[left] == self.ranks[right]);
        if self.reduced {
            edge.distance = self.metric.distance(edge.distance);
        }
        self.edges.push(edge);
    }
}

struct Search<'a, 'b> {
    tree: &'a SpaceTree<'b>,
    core: &'a [f64],
    membership: &'a [usize],
    bounds: &'a mut [f64],
    candidates: Vec<Option<SpanningEdge>>,
    alpha: f64,
}

impl Search<'_, '_> {
    fn visit(&mut self, query: usize, reference: usize, lower: f64) {
        if lower >= self.bounds[query] {
            return;
        }
        if self.tree.is_leaf(query) && self.tree.is_leaf(reference) {
            self.leaves(query, reference);
        } else {
            let [left, right] = self.tree.split_pair(query, self.tree, reference, true);
            let low = self.tree.between(left.0, self.tree, left.1);
            let high = self.tree.between(right.0, self.tree, right.1);
            if low < high {
                self.visit(left.0, left.1, low);
                self.visit(right.0, right.1, high);
            } else {
                self.visit(right.0, right.1, high);
                self.visit(left.0, left.1, low);
            }
        }
    }

    fn leaves(&mut self, query: usize, reference: usize) {
        let points = self.tree.members(query);
        let other_points = self.tree.members(reference);
        let first_component = self.membership[points[0]];
        if points
            .iter()
            .chain(other_points)
            .all(|&index| self.membership[index] == first_component)
        {
            return;
        }
        let mut upper = 0.0_f64;
        let mut lower_candidate = f64::INFINITY;
        for &point in points {
            let component = self.membership[point];
            let current = &mut self.candidates[component];
            let mut bound = current.map_or(f64::INFINITY, |edge| edge.distance);
            if self.core[point] > bound {
                continue;
            }
            for &other in other_points {
                if self.membership[other] == component || self.core[other] >= bound {
                    continue;
                }
                let distance = self
                    .tree
                    .metric
                    .reduced(self.tree.points.row(point), self.tree.points.row(other));
                let distance = if self.tree.geometry == Geometry::Kd {
                    distance / self.alpha
                } else {
                    // The Ball-tree path applies alpha to the distance before
                    // undoing it, so both operations affect rounding.
                    self.tree.metric.distance(distance) * self.alpha / self.alpha
                };
                let weight = self.core[point].max(self.core[other]).max(distance);
                if weight < bound {
                    bound = weight;
                    *current = Some(SpanningEdge {
                        source: point,
                        target: other,
                        distance: weight,
                    });
                }
            }
            upper = upper.max(bound);
            lower_candidate = lower_candidate.min(bound);
        }
        let radius = self.tree.nodes[query].radius;
        let radius = if self.tree.geometry == Geometry::Kd {
            self.tree.metric.to_reduced(radius)
        } else {
            radius
        };
        self.update_bounds(query, upper.min(lower_candidate + 2.0 * radius));
    }

    fn update_bounds(&mut self, mut node: usize, mut upper: f64) {
        while upper < self.bounds[node] {
            self.bounds[node] = upper;
            if node == 0 {
                break;
            }
            node = (node - 1) / 2;
            let left = 2 * node + 1;
            let right = left + 1;
            upper = self.bounds[left].max(self.bounds[right]);
            if self.tree.geometry == Geometry::Ball {
                // The reference applies a radius correction to each child
                // before using the smaller threshold, including in exact mode.
                let radius = self.tree.nodes[node].radius;
                let adjust = |child: usize| 2.0 * (radius - self.tree.nodes[child].radius);
                let translated =
                    (self.bounds[left] + adjust(left)).min(self.bounds[right] + adjust(right));
                if translated > 0.0 {
                    upper = upper.min(translated);
                }
            }
        }
    }
}
