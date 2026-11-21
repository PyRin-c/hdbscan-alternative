//! Flare-sensitive branch detection within HDBSCAN clusters.

use crate::{
    ClusterSelection, ClusterSelectionThresholds, CondensedEdge, FeatureMetric, FitError,
    LinkageMerge, Prediction, SpanningEdge,
    cluster::{self, FlatClustering, SelectionDiagnostic},
    hierarchy::{edges_in_linkage_order, root, single_linkage},
    ordering::pairwise_sum,
    sparse::spanning_forest,
    spatial::{KdIndex, Metric, Neighbor, Points},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BranchDetectionMethod {
    Core,
    #[default]
    Full,
}

#[derive(Clone, Copy, Debug)]
pub struct BranchOptions {
    pub method: BranchDetectionMethod,
    pub label_sides_as_branches: bool,
    /// Preserve connected components when supplied labels span disjoint groups.
    pub cluster_labels_overridden: bool,
    pub min_cluster_size: usize,
    pub selection: ClusterSelection,
    pub thresholds: ClusterSelectionThresholds,
}

impl Default for BranchOptions {
    fn default() -> Self {
        Self {
            method: BranchDetectionMethod::Full,
            label_sides_as_branches: false,
            cluster_labels_overridden: false,
            min_cluster_size: 5,
            selection: ClusterSelection::default(),
            thresholds: ClusterSelectionThresholds::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ApproximationEdge {
    pub source: usize,
    pub target: usize,
    pub centrality: f64,
    pub reachability: f64,
}

#[derive(Debug)]
pub struct ClusterBranches {
    pub points: Vec<usize>,
    pub graph: Vec<ApproximationEdge>,
    pub linkage: Option<Vec<LinkageMerge>>,
    pub condensed: Option<Vec<CondensedEdge>>,
    pub persistences: Vec<f64>,
    pub exemplars: Option<Vec<Vec<usize>>>,
}

#[derive(Debug)]
pub struct BranchResult {
    pub labels: Vec<i64>,
    pub probabilities: Vec<f64>,
    pub branch_labels: Vec<i64>,
    pub branch_probabilities: Vec<f64>,
    pub centralities: Vec<f64>,
    pub cluster_labels: Vec<i64>,
    pub cluster_probabilities: Vec<f64>,
    pub clusters: Vec<ClusterBranches>,
    label_sides_as_branches: bool,
}

enum LocalHierarchy {
    Components(Vec<SpanningEdge>),
    Linkage(Result<Vec<LinkageMerge>, FitError>),
}

enum LocalAssignment {
    Components(Vec<SpanningEdge>),
    Branches(FlatClustering),
    InvalidLinkage(FitError),
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BranchPrediction {
    pub label: i64,
    pub probability: f64,
    pub cluster_label: i64,
    pub cluster_probability: f64,
    pub branch_label: i64,
    pub branch_probability: f64,
}

impl BranchResult {
    /// Adds branch labels and strengths to an approximate cluster prediction.
    pub fn classify(&self, prediction: Prediction) -> Result<BranchPrediction, FitError> {
        let mut row = BranchPrediction {
            label: prediction.label,
            probability: prediction.probability,
            cluster_label: prediction.label,
            cluster_probability: prediction.probability,
            branch_label: 0,
            branch_probability: 1.0,
        };
        if prediction.label >= 0 {
            let group = self
                .clusters
                .get(prediction.label as usize)
                .ok_or(FitError::InvalidHierarchy)?;
            let minimum = if self.label_sides_as_branches { 2 } else { 3 };
            if group.persistences.len() >= minimum {
                let point = usize::try_from(prediction.connecting_point)
                    .map_err(|_| FitError::InvalidHierarchy)?;
                row.label = *self.labels.get(point).ok_or(FitError::InvalidHierarchy)?;
                row.branch_label = self.branch_labels[point];
                row.branch_probability = self.branch_probabilities[point];
                row.probability = 0.5 * (row.probability + row.branch_probability);
            }
        }
        Ok(row)
    }
}

pub struct BranchDetectionData {
    values: Vec<f64>,
    n_features: usize,
    n_samples: usize,
    points: Vec<usize>,
    metric: Metric<'static>,
    index: KdIndex,
    neighbors: Vec<i64>,
    core_distances: Vec<f64>,
    neighbors_per_point: usize,
}

impl BranchDetectionData {
    /// Prepares neighbors for the labeled training points, using all finite rows.
    pub fn new(
        values: &[f64],
        dimensions: usize,
        labels: &[i64],
        metric: FeatureMetric,
        min_samples: usize,
        leaf_size: usize,
    ) -> Result<Self, FitError> {
        Self::new_with_distance(
            values,
            dimensions,
            labels,
            metric.try_into()?,
            min_samples,
            leaf_size,
        )
    }

    /// Prepares branch detection data with a distance supported by the KD-tree.
    pub fn new_with_distance(
        values: &[f64],
        dimensions: usize,
        labels: &[i64],
        metric: Metric<'static>,
        min_samples: usize,
        leaf_size: usize,
    ) -> Result<Self, FitError> {
        if dimensions == 0 || labels.is_empty() {
            return Err(FitError::InvalidShape);
        }
        if labels.len().checked_mul(dimensions) != Some(values.len()) {
            return Err(FitError::LengthMismatch);
        }
        if min_samples == 0 {
            return Err(FitError::InvalidMinSamples);
        }
        if leaf_size == 0 {
            return Err(FitError::InvalidLeafSize);
        }
        if !metric.supports_kd() {
            return Err(FitError::UnsupportedMetric);
        }
        let metric = metric.validated(dimensions)?;
        let mut packed = Vec::new();
        let mut points = Vec::new();
        for (point, row) in values.chunks_exact(dimensions).enumerate() {
            if row.iter().all(|x| x.is_finite()) {
                points.push(point);
                packed.extend_from_slice(row);
            }
        }
        if points.is_empty() {
            return Err(FitError::NonFiniteInput);
        }
        let query = Points::new(&packed, dimensions);
        let index = KdIndex::new(query, metric, leaf_size);
        let tree = index.tree(query, metric);
        let mut neighbors = vec![
            -1;
            query
                .len()
                .checked_mul(min_samples)
                .ok_or(FitError::InvalidShape)?
        ];
        let mut core_distances = vec![f64::NAN; query.len()];
        if points.iter().any(|&point| labels[point] >= 0) {
            if min_samples > query.len() {
                return Err(FitError::InvalidMinSamples);
            }
            let mut heap = vec![
                Neighbor {
                    point: 0,
                    distance: 0.0
                };
                min_samples
            ];
            for (point, _) in points
                .iter()
                .enumerate()
                .filter(|&(_, raw)| labels[*raw] >= 0)
            {
                tree.query(query.row(point), &mut heap);
                core_distances[point] = heap[min_samples - 1].distance;
                for (slot, neighbor) in neighbors[point * min_samples..(point + 1) * min_samples]
                    .iter_mut()
                    .zip(&heap)
                {
                    *slot = neighbor.point as i64;
                }
            }
        }
        Ok(Self {
            values: packed,
            n_features: dimensions,
            n_samples: labels.len(),
            points,
            metric,
            index,
            neighbors,
            core_distances,
            neighbors_per_point: min_samples,
        })
    }

    pub fn finite_rows(&self) -> &[usize] {
        &self.points
    }

    pub fn neighbors(&self) -> &[i64] {
        &self.neighbors
    }

    pub fn core_distances(&self) -> &[f64] {
        &self.core_distances
    }

    /// Detects branches from cluster labels, probabilities, and the fitted tree.
    ///
    /// The MST uses finite-row indices. Point IDs in the result use the original
    /// training row numbers, and the cluster hierarchies use local indices.
    /// With missing rows, each finite training index must be present in both
    /// inputs, and the maximum supplied label determines which groups are checked.
    pub fn detect(
        &self,
        labels: &[i64],
        probabilities: &[f64],
        mst: &[SpanningEdge],
        options: BranchOptions,
    ) -> Result<BranchResult, FitError> {
        let wrong_length = labels.len() != self.n_samples || probabilities.len() != self.n_samples;
        let supplied = labels.len().min(probabilities.len());
        if wrong_length
            && (self.points.len() == self.n_samples
                || self.points.last().is_some_and(|&row| row >= supplied))
        {
            return Err(FitError::LengthMismatch);
        }
        if options.min_cluster_size < 2 {
            return Err(FitError::InvalidMinClusterSize);
        }
        options.thresholds.validate()?;
        if mst
            .iter()
            .any(|edge| edge.source >= self.points.len() || edge.target >= self.points.len())
        {
            return Err(FitError::InvalidHierarchy);
        }
        let finite_labels: Vec<_> = self.points.iter().map(|&row| labels[row]).collect();
        let cluster_count = usize::try_from(labels.iter().copied().max().unwrap_or(-1).max(-1))
            .ok()
            .map_or(0, |label| label.saturating_add(1))
            // At least one group in this prefix is empty if there are more labels.
            .min(self.points.len() + 1);
        let mut members = vec![Vec::new(); cluster_count];
        for (point, &label) in finite_labels.iter().enumerate() {
            if let Ok(label) = usize::try_from(label)
                && let Some(group) = members.get_mut(label)
            {
                group.push(point);
            }
        }
        let mut trees = vec![Vec::new(); cluster_count];
        for edge in mst {
            let label = finite_labels[edge.source];
            if label == finite_labels[edge.target]
                && let Ok(label) = usize::try_from(label)
                && let Some(tree) = trees.get_mut(label)
            {
                tree.push(*edge);
            }
        }
        let mut output = BranchResult {
            labels: vec![-1; self.n_samples],
            probabilities: vec![0.0; self.n_samples],
            branch_labels: vec![0; self.n_samples],
            branch_probabilities: vec![0.0; self.n_samples],
            centralities: vec![0.0; self.n_samples],
            cluster_labels: vec![-1; self.n_samples],
            cluster_probabilities: vec![0.0; self.n_samples],
            clusters: Vec::with_capacity(cluster_count),
            label_sides_as_branches: options.label_sides_as_branches,
        };
        for (&point, label) in self.points.iter().zip(finite_labels) {
            output.probabilities[point] = probabilities[point];
            output.branch_probabilities[point] = 1.0;
            output.cluster_labels[point] = label;
            output.cluster_probabilities[point] = probabilities[point];
        }
        let mut local_ids = vec![usize::MAX; self.points.len()];
        let mut pending = Vec::with_capacity(cluster_count);
        for (group, tree) in members.into_iter().zip(trees) {
            if group.is_empty() {
                return Err(FitError::ZeroBranchWeight);
            }
            for (local, &point) in group.iter().enumerate() {
                local_ids[point] = local;
            }
            let centers = self.centralities(&group, probabilities)?;
            let mut graph = self.graph(&group, &local_ids, &tree, &centers, options.method)?;
            let forest = branch_forest(group.len(), &graph);
            let rows: Vec<_> = group.iter().map(|&point| self.points[point]).collect();
            for (&row, &centrality) in rows.iter().zip(&centers) {
                output.centralities[row] = centrality;
            }
            for edge in &mut graph {
                edge.source = rows[edge.source];
                edge.target = rows[edge.target];
            }
            for point in group {
                local_ids[point] = usize::MAX;
            }
            let hierarchy = if options.cluster_labels_overridden && forest.len() < rows.len() - 1 {
                LocalHierarchy::Components(forest)
            } else {
                if forest.is_empty() {
                    return Err(FitError::InsufficientTreePoints);
                }
                // Check graph construction for each cluster before using its linkage.
                LocalHierarchy::Linkage(branch_linkage(forest.len() + 1, &forest))
            };
            pending.push((
                ClusterBranches {
                    points: rows,
                    graph,
                    linkage: None,
                    condensed: None,
                    persistences: Vec::new(),
                    exemplars: None,
                },
                hierarchy,
            ));
        }

        let mut assignments = Vec::with_capacity(cluster_count);
        for (mut group, hierarchy) in pending {
            let linkage = match hierarchy {
                LocalHierarchy::Components(forest) => {
                    assignments.push((group, LocalAssignment::Components(forest)));
                    continue;
                }
                LocalHierarchy::Linkage(Ok(linkage)) => linkage,
                LocalHierarchy::Linkage(Err(error)) => {
                    assignments.push((group, LocalAssignment::InvalidLinkage(error)));
                    continue;
                }
            };
            let n_linkage_points = linkage.len() + 1;
            let mut condensed = if n_linkage_points == group.points.len() {
                cluster::condense(&linkage, n_linkage_points, options.min_cluster_size)
            } else {
                cluster::condense_branch(&linkage, n_linkage_points, options.min_cluster_size)?
            };
            if options.thresholds.persistence > 0.0 {
                cluster::simplify(
                    &mut condensed,
                    n_linkage_points,
                    options.thresholds.persistence,
                )?;
            }
            let flat = cluster::select_clusters(
                &condensed,
                n_linkage_points,
                options.selection,
                options.thresholds,
                false,
            )?;
            group.linkage = Some(linkage);
            group.condensed = Some(condensed);
            assignments.push((group, LocalAssignment::Branches(flat)));
        }

        let minimum = if options.label_sides_as_branches {
            2
        } else {
            3
        };
        let mut label_offset = 0_i64;
        for (mut group, assignment) in assignments {
            match assignment {
                LocalAssignment::InvalidLinkage(error) => return Err(error),
                LocalAssignment::Components(forest) => {
                    let count =
                        assign_components(&mut output, &group.points, &forest, label_offset);
                    group.persistences = vec![0.0; count];
                    label_offset += count as i64;
                }
                LocalAssignment::Branches(flat) => {
                    if flat.selected.len() >= minimum {
                        if flat.labels.len() != group.points.len() {
                            return Err(FitError::BranchLabelShape {
                                values: flat.labels.len(),
                                rows: group.points.len(),
                            });
                        }
                        group.exemplars = group
                            .condensed
                            .as_ref()
                            .map(|tree| exemplar_points(tree, &flat.selected, &group.points));
                    }
                    label_offset += assign_branches(
                        &mut output,
                        &group.points,
                        &flat,
                        label_offset,
                        options.label_sides_as_branches,
                    );
                    group.persistences = flat.persistence;
                }
            }
            output.clusters.push(group);
        }
        Ok(output)
    }

    fn centralities(&self, group: &[usize], probabilities: &[f64]) -> Result<Vec<f64>, FitError> {
        let data = Points::new(&self.values, self.n_features);
        let weights: Vec<_> = group
            .iter()
            .map(|&point| probabilities[self.points[point]])
            .collect();
        let total = pairwise_sum(&weights);
        if total == 0.0 {
            return Err(FitError::ZeroBranchWeight);
        }
        let mut center = vec![0.0; self.n_features];
        if self.n_features == 1 {
            let weighted: Vec<_> = group
                .iter()
                .zip(&weights)
                .map(|(&point, &weight)| data.row(point)[0] * weight)
                .collect();
            center[0] = pairwise_sum(&weighted) / total;
        } else {
            for (&point, &weight) in group.iter().zip(&weights) {
                for (sum, &value) in center.iter_mut().zip(data.row(point)) {
                    *sum += value * weight;
                }
            }
            for value in &mut center {
                *value /= total;
            }
        }
        Ok(group
            .iter()
            .map(|&point| {
                self.metric
                    .distance(self.metric.reduced(&center, data.row(point)))
                    .recip()
            })
            .collect())
    }

    fn graph(
        &self,
        group: &[usize],
        local: &[usize],
        tree: &[SpanningEdge],
        centrality: &[f64],
        method: BranchDetectionMethod,
    ) -> Result<Vec<ApproximationEdge>, FitError> {
        let data = Points::new(&self.values, self.n_features);
        let core = |a, b| max(self.core_distances[a], self.core_distances[b]);
        let edge = |a: usize, b: usize, reachability| ApproximationEdge {
            source: a.min(b),
            target: a.max(b),
            centrality: max(centrality[a], centrality[b]),
            reachability,
        };
        let mut graph = Vec::new();
        match method {
            BranchDetectionMethod::Core => {
                cluster::report(SelectionDiagnostic::BranchGraphExtrema { core: true });
                graph.extend(
                    tree.iter()
                        .map(|e| edge(local[e.source], local[e.target], e.distance)),
                );
                for (a, &point) in group.iter().enumerate() {
                    let first = point * self.neighbors_per_point;
                    for &neighbor in &self.neighbors[first..first + self.neighbors_per_point] {
                        let neighbor = if neighbor < 0 {
                            data.len() - 1
                        } else {
                            neighbor as usize
                        };
                        let b = local[neighbor];
                        if b != usize::MAX {
                            graph.push(edge(a, b, core(a, b)));
                        }
                    }
                }
                graph.sort_by(|a, b| {
                    (a.source, a.target)
                        .cmp(&(b.source, b.target))
                        .then_with(|| a.reachability.total_cmp(&b.reachability))
                });
                graph.dedup_by(|a, b| {
                    a.source == b.source && a.target == b.target && a.reachability == b.reachability
                });
            }
            BranchDetectionMethod::Full => {
                let radius = tree
                    .iter()
                    .map(|edge| edge.distance)
                    .reduce(f64::max)
                    .ok_or(FitError::EmptyBranchTree)?;
                cluster::report(SelectionDiagnostic::BranchGraphExtrema { core: false });
                let order = self.index.tree(data, self.metric);
                for (a, &left) in group.iter().enumerate() {
                    for &right in order.traversal_order() {
                        let b = local[right];
                        if b == usize::MAX || b <= a {
                            continue;
                        }
                        let distance = self
                            .metric
                            .distance(self.metric.reduced(data.row(left), data.row(right)));
                        if distance <= radius {
                            graph.push(edge(a, b, max(distance, core(a, b))));
                        }
                    }
                }
            }
        }
        Ok(graph)
    }
}

fn exemplar_points(
    condensed: &[CondensedEdge],
    selected: &[usize],
    rows: &[usize],
) -> Vec<Vec<usize>> {
    let n_points = rows.len();
    let max_node = condensed
        .iter()
        .map(|edge| edge.parent)
        .max()
        .unwrap_or(n_points);
    let mut children = vec![Vec::new(); max_node + 1 - n_points];
    for edge in condensed {
        children[edge.parent - n_points].push(edge);
    }
    selected
        .iter()
        .map(|&root| {
            let mut examples = Vec::new();
            let mut pending = vec![root];
            while let Some(node) = pending.pop() {
                let edges = &children[node - n_points];
                let before = pending.len();
                pending.extend(
                    edges
                        .iter()
                        .rev()
                        .filter(|edge| edge.child_size > 1)
                        .map(|edge| edge.child),
                );
                if pending.len() == before {
                    let peak = edges.iter().map(|edge| edge.lambda).fold(0.0, f64::max);
                    examples.extend(
                        edges
                            .iter()
                            .filter(|edge| edge.lambda == peak)
                            .map(|edge| rows[edge.child]),
                    );
                }
            }
            examples
        })
        .collect()
}

fn max(left: f64, right: f64) -> f64 {
    if left.is_nan() || left > right {
        left
    } else {
        right
    }
}

fn branch_forest(n_points: usize, graph: &[ApproximationEdge]) -> Vec<SpanningEdge> {
    let mut ordered: Vec<_> = graph
        .iter()
        .filter(|edge| edge.source != edge.target)
        .collect();
    ordered.sort_by_key(|edge| (edge.source, edge.target));
    let mut entries = Vec::<SpanningEdge>::new();
    for edge in ordered {
        if let Some(last) = entries.last_mut()
            && (last.source, last.target) == (edge.source, edge.target)
        {
            last.distance += edge.centrality;
        } else {
            entries.push(SpanningEdge {
                source: edge.source,
                target: edge.target,
                distance: edge.centrality,
            });
        }
    }
    entries.retain(|edge| edge.distance != 0.0);
    spanning_forest(n_points, entries)
}

fn branch_linkage(n_points: usize, forest: &[SpanningEdge]) -> Result<Vec<LinkageMerge>, FitError> {
    if n_points < 2
        || forest
            .iter()
            .any(|edge| edge.source >= 2 * n_points - 1 || edge.target >= 2 * n_points - 1)
    {
        return Err(FitError::InvalidHierarchy);
    }
    if forest
        .iter()
        .all(|edge| edge.source < n_points && edge.target < n_points)
    {
        return Ok(single_linkage(edges_in_linkage_order(forest), n_points));
    }
    // An incomplete forest starts its merge IDs at edges.len() + 1, so some
    // raw endpoints coincide with internal nodes whose size starts at zero.
    let mut parents: Vec<_> = (0..2 * n_points - 1).collect();
    let mut sizes = vec![0_usize; parents.len()];
    sizes[..n_points].fill(1);
    let final_root = parents.len() - 1;
    let mut tree = Vec::with_capacity(forest.len());
    for (row, edge) in edges_in_linkage_order(forest).enumerate() {
        let parent = n_points + row;
        let left = incomplete_forest_root(&mut parents, edge.source)?;
        let right = if edge.target == final_root {
            // The last merge ID aliases the other endpoint in the reference.
            left
        } else {
            incomplete_forest_root(&mut parents, edge.target)?
        };
        let size = sizes[left].saturating_add(sizes[right]);
        parents[left] = parent;
        parents[right] = parent;
        sizes[parent] = size;
        tree.push(LinkageMerge {
            left,
            right,
            distance: edge.distance,
            size,
        });
    }
    Ok(tree)
}

fn incomplete_forest_root(parents: &mut [usize], node: usize) -> Result<usize, FitError> {
    let mut point = node;
    for _ in 0..parents.len() {
        if point == parents[point] {
            return Ok(root(parents, node));
        }
        point = parents[point];
    }
    Err(FitError::InvalidHierarchy)
}

fn assign_components(
    output: &mut BranchResult,
    rows: &[usize],
    forest: &[SpanningEdge],
    offset: i64,
) -> usize {
    let mut parents: Vec<_> = (0..rows.len()).collect();
    for edge in forest {
        let source = root(&mut parents, edge.source);
        let target = root(&mut parents, edge.target);
        parents[target] = source;
    }
    let mut labels = vec![-1; rows.len()];
    let mut next = 0;
    for (point, &row) in rows.iter().enumerate() {
        let label = &mut labels[root(&mut parents, point)];
        if *label == -1 {
            *label = next;
            next += 1;
        }
        output.branch_labels[row] = *label;
        output.labels[row] = offset + *label;
        output.probabilities[row] = 0.5 * (output.probabilities[row] + 1.0);
    }
    next as usize
}

fn assign_branches(
    output: &mut BranchResult,
    rows: &[usize],
    flat: &FlatClustering,
    label_offset: i64,
    label_sides: bool,
) -> i64 {
    let minimum = if label_sides { 2 } else { 3 };
    if flat.persistence.len() < minimum {
        for &row in rows {
            output.labels[row] = label_offset;
        }
        return 1;
    }
    let has_center = i64::from(flat.labels.contains(&-1));
    for (&row, (&branch, &strength)) in rows.iter().zip(flat.labels.iter().zip(&flat.probabilities))
    {
        output.labels[row] = label_offset + branch + has_center;
        output.branch_labels[row] = branch;
        output.branch_probabilities[row] = strength;
        output.probabilities[row] = (output.probabilities[row] + strength) * 0.5;
    }
    flat.persistence.len() as i64 + has_center
}
