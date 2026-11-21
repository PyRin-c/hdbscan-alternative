use crate::{FitError, LinkageMerge, SpanningEdge};

mod edge_order;
#[cfg(target_env = "msvc")]
pub(crate) use edge_order::partition_msvc_at_median;

pub(crate) fn minimum_spanning_tree(
    core: &[f64],
    alpha: f64,
    distance: impl Fn(usize, usize) -> Result<f64, FitError>,
) -> Result<Vec<SpanningEdge>, FitError> {
    prim_tree(core, alpha, distance, f64::INFINITY)
}

pub(crate) fn prim_tree(
    core: &[f64],
    alpha: f64,
    distance: impl Fn(usize, usize) -> Result<f64, FitError>,
    distance_bound: f64,
) -> Result<Vec<SpanningEdge>, FitError> {
    let mut edges = Vec::with_capacity(core.len() - 1);
    let mut visited = vec![false; core.len()];
    let mut best_distances = vec![distance_bound; core.len()];
    let mut best_sources = vec![0; core.len()];
    let mut current = 0;
    for _ in 1..core.len() {
        visited[current] = true;
        let mut next = None;
        for target in 0..core.len() {
            if visited[target] {
                continue;
            }
            let pair_distance = distance(current, target)?;
            let scaled = if pair_distance.is_infinite() {
                pair_distance
            } else {
                pair_distance / alpha
            };
            let weight = scaled.max(core[current]).max(core[target]);
            if !scaled.is_nan() && weight < best_distances[target] {
                best_distances[target] = weight;
                best_sources[target] = current;
            }
            if next.is_none_or(|previous| best_distances[target] < best_distances[previous]) {
                next = Some(target);
            }
        }
        let target = next.expect("one unvisited point remains");
        if distance_bound.is_finite() && best_distances[target] == distance_bound {
            // Once all remaining candidates exceed the bound, the vector path
            // emits the initial endpoints for each of the unfilled rows.
            edges.resize(
                core.len() - 1,
                SpanningEdge {
                    source: 0,
                    target: 0,
                    distance: distance_bound,
                },
            );
            break;
        }
        edges.push(SpanningEdge {
            source: best_sources[target],
            target,
            distance: best_distances[target],
        });
        current = target;
    }
    Ok(edges)
}

pub(crate) fn edges_in_linkage_order(
    edges: &[SpanningEdge],
) -> impl Iterator<Item = SpanningEdge> + use<'_> {
    edge_order::sort_indices(edges)
        .into_iter()
        .map(|index| edges[index])
}

pub(crate) fn root(parents: &mut [usize], mut node: usize) -> usize {
    while node != parents[node] {
        parents[node] = parents[parents[node]];
        node = parents[node];
    }
    node
}

pub(crate) fn single_linkage(
    edges: impl IntoIterator<Item = SpanningEdge>,
    n_samples: usize,
) -> Vec<LinkageMerge> {
    let mut parents: Vec<_> = (0..2 * n_samples - 1).collect();
    let mut sizes = vec![1_usize; parents.len()];
    let mut linkage = Vec::with_capacity(n_samples - 1);
    for (step, edge) in edges.into_iter().enumerate() {
        let left = root(&mut parents, edge.source);
        let right = root(&mut parents, edge.target);
        let parent = n_samples + step;
        sizes[parent] = sizes[left].saturating_add(sizes[right]);
        parents[left] = parent;
        parents[right] = parent;
        linkage.push(LinkageMerge {
            left,
            right,
            distance: edge.distance,
            size: sizes[parent],
        });
    }
    linkage
}
