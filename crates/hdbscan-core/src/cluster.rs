use std::collections::VecDeque;

use crate::{
    ClusterSelection, ClusterSelectionMethod, ClusterSelectionThresholds, CondensedEdge, FitError,
    LinkageMerge,
};

mod diagnostics;
mod simplify;

pub(crate) use diagnostics::report;
pub use diagnostics::{SelectionDiagnostic, capture_selection_diagnostics};
pub(crate) use simplify::simplify;

pub(crate) fn condense(
    linkage: &[LinkageMerge],
    n_samples: usize,
    min_cluster_size: usize,
) -> Vec<CondensedEdge> {
    condense_with_capacity(linkage, n_samples, min_cluster_size, n_samples)
}

/// A disconnected branch forest can reuse a cluster ID as an endpoint.
/// Check the part reachable from the root and count repeated descendants.
pub(crate) fn condense_branch(
    linkage: &[LinkageMerge],
    n_samples: usize,
    minimum: usize,
) -> Result<Vec<CondensedEdge>, FitError> {
    let capacity = 2 * n_samples;
    let root = n_samples + linkage.len() - 1;
    let mut counts = vec![None; root + 1];
    counts[..n_samples].fill(Some((1_u128, 0_u128)));
    let mut visiting = vec![false; linkage.len()];
    let mut pending = vec![(root, false)];
    while let Some((node, expanded)) = pending.pop() {
        if counts[node].is_some() {
            continue;
        }
        let index = node - n_samples;
        let merge = &linkage[index];
        let children = [merge.left, merge.right];
        if !expanded {
            if visiting[index] || children.iter().any(|&child| child > root) {
                return Err(FitError::InvalidHierarchy);
            }
            visiting[index] = true;
            pending.push((node, true));
            pending.extend(children.map(|child| (child, false)));
            continue;
        }
        let large = children.map(|child| {
            child
                .checked_sub(n_samples)
                .map_or(1, |row| linkage[row].size)
                >= minimum
        });
        let mut leaves = 0_u128;
        let mut required = if large[0] && large[1] { 2_u128 } else { 0 };
        for (child, large) in children.into_iter().zip(large) {
            let (child_leaves, child_required) = counts[child].ok_or(FitError::InvalidHierarchy)?;
            leaves = leaves.saturating_add(child_leaves);
            required = required.saturating_add(if large { child_required } else { child_leaves });
        }
        counts[node] = Some((leaves, required));
    }
    let (_, required) = counts[root].ok_or(FitError::InvalidHierarchy)?;
    if required > capacity as u128 {
        return Err(FitError::CondensedTreeCapacity { capacity, required });
    }
    Ok(condense_with_capacity(
        linkage,
        n_samples,
        minimum,
        required as usize,
    ))
}

/// The vector Prim path can export repeated descendants when its distance
/// bound is reached. Each visit uses the last cluster ID assigned to the node.
pub(crate) fn condense_repeated(
    linkage: &[LinkageMerge],
    n_samples: usize,
    minimum: usize,
) -> Result<Vec<CondensedEdge>, FitError> {
    let capacity = 2 * n_samples;
    let first = linkage.partition_point(|edge| edge.distance < f64::MAX);
    let mut visits = 1_u128;
    let mut required = 0_u128;
    let mut reached_subtree = true;
    for edge in linkage[first..].iter().rev() {
        if edge.left != edge.right {
            return Err(FitError::InvalidHierarchy);
        }
        if edge.size / 2 < minimum {
            required = required.saturating_add(visits.saturating_mul(edge.size as u128));
            reached_subtree = false;
            break;
        }
        visits = visits.saturating_mul(2);
        required = required.saturating_add(visits);
    }
    if reached_subtree {
        let subtree = condense(&linkage[..first], n_samples, minimum);
        required = required.saturating_add(visits.saturating_mul(subtree.len() as u128));
    }
    if required > capacity as u128 {
        return Err(FitError::CondensedTreeCapacity { capacity, required });
    }
    Ok(condense_with_capacity(
        linkage,
        n_samples,
        minimum,
        required as usize,
    ))
}

fn condense_with_capacity(
    linkage: &[LinkageMerge],
    n_samples: usize,
    minimum: usize,
    capacity: usize,
) -> Vec<CondensedEdge> {
    let root = n_samples + linkage.len() - 1;
    let mut relabel = vec![n_samples; root + 1];
    let mut pending = VecDeque::from([root]);
    let mut descendants = VecDeque::new();
    let mut result = Vec::with_capacity(capacity);
    let mut next_cluster = n_samples + 1;
    let size = |node: usize| {
        node.checked_sub(n_samples)
            .map_or(1, |index| linkage[index].size)
    };
    while let Some(node) = pending.pop_front() {
        let merge = linkage[node - n_samples];
        let lambda = if merge.distance > 0.0 {
            merge.distance.recip()
        } else {
            f64::INFINITY
        };
        let split = size(merge.left) >= minimum && size(merge.right) >= minimum;
        for child in [merge.left, merge.right] {
            let child_size = size(child);
            if split {
                result.push(CondensedEdge {
                    parent: relabel[node],
                    child: next_cluster,
                    lambda,
                    child_size,
                });
                relabel[child] = next_cluster;
                next_cluster += 1;
                pending.push_back(child);
            } else if child_size >= minimum {
                relabel[child] = relabel[node];
                pending.push_back(child);
            } else {
                descendants.push_back(child);
                while let Some(descendant) = descendants.pop_front() {
                    if descendant < n_samples {
                        result.push(CondensedEdge {
                            parent: relabel[node],
                            child: descendant,
                            lambda,
                            child_size: 1,
                        });
                    } else {
                        let branch = linkage[descendant - n_samples];
                        descendants.extend([branch.left, branch.right]);
                    }
                }
            }
        }
    }
    result
}

#[derive(Clone, Default)]
struct Cluster {
    parent: usize,
    children: Vec<usize>,
    size: usize,
    birth: f64,
    death: f64,
    stability: f64,
}

pub(crate) struct FlatClustering {
    pub(crate) labels: Vec<i64>,
    pub(crate) probabilities: Vec<f64>,
    pub(crate) persistence: Vec<f64>,
    pub(crate) selected: Vec<usize>,
}

pub(crate) fn select_clusters(
    tree: &[CondensedEdge],
    n_samples: usize,
    selection: ClusterSelection,
    thresholds: ClusterSelectionThresholds,
    strict_membership: bool,
) -> Result<FlatClustering, FitError> {
    let cluster_count = tree
        .iter()
        .map(|edge| edge.parent - n_samples)
        .max()
        .unwrap_or(0)
        + 1;
    let mut clusters = vec![Cluster::default(); cluster_count];
    let mut max_lambda = 0.0_f64;
    let mut root_size = 0;
    let mut root_epsilon = 0.0_f64;
    let mut root_reciprocal_inputs = [1.0; 3];
    for edge in tree {
        let parent = edge.parent - n_samples;
        let cluster = &mut clusters[parent];
        cluster.death = cluster.death.max(edge.lambda);
        cluster.stability += (edge.lambda - cluster.birth) * edge.child_size as f64;
        max_lambda = max_lambda.max(edge.lambda);
        let epsilon = edge.lambda.recip();
        root_epsilon = root_epsilon.max(epsilon);
        if selection.allow_single_cluster && !epsilon.is_normal() {
            if edge.lambda == 0.0 {
                root_reciprocal_inputs[0] = edge.lambda;
            } else if epsilon.is_infinite() {
                root_reciprocal_inputs[1] = edge.lambda;
            } else if epsilon.is_subnormal() && edge.lambda.to_bits() & 0x000f_ffff_ffff_ffff != 0 {
                // A reciprocal of a power of two can be exactly subnormal.
                root_reciprocal_inputs[2] = edge.lambda;
            }
        }
        if edge.child >= n_samples {
            if epsilon.is_infinite() || epsilon.is_subnormal() {
                report(SelectionDiagnostic::BranchLambdaReciprocal(edge.lambda));
            }
            let child_node = edge.child - n_samples;
            cluster.children.push(child_node);
            if parent == 0 {
                root_size += edge.child_size;
            }
            let child = &mut clusters[child_node];
            child.parent = parent;
            child.birth = edge.lambda;
            child.size = edge.child_size;
        }
    }
    if root_reciprocal_inputs != [1.0; 3] {
        report(SelectionDiagnostic::RootLambdaReciprocal(
            root_reciprocal_inputs,
        ));
    }
    clusters[0].size = root_size;

    let mut candidates = vec![false; cluster_count];
    match selection.method {
        ClusterSelectionMethod::Eom => {
            let first_candidate = usize::from(!selection.allow_single_cluster);
            let max_cluster_size = if thresholds.max_cluster_size <= 0.0 {
                n_samples as f64
            } else {
                thresholds.max_cluster_size
            };
            // Compare each cluster to the optimal selection within its descendants.
            for node in (first_candidate..cluster_count).rev() {
                let cluster = &clusters[node];
                let epsilon = if node == 0 {
                    root_epsilon
                } else {
                    cluster.birth.recip()
                };
                let too_large = cluster.size as f64 > max_cluster_size;
                // The reference rounds the running sum after each child, in tree order.
                let children_stability = cluster.children.iter().fold(0.0, |sum, &child| {
                    f64::from((sum + clusters[child].stability) as f32)
                });
                if children_stability > cluster.stability
                    || too_large
                    || epsilon > thresholds.epsilon_max
                {
                    clusters[node].stability = children_stability;
                } else {
                    candidates[node] = true;
                }
            }
            let mut has_selected_ancestor = vec![false; cluster_count];
            has_selected_ancestor[0] = candidates[0];
            for node in 1..cluster_count {
                let covered = has_selected_ancestor[clusters[node].parent];
                candidates[node] &= !covered;
                has_selected_ancestor[node] = covered || candidates[node];
            }
        }
        ClusterSelectionMethod::Leaf => {
            for node in 1..cluster_count {
                candidates[node] = clusters[node].children.is_empty();
            }
        }
    }

    if thresholds.epsilon != 0.0 && !candidates[0] {
        candidates = merge_close_clusters(
            &clusters,
            &candidates,
            selection.allow_single_cluster,
            thresholds.epsilon,
        )?;
    }

    let mut cluster_labels = vec![-1; cluster_count];
    let mut selected = Vec::new();
    for node in 0..cluster_count {
        let parent_label = cluster_labels[clusters[node].parent];
        if candidates[node] {
            cluster_labels[node] = selected.len() as i64;
            selected.push(node);
        } else {
            cluster_labels[node] = parent_label;
        }
    }

    let mut labels = vec![-1; n_samples];
    let mut probabilities = vec![0.0; n_samples];
    let selected_root = selected.first() == Some(&0);
    let root_cutoff = if thresholds.epsilon == 0.0 {
        clusters[0].death
    } else {
        thresholds.epsilon.recip()
    };
    for edge in tree.iter().filter(|edge| edge.child < n_samples) {
        let label = cluster_labels[edge.parent - n_samples];
        let Ok(index) = usize::try_from(label) else {
            labels[edge.child] = label;
            continue;
        };
        let cluster = &clusters[selected[index]];
        let member = if label == 0 && selected_root {
            selected.len() == 1 && edge.lambda >= root_cutoff
        } else {
            !strict_membership || edge.lambda > cluster.birth
        };
        if !member {
            continue;
        }
        labels[edge.child] = label;
        probabilities[edge.child] = if cluster.death == 0.0 || !edge.lambda.is_finite() {
            1.0
        } else {
            edge.lambda.min(cluster.death) / cluster.death
        };
    }

    let mut counts = vec![0; selected.len()];
    for label in labels
        .iter()
        .filter_map(|label| usize::try_from(*label).ok())
    {
        counts[label] += 1;
    }
    let persistence = selected
        .iter()
        .copied()
        .zip(counts)
        .map(|(node, size)| {
            if size == 0 || max_lambda == 0.0 || !max_lambda.is_finite() {
                1.0
            } else {
                clusters[node].stability / (size as f64 * max_lambda)
            }
        })
        .collect();
    Ok(FlatClustering {
        labels,
        probabilities,
        persistence,
        selected: selected.into_iter().map(|node| n_samples + node).collect(),
    })
}

fn merge_close_clusters(
    clusters: &[Cluster],
    candidates: &[bool],
    allow_single_cluster: bool,
    epsilon: f64,
) -> Result<Vec<bool>, FitError> {
    let mut merged = vec![false; candidates.len()];
    // Find the highest viable ancestor once per branch. Equality is retained
    // for a selected cluster but does not stop the climb from a descendant.
    let mut target: Vec<_> = (0..candidates.len()).map(Ok).collect();
    for node in 1..candidates.len() {
        let parent = clusters[node].parent;
        if parent == 0 {
            if allow_single_cluster {
                target[node] = Ok(parent);
            }
        } else if clusters[parent].birth == 0.0 {
            target[node] = Err(FitError::ZeroLambdaEpsilon);
        } else if clusters[parent].birth.recip() > epsilon {
            target[node] = Ok(parent);
        } else {
            target[node] = target[parent];
        }
        if candidates[node] {
            if clusters[node].birth == 0.0 {
                return Err(FitError::ZeroLambdaEpsilon);
            }
            let replacement = if clusters[node].birth.recip() < epsilon {
                target[node]?
            } else {
                node
            };
            merged[replacement] = true;
        }
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_branch_hierarchy_rejects_a_reachable_cycle() {
        let linkage = [
            LinkageMerge {
                left: 1,
                right: 3,
                distance: 1.0,
                size: 1,
            },
            LinkageMerge {
                left: 3,
                right: 3,
                distance: 2.0,
                size: 2,
            },
        ];
        assert!(matches!(
            condense_branch(&linkage, 3, 2),
            Err(FitError::InvalidHierarchy)
        ));
    }
}
