use crate::{CondensedEdge, FitError, LinkageMerge, hierarchy::root};

/// Computes point-wise GLOSH scores from a fitted condensed tree.
///
/// Returned scores follow the original sample order. A finite departure density
/// and an infinite maximum density produce NaN, as in the reference Wheel.
pub fn outlier_scores(tree: &[CondensedEdge]) -> Result<Vec<f64>, FitError> {
    let Some(first) = tree.first() else {
        return Ok(Vec::new());
    };
    let n_samples = first.parent;
    if !(1..=tree.len()).contains(&n_samples) {
        return Err(FitError::InvalidHierarchy);
    }
    let mut max_lambda = vec![0.0_f64; tree.len() - n_samples + 1];
    for edge in tree {
        if !(n_samples..=tree.len()).contains(&edge.parent)
            || edge.child > tree.len()
            || (edge.child >= n_samples && edge.child <= edge.parent)
        {
            return Err(FitError::InvalidHierarchy);
        }
        let parent = edge.parent - n_samples;
        max_lambda[parent] = max_lambda[parent].max(edge.lambda);
    }
    // Condensed cluster edges appear before the edges of their descendants.
    for edge in tree.iter().rev().filter(|edge| edge.child >= n_samples) {
        let (parent, child) = (edge.parent - n_samples, edge.child - n_samples);
        max_lambda[parent] = max_lambda[parent].max(max_lambda[child]);
    }

    let mut scores = vec![0.0; n_samples];
    for edge in tree.iter().filter(|edge| edge.child < n_samples) {
        let maximum = max_lambda[edge.parent - n_samples];
        if maximum != 0.0 && edge.lambda.is_finite() {
            scores[edge.child] = if maximum.is_infinite() {
                // The reference's infinity divided by infinity is a negative quiet NaN.
                f64::from_bits(0xfff8_0000_0000_0000)
            } else {
                (maximum - edge.lambda) / maximum
            };
        }
    }
    Ok(scores)
}

/// Cuts a fitted single-linkage tree at a mutual-reachability distance.
///
/// Merges strictly below `cut_distance` are included. Components smaller than
/// `min_cluster_size` are noise (`-1`). Values at or below one keep singletons.
pub fn dbscan_clustering(
    tree: &[LinkageMerge],
    cut_distance: f64,
    min_cluster_size: i64,
) -> Result<Vec<i64>, FitError> {
    let n_samples = tree.len() + 1;
    let n_nodes = n_samples + tree.len();
    let mut parents: Vec<_> = (0..n_nodes).collect();
    let mut ranks = vec![0; n_nodes];
    for (step, merge) in tree.iter().enumerate() {
        if merge.distance < cut_distance {
            let node = n_samples + step;
            if merge.left >= node || merge.right >= node {
                return Err(FitError::InvalidHierarchy);
            }
            join(&mut parents, &mut ranks, merge.left, node);
            join(&mut parents, &mut ranks, merge.right, node);
        }
    }

    let mut counts = vec![0_usize; n_nodes];
    let representatives: Vec<_> = (0..n_samples)
        .map(|sample| {
            let representative = root(&mut parents, sample);
            counts[representative] += 1;
            representative
        })
        .collect();
    let minimum = usize::try_from(min_cluster_size.max(1)).unwrap_or(usize::MAX);
    let mut labels = vec![-1; n_nodes];
    let mut next_label = 0;
    for (label, count) in labels.iter_mut().zip(counts) {
        if count >= minimum {
            *label = next_label;
            next_label += 1;
        }
    }
    Ok(representatives.into_iter().map(|id| labels[id]).collect())
}

fn join(parents: &mut [usize], ranks: &mut [u8], left: usize, right: usize) {
    let mut left = root(parents, left);
    let mut right = root(parents, right);
    if left == right {
        return;
    }
    if ranks[left] < ranks[right] {
        std::mem::swap(&mut left, &mut right);
    }
    // Equal ranks keep the first representative. The public labels are ordered
    // by representative ID, so swapping this tie rule changes the labels.
    parents[right] = left;
    ranks[left] += u8::from(ranks[left] == ranks[right]);
}
