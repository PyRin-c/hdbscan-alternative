use crate::{CondensedEdge, FitError};

use super::diagnostics::{SelectionDiagnostic, report};

struct Branch {
    representative: usize,
    highest_lambda: f32,
    splits: bool,
    lambda_override: Option<f32>,
    new_id: usize,
}

pub(crate) fn simplify(
    tree: &mut Vec<CondensedEdge>,
    n_samples: usize,
    threshold: f64,
) -> Result<(), FitError> {
    let splits: Vec<_> = tree.iter().filter(|edge| edge.child >= n_samples).collect();
    let last = splits.last().ok_or(FitError::EmptyPersistenceHierarchy)?;
    let mut branches: Vec<_> = (0..=last.child - n_samples)
        .map(|index| Branch {
            representative: index,
            highest_lambda: 0.0,
            splits: false,
            lambda_override: None,
            new_id: 0,
        })
        .collect();
    let mut cast_inputs = [1.0, f64::INFINITY];
    for edge in tree.iter() {
        let lambda = edge.lambda as f32;
        if lambda.is_infinite() && edge.lambda.is_finite() {
            cast_inputs[0] = edge.lambda;
        }
        if edge.lambda < f64::from(f32::MIN_POSITIVE) && f64::from(lambda) != edge.lambda {
            cast_inputs[1] = cast_inputs[1].min(edge.lambda);
        }
        let branch = &mut branches[edge.parent - n_samples];
        branch.highest_lambda = branch.highest_lambda.max(lambda);
        branch.splits |= edge.child >= n_samples;
    }
    if cast_inputs != [1.0, f64::INFINITY] {
        report(SelectionDiagnostic::LambdaCast(cast_inputs));
    }
    let rounded_threshold = threshold as f32;
    let threshold_overflow = rounded_threshold.is_infinite() && threshold.is_finite();

    // Consider each sibling pair from the bottom up so a branch can become a
    // leaf before its own persistence is tested.
    for pair in splits.chunks_exact(2).rev() {
        let parent = pair[0].parent - n_samples;
        let left = pair[0].child - n_samples;
        let right = pair[1].child - n_samples;
        let split_lambda = pair[0].lambda as f32;
        let (weak, strong) = if branches[left].highest_lambda <= branches[right].highest_lambda {
            (left, right)
        } else {
            (right, left)
        };
        let max_lambda = branches[strong].highest_lambda;
        branches[parent].highest_lambda = branches[parent].highest_lambda.max(max_lambda);
        let lifetime = branches[weak].highest_lambda - split_lambda;
        if lifetime.is_nan() {
            report(SelectionDiagnostic::InvalidLifetime);
        }
        if threshold_overflow {
            report(SelectionDiagnostic::ThresholdCastOverflow);
        }
        if lifetime >= rounded_threshold {
            continue;
        }
        branches[parent].splits = branches[strong].splits;
        if branches[strong].splits {
            // Preserve the deeper split and detach the weak side at this split's density.
            branches[weak].lambda_override = Some(split_lambda);
        }
        branches[left].representative = parent;
        branches[right].representative = parent;
    }

    let mut next_id = n_samples;
    for node in 0..branches.len() {
        let parent = branches[node].representative;
        if parent == node {
            branches[node].new_id = next_id;
            next_id += 1;
        } else {
            branches[node].new_id = branches[parent].new_id;
        }
    }

    tree.retain_mut(|edge| {
        let branch = &branches[edge.parent - n_samples];
        edge.parent = branch.new_id;
        edge.lambda = f64::from(branch.lambda_override.unwrap_or(edge.lambda as f32));
        if edge.child >= n_samples {
            let child = edge.child - n_samples;
            if branches[child].representative != child {
                return false;
            }
            edge.child = branches[child].new_id;
        }
        true
    });
    Ok(())
}
