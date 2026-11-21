use crate::{
    FeatureMetric, FitError,
    distance::{dense_l1, dense_squared_l2},
};

/// Distance functions usable by the dense spatial fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TreeDistance<'a> {
    Euclidean,
    Manhattan,
    Chebyshev,
    Minkowski { p: f64, weights: Option<&'a [f64]> },
    StandardizedEuclidean(&'a [f64]),
    Mahalanobis(&'a [f64]),
    Hamming,
    Canberra,
    BrayCurtis,
    Jaccard,
    Dice,
    RogersTanimoto,
    RussellRao,
    SokalMichener,
    SokalSneath,
    Haversine,
}

impl TryFrom<FeatureMetric> for TreeDistance<'static> {
    type Error = FitError;

    fn try_from(metric: FeatureMetric) -> Result<Self, Self::Error> {
        match metric {
            FeatureMetric::Euclidean => Ok(Self::Euclidean),
            FeatureMetric::Manhattan => Ok(Self::Manhattan),
            FeatureMetric::Cosine => Err(FitError::UnsupportedMetric),
        }
    }
}

impl TreeDistance<'_> {
    /// Whether this distance supplies the box bounds used by a KD-tree.
    pub fn supports_kd(self) -> bool {
        matches!(
            self,
            Self::Euclidean | Self::Manhattan | Self::Chebyshev | Self::Minkowski { .. }
        )
    }

    pub(crate) fn validated(self, dimensions: usize) -> Result<Self, FitError> {
        match self {
            Self::Minkowski { p, weights } => {
                if p.is_nan()
                    || p < 1.0
                    || weights.is_some_and(|weights| {
                        weights.len() != dimensions
                            || weights
                                .iter()
                                .any(|weight| !weight.is_finite() || *weight < 0.0)
                    })
                {
                    return Err(FitError::InvalidDistanceOptions);
                }
                if p == f64::INFINITY {
                    return Ok(Self::Chebyshev);
                }
                if weights.is_none() {
                    if p == 1.0 {
                        return Ok(Self::Manhattan);
                    }
                    if p == 2.0 {
                        return Ok(Self::Euclidean);
                    }
                }
            }
            Self::StandardizedEuclidean(v) if v.len() != dimensions => {
                return Err(FitError::InvalidDistanceOptions);
            }
            Self::Mahalanobis(vi) if dimensions.checked_mul(dimensions) != Some(vi.len()) => {
                return Err(FitError::InvalidDistanceOptions);
            }
            Self::Haversine if dimensions != 2 => return Err(FitError::InvalidDistanceOptions),
            _ => {}
        }
        Ok(self)
    }

    pub(crate) fn reduced(self, left: &[f64], right: &[f64]) -> f64 {
        let differences = left.iter().zip(right).map(|(&x, &y)| (x - y).abs());
        match self {
            Self::Euclidean => dense_squared_l2(left, right),
            Self::Manhattan => dense_l1(left, right),
            Self::Chebyshev | Self::Minkowski { .. } => self.box_distance(differences),
            Self::StandardizedEuclidean(v) => differences.zip(v).map(|(d, &v)| d * d / v).sum(),
            Self::Mahalanobis(vi) => {
                let delta: Vec<_> = left.iter().zip(right).map(|(x, y)| x - y).collect();
                vi.chunks_exact(delta.len())
                    .zip(&delta)
                    .map(|(row, d)| row.iter().zip(&delta).map(|(v, x)| v * x).sum::<f64>() * d)
                    .sum()
            }
            Self::Hamming => {
                left.iter().zip(right).filter(|(x, y)| x != y).count() as f64 / left.len() as f64
            }
            Self::Canberra => differences
                .zip(left.iter().zip(right))
                .map(|(d, (x, y))| {
                    let denominator = x.abs() + y.abs();
                    if denominator == 0.0 {
                        0.0
                    } else {
                        d / denominator
                    }
                })
                .sum(),
            Self::BrayCurtis => {
                let denominator: f64 = left.iter().zip(right).map(|(x, y)| (x + y).abs()).sum();
                if denominator == 0.0 {
                    0.0
                } else {
                    differences.sum::<f64>() / denominator
                }
            }
            Self::Haversine => {
                let lat = ((left[0] - right[0]) * 0.5).sin();
                let lon = ((left[1] - right[1]) * 0.5).sin();
                lat * lat + left[0].cos() * right[0].cos() * lon * lon
            }
            metric => {
                let (both, different) =
                    left.iter()
                        .zip(right)
                        .fold((0_u64, 0_u64), |(both, different), (&x, &y)| {
                            let x = x != 0.0;
                            let y = y != 0.0;
                            (both + u64::from(x && y), different + u64::from(x != y))
                        });
                let n = left.len() as f64;
                let (both, different) = (both as f64, different as f64);
                match metric {
                    Self::Jaccard => {
                        if both + different == 0.0 {
                            0.0
                        } else {
                            different / (both + different)
                        }
                    }
                    Self::Dice => different / (2.0 * both + different),
                    Self::RogersTanimoto | Self::SokalMichener => 2.0 * different / (n + different),
                    Self::RussellRao => (n - both) / n,
                    Self::SokalSneath => 2.0 * different / (both + 2.0 * different),
                    _ => unreachable!(),
                }
            }
        }
    }

    pub(crate) fn box_distance(self, differences: impl Iterator<Item = f64>) -> f64 {
        match self {
            Self::Euclidean => differences.map(|d| d * d).sum(),
            Self::Manhattan => differences.sum(),
            Self::Chebyshev => differences.fold(0.0, f64::max),
            Self::Minkowski { p, weights: None } => differences.map(|d| d.powf(p)).sum(),
            Self::Minkowski {
                p,
                weights: Some(weights),
            } => differences.zip(weights).map(|(d, &w)| w * d.powf(p)).sum(),
            _ => unreachable!(),
        }
    }

    pub(crate) fn distance(self, reduced: f64) -> f64 {
        match self {
            Self::Euclidean | Self::StandardizedEuclidean(_) | Self::Mahalanobis(_) => {
                reduced.sqrt()
            }
            Self::Minkowski { p, .. } => reduced.powf(1.0 / p),
            Self::Haversine => 2.0 * reduced.sqrt().asin(),
            _ => reduced,
        }
    }

    pub(crate) fn to_reduced(self, distance: f64) -> f64 {
        match self {
            Self::Euclidean | Self::StandardizedEuclidean(_) | Self::Mahalanobis(_) => {
                distance * distance
            }
            Self::Minkowski { p, .. } => distance.powf(p),
            Self::Haversine => {
                let sine = (distance * 0.5).sin();
                sine * sine
            }
            _ => distance,
        }
    }
}
