//! Weighted `L²` Gram matrices, norms and inner products of [`LowRankSum`]s,
//! computed without assembling the field.

use nalgebra::DMatrix;

use crate::lowrank::LowRankSum;

impl<const P: usize, const Q: usize> LowRankSum<P, Q> {
    /// Weighted Gram matrix of the left factors, `G_kl = ⟨ρ_k, ρ_l⟩_w`, `M × M`.
    pub fn gram_left(&self) -> DMatrix<f64> {
        todo!("weighted Gram matrix of the left factors")
    }

    /// Weighted Gram matrix of the right factors, `M × M`.
    pub fn gram_right(&self) -> DMatrix<f64> {
        todo!("weighted Gram matrix of the right factors")
    }

    /// Squared weighted `L²` norm of the field, without assembling it.
    ///
    /// $$
    /// \|\rho\|^2 = \sum_{k,l} a_k a_l \, G^L_{kl} \, G^R_{kl}.
    /// $$
    pub fn norm_squared(&self) -> f64 {
        todo!("contract the two Gram matrices")
    }

    /// Weighted `L²` inner product with another sum on the same grids.
    pub fn inner(&self, other: &Self) -> f64 {
        let _ = other;
        todo!("contract the two cross Gram matrices")
    }

    /// Squared weighted `L²` distance to another sum on the same grids.
    pub fn distance_squared(&self, other: &Self) -> f64 {
        let _ = other;
        todo!("‖self‖² − 2⟨self, other⟩ + ‖other‖²")
    }
}
