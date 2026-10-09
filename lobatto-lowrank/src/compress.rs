//! Weighted compression of a [`LowRankSum`].
//!
//! The best rank-`r` approximation in the weighted norm, computed without
//! forming the `N_left × N_right` unfolding:
//!
//! 1. Scale the factor matrices by the square-root weights,
//!    $\tilde A = W_L^{1/2} [\rho_1 \dots \rho_M]$ and
//!    $\tilde B = W_R^{1/2} [\mu_1 \dots \mu_M]$.
//! 2. Thin QR of both, $\tilde A = Q_A R_A$, $\tilde B = Q_B R_B$.
//! 3. Core matrix $C = R_A \operatorname{diag}(a) R_B^T$, `M × M`.
//! 4. SVD $C = U \Sigma V^T$.
//! 5. Keep `r` triplets: $\rho_k = W_L^{-1/2} Q_A U_{:,k}$,
//!    $\mu_k = W_R^{-1/2} Q_B V_{:,k}$, $a_k = \sigma_k$.
//!
//! The discarded energy is $\sum_{k > r} \sigma_k^2$. Cost is
//! `O((N_left + N_right)·M² + M³)`.

use crate::lowrank::LowRankSum;

/// How many singular triplets a compression keeps.
///
/// The rank kept is the smallest `r` meeting the criterion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Truncation {
    /// Keep exactly `min(r, M)` terms.
    Rank(usize),
    /// Discarded energy at most `tol² · ‖ρ‖²`.
    RelativeTol(f64),
    /// Discarded energy at most `tol²`.
    AbsoluteTol(f64),
    /// The relative tolerance, capped at a maximum rank.
    RankOrTol(usize, f64),
}

/// Outcome of a compression.
#[derive(Debug, Clone, PartialEq)]
pub struct Compressed {
    /// Number of terms kept.
    pub kept: usize,
    /// Discarded singular values `σ_{r+1}, …, σ_M`, decreasing.
    pub discarded: Vec<f64>,
    /// Weighted `L²` norm of the field before truncation.
    pub norm: f64,
}

impl<const P: usize, const Q: usize> LowRankSum<P, Q> {
    /// Compress in place to the canonical form: orthonormal factors in the
    /// weighted norms, coefficients `a_k = σ_k ≥ 0` decreasing.
    ///
    /// Always runs the QR/SVD, even when nothing is discarded.
    pub fn compress(&mut self, t: Truncation) -> Compressed {
        let _ = t;
        todo!("weighted QR → SVD → truncation")
    }

    /// Bring to the canonical form without discarding anything.
    pub fn orthonormalize(&mut self) -> Compressed {
        self.compress(Truncation::Rank(self.len()))
    }
}
