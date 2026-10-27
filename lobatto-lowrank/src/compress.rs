//! Weighted compression of a [`LowRankSum`]: the best rank-`r`
//! approximation in the weighted norm, without forming the
//! `N_left × N_right` unfolding.
//!
//! 1. Scale the factor matrices by the square-root weights,
//!    $\tilde A = W_L^{1/2} [\rho_1 \dots \rho_M]$ and
//!    $\tilde B = W_R^{1/2} [\mu_1 \dots \mu_M]$.
//! 2. Thin QR of both by TSQR, in place: $\tilde A = Q_A R_A$,
//!    $\tilde B = Q_B R_B$, `Q` kept as Householder reflectors.
//! 3. Core matrix $C = R_A \operatorname{diag}(a) R_B^T$, at most `M × M`.
//! 4. SVD $C = U \Sigma V^T$ (faer), singular values decreasing.
//! 5. Keep `r` triplets: $\rho_k = W_L^{-1/2} Q_A U_{:,k}$,
//!    $\mu_k = W_R^{-1/2} Q_B V_{:,k}$, $a_k = \sigma_k$.
//!
//! The norm of the field is $(\sum_k \sigma_k^2)^{1/2}$ and the error of the
//! truncation $(\sum_{k > r} \sigma_k^2)^{1/2}$. Cost
//! `O((N_left + N_right)·M² + M³)`.

use crate::linalg::{scale_rows, thin_svd, tsqr_in_place};
use crate::lowrank::LowRankSum;

/// How many singular triplets a compression keeps: the smallest `r`
/// meeting the criterion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Truncation {
    /// Keep exactly `min(r, s)` terms, `s = min(N_left, N_right, M)`.
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
    /// Discarded singular values `σ_{r+1}, …, σ_s`, decreasing, with
    /// `s = min(N_left, N_right, M)`; the remaining ones are zero.
    pub discarded: Vec<f64>,
    /// Weighted `L²` norm of the field before truncation, `(Σ_k σ_k²)^{1/2}`.
    pub norm: f64,
}

impl<const P: usize, const Q: usize> LowRankSum<P, Q> {
    /// Compress in place to the canonical form: orthonormal factors in the
    /// weighted norms, coefficients `a_k = σ_k ≥ 0` decreasing.
    ///
    /// The result is the best approximation of the field in the weighted norm
    /// with the number of terms kept; its squared distance to the field
    /// before compression is `Σ discarded²`. An empty sum stays empty.
    ///
    /// Runs in parallel on the current Rayon pool. The result is the same up
    /// to round-off for any number of threads, and bit for bit for a given
    /// one.
    pub fn compress(&mut self, t: Truncation) -> Compressed {
        if self.is_empty() {
            return Compressed { kept: 0, discarded: Vec::new(), norm: 0.0 };
        }

        // 1–2. Scale and factorise both sides concurrently, each with half the
        // pool's worth of row blocks. The factors are overwritten by the QR.
        let blocks = (rayon::current_num_threads() / 2).max(1);
        let ((qr_a, mut r_a), (qr_b, r_b)) = rayon::join(
            || {
                scale_rows(&mut self.left.u, &self.left.sqrt_w);
                let qr = tsqr_in_place(&mut self.left.u, blocks);
                let r = qr.r();
                (qr, r)
            },
            || {
                scale_rows(&mut self.right.u, &self.right.sqrt_w);
                let qr = tsqr_in_place(&mut self.right.u, blocks);
                let r = qr.r();
                (qr, r)
            },
        );

        // 3. Core R_A diag(a) R_Bᵀ.
        for (mut col, &a) in r_a.column_iter_mut().zip(self.a.iter()) {
            col *= a;
        }
        let core = r_a * r_b.transpose();

        // 4. SVD and the number of terms kept.
        let (u, sigma, v) = thin_svd(&core);
        let s = sigma.len();
        let energy: f64 = sigma.iter().map(|x| x * x).sum();
        // Smallest r whose tail energy Σ_{k ≥ r} σ_k² is at most `threshold`.
        let smallest_rank = |threshold: f64| {
            let (mut r, mut tail) = (s, 0.0);
            while r > 0 && tail + sigma[r - 1] * sigma[r - 1] <= threshold {
                tail += sigma[r - 1] * sigma[r - 1];
                r -= 1;
            }
            r
        };
        let r = match t {
            Truncation::Rank(r) => r.min(s),
            Truncation::AbsoluteTol(tol) => smallest_rank(tol * tol),
            Truncation::RelativeTol(tol) => smallest_rank(tol * tol * energy),
            Truncation::RankOrTol(r_max, tol) => smallest_rank(tol * tol * energy).min(r_max),
        };

        // 5. New factors W^{-1/2} Q [U_r; 0] and W^{-1/2} Q [V_r; 0].
        let (rho, mu) = rayon::join(
            || {
                let mut rho = qr_a.apply_q(&self.left.u, u.columns(0, r));
                scale_rows(&mut rho, &self.left.sqrt_w.map(|w| 1.0 / w));
                rho
            },
            || {
                let mut mu = qr_b.apply_q(&self.right.u, v.columns(0, r));
                scale_rows(&mut mu, &self.right.sqrt_w.map(|w| 1.0 / w));
                mu
            },
        );

        let discarded = sigma.as_slice()[r..].to_vec();
        self.a = sigma.rows(0, r).into_owned();
        self.left.u = rho;
        self.right.u = mu;
        Compressed { kept: r, discarded, norm: energy.sqrt() }
    }

    /// Bring to the canonical form without discarding anything.
    pub fn orthonormalize(&mut self) -> Compressed {
        self.compress(Truncation::Rank(self.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lobatto_grid::{Extension, GridND};
    use nalgebra::{DMatrix, DVector};

    /// Left: periodic × Dirichlet (2D), right: Neumann (1D); mixed extensions.
    fn grids() -> (GridND<2>, GridND<1>) {
        let left = GridND::<2>::new(
            [3, 2],
            [4, 4],
            [2.0, 1.5],
            [-1.0, 0.5],
            [Extension::Periodic, Extension::Odd],
        );
        let right = GridND::<1>::new([3], [4], [1.0], [0.0], [Extension::Even]);
        (left, right)
    }

    /// `m` terms with pseudo-random factors and coefficients.
    fn random_sum(m: usize, seed: u64) -> LowRankSum<2, 1> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let (l, r) = grids();
        let mut s = LowRankSum::new(l, r);
        s.a = DVector::from_fn(m, |_, _| 1.0 + next());
        s.left.u = DMatrix::from_fn(s.n_left(), m, |_, _| next());
        s.right.u = DMatrix::from_fn(s.n_right(), m, |_, _| next());
        s
    }

    /// Max entry-wise difference of the assembled fields.
    fn field_diff(a: &LowRankSum<2, 1>, b: &LowRankSum<2, 1>) -> f64 {
        a.assemble()
            .iter()
            .zip(b.assemble())
            .fold(0.0, |m, (u, v)| f64::max(m, (u - v).abs()))
    }

    /// Singular values of the weighted assembled unfolding `W_L^{1/2} P W_R^{1/2}`.
    fn dense_singular_values(s: &LowRankSum<2, 1>) -> Vec<f64> {
        let (nl, nr) = (s.n_left(), s.n_right());
        let mut p = DMatrix::from_vec(nl, nr, s.assemble());
        let (wl, wr) = (s.left().mass_weights(), s.right().mass_weights());
        for j in 0..nr {
            for i in 0..nl {
                p[(i, j)] *= (wl[i] * wr[j]).sqrt();
            }
        }
        p.svd(false, false).singular_values.iter().copied().collect()
    }

    fn assert_canonical(s: &LowRankSum<2, 1>) {
        let m = s.len();
        let id = DMatrix::<f64>::identity(m, m);
        assert!((s.gram_left() - &id).abs().max() < 1e-12, "left factors not orthonormal");
        assert!((s.gram_right() - &id).abs().max() < 1e-12, "right factors not orthonormal");
        let a = s.coefficients();
        assert!(a.iter().all(|&x| x >= 0.0));
        assert!(a.windows(2).all(|w| w[0] >= w[1]), "coefficients not decreasing: {a:?}");
    }

    #[test]
    fn single_product_compresses_to_itself() {
        let (l, r) = grids();
        let mut s = LowRankSum::new_from_fn(l, r, |x| x[0].cos() * (x[1] - 0.5), |y| 1.0 + y[0]);
        s.scale(-2.0);
        let orig = s.clone();
        let c = s.compress(Truncation::RelativeTol(1e-14));
        assert_eq!(c.kept, 1);
        assert!(c.discarded.is_empty());
        assert!(field_diff(&s, &orig) < 1e-14);
        assert_canonical(&s);
        // a_1 = ‖ρ‖ ‖μ‖ (the sign moves into a factor).
        let expected = 2.0 * (orig.gram_left()[(0, 0)] * orig.gram_right()[(0, 0)]).sqrt();
        assert!((s.coefficients()[0] - expected).abs() < 1e-13 * expected);
        assert!((c.norm - expected).abs() < 1e-13 * expected);
    }

    #[test]
    fn redundant_terms_collapse_to_the_true_rank() {
        // A rank-3 field written as each product twice with half the
        // coefficient, plus a zero term: 7 terms, rank 3.
        let base = random_sum(3, 1);
        let mut half = base.clone();
        half.scale(0.5);
        let mut zero = random_sum(1, 2);
        zero.scale(0.0);
        let mut s = half.add(&half).add(&zero);
        assert_eq!(s.len(), 7);
        let c = s.compress(Truncation::RelativeTol(1e-12));
        assert_eq!(c.kept, 3);
        assert!(c.discarded.iter().all(|&x| x <= 1e-13 * c.norm), "{:?}", c.discarded);
        assert!(field_diff(&s, &base) < 1e-13);
        assert_canonical(&s);
    }

    #[test]
    fn rank_truncation_is_the_best_approximation() {
        let orig = random_sum(8, 3);
        let dense = dense_singular_values(&orig);
        let mut s = orig.clone();
        let c = s.compress(Truncation::Rank(4));
        assert_eq!(c.kept, 4);
        assert_eq!(c.discarded.len(), 4);
        // Singular values of the factored path = those of the dense unfolding.
        let sv: Vec<f64> = s.coefficients().iter().chain(&c.discarded).copied().collect();
        for (k, (x, y)) in sv.iter().zip(&dense).enumerate() {
            assert!((x - y).abs() < 1e-12 * dense[0], "σ_{k}: {x} vs {y}");
        }
        // Error of the truncation = discarded energy.
        let tail = c.discarded.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((orig.distance(&s) - tail).abs() < 1e-12 * c.norm);
        assert!((c.norm * c.norm - orig.norm_squared()).abs() < 1e-12 * c.norm * c.norm);
        assert_canonical(&s);
    }

    #[test]
    fn tolerance_criteria_keep_the_expected_rank() {
        // Canonical factors from a random sum, then σ_k = 2^{-k}, k = 0..8.
        let mut s = random_sum(8, 4);
        s.orthonormalize();
        s.a = DVector::from_fn(8, |k, _| 0.5f64.powi(k as i32));
        let energy: f64 = (0..8).map(|k| 0.25f64.powi(k)).sum();
        // Smallest r with Σ_{k ≥ r} 4^{-k} ≤ threshold.
        let expected = |threshold: f64| {
            (0..=8)
                .find(|&r| (r..8).map(|k| 0.25f64.powi(k as i32)).sum::<f64>() <= threshold)
                .unwrap()
        };
        for tol in [0.3, 0.1, 1e-2, 1e-3] {
            let mut t = s.clone();
            assert_eq!(t.compress(Truncation::AbsoluteTol(tol)).kept, expected(tol * tol), "abs {tol}");
            let mut t = s.clone();
            let want = expected(tol * tol * energy);
            assert_eq!(t.compress(Truncation::RelativeTol(tol)).kept, want, "rel {tol}");
            let mut t = s.clone();
            assert_eq!(t.compress(Truncation::RankOrTol(2, tol)).kept, want.min(2), "cap {tol}");
        }
    }

    #[test]
    fn compress_is_idempotent() {
        let mut s = random_sum(6, 5);
        let c1 = s.compress(Truncation::Rank(4));
        let once = s.clone();
        let c2 = s.compress(Truncation::Rank(4));
        assert_eq!(c2.kept, 4);
        assert!(c2.discarded.is_empty());
        assert!(field_diff(&s, &once) < 1e-13);
        for (x, y) in s.coefficients().iter().zip(once.coefficients()) {
            assert!((x - y).abs() < 1e-13 * c1.norm);
        }
    }

    #[test]
    fn gauge_invariance() {
        let orig = random_sum(5, 6);
        let mut g = orig.clone();
        for k in 0..5 {
            let c = 0.3 + k as f64;
            g.left.u.column_mut(k).scale_mut(c);
            g.right.u.column_mut(k).scale_mut(1.0 / c);
        }
        let (mut a, mut b) = (orig.clone(), g);
        a.compress(Truncation::Rank(3));
        b.compress(Truncation::Rank(3));
        assert!(field_diff(&a, &b) < 1e-12);
        for (x, y) in a.coefficients().iter().zip(b.coefficients()) {
            assert!((x - y).abs() < 1e-12 * x);
        }
    }

    #[test]
    fn orthonormalize_keeps_the_field() {
        let mut s = random_sum(5, 7);
        let orig = s.clone();
        let c = s.orthonormalize();
        assert_eq!(c.kept, 5);
        assert!(field_diff(&s, &orig) < 1e-13);
        assert_canonical(&s);
    }

    #[test]
    fn empty_sum_stays_empty() {
        let (l, r) = grids();
        let mut s = LowRankSum::new(l, r);
        let c = s.compress(Truncation::RelativeTol(1e-8));
        assert_eq!(c, Compressed { kept: 0, discarded: vec![], norm: 0.0 });
        assert!(s.is_empty());
    }

    #[test]
    fn more_terms_than_right_nodes() {
        // Right grid has 13 nodes; 20 terms give at most 13 singular values.
        let orig = random_sum(20, 8);
        let mut s = orig.clone();
        let c = s.orthonormalize();
        assert_eq!(c.kept + c.discarded.len(), s.n_right().min(20));
        assert!(field_diff(&s, &orig) < 1e-12);
    }
}
