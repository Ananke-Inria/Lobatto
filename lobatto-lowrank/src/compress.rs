//! Weighted compression of a [`LowRankSum`].
//!
//! The best rank-`r` approximation in the weighted norm, computed without
//! forming the `N_left × N_right` unfolding:
//!
//! 1. Scale the factor matrices by the square-root weights,
//!    $\tilde A = W_L^{1/2} [\rho_1 \dots \rho_M]$ and
//!    $\tilde B = W_R^{1/2} [\mu_1 \dots \mu_M]$.
//! 2. Thin Householder QR of both (nalgebra), $\tilde A = Q_A R_A$,
//!    $\tilde B = Q_B R_B$.
//! 3. Core matrix $C = R_A \operatorname{diag}(a) R_B^T$, at most `M × M`.
//! 4. SVD $C = U \Sigma V^T$, singular values sorted decreasing, by faer
//!    (nalgebra's SVD returns wrong singular values on some cores; see
//!    `core_svd`).
//! 5. Keep `r` triplets: $\rho_k = W_L^{-1/2} Q_A U_{:,k}$,
//!    $\mu_k = W_R^{-1/2} Q_B V_{:,k}$, $a_k = \sigma_k$.
//!
//! The squared norm of the field is $\sum_k \sigma_k^2$ and the discarded
//! energy $\sum_{k > r} \sigma_k^2$, both free of the cancellation that
//! affects [`LowRankSum::norm_squared`]. Linearly dependent factors (after an
//! addition) need no special care: they give near-zero rows of `R` and zero
//! singular values. Cost `O((N_left + N_right)·M² + M³)`.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::{diag::DiagMut, MatMut, MatRef, Par};
use nalgebra::{DMatrix, DVector};

use crate::lowrank::LowRankSum;

/// How many singular triplets a compression keeps.
///
/// The rank kept is the smallest `r` meeting the criterion.
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
    /// The result is the best approximation of the field in the weighted
    /// norm among sums with the number of terms kept (see the module docs);
    /// its squared distance to the field before compression is
    /// `Σ discarded²`. Always runs the QR/SVD, even when nothing is
    /// discarded. An empty sum stays empty.
    pub fn compress(&mut self, t: Truncation) -> Compressed {
        if self.is_empty() {
            return Compressed {
                kept: 0,
                discarded: Vec::new(),
                norm: 0.0,
            };
        }
        // The row scalings below are entry-wise products of each (contiguous)
        // column with the stored square-root weights or their reciprocals.

        // 1–2. Thin QR of the weight-scaled factor matrices W^{1/2} ρ, W^{1/2} μ.
        // The factors are replaced at the end, so they are scaled in place and
        // moved into the QR, which consumes them: no copy. The two sides are
        // independent and run as two Rayon tasks.
        let ((q_a, mut r_a), (q_b, r_b)) = rayon::join(
            || {
                for mut col in self.rho.column_iter_mut() {
                    col.component_mul_assign(&self.sqrt_w_left);
                }
                let qr = std::mem::take(&mut self.rho).qr();
                (qr.q(), qr.r())
            },
            || {
                for mut col in self.mu.column_iter_mut() {
                    col.component_mul_assign(&self.sqrt_w_right);
                }
                let qr = std::mem::take(&mut self.mu).qr();
                (qr.q(), qr.r())
            },
        );

        // 3. Core R_A diag(a) R_Bᵀ.
        for (mut col, &a) in r_a.column_iter_mut().zip(self.a.iter()) {
            col *= a;
        }
        let core = r_a * r_b.transpose();

        // 4. SVD, singular values sorted decreasing.
        let (u, sigma, v) = core_svd(&core);
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

        // 5. New factors W^{-1/2} Q U_r and W^{-1/2} Q V_r, coefficients σ_r.
        // The two sides again as two Rayon tasks.
        let (rho, mu) = rayon::join(
            || {
                let mut rho = q_a * u.columns(0, r);
                for mut col in rho.column_iter_mut() {
                    col.component_mul_assign(&self.inv_sqrt_w_left);
                }
                rho
            },
            || {
                let mut mu = q_b * v.columns(0, r);
                for mut col in mu.column_iter_mut() {
                    col.component_mul_assign(&self.inv_sqrt_w_right);
                }
                mu
            },
        );

        let discarded = sigma.as_slice()[r..].to_vec();
        self.a = sigma.rows(0, r).into_owned();
        self.rho = rho;
        self.mu = mu;
        Compressed {
            kept: r,
            discarded,
            norm: energy.sqrt(),
        }
    }

    /// Bring to the canonical form without discarding anything.
    pub fn orthonormalize(&mut self) -> Compressed {
        self.compress(Truncation::Rank(self.len()))
    }
}

/// Thin SVD `core = U diag(σ) Vᵀ` with `σ` decreasing; returns `V`, not `Vᵀ`.
///
/// Computed by faer, which reads `core` and writes `U`, `σ` and `V` through
/// views of the nalgebra column-major storage: no conversion copies.
/// nalgebra's own SVD (0.34 and 0.35) is not used: on some rank-deficient
/// cores with clustered small singular values it returns orthonormal `U` and
/// `V` but wrong singular values, which breaks the guarantee of [`LowRankSum::compress`].
///
/// In debug builds the result is checked on one probe vector `x`,
/// `‖C x − U Σ Vᵀ x‖ ≤ 1e-12 ‖C‖_F ‖x‖`, at cost `O(m·n)`.
fn core_svd(core: &DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
    let (m, n) = core.shape();
    let k = m.min(n);
    let mut u = DMatrix::zeros(m, k);
    let mut sigma = DVector::zeros(k);
    let mut v = DMatrix::zeros(n, k);
    let thin = ComputeSvdVectors::Thin;
    let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(m, n, thin, thin, Par::Seq, Default::default()));
    svd::svd(
        MatRef::from_column_major_slice(core.as_slice(), m, n),
        DiagMut::from_slice_mut(sigma.as_mut_slice()),
        Some(MatMut::from_column_major_slice_mut(u.as_mut_slice(), m, k)),
        Some(MatMut::from_column_major_slice_mut(v.as_mut_slice(), n, k)),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .expect("SVD of the core did not converge");

    #[cfg(debug_assertions)]
    {
        // A probe with no special structure: x_j = 1 + frac(j·φ⁻¹).
        let x = DVector::from_fn(n, |j, _| 1.0 + (j as f64 * 0.618_033_988_749_895).fract());
        let mut y = v.tr_mul(&x);
        y.component_mul_assign(&sigma);
        let err = (core * &x - &u * y).norm();
        debug_assert!(
            err <= 1e-12 * core.norm() * x.norm(),
            "inaccurate SVD of the core: residual {err:e}"
        );
    }
    (u, sigma, v)
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
        s.rho = DMatrix::from_fn(s.n_left(), m, |_, _| next());
        s.mu = DMatrix::from_fn(s.n_right(), m, |_, _| next());
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
            g.rho.column_mut(k).scale_mut(c);
            g.mu.column_mut(k).scale_mut(1.0 / c);
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
