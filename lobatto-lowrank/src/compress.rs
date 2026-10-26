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

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder;
use faer::linalg::qr::no_pivoting::factor as qr;
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::{diag::DiagMut, Conj, MatMut, MatRef, Par};
use nalgebra::{DMatrix, DMatrixView, DVector};
use rayon::prelude::*;

use crate::lowrank::{scale_rows, LowRankSum};

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
                scale_rows(&mut self.rho, &self.sqrt_w_left);
                let qr = tsqr_in_place(&mut self.rho, blocks);
                let r = qr.r();
                (qr, r)
            },
            || {
                scale_rows(&mut self.mu, &self.sqrt_w_right);
                let qr = tsqr_in_place(&mut self.mu, blocks);
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

        // 5. New factors W^{-1/2} Q [U_r; 0] and W^{-1/2} Q [V_r; 0].
        let (rho, mu) = rayon::join(
            || {
                let mut rho = qr_a.apply_q(&self.rho, u.columns(0, r));
                scale_rows(&mut rho, &self.sqrt_w_left.map(|w| 1.0 / w));
                rho
            },
            || {
                let mut mu = qr_b.apply_q(&self.mu, v.columns(0, r));
                scale_rows(&mut mu, &self.sqrt_w_right.map(|w| 1.0 / w));
                mu
            },
        );

        let discarded = sigma.as_slice()[r..].to_vec();
        self.a = sigma.rows(0, r).into_owned();
        self.rho = rho;
        self.mu = mu;
        Compressed { kept: r, discarded, norm: energy.sqrt() }
    }

    /// Bring to the canonical form without discarding anything.
    pub fn orthonormalize(&mut self) -> Compressed {
        self.compress(Truncation::Rank(self.len()))
    }
}

// ── TSQR ─────────────────────────────────────────────────────────────────────

/// Householder QR of `a` in place (faer's blocked QR, sequential): `R` on and
/// above the diagonal, the Householder vectors below. Returns the block
/// Householder factors.
fn householder_qr(a: MatMut<'_, f64>) -> DMatrix<f64> {
    let (n, m) = (a.nrows(), a.ncols());
    let block = qr::recommended_block_size::<f64>(n, m);
    let mut coeff = DMatrix::zeros(block, n.min(m));
    let mut mem = MemBuffer::new(qr::qr_in_place_scratch::<f64>(n, m, block, Par::Seq, Default::default()));
    qr::qr_in_place(
        a,
        MatMut::from_column_major_slice_mut(coeff.as_mut_slice(), block, n.min(m)),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    );
    coeff
}

/// Apply the `Q` of a [`householder_qr`] (`basis` the factorised matrix,
/// `coeff` its block factors) to `x` in place.
fn householder_apply(basis: MatRef<'_, f64>, coeff: &DMatrix<f64>, x: MatMut<'_, f64>) {
    let k = coeff.ncols();
    let mut mem = MemBuffer::new(householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
        basis.nrows(),
        coeff.nrows(),
        x.ncols(),
    ));
    householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        basis.subcols(0, k),
        MatRef::from_column_major_slice(coeff.as_slice(), coeff.nrows(), k),
        Conj::No,
        x,
        Par::Seq,
        MemStack::new(&mut mem),
    );
}

/// Disjoint mutable views of the row blocks `ranges` of `a`.
fn row_blocks<'a>(mut a: MatMut<'a, f64>, ranges: &[(usize, usize)]) -> Vec<MatMut<'a, f64>> {
    let mut blocks = Vec::with_capacity(ranges.len());
    for &(start, end) in &ranges[..ranges.len() - 1] {
        let (top, rest) = a.split_at_row_mut(end - start);
        blocks.push(top);
        a = rest;
    }
    blocks.push(a);
    blocks
}

/// One merge of the TSQR tree: two stacked `R` factors (`2M × M`),
/// factorised in place, and the block Householder factors.
struct Merge {
    stacked: DMatrix<f64>,
    coeff: DMatrix<f64>,
}

/// The thin QR `A = Q R` computed by [`tsqr_in_place`]. `Q` is kept as the
/// Householder reflectors of the row blocks (stored in the factorised `A`)
/// and of the tree merges.
pub(crate) struct TallSkinnyQr {
    /// Row ranges of the blocks.
    ranges: Vec<(usize, usize)>,
    /// Block Householder factors of each row block.
    coeffs: Vec<DMatrix<f64>>,
    /// The tree, from the leaves up: entry `i` of a level merges nodes
    /// `2i` and `2i + 1` of the level below, or is `None` when an odd last
    /// node goes up unchanged.
    levels: Vec<Vec<Option<Merge>>>,
    /// The `R` factor.
    r: DMatrix<f64>,
}

/// Thin QR of `a` (`N × M`) in place by TSQR: the rows are split into `p`
/// blocks, each factorised by faer's sequential QR, in parallel; their `R`
/// factors are then merged pairwise up a binary tree, each level in
/// parallel. `p` is `blocks`, capped so that every block has at least `M`
/// rows; with `p = 1` this is a plain Householder QR.
pub(crate) fn tsqr_in_place(a: &mut DMatrix<f64>, blocks: usize) -> TallSkinnyQr {
    let (n, m) = a.shape();
    let p = n.checked_div(m).map_or(1, |rows| blocks.min(rows).max(1));
    let ranges: Vec<(usize, usize)> = (0..p).map(|i| (i * n / p, (i + 1) * n / p)).collect();

    // Leaves: QR of each row block, and a copy of its R.
    let mut views = row_blocks(MatMut::from_column_major_slice_mut(a.as_mut_slice(), n, m), &ranges);
    let (coeffs, mut rs): (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) = views
        .par_iter_mut()
        .map(|b| {
            let coeff = householder_qr(b.rb_mut());
            let k = b.nrows().min(m);
            let mut r = DMatrix::zeros(k, m);
            MatMut::from_column_major_slice_mut(r.as_mut_slice(), k, m).copy_from_triangular_upper((*b).rb().subrows(0, k));
            (coeff, r)
        })
        .unzip();

    // Tree: QR of each pair of stacked R factors.
    let mut levels = Vec::new();
    while rs.len() > 1 {
        let (merges, next): (Vec<Option<Merge>>, Vec<DMatrix<f64>>) = rs
            .par_chunks(2)
            .map(|pair| match pair {
                [upper, lower] => {
                    let mut stacked = DMatrix::zeros(2 * m, m);
                    stacked.rows_mut(0, m).copy_from(upper);
                    stacked.rows_mut(m, m).copy_from(lower);
                    let coeff = householder_qr(MatMut::from_column_major_slice_mut(stacked.as_mut_slice(), 2 * m, m));
                    let r = stacked.rows(0, m).upper_triangle();
                    (Some(Merge { stacked, coeff }), r)
                }
                _ => (None, pair[0].clone()),
            })
            .unzip();
        levels.push(merges);
        rs = next;
    }
    let r = rs.pop().expect("at least one block");
    TallSkinnyQr { ranges, coeffs, levels, r }
}

impl TallSkinnyQr {
    /// The `R` factor, `min(N, M) × M` upper trapezoidal.
    pub(crate) fn r(&self) -> DMatrix<f64> {
        self.r.clone()
    }

    /// `Q [top; 0]`, `N × c`, for `top` with `min(N, M)` rows; `a` is the
    /// matrix factorised by [`tsqr_in_place`].
    pub(crate) fn apply_q(&self, a: &DMatrix<f64>, top: DMatrixView<f64>) -> DMatrix<f64> {
        let (n, m, c) = (a.nrows(), a.ncols(), top.ncols());

        // Down the tree: each merge turns its node's `M × c` block into the
        // blocks of its two children.
        let mut ys = vec![top.into_owned()];
        for level in self.levels.iter().rev() {
            ys = level
                .par_iter()
                .zip(ys.into_par_iter())
                .flat_map_iter(|(merge, y)| match merge {
                    Some(Merge { stacked, coeff }) => {
                        let mut z = DMatrix::zeros(2 * m, c);
                        z.rows_mut(0, m).copy_from(&y);
                        householder_apply(
                            MatRef::from_column_major_slice(stacked.as_slice(), 2 * m, m),
                            coeff,
                            MatMut::from_column_major_slice_mut(z.as_mut_slice(), 2 * m, c),
                        );
                        vec![z.rows(0, m).into_owned(), z.rows(m, m).into_owned()]
                    }
                    None => vec![y],
                })
                .collect();
        }

        // Leaves: each row block's reflectors applied to [y_i; 0].
        let basis = MatRef::from_column_major_slice(a.as_slice(), n, m);
        let mut out = DMatrix::zeros(n, c);
        let views = row_blocks(MatMut::from_column_major_slice_mut(out.as_mut_slice(), n, c), &self.ranges);
        views.into_par_iter().zip(ys.par_iter()).enumerate().for_each(|(i, (mut block, y))| {
            let (start, end) = self.ranges[i];
            let k = y.nrows();
            block.rb_mut().subrows_mut(0, k).copy_from(MatRef::from_column_major_slice(y.as_slice(), k, c));
            householder_apply(basis.subrows(start, end - start), &self.coeffs[i], block);
        });
        out
    }
}

// ── SVD ──────────────────────────────────────────────────────────────────────

/// Thin SVD `core = U diag(σ) Vᵀ` by faer, `σ` decreasing; returns `V`, not
/// `Vᵀ`.
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

    /// `core_svd` on cores `R_A diag(a) R_Bᵀ` built from the QR of
    /// rank-deficient factor matrices (rank 20 of 60, plus round-off-sized
    /// perturbations: many small clustered singular values). Square (both
    /// sides with more nodes than terms) and rectangular (40 right nodes for
    /// 60 terms).
    #[test]
    fn core_svd_reconstructs_rank_deficient_cores() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let (m, rank) = (60, 20);
        for n_right in [200, 40] {
            let mut factors = [200, n_right].map(|n| {
                let g = DMatrix::from_fn(n, rank, |_, _| next());
                let h = DMatrix::from_fn(rank, m, |_, _| next());
                &g * h + DMatrix::from_fn(n, m, |_, _| 1e-14 * next())
            });
            let [r_a, r_b] = factors.each_mut().map(|f| tsqr_in_place(f, rayon::current_num_threads()).r());
            let a = DVector::from_fn(m, |_, _| 1.0 + next());
            let core = r_a * DMatrix::from_diagonal(&a) * r_b.transpose();

            let (u, sigma, v) = core_svd(&core);
            let k = core.nrows().min(core.ncols());
            let rec = (&core - &u * DMatrix::from_diagonal(&sigma) * v.transpose()).norm() / core.norm();
            let orth_u = (u.transpose() * &u - DMatrix::<f64>::identity(k, k)).amax();
            let orth_v = (v.transpose() * &v - DMatrix::<f64>::identity(k, k)).amax();
            assert!(rec < 1e-13, "{n_right} right nodes: reconstruction {rec:e}");
            assert!(orth_u < 1e-13 && orth_v < 1e-13, "{n_right} right nodes: UᵀU {orth_u:e}, VᵀV {orth_v:e}");
            assert!(sigma.iter().all(|&s| s >= 0.0));
            assert!(sigma.as_slice().windows(2).all(|w| w[0] >= w[1]), "σ not decreasing");
        }
    }

    /// TSQR in pools of 1 (a plain QR), 2, 3 and 7 threads (that many row
    /// blocks): `Q [R; 0] = A`, orthonormal `Q`, and `R` equal to nalgebra's
    /// up to the signs of its rows. Tall (500 × 37) and wide (30 × 37).
    #[test]
    fn tsqr_is_a_qr_for_any_pool_size() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for (n, m) in [(500, 37), (30, 37)] {
            let a = DMatrix::from_fn(n, m, |_, _| next());
            let k = n.min(m);
            let r_ref = a.clone().qr().r();
            for threads in [1, 2, 3, 7] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                pool.install(|| {
                    let mut f = a.clone();
                    let qr = tsqr_in_place(&mut f, threads);
                    let r = qr.r();
                    let rec = (qr.apply_q(&f, r.as_view()) - &a).amax() / a.amax();
                    let q = qr.apply_q(&f, DMatrix::<f64>::identity(k, k).as_view());
                    let orth = (q.transpose() * &q - DMatrix::<f64>::identity(k, k)).amax();
                    let r_diff = (r.abs() - r_ref.abs()).amax() / r_ref.amax();
                    let case = format!("{n} × {m}, {threads} threads");
                    assert!(rec < 1e-13, "{case}: ‖QR − A‖ {rec:e}");
                    assert!(orth < 1e-13, "{case}: ‖QᵀQ − I‖ {orth:e}");
                    assert!(r_diff < 1e-12, "{case}: |R| vs nalgebra {r_diff:e}");
                });
            }
        }
    }
}
