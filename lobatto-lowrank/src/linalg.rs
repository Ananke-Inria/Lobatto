//! Dense linear algebra on nalgebra matrices, used by the compression and
//! the distance: thin QR of tall-skinny matrices by TSQR, thin SVD, and row
//! scaling. The factorisations are computed by faer on views of the nalgebra
//! storage.
//!
//! | Function | Contents |
//! |----------|----------|
//! | [`tsqr_in_place`] | Thin QR `A = Q R` of an `N × M` matrix in place, parallel over row blocks; `Q` kept as Householder reflectors ([`TallSkinnyQr`]). |
//! | [`thin_svd`] | Thin SVD `A = U diag(σ) Vᵀ`, `σ` decreasing. |
//! | [`scale_rows`] | `A[i, k] *= w[i]`, the columns in parallel. |

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder;
use faer::linalg::qr::no_pivoting::factor as qr;
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::{diag::DiagMut, Conj, MatMut, MatRef, Par};
use nalgebra::{DMatrix, DMatrixView, DVector};
use rayon::prelude::*;

/// `m[i, k] *= w[i]` for every column `k`, the columns in parallel.
pub fn scale_rows(m: &mut DMatrix<f64>, w: &DVector<f64>) {
    let n = m.nrows().max(1);
    m.as_mut_slice().par_chunks_mut(n).for_each(|col| {
        col.iter_mut().zip(w.iter()).for_each(|(x, w)| *x *= w);
    });
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

/// The thin QR `A = Q R` computed by [`tsqr_in_place`].
///
/// `Q` is kept as Householder reflectors: those of the row blocks are stored
/// in the factorised `A` itself, so [`apply_q`](Self::apply_q) takes `A`
/// back as an argument.
pub struct TallSkinnyQr {
    /// Shape of `A`.
    shape: (usize, usize),
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

/// Thin QR `A = Q R` of `a` (`N × M`) in place, by TSQR.
///
/// The rows are split into `p` blocks, each factorised by faer's sequential
/// QR, in parallel on the current Rayon pool; their `R` factors are then
/// merged pairwise up a binary tree, each level in parallel. `p` is
/// `blocks`, capped so that every block has at least `M` rows; with `p = 1`
/// this is a plain Householder QR. The result depends on `p` only at
/// round-off level.
///
/// On return `a` holds the Householder vectors of the row blocks; read the
/// factors with [`TallSkinnyQr::r`] and [`TallSkinnyQr::apply_q`].
///
/// ```
/// use lobatto_lowrank::linalg::tsqr_in_place;
/// use nalgebra::DMatrix;
///
/// let a = DMatrix::from_fn(400, 5, |i, j| (0.1 * (i * (j + 1)) as f64).sin());
/// let mut f = a.clone();
/// let qr = tsqr_in_place(&mut f, 4);
/// let r = qr.r();
/// let q_r = qr.apply_q(&f, r.as_view()); // Q · [R; 0] = A
/// assert!((q_r - &a).amax() < 1e-12 * a.amax());
/// ```
pub fn tsqr_in_place(a: &mut DMatrix<f64>, blocks: usize) -> TallSkinnyQr {
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
    TallSkinnyQr { shape: (n, m), ranges, coeffs, levels, r }
}

impl TallSkinnyQr {
    /// The `R` factor, `min(N, M) × M` upper trapezoidal.
    pub fn r(&self) -> DMatrix<f64> {
        self.r.clone()
    }

    /// `Q [top; 0]`, `N × c`, for `top` with `min(N, M)` rows; `a` is the
    /// matrix factorised by [`tsqr_in_place`]. `Q` is never formed.
    ///
    /// # Panics
    ///
    /// If `a` does not have the shape of the factorised matrix, or `top`
    /// does not have `min(N, M)` rows.
    pub fn apply_q(&self, a: &DMatrix<f64>, top: DMatrixView<f64>) -> DMatrix<f64> {
        let (n, m) = self.shape;
        assert_eq!(a.shape(), self.shape, "apply_q: `a` is not the factorised matrix");
        assert_eq!(top.nrows(), n.min(m), "apply_q: `top` must have min(N, M) rows");
        let c = top.ncols();

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

/// Thin SVD `a = U diag(σ) Vᵀ` by faer, `σ` decreasing: `U` is `m × k`, `σ`
/// has length `k` and `V` is `n × k`, `k = min(m, n)`. Returns `V`, not
/// `Vᵀ`.
///
/// # Panics
///
/// If the SVD does not converge.
pub fn thin_svd(a: &DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
    let (m, n) = a.shape();
    let k = m.min(n);
    let mut u = DMatrix::zeros(m, k);
    let mut sigma = DVector::zeros(k);
    let mut v = DMatrix::zeros(n, k);
    let thin = ComputeSvdVectors::Thin;
    let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(m, n, thin, thin, Par::Seq, Default::default()));
    svd::svd(
        MatRef::from_column_major_slice(a.as_slice(), m, n),
        DiagMut::from_slice_mut(sigma.as_mut_slice()),
        Some(MatMut::from_column_major_slice_mut(u.as_mut_slice(), m, k)),
        Some(MatMut::from_column_major_slice_mut(v.as_mut_slice(), n, k)),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .expect("thin SVD did not converge");
    (u, sigma, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `thin_svd` on cores `R_A diag(a) R_Bᵀ` built from the QR of
    /// rank-deficient factor matrices (rank 20 of 60, plus round-off-sized
    /// perturbations: many small clustered singular values). Square (both
    /// sides with more nodes than terms) and rectangular (40 right nodes for
    /// 60 terms).
    #[test]
    fn thin_svd_reconstructs_rank_deficient_cores() {
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

            let (u, sigma, v) = thin_svd(&core);
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
