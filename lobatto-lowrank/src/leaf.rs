//! [`Leaf`]: `r` functions on a grid, stored by their nodal values.
//!
//! A leaf holds a grid of dimension `P`, the square roots of its mass
//! weights, and an `N × r` basis matrix whose column `k` is the `k`-th
//! function at the `N` stored nodes, in the order of `GridND::get_x`. Inner
//! products are the discrete `L²` inner products given by the mass weights.
//! The two sides of a [`LowRankSum`](crate::LowRankSum) are leaves.

use lobatto_grid::GridND;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::linalg::{scale_rows, tsqr_in_place};

/// `r` functions on a grid of dimension `P`, stored by their nodal values.
#[derive(Debug, Clone)]
pub struct Leaf<const P: usize> {
    pub(crate) grid: GridND<P>,
    /// Square roots of the mass weights of the grid.
    pub(crate) sqrt_w: DVector<f64>,
    /// The basis, `N × r`; column `k` is the `k`-th function.
    pub(crate) u: DMatrix<f64>,
}

impl<const P: usize> Leaf<P> {
    /// A leaf with no function on `grid`.
    pub fn new(grid: GridND<P>) -> Self {
        let n = grid.total();
        Self {
            sqrt_w: DVector::from_vec(grid.mass_weights()).map(f64::sqrt),
            u: DMatrix::zeros(n, 0),
            grid,
        }
    }

    /// A leaf on `grid` with the basis `u` (`N × r`, nodal values).
    ///
    /// # Panics
    ///
    /// If `u` does not have one row per stored node of `grid`.
    pub fn with_basis(grid: GridND<P>, u: DMatrix<f64>) -> Self {
        assert_eq!(u.nrows(), grid.total(), "basis rows differ from the number of grid nodes");
        let mut leaf = Self::new(grid);
        leaf.u = u;
        leaf
    }

    /// A leaf on the same grid with the basis `u`.
    pub(crate) fn with_same_grid(&self, u: DMatrix<f64>) -> Self {
        Self { grid: self.grid.clone(), sqrt_w: self.sqrt_w.clone(), u }
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    /// The grid.
    pub fn grid(&self) -> &GridND<P> {
        &self.grid
    }

    /// Square roots of the mass weights of the grid, in its node order.
    pub fn sqrt_w(&self) -> &[f64] {
        self.sqrt_w.as_slice()
    }

    /// The basis, `N × r`: column `k` is the `k`-th function.
    pub fn basis(&self) -> &DMatrix<f64> {
        &self.u
    }

    /// The `k`-th function, one value per stored node.
    pub fn column(&self, k: usize) -> &[f64] {
        let n = self.u.nrows();
        &self.u.as_slice()[k * n..(k + 1) * n]
    }

    /// Number of functions `r`.
    pub fn rank(&self) -> usize {
        self.u.ncols()
    }

    /// Number of stored nodes `N`.
    pub fn n_nodes(&self) -> usize {
        self.u.nrows()
    }

    // ── Building ────────────────────────────────────────────────────────────

    /// Append the function `col` (one value per stored node).
    ///
    /// # Panics
    ///
    /// If `col` does not have one value per stored node.
    pub fn push(&mut self, col: &[f64]) {
        assert_eq!(col.len(), self.n_nodes(), "function length differs from the number of grid nodes");
        self.u.extend(col.iter().copied());
    }

    /// Append the functions of `other`, a leaf on the same grid.
    ///
    /// # Panics
    ///
    /// If `other` does not have the same number of nodes.
    pub fn extend(&mut self, other: &Self) {
        assert_eq!(other.n_nodes(), self.n_nodes(), "leaves with different numbers of nodes");
        self.u.extend(other.u.iter().copied());
    }

    /// The functions of `self`, then those of `other`, a leaf on the same
    /// grid.
    ///
    /// # Panics
    ///
    /// If `other` does not have the same number of nodes.
    pub fn concat(&self, other: &Self) -> Self {
        assert_eq!(other.n_nodes(), self.n_nodes(), "leaves with different numbers of nodes");
        let u = [self.u.as_slice(), other.u.as_slice()].concat();
        self.with_same_grid(DMatrix::from_vec(self.n_nodes(), self.rank() + other.rank(), u))
    }

    // ── Weighted inner products ─────────────────────────────────────────────

    /// Weighted Gram matrix, `r × r`: `G_kl = Σ_i w_i u_k(i) u_l(i)`.
    /// Exactly symmetric. Cost `N · r²`.
    pub fn gram(&self) -> DMatrix<f64> {
        weighted_gram(&self.sqrt_w, &self.u)
    }

    /// Weighted cross Gram matrix with `other`, a leaf on the same grid,
    /// `r × r'`: `C_kl = Σ_i w_i u_k(i) u'_l(i)`. Cost `N · r · r'`.
    pub fn cross_gram(&self, other: &Self) -> DMatrix<f64> {
        weighted_cross_gram(&self.sqrt_w, &self.u, &other.u)
    }

    /// The weighted integrals of the functions, `Σ_i w_i u_k(i)`, length `r`.
    pub fn integrals(&self) -> DVector<f64> {
        self.u.tr_mul(&self.sqrt_w.component_mul(&self.sqrt_w))
    }

    /// Replace the basis by an orthonormal one in the weighted inner product
    /// spanning the same functions, and return `R` (`min(N, r) × r`, upper
    /// trapezoidal) such that `old basis = new basis · R`.
    ///
    /// Thin QR of the weight-scaled basis by [`tsqr_in_place`] with `blocks`
    /// row blocks.
    pub fn orthonormalize(&mut self, blocks: usize) -> DMatrix<f64> {
        scale_rows(&mut self.u, &self.sqrt_w);
        let qr = tsqr_in_place(&mut self.u, blocks);
        let r = qr.r();
        let k = r.nrows();
        let mut q = qr.apply_q(&self.u, DMatrix::<f64>::identity(k, k).as_view());
        scale_rows(&mut q, &self.sqrt_w.map(|w| 1.0 / w));
        self.u = q;
        r
    }

    // ── Evaluation ──────────────────────────────────────────────────────────

    /// The functions at the points `xs` by interpolation, `n_points × r`:
    /// entry `(i, k)` is the `k`-th function at `xs[i]`.
    ///
    /// Uses the extended evaluation of `lobatto-grid`
    /// ([`GridND::eval_extended_plan`]): the nodal interpolant inside the
    /// domain, folded back by each direction's extension outside.
    pub fn eval(&self, xs: &[[f64; P]]) -> DMatrix<f64> {
        let mut out = DMatrix::zeros(xs.len(), self.rank());
        if xs.is_empty() || self.rank() == 0 {
            return out;
        }
        let plan = self.grid.eval_extended_plan(xs);
        for (k, mut col) in out.column_iter_mut().enumerate() {
            self.grid.eval_with_plan(&plan, self.column(k), col.as_mut_slice());
        }
        out
    }

    // ── Maps ────────────────────────────────────────────────────────────────

    /// `f(state, u_k)` on every function in place, in parallel over the
    /// functions, in no particular order; `init` makes one state per parallel
    /// task.
    pub fn map_in_place_with<T>(&mut self, init: impl Fn() -> T + Sync, f: impl Fn(&mut T, &mut [f64]) + Sync) {
        let n = self.n_nodes().max(1);
        self.u.as_mut_slice().par_chunks_mut(n).for_each_init(&init, &f);
    }

    /// A leaf on the same grid whose `k`-th function is written by
    /// `f(state, u_k, zeroed output)`, in parallel over the functions;
    /// `init` makes one state per parallel task.
    pub fn map_with<T>(&self, init: impl Fn() -> T + Sync, f: impl Fn(&mut T, &[f64], &mut [f64]) + Sync) -> Self {
        let n = self.n_nodes().max(1);
        let mut out = DMatrix::zeros(self.n_nodes(), self.rank());
        out.as_mut_slice()
            .par_chunks_mut(n)
            .zip(self.u.as_slice().par_chunks(n))
            .for_each_init(&init, |state, (dst, col)| f(state, col, dst));
        self.with_same_grid(out)
    }
}

/// The weighted cross Gram matrix `C_kl = Σ_i s_i² F(i, k) G(i, l)` of the
/// columns of `f` (`N × M`) and `g` (`N × M'`), `M × M'`, for the square-root
/// weights `s`.
fn weighted_cross_gram(s: &DVector<f64>, f: &DMatrix<f64>, g: &DMatrix<f64>) -> DMatrix<f64> {
    let mut wg = g.clone();
    scale_rows(&mut wg, &s.component_mul(s));
    f.transpose() * wg
}

/// The weighted Gram matrix of the columns of `f`, `M × M`, made exactly
/// symmetric by copying its upper triangle into the lower one.
fn weighted_gram(s: &DVector<f64>, f: &DMatrix<f64>) -> DMatrix<f64> {
    let mut g = weighted_cross_gram(s, f, f);
    g.fill_lower_triangle_with_upper_triangle();
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use lobatto_grid::Extension;

    /// Periodic × Neumann on `[-1, 1) × [0.5, 2]`, 3 × 2 elements of order 4.
    fn grid() -> GridND<2> {
        GridND::<2>::new([3, 2], [4, 4], [2.0, 1.5], [-1.0, 0.5], [Extension::Periodic, Extension::Even])
    }

    /// Four functions sampled at the nodes, the last a combination of the
    /// first two (rank 3).
    fn leaf() -> Leaf<2> {
        let g = grid();
        let x = g.get_x();
        let fs: [fn([f64; 2]) -> f64; 3] = [|x| x[0].sin() + x[1], |x| x[0] * x[1], |x| (x[1] - 1.0).powi(2)];
        let mut l = Leaf::new(g);
        for f in fs {
            l.push(&x.iter().map(|&p| f(p)).collect::<Vec<_>>());
        }
        let dep: Vec<f64> = l.column(0).iter().zip(l.column(1)).map(|(a, b)| 2.0 * a - b).collect();
        l.push(&dep);
        l
    }

    #[test]
    fn gram_matches_direct_weighted_sums() {
        let l = leaf();
        let w = l.grid().mass_weights();
        let g = l.gram();
        for k in 0..4 {
            for m in 0..4 {
                let exact: f64 = (0..l.n_nodes()).map(|i| w[i] * l.column(k)[i] * l.column(m)[i]).sum();
                assert!((g[(k, m)] - exact).abs() < 1e-13 * (1.0 + exact.abs()));
                assert_eq!(g[(k, m)], g[(m, k)]);
            }
        }
        assert_eq!(l.cross_gram(&l), weighted_cross_gram(&l.sqrt_w, &l.u, &l.u));
    }

    #[test]
    fn integrals_integrate_low_degree_polynomials_exactly() {
        // ∫∫ x₁ over [-1, 1) × [0.5, 2] = 2 · (2² − 0.5²) / 2 = 3.75.
        let g = grid();
        let x1: Vec<f64> = g.get_x().iter().map(|p| p[1]).collect();
        let l = Leaf::with_basis(g, DMatrix::from_vec(x1.len(), 1, x1));
        assert!((l.integrals()[0] - 3.75).abs() < 1e-13);
    }

    #[test]
    fn orthonormalize_spans_the_same_functions() {
        for blocks in [1, 3] {
            let mut l = leaf();
            let old = l.basis().clone();
            let r = l.orthonormalize(blocks);
            let id = DMatrix::<f64>::identity(l.rank(), l.rank());
            assert!((l.gram() - id).amax() < 1e-12, "{blocks} blocks: not orthonormal");
            assert!((l.basis() * &r - &old).amax() < 1e-12 * old.amax(), "{blocks} blocks: old ≠ new · R");
        }
    }

    #[test]
    fn eval_interpolates_at_the_nodes_and_wraps_periodically() {
        let l = leaf();
        let x = l.grid().get_x();
        let at_nodes = l.eval(&x);
        for k in 0..l.rank() {
            for (i, v) in at_nodes.column(k).iter().enumerate() {
                assert!((v - l.column(k)[i]).abs() < 1e-13);
            }
        }
        // Period 2 in direction 0.
        let e = l.eval(&[[0.3, 1.1], [2.3, 1.1]]);
        assert!((e.row(0) - e.row(1)).amax() < 1e-13);
    }

    #[test]
    fn concat_push_and_maps() {
        let l = leaf();
        let mut e = Leaf::new(grid());
        e.extend(&l);
        assert_eq!(e.basis(), l.basis());
        assert_eq!(l.concat(&l).rank(), 8);
        let mut m = l.clone();
        m.map_in_place_with(|| (), |_, c| c.iter_mut().for_each(|v| *v *= 2.0));
        let n = l.map_with(|| (), |_, src, dst| dst.iter_mut().zip(src).for_each(|(d, s)| *d = 2.0 * s));
        assert_eq!(m.basis(), &(l.basis() * 2.0));
        assert_eq!(n.basis(), m.basis());
    }

    #[test]
    #[should_panic(expected = "basis rows differ from the number of grid nodes")]
    fn with_basis_checks_rows() {
        Leaf::with_basis(grid(), DMatrix::zeros(3, 1));
    }
}
