//! The [`LowRankSum`] structure: construction, accessors, evaluation, full
//! grid and assembled field.
//!
//! A [`LowRankSum<P, Q>`] holds a field on a `P + Q` dimensional grid as a
//! sum of `M` products `a_k · ρ_k ⊗ μ_k`, with `ρ_k` a nodal field on the
//! `P`-dimensional left grid and `μ_k` a nodal field on the `Q`-dimensional
//! right grid.
//!
//! # Storage
//!
//! The factors are two column-major matrices, one factor per column:
//! [`rho`](LowRankSum::rho) is `N_left × M` and [`mu`](LowRankSum::mu) is
//! `N_right × M`. The coefficients ([`coefficients`](LowRankSum::coefficients))
//! have length `M`. A sum owns its two grids and the square roots of their
//! mass weights.
//!
//! # Full grid and assembled field
//!
//! [`LowRankSum::full_grid`] is the `P + Q` dimensional grid, left directions
//! first. [`LowRankSum::assemble`] returns the field on it, column-major with
//! the left directions fastest: the node pair `(i, j)` (left index `i`, right
//! index `j`) is at `i + j · N_left`. This is the column-major
//! `N_left × N_right` unfolding matrix.

use lobatto_grid::GridND;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// A field on a `P + Q` dimensional grid stored as a sum of `M` products of a
/// field on a `P`-dimensional left grid and a field on a `Q`-dimensional
/// right grid.
///
/// Norms, Gram matrices and compressions use the discrete `L²` inner product
/// given by the mass weights of each grid ([`GridND::mass_weights`]).
/// Additions concatenate the terms; [`LowRankSum::compress`] returns the
/// canonical form (orthonormal factors, `a_k = σ_k ≥ 0` decreasing).
#[derive(Debug, Clone)]
pub struct LowRankSum<const P: usize, const Q: usize> {
    pub(crate) left: GridND<P>,
    pub(crate) right: GridND<Q>,
    /// Square roots of the mass weights of the left grid.
    pub(crate) sqrt_w_left: DVector<f64>,
    /// Square roots of the mass weights of the right grid.
    pub(crate) sqrt_w_right: DVector<f64>,
    /// Coefficients `a_k`, length `M`.
    pub(crate) a: DVector<f64>,
    /// Left factors, `N_left × M`; column `k` is `ρ_k`.
    pub(crate) rho: DMatrix<f64>,
    /// Right factors, `N_right × M`; column `k` is `μ_k`.
    pub(crate) mu: DMatrix<f64>,
}

impl<const P: usize, const Q: usize> LowRankSum<P, Q> {
    // ── Construction ────────────────────────────────────────────────────────

    /// The empty sum, i.e. the zero field, on the grids `left` and `right`.
    pub fn new(left: GridND<P>, right: GridND<Q>) -> Self {
        let (nl, nr) = (left.total(), right.total());
        Self {
            sqrt_w_left: DVector::from_vec(left.mass_weights()).map(f64::sqrt),
            sqrt_w_right: DVector::from_vec(right.mass_weights()).map(f64::sqrt),
            left,
            right,
            a: DVector::zeros(0),
            rho: DMatrix::zeros(nl, 0),
            mu: DMatrix::zeros(nr, 0),
        }
    }

    /// A sum on the grids of `self` with the given terms.
    pub(crate) fn with_terms(&self, a: DVector<f64>, rho: DMatrix<f64>, mu: DMatrix<f64>) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            sqrt_w_left: self.sqrt_w_left.clone(),
            sqrt_w_right: self.sqrt_w_right.clone(),
            a,
            rho,
            mu,
        }
    }

    /// The empty sum on the same grids as `self`.
    pub fn zeros_like(&self) -> Self {
        self.with_terms(DVector::zeros(0), DMatrix::zeros(self.n_left(), 0), DMatrix::zeros(self.n_right(), 0))
    }

    /// The rank-one field `f(x_1, …, x_P) · g(x_{P+1}, …, x_{P+Q})`.
    ///
    /// `f` is sampled at the stored nodes of `left` ([`GridND::get_x`]), `g` at
    /// those of `right`, and the sum holds the single term `1 · ρ ⊗ μ`. On an
    /// `Odd` (Dirichlet) direction the end nodes are not stored, so the field
    /// is zero there whatever `f` or `g`.
    pub fn new_from_fn(
        left: GridND<P>,
        right: GridND<Q>,
        f: impl Fn([f64; P]) -> f64,
        g: impl Fn([f64; Q]) -> f64,
    ) -> Self {
        let rho: Vec<f64> = left.get_x().into_iter().map(f).collect();
        let mu: Vec<f64> = right.get_x().into_iter().map(g).collect();
        let mut s = Self::new(left, right);
        s.a = DVector::from_element(1, 1.0);
        s.rho = DMatrix::from_vec(rho.len(), 1, rho);
        s.mu = DMatrix::from_vec(mu.len(), 1, mu);
        s
    }

    /// Remove every term: the zero field on the same grids.
    pub fn clear(&mut self) {
        self.a = DVector::zeros(0);
        self.rho = DMatrix::zeros(self.n_left(), 0);
        self.mu = DMatrix::zeros(self.n_right(), 0);
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    /// The left grid.
    pub fn left(&self) -> &GridND<P> {
        &self.left
    }

    /// The right grid.
    pub fn right(&self) -> &GridND<Q> {
        &self.right
    }

    /// Number of stored nodes of the left grid, `N_left`.
    pub fn n_left(&self) -> usize {
        self.left.total()
    }

    /// Number of stored nodes of the right grid, `N_right`.
    pub fn n_right(&self) -> usize {
        self.right.total()
    }

    /// Square roots of the mass weights of the left grid, in its node order.
    ///
    /// The weights themselves are [`GridND::mass_weights`] of [`left`](Self::left).
    pub fn sqrt_w_left(&self) -> &[f64] {
        self.sqrt_w_left.as_slice()
    }

    /// Square roots of the mass weights of the right grid, in its node order.
    ///
    /// The weights themselves are [`GridND::mass_weights`] of [`right`](Self::right).
    pub fn sqrt_w_right(&self) -> &[f64] {
        self.sqrt_w_right.as_slice()
    }

    /// The coefficients `a_k`, length `M`.
    pub fn coefficients(&self) -> &[f64] {
        self.a.as_slice()
    }

    /// The left factors, `N_left × M`: column `k` is `ρ_k`.
    pub fn rho(&self) -> &DMatrix<f64> {
        &self.rho
    }

    /// The right factors, `N_right × M`: column `k` is `μ_k`.
    pub fn mu(&self) -> &DMatrix<f64> {
        &self.mu
    }

    /// The left factor `ρ_k`, one value per stored node of the left grid.
    pub fn rho_k(&self, k: usize) -> &[f64] {
        let n = self.rho.nrows();
        &self.rho.as_slice()[k * n..(k + 1) * n]
    }

    /// The right factor `μ_k`, one value per stored node of the right grid.
    pub fn mu_k(&self, k: usize) -> &[f64] {
        let n = self.mu.nrows();
        &self.mu.as_slice()[k * n..(k + 1) * n]
    }

    /// Number of terms `M`.
    pub fn len(&self) -> usize {
        self.a.len()
    }

    /// Whether the sum has no term (the zero field).
    pub fn is_empty(&self) -> bool {
        self.a.is_empty()
    }

    /// Whether `other` lives on equal grids.
    pub fn same_grids(&self, other: &Self) -> bool {
        self.left == other.left && self.right == other.right
    }

    /// The field at the node pair `(i, j)`, `Σ_k a_k ρ_k(i) μ_k(j)`.
    ///
    /// `i` is a flat left node index, `j` a flat right node index.
    pub fn value(&self, i: usize, j: usize) -> f64 {
        (0..self.len())
            .map(|k| self.a[k] * self.rho[(i, k)] * self.mu[(j, k)])
            .sum()
    }

    /// The field at the points `xs` of `R^N`, `N = P + Q`, by interpolation,
    /// written to `out`: `out[i]` is
    /// `Σ_k a_k ρ_k(x_1, …, x_P) μ_k(x_{P+1}, …, x_N)` at `xs[i]`.
    ///
    /// Each factor is interpolated on its grid by the extended evaluation of
    /// `lobatto-grid` ([`GridND::eval_extended_plan`]): the nodal interpolant
    /// inside the domain; outside, each direction is folded back by its
    /// extension (periodic wrap, even reflection, odd reflection with a sign
    /// flip). `N` must equal `P + Q` (checked at compile time):
    ///
    /// ```
    /// use lobatto_grid::{Extension, GridND};
    /// use lobatto_lowrank::LowRankSum;
    ///
    /// let grid = || GridND::<1>::new([4], [4], [1.0], [0.0], [Extension::Even]);
    /// let sum = LowRankSum::new_from_fn(grid(), grid(), |[x]| x * x, |[y]| 1.0 - y);
    /// let xs = [[0.3, 0.6], [0.5, 0.1]];
    /// let mut out = [0.0; 2];
    /// sum.get_values(&xs, &mut out);
    /// assert!((out[0] - 0.09 * 0.4).abs() < 1e-14);
    /// assert!((out[1] - 0.25 * 0.9).abs() < 1e-14);
    /// ```
    ///
    /// # Panics
    ///
    /// If `out.len() != xs.len()`.
    pub fn get_values<const N: usize>(&self, xs: &[[f64; N]], out: &mut [f64]) {
        const { assert!(N == P + Q, "a point of the full grid has dimension P + Q") };
        assert_eq!(
            out.len(),
            xs.len(),
            "get_values: {} points but an output of length {}",
            xs.len(),
            out.len()
        );
        out.fill(0.0);
        if self.is_empty() || xs.is_empty() {
            return;
        }
        let (xl, xr): (Vec<[f64; P]>, Vec<[f64; Q]>) = xs
            .iter()
            .map(|x| {
                (
                    std::array::from_fn(|d| x[d]),
                    std::array::from_fn(|d| x[P + d]),
                )
            })
            .unzip();
        let plan_l = self.left.eval_extended_plan(&xl);
        let plan_r = self.right.eval_extended_plan(&xr);
        let (mut rho_x, mut mu_x) = (vec![0.0; xs.len()], vec![0.0; xs.len()]);
        for k in 0..self.len() {
            self.left.eval_with_plan(&plan_l, self.rho_k(k), &mut rho_x);
            self.right.eval_with_plan(&plan_r, self.mu_k(k), &mut mu_x);
            let a = self.a[k];
            for ((o, &r), &m) in out.iter_mut().zip(&rho_x).zip(&mu_x) {
                *o += a * r * m;
            }
        }
    }

    // ── Full grid ───────────────────────────────────────────────────────────

    /// The full `N = P + Q` dimensional grid: the left directions, then the
    /// right ones. `N` must equal `P + Q` (checked at compile time):
    ///
    /// ```
    /// use lobatto_grid::{Extension, GridND};
    /// use lobatto_lowrank::LowRankSum;
    ///
    /// let grid = || GridND::<2>::new([2, 3], [3, 2], [1.0, 2.0], [0.0, -1.0], [Extension::Periodic; 2]);
    /// let sum = LowRankSum::new(grid(), grid());
    /// let g: GridND<4> = sum.full_grid();
    ///
    /// // Left nodes fastest: pair (i, j) sits at i + j·N_left.
    /// assert_eq!(g.total(), sum.n_left() * sum.n_right());
    /// assert_eq!(g.strides()[2], sum.n_left());
    /// ```
    ///
    /// A wrong dimension does not build:
    ///
    /// ```compile_fail
    /// # use lobatto_grid::{Extension, GridND};
    /// # use lobatto_lowrank::LowRankSum;
    /// # let grid = || GridND::<2>::new([2, 3], [3, 2], [1.0, 2.0], [0.0, -1.0], [Extension::Periodic; 2]);
    /// let sum = LowRankSum::new(grid(), grid());
    /// let g: GridND<5> = sum.full_grid();
    /// ```
    pub fn full_grid<const N: usize>(&self) -> GridND<N> {
        const { assert!(N == P + Q, "the full grid has dimension P + Q") };
        GridND::from_grids(std::array::from_fn(|d| {
            if d < P {
                self.left.axis(d).clone()
            } else {
                self.right.axis(d - P).clone()
            }
        }))
    }

    /// The full field `Σ_k a_k ρ_k(i) μ_k(j)` on [`full_grid`](Self::full_grid),
    /// in the layout of the module docs: `(ρ diag(a)) μᵀ`, length
    /// `N_left · N_right`, cost `M · N_left · N_right`. All zeros for an
    /// empty sum.
    pub fn assemble(&self) -> Vec<f64> {
        let mut rho_a = self.rho.clone();
        for (mut col, &a) in rho_a.column_iter_mut().zip(self.a.iter()) {
            col *= a;
        }
        (rho_a * self.mu.transpose()).data.into()
    }
}

/// `m[i, k] *= w[i]` for every column `k`, the columns in parallel.
pub(crate) fn scale_rows(m: &mut DMatrix<f64>, w: &DVector<f64>) {
    let n = m.nrows().max(1);
    m.as_mut_slice().par_chunks_mut(n).for_each(|col| {
        col.iter_mut().zip(w.iter()).for_each(|(x, w)| *x *= w);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use lobatto_grid::Extension;

    /// A 2 + 1 sum with mixed extensions on both sides, no terms.
    fn empty_sum() -> LowRankSum<2, 1> {
        let left = GridND::<2>::new(
            [2, 3],
            [3, 2],
            [1.0, 2.0],
            [0.0, -1.0],
            [Extension::Periodic, Extension::Odd],
        );
        let right = GridND::<1>::new([3], [4], [1.5], [0.5], [Extension::Even]);
        LowRankSum::new(left, right)
    }

    /// Append `a · f ⊗ g`, sampling `f` and `g` on the left and right nodes.
    fn push_sampled(
        s: &mut LowRankSum<2, 1>,
        a: f64,
        f: impl Fn([f64; 2]) -> f64,
        g: impl Fn([f64; 1]) -> f64,
    ) {
        let mut t = LowRankSum::new_from_fn(s.left().clone(), s.right().clone(), f, g);
        t.scale(a);
        s.add_assign(&t);
    }

    fn f0(x: [f64; 2]) -> f64 {
        x[0].sin() + x[1]
    }
    fn g0(y: [f64; 1]) -> f64 {
        y[0].cos()
    }
    fn f1(x: [f64; 2]) -> f64 {
        x[0] * x[1]
    }
    fn g1(y: [f64; 1]) -> f64 {
        y[0] * y[0]
    }

    fn two_term_sum() -> LowRankSum<2, 1> {
        let mut s = empty_sum();
        push_sampled(&mut s, 2.0, f0, g0);
        push_sampled(&mut s, -0.5, f1, g1);
        s
    }

    #[test]
    fn storage_shapes_and_column_slices() {
        let s = two_term_sum();
        assert_eq!(s.len(), 2);
        assert_eq!(s.coefficients(), &[2.0, -0.5]);
        assert_eq!(s.rho().shape(), (s.n_left(), 2));
        assert_eq!(s.mu().shape(), (s.n_right(), 2));
        let x = s.left().get_x();
        for (i, xi) in x.iter().enumerate() {
            assert_eq!(s.rho_k(1)[i], f1(*xi));
        }
        let mut e = s.clone();
        e.clear();
        assert!(e.is_empty());
        assert_eq!(e.rho().shape(), (s.n_left(), 0));
    }

    #[test]
    fn matches_separable_function_on_full_grid() {
        let s = two_term_sum();
        let p = s.assemble();
        let x = s.full_grid::<3>().get_x();
        assert_eq!(p.len(), x.len());
        for (pv, xv) in p.iter().zip(&x) {
            let (l, r) = ([xv[0], xv[1]], [xv[2]]);
            let exact = 2.0 * f0(l) * g0(r) - 0.5 * f1(l) * g1(r);
            assert!((pv - exact).abs() < 1e-14, "{pv} vs {exact} at {xv:?}");
        }
    }

    #[test]
    fn matches_value_at_every_pair() {
        let s = two_term_sum();
        let p = s.assemble();
        let nl = s.n_left();
        for j in 0..s.n_right() {
            for i in 0..nl {
                assert!((p[i + j * nl] - s.value(i, j)).abs() < 1e-14, "(i, j) = ({i}, {j})");
            }
        }
    }

    #[test]
    fn empty_sum_is_zero_field() {
        let s = empty_sum();
        let p = s.assemble();
        assert_eq!(p.len(), s.n_left() * s.n_right());
        assert!(p.iter().all(|&v| v == 0.0));
    }

    /// Left: periodic × Neumann, right: Neumann; order 4 everywhere.
    fn grids_2_1() -> (GridND<2>, GridND<1>) {
        let left = GridND::<2>::new(
            [3, 2],
            [4, 4],
            [2.0, 1.5],
            [-1.0, 0.5],
            [Extension::Periodic, Extension::Even],
        );
        let right = GridND::<1>::new([2], [4], [1.0], [0.0], [Extension::Even]);
        (left, right)
    }

    #[test]
    fn new_from_fn_is_rank_one_and_matches_at_nodes() {
        let (left, right) = grids_2_1();
        let s = LowRankSum::new_from_fn(left, right, f0, g0);
        assert_eq!(s.len(), 1);
        assert_eq!(s.coefficients()[0], 1.0);
        let xs = s.full_grid::<3>().get_x();
        let mut out = vec![f64::NAN; xs.len()];
        s.get_values(&xs, &mut out);
        for (x, v) in xs.iter().zip(&out) {
            let exact = f0([x[0], x[1]]) * g0([x[2]]);
            assert!((v - exact).abs() < 1e-13, "at {x:?}");
        }
    }

    #[test]
    fn get_values_is_exact_for_polynomials_off_the_nodes() {
        // Interpolation of order r = 4 is exact for degree ≤ 4 in the Neumann
        // directions; keep the periodic direction constant, where polynomials
        // are not periodic.
        let (left, right) = grids_2_1();
        let f = |x: [f64; 2]| 1.0 + x[1].powi(4) - 2.0 * x[1];
        let g = |y: [f64; 1]| y[0].powi(3) - 0.5;
        let s = LowRankSum::new_from_fn(left, right, f, g);
        let xs: Vec<[f64; 3]> = (0..20)
            .map(|k| {
                let t = k as f64 / 19.0;
                [-1.0 + 2.0 * t, 0.5 + 1.5 * (1.0 - t), (0.37 + t).fract()]
            })
            .collect();
        let mut out = vec![f64::NAN; xs.len()];
        s.get_values(&xs, &mut out);
        for (x, v) in xs.iter().zip(&out) {
            let exact = f([x[0], x[1]]) * g([x[2]]);
            assert!((v - exact).abs() < 1e-12, "at {x:?}");
        }
    }

    #[test]
    fn get_values_adds_over_terms() {
        let (left, right) = grids_2_1();
        let a = LowRankSum::new_from_fn(left.clone(), right.clone(), f0, g0);
        let b = LowRankSum::new_from_fn(left, right, f1, g1);
        let ab = a.add(&b);
        let x = [[0.3, 1.1, 0.8]];
        let (mut va, mut vb, mut vab) = ([0.0], [0.0], [0.0]);
        a.get_values(&x, &mut va);
        b.get_values(&x, &mut vb);
        ab.get_values(&x, &mut vab);
        assert!((vab[0] - (va[0] + vb[0])).abs() < 1e-14);
    }

    #[test]
    fn get_values_wraps_periodic_direction() {
        let (left, right) = grids_2_1();
        let s = LowRankSum::new_from_fn(left, right, f0, g0);
        // Period L = 2 in direction 0.
        let mut out = [0.0; 2];
        s.get_values(&[[0.3, 1.1, 0.8], [0.3 + 2.0, 1.1, 0.8]], &mut out);
        assert!((out[0] - out[1]).abs() < 1e-13);
    }

    #[test]
    fn get_values_reflects_odd_direction_with_sign_flip() {
        let left = GridND::<1>::new([3], [4], [1.0], [0.0], [Extension::Odd]);
        let right = GridND::<1>::new([2], [3], [1.0], [0.0], [Extension::Even]);
        let s = LowRankSum::new_from_fn(left, right, |[x]| (std::f64::consts::PI * x).sin(), |[y]| 1.0 + y);
        // Odd reflection about x = 0: u(-x) = -u(x).
        let (x, y) = (0.27, 0.4);
        let mut out = [0.0; 3];
        s.get_values(&[[-x, y], [x, y], [0.0, y]], &mut out);
        assert!((out[0] + out[1]).abs() < 1e-13);
        assert!(out[2].abs() < 1e-14);
    }

    #[test]
    fn get_values_batch_matches_points_one_at_a_time() {
        let (left, right) = grids_2_1();
        let a = LowRankSum::new_from_fn(left.clone(), right.clone(), f0, g0);
        let s = a.add(&LowRankSum::new_from_fn(left, right, f1, g1));
        let xs: Vec<[f64; 3]> = (0..50)
            .map(|k| {
                let t = k as f64 / 49.0;
                // Spans the domain and goes past it in every direction.
                [-2.0 + 5.0 * t, 0.0 + 2.5 * (1.0 - t), -0.5 + 2.0 * (0.37 + t).fract()]
            })
            .collect();
        let mut out = vec![f64::NAN; xs.len()];
        s.get_values(&xs, &mut out);
        for (x, v) in xs.iter().zip(&out) {
            let mut one = [f64::NAN];
            s.get_values(&[*x], &mut one);
            assert!((v - one[0]).abs() < 1e-14, "at {x:?}");
        }
    }

    #[test]
    fn get_values_of_empty_sum_and_empty_batch() {
        let (left, right) = grids_2_1();
        let empty = LowRankSum::new(left.clone(), right.clone());
        let mut out = vec![f64::NAN; 3];
        empty.get_values(&[[0.1, 0.9, 0.2]; 3], &mut out);
        assert!(out.iter().all(|&v| v == 0.0));

        let s = LowRankSum::new_from_fn(left, right, f0, g0);
        let mut none: [f64; 0] = [];
        s.get_values::<3>(&[], &mut none);
    }

    #[test]
    #[should_panic(expected = "get_values: 2 points but an output of length 1")]
    fn get_values_checks_output_length() {
        let (left, right) = grids_2_1();
        let s = LowRankSum::new_from_fn(left, right, f0, g0);
        s.get_values(&[[0.1, 0.9, 0.2]; 2], &mut [0.0]);
    }

    #[test]
    fn assembly_commutes_with_addition() {
        let a = two_term_sum();
        let mut b = empty_sum();
        push_sampled(&mut b, 3.0, |x| x[1].exp(), |y| y[0] - 1.0);
        let (pa, pb, pab) = (a.assemble(), b.assemble(), a.add(&b).assemble());
        for ((u, v), w) in pa.iter().zip(&pb).zip(&pab) {
            assert!((u + v - w).abs() < 1e-14);
        }
    }
}
