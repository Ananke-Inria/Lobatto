//! Vector-space operations on [`LowRankSum`]s: scaling, addition, weighted
//! `L²` Gram matrices, norms and inner products.
//!
//! Scaling and addition only touch the coefficients and factor columns: an
//! addition concatenates the terms and never compresses. The inner products
//! are computed from the factor matrices, without assembling the field: the
//! weighted Gram matrices are matrix products `Fᵀ (W G)` and the contractions
//! `aᵀ (G^L ∘ G^R) b` (`∘` the entry-wise product).

use nalgebra::{DMatrix, DVector};

use crate::lowrank::LowRankSum;

impl<const P: usize, const Q: usize> LowRankSum<P, Q> {
    // ── Scaling and addition ────────────────────────────────────────────────

    /// Multiply the field by `c` (the coefficients `a_k`, not the factors).
    pub fn scale(&mut self, c: f64) {
        self.a *= c;
    }

    /// The sum of two fields: the term lists concatenated, no compression.
    ///
    /// # Panics
    ///
    /// If the two sums are not on equal grids (see [`Self::same_grids`]).
    pub fn add(&self, other: &Self) -> Self {
        assert!(self.same_grids(other), "adding sums on different grids");
        // Column-major: the concatenated storage is self's, then other's.
        let cat = |x: &[f64], y: &[f64]| [x, y].concat();
        let m = self.len() + other.len();
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            sqrt_w_left: self.sqrt_w_left.clone(),
            sqrt_w_right: self.sqrt_w_right.clone(),
            inv_sqrt_w_left: self.inv_sqrt_w_left.clone(),
            inv_sqrt_w_right: self.inv_sqrt_w_right.clone(),
            a: DVector::from_vec(cat(self.a.as_slice(), other.a.as_slice())),
            rho: DMatrix::from_vec(self.n_left(), m, cat(self.rho.as_slice(), other.rho.as_slice())),
            mu: DMatrix::from_vec(self.n_right(), m, cat(self.mu.as_slice(), other.mu.as_slice())),
        }
    }

    /// In-place version of [`LowRankSum::add`].
    ///
    /// # Panics
    ///
    /// If the two sums are not on equal grids (see [`Self::same_grids`]).
    pub fn add_assign(&mut self, other: &Self) {
        assert!(self.same_grids(other), "adding sums on different grids");
        // Column-major: nalgebra's `extend` appends columns at the end of the
        // storage (amortised Vec growth, no zero fill), so building a sum term
        // by term stays linear.
        self.a.extend(other.a.iter().copied());
        self.rho.extend(other.rho.iter().copied());
        self.mu.extend(other.mu.iter().copied());
    }

    // ── Weighted inner products ─────────────────────────────────────────────

    /// Weighted Gram matrix of the left factors, `M × M`.
    ///
    /// $$
    /// G^L_{kl} = \langle \rho_k, \rho_l \rangle_w = \sum_i w^L_i \, \rho_k(i) \, \rho_l(i),
    /// $$
    ///
    /// with the left mass weights `w^L`: the discrete `L²` inner products of
    /// the left factors, computed as the matrix product `(S ρ)ᵀ (S ρ)` with
    /// `S` the square-root weights ([`sqrt_w_left`](Self::sqrt_w_left)).
    /// Exactly symmetric; the identity when the sum is in canonical form.
    /// Cost `N_left · M²`.
    pub fn gram_left(&self) -> DMatrix<f64> {
        weighted_gram(self.sqrt_w_left.as_slice(), &self.rho)
    }

    /// Weighted Gram matrix of the right factors, `M × M`.
    ///
    /// $G^R_{kl} = \langle \mu_k, \mu_l \rangle_w$ with the right mass weights
    /// ([`sqrt_w_right`](Self::sqrt_w_right) squared); see
    /// [`gram_left`](Self::gram_left).
    pub fn gram_right(&self) -> DMatrix<f64> {
        weighted_gram(self.sqrt_w_right.as_slice(), &self.mu)
    }

    /// Squared weighted `L²` norm of the field, without assembling it.
    ///
    /// $$
    /// \|\rho\|^2 = \sum_{i,j} w^L_i w^R_j \, \rho(i, j)^2
    /// = \sum_{k,l} a_k a_l \, G^L_{kl} \, G^R_{kl},
    /// $$
    ///
    /// the weighted Frobenius norm of the unfolding, from the two Gram
    /// matrices ([`gram_left`](Self::gram_left),
    /// [`gram_right`](Self::gram_right)): `aᵀ (G^L ∘ G^R) a`. Cost
    /// `(N_left + N_right) · M²`, against `M · N_left · N_right` for
    /// assembling. Zero for an empty sum.
    ///
    /// Accuracy: the sum runs over terms that may cancel, so the absolute
    /// error is about machine precision times
    /// $\sum_{k,l} |a_k a_l G^L_{kl} G^R_{kl}|$. When the terms nearly cancel
    /// (a field much smaller than its terms), the relative error grows
    /// accordingly; compress first for an accurate small norm. A tiny
    /// negative result from rounding is clamped to zero.
    pub fn norm_squared(&self) -> f64 {
        let h = self.gram_left().component_mul(&self.gram_right());
        self.a.dot(&(h * &self.a)).max(0.0)
    }

    /// Weighted `L²` inner product with another sum on the same grids, without
    /// assembling either field.
    ///
    /// With `self = Σ_k a_k ρ_k ⊗ μ_k` and `other = Σ_l b_l ρ'_l ⊗ μ'_l`,
    ///
    /// $$
    /// \langle \rho, \rho' \rangle = \sum_{k,l} a_k b_l \, C^L_{kl} \, C^R_{kl},
    /// \qquad C^L_{kl} = \langle \rho_k, \rho'_l \rangle_w, \quad
    /// C^R_{kl} = \langle \mu_k, \mu'_l \rangle_w,
    /// $$
    ///
    /// the cross Gram matrices (`M × M'`) in the weighted inner products of
    /// each grid, computed as matrix products, then `aᵀ (C^L ∘ C^R) b`. Cost
    /// `(N_left + N_right) · M · M'`. Zero if either sum is empty.
    /// `self.inner(self)` equals [`norm_squared`](Self::norm_squared) up to
    /// rounding.
    ///
    /// # Panics
    ///
    /// If the two sums are not on equal grids (see [`Self::same_grids`]).
    pub fn inner(&self, other: &Self) -> f64 {
        assert!(self.same_grids(other), "inner product of sums on different grids");
        let cl = weighted_cross_gram(self.sqrt_w_left.as_slice(), &self.rho, &other.rho);
        let cr = weighted_cross_gram(self.sqrt_w_right.as_slice(), &self.mu, &other.mu);
        self.a.dot(&(cl.component_mul(&cr) * &other.a))
    }

    /// Squared weighted `L²` distance to another sum on the same grids,
    /// without assembling either field.
    ///
    /// $$
    /// \|\rho - \rho'\|^2 = \|\rho\|^2 - 2 \langle \rho, \rho' \rangle + \|\rho'\|^2,
    /// $$
    ///
    /// from [`norm_squared`](Self::norm_squared) and [`inner`](Self::inner).
    ///
    /// Accuracy: the three terms cancel when the fields are close, so the
    /// absolute error is about machine precision times `‖ρ‖² + ‖ρ'‖²`, and
    /// the relative error grows like `ε · ‖ρ‖² / ‖ρ − ρ'‖²`. A distance below
    /// about `1e-8 · ‖ρ‖` is not resolved; compressing the difference gives an
    /// accurate one. A tiny negative result from rounding is clamped to zero.
    ///
    /// # Panics
    ///
    /// If the two sums are not on equal grids (see [`Self::same_grids`]).
    pub fn distance_squared(&self, other: &Self) -> f64 {
        let d = self.norm_squared() - 2.0 * self.inner(other) + other.norm_squared();
        d.max(0.0)
    }
}

/// `C = Fᵀ (W G)`, i.e. `C_kl = Σ_i w_i F(i, k) G(i, l)`: the weighted cross
/// Gram matrix of the columns of `f` (`N × M`) and `g` (`N × M'`), `M × M'`,
/// from the square-root weights `s`: `(S F)ᵀ (S G) = Fᵀ (S² G)`.
///
/// One row scaling of `g` by `s²`, then one matrix product (nalgebra's
/// `gemm`, which uses `matrixmultiply` for all dimensions above 5).
fn weighted_cross_gram(s: &[f64], f: &DMatrix<f64>, g: &DMatrix<f64>) -> DMatrix<f64> {
    let mut wg = g.clone();
    for mut col in wg.column_iter_mut() {
        for (x, &si) in col.iter_mut().zip(s) {
            *x *= si * si;
        }
    }
    f.transpose() * wg
}

/// `G = (S F)ᵀ (S F)`, the weighted Gram matrix of the columns of `f`, `M × M`,
/// from the square-root weights `s`.
///
/// The product is symmetric only up to rounding; the upper triangle is
/// mirrored into the lower one, so `G` is exactly symmetric.
fn weighted_gram(s: &[f64], f: &DMatrix<f64>) -> DMatrix<f64> {
    let mut g = weighted_cross_gram(s, f, f);
    let m = g.nrows();
    for k in 0..m {
        for l in k + 1..m {
            g[(l, k)] = g[(k, l)];
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use lobatto_grid::{Extension, GridND};

    /// Left: periodic × Neumann on `[-1, 1) × [0.5, 2]`; right: Neumann on `[0, 1]`; order 4.
    fn grids() -> (GridND<2>, GridND<1>) {
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

    /// Three terms with distinct, non-orthogonal factors.
    fn three_terms() -> LowRankSum<2, 1> {
        let (l, r) = grids();
        let mut s = LowRankSum::new_from_fn(l.clone(), r.clone(), |x| x[0].sin() + x[1], |y| y[0]);
        let mut t = LowRankSum::new_from_fn(l.clone(), r.clone(), |x| x[0] * x[1], |y| 1.0 - y[0] * y[0]);
        t.scale(-0.7);
        s.add_assign(&t);
        let mut u = LowRankSum::new_from_fn(l, r, |x| (x[1] - 1.0).powi(2), |y| y[0].exp());
        u.scale(2.5);
        s.add_assign(&u);
        s
    }

    #[test]
    fn gram_matches_direct_weighted_sum_and_is_symmetric() {
        let s = three_terms();
        let (gl, gr) = (s.gram_left(), s.gram_right());
        let (wl, wr) = (s.left().mass_weights(), s.right().mass_weights());
        assert_eq!(gl.shape(), (3, 3));
        assert_eq!(gr.shape(), (3, 3));
        for k in 0..3 {
            for l in 0..3 {
                let (rk, rl) = (s.rho_k(k), s.rho_k(l));
                let (mk, ml) = (s.mu_k(k), s.mu_k(l));
                let dl: f64 = (0..s.n_left()).map(|i| wl[i] * rk[i] * rl[i]).sum();
                let dr: f64 = (0..s.n_right()).map(|j| wr[j] * mk[j] * ml[j]).sum();
                assert!((gl[(k, l)] - dl).abs() < 1e-13 * (1.0 + dl.abs()));
                assert!((gr[(k, l)] - dr).abs() < 1e-13 * (1.0 + dr.abs()));
                assert_eq!(gl[(k, l)], gl[(l, k)]);
                assert_eq!(gr[(k, l)], gr[(l, k)]);
            }
        }
    }

    #[test]
    fn gram_integrates_low_degree_polynomials_exactly() {
        // ∫∫ (1 + x₁)² over [-1, 1) × [0.5, 2] = 2 · (3³ − 1.5³) / 3 = 15.75,
        // ∫ y⁴ over [0, 1] = 0.2; both of degree ≤ 2r − 1 = 7 per direction.
        let (l, r) = grids();
        let s = LowRankSum::new_from_fn(l, r, |x| 1.0 + x[1], |y| y[0] * y[0]);
        assert!((s.gram_left()[(0, 0)] - 15.75).abs() < 1e-12);
        assert!((s.gram_right()[(0, 0)] - 0.2).abs() < 1e-14);
    }

    #[test]
    fn gram_contraction_is_weighted_norm_of_assembled_field() {
        let s = three_terms();
        let (gl, gr) = (s.gram_left(), s.gram_right());
        let a = s.coefficients();
        let mut from_gram = 0.0;
        for k in 0..3 {
            for l in 0..3 {
                from_gram += a[k] * a[l] * gl[(k, l)] * gr[(k, l)];
            }
        }
        let p = s.assemble();
        let nl = s.n_left();
        let (wl, wr) = (s.left().mass_weights(), s.right().mass_weights());
        let assembled: f64 = p
            .iter()
            .enumerate()
            .map(|(n, v)| wl[n % nl] * wr[n / nl] * v * v)
            .sum();
        assert!((from_gram - assembled).abs() < 1e-12 * assembled);
        assert!((s.norm_squared() - assembled).abs() < 1e-12 * assembled);
    }

    #[test]
    fn norm_squared_of_rank_one_is_product_of_integrals() {
        // ∫∫ (1 + x₁)² = 15.75 on the left, ∫ y⁴ = 0.2 on the right; a = 3.
        let (l, r) = grids();
        let mut s = LowRankSum::new_from_fn(l, r, |x| 1.0 + x[1], |y| y[0] * y[0]);
        s.scale(3.0);
        assert!((s.norm_squared() - 9.0 * 15.75 * 0.2).abs() < 1e-12);
    }

    #[test]
    fn norm_squared_scales_quadratically() {
        let s = three_terms();
        let mut t = s.clone();
        t.scale(-2.5);
        assert!((t.norm_squared() - 6.25 * s.norm_squared()).abs() < 1e-12 * t.norm_squared());
    }

    #[test]
    fn norm_squared_of_cancelling_terms_is_rounding_level_and_nonnegative() {
        let s = three_terms();
        let mut minus = s.clone();
        minus.scale(-1.0);
        let z = s.add(&minus).norm_squared();
        assert!(z >= 0.0);
        assert!(z < 1e-13 * s.norm_squared(), "{z}");
    }

    #[test]
    fn norm_squared_of_empty_sum_is_zero() {
        let (l, r) = grids();
        assert_eq!(LowRankSum::new(l, r).norm_squared(), 0.0);
    }

    /// Two terms, different from those of `three_terms`.
    fn two_other_terms() -> LowRankSum<2, 1> {
        let (l, r) = grids();
        let mut s = LowRankSum::new_from_fn(l.clone(), r.clone(), |x| x[0].cos(), |y| y[0] + 0.5);
        let mut t = LowRankSum::new_from_fn(l, r, |x| x[1] * x[1], |y| (2.0 * y[0]).sin());
        t.scale(1.3);
        s.add_assign(&t);
        s
    }

    /// `Σ_ij w^L_i w^R_j P(i, j) P'(i, j)` from the assembled fields.
    fn assembled_inner(a: &LowRankSum<2, 1>, b: &LowRankSum<2, 1>) -> f64 {
        let nl = a.n_left();
        let (wl, wr) = (a.left().mass_weights(), a.right().mass_weights());
        a.assemble()
            .iter()
            .zip(b.assemble())
            .enumerate()
            .map(|(n, (u, v))| wl[n % nl] * wr[n / nl] * u * v)
            .sum()
    }

    #[test]
    fn inner_matches_assembled_fields() {
        let (a, b) = (three_terms(), two_other_terms());
        let exact = assembled_inner(&a, &b);
        assert!((a.inner(&b) - exact).abs() < 1e-12 * a.norm_squared());
    }

    #[test]
    fn inner_with_self_is_norm_squared() {
        let a = three_terms();
        assert!((a.inner(&a) - a.norm_squared()).abs() < 1e-13 * a.norm_squared());
    }

    #[test]
    fn inner_is_symmetric_and_bilinear() {
        let (a, b, c) = (three_terms(), two_other_terms(), three_terms().add(&two_other_terms()));
        let tol = 1e-12 * (a.norm_squared() + c.norm_squared());
        assert!((a.inner(&b) - b.inner(&a)).abs() < tol);
        assert!((a.inner(&b.add(&c)) - (a.inner(&b) + a.inner(&c))).abs() < tol);
        let mut b3 = b.clone();
        b3.scale(-3.0);
        assert!((a.inner(&b3) + 3.0 * a.inner(&b)).abs() < tol);
    }

    #[test]
    fn distance_squared_matches_assembled_difference() {
        let (a, b) = (three_terms(), two_other_terms());
        let mut minus_b = b.clone();
        minus_b.scale(-1.0);
        let d = a.add(&minus_b);
        let exact = assembled_inner(&d, &d);
        let tol = 1e-12 * (a.norm_squared() + b.norm_squared());
        assert!((a.distance_squared(&b) - exact).abs() < tol);
        assert!((a.distance_squared(&b) - b.distance_squared(&a)).abs() < tol);
    }

    #[test]
    fn distance_squared_to_self_is_rounding_level_and_nonnegative() {
        let a = three_terms();
        let d = a.distance_squared(&a.clone());
        assert!(d >= 0.0);
        assert!(d < 1e-13 * a.norm_squared(), "{d}");
    }

    #[test]
    fn inner_with_empty_sum_is_zero() {
        let a = three_terms();
        let empty = a.zeros_like();
        assert_eq!(a.inner(&empty), 0.0);
        assert_eq!(empty.inner(&a), 0.0);
        assert!((a.distance_squared(&empty) - a.norm_squared()).abs() < 1e-13 * a.norm_squared());
    }

    #[test]
    #[should_panic(expected = "inner product of sums on different grids")]
    fn inner_panics_on_different_grids() {
        let a = three_terms();
        let (l, _) = grids();
        let other_right = GridND::<1>::new([3], [4], [1.0], [0.0], [Extension::Even]);
        let b = LowRankSum::new_from_fn(l, other_right, |x| x[0], |y| y[0]);
        a.inner(&b);
    }

    #[test]
    fn gram_of_empty_sum_is_empty() {
        let (l, r) = grids();
        let s = LowRankSum::new(l, r);
        assert_eq!(s.gram_left().shape(), (0, 0));
        assert_eq!(s.gram_right().shape(), (0, 0));
    }
}
