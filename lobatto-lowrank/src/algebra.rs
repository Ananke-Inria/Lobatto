//! Vector-space operations on [`LowRankSum`]s: scaling, addition, weighted
//! `L²` Gram matrices, norms and inner products.
//!
//! Scaling and addition only touch the coefficients and factor columns: an
//! addition concatenates the terms and never compresses. The inner products
//! are computed from the factor matrices, without assembling the field: the
//! weighted Gram matrices are matrix products `Fᵀ (W G)` and the contractions
//! `aᵀ (G^L ∘ G^R) b` (`∘` the entry-wise product). The distance between
//! close fields goes through a thin QR of the concatenated factors instead,
//! which avoids the cancellation of the Gram contractions.

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::compress::tsqr_in_place;
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

    /// Append the term `a · ρ ⊗ μ`, `rho` of length `N_left`, `mu` of length
    /// `N_right` (nodal values in the order of `GridND::get_x`).
    ///
    /// No compression; amortised `O(N_left + N_right)` like
    /// [`add_assign`](Self::add_assign).
    ///
    /// # Panics
    ///
    /// If `rho` or `mu` does not have the length of its grid.
    pub fn push(&mut self, a: f64, rho: &[f64], mu: &[f64]) {
        assert_eq!(rho.len(), self.n_left(), "rho length differs from the left grid");
        assert_eq!(mu.len(), self.n_right(), "mu length differs from the right grid");
        self.a.extend([a]);
        self.rho.extend(rho.iter().copied());
        self.mu.extend(mu.iter().copied());
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

    /// Weighted `L²` distance to another sum on the same grids, accurate
    /// also for close fields, without assembling either field.
    ///
    /// With the factors of both sums side by side and scaled by the
    /// square-root weights, $\tilde A = W_L^{1/2} [\rho^a \mid \rho^b]$ and
    /// $\tilde B = W_R^{1/2} [\mu^a \mid \mu^b]$, and the coefficients
    /// $c = (a, -b)$, the weighted unfolding of the difference is
    ///
    /// $$
    /// \tilde A \operatorname{diag}(c) \tilde B^T
    /// = Q_A \, (R_A \operatorname{diag}(c) R_B^T) \, Q_B^T
    /// $$
    ///
    /// for thin QR factorisations $\tilde A = Q_A R_A$, $\tilde B = Q_B R_B$.
    /// The `Q` factors have orthonormal columns, so the distance is the
    /// Frobenius norm of the core $R_A \operatorname{diag}(c) R_B^T$, at most
    /// `(M_a + M_b) × (M_a + M_b)`. Only the `R` factors are formed; neither
    /// sum is modified and no difference sum is built: the two QR inputs
    /// (`N_left × (M_a + M_b)` and `N_right × (M_a + M_b)`) are the only
    /// large allocations. Cost `(N_left + N_right) · (M_a + M_b)²`.
    ///
    /// Accuracy: the QR is backward stable and the norm is a sum of squares,
    /// so the absolute error is about machine precision times `‖ρ‖ + ‖ρ'‖`
    /// on the distance itself. The expansion
    /// $\|\rho\|^2 - 2 \langle \rho, \rho' \rangle + \|\rho'\|^2$ from
    /// [`norm_squared`](Self::norm_squared) and [`inner`](Self::inner) would
    /// instead cancel for close fields and resolve no distance below about
    /// `1e-8 · ‖ρ‖`. Zero if both sums are empty.
    ///
    /// # Panics
    ///
    /// If the two sums are not on equal grids (see [`Self::same_grids`]).
    pub fn distance(&self, other: &Self) -> f64 {
        assert!(self.same_grids(other), "distance between sums on different grids");
        let (ma, m) = (self.len(), self.len() + other.len());
        // Column-major: the factor storages chained are the concatenated
        // columns, and the weights cycle once per column.
        let mut a = DMatrix::from_iterator(
            self.n_left(),
            m,
            self.rho.iter().chain(other.rho.iter()).zip(self.sqrt_w_left.iter().cycle()).map(|(x, s)| x * s),
        );
        let mut b = DMatrix::from_iterator(
            self.n_right(),
            m,
            self.mu.iter().chain(other.mu.iter()).zip(self.sqrt_w_right.iter().cycle()).map(|(x, s)| x * s),
        );
        // TSQR in place; only R is needed.
        let (mut r_a, r_b) = (tsqr_in_place(&mut a).r(&a), tsqr_in_place(&mut b).r(&b));
        for (k, mut col) in r_a.column_iter_mut().enumerate() {
            col *= if k < ma { self.a[k] } else { -other.a[k - ma] };
        }
        (r_a * r_b.transpose()).norm()
    }

    // ── Marginals ───────────────────────────────────────────────────────────

    /// The field summed over the right directions with the right grid's mass
    /// weights, `Σ_k a_k (Σ_j w_j μ_k[j]) ρ_k`: a nodal field on the left grid.
    ///
    /// The quadrature of the field in the right variables, computed as
    /// `ρ (a ∘ μᵀ w)` without assembling the field. Cost
    /// `(N_left + N_right) · M`. All zeros for an empty sum.
    pub fn marginal_left(&self) -> Vec<f64> {
        let mut c = self.mu.tr_mul(&self.sqrt_w_right.component_mul(&self.sqrt_w_right));
        c.component_mul_assign(&self.a);
        (&self.rho * c).data.into()
    }

    /// The field summed over the left directions,
    /// `Σ_k a_k (Σ_i w_i ρ_k[i]) μ_k`: a nodal field on the right grid.
    ///
    /// `μ (a ∘ ρᵀ w)`; see [`marginal_left`](Self::marginal_left).
    pub fn marginal_right(&self) -> Vec<f64> {
        let mut c = self.rho.tr_mul(&self.sqrt_w_left.component_mul(&self.sqrt_w_left));
        c.component_mul_assign(&self.a);
        (&self.mu * c).data.into()
    }

    /// Apply `f` to every left factor in place, `ρ_k ← f(ρ_k)`; the
    /// coefficients are unchanged. (A pointwise weight, a diffusion step.)
    ///
    /// `f` is called once per term on the contiguous column `ρ_k` (length
    /// `N_left`), in parallel over the terms and in no particular order.
    pub fn map_left_in_place(&mut self, f: impl Fn(&mut [f64]) + Sync) {
        // Column-major: the columns are the chunks of length N_left (`max(1)`
        // for an empty grid, whose storage is empty).
        let n = self.n_left().max(1);
        self.rho.as_mut_slice().par_chunks_mut(n).for_each(&f);
    }

    /// Apply `f` to every right factor in place, `μ_k ← f(μ_k)`.
    ///
    /// See [`map_left_in_place`](Self::map_left_in_place).
    pub fn map_right_in_place(&mut self, f: impl Fn(&mut [f64]) + Sync) {
        let n = self.n_right().max(1);
        self.mu.as_mut_slice().par_chunks_mut(n).for_each(&f);
    }

    /// Like [`map_left_in_place`](Self::map_left_in_place), with a scratch
    /// value for `f`: `init` is called once per parallel task (not per
    /// term), and `f(&mut scratch, ρ_k)` reuses that value for every term of
    /// the task. For work buffers without an allocation per term.
    pub fn map_left_in_place_with<T>(
        &mut self,
        init: impl Fn() -> T + Sync,
        f: impl Fn(&mut T, &mut [f64]) + Sync,
    ) {
        let n = self.n_left().max(1);
        self.rho.as_mut_slice().par_chunks_mut(n).for_each_init(&init, &f);
    }

    /// Like [`map_right_in_place`](Self::map_right_in_place), with a scratch
    /// value for `f`; see [`map_left_in_place_with`](Self::map_left_in_place_with).
    pub fn map_right_in_place_with<T>(
        &mut self,
        init: impl Fn() -> T + Sync,
        f: impl Fn(&mut T, &mut [f64]) + Sync,
    ) {
        let n = self.n_right().max(1);
        self.mu.as_mut_slice().par_chunks_mut(n).for_each_init(&init, &f);
    }

    /// The sum `Σ_k a_k · L(ρ_k) ⊗ R(μ_k)` for maps `L` of the left factors
    /// and `R` of the right factors, on the same grids and with the same
    /// number of terms. Each map writes its result into the output slice,
    /// which has the length of its input.
    ///
    /// The output slices are zero-filled before each call. The maps are
    /// called in parallel over the terms (and the two sides concurrently), in
    /// no particular order. `self` is left untouched; for an in-place update
    /// use [`map_left_in_place`](Self::map_left_in_place) and
    /// [`map_right_in_place`](Self::map_right_in_place).
    pub fn map(
        &self,
        left: impl Fn(&[f64], &mut [f64]) + Sync,
        right: impl Fn(&[f64], &mut [f64]) + Sync,
    ) -> Self {
        let (nl, nr, m) = (self.n_left(), self.n_right(), self.len());
        let (rho, mu) = rayon::join(
            || {
                let mut rho = DMatrix::zeros(nl, m);
                rho.as_mut_slice()
                    .par_chunks_mut(nl.max(1))
                    .zip(self.rho.as_slice().par_chunks(nl.max(1)))
                    .for_each(|(dst, src)| left(src, dst));
                rho
            },
            || {
                let mut mu = DMatrix::zeros(nr, m);
                mu.as_mut_slice()
                    .par_chunks_mut(nr.max(1))
                    .zip(self.mu.as_slice().par_chunks(nr.max(1)))
                    .for_each(|(dst, src)| right(src, dst));
                mu
            },
        );
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            sqrt_w_left: self.sqrt_w_left.clone(),
            sqrt_w_right: self.sqrt_w_right.clone(),
            inv_sqrt_w_left: self.inv_sqrt_w_left.clone(),
            inv_sqrt_w_right: self.inv_sqrt_w_right.clone(),
            a: self.a.clone(),
            rho,
            mu,
        }
    }

    /// Like [`map`](Self::map), but each parallel task gets its own state for
    /// each side: `init_left` / `init_right` are called once per task, and the
    /// state is passed to `left` / `right` for every factor the task handles.
    pub fn map_with<T, U>(
        &self,
        init_left: impl Fn() -> T + Sync,
        left: impl Fn(&mut T, &[f64], &mut [f64]) + Sync,
        init_right: impl Fn() -> U + Sync,
        right: impl Fn(&mut U, &[f64], &mut [f64]) + Sync,
    ) -> Self {
        let (nl, nr, m) = (self.n_left(), self.n_right(), self.len());
        let (rho, mu) = rayon::join(
            || {
                let mut rho = DMatrix::zeros(nl, m);
                rho.as_mut_slice()
                    .par_chunks_mut(nl.max(1))
                    .zip(self.rho.as_slice().par_chunks(nl.max(1)))
                    .for_each_init(&init_left, |t, (dst, src)| left(t, src, dst));
                rho
            },
            || {
                let mut mu = DMatrix::zeros(nr, m);
                mu.as_mut_slice()
                    .par_chunks_mut(nr.max(1))
                    .zip(self.mu.as_slice().par_chunks(nr.max(1)))
                    .for_each_init(&init_right, |u, (dst, src)| right(u, src, dst));
                mu
            },
        );
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            sqrt_w_left: self.sqrt_w_left.clone(),
            sqrt_w_right: self.sqrt_w_right.clone(),
            inv_sqrt_w_left: self.inv_sqrt_w_left.clone(),
            inv_sqrt_w_right: self.inv_sqrt_w_right.clone(),
            a: self.a.clone(),
            rho,
            mu,
        }
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
    fn inner_with_empty_sum_is_zero() {
        let a = three_terms();
        let empty = a.zeros_like();
        assert_eq!(a.inner(&empty), 0.0);
        assert_eq!(empty.inner(&a), 0.0);
    }

    #[test]
    fn distance_matches_assembled_difference_and_is_symmetric() {
        let (a, b) = (three_terms(), two_other_terms());
        let mut minus_b = b.clone();
        minus_b.scale(-1.0);
        let exact = assembled_inner(&a.add(&minus_b), &a.add(&minus_b)).sqrt();
        let tol = 1e-13 * (a.norm_squared().sqrt() + b.norm_squared().sqrt());
        assert!((a.distance(&b) - exact).abs() < tol);
        assert!((b.distance(&a) - exact).abs() < tol);
    }

    #[test]
    fn distance_resolves_close_fields() {
        // b = a + δ·t with δ = 1e-10: ‖a − b‖ = δ‖t‖, far below the
        // `1e-8 · ‖a‖` that ‖a‖² − 2⟨a, b⟩ + ‖b‖² resolves.
        let a = three_terms();
        let (l, r) = grids();
        let t = LowRankSum::new_from_fn(l, r, |x| x[0].cos() * x[1], |y| 1.0 + y[0]);
        let exact = 1e-10 * t.norm_squared().sqrt();
        let mut b = a.clone();
        let mut dt = t.clone();
        dt.scale(1e-10);
        b.add_assign(&dt);
        let d = a.distance(&b);
        assert!((d - exact).abs() < 1e-4 * exact, "{d:e} vs {exact:e}");
    }

    #[test]
    fn distance_to_self_is_rounding_level() {
        let a = three_terms();
        assert!(a.distance(&a.clone()) < 1e-14 * a.norm_squared().sqrt());
    }

    #[test]
    fn distance_with_empty_sums() {
        let a = three_terms();
        let empty = a.zeros_like();
        assert_eq!(empty.distance(&empty), 0.0);
        let norm = a.norm_squared().sqrt();
        assert!((a.distance(&empty) - norm).abs() < 1e-13 * norm);
        assert!((empty.distance(&a) - norm).abs() < 1e-13 * norm);
    }

    #[test]
    #[should_panic(expected = "distance between sums on different grids")]
    fn distance_panics_on_different_grids() {
        let a = three_terms();
        let (l, _) = grids();
        let other_right = GridND::<1>::new([3], [4], [1.0], [0.0], [Extension::Even]);
        a.distance(&LowRankSum::new_from_fn(l, other_right, |x| x[0], |y| y[0]));
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
    fn marginals_match_weighted_sums_of_assembled_field() {
        let s = three_terms();
        let (nl, nr) = (s.n_left(), s.n_right());
        let (wl, wr) = (s.left().mass_weights(), s.right().mass_weights());
        let p = s.assemble();
        let (ml, mr) = (s.marginal_left(), s.marginal_right());
        assert_eq!((ml.len(), mr.len()), (nl, nr));
        for i in 0..nl {
            let exact: f64 = (0..nr).map(|j| wr[j] * p[i + nl * j]).sum();
            assert!((ml[i] - exact).abs() < 1e-13 * (1.0 + exact.abs()));
        }
        for j in 0..nr {
            let exact: f64 = (0..nl).map(|i| wl[i] * p[i + nl * j]).sum();
            assert!((mr[j] - exact).abs() < 1e-13 * (1.0 + exact.abs()));
        }
    }

    #[test]
    fn marginals_of_rank_one_polynomial_are_exact() {
        // 3 (1 + x₁) y²: ∫ y² over [0, 1] = 1/3, ∫∫ (1 + x₁) over
        // [-1, 1) × [0.5, 2] = 2 · (3² − 1.5²) / 2 = 6.75.
        let (l, r) = grids();
        let mut s = LowRankSum::new_from_fn(l, r, |x| 1.0 + x[1], |y| y[0] * y[0]);
        s.scale(3.0);
        for (v, x) in s.marginal_left().iter().zip(s.left().get_x()) {
            assert!((v - (1.0 + x[1])).abs() < 1e-14);
        }
        for (v, y) in s.marginal_right().iter().zip(s.right().get_x()) {
            assert!((v - 3.0 * 6.75 * y[0] * y[0]).abs() < 1e-13);
        }
    }

    #[test]
    fn marginals_of_empty_sum_are_zero() {
        let (l, r) = grids();
        let s = LowRankSum::new(l, r);
        assert!(s.marginal_left().iter().all(|&v| v == 0.0));
        assert!(s.marginal_right().iter().all(|&v| v == 0.0));
        assert_eq!((s.marginal_left().len(), s.marginal_right().len()), (s.n_left(), s.n_right()));
    }

    #[test]
    fn push_matches_add_assign_of_rank_one_sums() {
        let s = three_terms();
        let mut p = s.zeros_like();
        for k in 0..s.len() {
            p.push(s.coefficients()[k], s.rho_k(k), s.mu_k(k));
        }
        assert_eq!(p.coefficients(), s.coefficients());
        assert_eq!(p.rho(), s.rho());
        assert_eq!(p.mu(), s.mu());
    }

    #[test]
    #[should_panic(expected = "rho length differs from the left grid")]
    fn push_panics_on_wrong_length() {
        let mut s = three_terms();
        let mu = s.mu_k(0).to_vec();
        s.push(1.0, &[0.0; 3], &mu);
    }

    #[test]
    fn map_in_place_applies_to_every_factor() {
        let s = three_terms();
        let mut t = s.clone();
        t.map_left_in_place(|r| r.iter_mut().for_each(|v| *v = 2.0 * *v + 1.0));
        t.map_right_in_place(|m| m.reverse());
        assert_eq!(t.coefficients(), s.coefficients());
        for k in 0..s.len() {
            assert!(t.rho_k(k).iter().zip(s.rho_k(k)).all(|(x, y)| *x == 2.0 * y + 1.0));
            assert!(t.mu_k(k).iter().eq(s.mu_k(k).iter().rev()));
        }
    }

    /// Enough terms for the parallel loops to split: every column is mapped
    /// exactly once, as by a serial loop over the columns.
    #[test]
    fn maps_with_many_terms_match_column_by_column() {
        let (l, r) = grids();
        let mut s = LowRankSum::new(l, r);
        for k in 0..200 {
            let kf = k as f64;
            let rho: Vec<f64> = (0..s.n_left()).map(|i| (0.3 * i as f64 + kf).sin()).collect();
            let mu: Vec<f64> = (0..s.n_right()).map(|j| (0.7 * j as f64 - kf).cos()).collect();
            s.push(1.0 + kf, &rho, &mu);
        }
        // A map that depends on the whole column: prefix sums.
        let prefix = |c: &mut [f64]| {
            for i in 1..c.len() {
                c[i] += c[i - 1];
            }
        };
        let mut t = s.clone();
        t.map_left_in_place(prefix);
        t.map_right_in_place(prefix);
        let mapped = s.map(
            |src, dst| {
                dst.copy_from_slice(src);
                prefix(dst);
            },
            |src, dst| {
                dst.copy_from_slice(src);
                prefix(dst);
            },
        );
        // The scratch variants, with a reused buffer: copy, prefix-sum in the
        // buffer, copy back. `init` runs once per task, at most once per term.
        let inits = std::sync::atomic::AtomicUsize::new(0);
        let init = || {
            inits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Vec::<f64>::new()
        };
        let via_buffer = |buf: &mut Vec<f64>, c: &mut [f64]| {
            buf.clear();
            buf.extend_from_slice(c);
            prefix(buf);
            c.copy_from_slice(buf);
        };
        let mut w = s.clone();
        w.map_left_in_place_with(init, via_buffer);
        w.map_right_in_place_with(init, via_buffer);
        let n_inits = inits.load(std::sync::atomic::Ordering::Relaxed);
        assert!((1..=2 * s.len()).contains(&n_inits), "{n_inits} init calls");
        // `map_with`, different state types per side: a reused buffer on the
        // left, a running total of the terms handled on the right.
        let mw = s.map_with(
            Vec::<f64>::new,
            |buf, src, dst| {
                buf.clear();
                buf.extend_from_slice(src);
                prefix(buf);
                dst.copy_from_slice(buf);
            },
            || 0usize,
            |handled, src, dst| {
                *handled += 1;
                dst.copy_from_slice(src);
                prefix(dst);
            },
        );
        assert_eq!(mw.coefficients(), s.coefficients());
        for k in 0..s.len() {
            let (mut rho, mut mu) = (s.rho_k(k).to_vec(), s.mu_k(k).to_vec());
            prefix(&mut rho);
            prefix(&mut mu);
            assert_eq!((t.rho_k(k), t.mu_k(k)), (&rho[..], &mu[..]), "term {k}");
            assert_eq!((mapped.rho_k(k), mapped.mu_k(k)), (&rho[..], &mu[..]), "term {k}");
            assert_eq!((w.rho_k(k), w.mu_k(k)), (&rho[..], &mu[..]), "term {k}");
            assert_eq!((mw.rho_k(k), mw.mu_k(k)), (&rho[..], &mu[..]), "term {k}");
        }
    }

    #[test]
    fn map_matches_map_in_place_and_keeps_self() {
        let s = three_terms();
        let left = |r: &[f64], out: &mut [f64]| {
            for (o, v) in out.iter_mut().zip(r) {
                *o = v * v;
            }
        };
        let right = |m: &[f64], out: &mut [f64]| {
            // Reversal: the output is a genuine function of the whole input.
            for (o, v) in out.iter_mut().zip(m.iter().rev()) {
                *o = 2.0 * v;
            }
        };
        let mapped = s.map(left, right);
        let mut t = s.clone();
        t.map_left_in_place(|r| r.iter_mut().for_each(|v| *v *= *v));
        t.map_right_in_place(|m| {
            m.reverse();
            m.iter_mut().for_each(|v| *v *= 2.0);
        });
        assert!(mapped.same_grids(&s));
        assert_eq!(mapped.coefficients(), t.coefficients());
        assert_eq!(mapped.rho(), t.rho());
        assert_eq!(mapped.mu(), t.mu());
        assert_eq!(s.rho(), three_terms().rho());
    }

    #[test]
    fn map_of_empty_sum_is_empty() {
        let (l, r) = grids();
        let s = LowRankSum::new(l, r).map(|_, _| unreachable!(), |_, _| unreachable!());
        assert!(s.is_empty());
        assert_eq!((s.rho().nrows(), s.mu().nrows()), (s.n_left(), s.n_right()));
    }

    #[test]
    fn gram_of_empty_sum_is_empty() {
        let (l, r) = grids();
        let s = LowRankSum::new(l, r);
        assert_eq!(s.gram_left().shape(), (0, 0));
        assert_eq!(s.gram_right().shape(), (0, 0));
    }
}
