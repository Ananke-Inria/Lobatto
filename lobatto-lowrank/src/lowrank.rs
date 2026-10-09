//! The [`LowRankSum`] structure, its construction and its basic operations.
//!
//! A [`LowRankSum<P, Q>`] holds a field on a `P + Q` dimensional grid as a
//! sum of `M` products `a_k · ρ_k ⊗ μ_k`, with `ρ_k` a nodal field on the
//! `P`-dimensional left grid and `μ_k` a nodal field on the `Q`-dimensional
//! right grid. Both grids are shared by every term through an [`Arc`].
//!
//! The operations here only touch the term list: no inner product, no
//! factorisation. Norms live in [`crate::inner`], the dense unfolding in
//! [`crate::dense`], the compression in [`crate::compress`].

use std::sync::Arc;

use lobatto_grid::GridND;

use crate::weights::tensor_weights;

/// One term `a · ρ ⊗ μ` of a [`LowRankSum`].
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    /// Scalar coefficient `a_k`.
    pub a: f64,
    /// Left factor `ρ_k`, one value per stored node of the left grid.
    pub rho: Vec<f64>,
    /// Right factor `μ_k`, one value per stored node of the right grid.
    pub mu: Vec<f64>,
}

/// A field on a `P + Q` dimensional grid stored as a sum of `M` products of a
/// left field on a `P`-dimensional grid and a right field on a
/// `Q`-dimensional grid.
///
/// The sum is not kept canonical between operations: an addition simply
/// concatenates the terms, and only [`LowRankSum::compress`] returns the
/// canonical form (orthonormal factors, `a_k = σ_k ≥ 0` decreasing).
#[derive(Clone)]
pub struct LowRankSum<const P: usize, const Q: usize> {
    pub(crate) left: Arc<GridND<P>>,
    pub(crate) right: Arc<GridND<Q>>,
    /// Tensor weights of the left grid.
    pub(crate) w_left: Vec<f64>,
    /// Tensor weights of the right grid.
    pub(crate) w_right: Vec<f64>,
    pub(crate) terms: Vec<Term>,
}

impl<const P: usize, const Q: usize> LowRankSum<P, Q> {
    // ── Construction ────────────────────────────────────────────────────────

    /// The empty sum, i.e. the zero field, on the given grids.
    pub fn new(left: Arc<GridND<P>>, right: Arc<GridND<Q>>) -> Self {
        let w_left = tensor_weights(&left);
        let w_right = tensor_weights(&right);
        Self {
            left,
            right,
            w_left,
            w_right,
            terms: Vec::new(),
        }
    }

    /// The empty sum on the same grids as `self`, reusing its weights.
    pub fn zeros_like(&self) -> Self {
        Self {
            left: Arc::clone(&self.left),
            right: Arc::clone(&self.right),
            w_left: self.w_left.clone(),
            w_right: self.w_right.clone(),
            terms: Vec::new(),
        }
    }

    /// A single product `a · ρ ⊗ μ`.
    ///
    /// # Panics
    ///
    /// If the factor lengths differ from the grids' `total()`.
    pub fn product(
        left: Arc<GridND<P>>,
        right: Arc<GridND<Q>>,
        a: f64,
        rho: Vec<f64>,
        mu: Vec<f64>,
    ) -> Self {
        let mut s = Self::new(left, right);
        s.push(a, rho, mu);
        s
    }

    /// Append the term `a · ρ ⊗ μ`.
    ///
    /// # Panics
    ///
    /// If the factor lengths differ from the grids' `total()`.
    pub fn push(&mut self, a: f64, rho: Vec<f64>, mu: Vec<f64>) {
        assert_eq!(
            rho.len(),
            self.left.total(),
            "left factor has {} entries, left grid has {} nodes",
            rho.len(),
            self.left.total()
        );
        assert_eq!(
            mu.len(),
            self.right.total(),
            "right factor has {} entries, right grid has {} nodes",
            mu.len(),
            self.right.total()
        );
        self.terms.push(Term { a, rho, mu });
    }

    /// Remove every term: the zero field on the same grids.
    pub fn clear(&mut self) {
        self.terms.clear();
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    /// The left grid.
    pub fn left(&self) -> &Arc<GridND<P>> {
        &self.left
    }

    /// The right grid.
    pub fn right(&self) -> &Arc<GridND<Q>> {
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

    /// Tensor weights of the left grid, in its node order.
    pub fn w_left(&self) -> &[f64] {
        &self.w_left
    }

    /// Tensor weights of the right grid, in its node order.
    pub fn w_right(&self) -> &[f64] {
        &self.w_right
    }

    /// The terms.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// Number of terms `M`.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether the sum has no term (the zero field).
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Whether `other` lives on the same grids, i.e. the same [`Arc`]s.
    pub fn same_grids(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.left, &other.left) && Arc::ptr_eq(&self.right, &other.right)
    }

    // ── Basic operations ────────────────────────────────────────────────────

    /// The field at the node pair `(i, j)`, `Σ_k a_k ρ_k(i) μ_k(j)`.
    ///
    /// `i` is a flat left node index, `j` a flat right node index.
    pub fn value(&self, i: usize, j: usize) -> f64 {
        self.terms.iter().map(|t| t.a * t.rho[i] * t.mu[j]).sum()
    }

    /// Multiply the field by `c` (the coefficients `a_k`, not the factors).
    pub fn scale(&mut self, c: f64) {
        for t in &mut self.terms {
            t.a *= c;
        }
    }

    /// The sum of two fields: the term lists concatenated, no compression.
    ///
    /// # Panics
    ///
    /// If the two sums are not on the same grids (see [`Self::same_grids`]).
    pub fn add(&self, other: &Self) -> Self {
        let mut s = self.clone();
        s.add_assign(other);
        s
    }

    /// In-place version of [`LowRankSum::add`].
    ///
    /// # Panics
    ///
    /// If the two sums are not on the same grids (see [`Self::same_grids`]).
    pub fn add_assign(&mut self, other: &Self) {
        assert!(
            self.same_grids(other),
            "adding sums on different grids (the grids must be the same Arcs)"
        );
        self.terms.extend_from_slice(&other.terms);
    }
}
