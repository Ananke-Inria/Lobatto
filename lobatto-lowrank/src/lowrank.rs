//! The [`LowRankSum`] structure, its construction and its basic operations.
//!
//! A [`LowRankSum<P, Q>`] holds a field on a `P + Q` dimensional grid as a
//! sum of `M` products `a_k · ρ_k ⊗ μ_k`, with `ρ_k` a nodal field on the
//! `P`-dimensional left grid and `μ_k` a nodal field on the `Q`-dimensional
//! right grid.
//!
//! A sum takes its two grids by value. Each side is a [`WeightedGrid`]: the
//! grid and its mass weights, computed once and shared through an [`Arc`] by
//! the sum and every sum derived from it (`clone`,
//! [`zeros_like`](LowRankSum::zeros_like), [`add`](LowRankSum::add)), so these
//! never copy a grid or its weights. Two sums built independently on equal
//! grids can still be added: sides are compared by grid equality.
//!
//! The operations here only touch the term list: no inner product, no
//! factorisation. Norms live in [`crate::inner`], the dense unfolding in
//! [`crate::dense`], the compression in [`crate::compress`].

use std::sync::Arc;

use lobatto_grid::GridND;

/// One side of a [`LowRankSum`]: a grid and its mass weights.
///
/// The weights are [`GridND::mass_weights`], computed once at construction.
/// They define the discrete `L²` inner product used by every norm, Gram
/// matrix and compression of the sums built on this side.
#[derive(Debug)]
pub struct WeightedGrid<const D: usize> {
    grid: GridND<D>,
    weights: Vec<f64>,
}

impl<const D: usize> WeightedGrid<D> {
    /// Bundle a grid with its mass weights.
    pub(crate) fn new(grid: GridND<D>) -> Self {
        let weights = grid.mass_weights();
        Self { grid, weights }
    }

    /// The grid.
    pub fn grid(&self) -> &GridND<D> {
        &self.grid
    }

    /// Mass weights, one per stored node, in the grid's node order.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Number of stored nodes.
    pub fn total(&self) -> usize {
        self.grid.total()
    }
}

/// One term `a · ρ ⊗ μ` of a [`LowRankSum`].
#[derive(Debug, Clone, PartialEq)]
pub struct LowRankTerm {
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
#[derive(Debug, Clone)]
pub struct LowRankSum<const P: usize, const Q: usize> {
    pub(crate) left: Arc<WeightedGrid<P>>,
    pub(crate) right: Arc<WeightedGrid<Q>>,
    pub(crate) terms: Vec<LowRankTerm>,
}

impl<const P: usize, const Q: usize> LowRankSum<P, Q> {
    // ── Construction ────────────────────────────────────────────────────────

    /// The empty sum, i.e. the zero field, on the grids `left` and `right`.
    pub fn new(left: GridND<P>, right: GridND<Q>) -> Self {
        Self {
            left: Arc::new(WeightedGrid::new(left)),
            right: Arc::new(WeightedGrid::new(right)),
            terms: Vec::new(),
        }
    }

    /// The empty sum on the same sides as `self`, sharing its grids and weights.
    pub fn zeros_like(&self) -> Self {
        Self {
            left: Arc::clone(&self.left),
            right: Arc::clone(&self.right),
            terms: Vec::new(),
        }
    }

    /// A single product `a · ρ ⊗ μ`.
    ///
    /// # Panics
    ///
    /// If the factor lengths differ from the grids' `total()`.
    pub fn product(left: GridND<P>, right: GridND<Q>, a: f64, rho: Vec<f64>, mu: Vec<f64>) -> Self {
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
            self.n_left(),
            "left factor has {} entries, left grid has {} nodes",
            rho.len(),
            self.n_left()
        );
        assert_eq!(
            mu.len(),
            self.n_right(),
            "right factor has {} entries, right grid has {} nodes",
            mu.len(),
            self.n_right()
        );
        self.terms.push(LowRankTerm { a, rho, mu });
    }

    /// Remove every term: the zero field on the same sides.
    pub fn clear(&mut self) {
        self.terms.clear();
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    /// The left side: grid and mass weights.
    pub fn left(&self) -> &WeightedGrid<P> {
        &self.left
    }

    /// The right side: grid and mass weights.
    pub fn right(&self) -> &WeightedGrid<Q> {
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

    /// Mass weights of the left grid, in its node order.
    pub fn w_left(&self) -> &[f64] {
        self.left.weights()
    }

    /// Mass weights of the right grid, in its node order.
    pub fn w_right(&self) -> &[f64] {
        self.right.weights()
    }

    /// The terms.
    pub fn terms(&self) -> &[LowRankTerm] {
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

    /// Whether `other` lives on the same grids: shared sides, or equal grids.
    pub fn same_grids(&self, other: &Self) -> bool {
        let left = Arc::ptr_eq(&self.left, &other.left) || self.left.grid == other.left.grid;
        let right =
            Arc::ptr_eq(&self.right, &other.right) || self.right.grid == other.right.grid;
        left && right
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
        assert!(self.same_grids(other), "adding sums on different grids");
        self.terms.extend_from_slice(&other.terms);
    }
}
