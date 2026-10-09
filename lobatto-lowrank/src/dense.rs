//! Conversion between a [`LowRankSum`] and the dense field on the full grid.
//!
//! # Layout of an assembled field
//!
//! An assembled field lives on the full `P + Q` grid, column-major with the
//! left directions fastest: node pair `(i, j)`, with `i` a left node index and
//! `j` a right node index, sits at flat index `i + j · N_left`, where
//! `N_left = left.total()`. Equivalently, the assembled field is the
//! column-major `N_left × N_right` unfolding matrix.
//!
//! Both directions form the dense field: tests and small sizes only.

use lobatto_grid::GridND;

use crate::lowrank::LowRankSum;

impl<const P: usize, const Q: usize> LowRankSum<P, Q> {
    /// Compress a field given on the full grid, in the layout of the module
    /// docs, by the SVD of its weighted unfolding.
    pub fn from_assembled(
        left: GridND<P>,
        right: GridND<Q>,
        p: &[f64],
        rank: usize,
    ) -> Self {
        let _ = (left, right, p, rank);
        todo!("dense weighted SVD of the unfolding")
    }

    /// The full field `Σ_k a_k ρ_k(i) μ_k(j)` in the layout of the module docs.
    pub fn assemble(&self) -> Vec<f64> {
        todo!("assemble the unfolding")
    }
}
