//! Tensor Gauss-Lobatto mass weights of a grid.
//!
//! The weight of a stored node is the product of its per-direction weights,
//! $w(i) = \prod_d w_d(i_d)$, in the grid's own column-major node order
//! (direction 0 fastest). The per-direction weights are the reference
//! Gauss-Lobatto weights scaled by the element width `h = L/n` and assembled
//! over the elements:
//!
//! | Node | Weight |
//! |------|--------|
//! | interior node of an element | its element weight |
//! | node shared by two elements, periodic seam included | sum of the two end weights |
//! | stored end node (`Even`) | its single end weight |
//! | unstored end node (`Odd`) | dropped |

use lobatto_grid::{Grid1D, GridND};

/// Per-direction mass weights of a 1D grid, one per stored node.
///
/// Length is `g.n_dof()`.
pub fn axis_weights(g: &Grid1D) -> Vec<f64> {
    let _ = g;
    todo!("assemble the scaled Gauss-Lobatto weights over the elements")
}

/// Tensor mass weights of an ND grid, in the order of `g.strides()`.
///
/// Length is `g.total()`.
pub fn tensor_weights<const D: usize>(g: &GridND<D>) -> Vec<f64> {
    let _ = g;
    todo!("tensor product of the axis weights")
}
