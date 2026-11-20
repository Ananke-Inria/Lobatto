//! One-dimensional tables on `[0, 1]` as fixed-size arrays: quadrature points
//! and weights, and the interpolation and differentiation matrices between two
//! point sets.
//!
//! The matrices follow the convention of the batched kernels in
//! [`crate::collocation_hc`]: row `j` is the target point, column `i` the
//! Lagrange basis function on the source points,
//! $$m_{ji} = L_i(y_j) \quad\text{or}\quad m_{ji} = L_i'(y_j),$$
//! so that applying `m` to nodal values on the `NI` source points gives the
//! values (or derivatives) of the interpolant at the `NO` target points.
//!
//! The tables are computed on each call from [`crate::quadrature`] and
//! [`crate::utilities`]; callers are expected to build them once, at setup.
//! The `_dyn` functions are the same tables with sizes known only at run time
//! (`Vec`s); the fixed-size functions are wrappers around them, so both give
//! the same bits.
//!
//! # Example
//! ```rust
//! use lobatto::collocation::Gauss;
//! use lobatto::tables::*;
//!
//! // Q4 GLL values -> 3 Gauss–Legendre points
//! let m = interpolation_matrix::<5, 3>(Gauss::Lobatto, Gauss::Legendre);
//! let (x, _) = points_weights::<5>(Gauss::Lobatto);
//! let (y, _) = points_weights::<3>(Gauss::Legendre);
//! let u = x.map(|x| x * x * x);
//! for j in 0..3 {
//!     let v: f64 = (0..5).map(|i| m[j][i] * u[i]).sum();
//!     assert!((v - y[j].powi(3)).abs() < 1e-14);
//! }
//! ```

use crate::collocation::Gauss;
use crate::quadrature::{gauss_legendre_on_unit_interval, gauss_lobatto_on_unit_interval};
use crate::utilities::{barycentric_weights, lagrangian_derivative_interpolation, lagrangian_interpolation_into};

/// Quadrature points (increasing) and weights of `N` points of `family` on `[0, 1]`.
///
/// Fixed-size version of [`points_weights_dyn`]. The weights sum to 1.
///
/// # Panics
/// Panics if `N < 2` for [`Gauss::Lobatto`] or `N < 1` for [`Gauss::Legendre`].
pub fn points_weights<const N: usize>(family: Gauss) -> ([f64; N], [f64; N]) {
    let (x, w) = points_weights_dyn(N, family);
    (core::array::from_fn(|i| x[i]), core::array::from_fn(|i| w[i]))
}

/// Quadrature points (increasing) and weights of `n` points of `family` on `[0, 1]`.
///
/// Same values as [`gauss_lobatto_on_unit_interval`] / [`gauss_legendre_on_unit_interval`].
/// The weights sum to 1.
///
/// # Panics
/// Panics if `n < 2` for [`Gauss::Lobatto`] or `n < 1` for [`Gauss::Legendre`].
pub fn points_weights_dyn(n: usize, family: Gauss) -> (Vec<f64>, Vec<f64>) {
    match family {
        Gauss::Lobatto => {
            assert!(n >= 2, "a Gauss–Lobatto rule needs at least 2 points (n = {n})");
            gauss_lobatto_on_unit_interval(n)
        }
        Gauss::Legendre => {
            assert!(n >= 1, "a Gauss–Legendre rule needs at least 1 point (n = {n})");
            gauss_legendre_on_unit_interval(n)
        }
    }
}

/// Values $L_i(y)$ of the Lagrange basis on the nodes `x` at one point `y`, written into `out`.
///
/// `bw` are the barycentric weights of `x` ([`barycentric_weights`]); `out`
/// has the length of `x`. If `y` equals a node bit for bit, `out` is the exact
/// unit vector of that node; otherwise the values are those of
/// [`lagrangian_interpolation_into`].
pub fn interpolation_row(x: &[f64], bw: &[f64], y: f64, out: &mut [f64]) {
    match x.iter().position(|&xi| xi == y) {
        Some(k) => {
            out.fill(0.0);
            out[k] = 1.0;
        }
        None => lagrangian_interpolation_into(x, bw, y, out),
    }
}

/// Interpolation matrix `m[j][i]` $= L_i(y_j)$ from `NI` points of `from` to `NO` points of `to`.
///
/// $L_i$ is the Lagrange polynomial of degree `NI − 1` on the `from` points,
/// $y_j$ the `to` points, both on `[0, 1]`. Exact on polynomials of degree
/// `≤ NI − 1`. Rows are [`interpolation_row`]: a target point equal to a source
/// point gives an exact unit row, so `interpolation_matrix::<N, N>(f, f)` is
/// exactly the identity.
///
/// # Panics
/// Same as [`points_weights`], for both point sets.
pub fn interpolation_matrix<const NI: usize, const NO: usize>(from: Gauss, to: Gauss) -> [[f64; NI]; NO] {
    let (x, _) = points_weights::<NI>(from);
    let (y, _) = points_weights::<NO>(to);
    let bw = barycentric_weights(&x);
    core::array::from_fn(|j| {
        let mut row = [0.0; NI];
        interpolation_row(&x, &bw, y[j], &mut row);
        row
    })
}

/// Differentiation matrix `m[j][i]` $= L_i'(y_j)$ from `NI` points of `from` to `NO` points of `to`.
///
/// Fixed-size version of [`derivative_matrix_dyn`] on the `from` and `to`
/// points. Exact on polynomials of degree `≤ NI − 1`.
/// `derivative_matrix::<NP, NP>(Gauss::Lobatto, Gauss::Lobatto)` is the matrix
/// `d` of the generic square kernels of [`crate::collocation_hc`], and equals
/// [`crate::collocation::CollocationBasis::diff_matrix_1d`] entry for entry.
///
/// # Panics
/// Same as [`points_weights`], for both point sets.
pub fn derivative_matrix<const NI: usize, const NO: usize>(from: Gauss, to: Gauss) -> [[f64; NI]; NO] {
    let (x, _) = points_weights::<NI>(from);
    let (y, _) = points_weights::<NO>(to);
    let m = derivative_matrix_dyn(&x, &y);
    core::array::from_fn(|j| core::array::from_fn(|i| m[j][i]))
}

/// Differentiation matrix `m[j][i]` $= L_i'(y_j)$ from the nodes `x` to the points `y`.
///
/// $L_i$ is the Lagrange polynomial of degree `x.len() − 1` on the nodes `x`
/// (distinct), $y_j$ any points; one row per point of `y`. Exact on
/// polynomials of degree `≤ x.len() − 1`.
pub fn derivative_matrix_dyn(x: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
    y.iter()
        .map(|&yj| (0..x.len()).map(|i| lagrangian_derivative_interpolation(x, yj, i)).collect())
        .collect()
}
