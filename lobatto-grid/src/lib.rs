//! # lobatto-grid
//!
//! Multi-element Gauss–Lobatto grids on tensor-product domains: geometry,
//! boundary extensions, and high-order Lagrange interpolation.
//!
//! A [`Grid1D`] describes `n` Gauss–Lobatto finite elements of polynomial order
//! `r` on an **arbitrary interval** `[x_l, x_l + L]` — any origin and length;
//! the `[0, 1]` reference element used for the Gauss–Lobatto nodes and basis
//! evaluation is an internal detail only. A [`GridND`] is the tensor product of
//! `N` independent 1D grids, one per direction.
//!
//! Each direction carries an [`Extension`] (`Periodic`, `Odd`, `Even`) that
//! encodes how the represented field extends beyond the domain — periodic
//! wrap, odd (Dirichlet) or even (Neumann) reflection — which fixes the DOF
//! layout and count, and lets interpolation fold arbitrary query positions
//! back into the domain consistently.
//!
//! ## Interpolation
//!
//! | Routine | Description |
//! |---------|-------------|
//! | [`Grid1D::eval_at`] / [`Grid1D::eval`], [`GridND::eval`] | Interpolate a DOF vector at in-domain positions. |
//! | [`Grid1D::eval_extended`], [`GridND::eval_extended`] | Same, for arbitrary positions (folded via the extension). |
//! | [`Grid1D::convect`], [`GridND::convect`] | Interpolate at the image of every grid node under a flow map. |
//! | [`GridND::eval_plan`] / [`GridND::convect_plan`] + [`GridND::eval_with_plan`] | Precompute all position-dependent work once, apply many times with varying DOF data. |
//!
//! All routines are generic over the DOF scalar via [`Scalar`] — in
//! particular `f64` and `Complex<f64>` fields share the same code paths and
//! plans.
//!
//! ## Differentiation and indexing
//!
//! | Routine | Description |
//! |---------|-------------|
//! | [`Grid1D::derivative`], [`GridND::derivative`] | Nodal derivative, element by element (no assembled matrix), averaged at shared nodes; ND along every fibre of one direction, Rayon-parallel. |
//! | [`grid::nd_col_major_strides`], [`grid::nd_fiber_base`], [`grid::nd_multi_indices`], [`grid::nd_multi_index`] | Column-major strides of a DOF tensor, the start index of its `s`-th fibre along a direction, its multi-indices in column-major order, and the multi-index of one flat index. |
//!
//! ## Quick start
//!
//! ```rust
//! use lobatto_grid::{Extension, GridND};
//!
//! // 2D grid: 8 elements of order 4 per direction on [0,2]×[−1,0].
//! let grid = GridND::<2>::new(
//!     [8; 2],                            // elements per direction n_d
//!     [4; 2],                            // polynomial orders r_d
//!     [2.0, 1.0],                        // domain lengths L_d
//!     [0.0, -1.0],                       // left boundaries x_{l,d}
//!     [Extension::Periodic; 2],          // boundary extensions
//! );
//!
//! // A field sampled at the grid nodes ...
//! let u: Vec<f64> = grid.get_x().iter().map(|&[x, y]| (x + y).sin()).collect();
//!
//! // ... interpolated at arbitrary points.
//! let pts = [[0.3, -0.4], [1.7, -0.1]];
//! let mut vals = [0.0_f64; 2];
//! grid.eval(&pts, &u, &mut vals);
//! ```
//!
//! This crate underpins the HoFFT transforms and screened Poisson solvers of
//! the `lobatto-fft` crate, which hold a grid internally and expose it via
//! `grid()` accessors.

pub mod grid;
pub mod interp;

pub use grid::{Extension, Grid1D, GridND};
pub use interp::{EvalPlan, Scalar};
