//! # quadratix
//!
//! High-order spectral-element solvers for the periodic Laplace equation
//!
//! ```text
//!   (α M + β K) u = f   on [0, L]^N
//! ```
//!
//! using Gauss-Lobatto-Legendre (GLL) quadrature.
//!
//! ## Modules
//!
//! - [`quadrature`] — Gauss and Gauss-Lobatto quadrature points and weights.
//! - [`calculator`] — Multi-dimensional GLL differentiation and mass operators.
//! - [`hc_calculations`] — Hard-coded, SIMD-friendly differentiation kernels for Q3 and Q4 elements.
//! - [`hofft`] — High-order FFT (HoFFT) solver: *n* elements × order *r* per direction.
//! - [`hot`] — High-order Tensor (HoT) solver: single spectral element, order *r* per direction.

pub mod calculator;
pub mod hc_calculations;
pub mod hofft;
pub mod hot;
pub(crate) mod lagrangian_interpolation;
pub mod quadrature;
