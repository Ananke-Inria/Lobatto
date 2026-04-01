//! # quadratix
//!
//! This library offers various functionnality around the use of Gauss-Legendre and Gauss-Lobatto points.
//!
//! ## Modules
//!
//! - [`quadrature`] — Gauss and Gauss-Lobatto quadrature points and weights.
//! - [`basis`] — Lagrangian basis functions based upon the Gauss-Lobatto or Gauss-Legendre points in several dimensions
//! - [`hc`] — Hard-coded, SIMD-friendly differentiation kernels for Q3 up to Q6 elements.
//! - [`hofft`] — HoFFT transform planners and solver: *n* elements × order *r* per direction.
//! - [`hot`] — High-order Tensor (HoT) solver: single spectral element, order *r* per direction.

pub mod quadrature;

pub mod basis;

pub(crate) mod utilities;

pub mod hc;

pub mod hofft;

pub mod hot;
