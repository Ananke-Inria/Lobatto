//! # lobatto
//!
//! Gauss–Lobatto quadrature and interpolation points, spectral finite-element lagrangian basis operations
//! with efficient hardcoded versions for low order low dimensions cases
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`quadrature`] | GL and GLL nodes and weights on `[-1, 1]` and `[0, 1]`. |
//! | [`quadrature_hc`] | Hardcoded GLL nodes and weights for n = 2..14, certified to 1e-15. |
//! | [`collocation`] | Multi-dimensional collocation basis (tensor-product Gauss/GLL quadrature + Lagrange interpolation). |
//! | [`collocation_hc`] | Hardcoded SIMD-friendly operations for Q3–Q6 elements in dimension up to 3, plus generic-order square and rectangular (`NI → NO` points) fallbacks. |
//! | [`tables`] | 1D points, weights, interpolation and differentiation matrices between GL/GLL point sets, as fixed-size arrays. |
//! | [`utilities`] | Lagrange interpolation in barycentric form. |

pub mod quadrature;
pub mod quadrature_hc;

pub mod collocation;
pub mod collocation_hc;

pub mod tables;
pub mod utilities;
