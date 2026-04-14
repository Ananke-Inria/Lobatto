//! # lobatto
//!
//! Gauss–Lobatto–Legendre (GLL) and Gauss–Legendre (GL) quadrature rules,
//! spectral-element basis operators, and hardcoded differentiation kernels.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`quadrature`] | GL and GLL nodes and weights on $[-1,1]$ and $[0,1]$. |
//! | [`quadrature_hc`] | Hardcoded GLL nodes and weights for $n = 2\ldots14$, certified to $10^{-15}$. |
//! | [`basis`] | [`LagrangianBasis`]: multi-dimensional GLL/GL operations on tensor-product grids. |
//! | [`basis_hc`] | Hardcoded SIMD-friendly operations for $Q_3$–$Q_6$ elements and dimension less or equal 3. |
//! | [`utilities`] | Lagrange interpolation in barycentric form. |

pub mod quadrature;
pub mod quadrature_hc;

pub mod basis;
pub use basis::LagrangianBasis;

pub mod basis_hc;

pub mod utilities;
