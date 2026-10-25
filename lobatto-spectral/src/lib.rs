//! # lobatto-spectral
//!
//! High-order screened Poisson solvers on Gauss-Lobatto grids by dense fast
//! diagonalisation.
//!
//! Solves $(\alpha M + \beta K)\, u = f$ on tensor-product domains in arbitrary
//! dimension `N`, discretised by `n` Gauss-Lobatto spectral elements of order
//! `r` per direction (grids from the
//! [`lobatto-grid`](https://crates.io/crates/lobatto-grid) crate). The
//! discretisation and the public API are those of the solvers in
//! [`lobatto-fft`](https://crates.io/crates/lobatto-fft); the two crates
//! differ only in how the per-direction operator is diagonalised:
//!
//! | Crate | Per-direction diagonalisation | Sweet spot |
//! |-------|-------------------------------|------------|
//! | `lobatto-spectral` | dense eigendecomposition of the `n·r × n·r` operator, `O(n·r)` per DOF per direction | small problems |
//! | `lobatto-fft` | FFT across elements, `r × r` symbols per Fourier mode, `O(log n)` per DOF per direction | larger problems |
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`solver`] | [`solver::Poisson`] (1D) and [`solver::PoissonND`] (ND) with Periodic, Dirichlet and Neumann conditions per direction, solves and operator functions. |

pub mod solver;
