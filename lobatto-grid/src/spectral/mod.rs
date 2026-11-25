//! High-order screened Poisson solvers on Gauss-Lobatto grids by dense fast
//! diagonalisation.
//!
//! Solves $(\alpha M + \beta K)\, u = f$ on tensor-product domains in arbitrary
//! dimension `N`, discretised by `n` Gauss-Lobatto spectral elements of order
//! `r` per direction (grids of [`crate::grid`]). The discretisation and the
//! public API are those of the solvers in `fft` ([`crate::fft`] with the
//! `fft` feature); the two modules differ only in how the per-direction
//! operator is diagonalised:
//!
//! | Module | Per-direction diagonalisation | Sweet spot |
//! |-------|-------------------------------|------------|
//! | `spectral` | dense eigendecomposition of the `n·r × n·r` operator, `O(n·r)` per DOF per direction | small problems |
//! | `fft` | FFT across elements, `r × r` symbols per Fourier mode, `O(log n)` per DOF per direction | larger problems |
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`solver`] | [`solver::Poisson`] (1D) and [`solver::PoissonND`] (ND) with Periodic, Dirichlet and Neumann conditions per direction, solves and operator functions. |
//!
//! This module is compiled with the `spectral` feature (on by default). Until
//! lobatto-grid 0.1.9 it was the separate crate `lobatto-spectral`.

pub mod solver;
