//! # lobatto-tensor
//!
//! High-order tensor-product screened Poisson solvers on GLL grids.
//!
//! Solves $(\alpha M + \beta K)\, u = f$ on tensor-product domains in arbitrary
//! dimension $N$, using GLL spectral-element discretisations from the
//! [`lobatto`](https://crates.io/crates/lobatto) crate.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`hot`] | **HoT** — single spectral element per direction; direct eigendecomposition of the local symbol. Supports Periodic, Dirichlet, and Neumann BCs. |
//! | [`hofft`] | **HoFFT** — multi-element solver; FFT diagonalisation over inter-element coupling modes. |

pub mod hot;

pub mod hofft;
