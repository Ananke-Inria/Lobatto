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
//! | [`hofft`] | **HoFFT** — multi-element solver; FFT diagonalisation over inter-element coupling modes.  Supports Periodic, Dirichlet, and Neumann BCs. |

pub mod hofft;
