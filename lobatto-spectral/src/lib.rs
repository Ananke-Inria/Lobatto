//! # lobatto-tensor
//!
//! High-order screened Poisson solvers on Gauss-Lobatto grids.
//!
//! Solves $(\alpha M + \beta K)\, u = f$ on tensor-product domains in arbitrary
//! dimension $N$, using GLL spectral-element discretisations from the
//! [`lobatto`](https://crates.io/crates/lobatto) crate.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`solver`] | **solver** — single spectral element per direction; direct eigendecomposition of the local symbol. Supports Periodic, Dirichlet, and Neumann BCs. |

pub mod solver;
