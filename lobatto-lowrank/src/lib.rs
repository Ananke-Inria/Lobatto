//! # lobatto-lowrank
//!
//! Low-rank sums of products of fields on two Gauss-Lobatto grids.
//!
//! A field on an `N`-dimensional tensor-product grid is split at a cut
//! between the first `P` directions (the *left* grid) and the last `Q`
//! directions (the *right* grid), and stored as a sum of `M` products
//!
//! $$
//! \rho(x) = \sum_{k=1}^{M} a_k \, \rho_k(x_1, \dots, x_P) \, \mu_k(x_{P+1}, \dots, x_{P+Q}),
//! $$
//!
//! with scalars $a_k$, each $\rho_k$ a nodal field on the left grid and each
//! $\mu_k$ a nodal field on the right grid. Grids come from
//! [`lobatto-grid`](https://crates.io/crates/lobatto-grid); a sum owns its two
//! grids and their mass weights, and two sums can be added when their grids
//! are equal.
//!
//! All norms, Gram matrices and compressions use the discrete `L²` inner
//! product of the box, given by the mass weights of each grid
//! (`GridND::mass_weights`), never the plain Euclidean one.
//!
//! ## Status
//!
//! The structure, the full grid, the assembled field and point evaluation
//! ([`lowrank`]), the algebra (scaling, addition, Gram matrices, norm, inner
//! product, distance) and the weighted QR → SVD compression ([`compress`])
//! are implemented.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`lowrank`] | [`LowRankSum`]: factor-matrix storage, construction, accessors, value, point evaluation, full grid and assembled field. |
//! | [`algebra`] | Scaling, addition, weighted Gram matrices, norm, inner product and distance, without assembling. |
//! | [`compress`] | [`Truncation`] criteria, the weighted QR → SVD compression, and its dense counterpart from an assembled field. |

pub mod algebra;
pub mod compress;
pub mod lowrank;

pub use compress::{Compressed, Truncation};
pub use lowrank::LowRankSum;
