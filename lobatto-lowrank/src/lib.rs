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
//! [`lobatto-grid`](https://crates.io/crates/lobatto-grid) and are shared by
//! every term.
//!
//! All norms, Gram matrices and compressions use the discrete `L²` inner
//! product of the box, given by the tensor Gauss-Lobatto weights of each grid,
//! never the plain Euclidean one.
//!
//! ## Status
//!
//! The structure and its basic operations ([`lowrank`]) are implemented; the
//! algorithms (weights, assembly, Gram matrices, QR/SVD compression) are not
//! yet and panic with `todo!()`. Since [`LowRankSum::new`] computes the grid
//! weights, no sum can be built until [`weights`] is implemented.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`weights`] | Tensor Gauss-Lobatto mass weights of a grid, in its node order. |
//! | [`lowrank`] | [`LowRankSum`] and [`Term`]: construction, accessors, value, scaling, addition. |
//! | [`inner`] | Weighted Gram matrices, norm, inner product and distance, without assembling. |
//! | [`dense`] | Conversion to and from the dense field on the full grid (tests, small sizes). |
//! | [`compress`] | [`Truncation`] criteria and the weighted QR → SVD compression. |

pub mod compress;
pub mod dense;
pub mod inner;
pub mod lowrank;
pub mod weights;

pub use compress::{Compressed, Truncation};
pub use lowrank::{LowRankSum, Term};
