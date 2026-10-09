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
//! [`lobatto-grid`](https://crates.io/crates/lobatto-grid); each side is a
//! [`WeightedGrid`], the grid bundled with its mass weights, shared by every
//! term and every sum derived from it. Sums built independently on equal
//! grids can still be added.
//!
//! All norms, Gram matrices and compressions use the discrete `L²` inner
//! product of the box, given by the mass weights of each grid
//! (`GridND::mass_weights`), never the plain Euclidean one.
//!
//! ## Status
//!
//! The structure and its basic operations ([`lowrank`]) are implemented; the
//! algorithms (assembly, Gram matrices, QR/SVD compression) are not yet and
//! panic with `todo!()`.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`lowrank`] | [`WeightedGrid`], [`LowRankSum`] and [`LowRankTerm`]: construction, accessors, value, scaling, addition. |
//! | [`inner`] | Weighted Gram matrices, norm, inner product and distance, without assembling. |
//! | [`dense`] | Conversion to and from the dense field on the full grid (tests, small sizes). |
//! | [`compress`] | [`Truncation`] criteria and the weighted QR → SVD compression. |

pub mod compress;
pub mod dense;
pub mod inner;
pub mod lowrank;

pub use compress::{Compressed, Truncation};
pub use lowrank::{LowRankSum, LowRankTerm, WeightedGrid};
