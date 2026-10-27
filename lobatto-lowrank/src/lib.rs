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
//! [`lobatto-grid`](https://crates.io/crates/lobatto-grid); two sums can be
//! combined when their grids are equal.
//!
//! Norms, Gram matrices and compressions use the discrete `L²` inner product
//! given by the mass weights of each grid (`GridND::mass_weights`).
//!
//! ## Quick start
//!
//! ```
//! use lobatto_grid::{Extension, GridND};
//! use lobatto_lowrank::{LowRankSum, Truncation};
//!
//! // Left and right grids: 4 × 4 elements of order 4 on [-1, 1]², Neumann.
//! let grid = || GridND::<2>::new([4, 4], [4, 4], [2.0; 2], [-1.0; 2], [Extension::Even; 2]);
//!
//! // Nodal factors (the two grids are equal, so they share their nodes).
//! let x = grid().get_x();
//! let cos: Vec<f64> = x.iter().map(|p| p[0].cos()).collect();
//! let sin: Vec<f64> = x.iter().map(|p| p[0].sin()).collect();
//!
//! // cos(x₁ − y₁) + cos(x₁ + y₁), expanded into four products.
//! let mut f = LowRankSum::new(grid(), grid());
//! f.push(1.0, &cos, &cos);
//! f.push(1.0, &sin, &sin);
//! f.push(1.0, &cos, &cos);
//! f.push(-1.0, &sin, &sin);
//!
//! // The field is 2 cos x₁ cos y₁: compression finds rank one.
//! let c = f.compress(Truncation::RelativeTol(1e-12));
//! assert_eq!(c.kept, 1);
//!
//! // Weighted L² norm: 2 · (2 ∫ cos² over [-1, 1])² = 4 (1 + sin 2 / 2).
//! let norm = 4.0 * (1.0 + 2f64.sin() / 2.0);
//! assert!((c.norm - norm).abs() < 1e-8 * norm);
//!
//! // Interpolation at points of the 4D box, without assembling the field.
//! let mut out = [0.0];
//! f.get_values(&[[0.3, -0.2, 0.5, 0.9]], &mut out);
//! assert!((out[0] - 2.0 * 0.3f64.cos() * 0.5f64.cos()).abs() < 1e-5);
//! ```
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`lowrank`] | [`LowRankSum`]: storage, construction, accessors, point evaluation, full grid and assembled field. |
//! | [`leaf`] | [`Leaf`]: `r` functions on a grid (one side of a sum): weighted Gram matrices, orthonormalisation, evaluation, maps. |
//! | [`algebra`] | Scaling, addition, [`push`](LowRankSum::push), weighted Gram matrices, norm, inner product, distance, marginals and maps of the factors. |
//! | [`compress`] | [`Truncation`] criteria and the weighted QR → SVD compression to canonical form. |
//! | [`linalg`] | Dense building blocks on nalgebra matrices: TSQR thin QR, thin SVD, row scaling. |

pub mod algebra;
pub mod compress;
pub mod leaf;
pub mod linalg;
pub mod lowrank;

pub use compress::{Compressed, Truncation};
pub use leaf::Leaf;
pub use lowrank::LowRankSum;
