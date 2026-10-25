# lobatto-lowrank

Low-rank sums of products of fields on two Gauss–Lobatto grids, with compression in the weighted L² norm.

A field on an `N`-dimensional tensor-product grid is split between the first `P` directions (the *left* grid) and the last `Q = N − P` directions (the *right* grid), and stored as a sum of `M` products

```text
ρ(x) = Σ_k a_k · ρ_k(x_1, …, x_P) · μ_k(x_{P+1}, …, x_N),    k = 1, …, M
```

with scalars `a_k`, each `ρ_k` a nodal field on the left grid and each `μ_k` a nodal field on the right grid. Storage is `M·(N_left + N_right + 1)` values instead of `N_left · N_right` for the full field, which is never formed unless asked for.

Grids come from [`lobatto-grid`](https://crates.io/crates/lobatto-grid): multi-element Gauss–Lobatto discretisations with a periodic, Dirichlet (odd) or Neumann (even) extension per direction. Every norm, inner product and compression uses the discrete L² inner product of the box, given by the Gauss–Lobatto mass weights of each grid, never the plain Euclidean one.

## Provides

`LowRankSum<P, Q>`, with:

- **construction**: empty sum on two grids, rank-one sum from two functions (`new_from_fn`), term-by-term building from nodal values (`push`);
- **algebra without assembling**: scaling, addition (`add`, `add_assign`; terms concatenated, no compression), weighted Gram matrices, norm, inner product and distance between sums;
- **compression**: best approximation in the weighted norm by a thin QR of each factor matrix and an SVD of the small core matrix, with `Truncation` by rank, relative or absolute tolerance, or both; `orthonormalize` brings a sum to canonical form without discarding anything;
- **evaluation**: interpolation at arbitrary points of the full box (`get_values`; points located once, plans reused for every term), node values, and, for small sizes and tests, the full grid and the assembled field;
- **marginals and factor maps**: the field integrated over the right or left variables (`marginal_left`, `marginal_right`), and maps applied to every left or right factor (`map_left_in_place`, `map_right_in_place`, `map`).

Factors and Gram matrices are [`nalgebra`](https://crates.io/crates/nalgebra) `DMatrix<f64>` values.

## Quick start

```toml
[dependencies]
lobatto-grid = "0.1"
lobatto-lowrank = "0.3"
```

```rust
use lobatto_grid::{Extension, GridND};
use lobatto_lowrank::{LowRankSum, Truncation};

// Left and right grids: 4 × 4 elements of order 4 on [-1, 1]², Neumann.
let grid = || GridND::<2>::new([4, 4], [4, 4], [2.0; 2], [-1.0; 2], [Extension::Even; 2]);

// Nodal factors (the two grids are equal, so they share their nodes).
let x = grid().get_x();
let cos: Vec<f64> = x.iter().map(|p| p[0].cos()).collect();
let sin: Vec<f64> = x.iter().map(|p| p[0].sin()).collect();

// cos(x₁ − y₁) + cos(x₁ + y₁), expanded into four products.
let mut f = LowRankSum::new(grid(), grid());
f.push(1.0, &cos, &cos);
f.push(1.0, &sin, &sin);
f.push(1.0, &cos, &cos);
f.push(-1.0, &sin, &sin);

// The field is 2 cos x₁ cos y₁: compression finds rank one.
let c = f.compress(Truncation::RelativeTol(1e-12));
assert_eq!(c.kept, 1);

// Weighted L² norm: 2 · (2 ∫ cos² over [-1, 1])² = 4 (1 + sin 2 / 2).
let norm = 4.0 * (1.0 + 2f64.sin() / 2.0);
assert!((c.norm - norm).abs() < 1e-8 * norm);

// Interpolation at points of the 4D box, without assembling the field.
let mut out = [0.0];
f.get_values(&[[0.3, -0.2, 0.5, 0.9]], &mut out);
assert!((out[0] - 2.0 * 0.3f64.cos() * 0.5f64.cos()).abs() < 1e-5);
```

## Compression

The squared weighted norm of a sum is the Frobenius norm of its unfolding scaled by the square-root mass weights, `W_L^{1/2} P W_R^{1/2}`. `compress` computes the SVD of that matrix without forming it:

1. scale the factor matrices by the square-root weights, `Ã = W_L^{1/2} [ρ_1 … ρ_M]`, `B̃ = W_R^{1/2} [μ_1 … μ_M]`;
2. thin QR of both, `Ã = Q_A R_A`, `B̃ = Q_B R_B`;
3. core matrix `C = R_A diag(a) R_Bᵀ`, at most `M × M`, and its SVD `C = U Σ Vᵀ`;
4. keep `r` triplets: `ρ_k = W_L^{-1/2} Q_A U_k`, `μ_k = W_R^{-1/2} Q_B V_k`, `a_k = σ_k`.

The result is in canonical form (orthonormal factors in the weighted norms, `a_k` decreasing) and is the best rank-`r` approximation in the weighted norm. The discarded singular values are returned, so the error is known exactly. Cost `O((N_left + N_right)·M² + M³)`, linear in the grid sizes.

## Example

`examples/gaussian_coupling.rs` builds a Gaussian on `[-1, 1]² × [-1, 1]²` with a coupling term across the cut, `exp(−(|x|² + |y|²)/2a²) · exp(−|x − y|²/2s²)`, from 729 separable terms (a quadrature of the coupling as a convolution). On small grids it compresses all terms at once and checks the singular values against a dense SVD. On grids of up to `161² + 161²` nodes (`6.7·10⁸` nodes on the full grid, never formed) it adds terms in batches with a compression after each batch, then reports rank against tolerance, the pointwise error at random points, and memory against the full field.

```sh
cargo run --release --example gaussian_coupling
```

## Related crates

Part of the [QuadratiX](https://github.com/OndoMathX/QuadratiX) workspace:

- [`lobatto`](https://crates.io/crates/lobatto): Gauss–Lobatto points, quadrature and Lagrange bases;
- [`lobatto-grid`](https://crates.io/crates/lobatto-grid): the grids, interpolation and mass weights used here;
- [`lobatto-fft`](https://crates.io/crates/lobatto-fft), [`lobatto-spectral`](https://crates.io/crates/lobatto-spectral): screened Poisson solvers on the same grids.

## License

LGPL-3.0-only.
