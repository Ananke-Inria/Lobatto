# lobatto-grid

Multi-element Gauss–Lobatto grids on tensor-product domains: geometry, boundary extensions, high-order Lagrange interpolation, and screened Poisson solvers on those grids.

`Grid1D` describes n Gauss–Lobatto finite elements of polynomial order r on an arbitrary interval [x_l, x_l + L]; `GridND<N>` is the tensor product of N independent 1D grids. Each direction carries an `Extension` (`Periodic`, `Odd`/Dirichlet, `Even`/Neumann) that fixes the DOF layout and lets interpolation fold arbitrary query positions back into the domain.

Provides:

- `eval` / `eval_extended`: batched high-order interpolation of DOF vectors (sum-factored, element-grouped, Rayon-parallel),
- `convect`: interpolation at the image of every grid node under a flow map,
- `EvalPlan`: precomputed plans for repeated evaluation at fixed points with varying data — the apply is allocation-free gather + contraction,
- generic scalar support (`f64` and `Complex<f64>` share code paths and plans),
- `derivative`: element-wise nodal derivative in 1D and along any direction in ND (no assembled matrix, average at shared nodes),
- `nd_col_major_strides` / `nd_fiber_base` / `nd_multi_indices` / `nd_multi_index`: column-major DOF-tensor indexing helpers (strides, fibre start offsets, multi-indices in order, multi-index of a flat index),
- two screened Poisson solvers, behind Cargo features (below).

---

## Solvers (features `fft` and `spectral`)

Both solve (αI − βΔ) u = f, i.e. (αM + βK) u = f, on a box discretised by n_d elements of order r_d per direction, with periodic, Dirichlet or Neumann conditions chosen per direction. They share the discretisation and the public interface (`solver::Poisson`, `solver::PoissonND<N>`, `solver::BoundaryCondition`) and differ in how the per-direction operator is diagonalised:

| Module | Feature | Per-direction diagonalisation | Sweet spot |
|--------|---------|-------------------------------|------------|
| `fft` | `fft` | high-order FFT (HoFFT): FFT across the elements, `r × r` symbols per Fourier mode, O(N_dof log N_dof) | many elements per direction |
| `spectral` | `spectral` | dense eigendecomposition of the `n·r × n·r` operator, O(Σ_d n_d r_d · N_dof) | small `n·r` |

Both features are on by default. To compile only the grid and the interpolation (no `rustfft`, no `nalgebra`):

```toml
[dependencies]
lobatto-grid = { version = "0.1.9", default-features = false }
```

Until version 0.1.9 the two modules were the separate crates `lobatto-fft` and `lobatto-spectral`; replace `lobatto_fft::` by `lobatto_grid::fft::` and `lobatto_spectral::` by `lobatto_grid::spectral::`.

### Boundary conditions

| BC          | Extension | N_dof per direction |
|-------------|-----------|---------------------|
| `Periodic`  | none      | n·r                 |
| `Dirichlet` | odd       | n·r − 1             |
| `Neumann`   | even      | n·r + 1             |

The DOF vector is in x-increasing order, direction 0 varying fastest in ND: the layout of `Grid1D::get_x` / `GridND::get_x`.

### `fft`: high-order FFT

Based on the HoFFT of Caforio & Imperiale (2019): a single FFT of length n_d (2·n_d for non-periodic conditions, by odd or even extension) per direction decouples the Fourier modes; each mode then needs an independent r × r Hermitian system whose eigendecomposition is precomputed at setup. Data and coefficients may be complex (e.g. a damped Helmholtz problem).

```rust
use lobatto_grid::fft::solver::{BoundaryCondition, PoissonND};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

// (α I − β Δ) u = f on [0, 1]^2 with n = 8 elements of order r = 4 per direction.
let solver = PoissonND::<2>::new([1.0; 2], [0.0; 2], [8; 2], [4; 2], [BoundaryCondition::Periodic; 2]);
let mut f: Vec<Complex<f64>> = solver
    .get_x()
    .iter()
    .map(|&[x, y]| Complex::new((PI * x).sin() * (PI * y).sin(), 0.0))
    .collect();
solver.solve_in_place(&mut f, 1.0, 1.0); // α = 1, β = 1; f now holds u at the nodes
```

**Reference** — Caforio, F. and Imperiale, S. (2019). *High-order discrete Fourier transform for the solution of the Poisson equation*. SIAM Journal on Scientific Computing, 41(5), A2747–A2771.

### `spectral`: dense fast diagonalisation

The generalised eigendecomposition K V = M V diag(μ), Vᵀ M V = I, is computed once per direction (dense, n·r × n·r); in N dimensions the operator is a Kronecker sum, so a solve is a forward transform by V⁽ᵈ⁾ᵀ per direction, a division by λ = α + β Σ_d μ⁽ᵈ⁾, and the inverse transform. Real data and coefficients.

```rust
use lobatto_grid::spectral::solver::{BoundaryCondition, PoissonND};
use std::f64::consts::PI;

// (α I − β Δ) u = f on [0, 1]^3 with 2 elements of order r = 10 per direction.
let solver = PoissonND::<3>::new([1.0; 3], [0.0; 3], [2; 3], [10; 3], [BoundaryCondition::Periodic; 3]);
let mut f: Vec<f64> = solver
    .get_x()
    .iter()
    .map(|&[x0, x1, x2]| (PI * x0).sin() * (PI * x1).sin() * (PI * x2).sin())
    .collect();
solver.solve_in_place(&mut f, 1.0, 1.0); // f now holds u at the nodes
```

### Operator functions (both modules)

The diagonalisation makes any function of L = αI + βM⁻¹K cost one solve: `apply_in_place(&mut u, alpha, beta, g)` computes g(L) u, `apply_in_place_dirs` (ND) gives any tensor product ⊗_d g_d(L_d), and `euler_step_in_place`, `euler_split_step_in_place`, `exp_in_place` are the implicit Euler steps (unsplit and directionally split) and the exact semigroup exp(−tL). `solve_*` take the strong-form right-hand side and return (αM + βK)⁻¹ f; `apply_*` take the nodal vector. See the module documentation for the full API.

---

## Examples, tests, benchmarks

```sh
cargo run --release --example fft_laplace_5d
cargo run --release --example spectral_laplace_5d
cargo test --release
cargo bench
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| [`lobatto`](https://crates.io/crates/lobatto) | GLL quadrature, collocation basis, differentiation and interpolation |
| [`realfft`](https://crates.io/crates/realfft) | Real FFT for trigonometric (equidistant) directions |
| [`rustfft`](https://crates.io/crates/rustfft) | Forward/inverse FFT plans (`fft` feature) |
| [`nalgebra`](https://nalgebra.org) | Eigendecompositions and small dense products (`fft`, `spectral` features) |
| [`rayon`](https://github.com/rayon-rs/rayon) | Data-parallel fibre sweeps |

Part of the [Lobatto](https://github.com/Ananke-Inria/Lobatto) workspace.

## License

Licensed under the **GNU Lesser General Public License v3.0** (LGPL-3.0-only).
