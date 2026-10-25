# lobatto-spectral

High-order screened Poisson solvers on Gauss-Lobatto grids by dense fast
diagonalisation.

Solves (αI − βΔ) u = f on a box [x_l, x_l + L]^N discretised by `n` spectral
finite elements of polynomial order `r` per direction, with periodic, Dirichlet
or Neumann boundary conditions chosen independently per direction. Grids come
from [`lobatto-grid`](https://crates.io/crates/lobatto-grid).

The discretisation and the public interface are those of the solvers in
[`lobatto-fft`](https://crates.io/crates/lobatto-fft), restricted to real data
and coefficients. The two crates differ in how the per-direction operator is
diagonalised:

| Crate | Per-direction diagonalisation | Sweet spot |
|-------|-------------------------------|------------|
| `lobatto-spectral` | dense eigendecomposition of the `n·r × n·r` operator | small `n·r` |
| `lobatto-fft` | FFT across the elements, `r × r` symbols per Fourier mode | many elements per direction |

---

## Method

With `M` the lumped mass matrix and `K` the stiffness matrix of the 1D
spectral-element discretisation, the generalised eigendecomposition

    K V = M V diag(μ),   Vᵀ M V = I

is computed once per direction at construction (dense, `n·r × n·r`). In `N`
dimensions the operator is a Kronecker sum, so its eigenvectors are
`V⁽⁰⁾ ⊗ … ⊗ V⁽ᴺ⁻¹⁾` and its eigenvalues

    λ_{j_0,...,j_{N-1}} = α + β ∑_d μ⁽ᵈ⁾_{j_d}.

Each call to `solve_in_place` runs four steps:

1. **Mass scaling** — f ← M f (diagonal, embarrassingly parallel).
2. **Forward transforms** — apply V⁽ᵈ⁾ᵀ direction by direction (sum factorisation).
3. **Spectral division** — f̂_j ← f̂_j / λ_j (pseudo-inverse).
4. **Inverse transforms** — apply V⁽ᵈ⁾ direction by direction.

Steps 2 and 4 are parallelised over independent fibres with
[Rayon](https://github.com/rayon-rs/rayon). Setup costs `O(Σ_d (n_d r_d)³)`,
one solve `O(Σ_d n_d r_d · N_dof)`.

Step 3 is one instance of a functional calculus: `apply_in_place` evaluates
any `g(λ)` instead of `1/λ`, so implicit Euler steps and the exact semigroup
cost one solve each.

---

## Boundary conditions

| BC          | Extension | `N_dof` per direction |
|-------------|-----------|-----------------------|
| `Periodic`  | none      | `n·r`                 |
| `Dirichlet` | odd       | `n·r − 1`             |
| `Neumann`   | even      | `n·r + 1`             |

The DOF vector is in x-increasing order, direction 0 varying fastest in ND:
the layout of `Grid1D::get_x` / `GridND::get_x`.

---

## Quick start

```toml
[dependencies]
lobatto-spectral = "0.2"
```

```rust
use lobatto_spectral::solver::{BoundaryCondition, PoissonND};
use std::f64::consts::PI;

// (α I − β Δ) u = f on [0, 1]^3 with 2 elements of order r = 10 per direction.
let solver = PoissonND::<3>::new(
    [1.0; 3],                          // domain lengths  L_d
    [0.0; 3],                          // left boundaries x_{l,d}
    [2; 3],                            // elements per direction n_d
    [10; 3],                           // polynomial orders r_d
    [BoundaryCondition::Periodic; 3],  // boundary conditions
);

let x = solver.get_x(); // tensor-product GLL grid, length 20^3 = 8 000

let mut f: Vec<f64> = x
    .iter()
    .map(|&[x0, x1, x2]| {
        // evaluate the right-hand side f here
        (PI * x0).sin() * (PI * x1).sin() * (PI * x2).sin()
    })
    .collect();

let (alpha, beta) = (1.0, 1.0);
solver.solve_in_place(&mut f, alpha, beta); // f now contains u at the GLL nodes

// Interpolate the solution anywhere through the grid.
let pts = [[0.3, 0.4, 0.5]];
let mut vals = [0.0];
solver.grid().eval(&pts, &f, &mut vals);
```

For a complete 5-D run see
[`examples/laplace_5d.rs`](https://github.com/OndoMathX/QuadratiX/blob/main/lobatto-spectral/examples/laplace_5d.rs).

---

## API

### `solver::BoundaryCondition`

```rust
pub enum BoundaryCondition { Periodic, Dirichlet, Neumann }
```

### `solver::Poisson` — 1D solver

```rust
let solver = Poisson::new(l, xl, n, r, bc);
let x: Vec<f64> = solver.get_x();                    // N_dof GLL nodes
let grid = solver.grid();                            // &lobatto_grid::Grid1D
solver.solve_in_place(&mut f, alpha, beta);          // overwrites f with u
solver.apply_in_place(&mut u, alpha, beta, g);       // u ← g(αI + βM⁻¹K) u
solver.apply_mu_in_place(&mut u, g);                 // u ← g(M⁻¹K) u
solver.euler_step_in_place(&mut u, alpha, beta, dt); // (I + dt L)⁻¹ u
solver.exp_in_place(&mut u, alpha, beta, t);         // exp(−t L) u
solver.mu_max();                                     // spectral radius of M⁻¹K
```

### `solver::PoissonND<N>` — N-dimensional solver

```rust
let solver = PoissonND::<N>::new(ls, xls, ns, rs, bcs);
let x: Vec<[f64; N]> = solver.get_x();               // ∏_d N_dof^(d) grid points
let grid = solver.grid();                            // &lobatto_grid::GridND<N>
solver.solve_in_place(&mut f, alpha, beta);
solver.apply_in_place(&mut u, alpha, beta, g);       // g: Fn(f64) -> f64 + Sync
solver.apply_in_place_dirs(&mut u, g);               // g: Fn([f64; N]) -> f64 + Sync
solver.euler_step_in_place(&mut u, alpha, beta, dt);
solver.euler_split_step_in_place(&mut u, alpha, beta, dt);
solver.exp_in_place(&mut u, alpha, beta, t);
```

**Input convention** — `solve_in_place` takes the strong-form right-hand side
and returns `(αM + βK)⁻¹ f`; the `apply_*` methods take the **nodal vector**
and return `g(αI + βM⁻¹K) u` (with `g(λ) = 1/λ`, that is `(αM + βK)⁻¹ M u`).

**Pseudo-inverse** — the solves annihilate eigenvalues with
`|λ| ≤ 1e-12 · (|α| + |β| max|μ|)`, which filters the round-off constant mode
of a pure periodic/Neumann Laplacian. The `apply_*` methods apply no cutoff.

---

## Running the example

```sh
cargo run --example laplace_5d --release
```

## Running tests

```sh
cargo test --release
```

## Running benchmarks

```sh
cargo bench
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| [`lobatto`](https://crates.io/crates/lobatto) | GLL quadrature, collocation basis, differentiation matrices |
| [`lobatto-grid`](https://crates.io/crates/lobatto-grid) | Grid geometry, DOF layout, interpolation |
| [`nalgebra`](https://nalgebra.org) | Dense symmetric eigendecomposition |
| [`rayon`](https://github.com/rayon-rs/rayon) | Data-parallel fibre sweeps |

---

## License

Licensed under the **GNU Lesser General Public License v3.0** (LGPL-3.0-only).
See [lgpl-3.0.txt](lgpl-3.0.txt) for the full license text.
