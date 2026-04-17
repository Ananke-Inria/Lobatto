# lobatto-spectral

High-order screened Poisson solvers on Gauss-Lobatto grids.

Solves (αI − βΔ) u = f on [x_l, x_l + L]^N using a **single**
spectral finite-element of polynomial order r per direction. Supports periodic, Dirichlet, and Neumann boundary conditions, independently per direction.

For problems with many elements per direction, see
[`lobatto-fft`](https://crates.io/crates/lobatto-fft), which uses the multi-element HoFFT algorithm.

---

## Method

With one element per direction the global operator factorises as a tensor product of 1D
operators. At construction the generalised eigendecomposition

M^{-1/2}(αM + βK)M^{-1/2} = W Λ W^T,   V = M^{-1/2} W

is computed once per direction. The ND eigenvalues are then

λ_{j_0,...,j_{N-1}} = α + β ∑_{d=0}^{N-1} λ^(d)_{j_d}

and each call to `solve_in_place` runs four steps:

1. **Mass scaling** — f ← M_ND f (diagonal, embarrassingly parallel).
2. **Forward transform** — apply V^(d)T direction by direction (sum factorisation).
3. **Spectral division** — f̂_j ← f̂_j / λ_j (pseudo-inverse).
4. **Inverse transform** — apply V^(d) direction by direction.

Steps 2 and 4 are parallelised over independent fibres with [Rayon](https://github.com/rayon-rs/rayon).

---

## Boundary conditions

| BC          | Active nodes  | N_dof | 
|-------------|---------------|-------|
| `Periodic`  | 0, ..., r-1   | r     | 
| `Dirichlet` | 1, ..., r-1   | r-1   | 
| `Neumann`   | 0, ..., r     | r+1   |

---

## Quick start

```toml
[dependencies]
lobatto-spectral = "0.1.0"
```

```rust
use lobatto_spectral::solver::{BoundaryCondition, PoissonND};
use std::f64::consts::PI;

// Solve (I α -  β Δ) u = f on [0, 1]^3 with a single element of order r=20 per direction.
let solver = PoissonND::<3>::new(
    [1.0; 3],                          // domain lengths  L_d
    [0.0; 3],                          // left boundaries x_{l,d}
    [20;  3],                          // polynomial orders r_d
    1.0, 1.0,                          // α, β
    [BoundaryCondition::Periodic; 3],  // boundary conditions
);

let x = solver.get_x(); // tensor-product GLL grid, length 20^3 = 8 000

 
let mut f: Vec<f64> = x
    .iter()
    .map(|&[x0, x1, x2]| {
        // evaluate the right-hand side f here
        (PI * x0).sin()*(PI * x1).sin()*(PI * x2).sin()
    })
    .collect();

solver.solve_in_place(&mut f); // f now contains the solution u at the GLL nodes
```

For a complete 5-D benchmark see
[`examples/laplace_5d.rs`](https://github.com/OndoMathX/QuadratiX/blob/main/lobatto-spectral/examples/laplace_5d.rs).

---

## API

### `solver::BoundaryCondition`

```rust
pub enum BoundaryCondition { Periodic, Dirichlet, Neumann }
```

### `solver::Poisson` — 1D solver

```rust
let solver = Poisson::new(l, xl, r, alpha, beta, bc);
let x: Vec<f64>   = solver.get_x();           // N_dof GLL nodes
let val: f64      = solver.eval(x_pt, &u);    // interpolate u at x_pt
solver.solve_in_place(&mut f);                // overwrites f with u
```

### `solver::PoissonND<N>` — N-dimensional solver

```rust
let solver = PoissonND::<N>::new(ls, xls, rs, alpha, beta, bcs);
let x: Vec<[f64; N]> = solver.get_x();       // ∏_d N_dof^(d) grid points
let vals             = solver.eval(&pts, &u); // batch interpolation
solver.solve_in_place(&mut f);               // overwrites f with u
```

---

## Running the example

```sh
cargo run --example laplace_5d --release
```

## Running tests

```sh
cargo test
```

## Running benchmarks

```sh
cargo bench
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| [`lobatto`](https://crates.io/crates/lobatto) | GLL quadrature, collocation basis, differentiation and interpolation |
| [`nalgebra`](https://nalgebra.org) | Eigendecomposition and matrix-vector products |
| [`rayon`](https://github.com/rayon-rs/rayon) | Data-parallel fibre sweeps |

---

## License

Licensed under the **GNU Lesser General Public License v3.0** (LGPL-3.0-only).
See [lgpl-3.0.txt](lgpl-3.0.txt) for the full license text.
