# lobatto-spectral

High-order screened Poisson solvers on Gauss-Lobatto grids.

Solves $(\alpha I - \beta \Delta)~ u = f$ on $[x_l, x_l + L]^N$ using a **single**  
spectral finite-element of polynomial order $r$ per direction.  Supports periodic, Dirichlet, and Neumann boundary conditions, independently
per direction.

For problems with many elements per direction, see
[`lobatto-fft`](../lobatto-fft/README.md), which uses the multi-element HoFFT algorithm.

---

## Method

With one element per direction the global operator factorises as a tensor product of 1D
operators.  At construction the generalised eigendecomposition

$$M^{-1/2}(\alpha M + \beta K)M^{-1/2} = W \Lambda W^T, \quad V = M^{-1/2} W$$

is computed once per direction.  The ND eigenvalues are then

$$\lambda_{j_0,\ldots,j_{N-1}} = \alpha + \beta \sum_{d=0}^{N-1} \lambda^{(d)}_{j_d},$$

and each call to `solve_in_place` runs four steps:

1. **Mass scaling** — $f \leftarrow M_\mathrm{ND}\, f$ (diagonal, embarrassingly parallel).
2. **Forward transform** — apply $V^{(d)\,T}$ direction by direction (sum factorisation).
3. **Spectral division** — $\hat{f}_{j} \leftarrow \hat{f}_{j} / \lambda_{j}$ (pseudo-inverse).
4. **Inverse transform** — apply $V^{(d)}$ direction by direction.

Steps 2 and 4 are parallelised over independent fibres with [Rayon](https://github.com/rayon-rs/rayon).

---

## Boundary conditions

| BC          | Active nodes            | $N_\mathrm{dof}$ | Enforced condition              |
|-------------|-------------------------|------------------|---------------------------------|
| `Periodic`  | $0, \ldots, r-1$        | $r$              | $u(x_l) = u(x_l+L)$            |
| `Dirichlet` | $1, \ldots, r-1$        | $r-1$            | $u(x_l) = u(x_l+L) = 0$        |
| `Neumann`   | $0, \ldots, r$          | $r+1$            | $u'(x_l) = u'(x_l+L) = 0$      |

---

## Quick start

```toml
[dependencies]
lobatto-spectral = { path = "path/to/lobatto-spectral" }
```

```rust
use lobatto_spectral::solver::{BoundaryCondition, PoissonND};
use std::f64::consts::PI;

// Solve (M + K) u = f on [0, 1]^3 with a single element of order r=20 per direction.
let solver = PoissonND::<3>::new(
    [1.0; 3],                          // domain lengths  L_d
    [0.0; 3],                          // left boundaries x_{l,d}
    [20;  3],                          // polynomial orders r_d
    1.0, 1.0,                          // α, β
    [BoundaryCondition::Periodic; 3],  // boundary conditions
);

let x = solver.get_x(); // tensor-product GLL grid, length 20^3 = 8 000

let w = 2.0 * PI;
let mut f: Vec<f64> = x
    .iter()
    .map(|&[x0, x1, x2]| {
        // example: u = exp(sin(wx0)) * exp(sin(wx1)) * exp(sin(wx2))
        // evaluate the right-hand side α u - β Δu here
        0.0
    })
    .collect();

solver.solve_in_place(&mut f); // f now contains the solution u at the GLL nodes
```

For a complete 5-D benchmark see [`examples/laplace_5d.rs`](examples/laplace_5d.rs).

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
| [`lobatto`](../lobatto) | GLL quadrature, collocation basis, differentiation and interpolation |
| [`nalgebra`](https://nalgebra.org) | Eigendecomposition and matrix-vector products |
| [`rayon`](https://github.com/rayon-rs/rayon) | Data-parallel fibre sweeps |

---

## License

Licensed under the **GNU Lesser General Public License v3.0** (LGPL-3.0-only).
See [lgpl-3.0.txt](lgpl-3.0.txt) for the full license text.
