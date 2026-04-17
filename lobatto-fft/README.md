# lobatto-fft

High-order FFT (**HoFFT**) on Gauss–Lobatto grids and corresponding high-order solver for Poisson problems.

Solves $(\alpha I - \beta \Delta)~ u = f$ on tensor-product domains
$\prod_{d=0}^{N-1} [x_{l,d},\, x_{l,d} + L_d]$
using a spectral-finite element discretisation with $n_d$ elements of polynomial order $r_d$ per direction,
for any combination of periodic, Dirichlet, and Neumann boundary conditions.
The solver runs in $\mathcal{O}(N_\mathrm{dof} \log N_\mathrm{dof})$ after a one-time setup.

For the single-element limit ($n_d = 1$, all FFT overhead removed), see
[`lobatto-spectral`](../lobatto-spectral/README.md).

---

## Method

The HoFFT algorithm of Caforio & Imperiale (2019) exploits the (near-)circulant
inter-element coupling of the GLL operator.  A single FFT per direction decouples
all Fourier modes, reducing the global system to a collection of independent
$r \times r$ Hermitian problems — the **local symbols** $S(k)$ — whose
eigendecompositions are precomputed at setup time.

Each call to `solve_in_place` runs four steps:

1. **Mass scaling** — $f \leftarrow \tilde{M}\, f$ where $\tilde{M}$ is the
   diagonal periodic mass matrix.
2. **Forward HoFFT** — apply [`hofft::EngineND::forward`] direction by direction
   (sum factorisation, Rayon-parallel).
3. **Per-mode symbol solve** — for each mode tuple $(k_0,\ldots,k_{N-1})$:
   $$\hat{u}_k = V_k \left(\alpha I + \beta \Lambda_k\right)^{-1} V_k^\dagger \hat{f}_k$$
   where $V_k$ and $\Lambda_k$ are the precomputed eigenvectors and eigenvalues of the
   per-direction symbols (Rayon-parallel over mode tuples).
4. **Inverse HoFFT** — apply [`hofft::EngineND::inverse`] direction by direction.

Non-periodic BCs are handled by an odd (Dirichlet) or even (Neumann) extension of the
domain to length $2n$ before the FFT, retaining only the independent Fourier coefficients
(Lemma 3.2 of \[1\]).

**Reference** — \[1\] Caforio, F. and Imperiale, S. (2019). *High-order discrete Fourier
transform for the solution of the Poisson equation*. SIAM Journal on Scientific
Computing, 41(5), A2747–A2771.

---

## Boundary conditions

| BC          | Extension | $n_\mathrm{fft}$ | $N_\mathrm{dof}$ per direction |
|-------------|-----------|------------------|-------------------------------|
| `Periodic`  | none      | $n$              | $n r$                         |
| `Dirichlet` | Odd       | $2n$             | $n r - 1$                     |
| `Neumann`   | Even      | $2n$             | $n r + 1$                     |

Each spatial direction carries an independent boundary condition, enabling mixed
configurations (e.g. Dirichlet in $x$, Neumann in $y$).

---

## Quick start

```toml
[dependencies]
lobatto-fft = { path = "path/to/lobatto-fft" }
rustfft = "6"
```

```rust
use lobatto_fft::solver::{BoundaryCondition, PoissonND};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

// Solve (M + K) u = f on [0, 1]^2 with n=8 elements of order r=4 per direction.
let solver = PoissonND::<2>::new(
    [1.0; 2],                          // domain lengths  L_d
    [0.0; 2],                          // left boundaries x_{l,d}
    [8;   2],                          // elements per direction n_d
    [4;   2],                          // polynomial orders r_d
    1.0, 1.0,                          // α, β
    [BoundaryCondition::Periodic; 2],  // boundary conditions
);

let x = solver.get_x(); // tensor-product GLL grid, length (8·4)² = 1024

let w = 2.0 * PI;
let mut f: Vec<Complex<f64>> = x
    .iter()
    .map(|&[xi, yi]| {
        // evaluate the right-hand side α u(xi, yi) − β Δu(xi, yi) here
        Complex::new((PI * xi).sin() * (PI * yi).sin(), 0.0)
    })
    .collect();

solver.solve_in_place(&mut f); // f now contains the solution u at the GLL nodes
```

For a complete 5-D accuracy and timing benchmark see
[`examples/laplace_5d.rs`](examples/laplace_5d.rs).

---

## API

### `solver::BoundaryCondition`

```rust
pub enum BoundaryCondition { Periodic, Dirichlet, Neumann }
```

### `solver::Poisson` — 1D solver

```rust
let solver = Poisson::new(l, xl, n, r, alpha, beta, bc);
let x: Vec<f64>         = solver.get_x();          // N_dof GLL nodes
solver.solve_in_place(&mut f);                      // overwrites f with u
// access the underlying engine for interpolation or advection:
let val = solver.planner.eval(x_pt, &f);
```

### `solver::PoissonND<N>` — N-dimensional solver

```rust
let solver = PoissonND::<N>::new(ls, xls, ns, rs, alpha, beta, bcs);
let x: Vec<[f64; N]>    = solver.get_x();          // ∏_d N_dof^(d) grid points
solver.solve_in_place(&mut f);                      // overwrites f with u
// interpolate at arbitrary off-grid points:
let mut vals = vec![Complex::default(); pts.len()];
solver.fft.eval(&pts, &f, &mut vals);
// advect the solution by a flow map λ:
let f_advected = solver.fft.convect(|x| lambda(x), &f);
```

### `hofft::Engine` / `hofft::EngineND<N>` — transform engines

The engines are exposed directly through `solver.planner` (1D) and `solver.fft` (ND),
but can also be constructed independently for use without the Poisson solver:

```rust
use lobatto_fft::hofft::{Engine, EngineND, Extension};

let engine = Engine::new(n, r, l, xl, Extension::Periodic);
engine.forward(&mut buf);   // in-place forward HoFFT
engine.inverse(&mut buf);   // in-place inverse HoFFT
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
| [`rustfft`](https://github.com/ejmahler/RustFFT) | Forward/inverse FFT plans |
| [`nalgebra`](https://nalgebra.org) | Eigendecomposition and matrix-vector products |
| [`rayon`](https://github.com/rayon-rs/rayon) | Data-parallel fibre sweeps and mode-tuple loop |

---

## License

Licensed under the **GNU Lesser General Public License v3.0** (LGPL-3.0-only).
See [lgpl-3.0.txt](lgpl-3.0.txt) for the full license text.
