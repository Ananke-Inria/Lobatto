# lobatto-fft

High-order FFT (**HoFFT**) on Gauss–Lobatto grids and corresponding high-order solver for Poisson problems.

Solves (αI − βΔ) u = f on tensor-product domains using a spectral-finite element discretisation with n_d elements of polynomial order r_d per direction, for any combination of periodic, Dirichlet, and Neumann boundary conditions.
The solver runs in O(N_dof log N_dof) after a one-time setup.

---

## Method

The method is based on the  HoFFT algorithm of Caforio & Imperiale (2019). It uses a single FFT per direction and per type of nodes to decouple all Fourier modes, reducing the global system to a collection of independent
r × r Hermitian problems — the **local symbols** S(k) — whose
eigendecompositions are precomputed at setup time.

Each call to `solve_in_place` runs four steps:

1. **Mass scaling** — f ← M f where M is the
   diagonal mass matrix.
2. **Forward HoFFT** — apply [`hofft::EngineND::forward`] direction by direction
   (sum factorisation, Rayon-parallel).
3. **Per-mode symbol solve** — for each mode tuple (k_0, ..., k_{N-1}):

   û_k = V_k (αI + β Λ_k)^{-1} V_k† f̂_k

   where V_k and Λ_k are the precomputed eigenvectors and eigenvalues of the
   per-direction symbols (Rayon-parallel over mode tuples).
4. **Inverse HoFFT** — apply [`hofft::EngineND::inverse`] direction by direction.

Non-periodic BCs are handled by an odd (Dirichlet) or even (Neumann) extension of the
domain to length 2n before the FFT, retaining only the independent Fourier coefficients
(Lemma 3.2 of \[1\]).

**Reference** — \[1\] Caforio, F. and Imperiale, S. (2019). *High-order discrete Fourier
transform for the solution of the Poisson equation*. SIAM Journal on Scientific
Computing, 41(5), A2747–A2771.

---

## Boundary conditions

| BC          | Extension | n_fft | N_dof per direction |
|-------------|-----------|-------|---------------------|
| `Periodic`  | none      | n     | n·r                 |
| `Dirichlet` | Odd       | 2n    | n·r − 1             |
| `Neumann`   | Even      | 2n    | n·r + 1             |

Each spatial direction carries an independent boundary condition, enabling mixed
configurations (e.g. Dirichlet in x, Neumann in y).

---

## Quick start

```toml
[dependencies]
lobatto-fft = "0.4"
```

```rust
use lobatto_fft::solver::{BoundaryCondition, PoissonND};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

// Solve (I α -  β Δ) u = f on [0, 1]^2 with n=8 elements of order r=4 per direction.
let solver = PoissonND::<2>::new(
    [1.0; 2],                          // domain lengths  
    [0.0; 2],                          // left boundaries 
    [8;   2],                          // elements per direction n_d
    [4;   2],                          // polynomial orders r_d
    [BoundaryCondition::Periodic; 2],  // boundary conditions
);

let x = solver.get_x(); // tensor-product GLL grid, length (8·4)² = 1024

let mut f: Vec<Complex<f64>> = x
    .iter()
    .map(|&[xi, yi]| {
        // evaluate the right-hand side α u(xi, yi) − β Δu(xi, yi) here
        Complex::new((PI * xi).sin() * (PI * yi).sin(), 0.0)
    })
    .collect();

solver.solve_in_place(&mut f, 1.0, 1.0); // α = 1, β = 1; f now contains u at the GLL nodes
```

For a complete 5-D accuracy and timing benchmark see
[`examples/laplace_5d.rs`](https://github.com/OndoMathX/QuadratiX/blob/main/lobatto-fft/examples/laplace_5d.rs).

---

## API

### `solver::BoundaryCondition`

```rust
pub enum BoundaryCondition { Periodic, Dirichlet, Neumann }
```

### `solver::Poisson` — 1D solver

```rust
let solver = Poisson::new(l, xl, n, r, bc);
let x: Vec<f64>         = solver.get_x();          // N_dof GLL nodes
solver.solve_in_place(&mut f, alpha, beta);         // overwrites f with u; alpha, beta: f64 or Complex<f64>
// access the underlying engine for interpolation or advection:
let val = solver.planner.eval_at(x_pt, &f);          // single point
let mut vals = vec![Complex::default(); pts.len()];
solver.planner.eval(&pts, &f, &mut vals);            // batch, caller-provided buffer
```

### `solver::PoissonND<N>` — N-dimensional solver

```rust
let solver = PoissonND::<N>::new(ls, xls, ns, rs, bcs);
let x: Vec<[f64; N]>    = solver.get_x();          // ∏_d N_dof^(d) grid points
solver.solve_in_place(&mut f, alpha, beta);         // overwrites f with u
// interpolate at arbitrary off-grid points:
let mut vals = vec![Complex::default(); pts.len()];
solver.fft.eval(&pts, &f, &mut vals);
// advect the solution by a flow map λ (caller-provided output buffer):
let mut f_advected = vec![Complex::default(); f.len()];
solver.fft.convect(|x| lambda(x), &f, &mut f_advected);
```

The coefficients `alpha`, `beta` are arguments of the solve, not of the solver: nothing
precomputed depends on them, so one solver serves every operator on its grid (varying
time steps, frequency sweeps, complex-shifted resolvents). They may be real (`f64`) or
complex (`Complex<f64>`) — e.g. `Complex::new(-omega * omega, omega * c)` for a damped
Helmholtz problem — at no extra cost: they only enter the per-mode eigenvalues
`α + β·μ`, the eigendecompositions stay Hermitian.
`solve_in_place_real(&mut f, alpha: f64, beta: f64)` is the real-in/real-out variant.

### Operator functions

The per-mode diagonalisation makes any function of `L = α I + β M⁻¹K` cost exactly
one solve — same transforms, same eigenvector products, a multiplication by `g(λ)`
instead of a division by `λ`:

- `apply_in_place(&mut u, alpha, beta, g)` computes `u ← g(L) u`, `g` evaluated at the
  tensor eigenvalue `λ = α + β Σ_d μ^(d)` (complex when `α`, `β` are);
- `apply_in_place_dirs(&mut u, g)` (ND) hands `g` the per-direction eigenvalues
  `(μ^(0), …, μ^(N-1))`, giving any tensor product `⊗_d g_d(L_d)` — exact, because
  the 1D operators act on different tensor factors;
- `apply_mu_in_place` (1D) hands `g` the raw stiffness eigenvalue `μ`;
- `apply_in_place_real` is the real-in/real-out wrapper.

```rust
// implicit Euler, unsplit, one application
solver.apply_in_place(&mut u, alpha, beta, |lam| 1.0 / (1.0 + dt * lam));
// implicit Euler, directionally split Π_d (I + dt β L_d)⁻¹ (α = 0), one application
solver_nd.apply_in_place_dirs(&mut u, |mu| {
    Complex::from(mu.iter().map(|&m| 1.0 / (1.0 + dt * beta * m)).product::<f64>())
});
// exact semigroup exp(-t L)
solver.apply_in_place(&mut u, alpha, beta, |lam| (-t * lam).exp());
```

Conveniences without closures: `euler_step_in_place(&mut u, alpha, beta, dt)`,
`euler_split_step_in_place(&mut u, alpha, beta, dt)` (ND; the split step satisfies the
1D discrete maximum principle in every direction, `α` split evenly as `α/N`) and
`exp_in_place(&mut u, alpha, beta, t)`. `m` sub-steps `(1 + dt λ/m)^{-m}` or the exact
exponential are therefore cheaper than `m` solves.

**Input convention** — `solve_*` take the strong-form right-hand side and return
`(αM + βK)⁻¹ f`; `apply_*` take the **nodal vector** and return `g(αI + βM⁻¹K) f`
(with `g(λ) = 1/λ`, that is `(αM + βK)⁻¹ M f`).

**Pseudo-inverse cutoff** — the solves annihilate eigenvalues with
`|λ| ≤ pinv_rel_tol · (|α| + |β| max|μ|)`, which filters the roundoff constant
mode of a pure periodic/Neumann Laplacian. Default `DEFAULT_PINV_REL_TOL = 1e-12`;
change it with `with_pinv_rel_tol(tol)` / `set_pinv_rel_tol(tol)`, inspect the
absolute value with `pinv_tol(alpha, beta)` and the spectral radius with `mu_max()`.
The `apply_*` methods apply no cutoff.

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
| [`lobatto`](https://crates.io/crates/lobatto) | GLL quadrature, collocation basis, differentiation and interpolation |
| [`rustfft`](https://github.com/ejmahler/RustFFT) | Forward/inverse FFT plans |
| [`nalgebra`](https://nalgebra.org) | Eigendecomposition and matrix-vector products |
| [`rayon`](https://github.com/rayon-rs/rayon) | Data-parallel fibre sweeps and mode-tuple loop |

---

## License

Licensed under the **GNU Lesser General Public License v3.0** (LGPL-3.0-only).
See [lgpl-3.0.txt](lgpl-3.0.txt) for the full license text.
