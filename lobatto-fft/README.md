# quadratix

A Rust library for high-order numerical methods based on Gauss-Lobatto and Gauss-Legendre quadrature. 

## Features

- Computation of quadrature points and weights.
- Differentiation and interpolation of functions in a Lagrangian basis on Gauss-Lobatto points.
- High-order spectral-element solvers for the periodic Laplace equation on an N-dimensional unit hypercube.

## Modules

| Module | Description |
|---|---|
| `quadrature` | Gauss and Gauss-Lobatto quadrature points and weights |
| `Basis` | Multi-dimensional GLL differentiation and mass operators |
| `hc_calculations` | Hard-coded, SIMD-friendly differentiation kernels for Q3–Q6 elements |
| `hofft_solver` | HoFFT solver: `n` elements × order `r` per direction |
| `hot` | HoT solver: single spectral element, order `r` per direction |

## Getting started

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
quadratix = { path = "path/to/quadratix" }
```

### Quick example

```rust
use quadratix::hofft::solver::PoissonSolverND;

// 3-D solver: 8 elements × order 4, domain [0, 1]^3
let alpha = 1.0;
let beta  = 1.0;
let solver = PoissonSolverND::<3>::new(
    [1.0; 3],   // domain lengths
    [0.0; 3],   // domain origins
    [8;   3],   // number of elements per direction
    [4;   3],   // polynomial order per direction
    alpha,
    beta,
);

let x = solver.get_x();          // GLL nodes, shape [n_dof][3]
let mut f = vec![0.0; x.len()];  // fill with right-hand side values
solver.solve_in_place(&mut f);   // f is overwritten with the solution
```

See [`examples/laplace_5d.rs`](examples/laplace_5d.rs) for a full 5-D benchmark comparing HoFFT and HoT.

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

## Dependencies

- [rustfft](https://github.com/ejmahler/RustFFT) — FFT-based diagonalization
- [nalgebra](https://nalgebra.org) — linear algebra primitives
- [rayon](https://github.com/rayon-rs/rayon) — data-parallel iterators

## License

This library is licensed under the **GNU Lesser General Public License v3.0** (LGPLv3).
See [lgpl-3.0.txt](lgpl-3.0.txt) for the full license text.
