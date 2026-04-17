# lobatto

Gauss–Lobatto and Gauss–Legendre quadrature, tensor-product Lagrangian collocation bases, and spectral differentiation operators for high-order finite-element methods in Rust.

## Modules

| Module | Description |
|---|---|
| `quadrature` | GL and GLL nodes and weights on `[-1, 1]` and `[0, 1]` |
| `quadrature_hc` | Hard-coded GLL nodes and weights for n = 2..14, certified to 1e-15 |
| `collocation` | Multi-dimensional tensor-product collocation basis (quadrature, differentiation, interpolation) |
| `collocation_hc` | Hard-coded SIMD-friendly differentiation kernels for Q3–Q6 elements in up to 3D |
| `utilities` | Lagrange interpolation in barycentric form |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
lobatto = "0.1"
```

## Examples

### Quadrature

Integrate `x^8` over `[-1, 1]` with a 9-point GLL rule (exact for polynomials up to degree 15):

```rust
use lobatto::quadrature::gauss_lobatto;

let (points, weights) = gauss_lobatto(9);
let integral: f64 = points.iter().zip(&weights).map(|(&x, &w)| w * x.powi(8)).sum();
assert!((integral - 2.0 / 9.0).abs() < 1e-15);
```

### Collocation basis

Build a 2D Q4 tensor-product basis (5×5 GLL nodes), differentiate a function, and evaluate the interpolant at an arbitrary point:

```rust
use lobatto::collocation::{CollocationBasis, Gauss};

// 2D Q4 element: 5×5 GLL nodes on [0,1]^2
let basis = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto)]);

// DOF values: f(x, y) = x^2 + y^2 at the collocation points
let u: Vec<f64> = basis.points().iter().map(|p| p[0].powi(2) + p[1].powi(2)).collect();

// Differentiate along x 
let mut du_dx = vec![0.0; u.len()];
basis.diff(0, &u, &mut du_dx);

// Evaluate the interpolant at an arbitrary point
let vals = basis.eval(&[vec![0.3, 0.7]], &u);
let expected = 0.3f64.powi(2) + 0.7f64.powi(2);
assert!((vals[0] - expected).abs() < 1e-13);

// Evaluate the interpolant derivative at an arbitrary point
let vals = basis.eval(&[vec![0.7, 0.3]], &du_dx);
let expected = 2.0*0.7;
assert!((vals[0] - expected).abs() < 1e-13);
```

## Dependencies

- [nalgebra](https://nalgebra.org) — dense linear algebra (eigendecomposition, matrix views)
- [rayon](https://github.com/rayon-rs/rayon) — data-parallel evaluation via `par_iter`

## License

Licensed under the **GNU Lesser General Public License v3.0** (LGPLv3).
