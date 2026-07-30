# lobatto-grid

Multi-element Gauss–Lobatto grids on tensor-product domains: geometry, boundary extensions, and high-order Lagrange interpolation.

`Grid1D` describes n Gauss–Lobatto finite elements of polynomial order r on an arbitrary interval [x_l, x_l + L]; `GridND<N>` is the tensor product of N independent 1D grids. Each direction carries an `Extension` (`Periodic`, `Odd`/Dirichlet, `Even`/Neumann) that fixes the DOF layout and lets interpolation fold arbitrary query positions back into the domain.

Provides:

- `eval` / `eval_extended`: batched high-order interpolation of DOF vectors (sum-factored, element-grouped, Rayon-parallel),
- `convect`: interpolation at the image of every grid node under a flow map,
- `EvalPlan`: precomputed plans for repeated evaluation at fixed points with varying data — the apply is allocation-free gather + contraction,
- generic scalar support (`f64` and `Complex<f64>` share code paths and plans).

Used by [`lobatto-fft`](https://crates.io/crates/lobatto-fft) as the grid layer under its HoFFT transforms and screened Poisson solvers.

Part of the [QuadratiX](https://github.com/OndoMathX/QuadratiX) workspace.
