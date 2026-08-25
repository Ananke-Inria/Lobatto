//! # lobatto-fft
//!
//! High-order FFT (**HoFFT**) on Gauss–Lobatto grids and corresponding high-order solver for Poisson problems.
//!
//! The solver solves $(\alpha I - \beta \Delta)~u = f$ for complex-valued right-hand side on a tensor-product domain
//! $\prod_{d=0}^{N-1} [x_{l,d}, ~ x_{l,d} + L_d]$
//! using a grid based on Gauss–Lobatto points with `n_d` elements of polynomial
//! order `r_d` per direction `d`.
//!
//! The method is based on the **High-order FFT (HoFFT)** of Caforio & Imperiale (2019):
//! A single FFT of length `n_d`
//! (or `2·n_d` for non-periodic BCs) per direction decouples the Fourier modes.
//! Each decoupled mode tuple $(k_0, \ldots, k_{N-1})$ then requires inverting an
//! independent `r × r` Hermitian system whose eigendecomposition is precomputed
//! at setup time, giving an `O(N_dof·log N_dof)`
//! solver with small constants.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`hofft`] | HoFFT transform engine: [`hofft::Engine`] (1D) and [`hofft::EngineND`] (ND).|
//! | [`solver`] | Screened Poisson solvers: [`solver::Poisson`] (1D) and [`solver::PoissonND`] (ND). Supports `Periodic`, `Dirichlet`, and `Neumann` boundary conditions per direction. |
//!
//! Grid geometry and interpolation (point evaluation, convection, precomputed
//! evaluation plans) live in the [`lobatto_grid`] crate; engines and solvers
//! expose the grid they are built on via their `grid()` accessors.
//!
//! ## Quick start
//!
//! ```rust
//! use lobatto_fft::solver::*;
//! use rustfft::num_complex::Complex;
//! use std::f64::consts::PI;
//!
//! // Solve (α I - β Δ) u = f on [0,1]^2 with n=8 elements of order r=4 per direction
//! // with α = 1, β = 1 and periodic boundary conditions.
//! let solver = PoissonND::<2>::new(
//!     [1.0; 2],                          // domain lengths L_d
//!     [0.0; 2],                          // left boundaries x_{l,d}
//!     [8;   2],                          // elements per direction n_d
//!     [4;   2],                          // polynomial orders r_d
//!     [BoundaryCondition::Periodic; 2],  // boundary conditions
//! );
//!
//! let x = solver.get_x(); // Gauss-Lobatto grid, length (8·4)² = 1024
//!
//! let mut f: Vec<Complex<f64>> = x
//!     .iter()
//!     .map(|&[xi, yi]| {
//!         // evaluate the right-hand side f(x,y) here (= sin(pi x) sin(pi y) for the example)
//!         Complex::new((PI * xi).sin() * (PI * yi).sin(), 0.0)
//!     })
//!     .collect();
//!
//! solver.solve_in_place(&mut f, 1.0, 1.0); // f now contains the solution u at the Gauss-Lobatto nodes
//! ```
//!
//! The coefficients `α`, `β` are arguments of the solve, not of the solver:
//! nothing precomputed depends on them, so one solver serves every operator
//! on its grid. They may also be complex — e.g. a damped Helmholtz problem
//! `(-ω² + iωc) M + K` — by passing `Complex<f64>` values:
//!
//! ```rust
//! # use lobatto_fft::solver::*;
//! # use rustfft::num_complex::Complex;
//! let solver = PoissonND::<2>::new([1.0; 2], [0.0; 2], [8; 2], [4; 2],
//!     [BoundaryCondition::Dirichlet; 2]);
//! let mut f = vec![Complex::new(1.0, 0.0); solver.get_x().len()];
//! let c = 0.5;
//! for omega in [1.0, 2.0, 3.0] {
//!     // frequency sweep on one solver
//!     solver.solve_in_place(&mut f, Complex::new(-omega * omega, omega * c), 1.0);
//! }
//! ```
//!
//! ## Operator functions
//!
//! The solve diagonalises `L = α I + β M⁻¹K` mode by mode, so any function of
//! the operator costs exactly one solve (two transform passes and the per-mode
//! `r^N × r^N` products): the `apply_*` methods multiply by `g(λ)` instead of
//! dividing by `λ`. With per-direction eigenvalues $\mu^{(d)}$ of the 1D
//! operators $L_d = M_d^{-1} K_d$,
//! $$g(L)\,f \quad\text{at}\quad \lambda = \alpha + \beta \sum_d \mu^{(d)}, \qquad
//! \Bigl(\bigotimes_d g_d(L_d)\Bigr) f \quad\text{at}\quad \prod_d g_d(\mu^{(d)}),$$
//! the second identity being exact because the $L_d$ act on different tensor
//! factors. This gives, e.g., the directionally split implicit Euler step
//! $\prod_d (I + \Delta t\, L_d)^{-1}$ — which inherits the one-dimensional
//! discrete maximum principle — and the exact semigroup $\exp(-tL)$.
//!
//! ```rust
//! # use lobatto_fft::solver::*;
//! # use rustfft::num_complex::Complex;
//! # let solver_nd = PoissonND::<2>::new([1.0; 2], [0.0; 2], [4; 2], [3; 2],
//! #     [BoundaryCondition::Periodic; 2]);
//! # let mut u = vec![Complex::new(1.0, 0.0); solver_nd.get_x().len()];
//! # let (dt, t) = (0.01, 0.1);
//! let (alpha, beta) = (0.0, 1.0);
//! // implicit Euler, unsplit, one application
//! solver_nd.apply_in_place(&mut u, alpha, beta, |lam| 1.0 / (1.0 + dt * lam));
//! // implicit Euler, directionally split (α = 0 here), one application
//! solver_nd.apply_in_place_dirs(&mut u, |mu| {
//!     Complex::from(mu.iter().map(|&m| 1.0 / (1.0 + dt * beta * m)).product::<f64>())
//! });
//! // exact semigroup
//! solver_nd.apply_in_place(&mut u, alpha, beta, |lam| (-t * lam).exp());
//! // the same three, without closures
//! solver_nd.euler_step_in_place(&mut u, alpha, beta, dt);
//! solver_nd.euler_split_step_in_place(&mut u, alpha, beta, dt);
//! solver_nd.exp_in_place(&mut u, alpha, beta, t);
//! ```
//!
//! **Input convention.** `solve_*` take the strong-form right-hand side and
//! return `(αM + βK)⁻¹ f`; `apply_*` take the **nodal vector** and return
//! `g(αI + βM⁻¹K) f` — with `g(λ) = 1/λ` that is `(αM + βK)⁻¹ M f`.
//!
//! **Pseudo-inverse cutoff.** The solves annihilate eigenvalues with
//! `|λ| ≤ pinv_rel_tol · (|α| + |β|·max|μ|)` (the roundoff constant mode of a
//! pure Neumann/periodic Laplacian); the default
//! [`solver::DEFAULT_PINV_REL_TOL`]` = 1e-12` can be changed with
//! `with_pinv_rel_tol` / `set_pinv_rel_tol`; `pinv_tol(alpha, beta)` returns
//! the absolute cutoff. The `apply_*` methods apply no cutoff — `g` decides
//! what happens at `λ = 0`.

pub mod hofft;
pub mod solver;
