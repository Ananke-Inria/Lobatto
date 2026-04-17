//! # lobatto-fft
//!
//! High-order FFT (**HoFFT**) on Gauss–Lobatto grids and corresponding high-order solver for Poisson problems.
//!
//! The solver solves $(\alpha I - \beta \Delta)~u = f$ for complex-valued right-hand side on a tensor-product domain
//! $\prod_{d=0}^{N-1} [x_{l,d}, ~ x_{l,d} + L_d]$
//! using a grid based on Gauss–Lobatto points with $n_d$ elements of polynomial
//! order $r_d$ per direction $d$.
//!
//! The method is based on the **High-order FFT (HoFFT)** of Caforio & Imperiale (2019):
//! A single FFT of length $n_d$
//! (or $2n_d$ for non-periodic BCs) per direction decouples the Fourier modes.
//! Each decoupled mode tuple $(k_0, \ldots, k_{N-1})$ then requires inverting an
//! independent $r \times r$ Hermitian system whose eigendecomposition is precomputed
//! at setup time, giving an $\mathcal{O}(N_\mathrm{dof} \log N_\mathrm{dof})$
//! solver with small constants.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`hofft`] | HoFFT transform engine: [`hofft::Engine`] (1D) and [`hofft::EngineND`] (ND).|
//! | [`solver`] | Screened Poisson solvers: [`solver::Poisson`] (1D) and [`solver::PoissonND`] (ND). Supports `Periodic`, `Dirichlet`, and `Neumann` boundary conditions per direction. |
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
//!     1.0, 1.0,                          // α, β
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
//! solver.solve_in_place(&mut f); // f now contains the solution u at the Gauss-Lobatto nodes
//! ```

pub mod hofft;
pub mod solver;
