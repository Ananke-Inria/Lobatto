//! # QuadratiX
//!
//! A high-order spectral-element library for the fast solution of the
//! **screened Poisson equation**
//! $$(\alpha M + \beta K)\, u = f$$
//! on tensor-product domains in arbitrary dimension $N$.
//!
//! $M$ denotes the GLL mass matrix, $K = D^T M D$ the stiffness matrix built from
//! the Gauss–Lobatto–Legendre differentiation matrix $D$, and $\alpha$, $\beta$
//! are real scalar coefficients.  Setting $\alpha = 0$ recovers the Laplacian;
//! setting $\beta = 0$ recovers a weighted $L^2$-projection.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`quadrature`] | Gauss–Legendre (GL) and Gauss–Lobatto–Legendre (GLL) nodes and weights on $[-1,1]$ and $[0,1]$. |
//! | [`basis`] | Multi-dimensional GLL/GL differentiation and mass-matrix operators on tensor-product grids. |
//! | [`utilities`] | Lagrange interpolation in barycentric form (internal). |
//! | [`hc`] | Hard-coded SIMD-friendly differentiation kernels for $Q_3$–$Q_6$ elements. |
//! | [`hofft`] | **HoFFT** — $n$ elements of order $r$ per direction; solves via FFT diagonalisation over inter-element coupling modes. |
//! | [`hot`] | **HoT** — single spectral element of order $r$ per direction; solves via a direct eigendecomposition of the local symbol. |
//!
//! ## Choosing a solver
//!
//! | Solver | When to use |
//! |--------|-------------|
//! | [`hot::PoissonSolver1D`] / [`hot::PoissonSolverND`] | Single-element, very-high-order problems or when $h$-refinement is not needed. Supports Periodic, Dirichlet, and Neumann BCs. Cost: $O(r^{2N})$ build, $O(r^{2N})$ solve. |
//! | [`hofft::PoissonSolver1D`] / [`hofft::PoissonSolverND`] | Multi-element problems with $n \gg 1$ and moderate order $r$. Cost: $O(n r^2 \log n)$ per solve. |

pub mod quadrature;

pub mod basis;

pub(crate) mod utilities;

pub mod hc;

pub mod hofft;

pub mod hot;
