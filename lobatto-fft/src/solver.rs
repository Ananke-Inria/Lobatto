//! HoFFT screened Poisson solvers.
//!
//! Provides [`Poisson`] and [`PoissonND`], which solve
//! $(\alpha M + \beta K)\, u = f$ on tensor-product Gauss-Lobatto grids using
//! the HoFFT transform machinery from [`crate::hofft`].
//!
//! The coefficients $\alpha$ and $\beta$ are **per-call parameters** of the
//! solves, not of the solver: nothing precomputed depends on them, so one
//! solver serves every operator on its grid (varying time steps, frequency
//! sweeps). They may be real or complex (e.g. $\alpha = -\omega^2 + i\omega c$
//! for damped time-harmonic problems); the solve methods accept anything
//! convertible into `Complex<f64>`, so plain `f64` arguments work.
//!
//! Both solvers are set up once and reused for many right-hand sides.
//! Construction precomputes the FFT plans and the eigendecomposition of
//! all independent local stiffness symbols; each call to `solve_in_place`
//! runs in `O(N_dof·log N_dof)` arithmetic operations.
//!
//! | Type | Description |
//! |------|-------------|
//! | [`BoundaryCondition`] | Selects `Periodic`, `Dirichlet`, or `Neumann` for each direction. |
//! | [`Poisson`] | 1D solver; single-threaded symbol solve. |
//! | [`PoissonND`] | `N`-dimensional solver; Rayon-parallel symbol solve. |
//!
//! ## Boundary conditions and extension
//!
//! Non-periodic BCs are handled by extending the `n`-element physical domain to a
//! `2n`-element periodic domain with odd (Dirichlet) or even (Neumann) symmetry,
//! applying a standard FFT of length `2n`, then retaining only the
//! `N_dof` independent Fourier coefficients; see [`crate::hofft::Extension`]
//! and Caforio & Imperiale (2019), SIAM JSC 41(5):
//!
//! | BC          | Extension | `n_fft` | `N_dof` |
//! |-------------|-----------|------------------|------------------|
//! | `Periodic`  | none      | `n`              | `n·r`            |
//! | `Dirichlet` | Odd       | `2n`             | `n·r - 1`        |
//! | `Neumann`   | Even      | `2n`             | `n·r + 1`        |
//!
//!
//! ## Data layout
//!
//! The **physical DOF buffer** (length `N_dof`) is in `x`-increasing order:
//!
//! | BC          | Flat index of node `j` in element `k` |
//! |-------------|---------------------------------------|
//! | `Periodic`  | `k·r + j`                             |
//! | `Dirichlet` | `k·r + j - 1` (left boundary excluded) |
//! | `Neumann`   | `k·r + j`, plus the right endpoint at `n·r` |

use crate::hofft::{Engine, EngineND, Extension};
use lobatto_grid::grid::{nd_fiber_base, nd_multi_index, nd_multi_indices};
use lobatto::collocation::{CollocationBasis, Gauss};
use nalgebra::*;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

/// Boundary condition for one spatial direction of the screened Poisson problem.
///
/// Controls how the physical domain `[x_l, x_l + L]` is extended before the FFT
/// and determines the number of physical DOFs `N_dof`
/// (see [`crate::hofft::Extension`]).
/// Each direction may carry an independent BC, enabling mixed configurations in ND.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Periodic identification: `u(x_l) = u(x_l + L)`.
    ///
    /// Uses a plain FFT of length `n`, giving `N_dof = n·r` DOFs.
    Periodic,
    /// Homogeneous Dirichlet: `u(x_l) = u(x_l + L) = 0`.
    ///
    /// Uses an odd (anti-symmetric) extension to length `2n`.
    /// The two boundary nodes are Dirichlet zeros and not stored,
    /// giving `N_dof = n·r - 1` DOFs.
    Dirichlet,
    /// Homogeneous Neumann: `u'(x_l) = u'(x_l + L) = 0`.
    ///
    /// Uses an even (symmetric) extension to length `2n`.
    /// Both boundary nodes are included in the DOF vector,
    /// giving `N_dof = n·r + 1` DOFs.
    Neumann,
}

impl BoundaryCondition {
    /// Convert to the corresponding [`Extension`] type.
    pub(crate) fn extension(self) -> Extension {
        match self {
            BoundaryCondition::Periodic => Extension::Periodic,
            BoundaryCondition::Dirichlet => Extension::Odd,
            BoundaryCondition::Neumann => Extension::Even,
        }
    }
}

// ── 1D solver ────────────────────────────────────────────────────────────────

/// 1D HoFFT screened Poisson solver.
///
/// Solves $(\alpha M + \beta K)\, u = f$ on `[x_l, x_l + L]` using a Gauss-Lobatto
/// spectral-element discretisation with `n` elements of polynomial order `r`.
///
/// ## Algorithm
///
/// The inter-element coupling matrix has a block-circulant structure (periodic)
/// or a near-circulant structure (Dirichlet/Neumann via odd/even extension).
/// A single FFT of length `n_fft` decouples the `n_fft` Fourier
/// modes, reducing the global `n·r × n·r` system to `n_fft/2 + 1`
/// independent **local symbols** $S(k)$ of size `r × r`.
///
/// Each $S(k)$ splits as $S(k) = \alpha \tilde{M} + \beta \tilde{K}(k)$, where
/// $\tilde{M}$ is the (phase-free, diagonal) periodic mass matrix and
/// $\tilde{K}(k)$ is the Hermitian stiffness symbol assembled from the periodic
/// element matrices modulated by the inter-element phase $e^{-2\pi i k / n}$:
/// $$\tilde{K}(k)_{ij} = \frac{1}{h}\bigl(K_{ij} + K_{ri}\,e^{-2\pi ik/n}\bigr), \quad i,j = 1,\ldots,r-1,$$
/// with appropriate modifications for the `(0, 0)` entry that couples the two sides
/// of an element interface.
///
/// The generalised eigendecomposition $\tilde{K}(k) V_k = \tilde{M} V_k \mathrm{diag}(\mu_k)$,
/// precomputed at build time, is independent of $\alpha$ and $\beta$; the
/// coefficients only enter the per-mode eigenvalues
/// $\lambda_{k,j} = \alpha + \beta\,\mu_{k,j}$ applied in the solve
/// $$\hat{u}_k = V_k \Lambda_k^{-1} V_k^\dagger \hat{f}_k.$$
/// This is what allows complex $\alpha$, $\beta$ with a purely Hermitian
/// eigensolver.
///
/// Physical DOF layout: `x`-increasing, flat index `k·r + j` for local node `j`
/// of element `k` (with offset adjustments for Dirichlet; see [`Engine`]).
pub struct Poisson {
    /// Polynomial order `r` per element.
    r: usize,

    /// HoFFT engine: handles extension, packing, forward/inverse FFT, and interpolation.
    pub planner: Engine,

    /// Real eigenvalues of the mass-normalised **stiffness** symbol for each
    /// independent Fourier mode `k`: $\mu_j(k)$ of
    /// $\tilde{M}^{-1/2} \tilde{K}(k) \tilde{M}^{-1/2}$, length `r`.
    eigenvalues: Vec<DVector<f64>>,
    /// $\tilde{M}$-orthonormal eigenvectors $V_k$ satisfying
    /// $\tilde{K}(k) V_k = \tilde{M} V_k \mathrm{diag}(\mu_k)$;
    /// `eigenvectors[k]` is the `r × r` complex matrix.
    eigenvectors: Vec<DMatrix<Complex<f64>>>,
    /// Diagonal of the **periodic** local mass matrix (length `r`):
    /// $\tilde{M}_{jj} = h w_j$, with DOF 0 accumulating contributions from both sides
    /// of the element interface: $\tilde{M}_{00} = 2h w_0$.
    periodic_mass_matrix: DVector<f64>,
    /// Spectral radius `max |μ|` of the stiffness symbols.
    mu_max: f64,
    /// Boundary condition type.
    bc: BoundaryCondition,
}

/// Work buffers of one 1D application ([`Poisson::apply_mu_with`]): the
/// engine's extended buffer and FFT scratch ([`Engine::work_lens`]), and two
/// per-mode vectors of length `r`. One per thread in the directional solves.
struct Work1D {
    buf: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    col: DVector<Complex<f64>>,
    v: DVector<Complex<f64>>,
}

/// Spectral radius `max |μ|` over a set of per-mode real eigenvalue vectors.
fn max_abs_eigenvalue(eigenvalues: &[DVector<f64>]) -> f64 {
    eigenvalues
        .iter()
        .flat_map(|l| l.iter())
        .fold(0.0f64, |m, &l| m.max(l.abs()))
}

/// Pseudo-inverse of the eigenvalue `lam`: `conj(λ)/|λ|²` (one real division,
/// no square root), or `0` when `λ` is exactly zero.
#[inline(always)]
fn pinv(lam: Complex<f64>) -> Complex<f64> {
    let norm_sqr = lam.norm_sqr();
    if norm_sqr > 0.0 {
        lam.conj() / norm_sqr
    } else {
        Complex::new(0.0, 0.0)
    }
}

/// Element-level data from which the `r × r` stiffness symbols are assembled.
///
/// The symbol of a mode with inter-element phase `φ` splits as
/// $$\tilde{K}(\varphi) = \tilde{K}(0) + D(\varphi),$$
/// where `K̃(0)` is the phase-free part (it annihilates the constant vector)
/// and `D(φ)` collects the terms multiplied by `cos φ − 1` and `e^{-iφ} − 1`.
/// `D` is formed from `sin(φ/2)` rather than by subtracting `1` from
/// `cos φ`, so that its entries, of order `φ` and `φ²`, carry full relative
/// precision even for the lowest modes where `φ ≪ 1`.
struct LocalSymbol {
    r: usize,
    /// Phase-free symbol `K̃(0)` (real symmetric, `r × r`).
    k0: DMatrix<f64>,
    /// `K_{r,i} / h` for `i = 0..=r`: the last row of the element stiffness
    /// matrix, which couples the two sides of an element interface.
    k_row_r: DVector<f64>,
    /// Diagonal of the periodic local mass matrix `M̃` (length `r`):
    /// `h·w_j`, with the interface node `j = 0` weighted `2h·w_0`.
    mass: DVector<f64>,
}

impl LocalSymbol {
    fn new(r: usize, h: f64, basis: &CollocationBasis) -> Self {
        let mass_matrix = basis.weights_matrix();
        let diff_matrix = basis.diff_matrix(0);
        // Element stiffness matrix K = Dᵀ M D on the reference element.
        let stiffness: DMatrix<f64> = diff_matrix.transpose() * &mass_matrix * diff_matrix;

        let mass = DVector::from_fn(r, |i, _| {
            if i == 0 {
                2.0 * h * mass_matrix[(0, 0)]
            } else {
                h * mass_matrix[(i, i)]
            }
        });
        let k_row_r = DVector::from_fn(r + 1, |i, _| stiffness[(r, i)] / h);

        // K̃(0): interior block K_{ij}/h; interface row/column accumulate the
        // contributions of both neighbouring elements with phase 1.
        let mut k0 = DMatrix::<f64>::zeros(r, r);
        for i in 1..r {
            for j in 1..r {
                k0[(i, j)] = stiffness[(i, j)] / h;
            }
        }
        k0[(0, 0)] = 2.0 * stiffness[(0, 0)] / h + 2.0 * k_row_r[0];
        for i in 1..r {
            k0[(0, i)] = stiffness[(0, i)] / h + k_row_r[i];
            k0[(i, 0)] = k0[(0, i)];
        }

        Self {
            r,
            k0,
            k_row_r,
            mass,
        }
    }

    /// Phase-dependent part `D(φ) = K̃(φ) − K̃(0)`, formed without cancellation.
    ///
    /// With `s = sin(φ/2)`, `c = cos(φ/2)`:
    /// `cos φ − 1 = −2s²` and `e^{-iφ} − 1 = −2s(s + ic)`, hence
    /// `D_00 = −4s²·K_{0r}/h`, `D_{0i} = −2s(s + ic)·K_{ri}/h`, `D_{i0} = conj(D_{0i})`,
    /// all other entries zero.
    fn phase_part(&self, phi: f64) -> DMatrix<Complex<f64>> {
        let (s, c) = (0.5 * phi).sin_cos();
        let mut d = DMatrix::<Complex<f64>>::zeros(self.r, self.r);
        d[(0, 0)] = Complex::new(-4.0 * s * s * self.k_row_r[0], 0.0);
        let factor = -2.0 * s * Complex::new(s, c);
        for i in 1..self.r {
            d[(0, i)] = factor * self.k_row_r[i];
            d[(i, 0)] = d[(0, i)].conj();
        }
        d
    }

    /// Full symbol `K̃(φ) = K̃(0) + D(φ)`.
    fn symbol(&self, phi: f64) -> DMatrix<Complex<f64>> {
        self.k0.map(|v| Complex::new(v, 0.0)) + self.phase_part(phi)
    }

    /// Deflated eigendecomposition of the mass-normalised phase-free symbol.
    ///
    /// Diagonalises $A_0 = \tilde{M}^{-1/2} \tilde{K}(0) \tilde{M}^{-1/2}$, then
    /// replaces the eigenpair belonging to the numerically zero eigenvalue by
    /// the exact one: eigenvalue `0`, eigenvector `e ∝ M̃^{1/2}·𝟙` (the constant
    /// nodal vector, which `K̃(0)` annihilates). The `r − 1` upper eigenvectors
    /// are re-orthonormalised against `e` by modified Gram-Schmidt and sorted
    /// by ascending eigenvalue.
    fn deflated_basis(&self) -> DeflatedBasis {
        let r = self.r;
        let m_inv_sqrt = self.mass.map(|m| 1.0 / m.sqrt());
        let a0 = DMatrix::from_fn(r, r, |i, j| m_inv_sqrt[i] * self.k0[(i, j)] * m_inv_sqrt[j]);
        let eig = SymmetricEigen::new(a0);

        // The computed eigenvalue closest to zero is the constant mode.
        let null = (0..r)
            .min_by(|&i, &j| {
                eig.eigenvalues[i]
                    .abs()
                    .partial_cmp(&eig.eigenvalues[j].abs())
                    .expect("finite eigenvalues")
            })
            .expect("r ≥ 1");

        let mut upper: Vec<(f64, DVector<f64>)> = (0..r)
            .filter(|&i| i != null)
            .map(|i| (eig.eigenvalues[i], eig.eigenvectors.column(i).into_owned()))
            .collect();
        upper.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite eigenvalues"));

        let mut w0 = DMatrix::<f64>::zeros(r, r);
        let mut lambda = DVector::<f64>::zeros(r);
        w0.set_column(0, &self.mass.map(f64::sqrt).normalize());
        for (j, (mu, mut v)) in upper.into_iter().enumerate() {
            let j = j + 1;
            for i in 0..j {
                let q = w0.column(i);
                let proj = q.dot(&v);
                v.axpy(-proj, &q, 1.0);
            }
            v.normalize_mut();
            w0.set_column(j, &v);
            lambda[j] = mu;
        }

        DeflatedBasis { w0, lambda }
    }

    /// Lowest eigenvalue of the mode with phase `φ`, accurate to relative round-off.
    ///
    /// In the deflated basis `W0 = [e | W_u]` of `A0`, the mass-normalised
    /// symbol of the mode reads
    /// $$W_0^\dagger \tilde{M}^{-1/2} \tilde{K}(\varphi) \tilde{M}^{-1/2} W_0
    /// = \begin{pmatrix} a & b^\dagger \\ b & L' \end{pmatrix},
    /// \qquad L' = \mathrm{diag}(\Lambda_u) + W_u^\dagger \hat{D} W_u,$$
    /// with `a = O(φ²/h²)`, `b = O(φ/h²)` and `L' = O(1/h²)`, where
    /// `D̂ = M̃^{-1/2} D(φ) M̃^{-1/2}`. Its lowest eigenvalue is the root below
    /// `λ'_min` of the secular function
    /// $$f(\mu) = a - \mu - b^\dagger (L' - \mu)^{-1} b .$$
    /// The eigensolver's value `mu_lo` is within `ε·μ_max` of that root, and
    /// one Newton step from it,
    /// $$y = (L' - \mu_\mathrm{lo})^{-1} b, \qquad
    /// \mu_0 = \mu_\mathrm{lo} + \frac{a - \mu_\mathrm{lo} - b^\dagger y}{1 + \|y\|^2},$$
    /// squares that error: all terms are of order `φ²/h²` with full relative
    /// precision, nothing of size `μ_max` cancels. For `φ = 0` (`D = 0`) the
    /// result is exactly `0`.
    fn lowest_eigenvalue(&self, basis: &DeflatedBasis, phi: f64, mu_lo: f64) -> f64 {
        let r = self.r;
        let m_inv_sqrt = self.mass.map(|m| 1.0 / m.sqrt());
        let mut d_hat = self.phase_part(phi);
        for i in 0..r {
            for j in 0..r {
                d_hat[(i, j)] *= m_inv_sqrt[i] * m_inv_sqrt[j];
            }
        }
        let w0 = basis.w0.map(|v| Complex::new(v, 0.0));
        let proj = w0.adjoint() * d_hat * &w0;

        let a = proj[(0, 0)].re;
        if r == 1 {
            return a;
        }
        let b = proj.view((1, 0), (r - 1, 1)).into_owned();
        let mut shifted = proj.view((1, 1), (r - 1, r - 1)).into_owned();
        for j in 0..r - 1 {
            shifted[(j, j)] += basis.lambda[j + 1] - mu_lo;
        }
        let y = shifted
            .lu()
            .solve(&b)
            .expect("L' − μ_lo is positive definite: μ_lo lies below the upper branches");
        mu_lo + (a - mu_lo - b.dotc(&y).re) / (1.0 + y.norm_squared())
    }
}

/// Orthonormal eigenbasis of `A0 = M̃^{-1/2} K̃(0) M̃^{-1/2}` with the constant
/// mode deflated exactly (see [`LocalSymbol::deflated_basis`]).
struct DeflatedBasis {
    /// Orthonormal eigenvectors `W0`, column 0 being the exact null vector `e`.
    w0: DMatrix<f64>,
    /// Eigenvalues `Λ0`: `lambda[0] == 0` exactly, the upper `r − 1` positive
    /// and ascending.
    lambda: DVector<f64>,
}

impl Poisson {
    /// Construct the 1D HoFFT solver.
    ///
    /// Assembles the local stiffness symbols $\tilde{K}(k)$ for all independent
    /// Fourier modes and precomputes their generalised eigendecompositions.
    ///
    /// # Arguments
    /// - `l`: domain length `L`.
    /// - `xl`: left boundary position `x_l`.
    /// - `n`: number of elements.
    /// - `r`: polynomial order (`r + 1` Gauss-Lobatto nodes per element).
    /// - `bc`: boundary condition type.
    ///
    /// The operator coefficients `α`, `β` are passed to the solve methods.
    pub fn new(l: f64, xl: f64, n: usize, r: usize, bc: BoundaryCondition) -> Self {
        let lobatto_basis = CollocationBasis::new(vec![(r + 1, Gauss::Lobatto)]);

        let h = l / (n as f64);

        let planner = Engine::new(n, r, l, xl, bc.extension());

        let n_freq = match bc {
            BoundaryCondition::Periodic => n,
            BoundaryCondition::Dirichlet => n + 1,
            BoundaryCondition::Neumann => n + 1,
        };

        let n_fft = match bc {
            BoundaryCondition::Periodic => n,
            BoundaryCondition::Dirichlet | BoundaryCondition::Neumann => 2 * n,
        };

        let local = LocalSymbol::new(r, h, &lobatto_basis);
        let periodic_mass_matrix = local.mass.clone();

        // Stiffness symbol K̃(k) = K̃(0) + D(φ_k) for each frequency. The mass
        // part α·M̃ is diagonal and phase-free, so α and β are applied to the
        // eigenvalues at solve time instead of being baked into the symbol.
        // Inter-element phase φ_k = 2πk/n_fft, reduced to (-π, π]: the symbol
        // is 2π-periodic in φ, and sin(φ/2), which sets the lowest eigenvalue,
        // keeps full relative precision only when |φ/2| is not close to π.
        let phis: Vec<f64> = (0..n_freq)
            .map(|k| {
                let k = if 2 * k > n_fft { k as f64 - n_fft as f64 } else { k as f64 };
                2.0 * PI * k / (n_fft as f64)
            })
            .collect();
        let symbol: Vec<DMatrix<Complex<f64>>> =
            phis.iter().map(|&phi| local.symbol(phi)).collect();

        let mass_sqrt_inv_c: DMatrix<Complex<f64>> =
            DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                Complex::new(1.0 / periodic_mass_matrix[i].sqrt(), 0.0)
            }));

        // Generalised eigendecomposition via mass normalisation:
        //   M^{-1/2} K̃(k) M^{-1/2} W = W diag(μ)   ⟹   eigenvectors V = M^{-1/2} W
        let eigs: Vec<SymmetricEigen<Complex<f64>, Dyn>> = symbol
            .iter()
            .map(|s| SymmetricEigen::new(&mass_sqrt_inv_c * s * &mass_sqrt_inv_c))
            .collect();
        let mut eigenvalues: Vec<DVector<f64>> =
            eigs.iter().map(|e| e.eigenvalues.clone()).collect();

        // The eigensolver returns the lowest eigenvalue of each mode with an
        // absolute error ~ε·μ_max, i.e. a relative error growing like n²;
        // recompute it without cancellation (the eigenvectors are accurate).
        let basis = local.deflated_basis();
        for (mu, &phi) in eigenvalues.iter_mut().zip(&phis) {
            let j0 = mu.imin();
            mu[j0] = local.lowest_eigenvalue(&basis, phi, mu[j0]);
        }

        // M-orthonormal eigenvectors of M^{-1} K̃(k)
        let eigenvectors: Vec<DMatrix<Complex<f64>>> = eigs
            .iter()
            .map(|e| &mass_sqrt_inv_c * e.eigenvectors.clone())
            .collect();

        let mu_max = max_abs_eigenvalue(&eigenvalues);

        Self {
            r,
            planner,
            eigenvalues,
            eigenvectors,
            periodic_mass_matrix,
            mu_max,
            bc,
        }
    }

    /// Spectral radius `max |μ|` of the stiffness operator `M⁻¹K` on this grid.
    pub fn mu_max(&self) -> f64 {
        self.mu_max
    }

    /// Return the Gauss-Lobatto nodes in `x`-increasing order on `[x_l, x_l + L]`.
    ///
    /// - [`BoundaryCondition::Periodic`]:  `n·r` nodes — left endpoint included, right excluded.
    /// - [`BoundaryCondition::Dirichlet`]: `n·r - 1` nodes — both boundary endpoints excluded.
    /// - [`BoundaryCondition::Neumann`]:   `n·r + 1` nodes — both boundary endpoints included.
    pub fn get_x(&self) -> Vec<f64> {
        self.planner.get_x()
    }

    /// The [`lobatto_grid::Grid1D`] the solver is built on.
    ///
    /// Interpolation of the solution (e.g. [`lobatto_grid::Grid1D::eval`],
    /// [`lobatto_grid::Grid1D::convect`]) is provided by the grid directly.
    pub fn grid(&self) -> &lobatto_grid::Grid1D {
        self.planner.grid()
    }

    /// Solve `(α M + β K) u = f` in place.
    ///
    /// On input, `f` holds the strong-form right-hand side at the Gauss-Lobatto nodes
    /// returned by [`get_x`](Self::get_x).
    /// On output, `f` contains the FEM solution `u` at the same nodes.
    /// `alpha`, `beta` are real (`f64`) or complex (`Complex<f64>`); the
    /// precomputed eigendecomposition does not depend on them, so any pair —
    /// e.g. a complex-shifted resolvent `α + s` — costs one ordinary solve.
    ///
    /// Executes the three-step HoFFT solve:
    /// 1. Mass scaling + forward FFT.
    /// 2. Per-mode symbol solve: $\hat{u}_k = V_k \Lambda_k^{-1} V_k^\dagger \hat{f}_k$.
    /// 3. Inverse FFT.
    ///
    /// An exactly zero eigenvalue `λ = α + β·μ` is annihilated: the constant
    /// mode of a pure periodic/Neumann Laplacian (`α = 0`) has `μ = 0` exactly,
    /// so its kernel is projected out. No other cutoff is applied; a nearly
    /// singular operator is the caller's responsibility.
    pub fn solve_in_place(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
    ) {
        self.apply_in_place(f, alpha, beta, pinv);
    }

    /// Apply the operator function `g(L)` with `L = α I + β M⁻¹K` to the
    /// **nodal vector** `f` in place: `u = g(L) f`.
    ///
    /// `g` is evaluated at every per-mode eigenvalue $\lambda_{k,j} = \alpha + \beta\,\mu_{k,j}$
    /// (complex when `α`, `β` are), so any function of the operator costs
    /// exactly one solve: `solve_in_place` is `g(λ) = 1/λ`, an implicit Euler
    /// step of length `dt` is `g(λ) = 1/(1 + dt·λ)`, its `m`-fold sub-step
    /// product is `g(λ) = (1 + dt·λ/m)^{-m}`, the exact semigroup is
    /// `g(λ) = exp(-t·λ)`.
    ///
    /// **Input convention.** Unlike the `solve_*` methods, whose input is the
    /// strong-form right-hand side (so that `solve` returns `(αM + βK)⁻¹ f`),
    /// the `apply_*` methods take the nodal values of `f` itself:
    /// `apply` with `g(λ) = 1/λ` returns `(αI + βM⁻¹K)⁻¹ f = (αM + βK)⁻¹ M f`.
    /// No pseudo-inverse cutoff is applied — `g` is responsible for the
    /// behaviour at `λ = 0`.
    pub fn apply_in_place<G: Fn(Complex<f64>) -> Complex<f64>>(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
        g: G,
    ) {
        let (a, b) = (alpha.into(), beta.into());
        self.apply_mu_in_place(f, |mu| g(a + b * mu));
    }

    /// Like [`apply_in_place`](Self::apply_in_place), but `g` receives the raw
    /// real eigenvalue `μ` of the stiffness operator `M⁻¹K` (neither `α` nor
    /// `β` applied): `u = g(M⁻¹K) f`.
    pub fn apply_mu_in_place<G: Fn(f64) -> Complex<f64>>(&self, f: &mut [Complex<f64>], g: G) {
        self.apply_mu_with(f, &g, &mut self.work());
    }

    /// Work buffers for [`apply_mu_with`](Self::apply_mu_with).
    fn work(&self) -> Work1D {
        let (buf_len, scratch_len) = self.planner.work_lens();
        let zero = Complex::<f64>::default();
        Work1D {
            buf: vec![zero; buf_len],
            scratch: vec![zero; scratch_len],
            col: DVector::zeros(self.r),
            v: DVector::zeros(self.r),
        }
    }

    /// [`apply_mu_in_place`](Self::apply_mu_in_place) with the work buffers
    /// supplied by the caller, so that repeated applications (one per fibre
    /// in [`PoissonND::apply_mu_in_place_dir`]) do not allocate.
    fn apply_mu_with(&self, f: &mut [Complex<f64>], g: &impl Fn(f64) -> Complex<f64>, w: &mut Work1D) {
        // Step 1: mass-scale f in place, then forward FFT.
        // Local node index j determines the mass weight:
        //   Periodic / Neumann:  j = idx % r
        //   Dirichlet:           j = (idx + 1) % r  (elem 0 starts at j=1)
        let j_offset = if self.bc == BoundaryCondition::Dirichlet {
            1
        } else {
            0
        };
        for (idx, v) in f.iter_mut().enumerate() {
            *v *= self.periodic_mass_matrix[(idx + j_offset) % self.r];
        }
        self.planner.forward_with(f, &mut w.buf, &mut w.scratch);

        // Step 2: symbol application via eigendecomposition.
        // For each mode k: v = Q†·g_k, pointwise multiply by g(μ), result = Q·v.
        for k in 0..self.eigenvalues.len() {
            self.planner.get_values(k, f, w.col.as_mut_slice());
            // Q† · col
            self.eigenvectors[k].ad_mul_to(&w.col, &mut w.v);
            for j in 0..self.r {
                w.v[j] *= g(self.eigenvalues[k][j]);
            }
            // Q · v
            w.col.gemv(
                Complex::new(1.0, 0.0),
                &self.eigenvectors[k],
                &w.v,
                Complex::new(0.0, 0.0),
            );
            self.planner.set_values(k, f, w.col.as_slice());
        }

        // Step 3: recover symmetric modes, inverse FFT.
        self.planner.inverse_with(f, &mut w.buf, &mut w.scratch);
    }

    /// Real-input variant of [`solve_in_place`](Self::solve_in_place).
    ///
    /// For real $\alpha$, $\beta$ the operator $(\alpha M + \beta K)$ is real,
    /// so a real right-hand side has an exactly real solution: `f` is copied
    /// into an internal complex buffer, solved, and the real parts are written
    /// back. Costs the same as the complex solve plus the two conversion passes
    /// and one temporary complex buffer of `f.len()` entries.
    pub fn solve_in_place_real(&self, f: &mut [f64], alpha: f64, beta: f64) {
        let mut buf: Vec<Complex<f64>> = f.iter().map(|&v| Complex::new(v, 0.0)).collect();
        self.solve_in_place(&mut buf, alpha, beta);
        for (o, b) in f.iter_mut().zip(&buf) {
            *o = b.re;
        }
    }

    /// Real-input variant of [`apply_in_place`](Self::apply_in_place) for real
    /// `α`, `β`: `f` is copied into a complex buffer, `g(L)` is applied, and
    /// the **real part** is written back.
    ///
    /// `g` may still return complex values (e.g. one factor of a conjugate pole
    /// pair); only the real part of the result is kept.
    pub fn apply_in_place_real<G: Fn(Complex<f64>) -> Complex<f64>>(
        &self,
        f: &mut [f64],
        alpha: f64,
        beta: f64,
        g: G,
    ) {
        let mut buf: Vec<Complex<f64>> = f.iter().map(|&v| Complex::new(v, 0.0)).collect();
        self.apply_in_place(&mut buf, alpha, beta, g);
        for (o, b) in f.iter_mut().zip(&buf) {
            *o = b.re;
        }
    }

    /// Implicit Euler step `u = (I + dt·L)⁻¹ f` with `L = α I + β M⁻¹K`, on the
    /// nodal vector `f` (see [`apply_in_place`](Self::apply_in_place)).
    pub fn euler_step_in_place(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
        dt: f64,
    ) {
        self.apply_in_place(f, alpha, beta, |lam| 1.0 / (1.0 + dt * lam));
    }

    /// Exact semigroup `u = exp(-t·L) f` with `L = α I + β M⁻¹K`, on the nodal
    /// vector `f` (see [`apply_in_place`](Self::apply_in_place)).
    pub fn exp_in_place(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
        t: f64,
    ) {
        self.apply_in_place(f, alpha, beta, |lam| (-t * lam).exp());
    }
}

// ── ND solver ────────────────────────────────────────────────────────────────

/// `N`-dimensional HoFFT screened Poisson solver.
///
/// Extends [`Poisson`] to `N` spatial dimensions using **sum factorisation**:
/// the forward and inverse FFTs are applied direction by direction, and the
/// per-mode symbol solve is applied independently to each mode tuple
/// $(k_0, \ldots, k_{N-1})$ via a tensor-product eigendecomposition.
///
/// `N` is a compile-time constant (const generic); the compiler can unroll
/// the per-direction loops.
///
/// ## ND algorithm
///
/// 1. **Mass scaling**: $f_p \leftarrow \tilde{M}_p f_p$ for each DOF `p`,
///    where $\tilde{M}_p = \prod_d \tilde{M}^{(d)}_{j_d(p)}$.
/// 2. **Forward ND FFT**: apply [`EngineND::forward`] direction by direction.
/// 3. **ND symbol solve**: for each mode tuple $(k_0,\ldots,k_{N-1})$, apply the
///    tensor-product pseudo-inverse
///    $$\hat{u}_{k,j} \leftarrow \hat{f}_{k,j} \Big/ \Bigl(\alpha + \beta \sum_d \mu^{(d)}_{k_d, j_d}\Bigr)$$
///    after projecting onto and back from the per-direction eigenbases $V_{k_d}^{(d)}$.
/// 4. **Inverse ND FFT**: apply [`EngineND::inverse`] direction by direction.
///
/// Step 3 is parallelised over independent mode tuples with Rayon.
///
/// ## Operator functions
///
/// The divide in step 3 is one instance of a functional calculus: every mode
/// carries the vector of per-direction eigenvalues
/// $(\mu^{(0)}, \ldots, \mu^{(N-1)})$, so
/// [`apply_in_place`](Self::apply_in_place) evaluates any `g(λ)` at
/// $\lambda = \alpha + \beta \sum_d \mu^{(d)}$ and
/// [`apply_in_place_dirs`](Self::apply_in_place_dirs) any tensor product
/// $\bigotimes_d g_d(L_d)$ — including the directionally split implicit Euler
/// step — at the cost of one solve.
pub struct PoissonND<const N: usize> {
    /// One 1D solver per spatial direction, providing the per-direction
    /// eigendecompositions.
    solvers: [Poisson; N],
    /// ND HoFFT engine for the forward/inverse transforms, interpolation, and advection.
    pub fft: EngineND<N>,
    /// Diagonal of the Kronecker-product periodic mass matrix
    /// $\tilde{M}^{(0)} \otimes \cdots \otimes \tilde{M}^{(N-1)}$,
    /// length $\prod_d r_d$, direction 0 (`x`) varies fastest.
    periodic_mass_matrix: DVector<f64>,
    /// Number of independent Fourier modes per direction:
    /// `n_d` for `Periodic`, `n_d + 1` for `Odd`/`Even`.
    n_freqs: [usize; N],
    /// Total number of independent mode tuples: the product of `n_freqs`.
    n_freq_total: usize,
    /// Per direction, for each DOF `dof_d`, its contribution `j_d · r_strides[d]`
    /// to the index of its weight in `periodic_mass_matrix`: local node
    /// `j_d = (dof_d + 1) % r_d` for Dirichlet (the first DOF is node `j = 1`),
    /// `dof_d % r_d` otherwise.
    mass_index: [Vec<usize>; N],
    /// Bound `Σ_d max |μ^(d)|` on the tensor stiffness eigenvalue magnitude.
    mu_max: f64,
}

impl<const N: usize> PoissonND<N> {
    /// Construct the ND HoFFT solver.
    ///
    /// # Arguments
    /// - `ls[d]`: domain length `L_d` in direction `d`.
    /// - `xls[d]`: left boundary position $x_{l,d}$ in direction `d`.
    /// - `ns[d]`: number of elements in direction `d`.
    /// - `rs[d]`: polynomial order `r_d` in direction `d`.
    /// - `bc[d]`: boundary condition for direction `d`.
    ///
    /// The operator coefficients `α`, `β` are passed to the solve methods.
    pub fn new(
        ls: [f64; N],
        xls: [f64; N],
        ns: [usize; N],
        rs: [usize; N],
        bc: [BoundaryCondition; N],
    ) -> Self {
        let solvers = std::array::from_fn(|d| Poisson::new(ls[d], xls[d], ns[d], rs[d], bc[d]));

        // Build the diagonal of M^{(0)} ⊗ … ⊗ M^{(N-1)} iteratively.
        // Each 1D mass matrix is diagonal (Gauss-Lobatto quadrature), so the Kronecker product
        // is also diagonal: entry (i_0,…,i_{N-1}) = ∏_d M^{(d)}[i_d, i_d].
        let periodic_mass_matrix =
            (1..N).fold(solvers[0].periodic_mass_matrix.clone(), |acc, d| {
                let m_d = &solvers[d].periodic_mass_matrix;
                DVector::from_iterator(
                    acc.len() * m_d.len(),
                    m_d.iter().flat_map(|&w| acc.iter().map(move |&v| v * w)),
                )
            });

        let extension = std::array::from_fn(|d| bc[d].extension());

        let fft = EngineND::new(ns, rs, ls, xls, extension);

        // Number of independent Fourier modes per direction.
        let n_freqs: [usize; N] = std::array::from_fn(|d| solvers[d].eigenvalues.len());
        let n_freq_total = n_freqs.iter().product();

        // Mass weight index per DOF and direction: Dirichlet DOFs start at j=1.
        let mass_index: [Vec<usize>; N] = std::array::from_fn(|d| {
            let offset = usize::from(bc[d] == BoundaryCondition::Dirichlet);
            (0..fft.ndofs[d]).map(|dof| ((dof + offset) % fft.rs[d]) * fft.r_strides[d]).collect()
        });

        // `Σ_d max |μ^(d)|` bounds the combined eigenvalue magnitude `|Σ_d μ^(d)|`.
        let mu_max = solvers.iter().map(|s| s.mu_max).sum::<f64>();

        Self {
            solvers,
            fft,
            periodic_mass_matrix,
            n_freqs,
            n_freq_total,
            mass_index,
            mu_max,
        }
    }

    /// Bound `Σ_d max |μ^(d)|` on the spectral radius of the tensor stiffness
    /// operator `M⁻¹K`.
    pub fn mu_max(&self) -> f64 {
        self.mu_max
    }

    /// Return the tensor-product Gauss-Lobatto grid as a `Vec<[f64; N]>` of length
    /// $N_\mathrm{dof} = \prod_d N_\mathrm{dof}^{(d)}$.
    ///
    /// Entry `p` holds coordinates $(x_0, \ldots, x_{N-1})$ for DOF `p`.
    /// Direction 0 (`x`) varies fastest.
    pub fn get_x(&self) -> Vec<[f64; N]> {
        self.fft.get_x()
    }

    /// The [`lobatto_grid::GridND`] the solver is built on.
    ///
    /// Interpolation of the solution (e.g. [`lobatto_grid::GridND::eval`],
    /// [`lobatto_grid::GridND::convect`], the [`lobatto_grid::EvalPlan`] API)
    /// is provided by the grid directly.
    pub fn grid(&self) -> &lobatto_grid::GridND<N> {
        self.fft.grid()
    }

    /// Solve `(α M + β K) u = f` in place using the ND HoFFT algorithm.
    ///
    /// On input, `f` holds the strong-form right-hand side at the tensor-product
    /// Gauss-Lobatto grid returned by [`get_x`](Self::get_x).
    /// On output, `f` contains the FEM solution `u` at the same grid.
    /// `alpha`, `beta` are real (`f64`) or complex (`Complex<f64>`) and may
    /// change from call to call at no cost (see [`Poisson::solve_in_place`]).
    ///
    /// See the struct-level documentation for the four-step ND algorithm.
    /// Step 3 (symbol solve) is parallelised over independent mode tuples with Rayon.
    /// An exactly zero eigenvalue `λ = α + β·Σ_d μ^(d)` (the constant mode of a
    /// pure periodic/Neumann Laplacian) is annihilated; no other cutoff is applied.
    pub fn solve_in_place(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
    ) {
        self.apply_in_place(f, alpha, beta, pinv);
    }

    /// Apply the operator function `g(L)` with `L = α I + β M⁻¹K` to the
    /// **nodal vector** `f` in place: `u = g(L) f`, `g` evaluated at the tensor
    /// eigenvalue $\lambda = \alpha + \beta \sum_d \mu^{(d)}_{k_d, j_d}$ of every mode.
    ///
    /// One solve's cost for any `g`; see [`Poisson::apply_in_place`] for the
    /// functional calculus and the **nodal-vector input convention** (the
    /// `solve_*` methods take the strong-form right-hand side instead).
    /// `g` must be `Sync`: it is called from the Rayon mode loop.
    ///
    /// ```
    /// # use lobatto_fft::solver::*;
    /// # use rustfft::num_complex::Complex;
    /// let solver = PoissonND::<2>::new([1.0; 2], [0.0; 2], [4; 2], [3; 2],
    ///     [BoundaryCondition::Periodic; 2]);
    /// let mut u = vec![Complex::new(1.0, 0.0); solver.get_x().len()];
    /// let dt = 0.01;
    /// // implicit Euler step (I + dt L)^{-1} u with L = M⁻¹K (α = 0, β = 1);
    /// // the closure captures `dt` by copy
    /// solver.apply_in_place(&mut u, 0.0, 1.0, move |lam| 1.0 / (1.0 + dt * lam));
    /// ```
    pub fn apply_in_place<G: Fn(Complex<f64>) -> Complex<f64> + Sync>(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
        g: G,
    ) {
        let (a, b) = (alpha.into(), beta.into());
        self.apply_in_place_dirs(f, |mu| g(a + b * mu.iter().sum::<f64>()));
    }

    /// Like [`apply_in_place`](Self::apply_in_place), but `g` receives the raw
    /// per-direction stiffness eigenvalues `(μ⁽⁰⁾, …, μ⁽ᴺ⁻¹⁾)` of the mode
    /// (neither `α` nor `β` applied) instead of their combination.
    ///
    /// Because the 1D operators `L_d = M_d⁻¹K_d` act on different tensor
    /// factors, this gives any tensor-product operator function
    /// $\bigotimes_d g_d(L_d)$ — in particular the directionally split
    /// implicit Euler step $\prod_d (I + dt\,L_d)^{-1}$ — at the cost of one
    /// application. How `α` is distributed over the directions is up to `g`
    /// ([`euler_split_step_in_place`](Self::euler_split_step_in_place) uses
    /// the even split `α/N`).
    ///
    /// ```
    /// # use lobatto_fft::solver::*;
    /// # use rustfft::num_complex::Complex;
    /// let solver = PoissonND::<2>::new([1.0; 2], [0.0; 2], [4; 2], [3; 2],
    ///     [BoundaryCondition::Periodic; 2]);
    /// let mut u = vec![Complex::new(1.0, 0.0); solver.get_x().len()];
    /// let (dt, beta) = (0.01, 2.0);
    /// // split implicit Euler: Π_d (I + dt β M_d⁻¹K_d)^{-1} u
    /// solver.apply_in_place_dirs(&mut u, move |mu| {
    ///     Complex::from(mu.iter().map(|&m| 1.0 / (1.0 + dt * beta * m)).product::<f64>())
    /// });
    /// ```
    pub fn apply_in_place_dirs<G: Fn([f64; N]) -> Complex<f64> + Sync>(
        &self,
        f: &mut [Complex<f64>],
        g: G,
    ) {
        // Step 1: mass-scale in place, then forward FFT. The weight of DOF
        // (dof_0, …, dof_{N-1}) is entry Σ_d mass_index[d][dof_d] of the
        // Kronecker mass diagonal; one fibre along direction 0 at a time, so
        // the other directions' part is computed once per fibre.
        let rs = &self.fft.rs;
        let r_strides = &self.fft.r_strides;
        let r_total = self.fft.r_total;
        let n0 = self.fft.ndofs[0];

        f.par_chunks_mut(n0).enumerate().for_each(|(q, fibre)| {
            let i = nd_multi_index(q * n0, &self.fft.ndofs);
            let outer: usize = (1..N).map(|d| self.mass_index[d][i[d]]).sum();
            for (fp, &j0) in fibre.iter_mut().zip(&self.mass_index[0]) {
                *fp *= self.periodic_mass_matrix[outer + j0];
            }
        });

        self.fft.forward(f);

        // Step 2: symbol solve — independent mode tuples in parallel, with a
        // per-thread scratch: the mode tuple's compact-buffer addresses `m`
        // (built once, used by the gather and the scatter; same entries as
        // `get_values` / `set_values` without their per-value index
        // arithmetic), its values `v` (fully overwritten by the gather) and
        // one per-direction product `x`. Hot path: keep it allocation-free
        // (`ad_mul_to` / `mul_to`, not `ad_mul` / `*`); one temporary per
        // product made the whole solve three times slower.
        // SAFETY: distinct k_flat values write to disjoint positions in f, and
        // each reads only positions it writes (tested in `tests`).
        let ptr = f.as_mut_ptr() as usize;
        let r_max = rs.iter().copied().max().unwrap_or(0);
        (0..self.n_freq_total).into_par_iter().for_each_init(
            || {
                (
                    self.fft.mode_indices_buffer(),
                    DVector::<Complex<f64>>::zeros(r_total),
                    DVector::<Complex<f64>>::zeros(r_max),
                )
            },
            |(m, v, x), k_flat| {
                let ks = nd_multi_index(k_flat, &self.n_freqs);
                self.fft.mode_indices(&ks, m);

                {
                    let f_ro = unsafe {
                        std::slice::from_raw_parts(ptr as *const Complex<f64>, self.fft.total)
                    };
                    for ((vj, &o), &c) in v.iter_mut().zip(&m.offsets).zip(&m.coeffs) {
                        *vj = c * f_ro[o];
                    }
                }

                // Apply Q†_d for each direction (sum factorisation).
                for d in 0..N {
                    let r_d = rs[d];
                    let step = r_strides[d] - 1;
                    let mut x = x.rows_mut(0, r_d);
                    for s in 0..r_total / r_d {
                        let bj = nd_fiber_base(s, d, rs, r_strides);
                        self.solvers[d].eigenvectors[ks[d]]
                            .ad_mul_to(&v.rows_with_step(bj, r_d, step), &mut x);
                        v.rows_with_step_mut(bj, r_d, step).copy_from(&x);
                    }
                }

                // Pointwise application of g to the per-direction eigenvalues
                // μ^{(d)}_{k_d, j_d} of the mode.
                let eigs: [&[f64]; N] = std::array::from_fn(|d| self.solvers[d].eigenvalues[ks[d]].as_slice());
                for (vj, js) in v.iter_mut().zip(nd_multi_indices(*rs)) {
                    *vj *= g(std::array::from_fn(|d| eigs[d][js[d]]));
                }

                // Apply Q_d for each direction.
                for d in 0..N {
                    let r_d = rs[d];
                    let step = r_strides[d] - 1;
                    let mut x = x.rows_mut(0, r_d);
                    for s in 0..r_total / r_d {
                        let bj = nd_fiber_base(s, d, rs, r_strides);
                        self.solvers[d].eigenvectors[ks[d]]
                            .mul_to(&v.rows_with_step(bj, r_d, step), &mut x);
                        v.rows_with_step_mut(bj, r_d, step).copy_from(&x);
                    }
                }

                {
                    let f_rw = unsafe {
                        std::slice::from_raw_parts_mut(ptr as *mut Complex<f64>, self.fft.total)
                    };
                    for &(j_flat, o) in &m.canonical {
                        f_rw[o] = v[j_flat];
                    }
                }
            },
        );

        // Step 3: inverse FFT.
        self.fft.inverse(f);
    }

    /// Solve `(α M_d + β K_d) ⊗ M_rest · u = f` in place: the screened
    /// Poisson problem with diffusion along direction `d` only.
    ///
    /// The operator acts as the mass matrix in every other direction, so
    /// `M_rest` cancels and the solve splits into independent 1D solves, one
    /// per fibre along `d`:
    /// $$u = \bigl((\alpha M_d + \beta K_d)^{-1} M_d \otimes I\bigr) f.$$
    /// The input convention is that of [`solve_in_place`](Self::solve_in_place)
    /// (strong-form right-hand side), and the result equals
    /// [`apply_in_place_dirs`](Self::apply_in_place_dirs) with a `g` that
    /// keeps only `μ⁽ᵈ⁾`, but without the transforms in the other directions:
    /// one 1D HoFFT solve per fibre instead of `2N` full-array sweeps. An
    /// exactly zero eigenvalue is annihilated, as in
    /// [`Poisson::solve_in_place`].
    ///
    /// # Panics
    ///
    /// If `d ≥ N` or `f` does not have one value per DOF.
    pub fn solve_in_place_dir(
        &self,
        d: usize,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
    ) {
        self.apply_in_place_dir(d, f, alpha, beta, pinv);
    }

    /// Apply `g(L_d)` with `L_d = α I + β M_d⁻¹K_d` along direction `d` only,
    /// to the **nodal vector** `f` in place: the 1D
    /// [`Poisson::apply_in_place`] on every fibre along `d`.
    ///
    /// See [`solve_in_place_dir`](Self::solve_in_place_dir) for the
    /// decoupling and [`Poisson::apply_in_place`] for the functional calculus
    /// and the nodal-vector input convention. `g` must be `Sync`: it is
    /// called from the Rayon fibre loop.
    ///
    /// # Panics
    ///
    /// If `d ≥ N` or `f` does not have one value per DOF.
    pub fn apply_in_place_dir<G: Fn(Complex<f64>) -> Complex<f64> + Sync>(
        &self,
        d: usize,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
        g: G,
    ) {
        let (a, b) = (alpha.into(), beta.into());
        self.apply_mu_in_place_dir(d, f, |mu| g(a + b * mu));
    }

    /// Like [`apply_in_place_dir`](Self::apply_in_place_dir), but `g` receives
    /// the raw eigenvalue `μ⁽ᵈ⁾` of `M_d⁻¹K_d` (neither `α` nor `β` applied):
    /// `u = (g(M_d⁻¹K_d) ⊗ I) f`.
    ///
    /// Direction 0 has contiguous fibres, solved in place. For `d > 0` each
    /// strided fibre is gathered into a per-thread buffer, solved and
    /// scattered back, fibres in parallel. All work buffers (the fibre, the
    /// 1D engine's extended buffer and FFT scratch, the per-mode vectors) are
    /// per thread: no allocation per fibre.
    ///
    /// # Panics
    ///
    /// If `d ≥ N` or `f` does not have one value per DOF.
    pub fn apply_mu_in_place_dir<G: Fn(f64) -> Complex<f64> + Sync>(
        &self,
        d: usize,
        f: &mut [Complex<f64>],
        g: G,
    ) {
        assert!(d < N, "direction {d} out of range for N = {N}");
        let total = self.fft.total;
        assert_eq!(f.len(), total, "expected {total} DOFs, got {}", f.len());
        let (ndofs, strides) = (&self.fft.ndofs, &self.fft.strides);
        let (m, stride) = (ndofs[d], strides[d]);
        let solver = &self.solvers[d];
        if stride == 1 {
            f.par_chunks_mut(m)
                .for_each_init(|| solver.work(), |w, fibre| solver.apply_mu_with(fibre, &g, w));
            return;
        }
        // SAFETY: nd_fiber_base yields disjoint index sets for distinct `s`,
        // so concurrent reads and writes through the raw pointer never alias.
        let ptr = f.as_mut_ptr() as usize;
        (0..total / m).into_par_iter().for_each_init(
            || (vec![Complex::<f64>::default(); m], solver.work()),
            |(fibre, w), s| {
                let base = nd_fiber_base(s, d, ndofs, strides);
                let p = ptr as *mut Complex<f64>;
                for (k, x) in fibre.iter_mut().enumerate() {
                    *x = unsafe { *p.add(base + k * stride) };
                }
                solver.apply_mu_with(fibre, &g, w);
                for (k, &x) in fibre.iter().enumerate() {
                    unsafe { *p.add(base + k * stride) = x };
                }
            },
        );
    }

    /// Real-input variant of [`solve_in_place`](Self::solve_in_place).
    ///
    /// For real $\alpha$, $\beta$ the operator $(\alpha M + \beta K)$ is real,
    /// so a real right-hand side has an exactly real solution: `f` is copied
    /// into an internal complex buffer, solved, and the real parts are written
    /// back. Costs the same as the complex solve plus the two (parallel)
    /// conversion passes and one temporary complex buffer of `f.len()` entries.
    pub fn solve_in_place_real(&self, f: &mut [f64], alpha: f64, beta: f64) {
        let mut buf: Vec<Complex<f64>> = f.par_iter().map(|&v| Complex::new(v, 0.0)).collect();
        self.solve_in_place(&mut buf, alpha, beta);
        f.par_iter_mut()
            .zip(buf.par_iter())
            .for_each(|(o, b)| *o = b.re);
    }

    /// Real-input variant of [`apply_in_place`](Self::apply_in_place) for real
    /// `α`, `β`: `f` is copied into a complex buffer, `g(L)` is applied, and
    /// the **real part** is written back (`g` may return complex values; only
    /// the real part of the result is kept).
    pub fn apply_in_place_real<G: Fn(Complex<f64>) -> Complex<f64> + Sync>(
        &self,
        f: &mut [f64],
        alpha: f64,
        beta: f64,
        g: G,
    ) {
        let mut buf: Vec<Complex<f64>> = f.par_iter().map(|&v| Complex::new(v, 0.0)).collect();
        self.apply_in_place(&mut buf, alpha, beta, g);
        f.par_iter_mut()
            .zip(buf.par_iter())
            .for_each(|(o, b)| *o = b.re);
    }

    /// Unsplit implicit Euler step `u = (I + dt·L)⁻¹ f` with `L = α I + β M⁻¹K`,
    /// on the nodal vector `f` (see [`apply_in_place`](Self::apply_in_place)).
    pub fn euler_step_in_place(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
        dt: f64,
    ) {
        self.apply_in_place(f, alpha, beta, |lam| 1.0 / (1.0 + dt * lam));
    }

    /// Directionally split implicit Euler step on the nodal vector `f`:
    /// $$u = \prod_d \bigl(I + dt\,(\tfrac{\alpha}{N} I + \beta\, M_d^{-1} K_d)\bigr)^{-1} f,$$
    /// the mass term being split evenly over the `N` directions. The factors
    /// commute, so the product is the exact tensor product of the 1D steps and
    /// costs one application (see [`apply_in_place_dirs`](Self::apply_in_place_dirs)).
    /// Each 1D factor satisfies the one-dimensional discrete maximum principle
    /// (for `dt` large enough relative to `h²`), which the unsplit step does not.
    pub fn euler_split_step_in_place(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
        dt: f64,
    ) {
        let a_n = alpha.into() / N as f64;
        let b = beta.into();
        self.apply_in_place_dirs(f, |mu| {
            mu.iter()
                .map(|&m| 1.0 / (1.0 + dt * (a_n + b * m)))
                .product()
        });
    }

    /// Exact semigroup `u = exp(-t·L) f` with `L = α I + β M⁻¹K`, on the nodal
    /// vector `f` (see [`apply_in_place`](Self::apply_in_place)). Exact in ND:
    /// the exponential of the Kronecker sum is the Kronecker product of the
    /// 1D exponentials.
    pub fn exp_in_place(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
        t: f64,
    ) {
        self.apply_in_place(f, alpha, beta, |lam| (-t * lam).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_symbol_eigendecomposition() {
        const ORTHO_TOL: f64 = 1e-10;
        const RECON_TOL: f64 = 1e-8;

        for &r in &[1, 2, 3, 4, 5, 6] {
            for &n in &[4, 16, 32, 64, 128] {
                let (alpha, beta) = (Complex::new(1.0, 0.0), Complex::new(1.0, 0.0));
                let solver = Poisson::new(1.0, 0.0, n, r, BoundaryCondition::Periodic);
                let id = DMatrix::<Complex<f64>>::identity(r, r);

                let m_c = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                    Complex::new(solver.periodic_mass_matrix[i], 0.0)
                }));
                let m_inv_c = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                    Complex::new(1.0 / solver.periodic_mass_matrix[i], 0.0)
                }));

                for k in 0..n {
                    let v = &solver.eigenvectors[k];

                    // 1. V† M V = I
                    let vtmv = v.adjoint() * &m_c * v;
                    let ortho_err = (&vtmv - &id).norm();
                    assert!(
                        ortho_err < ORTHO_TOL,
                        "r={r} n={n} k={k}: eigenvectors not M-orthonormal (err={ortho_err:.2e})"
                    );

                    // 2. V V† = M^{-1}
                    let vvt = v * v.adjoint();
                    let vvt_err = (&vvt - &m_inv_c).norm();
                    assert!(
                        vvt_err < ORTHO_TOL,
                        "r={r} n={n} k={k}: V V† ≠ M⁻¹ (err={vvt_err:.2e})"
                    );

                    // 3. Round-trip with the operator eigenvalues λ = α + β·μ and the
                    //    solver's pseudo-inverse: Q diag(1/λ) Q† M V Λ = V
                    let lambda = |i: usize| alpha + beta * solver.eigenvalues[k][i];
                    let lambda_diag = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| lambda(i)));
                    let lambda_inv =
                        DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| pinv(lambda(i))));
                    let rhs = v * &lambda_inv * v.adjoint() * &m_c * v * &lambda_diag;
                    let recon_err = (v - &rhs).norm() / v.norm().max(1.0);
                    assert!(
                        recon_err < RECON_TOL,
                        "r={r} n={n} k={k}: eigendecomposition round-trip failed (err={recon_err:.2e})"
                    );
                }
            }
        }
    }

    /// The deflated basis of `A0` is orthonormal, diagonalises `A0` to
    /// round-off, and carries the constant mode as an exact zero eigenvalue.
    #[test]
    fn test_deflated_basis() {
        for r in 1..=6 {
            for n in [4usize, 64, 4096] {
                let h = 1.0 / n as f64;
                let basis = CollocationBasis::new(vec![(r + 1, Gauss::Lobatto)]);
                let local = LocalSymbol::new(r, h, &basis);
                let DeflatedBasis { w0, lambda } = local.deflated_basis();

                let ortho_err = (w0.transpose() * &w0 - DMatrix::<f64>::identity(r, r)).norm();
                assert!(ortho_err < 1e-14, "r={r} n={n}: W0 not orthonormal ({ortho_err:.2e})");

                assert_eq!(lambda[0], 0.0, "r={r} n={n}: deflated eigenvalue not exactly 0");
                for j in 1..r {
                    assert!(lambda[j] > 0.0, "r={r} n={n}: upper eigenvalue {j} not positive");
                    assert!(lambda[j] >= lambda[j - 1], "r={r} n={n}: eigenvalues not ascending");
                }

                let e = local.mass.map(f64::sqrt).normalize();
                assert_eq!(w0.column(0), e, "r={r} n={n}: column 0 is not the exact null vector");

                let m_inv_sqrt = local.mass.map(|m| 1.0 / m.sqrt());
                let a0 = DMatrix::from_fn(r, r, |i, j| {
                    m_inv_sqrt[i] * local.k0[(i, j)] * m_inv_sqrt[j]
                });
                let residual = (&a0 * &w0 - &w0 * DMatrix::from_diagonal(&lambda)).norm();
                let scale = a0.norm().max(1.0);
                assert!(
                    residual < 1e-14 * scale,
                    "r={r} n={n}: A0 W0 ≠ W0 Λ0 (residual {residual:.2e}, |A0| = {scale:.2e})"
                );
            }
        }
    }

    /// The pseudo-inverse annihilates the operator kernel.
    ///
    /// With `α = 0` and periodic BCs the `k = 0` symbol has a constant-mode
    /// eigenvalue that is exactly zero (deflated evaluation), so the solve
    /// projects the constant mode out without any cutoff. The observable
    /// spec: the solution is invariant under a constant shift of the RHS.
    #[test]
    fn test_constant_mode_annihilated_1d() {
        let (n, r) = (16, 4);
        let solver = Poisson::new(1.0, 0.0, n, r, BoundaryCondition::Periodic);
        let x = solver.get_x();

        let f = |x: f64| (2.0 * PI * x).sin() + 0.3 * (4.0 * PI * x).cos();
        let mut u1: Vec<Complex<f64>> = x.iter().map(|&xi| Complex::new(f(xi), 0.0)).collect();
        let mut u2: Vec<Complex<f64>> =
            x.iter().map(|&xi| Complex::new(f(xi) + 1.0, 0.0)).collect();
        solver.solve_in_place(&mut u1, 0.0, 1.0);
        solver.solve_in_place(&mut u2, 0.0, 1.0);

        let max_diff = u1
            .iter()
            .zip(&u2)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        let scale = u1.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
        assert!(
            max_diff <= 1e-12 * scale.max(1.0),
            "constant RHS shift leaked into the solution (max diff = {max_diff:.2e})"
        );
    }

    /// ND version of [`test_constant_mode_annihilated_1d`]: the ND eigenvalue
    /// `μ^(0)_0 + μ^(1)_0` of the constant mode is the sum of two exact zeros.
    #[test]
    fn test_constant_mode_annihilated_2d() {
        let solver = PoissonND::<2>::new(
            [1.0; 2],
            [0.0; 2],
            [16; 2],
            [4; 2],
            [BoundaryCondition::Periodic; 2],
        );
        let x = solver.get_x();

        let f = |p: &[f64; 2]| (2.0 * PI * p[0]).sin() * (2.0 * PI * p[1]).cos();
        let mut u1: Vec<Complex<f64>> = x.iter().map(|p| Complex::new(f(p), 0.0)).collect();
        let mut u2: Vec<Complex<f64>> = x.iter().map(|p| Complex::new(f(p) + 1.0, 0.0)).collect();
        solver.solve_in_place(&mut u1, 0.0, 1.0);
        solver.solve_in_place(&mut u2, 0.0, 1.0);

        let max_diff = u1
            .iter()
            .zip(&u2)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        let scale = u1.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
        assert!(
            max_diff <= 1e-12 * scale.max(1.0),
            "constant RHS shift leaked into the solution (max diff = {max_diff:.2e})"
        );
    }

    /// Check that the tensorial structure recovers the inverse of the local mass matrix.
    ///
    /// V_ND = V_0^{(0)} ⊗ ... ⊗ V_{N-1}^{(0)} (eigenvectors at k=0 for each direction)
    /// satisfies V_ND * V_ND† = tilde_M_ND^{-1} = diag(1 / periodic_mass_matrix).
    #[test]
    fn test_nd_mass_matrix_inverse_tensorial() {
        const TOL: f64 = 1e-12;

        let kron = |a: &DMatrix<Complex<f64>>, b: &DMatrix<Complex<f64>>| {
            let (ra, ca) = (a.nrows(), a.ncols());
            let (rb, cb) = (b.nrows(), b.ncols());
            DMatrix::from_fn(ra * rb, ca * cb, |i, j| {
                a[(i / rb, j / cb)] * b[(i % rb, j % cb)]
            })
        };

        // 1D
        {
            let solver = PoissonND::<1>::new(
                [1.0],
                [0.0],
                [4],
                [3],
                [BoundaryCondition::Periodic; 1],
            );
            let v = &solver.solvers[0].eigenvectors[0];
            let vvt = v * v.adjoint();
            let r = v.nrows();
            for i in 0..r {
                for j in 0..r {
                    let expected = if i == j {
                        Complex::new(1.0 / solver.solvers[0].periodic_mass_matrix[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                    assert!((vvt[(i, j)] - expected).norm() < TOL, "1D [{i},{j}]");
                }
            }
        }

        // 2D: V_ND = V_0 ⊗ V_1
        {
            let solver = PoissonND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [4; 2],
                [2; 2],
                [BoundaryCondition::Periodic; 2],
            );
            let v_nd = kron(
                &solver.solvers[0].eigenvectors[0],
                &solver.solvers[1].eigenvectors[0],
            );
            let vvt = &v_nd * v_nd.adjoint();
            let r_total = v_nd.nrows();
            for i in 0..r_total {
                for j in 0..r_total {
                    let expected = if i == j {
                        Complex::new(1.0 / solver.periodic_mass_matrix[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                    assert!((vvt[(i, j)] - expected).norm() < TOL, "2D [{i},{j}]");
                }
            }
        }

        // 3D: V_ND = V_0 ⊗ V_1 ⊗ V_2
        {
            let solver = PoissonND::<3>::new(
                [1.0; 3],
                [0.0; 3],
                [4; 3],
                [2; 3],
                [BoundaryCondition::Periodic; 3],
            );
            let v_nd = kron(
                &kron(
                    &solver.solvers[0].eigenvectors[0],
                    &solver.solvers[1].eigenvectors[0],
                ),
                &solver.solvers[2].eigenvectors[0],
            );
            let vvt = &v_nd * v_nd.adjoint();
            let r_total = v_nd.nrows();
            for i in 0..r_total {
                for j in 0..r_total {
                    let expected = if i == j {
                        Complex::new(1.0 / solver.periodic_mass_matrix[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                    let scale = expected.norm().max(1.0_f64);
                    assert!(
                        (vvt[(i, j)] - expected).norm() / scale < TOL,
                        "3D [{i},{j}]"
                    );
                }
            }
        }
    }

    /// Mixed 3D grid: periodic, Dirichlet and Neumann directions, strides 1,
    /// 6 and 30.
    fn solver_3d() -> PoissonND<3> {
        PoissonND::<3>::new(
            [1.0, 2.0, 0.5],
            [0.0, -1.0, 0.3],
            [2, 3, 2],
            [3, 2, 4],
            [BoundaryCondition::Periodic, BoundaryCondition::Dirichlet, BoundaryCondition::Neumann],
        )
    }

    /// Complex test field with no special structure.
    fn field(total: usize) -> Vec<Complex<f64>> {
        (0..total)
            .map(|p| Complex::new((1.7 * p as f64).sin(), (0.9 * p as f64).cos()))
            .collect()
    }

    /// The directional solve is the 1D solve on every fibre, bit for bit.
    #[test]
    fn test_dir_solve_is_1d_solve_on_every_fibre() {
        let s = solver_3d();
        let (ndofs, strides, total) = (s.fft.ndofs, s.fft.strides, s.fft.total);
        let f = field(total);
        let (alpha, beta) = (Complex::new(0.4, 0.2), 1.3);
        for d in 0..3 {
            let mut u = f.clone();
            s.solve_in_place_dir(d, &mut u, alpha, beta);
            for k in 0..total / ndofs[d] {
                let base = nd_fiber_base(k, d, &ndofs, &strides);
                let mut fibre: Vec<Complex<f64>> =
                    (0..ndofs[d]).map(|j| f[base + j * strides[d]]).collect();
                s.solvers[d].solve_in_place(&mut fibre, alpha, beta);
                for (j, v) in fibre.iter().enumerate() {
                    assert_eq!(u[base + j * strides[d]], *v, "d={d} fibre {k} j={j}");
                }
            }
        }
    }

    /// The directional application equals the full ND machinery with a `g`
    /// that keeps only `μ⁽ᵈ⁾`: the other directions' transforms cancel.
    #[test]
    fn test_dir_apply_matches_nd_apply_keeping_one_direction() {
        let s = solver_3d();
        let f = field(s.fft.total);
        let (alpha, beta, dt) = (0.3, 1.1, 0.05);
        for d in 0..3 {
            let mut u = f.clone();
            s.apply_in_place_dir(d, &mut u, alpha, beta, |lam| 1.0 / (1.0 + dt * lam));
            let mut v = f.clone();
            s.apply_in_place_dirs(&mut v, |mu| {
                Complex::from(1.0 / (1.0 + dt * (alpha + beta * mu[d])))
            });
            let diff = u.iter().zip(&v).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(diff < 1e-12, "d={d}: {diff:.2e}");
        }
    }

    #[test]
    #[should_panic(expected = "direction 3 out of range for N = 3")]
    fn test_dir_solve_panics_on_out_of_range_direction() {
        let s = solver_3d();
        let mut f = vec![Complex::new(0.0, 0.0); s.fft.total];
        s.solve_in_place_dir(3, &mut f, 1.0, 1.0);
    }

    /// The fibres of every direction partition the DOF array: the `unsafe`
    /// fibre loops (transforms, directional solves) write disjoint entries.
    fn assert_fibres_partition<const N: usize>(s: &PoissonND<N>) {
        let (ndofs, strides, total) = (s.fft.ndofs, s.fft.strides, s.fft.total);
        for d in 0..N {
            let mut seen = vec![false; total];
            for k in 0..total / ndofs[d] {
                let base = nd_fiber_base(k, d, &ndofs, &strides);
                for j in 0..ndofs[d] {
                    let p = base + j * strides[d];
                    assert!(!seen[p], "d={d}: index {p} in two fibres");
                    seen[p] = true;
                }
            }
            assert!(seen.iter().all(|&b| b), "d={d}: fibres do not cover the array");
        }
    }

    /// The mode tuples of the per-mode step write disjoint entries that
    /// cover the compact array (the canonical entries of `mode_indices`), and
    /// each tuple reads only entries it writes itself (the offsets with a
    /// nonzero coefficient): the `unsafe` mode loop never touches an entry
    /// from two threads.
    fn assert_mode_tuples_own_their_entries<const N: usize>(s: &PoissonND<N>) {
        let mut m = s.fft.mode_indices_buffer();
        let mut owner = vec![usize::MAX; s.fft.total];
        for (k_flat, ks) in nd_multi_indices(s.n_freqs).enumerate() {
            s.fft.mode_indices(&ks, &mut m);
            for &(_, p) in &m.canonical {
                assert_eq!(owner[p], usize::MAX, "entry {p} written by tuples {} and {k_flat}", owner[p]);
                owner[p] = k_flat;
            }
        }
        assert!(owner.iter().all(|&o| o != usize::MAX), "some entries are written by no tuple");
        for (k_flat, ks) in nd_multi_indices(s.n_freqs).enumerate() {
            s.fft.mode_indices(&ks, &mut m);
            for (&p, c) in m.offsets.iter().zip(&m.coeffs).filter(|(_, c)| c.norm() > 0.0) {
                assert_eq!(owner[p], k_flat, "tuple {k_flat} reads entry {p} (coefficient {c}) owned by tuple {}", owner[p]);
            }
        }
    }

    /// `mode_indices` gives, bit for bit, the values `get_values` gathers and
    /// the entries `set_values` writes, on every independent mode tuple.
    fn assert_mode_indices_match_get_set_values<const N: usize>(s: &PoissonND<N>) {
        let (total, r_total) = (s.fft.total, s.fft.r_total);
        let g: Vec<Complex<f64>> = (0..total).map(|p| Complex::new((1.7 * p as f64).sin(), -(0.3 * p as f64).cos())).collect();
        let v_in: Vec<Complex<f64>> = (0..r_total).map(|j| Complex::new(j as f64 + 0.5, -(j as f64))).collect();
        let mut m = s.fft.mode_indices_buffer();
        let mut v = vec![Complex::new(0.0, 0.0); r_total];
        for ks in nd_multi_indices(s.n_freqs) {
            s.fft.mode_indices(&ks, &mut m);
            s.fft.get_values(&ks, &g, &mut v);
            for (j, ((&o, &c), x)) in m.offsets.iter().zip(&m.coeffs).zip(&v).enumerate() {
                let y = c * g[o];
                assert!(y.re.to_bits() == x.re.to_bits() && y.im.to_bits() == x.im.to_bits(), "{ks:?}, j = {j}: {y} vs {x}");
            }
            let (mut a, mut b) = (g.clone(), g.clone());
            s.fft.set_values(&ks, &mut a, &v_in);
            for &(j, o) in &m.canonical {
                b[o] = v_in[j];
            }
            assert!(a == b, "{ks:?}: scatter differs from set_values");
        }
    }

    #[test]
    fn test_unsafe_loops_touch_disjoint_entries() {
        let bc = [BoundaryCondition::Periodic, BoundaryCondition::Dirichlet, BoundaryCondition::Neumann];
        let s1 = PoissonND::<1>::new([1.0], [0.0], [3], [4], [BoundaryCondition::Dirichlet]);
        let s3 = solver_3d();
        let s5 = PoissonND::<5>::new([1.0; 5], [0.0; 5], [2, 1, 3, 1, 2], [2, 3, 2, 2, 3], [bc[0], bc[1], bc[2], bc[2], bc[0]]);
        assert_fibres_partition(&s1);
        assert_fibres_partition(&s3);
        assert_fibres_partition(&s5);
        assert_mode_tuples_own_their_entries(&s1);
        assert_mode_tuples_own_their_entries(&s3);
        assert_mode_tuples_own_their_entries(&s5);
    }

    #[test]
    fn test_mode_indices_match_get_set_values() {
        use BoundaryCondition::*;
        let s1 = PoissonND::<1>::new([1.0], [0.0], [3], [4], [Dirichlet]);
        let s3 = solver_3d();
        let s4 = PoissonND::<4>::new([1.0; 4], [0.0; 4], [3, 2, 4, 1], [4, 3, 5, 2], [Neumann, Dirichlet, Periodic, Neumann]);
        let s5 = PoissonND::<5>::new([1.0; 5], [0.0; 5], [2, 1, 3, 1, 2], [2, 3, 2, 2, 3], [Periodic, Dirichlet, Neumann, Neumann, Periodic]);
        assert_mode_indices_match_get_set_values(&s1);
        assert_mode_indices_match_get_set_values(&s3);
        assert_mode_indices_match_get_set_values(&s4);
        assert_mode_indices_match_get_set_values(&s5);
    }
}
