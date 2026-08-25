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

use crate::hofft::{nd_col_major_strides, nd_fiber_base, Engine, EngineND, Extension};
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
    /// Relative pseudo-inverse cutoff (default [`DEFAULT_PINV_REL_TOL`]); the
    /// absolute cutoff used by the solves is [`pinv_tol`](Self::pinv_tol).
    pinv_rel_tol: f64,
    /// Spectral radius `max |μ|` of the stiffness symbols, precomputed for the
    /// cutoff bound.
    mu_max: f64,
    /// Boundary condition type.
    bc: BoundaryCondition,
}

/// Default relative pseudo-inverse cutoff of the solvers (`1e-12`).
///
/// Per-mode eigenvalues with `|λ| ≤ pinv_rel_tol · (|α| + |β|·max |μ|)`
/// are treated as zero: analytically zero eigenvalues
/// — the constant mode when `α = 0` with periodic or Neumann conditions — are
/// computed by the eigensolver as roundoff `~ε·λ_max`, and the relative bound
/// filters them regardless of problem size.
pub const DEFAULT_PINV_REL_TOL: f64 = 1e-12;

/// Spectral radius `max |μ|` over a set of per-mode real eigenvalue vectors.
fn max_abs_eigenvalue(eigenvalues: &[DVector<f64>]) -> f64 {
    eigenvalues
        .iter()
        .flat_map(|l| l.iter())
        .fold(0.0f64, |m, &l| m.max(l.abs()))
}

/// Pseudo-inverse of the eigenvalue `lam`: `conj(λ)/|λ|²` (one real division,
/// no square root), or `0` when `|λ|² ≤ tol_sqr`.
#[inline(always)]
fn pinv(lam: Complex<f64>, tol_sqr: f64) -> Complex<f64> {
    let norm_sqr = lam.norm_sqr();
    if norm_sqr > tol_sqr {
        lam.conj() / norm_sqr
    } else {
        Complex::new(0.0, 0.0)
    }
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

        let coeff_phase = if bc == BoundaryCondition::Periodic {
            2.0
        } else {
            1.0
        };

        let mut symbol = vec![DMatrix::<Complex::<f64>>::zeros(r, r); n_freq];
        let mass_matrix = lobatto_basis.weights_matrix();
        let periodic_mass_matrix = DVector::from_fn(r, |i, _| {
            if i == 0 {
                h * 2.0 * mass_matrix[(0, 0)]
            } else {
                h * mass_matrix[(i, i)]
            }
        });
        let diff_matrix = lobatto_basis.diff_matrix(0);

        // Construct the stiffness matrix
        let stiffness_matrix: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> =
            diff_matrix.transpose() * mass_matrix.clone() * diff_matrix;

        // Compute the stiffness symbol K̃(k) for each frequency. The mass part
        // α·M̃ is diagonal and phase-free, so α and β are applied to the
        // eigenvalues at solve time instead of being baked into the symbol.
        for (k, s) in symbol.iter_mut().enumerate() {
            let phase = -coeff_phase * PI * (k as f64) / (n as f64);
            let cos = phase.cos();
            let sin = phase.sin();
            let exp = cos + Complex::<f64>::I * sin;

            for i in 1..r {
                for j in 1..r {
                    s[(i, j)] = (stiffness_matrix[(i, j)] / h).into();
                }
            }

            s[(0, 0)] = (2.0 * stiffness_matrix[(0, 0)] / h
                + 2.0 * stiffness_matrix[(0, r)] * cos / h)
                .into();

            for i in 1..r {
                s[(0, i)] = stiffness_matrix[(0, i)] / h + stiffness_matrix[(r, i)] * exp / h;
                s[(i, 0)] = s[(0, i)].conj();
            }
        }

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
        let eigenvalues: Vec<DVector<f64>> = eigs.iter().map(|e| e.eigenvalues.clone()).collect();

        // M-orthonormal eigenvectors of M^{-1} K̃(k)
        let eigenvectors: Vec<DMatrix<Complex<f64>>> = eigs
            .iter()
            .map(|e| &mass_sqrt_inv_c * e.eigenvectors.clone())
            .collect();

        // Spectral radius for the relative pseudo-inverse cutoff: an
        // analytically zero eigenvalue (the constant mode of the k = 0 symbol
        // when α = 0) is computed as roundoff ~ε·λ_max, which outgrows any
        // absolute threshold as the spectral radius grows with n and r. The
        // cutoff is scaled with the bound |α| + |β|·max |μ| on |α + β·μ|.
        let mu_max = max_abs_eigenvalue(&eigenvalues);

        Self {
            r,
            planner,
            eigenvalues,
            eigenvectors,
            periodic_mass_matrix,
            pinv_rel_tol: DEFAULT_PINV_REL_TOL,
            mu_max,
            bc,
        }
    }

    /// Builder-style setter for the relative pseudo-inverse cutoff
    /// (see [`DEFAULT_PINV_REL_TOL`]).
    ///
    /// # Panics
    /// If `rel_tol` is negative or NaN.
    pub fn with_pinv_rel_tol(mut self, rel_tol: f64) -> Self {
        self.set_pinv_rel_tol(rel_tol);
        self
    }

    /// Set the relative pseudo-inverse cutoff (see [`DEFAULT_PINV_REL_TOL`]).
    ///
    /// # Panics
    /// If `rel_tol` is negative or NaN.
    pub fn set_pinv_rel_tol(&mut self, rel_tol: f64) {
        assert!(rel_tol >= 0.0, "pinv_rel_tol must be non-negative (got {rel_tol})");
        self.pinv_rel_tol = rel_tol;
    }

    /// Relative pseudo-inverse cutoff (default [`DEFAULT_PINV_REL_TOL`]).
    pub fn pinv_rel_tol(&self) -> f64 {
        self.pinv_rel_tol
    }

    /// Absolute pseudo-inverse cutoff used by
    /// [`solve_in_place`](Self::solve_in_place) for the coefficients `α`, `β`:
    /// `pinv_rel_tol · (|α| + |β|·max |μ|)`.
    pub fn pinv_tol(&self, alpha: impl Into<Complex<f64>>, beta: impl Into<Complex<f64>>) -> f64 {
        self.pinv_rel_tol * (alpha.into().norm() + beta.into().norm() * self.mu_max)
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
    /// Eigenvalues below the pseudo-inverse cutoff [`pinv_tol`](Self::pinv_tol)
    /// are annihilated (the kernel of a pure Neumann/periodic Laplacian is
    /// projected out).
    pub fn solve_in_place(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
    ) {
        let (alpha, beta) = (alpha.into(), beta.into());
        let tol = self.pinv_tol(alpha, beta);
        let tol_sqr = tol * tol;
        self.apply_in_place(f, alpha, beta, |lam| pinv(lam, tol_sqr));
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
        self.planner.forward(f);

        // Step 2: symbol application via eigendecomposition.
        // For each mode k: v = Q†·g_k, pointwise multiply by g(μ), result = Q·v.
        let n_freq = self.eigenvalues.len();
        let mut col = DVector::<Complex<f64>>::zeros(self.r);
        for k in 0..n_freq {
            self.planner.get_values(k, f, col.as_mut_slice());
            // Q† · col
            let mut v = self.eigenvectors[k].ad_mul(&col);
            for j in 0..self.r {
                v[j] *= g(self.eigenvalues[k][j]);
            }
            // Q · v
            col.gemv(
                Complex::new(1.0, 0.0),
                &self.eigenvectors[k],
                &v,
                Complex::new(0.0, 0.0),
            );
            self.planner.set_values(k, f, col.as_slice());
        }

        // Step 3: recover symmetric modes, inverse FFT.
        self.planner.inverse(f);
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
    /// Column-major strides for the mode-tuple tensor.
    n_freq_strides: [usize; N],
    /// Per-direction DOF index offset for mass scaling: 1 for Dirichlet (first DOF
    /// is local node `j = 1`), 0 otherwise.
    j_offsets: [usize; N],
    /// Relative pseudo-inverse cutoff (default [`DEFAULT_PINV_REL_TOL`]); the
    /// absolute cutoff used by the solves is [`pinv_tol`](Self::pinv_tol).
    pinv_rel_tol: f64,
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
        let n_freq_strides = nd_col_major_strides(&n_freqs);

        // DOF index offset for mass scaling: Dirichlet DOFs start at j=1.
        let j_offsets: [usize; N] = std::array::from_fn(|d| {
            if bc[d] == BoundaryCondition::Dirichlet {
                1
            } else {
                0
            }
        });

        // Bound for the relative pseudo-inverse cutoff (see [`Poisson::new`]):
        // `|α| + |β|·Σ_d max |μ^(d)|` bounds the combined eigenvalue magnitude
        // `|α + β·Σ_d μ^(d)|`.
        let mu_max = solvers.iter().map(|s| s.mu_max).sum::<f64>();

        Self {
            solvers,
            fft,
            periodic_mass_matrix,
            n_freqs,
            n_freq_total,
            n_freq_strides,
            j_offsets,
            pinv_rel_tol: DEFAULT_PINV_REL_TOL,
            mu_max,
        }
    }

    /// Builder-style setter for the relative pseudo-inverse cutoff
    /// (see [`DEFAULT_PINV_REL_TOL`]).
    ///
    /// # Panics
    /// If `rel_tol` is negative or NaN.
    pub fn with_pinv_rel_tol(mut self, rel_tol: f64) -> Self {
        self.set_pinv_rel_tol(rel_tol);
        self
    }

    /// Set the relative pseudo-inverse cutoff (see [`DEFAULT_PINV_REL_TOL`]).
    ///
    /// # Panics
    /// If `rel_tol` is negative or NaN.
    pub fn set_pinv_rel_tol(&mut self, rel_tol: f64) {
        assert!(rel_tol >= 0.0, "pinv_rel_tol must be non-negative (got {rel_tol})");
        self.pinv_rel_tol = rel_tol;
    }

    /// Relative pseudo-inverse cutoff (default [`DEFAULT_PINV_REL_TOL`]).
    pub fn pinv_rel_tol(&self) -> f64 {
        self.pinv_rel_tol
    }

    /// Absolute pseudo-inverse cutoff used by
    /// [`solve_in_place`](Self::solve_in_place) for the coefficients `α`, `β`:
    /// `pinv_rel_tol · (|α| + |β|·Σ_d max |μ^(d)|)`.
    pub fn pinv_tol(&self, alpha: impl Into<Complex<f64>>, beta: impl Into<Complex<f64>>) -> f64 {
        self.pinv_rel_tol * (alpha.into().norm() + beta.into().norm() * self.mu_max)
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
    /// Eigenvalues below the cutoff [`pinv_tol`](Self::pinv_tol) are annihilated.
    pub fn solve_in_place(
        &self,
        f: &mut [Complex<f64>],
        alpha: impl Into<Complex<f64>>,
        beta: impl Into<Complex<f64>>,
    ) {
        let (alpha, beta) = (alpha.into(), beta.into());
        let tol = self.pinv_tol(alpha, beta);
        let tol_sqr = tol * tol;
        self.apply_in_place(f, alpha, beta, |lam| pinv(lam, tol_sqr));
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
        // Step 1: mass-scale in place, then forward FFT.
        // Mass weight for DOF p in direction d: j_d = (dof_d + j_offset_d) % r_d.
        let rs = &self.fft.rs;
        let r_strides = &self.fft.r_strides;
        let r_total = self.fft.r_total;

        f.par_iter_mut().enumerate().for_each(|(p, fp)| {
            let lj = (0..N).fold(0, |acc, d| {
                let dof_d = (p / self.fft.strides[d]) % self.fft.ndofs[d];
                acc + ((dof_d + self.j_offsets[d]) % rs[d]) * r_strides[d]
            });
            *fp *= self.periodic_mass_matrix[lj];
        });

        self.fft.forward(f);

        // Step 2: symbol solve — independent mode tuples in parallel.
        // SAFETY: distinct k_flat values write to disjoint positions in f.
        let ptr = f.as_mut_ptr() as usize;
        (0..self.n_freq_total).into_par_iter().for_each(|k_flat| {
            let ks: [usize; N] =
                std::array::from_fn(|d| (k_flat / self.n_freq_strides[d]) % self.n_freqs[d]);

            let mut v = DVector::<Complex<f64>>::zeros(r_total);

            {
                let f_ro = unsafe {
                    std::slice::from_raw_parts(ptr as *const Complex<f64>, self.fft.total)
                };
                self.fft.get_values(&ks, f_ro, v.as_mut_slice());
            }

            // Apply Q†_d for each direction (sum factorisation).
            for d in 0..N {
                let r_d = rs[d];
                let step = r_strides[d] - 1;
                for s in 0..r_total / r_d {
                    let bj = nd_fiber_base(s, d, rs, r_strides);
                    let x = self.solvers[d].eigenvectors[ks[d]]
                        .ad_mul(&v.rows_with_step(bj, r_d, step));
                    v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                }
            }

            // Pointwise application of g to the per-direction eigenvalues
            // μ^{(d)}_{k_d, j_d} of the mode.
            for j_flat in 0..r_total {
                let mut mus = [0.0f64; N];
                for d in 0..N {
                    let j_d = (j_flat / r_strides[d]) % rs[d];
                    mus[d] = self.solvers[d].eigenvalues[ks[d]][j_d];
                }
                v[j_flat] *= g(mus);
            }

            // Apply Q_d for each direction.
            for d in 0..N {
                let r_d = rs[d];
                let step = r_strides[d] - 1;
                for s in 0..r_total / r_d {
                    let bj = nd_fiber_base(s, d, rs, r_strides);
                    let x = &self.solvers[d].eigenvectors[ks[d]] * v.rows_with_step(bj, r_d, step);
                    v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                }
            }

            {
                let f_rw = unsafe {
                    std::slice::from_raw_parts_mut(ptr as *mut Complex<f64>, self.fft.total)
                };
                self.fft.set_values(&ks, f_rw, v.as_slice());
            }
        });

        // Step 3: inverse FFT.
        self.fft.inverse(f);
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
                    //    solver's own pseudo-inverse cutoff: Q diag(1/λ) Q† M V Λ = V
                    let lambda = |i: usize| alpha + beta * solver.eigenvalues[k][i];
                    let lambda_diag = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| lambda(i)));
                    let lambda_inv = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                        let lam = lambda(i);
                        if lam.norm() > solver.pinv_tol(alpha, beta) {
                            1.0 / lam
                        } else {
                            Complex::new(0.0, 0.0)
                        }
                    }));
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

    /// Regression: the pseudo-inverse must annihilate the operator kernel.
    ///
    /// With `α = 0` and periodic BCs the `k = 0` symbol has an analytically zero
    /// constant-mode eigenvalue, computed by the eigensolver as roundoff on the
    /// order of `ε·λ_max`. At `(n, r) = (16, 4)` that roundoff exceeds `1e-12`,
    /// so an *absolute* pseudo-inverse cutoff fails to filter it and the
    /// constant mode of the RHS is amplified by `~1/λ ≈ 1e12`. The observable
    /// spec: the solution is invariant under a constant shift of the RHS.
    #[test]
    fn test_pinv_filters_constant_mode_1d() {
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

    /// ND version of [`test_pinv_filters_constant_mode_1d`]: the roundoff of the
    /// zero ND eigenvalue `λ^(0)_0 + λ^(1)_0` is the sum of two per-direction
    /// roundoffs, so it crosses an absolute `1e-12` cutoff even sooner.
    #[test]
    fn test_pinv_filters_constant_mode_2d() {
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
}
