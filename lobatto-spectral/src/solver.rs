//! Screened Poisson solvers by dense fast diagonalisation.
//!
//! Provides [`Poisson`] and [`PoissonND`], which solve
//! $(\alpha M + \beta K)\, u = f$ on Gauss-Lobatto grids of `n` spectral
//! elements of order `r` per direction ([`lobatto_grid::Grid1D`],
//! [`lobatto_grid::GridND`]). The discretisation, the DOF layout and the
//! public interface are those of the `lobatto-fft` solvers, restricted to
//! real data and real coefficients.
//!
//! The coefficients $\alpha$ and $\beta$ are **per-call parameters** of the
//! solves, not of the solver: nothing precomputed depends on them, so one
//! solver serves every operator on its grid (varying time steps, frequency
//! sweeps).
//!
//! ## Algorithm
//!
//! With $M$ the lumped (diagonal) mass matrix and $K$ the stiffness matrix
//! of the 1D spectral-element discretisation, the generalised
//! eigendecomposition
//! $$K V = M V \operatorname{diag}(\mu), \qquad V^T M V = I,$$
//! is computed once at construction (dense, `n·r × n·r`). Any function of
//! the operator `L = M⁻¹K` then costs two dense products:
//! $$u = g(L)\, f = V \operatorname{diag}\bigl(g(\mu)\bigr) V^T M f,$$
//! and the solve is `g(μ) = 1/(α + β μ)`.
//!
//! In `N` dimensions the operator has the tensor-product structure
//! $$A = \alpha\, M_0 \otimes \cdots \otimes M_{N-1}
//! \;+\; \beta \sum_{d} M_0 \otimes \cdots \otimes K_d \otimes \cdots \otimes M_{N-1},$$
//! so its eigenvectors are $V^{(0)} \otimes \cdots \otimes V^{(N-1)}$ and
//! its eigenvalues $\alpha + \beta \sum_d \mu^{(d)}_{j_d}$: the transforms
//! are applied direction by direction (sum factorisation), parallelised
//! with Rayon over independent fibres.
//!
//! Setup costs `O(Σ_d (n_d r_d)³)`, one solve `O(Σ_d n_d r_d · N_dof)`.
//! This is the right tool for small problems; for larger ones,
//! `lobatto-fft` diagonalises the same operator through the FFT at
//! `O(log n)` per DOF per direction.
//!
//! ## Boundary conditions and data layout
//!
//! | BC          | Extension | `N_dof` per direction |
//! |-------------|-----------|-----------------------|
//! | `Periodic`  | none      | `n·r`                 |
//! | `Dirichlet` | odd       | `n·r - 1`             |
//! | `Neumann`   | even      | `n·r + 1`             |
//!
//! The DOF vector is in `x`-increasing order (direction 0 fastest in ND),
//! exactly the layout of [`lobatto_grid::Grid1D::get_x`] and
//! [`lobatto_grid::GridND::get_x`].

use lobatto::collocation::{CollocationBasis, Gauss};
use lobatto_grid::grid::nd_fiber_base;
use lobatto_grid::{Extension, Grid1D, GridND};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rayon::prelude::*;

/// Boundary condition for one spatial direction of the screened Poisson problem.
///
/// Each direction may carry an independent BC, enabling mixed configurations in ND.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Periodic identification: `u(x_l) = u(x_l + L)`. `N_dof = n·r`.
    Periodic,
    /// Homogeneous Dirichlet: `u(x_l) = u(x_l + L) = 0`. The two boundary
    /// nodes are not stored, `N_dof = n·r - 1`.
    Dirichlet,
    /// Homogeneous Neumann: `u'(x_l) = u'(x_l + L) = 0`. Both boundary
    /// nodes are stored, `N_dof = n·r + 1`.
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

/// Relative pseudo-inverse cutoff of the solves.
///
/// Per-mode eigenvalues with `|λ| ≤ PINV_REL_TOL · (|α| + |β|·max |μ|)` are
/// treated as zero. The analytically zero eigenvalue of the constant mode
/// (periodic or Neumann conditions with `α = 0`) is returned by the dense
/// eigensolver as round-off of order `ε·max |μ|`, which this relative bound
/// filters regardless of problem size.
const PINV_REL_TOL: f64 = 1e-12;

/// Pseudo-inverse of the eigenvalue `lam`: `1/λ`, or `0` when `|λ| ≤ tol`.
#[inline(always)]
fn pinv(lam: f64, tol: f64) -> f64 {
    if lam.abs() > tol {
        1.0 / lam
    } else {
        0.0
    }
}

/// `out = Vᵀ x` for a square column-major matrix `v`.
fn tr_mul(v: &DMatrix<f64>, x: &[f64], out: &mut [f64]) {
    let n = v.nrows();
    let data = v.as_slice();
    for (k, o) in out.iter_mut().enumerate() {
        *o = data[k * n..(k + 1) * n]
            .iter()
            .zip(x)
            .map(|(a, b)| a * b)
            .sum();
    }
}

/// `out = V x` for a square column-major matrix `v`.
fn mul(v: &DMatrix<f64>, x: &[f64], out: &mut [f64]) {
    let n = v.nrows();
    let data = v.as_slice();
    out.iter_mut().for_each(|o| *o = 0.0);
    for (k, &xk) in x.iter().enumerate() {
        for (o, a) in out.iter_mut().zip(&data[k * n..(k + 1) * n]) {
            *o += a * xk;
        }
    }
}

// ── 1D solver ────────────────────────────────────────────────────────────────

/// 1D screened Poisson solver by dense diagonalisation.
///
/// Solves $(\alpha M + \beta K)\, u = f$ on `[x_l, x_l + L]` using `n`
/// Gauss-Lobatto spectral elements of polynomial order `r`.
///
/// ## Construction
///
/// The global lumped mass `M` (diagonal) and stiffness `K` are assembled
/// from the reference element matrices through the DOF map of the grid,
/// then the generalised eigendecomposition
/// $$M^{-1/2} K M^{-1/2} = W \operatorname{diag}(\mu) W^T, \qquad V = M^{-1/2} W$$
/// is stored. `α` and `β` enter only through the per-mode eigenvalues
/// `λ_j = α + β μ_j` at solve time.
pub struct Poisson {
    /// The grid: geometry, DOF layout, interpolation.
    grid: Grid1D,
    /// Diagonal of the lumped mass matrix `M`, length `N_dof`.
    mass_diag: DVector<f64>,
    /// Eigenvalues `μ_j` of the stiffness operator `M⁻¹K`.
    eigenvalues: DVector<f64>,
    /// `M`-orthonormal eigenvectors `V` (`N_dof × N_dof`): `Vᵀ M V = I`.
    eigenvectors: DMatrix<f64>,
    /// Spectral radius `max |μ|`.
    mu_max: f64,
}

impl Poisson {
    /// Construct the 1D solver.
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
        let grid = Grid1D::new(n, r, l, xl, bc.extension());
        let (mass_diag, stiffness) = assemble(&grid);
        let n_dof = grid.n_dof();

        // Generalised eigendecomposition via mass normalisation:
        //   M^{-1/2} K M^{-1/2} W = W diag(μ)   ⟹   eigenvectors V = M^{-1/2} W
        let m_inv_sqrt = mass_diag.map(|m| 1.0 / m.sqrt());
        let a = DMatrix::from_fn(n_dof, n_dof, |i, j| {
            m_inv_sqrt[i] * stiffness[(i, j)] * m_inv_sqrt[j]
        });
        let SymmetricEigen {
            eigenvalues,
            eigenvectors: w,
        } = SymmetricEigen::new(a);
        let eigenvectors = DMatrix::from_fn(n_dof, n_dof, |i, j| m_inv_sqrt[i] * w[(i, j)]);
        let mu_max = eigenvalues.iter().fold(0.0_f64, |m, &l| m.max(l.abs()));

        Self {
            grid,
            mass_diag,
            eigenvalues,
            eigenvectors,
            mu_max,
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
        self.grid.get_x()
    }

    /// The [`lobatto_grid::Grid1D`] the solver is built on.
    ///
    /// Interpolation of the solution (e.g. [`lobatto_grid::Grid1D::eval`],
    /// [`lobatto_grid::Grid1D::convect`]) is provided by the grid directly.
    pub fn grid(&self) -> &Grid1D {
        &self.grid
    }

    /// Solve `(α M + β K) u = f` in place.
    ///
    /// On input, `f` holds the strong-form right-hand side at the Gauss-Lobatto
    /// nodes returned by [`get_x`](Self::get_x). On output, `f` contains the
    /// FEM solution `u` at the same nodes. `alpha`, `beta` may change from
    /// call to call at no cost: the eigendecomposition does not depend on them.
    ///
    /// Eigenvalues with `|α + β·μ| ≤ 1e-12 · (|α| + |β|·max |μ|)` are
    /// annihilated: the kernel of a pure periodic/Neumann Laplacian (`α = 0`)
    /// is projected out.
    pub fn solve_in_place(&self, f: &mut [f64], alpha: f64, beta: f64) {
        let tol = PINV_REL_TOL * (alpha.abs() + beta.abs() * self.mu_max);
        self.apply_in_place(f, alpha, beta, |lam| pinv(lam, tol));
    }

    /// Apply the operator function `g(L)` with `L = α I + β M⁻¹K` to the
    /// **nodal vector** `f` in place: `u = g(L) f`.
    ///
    /// `g` is evaluated at every eigenvalue `λ_j = α + β·μ_j`, so any
    /// function of the operator costs exactly one solve: `solve_in_place` is
    /// `g(λ) = 1/λ`, an implicit Euler step of length `dt` is
    /// `g(λ) = 1/(1 + dt·λ)`, its `m`-fold sub-step product is
    /// `g(λ) = (1 + dt·λ/m)^{-m}`, the exact semigroup is `g(λ) = exp(-t·λ)`.
    ///
    /// **Input convention.** Unlike the `solve_*` methods, whose input is the
    /// strong-form right-hand side (so that `solve` returns `(αM + βK)⁻¹ f`),
    /// the `apply_*` methods take the nodal values of `f` itself:
    /// `apply` with `g(λ) = 1/λ` returns `(αI + βM⁻¹K)⁻¹ f = (αM + βK)⁻¹ M f`.
    /// No pseudo-inverse cutoff is applied — `g` is responsible for the
    /// behaviour at `λ = 0`.
    pub fn apply_in_place<G: Fn(f64) -> f64>(&self, f: &mut [f64], alpha: f64, beta: f64, g: G) {
        self.apply_mu_in_place(f, |mu| g(alpha + beta * mu));
    }

    /// Like [`apply_in_place`](Self::apply_in_place), but `g` receives the raw
    /// eigenvalue `μ` of the stiffness operator `M⁻¹K` (neither `α` nor `β`
    /// applied): `u = g(M⁻¹K) f`.
    pub fn apply_mu_in_place<G: Fn(f64) -> f64>(&self, f: &mut [f64], g: G) {
        let n_dof = self.grid.n_dof();
        assert_eq!(f.len(), n_dof, "expected {n_dof} DOFs, got {}", f.len());
        self.apply_mu_with(f, &g, &mut vec![0.0; n_dof]);
    }

    /// [`apply_mu_in_place`](Self::apply_mu_in_place) with the work vector
    /// `hat` (length `N_dof`) supplied by the caller, so that repeated
    /// applications (one per fibre in [`PoissonND::apply_mu_in_place_dir`])
    /// do not allocate.
    fn apply_mu_with(&self, f: &mut [f64], g: &impl Fn(f64) -> f64, hat: &mut [f64]) {
        // 1. Mass scaling.
        for (fi, m) in f.iter_mut().zip(self.mass_diag.iter()) {
            *fi *= m;
        }
        // 2. Vᵀ, pointwise g(μ), V.
        tr_mul(&self.eigenvectors, f, hat);
        for (h, &mu) in hat.iter_mut().zip(self.eigenvalues.iter()) {
            *h *= g(mu);
        }
        mul(&self.eigenvectors, hat, f);
    }

    /// Implicit Euler step `u = (I + dt·L)⁻¹ f` with `L = α I + β M⁻¹K`, on the
    /// nodal vector `f` (see [`apply_in_place`](Self::apply_in_place)).
    pub fn euler_step_in_place(&self, f: &mut [f64], alpha: f64, beta: f64, dt: f64) {
        self.apply_in_place(f, alpha, beta, |lam| 1.0 / (1.0 + dt * lam));
    }

    /// Exact semigroup `u = exp(-t·L) f` with `L = α I + β M⁻¹K`, on the nodal
    /// vector `f` (see [`apply_in_place`](Self::apply_in_place)).
    pub fn exp_in_place(&self, f: &mut [f64], alpha: f64, beta: f64, t: f64) {
        self.apply_in_place(f, alpha, beta, |lam| (-t * lam).exp());
    }
}

/// Assemble the lumped mass diagonal and the stiffness matrix of a 1D grid.
///
/// The mass diagonal is [`Grid1D::mass_weights`]. Each element contributes its
/// reference stiffness scaled by `1/h` at the DOFs given by
/// [`Grid1D::eval_dof_index`], which handles the periodic wrap, the shared
/// interface nodes and the Dirichlet endpoints (sentinel `n_dof`, skipped).
fn assemble(grid: &Grid1D) -> (DVector<f64>, DMatrix<f64>) {
    let (n, r, h, n_dof) = (grid.n(), grid.r(), grid.h(), grid.n_dof());
    let basis = CollocationBasis::new(vec![(r + 1, Gauss::Lobatto)]);
    let mass_ref = basis.weights_matrix();
    let diff = basis.diff_matrix(0);
    let stiffness_ref = diff.transpose() * &mass_ref * &diff;

    let mass = DVector::from_vec(grid.mass_weights());
    let mut stiffness = DMatrix::<f64>::zeros(n_dof, n_dof);
    for k in 0..n {
        let dofs: Vec<usize> = (0..=r).map(|j| grid.eval_dof_index(k, j)).collect();
        for (i, &gi) in dofs.iter().enumerate() {
            if gi == n_dof {
                continue;
            }
            for (j, &gj) in dofs.iter().enumerate() {
                if gj != n_dof {
                    stiffness[(gi, gj)] += stiffness_ref[(i, j)] / h;
                }
            }
        }
    }
    (mass, stiffness)
}

// ── ND solver ────────────────────────────────────────────────────────────────

/// `N`-dimensional screened Poisson solver by dense diagonalisation.
///
/// Extends [`Poisson`] to `N` spatial dimensions with a tensor-product
/// eigendecomposition: the eigenvectors are
/// $V^{(0)} \otimes \cdots \otimes V^{(N-1)}$ and the eigenvalues
/// $\lambda = \alpha + \beta \sum_d \mu^{(d)}_{j_d}$.
///
/// ## ND algorithm
///
/// 1. **Mass scaling**: $f_p \leftarrow M_p f_p$, `M_p = ∏_d M^{(d)}_{j_d(p)}`.
/// 2. **Forward transforms**: apply $V^{(d)T}$ along the fibres of each direction `d`.
/// 3. **Pointwise**: $\hat f_j \leftarrow g(\lambda_j)\, \hat f_j$.
/// 4. **Inverse transforms**: apply $V^{(d)}$ along the fibres of each direction `d`.
///
/// Steps 1 and 3 are element-wise parallel; steps 2 and 4 are parallelised
/// over independent fibres with Rayon.
///
/// ## Operator functions
///
/// Every mode carries the vector of per-direction eigenvalues
/// $(\mu^{(0)}, \ldots, \mu^{(N-1)})$, so
/// [`apply_in_place`](Self::apply_in_place) evaluates any `g(λ)` at
/// $\lambda = \alpha + \beta \sum_d \mu^{(d)}$ and
/// [`apply_in_place_dirs`](Self::apply_in_place_dirs) any tensor product
/// $\bigotimes_d g_d(L_d)$ of per-direction operator functions.
pub struct PoissonND<const N: usize> {
    /// One 1D solver per spatial direction.
    solvers: [Poisson; N],
    /// The tensor-product grid.
    grid: GridND<N>,
    /// Bound `Σ_d max |μ^(d)|` on the tensor stiffness eigenvalue magnitude.
    mu_max: f64,
}

impl<const N: usize> PoissonND<N> {
    /// Construct the ND solver.
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
        let solvers: [Poisson; N] =
            std::array::from_fn(|d| Poisson::new(ls[d], xls[d], ns[d], rs[d], bc[d]));
        let grid = GridND::new(ns, rs, ls, xls, std::array::from_fn(|d| bc[d].extension()));
        let mu_max = solvers.iter().map(|s| s.mu_max).sum();
        Self {
            solvers,
            grid,
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
        self.grid.get_x()
    }

    /// The [`lobatto_grid::GridND`] the solver is built on.
    ///
    /// Interpolation of the solution (e.g. [`lobatto_grid::GridND::eval`],
    /// [`lobatto_grid::GridND::convect`], the [`lobatto_grid::EvalPlan`] API)
    /// is provided by the grid directly.
    pub fn grid(&self) -> &GridND<N> {
        &self.grid
    }

    /// Solve `(α M + β K) u = f` in place using the ND algorithm.
    ///
    /// On input, `f` holds the strong-form right-hand side at the tensor-product
    /// Gauss-Lobatto grid returned by [`get_x`](Self::get_x).
    /// On output, `f` contains the FEM solution `u` at the same grid.
    /// `alpha`, `beta` may change from call to call at no cost.
    ///
    /// Eigenvalues with `|λ| ≤ 1e-12 · (|α| + |β|·Σ_d max |μ^(d)|)` are
    /// annihilated (see [`Poisson::solve_in_place`]).
    pub fn solve_in_place(&self, f: &mut [f64], alpha: f64, beta: f64) {
        let tol = PINV_REL_TOL * (alpha.abs() + beta.abs() * self.mu_max);
        self.apply_in_place(f, alpha, beta, |lam| pinv(lam, tol));
    }

    /// Apply the operator function `g(L)` with `L = α I + β M⁻¹K` to the
    /// **nodal vector** `f` in place: `u = g(L) f`, `g` evaluated at the tensor
    /// eigenvalue $\lambda = \alpha + \beta \sum_d \mu^{(d)}_{j_d}$ of every mode.
    ///
    /// One solve's cost for any `g`; see [`Poisson::apply_in_place`] for the
    /// functional calculus and the **nodal-vector input convention** (the
    /// `solve_*` methods take the strong-form right-hand side instead).
    /// `g` must be `Sync`: it is called from the Rayon mode loop.
    ///
    /// ```
    /// # use lobatto_spectral::solver::*;
    /// let solver = PoissonND::<2>::new([1.0; 2], [0.0; 2], [4; 2], [3; 2],
    ///     [BoundaryCondition::Periodic; 2]);
    /// let mut u = vec![1.0; solver.get_x().len()];
    /// let dt = 0.01;
    /// // implicit Euler step (I + dt L)^{-1} u with L = M⁻¹K (α = 0, β = 1);
    /// // the closure captures `dt` by copy
    /// solver.apply_in_place(&mut u, 0.0, 1.0, move |lam| 1.0 / (1.0 + dt * lam));
    /// ```
    pub fn apply_in_place<G: Fn(f64) -> f64 + Sync>(
        &self,
        f: &mut [f64],
        alpha: f64,
        beta: f64,
        g: G,
    ) {
        self.apply_in_place_dirs(f, |mu| g(alpha + beta * mu.iter().sum::<f64>()));
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
    /// # use lobatto_spectral::solver::*;
    /// let solver = PoissonND::<2>::new([1.0; 2], [0.0; 2], [4; 2], [3; 2],
    ///     [BoundaryCondition::Periodic; 2]);
    /// let mut u = vec![1.0; solver.get_x().len()];
    /// let (dt, beta) = (0.01, 2.0);
    /// // split implicit Euler: Π_d (I + dt β M_d⁻¹K_d)^{-1} u
    /// solver.apply_in_place_dirs(&mut u, move |mu| {
    ///     mu.iter().map(|&m| 1.0 / (1.0 + dt * beta * m)).product::<f64>()
    /// });
    /// ```
    pub fn apply_in_place_dirs<G: Fn([f64; N]) -> f64 + Sync>(&self, f: &mut [f64], g: G) {
        let ndofs = self.grid.ndofs();
        let strides = self.grid.strides();
        let total = self.grid.total();
        assert_eq!(f.len(), total, "expected {total} DOFs, got {}", f.len());

        // 1. Mass scaling by the diagonal of M_0 ⊗ … ⊗ M_{N-1}.
        f.par_iter_mut().enumerate().for_each(|(p, fp)| {
            *fp *= (0..N)
                .map(|d| self.solvers[d].mass_diag[(p / strides[d]) % ndofs[d]])
                .product::<f64>();
        });

        // 2. Forward transforms Vᵀ_d along the fibres of each direction.
        for d in 0..N {
            self.sweep(f, d, true);
        }

        // 3. Pointwise g of the per-direction eigenvalues of the mode.
        f.par_iter_mut().enumerate().for_each(|(p, fp)| {
            let mus: [f64; N] =
                std::array::from_fn(|d| self.solvers[d].eigenvalues[(p / strides[d]) % ndofs[d]]);
            *fp *= g(mus);
        });

        // 4. Inverse transforms V_d.
        for d in 0..N {
            self.sweep(f, d, false);
        }
    }

    /// Apply `Vᵀ_d` (`transpose`) or `V_d` to every fibre of `f` along
    /// direction `d`, fibres in parallel.
    // Hot path: per-task buffers only (`for_each_init`), no allocation per
    // fibre; two `Vec`s per fibre here made the solve twice as slow.
    fn sweep(&self, f: &mut [f64], d: usize, transpose: bool) {
        let ndofs = self.grid.ndofs();
        let strides = self.grid.strides();
        let n_d = ndofs[d];
        let stride = strides[d];
        let v = &self.solvers[d].eigenvectors;
        let product = |fiber: &[f64], out: &mut [f64]| {
            if transpose {
                tr_mul(v, fiber, out);
            } else {
                mul(v, fiber, out);
            }
        };
        if stride == 1 {
            // Contiguous fibres: no gather; the product goes through `out`.
            f.par_chunks_mut(n_d).for_each_init(
                || vec![0.0; n_d],
                |out, fiber| {
                    product(fiber, out);
                    fiber.copy_from_slice(out);
                },
            );
            return;
        }
        // SAFETY: nd_fiber_base yields disjoint index sets for distinct `s`
        // (tested in `tests`), so concurrent reads and writes through the raw
        // pointer never alias.
        let ptr = f.as_mut_ptr() as usize;
        (0..f.len() / n_d).into_par_iter().for_each_init(
            || (vec![0.0; n_d], vec![0.0; n_d]),
            |(fiber, out), s| {
                let base = nd_fiber_base(s, d, &ndofs, &strides);
                let p = ptr as *mut f64;
                for (k, x) in fiber.iter_mut().enumerate() {
                    *x = unsafe { *p.add(base + k * stride) };
                }
                product(fiber, out);
                for (k, &o) in out.iter().enumerate() {
                    unsafe { *p.add(base + k * stride) = o };
                }
            },
        );
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
    /// one pass over the data instead of `2N` sweeps. Eigenvalues with
    /// `|λ| ≤ 1e-12 · (|α| + |β|·max |μ⁽ᵈ⁾|)` are annihilated, as in
    /// [`Poisson::solve_in_place`].
    ///
    /// # Panics
    ///
    /// If `d ≥ N` or `f` does not have one value per DOF.
    pub fn solve_in_place_dir(&self, d: usize, f: &mut [f64], alpha: f64, beta: f64) {
        assert!(d < N, "direction {d} out of range for N = {N}");
        let tol = PINV_REL_TOL * (alpha.abs() + beta.abs() * self.solvers[d].mu_max);
        self.apply_in_place_dir(d, f, alpha, beta, |lam| pinv(lam, tol));
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
    pub fn apply_in_place_dir<G: Fn(f64) -> f64 + Sync>(
        &self,
        d: usize,
        f: &mut [f64],
        alpha: f64,
        beta: f64,
        g: G,
    ) {
        self.apply_mu_in_place_dir(d, f, |mu| g(alpha + beta * mu));
    }

    /// Like [`apply_in_place_dir`](Self::apply_in_place_dir), but `g` receives
    /// the raw eigenvalue `μ⁽ᵈ⁾` of `M_d⁻¹K_d` (neither `α` nor `β` applied):
    /// `u = (g(M_d⁻¹K_d) ⊗ I) f`.
    ///
    /// Direction 0 has contiguous fibres, solved in place. For `d > 0` each
    /// strided fibre is gathered into a per-thread buffer, solved and
    /// scattered back, fibres in parallel. Per-thread work vectors: no
    /// allocation per fibre.
    ///
    /// # Panics
    ///
    /// If `d ≥ N` or `f` does not have one value per DOF.
    pub fn apply_mu_in_place_dir<G: Fn(f64) -> f64 + Sync>(&self, d: usize, f: &mut [f64], g: G) {
        assert!(d < N, "direction {d} out of range for N = {N}");
        let total = self.grid.total();
        assert_eq!(f.len(), total, "expected {total} DOFs, got {}", f.len());
        let (ndofs, strides) = (self.grid.ndofs(), self.grid.strides());
        let (m, stride) = (ndofs[d], strides[d]);
        let solver = &self.solvers[d];
        if stride == 1 {
            f.par_chunks_mut(m)
                .for_each_init(|| vec![0.0; m], |hat, fibre| solver.apply_mu_with(fibre, &g, hat));
            return;
        }
        // SAFETY: nd_fiber_base yields disjoint index sets for distinct `s`,
        // so concurrent reads and writes through the raw pointer never alias.
        let ptr = f.as_mut_ptr() as usize;
        (0..total / m).into_par_iter().for_each_init(
            || (vec![0.0; m], vec![0.0; m]),
            |(fibre, hat), s| {
                let base = nd_fiber_base(s, d, &ndofs, &strides);
                let p = ptr as *mut f64;
                for (k, x) in fibre.iter_mut().enumerate() {
                    *x = unsafe { *p.add(base + k * stride) };
                }
                solver.apply_mu_with(fibre, &g, hat);
                for (k, &x) in fibre.iter().enumerate() {
                    unsafe { *p.add(base + k * stride) = x };
                }
            },
        );
    }

    /// Unsplit implicit Euler step `u = (I + dt·L)⁻¹ f` with `L = α I + β M⁻¹K`,
    /// on the nodal vector `f` (see [`apply_in_place`](Self::apply_in_place)).
    pub fn euler_step_in_place(&self, f: &mut [f64], alpha: f64, beta: f64, dt: f64) {
        self.apply_in_place(f, alpha, beta, |lam| 1.0 / (1.0 + dt * lam));
    }

    /// Directionally split implicit Euler step on the nodal vector `f`:
    /// $$u = \prod_d \bigl(I + dt\,(\tfrac{\alpha}{N} I + \beta\, M_d^{-1} K_d)\bigr)^{-1} f,$$
    /// the mass term being split evenly over the `N` directions. The factors
    /// commute, so the product is the exact tensor product of the 1D steps and
    /// costs one application (see [`apply_in_place_dirs`](Self::apply_in_place_dirs)).
    /// Each 1D factor satisfies the one-dimensional discrete maximum principle
    /// (for `dt` large enough relative to `h²`), which the unsplit step does not.
    pub fn euler_split_step_in_place(&self, f: &mut [f64], alpha: f64, beta: f64, dt: f64) {
        let a_n = alpha / N as f64;
        self.apply_in_place_dirs(f, |mu| {
            mu.iter()
                .map(|&m| 1.0 / (1.0 + dt * (a_n + beta * m)))
                .product()
        });
    }

    /// Exact semigroup `u = exp(-t·L) f` with `L = α I + β M⁻¹K`, on the nodal
    /// vector `f` (see [`apply_in_place`](Self::apply_in_place)). Exact in ND:
    /// the exponential of the Kronecker sum is the Kronecker product of the
    /// 1D exponentials.
    pub fn exp_in_place(&self, f: &mut [f64], alpha: f64, beta: f64, t: f64) {
        self.apply_in_place(f, alpha, beta, |lam| (-t * lam).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const ALL_BC: [BoundaryCondition; 3] = [
        BoundaryCondition::Periodic,
        BoundaryCondition::Dirichlet,
        BoundaryCondition::Neumann,
    ];

    /// `Vᵀ M V = I` and `K V = M V diag(μ)` for the assembled operator, all
    /// boundary conditions, one and several elements.
    #[test]
    fn eigendecomposition() {
        for bc in ALL_BC {
            for n in [1usize, 2, 5] {
                for r in 1..=6 {
                    let grid = Grid1D::new(n, r, 1.3, -0.2, bc.extension());
                    if grid.n_dof() == 0 {
                        continue; // Dirichlet with n·r = 1
                    }
                    let solver = Poisson::new(1.3, -0.2, n, r, bc);
                    let (mass, stiffness) = assemble(&grid);
                    let n_dof = grid.n_dof();
                    let v = &solver.eigenvectors;
                    let m = DMatrix::from_diagonal(&mass);

                    let ortho = (v.transpose() * &m * v - DMatrix::<f64>::identity(n_dof, n_dof))
                        .norm();
                    assert!(ortho < 1e-10, "{bc:?} n={n} r={r}: Vᵀ M V ≠ I ({ortho:.2e})");

                    let lambda = DMatrix::from_diagonal(&solver.eigenvalues);
                    let residual = (&stiffness * v - &m * v * &lambda).norm();
                    let scale = stiffness.norm().max(1.0);
                    assert!(
                        residual < 1e-12 * scale,
                        "{bc:?} n={n} r={r}: K V ≠ M V Λ (residual {residual:.2e})"
                    );
                }
            }
        }
    }

    /// The assembled stiffness annihilates constants for periodic and
    /// Neumann conditions and is positive definite for Dirichlet.
    #[test]
    fn stiffness_kernel() {
        for n in [1usize, 3] {
            for r in 2..=5 {
                for bc in ALL_BC {
                    let grid = Grid1D::new(n, r, 1.0, 0.0, bc.extension());
                    let (_, stiffness) = assemble(&grid);
                    let ones = DVector::from_element(grid.n_dof(), 1.0);
                    let k1 = (&stiffness * &ones).norm();
                    let solver = Poisson::new(1.0, 0.0, n, r, bc);
                    let mu_min = solver.eigenvalues.min();
                    match bc {
                        BoundaryCondition::Dirichlet => {
                            assert!(mu_min > 1.0, "Dirichlet n={n} r={r}: μ_min = {mu_min:.2e}");
                        }
                        _ => {
                            assert!(
                                k1 < 1e-12 * stiffness.norm(),
                                "{bc:?} n={n} r={r}: K·1 = {k1:.2e}"
                            );
                            assert!(
                                mu_min.abs() < 1e-12 * solver.mu_max,
                                "{bc:?} n={n} r={r}: μ_min = {mu_min:.2e}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// With `α = 0` the solution is invariant under a constant shift of the
    /// right-hand side (the constant mode is projected out), 1D and 2D.
    #[test]
    fn constant_mode_annihilated() {
        let f = |x: f64| (2.0 * PI * x).sin() + 0.3 * (4.0 * PI * x).cos();
        for bc in [BoundaryCondition::Periodic, BoundaryCondition::Neumann] {
            let solver = Poisson::new(1.0, 0.0, 3, 4, bc);
            let x = solver.get_x();
            let mut u1: Vec<f64> = x.iter().map(|&xi| f(xi)).collect();
            let mut u2: Vec<f64> = x.iter().map(|&xi| f(xi) + 1.0).collect();
            solver.solve_in_place(&mut u1, 0.0, 1.0);
            solver.solve_in_place(&mut u2, 0.0, 1.0);
            let diff = u1.iter().zip(&u2).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(diff < 1e-12, "{bc:?} 1D: constant shift leaked ({diff:.2e})");

            let solver = PoissonND::<2>::new([1.0; 2], [0.0; 2], [3; 2], [4; 2], [bc; 2]);
            let x = solver.get_x();
            let mut u1: Vec<f64> = x.iter().map(|p| f(p[0]) * f(p[1])).collect();
            let mut u2: Vec<f64> = x.iter().map(|p| f(p[0]) * f(p[1]) + 1.0).collect();
            solver.solve_in_place(&mut u1, 0.0, 1.0);
            solver.solve_in_place(&mut u2, 0.0, 1.0);
            let diff = u1.iter().zip(&u2).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(diff < 1e-12, "{bc:?} 2D: constant shift leaked ({diff:.2e})");
        }
    }

    /// The ND solver with one direction equals the 1D solver, and the 2D
    /// solve of a tensor-product right-hand side equals the tensor product
    /// of the 1D eigen-solves (checked through `apply_in_place_dirs`).
    #[test]
    fn nd_consistency() {
        let f = |x: f64| (2.0 * PI * x).sin().exp();
        let s1 = Poisson::new(1.0, 0.0, 2, 3, BoundaryCondition::Periodic);
        let s2 = PoissonND::<1>::new([1.0], [0.0], [2], [3], [BoundaryCondition::Periodic]);
        let x = s1.get_x();
        let mut u1: Vec<f64> = x.iter().map(|&xi| f(xi)).collect();
        let mut u2 = u1.clone();
        s1.solve_in_place(&mut u1, 0.7, 1.3);
        s2.solve_in_place(&mut u2, 0.7, 1.3);
        let diff = u1.iter().zip(&u2).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(diff < 1e-13, "ND<1> ≠ 1D ({diff:.2e})");

        // Split Euler step = product of the 1D steps on a separable field.
        let (dt, alpha, beta) = (0.05, 0.4, 1.0);
        let sx = Poisson::new(1.0, 0.0, 2, 4, BoundaryCondition::Dirichlet);
        let sy = Poisson::new(2.0, 0.0, 3, 3, BoundaryCondition::Neumann);
        let s2d = PoissonND::<2>::new(
            [1.0, 2.0],
            [0.0; 2],
            [2, 3],
            [4, 3],
            [BoundaryCondition::Dirichlet, BoundaryCondition::Neumann],
        );
        let gx = |x: f64| (PI * x).sin();
        let gy = |y: f64| (PI * y).cos();
        let mut ux: Vec<f64> = sx.get_x().iter().map(|&x| gx(x)).collect();
        let mut uy: Vec<f64> = sy.get_x().iter().map(|&y| gy(y)).collect();
        sx.euler_step_in_place(&mut ux, alpha / 2.0, beta, dt);
        sy.euler_step_in_place(&mut uy, alpha / 2.0, beta, dt);
        let mut u: Vec<f64> = s2d.get_x().iter().map(|p| gx(p[0]) * gy(p[1])).collect();
        s2d.euler_split_step_in_place(&mut u, alpha, beta, dt);
        let nx = ux.len();
        let diff = u
            .iter()
            .enumerate()
            .map(|(p, &v)| (v - ux[p % nx] * uy[p / nx]).abs())
            .fold(0.0, f64::max);
        assert!(diff < 1e-13, "split Euler ≠ tensor product of 1D steps ({diff:.2e})");
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

    /// The directional solve is the 1D solve on every fibre, bit for bit.
    #[test]
    fn dir_solve_is_1d_solve_on_every_fibre() {
        let s = solver_3d();
        let (ndofs, strides, total) = (s.grid.ndofs(), s.grid.strides(), s.grid.total());
        let f: Vec<f64> = (0..total).map(|p| (1.7 * p as f64).sin()).collect();
        for d in 0..3 {
            let mut u = f.clone();
            s.solve_in_place_dir(d, &mut u, 0.4, 1.3);
            for k in 0..total / ndofs[d] {
                let base = nd_fiber_base(k, d, &ndofs, &strides);
                let mut fibre: Vec<f64> = (0..ndofs[d]).map(|j| f[base + j * strides[d]]).collect();
                s.solvers[d].solve_in_place(&mut fibre, 0.4, 1.3);
                for (j, v) in fibre.iter().enumerate() {
                    assert_eq!(u[base + j * strides[d]], *v, "d={d} fibre {k} j={j}");
                }
            }
        }
    }

    /// The directional application equals the full ND machinery with a `g`
    /// that keeps only `μ⁽ᵈ⁾`: the other directions' transforms cancel.
    #[test]
    fn dir_apply_matches_nd_apply_keeping_one_direction() {
        let s = solver_3d();
        let f: Vec<f64> = (0..s.grid.total()).map(|p| (0.9 * p as f64).cos()).collect();
        let (alpha, beta, dt) = (0.3, 1.1, 0.05);
        for d in 0..3 {
            let mut u = f.clone();
            s.apply_in_place_dir(d, &mut u, alpha, beta, |lam| 1.0 / (1.0 + dt * lam));
            let mut v = f.clone();
            s.apply_in_place_dirs(&mut v, |mu| 1.0 / (1.0 + dt * (alpha + beta * mu[d])));
            let diff = u.iter().zip(&v).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(diff < 1e-12, "d={d}: {diff:.2e}");
        }
    }

    #[test]
    #[should_panic(expected = "direction 3 out of range for N = 3")]
    fn dir_solve_panics_on_out_of_range_direction() {
        let s = solver_3d();
        let mut f = vec![0.0; s.grid.total()];
        s.solve_in_place_dir(3, &mut f, 1.0, 1.0);
    }

    /// The fibres of every direction partition the DOF array: the `unsafe`
    /// fibre loops (sweeps, directional solves) write disjoint entries.
    fn assert_fibres_partition<const N: usize>(s: &PoissonND<N>) {
        let (ndofs, strides, total) = (s.grid.ndofs(), s.grid.strides(), s.grid.total());
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

    #[test]
    fn unsafe_fibre_loops_touch_disjoint_entries() {
        let bc = ALL_BC;
        assert_fibres_partition(&PoissonND::<1>::new([1.0], [0.0], [3], [4], [bc[1]]));
        assert_fibres_partition(&solver_3d());
        assert_fibres_partition(&PoissonND::<5>::new(
            [1.0; 5],
            [0.0; 5],
            [2, 1, 3, 1, 2],
            [2, 3, 2, 2, 3],
            [bc[0], bc[1], bc[2], bc[2], bc[0]],
        ));
    }
}
