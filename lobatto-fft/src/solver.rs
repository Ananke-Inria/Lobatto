//! HoFFT screened Poisson solvers.
//!
//! Provides [`Poisson`] and [`PoissonND`], which solve
//! $(\alpha M + \beta K)\, u = f$ on tensor-product Gauss-Lobatto grids using
//! the HoFFT transform machinery from [`crate::hofft`].
//!
//! Both solvers are set up once and reused for many right-hand sides.
//! Construction precomputes the FFT plans and the eigendecomposition of
//! all independent local symbols; each call to `solve_in_place` runs in
//! `O(N_dof·log N_dof)` arithmetic operations.
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
/// Each $S(k)$ is a Hermitian matrix assembled from the periodic element matrices
/// modulated by the inter-element phase $e^{-2\pi i k / n}$:
/// $$S(k)_{ij} = \alpha h M_{ij} + \frac{\beta}{h}\bigl(K_{ij} + K_{ri}\,e^{-2\pi ik/n}\bigr), \quad i,j = 1,\ldots,r-1,$$
/// with appropriate modifications for the `(0, 0)` entry that couples the two sides
/// of an element interface.
///
/// The generalised eigendecomposition $S(k) V_k = M V_k \Lambda_k$,
/// precomputed at build time, yields the per-mode solve
/// $$\hat{u}_k = V_k \Lambda_k^{-1} V_k^\dagger \hat{f}_k.$$
///
/// Physical DOF layout: `x`-increasing, flat index `k·r + j` for local node `j`
/// of element `k` (with offset adjustments for Dirichlet; see [`Engine`]).
pub struct Poisson {
    /// Polynomial order `r` per element.
    r: usize,

    /// HoFFT engine: handles extension, packing, forward/inverse FFT, and interpolation.
    pub planner: Engine,

    /// Real eigenvalues of the mass-normalised symbol for each independent Fourier mode `k`:
    /// $\lambda_j(k)$ of $M^{-1/2} S(k) M^{-1/2}$, length `r`.
    eigenvalues: Vec<DVector<f64>>,
    /// $M$-orthonormal eigenvectors $V_k$ satisfying $S(k) V_k = M V_k \Lambda_k$;
    /// `eigenvectors[k]` is the `r × r` complex matrix.
    eigenvectors: Vec<DMatrix<Complex<f64>>>,
    /// Diagonal of the **periodic** local mass matrix (length `r`):
    /// $\tilde{M}_{jj} = h w_j$, with DOF 0 accumulating contributions from both sides
    /// of the element interface: $\tilde{M}_{00} = 2h w_0$.
    periodic_mass_matrix: DVector<f64>,
    /// Pseudo-inverse cutoff for the per-mode solve: `1e-12` × the spectral
    /// radius over all symbols, so that analytically zero eigenvalues computed
    /// as roundoff (`~ε·λ_max`) are filtered regardless of problem size.
    pinv_tol: f64,
    /// Boundary condition type.
    bc: BoundaryCondition,
}

impl Poisson {
    /// Construct the 1D HoFFT solver.
    ///
    /// Assembles the local symbol matrices $S(k)$ for all independent Fourier
    /// modes and precomputes their generalised eigendecompositions.
    ///
    /// # Arguments
    /// - `l`: domain length `L`.
    /// - `xl`: left boundary position `x_l`.
    /// - `n`: number of elements.
    /// - `r`: polynomial order (`r + 1` Gauss-Lobatto nodes per element).
    /// - `alpha`, `beta`: operator coefficients.
    /// - `bc`: boundary condition type.
    pub fn new(
        l: f64,
        xl: f64,
        n: usize,
        r: usize,
        alpha: f64,
        beta: f64,
        bc: BoundaryCondition,
    ) -> Self {
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

        // Compute the symbol of the operator for each frequency
        for (k, s) in symbol.iter_mut().enumerate() {
            let phase = -coeff_phase * PI * (k as f64) / (n as f64);
            let cos = phase.cos();
            let sin = phase.sin();
            let exp = cos + Complex::<f64>::I * sin;

            for i in 1..r {
                for j in 1..r {
                    s[(i, j)] = (alpha * h * mass_matrix[(i, j)]
                        + beta * stiffness_matrix[(i, j)] / h)
                        .into();
                }
            }

            s[(0, 0)] = (2.0 * alpha * h * mass_matrix[(0, 0)]
                + 2.0 * beta * stiffness_matrix[(0, 0)] / h
                + 2.0 * beta * stiffness_matrix[(0, r)] * cos / h)
                .into();

            for i in 1..r {
                s[(0, i)] =
                    beta * stiffness_matrix[(0, i)] / h + beta * stiffness_matrix[(r, i)] * exp / h;
                s[(i, 0)] = s[(0, i)].conj();
            }
        }

        let mass_sqrt_inv_c: DMatrix<Complex<f64>> =
            DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                Complex::new(1.0 / periodic_mass_matrix[i].sqrt(), 0.0)
            }));

        // Generalised eigendecomposition via mass normalisation:
        //   M^{-1/2} S(k) M^{-1/2} W = W Λ   ⟹   eigenvectors V = M^{-1/2} W
        let eigs: Vec<SymmetricEigen<Complex<f64>, Dyn>> = symbol
            .iter()
            .map(|s| SymmetricEigen::new(&mass_sqrt_inv_c * s * &mass_sqrt_inv_c))
            .collect();
        let eigenvalues: Vec<DVector<f64>> = eigs.iter().map(|e| e.eigenvalues.clone()).collect();

        // M-orthonormal eigenvectors of α I + β M^{-1} K
        let eigenvectors: Vec<DMatrix<Complex<f64>>> = eigs
            .iter()
            .map(|e| &mass_sqrt_inv_c * e.eigenvectors.clone())
            .collect();

        // Relative pseudo-inverse cutoff: an analytically zero eigenvalue (the
        // constant mode of the k = 0 symbol when α = 0) is computed as roundoff
        // ~ε·λ_max, which outgrows any absolute threshold as the spectral
        // radius grows with n and r.
        let pinv_tol = 1e-12
            * eigenvalues
                .iter()
                .flat_map(|l| l.iter())
                .fold(0.0f64, |m, &l| m.max(l.abs()));

        Self {
            r,
            planner,
            eigenvalues,
            eigenvectors,
            periodic_mass_matrix,
            pinv_tol,
            bc,
        }
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
    ///
    /// Executes the three-step HoFFT solve:
    /// 1. Mass scaling + forward FFT.
    /// 2. Per-mode symbol solve: $\hat{u}_k = V_k \Lambda_k^{-1} V_k^\dagger \hat{f}_k$.
    /// 3. Inverse FFT.
    pub fn solve_in_place(&self, f: &mut [Complex<f64>]) {
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

        // Step 2: symbol solve via eigendecomposition.
        // For each mode k: v = Q†·g_k, pointwise divide by λ, result = Q·v.
        let n_freq = self.eigenvalues.len();
        let mut col = DVector::<Complex<f64>>::zeros(self.r);
        for k in 0..n_freq {
            self.planner.get_values(k, f, col.as_mut_slice());
            // Q† · col
            let mut v = self.eigenvectors[k].ad_mul(&col);
            // Pointwise pseudo-inverse
            for j in 0..self.r {
                let lam = self.eigenvalues[k][j];
                if lam.abs() > self.pinv_tol {
                    v[j] /= lam;
                } else {
                    v[j] = Complex::new(0.0, 0.0);
                }
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
    /// The operator $(\alpha M + \beta K)$ is real, so a real right-hand side
    /// has an exactly real solution: `f` is copied into an internal complex
    /// buffer, solved, and the real parts are written back. Costs the same as
    /// the complex solve plus the two conversion passes and one temporary
    /// complex buffer of `f.len()` entries.
    pub fn solve_in_place_real(&self, f: &mut [f64]) {
        let mut buf: Vec<Complex<f64>> = f.iter().map(|&v| Complex::new(v, 0.0)).collect();
        self.solve_in_place(&mut buf);
        for (o, b) in f.iter_mut().zip(&buf) {
            *o = b.re;
        }
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
///    $$\hat{u}_{k,j} \leftarrow \hat{f}_{k,j} \Big/ \Bigl(\alpha + \beta \sum_d \lambda^{(d)}_{k_d, j_d}\Bigr)$$
///    after projecting onto and back from the per-direction eigenbases $V_{k_d}^{(d)}$.
/// 4. **Inverse ND FFT**: apply [`EngineND::inverse`] direction by direction.
///
/// Step 3 is parallelised over independent mode tuples with Rayon.
pub struct PoissonND<const N: usize> {
    /// One 1D solver per spatial direction (built with `α = 0`, `β = 1` to obtain
    /// the per-direction eigenvectors; actual `α`, `β` applied at the ND level).
    solvers: [Poisson; N],
    /// ND HoFFT engine for the forward/inverse transforms, interpolation, and advection.
    pub fft: EngineND<N>,
    /// Diagonal of the Kronecker-product periodic mass matrix
    /// $\tilde{M}^{(0)} \otimes \cdots \otimes \tilde{M}^{(N-1)}$,
    /// length $\prod_d r_d$, direction 0 (`x`) varies fastest.
    periodic_mass_matrix: DVector<f64>,
    /// Coefficient `α` of the mass term.
    alpha: f64,
    /// Coefficient `β` of the stiffness term.
    beta: f64,
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
    /// Pseudo-inverse cutoff for the ND symbol solve: `1e-12` × an upper bound
    /// on the combined eigenvalue magnitude (see [`PoissonND::new`]).
    pinv_tol: f64,
}

impl<const N: usize> PoissonND<N> {
    /// Construct the ND HoFFT solver.
    ///
    /// # Arguments
    /// - `ls[d]`: domain length `L_d` in direction `d`.
    /// - `xls[d]`: left boundary position $x_{l,d}$ in direction `d`.
    /// - `ns[d]`: number of elements in direction `d`.
    /// - `rs[d]`: polynomial order `r_d` in direction `d`.
    /// - `alpha`, `beta`: operator coefficients.
    /// - `bc[d]`: boundary condition for direction `d`.
    pub fn new(
        ls: [f64; N],
        xls: [f64; N],
        ns: [usize; N],
        rs: [usize; N],
        alpha: f64,
        beta: f64,
        bc: [BoundaryCondition; N],
    ) -> Self {
        let solvers =
            std::array::from_fn(|d| Poisson::new(ls[d], xls[d], ns[d], rs[d], 0.0, 1.0, bc[d]));

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

        // Relative pseudo-inverse cutoff (see [`Poisson::new`]): scale with an
        // upper bound `|α| + |β|·Σ_d max |λ^(d)|` on the combined eigenvalue
        // magnitude `α + β·Σ_d λ^(d)`.
        let pinv_tol = 1e-12
            * (alpha.abs()
                + beta.abs()
                    * solvers
                        .iter()
                        .map(|s| {
                            s.eigenvalues
                                .iter()
                                .flat_map(|l| l.iter())
                                .fold(0.0f64, |m, &l| m.max(l.abs()))
                        })
                        .sum::<f64>());

        Self {
            solvers,
            fft,
            periodic_mass_matrix,
            alpha,
            beta,
            n_freqs,
            n_freq_total,
            n_freq_strides,
            j_offsets,
            pinv_tol,
        }
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
    ///
    /// See the struct-level documentation for the four-step ND algorithm.
    /// Step 3 (symbol solve) is parallelised over independent mode tuples with Rayon.
    pub fn solve_in_place(&self, f: &mut [Complex<f64>]) {
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

            // Pointwise divide by ND eigenvalue λ = α + Σ_d β λ^{(d)}_{k_d, j_d}.
            for j_flat in 0..r_total {
                let lambda = self.alpha
                    + (0..N)
                        .map(|d| {
                            let j_d = (j_flat / r_strides[d]) % rs[d];
                            self.beta * self.solvers[d].eigenvalues[ks[d]][j_d]
                        })
                        .sum::<f64>();
                if lambda.abs() > self.pinv_tol {
                    v[j_flat] /= lambda;
                } else {
                    v[j_flat] = Complex::new(0.0, 0.0);
                }
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
    /// The operator $(\alpha M + \beta K)$ is real, so a real right-hand side
    /// has an exactly real solution: `f` is copied into an internal complex
    /// buffer, solved, and the real parts are written back. Costs the same as
    /// the complex solve plus the two (parallel) conversion passes and one
    /// temporary complex buffer of `f.len()` entries.
    pub fn solve_in_place_real(&self, f: &mut [f64]) {
        let mut buf: Vec<Complex<f64>> = f.par_iter().map(|&v| Complex::new(v, 0.0)).collect();
        self.solve_in_place(&mut buf);
        f.par_iter_mut()
            .zip(buf.par_iter())
            .for_each(|(o, b)| *o = b.re);
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
                let solver = Poisson::new(1.0, 0.0, n, r, 1.0, 1.0, BoundaryCondition::Periodic);
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

                    // 3. Check Q† M Q = I (already covered above) and eigenvalues are real.
                    //    Also verify round-trip: Q diag(1/λ) Q† M V Λ = V
                    let lambda_diag = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                        Complex::new(solver.eigenvalues[k][i], 0.0)
                    }));
                    let lambda_inv = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                        let lam = solver.eigenvalues[k][i];
                        Complex::new(if lam.abs() > 1e-12 { 1.0 / lam } else { 0.0 }, 0.0)
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
        let solver = Poisson::new(1.0, 0.0, n, r, 0.0, 1.0, BoundaryCondition::Periodic);
        let x = solver.get_x();

        let f = |x: f64| (2.0 * PI * x).sin() + 0.3 * (4.0 * PI * x).cos();
        let mut u1: Vec<Complex<f64>> = x.iter().map(|&xi| Complex::new(f(xi), 0.0)).collect();
        let mut u2: Vec<Complex<f64>> =
            x.iter().map(|&xi| Complex::new(f(xi) + 1.0, 0.0)).collect();
        solver.solve_in_place(&mut u1);
        solver.solve_in_place(&mut u2);

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
            0.0,
            1.0,
            [BoundaryCondition::Periodic; 2],
        );
        let x = solver.get_x();

        let f = |p: &[f64; 2]| (2.0 * PI * p[0]).sin() * (2.0 * PI * p[1]).cos();
        let mut u1: Vec<Complex<f64>> = x.iter().map(|p| Complex::new(f(p), 0.0)).collect();
        let mut u2: Vec<Complex<f64>> = x.iter().map(|p| Complex::new(f(p) + 1.0, 0.0)).collect();
        solver.solve_in_place(&mut u1);
        solver.solve_in_place(&mut u2);

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

    /// Check that with beta=0 the tensorial structure recovers the inverse of the local mass matrix.
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
                1.0,
                0.0,
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
                1.0,
                0.0,
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
                1.0,
                0.0,
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
