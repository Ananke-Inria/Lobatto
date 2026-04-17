//! High-order screened Poisson solver based on Lagrangian functions on Gauss-Lobatto points.
//!
//! Solves the screened Poisson equation
//! $$(\alpha I - \Delta K)~u = f$$
//! on the box $[x_{l}, x_{l}+L]^N$ using a **single** high-order spectral element
//! of polynomial order $r$ per direction.  Supports periodic, Dirichlet, and Neumann
//! boundary conditions via the [`BoundaryCondition`] enum.
//!
//! ## Boundary conditions and DOF layout
//!
//! The set of active nodes depends on the boundary condition:
//!
//! | BC          | Active nodes            | $N_\mathrm{dof}$ | Enforced condition              |
//! |-------------|-------------------------|------------------|---------------------------------|
//! | `Periodic`  | $0, \ldots, r-1$        | $r$              | $u(x_l) = u(x_l+L)$            |
//! | `Dirichlet` | $1, \ldots, r-1$        | $r-1$            | $u(x_l) = u(x_l+L) = 0$        |
//! | `Neumann`   | $0, \ldots, r$          | $r+1$            | $u'(x_l) = u'(x_l+L) = 0$      |
//!
//! ## Algorithm
//!
//! Denoting $M$ the mass-lumped matrix and $K$ the
//! stiffness matrix built from the spectral finite-element method on one hypercube, we define  $A = \alpha M + \beta K$ a symmetric positive semi-definite.
//! Its generalised eigendecomposition $A V = M V \Lambda$ is computed once at construction:
//! $$M^{-1/2} A M^{-1/2} = W \Lambda W^T, \quad V = M^{-1/2} W,$$
//! yielding $M$-orthonormal columns $V^T M V = I$.
//!
//! Given the right-hand side $f$, the solve then executes four steps:
//! 1. **Mass scaling**: $g \leftarrow M f$.
//! 2. **Forward transform**: $\hat{g} \leftarrow V^T g$.
//! 3. **Spectral division**: $\hat{u}_j \leftarrow \hat{g}_j / \lambda_j$
//!    (near-zero eigenvalues are set to zero — pseudo-inverse).
//! 4. **Inverse transform**: $u \leftarrow V \hat{u}$.
//!
//! The $N$-dimensional solver [`PoissonND`] exploits the tensor-product structure.
//! The eigenvectors and eigenvalues factorise as
//! $$V_\mathrm{ND} = V^{(0)} \otimes \cdots \otimes V^{(N-1)},
//!   \quad \lambda_{j_0,\ldots,j_{N-1}} = \alpha + \beta \sum_{d=0}^{N-1} \lambda^{(d)}_{j_d},$$
//! so steps 2–4 are applied direction by direction (sum factorisation),
//! parallelised with Rayon over independent fibres.

use lobatto::collocation::{CollocationBasis, Gauss};
use lobatto::utilities::{barycentric_weights, lagrangian_interpolation};
use nalgebra::*;
use rayon::prelude::*;

/// Boundary condition type for the HoT solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Periodic: right endpoint identified with left endpoint.
    /// $N_\mathrm{dof} = r$.
    Periodic,
    /// Homogeneous Dirichlet: $u = 0$ at both endpoints; boundary nodes excluded.
    /// $N_\mathrm{dof} = r - 1$.
    Dirichlet,
    /// Homogeneous Neumann (natural BC): $\partial_n u = 0$ at both endpoints;
    /// all nodes including the endpoints are active.
    /// $N_\mathrm{dof} = r + 1$.
    Neumann,
}

// ── 1D ──────────────────────────────────────────────────────────────────────

/// 1D HoT screened Poisson solver on a single element of polynomial order $r$.
///
/// Solves $(\alpha M + \beta K)\, u = f$ on $[x_l, x_l + L]$ using a single
/// GLL spectral element of order $r$.  The active DOFs depend on the boundary
/// condition (see the module-level table).
///
/// ## Construction
///
/// At build time the generalised eigendecomposition
/// $$M^{-1/2}(\alpha M + \beta K)M^{-1/2} = W \Lambda W^T, \quad V = M^{-1/2} W$$
/// is computed and the pseudo-inverse $A^+ = V \Lambda^+ V^T M$ is stored.
/// The solve is then a single matrix-vector product: $u = A^+ f$.
pub struct Poisson {
    /// Domain length $L$.
    l: f64,
    /// Position of the left boundary $x_l$.
    xl: f64,
    /// Polynomial order $r$ (the GLL basis has $r+1$ nodes).
    r: usize,
    /// Number of active DOFs $N_\mathrm{dof}$ (depends on BC; see module-level table).
    pub ndof: usize,
    /// Boundary condition type.
    bc: BoundaryCondition,
    /// GLL basis with $r+1$ nodes on $\[0,1\]$.
    lobatto_basis: CollocationBasis,
    /// Pre-computed pseudo-inverse $A^+ = V \Lambda^+ V^T M$ of the local symbol.
    symbol_pseudo_inv: DMatrix<f64>,
    /// Eigenvalues $\lambda_j$ of the mass-normalised symbol
    /// $M^{-1/2}(\alpha M + \beta K)M^{-1/2}$.
    pub eigenvalues: DVector<f64>,
    /// $M$-orthonormal eigenvectors: columns of $V = M^{-1/2} W$ where $W$
    /// diagonalises $M^{-1/2}(\alpha M + \beta K)M^{-1/2}$.
    /// Satisfy $V^T M V = I$.
    pub eigenvectors: DMatrix<f64>,
    /// Diagonal of the active mass matrix $M$, length $N_\mathrm{dof}$.
    pub mass_diag: DVector<f64>,
    /// Barycentric weights for the $r+1$ GLL nodes on $\[0,1\]$ (used in [`eval`](Self::eval)).
    bary_weights: Vec<f64>,
}

impl Poisson {
    /// Construct the 1D HoT solver.
    ///
    /// Assembles the local operator $\alpha M + \beta K$ for the chosen boundary
    /// condition, then computes and stores its generalised eigendecomposition.
    ///
    /// # Arguments
    /// - `l`: domain length $L$.
    /// - `xl`: position of the left boundary $x_l$.
    /// - `r`: polynomial order ($r+1$ GLL nodes on $[x_l, x_l+L]$).
    /// - `alpha`, `beta`: operator coefficients.
    /// - `bc`: boundary condition type.
    pub fn new(l: f64, xl: f64, r: usize, alpha: f64, beta: f64, bc: BoundaryCondition) -> Self {
        let lobatto_basis = CollocationBasis::new(vec![(r + 1, Gauss::Lobatto)]);
        let mass_matrix = lobatto_basis.weights_matrix();
        let diff_matrix = lobatto_basis.diff_matrix(0);
        let stiffness_matrix = diff_matrix.transpose() * &mass_matrix * &diff_matrix;
        let h = l;

        let (ndof, mass_diag, symbol) = match bc {
            BoundaryCondition::Periodic => {
                // DOFs: nodes 0..r-1. Node r is identified with node 0 (periodic wrap).
                let ndof = r;
                let mass_diag = DVector::from_fn(r, |i, _| {
                    // DOF 0 accumulates mass from both sides of the periodic boundary.
                    if i == 0 {
                        h * 2.0 * mass_matrix[(0, 0)]
                    } else {
                        h * mass_matrix[(i, i)]
                    }
                });
                let mut sym = DMatrix::<f64>::zeros(r, r);
                // (0,0) receives stiffness contributions from both sides of the wrap.
                sym[(0, 0)] = 2.0 * alpha * h * mass_matrix[(0, 0)]
                    + 2.0 * beta * (stiffness_matrix[(0, 0)] + stiffness_matrix[(0, r)]) / h;
                for i in 1..r {
                    let val = beta * (stiffness_matrix[(0, i)] + stiffness_matrix[(r, i)]) / h;
                    sym[(0, i)] = val;
                    sym[(i, 0)] = val;
                }
                for i in 1..r {
                    for j in 1..r {
                        sym[(i, j)] =
                            alpha * h * mass_matrix[(i, j)] + beta * stiffness_matrix[(i, j)] / h;
                    }
                }
                (ndof, mass_diag, sym)
            }

            BoundaryCondition::Dirichlet => {
                // DOFs: interior nodes 1..r-1. Both endpoints are enforced to zero.
                let ndof = r - 1;
                let mass_diag = DVector::from_fn(ndof, |i, _| h * mass_matrix[(i + 1, i + 1)]);
                let mut sym = DMatrix::<f64>::zeros(ndof, ndof);
                for i in 0..ndof {
                    for j in 0..ndof {
                        sym[(i, j)] = beta * stiffness_matrix[(i + 1, j + 1)] / h;
                        if i == j {
                            sym[(i, j)] += alpha * h * mass_matrix[(i + 1, i + 1)];
                        }
                    }
                }
                (ndof, mass_diag, sym)
            }

            BoundaryCondition::Neumann => {
                // DOFs: all nodes 0..r. Zero-flux Neumann is a natural BC (no boundary term).
                let ndof = r + 1;
                let mass_diag = DVector::from_fn(ndof, |i, _| h * mass_matrix[(i, i)]);
                let mut sym = DMatrix::<f64>::zeros(ndof, ndof);
                for i in 0..ndof {
                    for j in 0..ndof {
                        sym[(i, j)] = beta * stiffness_matrix[(i, j)] / h;
                        if i == j {
                            sym[(i, j)] += alpha * h * mass_matrix[(i, i)];
                        }
                    }
                }
                (ndof, mass_diag, sym)
            }
        };

        // Generalised eigendecomposition via mass normalisation:
        //   M^{-1/2} symbol M^{-1/2} W = W Λ   ⟹   eigenvectors V = M^{-1/2} W
        let mass_sqrt_inv = DMatrix::from_diagonal(&mass_diag.map(|v| 1.0 / v.sqrt()));
        let eig = SymmetricEigen::new(&mass_sqrt_inv * &symbol * &mass_sqrt_inv);
        let eigenvalues = eig.eigenvalues.clone();
        let eigenvectors = &mass_sqrt_inv * eig.eigenvectors;

        // Pseudo-inverse: V diag(1/λ) V^T  (zero out near-zero eigenvalues)
        const PINV_TOL: f64 = 1e-12;
        let lambda_pinv = eigenvalues.map(|v| if v.abs() > PINV_TOL { 1.0 / v } else { 0.0 });
        let symbol_pseudo_inv =
            &eigenvectors * DMatrix::from_diagonal(&lambda_pinv) * eigenvectors.transpose();

        let bary_weights = barycentric_weights(&lobatto_basis.points_1d(0));

        Self {
            l,
            xl,
            r,
            ndof,
            bc,
            lobatto_basis,
            symbol_pseudo_inv,
            eigenvalues,
            eigenvectors,
            mass_diag,
            bary_weights,
        }
    }

    /// Return the active GLL nodes on $[x_l, x_l+L]$, length $N_\mathrm{dof}$.
    ///
    /// - `Periodic`:  nodes $0, \ldots, r-1$ (right endpoint excluded — it is
    ///   identified with the left endpoint by periodicity).
    /// - `Dirichlet`: interior nodes $1, \ldots, r-1$ (both boundary nodes
    ///   are enforced to zero and are not returned).
    /// - `Neumann`:   all nodes $0, \ldots, r$ (including both endpoints).
    pub fn get_x(&self) -> Vec<f64> {
        let pts = self.lobatto_basis.points_1d(0);
        match self.bc {
            BoundaryCondition::Periodic => (0..self.r).map(|i| self.xl + self.l * pts[i]).collect(),
            BoundaryCondition::Dirichlet => {
                (1..self.r).map(|i| self.xl + self.l * pts[i]).collect()
            }
            BoundaryCondition::Neumann => (0..=self.r).map(|i| self.xl + self.l * pts[i]).collect(),
        }
    }

    /// Solve $(\alpha M + \beta K)\, u = f$ in place.
    ///
    /// On input, `f` contains the strong-form right-hand side evaluated at
    /// the active GLL nodes returned by [`get_x`](Self::get_x).
    /// On output, `f` contains the FEM solution $u$ at those same nodes.
    ///
    /// The solve is $u = A^+ (M f)$, a single matrix-vector product with the
    /// pre-computed pseudo-inverse.  Cost: $O(N_\mathrm{dof}^2)$.
    pub fn solve_in_place(&self, f: &mut Vec<f64>) {
        let g = DVector::from_fn(self.ndof, |i, _| self.mass_diag[i] * f[i]);
        let x = &self.symbol_pseudo_inv * &g;
        for i in 0..self.ndof {
            f[i] = x[i];
        }
    }

    /// Evaluate the GLL interpolant of the solution at an arbitrary point `x`.
    ///
    /// Uses the barycentric Lagrange formula to evaluate
    /// $u_h(x) = \sum_i u_i L_i(\xi(x))$
    /// where $\xi(x) = (x - x_l)/L \in [0,1]$.
    ///
    /// - `Periodic`: `x` is folded into $[x_l, x_l+L)$ by periodicity.
    /// - `Dirichlet`/`Neumann`: `x` is clamped to $[x_l, x_l+L]$.
    pub fn eval(&self, x: f64, u: &[f64]) -> f64 {
        let pts = self.lobatto_basis.points_1d(0);
        match self.bc {
            BoundaryCondition::Periodic => {
                let xi = (x - self.xl).rem_euclid(self.l) / self.l;
                let phi = lagrangian_interpolation(pts, &self.bary_weights, xi);
                // phi[r] (right endpoint) folds into DOF 0 by periodicity.
                let mut val = u[0] * (phi[0] + phi[self.r]);
                for i in 1..self.r {
                    val += u[i] * phi[i];
                }
                val
            }
            BoundaryCondition::Dirichlet => {
                let xi = ((x - self.xl) / self.l).clamp(0.0, 1.0);
                let phi = lagrangian_interpolation(pts, &self.bary_weights, xi);
                // Endpoint basis functions (phi[0], phi[r]) multiply zero-valued DOFs.
                (1..self.r).map(|i| u[i - 1] * phi[i]).sum()
            }
            BoundaryCondition::Neumann => {
                let xi = ((x - self.xl) / self.l).clamp(0.0, 1.0);
                let phi = lagrangian_interpolation(pts, &self.bary_weights, xi);
                (0..=self.r).map(|i| u[i] * phi[i]).sum()
            }
        }
    }
}

// ── ND ──────────────────────────────────────────────────────────────────────

/// $N$-dimensional HoT screened Poisson solver.
///
/// Extends [`Poisson`] to $N$ spatial dimensions using a **tensor-product
/// eigendecomposition**.  Each direction $d$ may have an independent polynomial
/// order $r_d$ and [`BoundaryCondition`].
///
/// ## Algorithm
///
/// The ND operator has the tensor-product structure
/// $$A_\mathrm{ND} = \alpha M_0 \otimes \cdots \otimes M_{N-1}
///   + \beta \sum_{d=0}^{N-1} M_0 \otimes \cdots \otimes K_d \otimes \cdots \otimes M_{N-1}.$$
/// Its eigendecomposition factorises direction by direction:
/// $$V_\mathrm{ND} = V^{(0)} \otimes \cdots \otimes V^{(N-1)},
///   \quad \lambda_{j_0,\ldots,j_{N-1}} = \alpha + \beta \sum_d \lambda^{(d)}_{j_d}.$$
///
/// The solve proceeds as:
/// 1. $f \leftarrow M_\mathrm{ND}\, f$ — pointwise multiplication by the diagonal ND mass matrix.
/// 2. For each $d$: apply $V^{(d)\,T}$ along fibres in direction $d$.
/// 3. $f_{j_0,\ldots,j_{N-1}} \leftarrow f_{j_0,\ldots,j_{N-1}} / \lambda_{j_0,\ldots,j_{N-1}}$
///    (pseudo-inverse: zero out near-zero eigenvalues).
/// 4. For each $d$: apply $V^{(d)}$ along fibres in direction $d$.
///
/// Steps 1 and 3 are embarrassingly parallel; steps 2 and 4 are parallelised
/// over independent fibres with Rayon.
pub struct PoissonND<const N: usize> {
    /// One 1D solver per spatial direction.
    pub solvers: [Poisson; N],
    /// Coefficient $\alpha$ of the mass term.
    pub alpha: f64,
    /// Coefficient $\beta$ of the stiffness term.
    pub beta: f64,
    /// Polynomial order $r_d$ per direction.
    pub rs: [usize; N],
    /// DOF counts per direction: $N_\mathrm{dof}^{(d)}$ of the $d$-th 1D solver.
    pub ndofs: [usize; N],
    /// Boundary condition per direction.
    pub bcs: [BoundaryCondition; N],
    /// Domain length $L_d$ per direction.
    pub ls: [f64; N],
    /// Left boundary position $x_{l,d}$ per direction.
    pub xls: [f64; N],
    /// Total number of DOFs: $\prod_d N_\mathrm{dof}^{(d)}$.
    pub r_total: usize,
    /// Column-major strides for the DOF tensor (direction 0 varies fastest).
    pub local_strides: [usize; N],
}

impl<const N: usize> PoissonND<N> {
    /// Construct the ND HoT solver.
    ///
    /// Builds one [`Poisson`] per direction with `alpha = 0, beta = 1`
    /// (pure stiffness) to obtain the per-direction eigenvectors; the full operator
    /// coefficients $\alpha$ and $\beta$ are applied at the ND level during the solve.
    ///
    /// # Arguments
    /// - `ls[d]`: domain length $L_d$ in direction $d$.
    /// - `xls[d]`: left boundary position $x_{l,d}$ in direction $d$.
    /// - `rs[d]`: polynomial order $r_d$ in direction $d$.
    /// - `bcs[d]`: boundary condition for direction $d$.
    /// - `alpha`, `beta`: operator coefficients.
    pub fn new(
        ls: [f64; N],
        xls: [f64; N],
        rs: [usize; N],
        alpha: f64,
        beta: f64,
        bcs: [BoundaryCondition; N],
    ) -> Self {
        // 1D solvers are built with (alpha=0, beta=1) so their eigenvectors diagonalise
        // the pure stiffness operator; alpha is applied at the ND level in solve_in_place.
        let solvers = std::array::from_fn(|d| Poisson::new(ls[d], xls[d], rs[d], 0.0, 1.0, bcs[d]));
        let ndofs: [usize; N] = std::array::from_fn(|d| solvers[d].ndof);
        let r_total = ndofs.iter().product();
        let local_strides = Self::compute_strides(&ndofs);
        Self {
            solvers,
            alpha,
            beta,
            rs,
            ndofs,
            bcs,
            ls,
            xls,
            r_total,
            local_strides,
        }
    }

    /// Return the tensor-product GLL grid as a `Vec<[f64; N]>` of length
    /// $\prod_d N_\mathrm{dof}^{(d)}$.
    ///
    /// Entry `p` holds the coordinates $(x_0, \ldots, x_{N-1})$ of the $p$-th
    /// DOF.  Direction 0 varies fastest (column-major ordering).
    pub fn get_x(&self) -> Vec<[f64; N]> {
        let x1d: [Vec<f64>; N] = std::array::from_fn(|d| self.solvers[d].get_x());
        (0..self.r_total)
            .map(|p| std::array::from_fn(|d| x1d[d][(p / self.local_strides[d]) % self.ndofs[d]]))
            .collect()
    }

    /// Solve $(\alpha M + \beta K)\, u = f$ in place using the ND HoT algorithm.
    ///
    /// On input, `f` contains the strong-form right-hand side at the
    /// tensor-product grid returned by [`get_x`](Self::get_x).
    /// On output, `f` contains the FEM solution $u$ at the same grid.
    ///
    /// # Parallelism
    /// Mass scaling (step 1) and spectral division (step 3) are element-wise
    /// parallel via `par_iter_mut`.  The eigenvector transforms (steps 2 and 4)
    /// are parallelised over independent fibres with Rayon.
    pub fn solve_in_place(&self, f: &mut Vec<f64>) {
        // Step 0: multiply by diagonal ND mass matrix (trivially parallel).
        f.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, fi)| {
                *fi *= (0..N)
                    .map(|d| self.solvers[d].mass_diag[(i / self.local_strides[d]) % self.ndofs[d]])
                    .product::<f64>();
            });

        // Step a: apply Q^T_d for each direction.
        for d in 0..N {
            let r_d = self.ndofs[d];
            let stride = self.local_strides[d];
            let eigvecs = &self.solvers[d].eigenvectors;
            // SAFETY: flat_base produces disjoint index sets for distinct `s`, so
            // concurrent reads and writes never alias.
            let ptr = f.as_mut_ptr() as usize;
            (0..self.r_total / r_d).into_par_iter().for_each(|s| {
                let bj = Self::flat_base_impl(s, d, &self.ndofs, &self.local_strides);
                let p = ptr as *mut f64;
                let fiber = DVector::from_fn(r_d, |k, _| unsafe { *p.add(bj + k * stride) });
                let x = eigvecs.tr_mul(&fiber);
                for k in 0..r_d {
                    unsafe { *p.add(bj + k * stride) = x[k] };
                }
            });
        }

        // Step b: divide by ND eigenvalue (pseudo-inverse: skip near-zero modes).
        const PINV_TOL: f64 = 1e-12;
        f.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(j_flat, fj)| {
                let lambda = self.alpha
                    + (0..N)
                        .map(|d| {
                            let j_d = (j_flat / self.local_strides[d]) % self.ndofs[d];
                            self.beta * self.solvers[d].eigenvalues[j_d]
                        })
                        .sum::<f64>();
                if lambda.abs() > PINV_TOL {
                    *fj /= lambda;
                } else {
                    *fj = 0.0;
                }
            });

        // Step c: apply Q_d for each direction (same structure as step a).
        for d in 0..N {
            let r_d = self.ndofs[d];
            let stride = self.local_strides[d];
            let eigvecs = &self.solvers[d].eigenvectors;
            // SAFETY: same non-overlap argument as step a.
            let ptr = f.as_mut_ptr() as usize;
            (0..self.r_total / r_d).into_par_iter().for_each(|s| {
                let bj = Self::flat_base_impl(s, d, &self.ndofs, &self.local_strides);
                let p = ptr as *mut f64;
                let fiber = DVector::from_fn(r_d, |k, _| unsafe { *p.add(bj + k * stride) });
                let x = eigvecs * &fiber;
                for k in 0..r_d {
                    unsafe { *p.add(bj + k * stride) = x[k] };
                }
            });
        }
    }

    /// Evaluate the GLL interpolant of `u` at a batch of arbitrary positions.
    ///
    /// For each query point $x$, computes
    /// $$u_h(x) = \sum_{j_0, \ldots, j_{N-1}} u_{j_0,\ldots,j_{N-1}}
    ///            \prod_d L^{(d)}_{j_d}(\xi_d(x_d))$$
    /// using sum factorisation over directions, with cost
    /// $O\bigl(\sum_d N_\mathrm{dof}^{(d)} \cdot \prod_{d'<d} N_\mathrm{dof}^{(d')}\bigr)$
    /// per point.  Points are processed in parallel with Rayon.
    pub fn eval(&self, xs: &[[f64; N]], u: &[f64]) -> Vec<f64> {
        xs.par_iter()
            .map(|x| {
                // Per-direction active basis values.
                let phi: [Vec<f64>; N] = std::array::from_fn(|d| {
                    let xi = match self.bcs[d] {
                        BoundaryCondition::Periodic => {
                            (x[d] - self.xls[d]).rem_euclid(self.ls[d]) / self.ls[d]
                        }
                        _ => ((x[d] - self.xls[d]) / self.ls[d]).clamp(0.0, 1.0),
                    };
                    let phi_1d = lagrangian_interpolation(
                        self.solvers[d].lobatto_basis.points_1d(0),
                        &self.solvers[d].bary_weights,
                        xi,
                    );
                    let r_d = self.rs[d];
                    match self.bcs[d] {
                        BoundaryCondition::Periodic => {
                            // Fold right-endpoint basis value into DOF 0.
                            let mut phi_d = phi_1d[..r_d].to_vec();
                            phi_d[0] += phi_1d[r_d];
                            phi_d
                        }
                        BoundaryCondition::Dirichlet => {
                            // Interior nodes only (endpoints are zero-valued DOFs).
                            phi_1d[1..r_d].to_vec()
                        }
                        BoundaryCondition::Neumann => {
                            // All r+1 nodes.
                            phi_1d.to_vec()
                        }
                    }
                });

                // Sum-factorised eval: direction 0 varies fastest (stride = 1).
                let c0 = self.ndofs[0];
                let rest0 = u.len() / c0;
                let mut v = vec![0.0_f64; rest0];
                for s in 0..rest0 {
                    let base = s * c0;
                    for j in 0..c0 {
                        v[s] += phi[0][j] * u[base + j];
                    }
                }
                for d in 1..N {
                    let c_d = self.ndofs[d];
                    let next = v.len() / c_d;
                    let mut v_next = vec![0.0_f64; next];
                    for s in 0..next {
                        let base = s * c_d;
                        for j in 0..c_d {
                            v_next[s] += phi[d][j] * v[base + j];
                        }
                    }
                    v = v_next;
                }
                v[0]
            })
            .collect()
    }

    // Column-major strides for a tensor-product array of shape `sizes`.
    fn compute_strides(sizes: &[usize; N]) -> [usize; N] {
        let mut st = [1usize; N];
        for d in 1..N {
            st[d] = st[d - 1] * sizes[d - 1];
        }
        st
    }

    // Flat start index of the `s`-th fiber along direction `d` in the DOF tensor.
    // SAFETY invariant: for a fixed `d`, distinct `s` values yield disjoint index sets.
    fn flat_base_impl(s: usize, d: usize, ndofs: &[usize; N], strides: &[usize; N]) -> usize {
        let mut base = 0usize;
        let mut rem = s;
        for d2 in 0..N {
            if d2 == d {
                continue;
            }
            base += (rem % ndofs[d2]) * strides[d2];
            rem /= ndofs[d2];
        }
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_1d_periodic_eigendecomposition() {
        const ORTHO_TOL: f64 = 1e-10;
        const RECON_TOL: f64 = 1e-10;

        for &r in &[2, 3, 4, 5, 6] {
            let solver = Poisson::new(1.0, 0.0, r, 1.0, 1.0, BoundaryCondition::Periodic);
            let m = DMatrix::from_diagonal(&solver.mass_diag);
            let id = DMatrix::<f64>::identity(r, r);
            let v = &solver.eigenvectors;

            // V^T M V = I  (M-orthonormality)
            let vtmv = v.transpose() * &m * v;
            let err = (&vtmv - &id).norm();
            assert!(err < ORTHO_TOL, "r={r}: V^T M V ≠ I  (err={err:.2e})");

            // V = symbol^{-1} * M * V * Λ  (generalised eigenproblem)
            let lambda = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| solver.eigenvalues[i]));
            let lhs = v.clone();
            let rhs = &solver.symbol_pseudo_inv * &m * v * &lambda;
            let rel = (&lhs - &rhs).norm() / rhs.norm();
            assert!(
                rel < RECON_TOL,
                "r={r}: V ≠ symbol^-1 · M·V·Λ  (rel={rel:.2e})"
            );
        }
    }

    #[test]
    fn test_1d_dirichlet_eigendecomposition() {
        const ORTHO_TOL: f64 = 1e-10;

        for &r in &[3, 4, 5, 6, 8] {
            let solver = Poisson::new(1.0, 0.0, r, 1.0, 1.0, BoundaryCondition::Dirichlet);
            let ndof = r - 1;
            let m = DMatrix::from_diagonal(&solver.mass_diag);
            let id = DMatrix::<f64>::identity(ndof, ndof);
            let v = &solver.eigenvectors;

            let vtmv = v.transpose() * &m * v;
            let err = (&vtmv - &id).norm();
            assert!(
                err < ORTHO_TOL,
                "Dirichlet r={r}: V^T M V ≠ I  (err={err:.2e})"
            );
        }
    }

    #[test]
    fn test_1d_neumann_eigendecomposition() {
        const ORTHO_TOL: f64 = 1e-10;

        for &r in &[2, 3, 4, 5, 6] {
            let solver = Poisson::new(1.0, 0.0, r, 1.0, 1.0, BoundaryCondition::Neumann);
            let ndof = r + 1;
            let m = DMatrix::from_diagonal(&solver.mass_diag);
            let id = DMatrix::<f64>::identity(ndof, ndof);
            let v = &solver.eigenvectors;

            let vtmv = v.transpose() * &m * v;
            let err = (&vtmv - &id).norm();
            assert!(
                err < ORTHO_TOL,
                "Neumann r={r}: V^T M V ≠ I  (err={err:.2e})"
            );
        }
    }
}
