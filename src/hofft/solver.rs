//! HoFFT solver for the screened Poisson equation `(α M + β K) u = f`.
//!
//! Solves `(α M + β K) u = f` on `[0, L]^N` using a spectral-element
//! discretisation: *n* elements of polynomial order *r* per direction,
//! so each direction has `n·r` DOFs.
//!
//! Algorithm based on
//!
//! - \[1\] Caforio, F. and Imperiale, S. (2019). *High-order discrete fourier transform for the solution of the Poisson equation*. SIAM Journal of Scientific Computing, 41(5), 2747-2771
//!
//! ## Algorithm (1D outline)
//!
//! 1. **Mass scaling** — multiply the load vector `f` by the diagonal
//!    mass matrix `M` to convert from strong-form RHS to FEM RHS.
//! 2. **Forward FFT** — apply a length-*n* FFT along each direction to decouple
//!    the *n* Fourier modes.
//! 3. **Symbol solve** — for each Fourier mode `k`, apply the pseudo-inverse of
//!    the local symbol matrix `S(k)` (an *r×r* Hermitian system) via its
//!    pre-computed eigendecomposition.
//! 4. **Inverse FFT** — transform back to physical space.
//!
//! The ND solver ([`PoissonSolverND`]) extends this by applying steps 2 and 4
//! direction by direction (sum factorisation) and solving the full `r^N × r^N`
//! block-diagonal system in step 3 using tensor-product eigendecompositions.
//! Steps 2 and 4 are parallelised with Rayon over independent fibers; step 3 is
//! parallelised over independent Fourier modes.

use super::planner::{HofftPlanner, HofftPlannerND};
use crate::basis::*;
use crate::utilities::{barycentric_weights, lagrangian_interpolation_all};
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

use nalgebra::*;

/// Boundary condition type for the HoFFT solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    Periodic,
    Dirichlet,
    Neumann,
}

/// 1D Poisson solver using HoFFT:
/// - a GLL spectral-element discretisation with *n* elements of order *r*,
/// - FFT diagonalisation over the *n* element-to-element coupling modes,
/// - per-mode *r×r* Hermitian eigendecomposition for the local solve.
///
/// The DOF layout in `get_x` / `solve_in_place` is *interleaved by mode*:
/// flat index `j * n + k` corresponds to local node `j` of element `k`.
pub struct PoissonSolver1D {
    /// Domain length.
    l: f64,
    /// The domain is [xl, xl+l].
    xl: f64,
    /// Number of elements.
    n: usize,
    /// Polynomial order per element (number of interior+boundary GLL nodes per element).
    r: usize,

    /// FFT planner for the forward and inverse transforms (length *n*, *r* rows).
    planner: HofftPlanner,

    /// GLL quadrature Basis for a single element of order *r*.
    lobatto_basis: Basis,
    /// Pseudo-inverse of each symbol matrix:
    /// `symbol[k] = α h M_loc + β/h K_loc(e^{2πi k/n})` (Per-Fourier-mode symbol matrices `S(k)`, size *r×r*, Hermitian).
    /// pre-computed via eigendecomposition.
    symbol_pseudo_inv: Vec<DMatrix<Complex<f64>>>,
    /// Real eigenvalues of the mass-normalised symbol for each Fourier mode `k`.
    /// `eigenvalues[k][j]` = λ_j of `M^{-1/2} S(k) M^{-1/2}`.
    eigenvalues: Vec<DVector<f64>>,
    /// M-orthonormal eigenvectors: `S(k) V(k) = M V(k) diag(λ)`.
    /// `eigenvectors[k]` is the *r×r* complex unitary matrix.
    eigenvectors: Vec<DMatrix<Complex<f64>>>,
    /// Diagonal of the periodic local mass matrix (length *r*).
    /// DOF 0 collects contributions from both sides of the periodic boundary.
    periodic_mass_matrix: DVector<f64>,
    /// Barycentric weights for the `r+1` reference GLL nodes on `[0, 1]`.
    bary_weights: Vec<f64>,
    /// Boundary condition type.
    bc: BoundaryCondition,
}

impl PoissonSolver1D {
    /// Construct the 1D solver.
    ///
    /// - `l`: domain length.
    /// - `n`: number of elements.
    /// - `r`: polynomial order (GLL nodes per element, excluding the shared endpoint).
    /// - `alpha`, `beta`: operator coefficients for `α M + β K`.
    /// - `bc`: boundary condition type ([`BoundaryCondition::Periodic`], [`BoundaryCondition::Dirichlet`], or [`BoundaryCondition::Neumann`]).
    pub fn new(
        l: f64,
        xl: f64,
        n: usize,
        r: usize,
        alpha: f64,
        beta: f64,
        bc: BoundaryCondition,
    ) -> Self {
        let lobatto_basis = Basis::new(vec![(r + 1, true)]);

        let h = l / (n as f64);

        let planner = if bc == BoundaryCondition::Periodic {
            HofftPlanner::new(n, r, l, xl)
        } else {
            HofftPlanner::new(2 * n, r, 2.0 * l, xl)
        };

        let n_freq = if bc == BoundaryCondition::Periodic {
            n
        } else {
            n + 1
        };

        let coeff_phase = if bc == BoundaryCondition::Periodic {
            2.0
        } else {
            1.0
        };

        let mut symbol = vec![DMatrix::<Complex::<f64>>::zeros(r, r); n_freq];
        let mass_matrix = lobatto_basis.mass_matrix();
        let periodic_mass_matrix = DVector::from_fn(r, |i, _| {
            if i == 0 {
                h * 2.0 * mass_matrix[(0, 0)]
            } else {
                h * mass_matrix[(i, i)]
            }
        });
        let diff_matrix = lobatto_basis.diff_matrix(0);

        //Construct the stiffness matrix
        let stiffness_matrix: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> =
            diff_matrix.transpose() * mass_matrix.clone() * diff_matrix;

        //Compute the symbol of the operator for each frequency
        for k in 0..n_freq {
            let phase = -coeff_phase * PI * (k as f64) / (n as f64);
            let cos = phase.cos();
            let sin = phase.sin();
            let exp = cos + Complex::<f64>::I * sin;

            for i in 1..r {
                for j in 1..r {
                    symbol[k][(i, j)] = (alpha * h * mass_matrix[(i, j)]
                        + beta * stiffness_matrix[(i, j)] / h)
                        .into();
                }
            }

            symbol[k][(0, 0)] = (2.0 * alpha * h * mass_matrix[(0, 0)]
                + 2.0 * beta * stiffness_matrix[(0, 0)] / h
                + 2.0 * beta * stiffness_matrix[(0, r)] * cos / h)
                .into();

            for i in 1..r {
                symbol[k][(0, i)] =
                    beta * stiffness_matrix[(0, i)] / h + beta * stiffness_matrix[(r, i)] * exp / h;
                symbol[k][(i, 0)] = symbol[k][(0, i)].conj();
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

        // Pseudo-inverse of α M + β K via eigendecomposition:
        //   S⁺ = Q diag(1/λ if |λ| > tol else 0) Q†
        const PINV_TOL: f64 = 1e-12;
        let symbol_pseudo_inv: Vec<DMatrix<Complex<f64>>> = eigenvalues
            .iter()
            .zip(eigenvectors.iter())
            .map(|(vals, q)| {
                let lambda_pinv =
                    vals.map(|v| Complex::new(if v.abs() > PINV_TOL { 1.0 / v } else { 0.0 }, 0.0));
                q * DMatrix::from_diagonal(&lambda_pinv) * q.adjoint()
            })
            .collect();

        let bary_weights = barycentric_weights(&lobatto_basis.points(0));

        Self {
            l,
            xl,
            n,
            r,
            planner,
            lobatto_basis,
            symbol_pseudo_inv,
            eigenvalues,
            eigenvectors,
            periodic_mass_matrix,
            bary_weights,
            bc,
        }
    }

    /// GLL nodes in increasing x order.
    ///
    /// - [`BoundaryCondition::Periodic`]:  `n*r` nodes — left endpoint `xl` included,
    ///   right endpoint `xl+l` excluded (identified with `xl`).
    /// - [`BoundaryCondition::Dirichlet`]: `n*r - 1` nodes — both endpoints excluded.
    /// - [`BoundaryCondition::Neumann`]:   `n*r + 1` nodes — both endpoints included.
    pub fn get_x(&self) -> Vec<f64> {
        let h = self.l / (self.n as f64);
        let gll = &self.lobatto_basis.points(0); // length r+1, gll[0]=0, gll[r]=1
        match self.bc {
            BoundaryCondition::Periodic => {
                // Each element k contributes j=0..r-1 (right endpoint = next element's left).
                let mut x = Vec::with_capacity(self.n * self.r);
                for k in 0..self.n {
                    for j in 0..self.r {
                        x.push(self.xl + h * ((k as f64) + gll[j]));
                    }
                }
                x
            }
            BoundaryCondition::Dirichlet => {
                // Skip xl (k=0, j=0) and xl+l (k=n-1, j=r).
                let mut x = Vec::with_capacity(self.n * self.r - 1);
                for j in 1..self.r {
                    x.push(self.xl + h * gll[j]);
                }
                for k in 1..self.n {
                    for j in 0..self.r {
                        x.push(self.xl + h * ((k as f64) + gll[j]));
                    }
                }
                x
            }
            BoundaryCondition::Neumann => {
                // Element 0: all r+1 nodes; elements 1..n-1: skip shared left endpoint.
                let mut x = Vec::with_capacity(self.n * self.r + 1);
                for j in 0..=self.r {
                    x.push(self.xl + h * gll[j]);
                }
                for k in 1..self.n {
                    for j in 1..=self.r {
                        x.push(self.xl + h * ((k as f64) + gll[j]));
                    }
                }
                x
            }
        }
    }

    /// Solve `(α M + β K) u = f` in place.
    ///
    /// On input, `f` contains the strong-form right-hand side evaluated at
    /// the GLL nodes returned by [`PoissonSolver1D::get_x`].
    /// On output, `f` contains the FEM solution `u` at the same nodes.
    pub fn solve_in_place(&self, f: &mut [f64]) {
        let n_dof = if self.bc == BoundaryCondition::Periodic {
            self.n * self.r
        } else {
            2 * self.n * self.r
        };

        let mut g = DVector::<Complex<f64>>::zeros(n_dof);

        match self.bc {
            BoundaryCondition::Periodic => {
                for j in 0..self.r {
                    for k in 0..self.n {
                        g[j * self.n + k] = Complex {
                            re: self.periodic_mass_matrix[j] * f[k * self.r + j],
                            im: 0.0,
                        };
                    }
                }
            }
            BoundaryCondition::Dirichlet => {
                // Odd extension on 2n elements: f[-1] = 0 (left BC), f[n*r] = 0 (right BC).
                // First half  k ∈ [0, n): xl excluded → x-inc index = k*r + j - 1,
                //             (k=0, j=0) is the left boundary → 0.
                // Second half k ∈ [n, 2n): odd mirror → -f[(2n-k)*r - j - 1],
                //             (k=n, j=0) is the right boundary → 0.
                for j in 0..self.r {
                    for k in 0..2 * self.n {
                        g[j * (2 * self.n) + k] = Complex {
                            re: if k < self.n {
                                if k == 0 && j == 0 {
                                    0.0
                                } else {
                                    self.periodic_mass_matrix[j] * f[k * self.r + j - 1]
                                }
                            } else {
                                if k == self.n && j == 0 {
                                    0.0
                                } else {
                                    -self.periodic_mass_matrix[j]
                                        * f[(2 * self.n - k) * self.r - j - 1]
                                }
                            },
                            im: 0.0,
                        };
                    }
                }
            }
            BoundaryCondition::Neumann => {
                // Even extension on 2n elements: f'(0) = 0, f'(l) = 0.
                // First half  k ∈ [0, n): xl included → x-inc index = k*r + j.
                // Second half k ∈ [n, 2n): even mirror → +f[(2n-k)*r - j].
                for j in 0..self.r {
                    for k in 0..2 * self.n {
                        g[j * (2 * self.n) + k] = Complex {
                            re: if k < self.n {
                                self.periodic_mass_matrix[j] * f[k * self.r + j]
                            } else {
                                self.periodic_mass_matrix[j] * f[(2 * self.n - k) * self.r - j]
                            },
                            im: 0.0,
                        };
                    }
                }
            }
        };

        self.planner.forward(g.as_mut_slice());

        let n_elem = if self.bc == BoundaryCondition::Periodic {
            self.n
        } else {
            2 * self.n
        };

        for k in 0..self.symbol_pseudo_inv.len() {
            let x = &self.symbol_pseudo_inv[k] * g.rows_with_step(k, self.r, n_elem - 1);
            g.rows_with_step_mut(k, self.r, n_elem - 1)
                .set_column(0, &x);
        }

        // For Dirichlet/Neumann: recover frequencies k ∈ {n+1,...,2n-1}
        // from the already-solved k ∈ {1,...,n-1} via Lemma 3.2 (Caforio & Imperiale 2019).
        //   j = 0:   û_{2N-k, 0}   = ±û_{k, 0}
        //   j > 0:   û_{2N-k, j}   = ±e^{-iπk/N} û_{k, R-j}
        // where + = Neumann, − = Dirichlet.
        if self.bc != BoundaryCondition::Periodic {
            let sign = if self.bc == BoundaryCondition::Dirichlet {
                -1.0_f64
            } else {
                1.0_f64
            };
            for k in 1..self.n {
                let km = 2 * self.n - k;
                let phase = Complex::new(0.0_f64, -PI * (k as f64) / (self.n as f64)).exp();
                // j = 0
                g[km] = sign * g[k];
                // j in 1..r
                for j in 1..self.r {
                    g[j * (2 * self.n) + km] = sign * phase * g[(self.r - j) * (2 * self.n) + k];
                }
            }
        }

        self.planner.inverse(g.as_mut_slice());

        match self.bc {
            BoundaryCondition::Periodic => {
                // Inverse of: g[j * n + k] ← f[k * r + j]
                for j in 0..self.r {
                    for k in 0..self.n {
                        f[k * self.r + j] = g[j * self.n + k].re;
                    }
                }
            }
            BoundaryCondition::Dirichlet => {
                // Inverse of first half only: g[j * 2n + k] ← f[k*r + j - 1]  (k<n, skip k=0,j=0)
                for j in 0..self.r {
                    for k in 0..self.n {
                        if k == 0 && j == 0 {
                            continue;
                        }
                        f[k * self.r + j - 1] = g[j * (2 * self.n) + k].re;
                    }
                }
            }
            BoundaryCondition::Neumann => {
                // Inverse of first half: g[j * 2n + k] ← f[k*r + j]  (k<n)
                for j in 0..self.r {
                    for k in 0..self.n {
                        f[k * self.r + j] = g[j * (2 * self.n) + k].re;
                    }
                }
                // Right endpoint: mirror at k=n, j=0 → HoFFT index 0*(2*n) + n = n
                f[self.n * self.r] = g[self.n].re;
            }
        }
    }

    /// Evaluate the interpolant of `u` at position `x`.
    ///
    /// `x` is first folded into `[xl, xl+l)` by periodicity, then the containing
    /// element `k` is located in O(1).  Only the `r+1` basis functions of that
    /// element are evaluated (O(r)); the shared right endpoint (local node `r`)
    /// is DOF 0 of element `(k+1) % n`.
    pub fn eval(&self, x: f64, u: &[f64]) -> f64 {
        let h = self.l / (self.n as f64);
        let y = (x - self.xl).rem_euclid(self.l) + self.xl;
        let k = (((y - self.xl) / h) as usize).min(self.n - 1);
        let xi = (y - self.xl - (k as f64) * h) / h;

        let phi =
            lagrangian_interpolation_all(&self.lobatto_basis.points(0), &self.bary_weights, xi);

        let mut val = 0.0_f64;
        for j in 0..self.r {
            val += u[j * self.n + k] * phi[j];
        }
        val += u[(k + 1) % self.n] * phi[self.r];
        val
    }

    /// Apply a flow map `lambda` to every grid node and interpolate `u` at the
    /// resulting positions.
    ///
    /// `u` must be in the same layout as the output of [`PoissonSolver1D::solve_in_place`]
    /// (i.e. the DOF vector indexed by [`PoissonSolver1D::get_x`]).
    /// `lambda` maps a physical position to its image under the flow.
    pub fn convect<F>(&self, lambda: F, u: &[f64]) -> Vec<f64>
    where
        F: Fn(f64) -> f64,
    {
        let xs = self.get_x();
        xs.iter().map(|&x| self.eval(lambda(x), u)).collect()
    }
}

/// N-dimensional HoFFT screened Poisson solver.
///
/// Extends [`PoissonSolver1D`] to `N` spatial dimensions using sum
/// factorisation: FFTs are applied direction by direction, and the symbol
/// pseudo-inverse is applied to each `(r_0 × … × r_{N-1})`-dimensional block
/// independently for each Fourier mode combination.
///
/// `N` is a compile-time constant (const generic), so the compiler can
/// unroll the per-direction loops.
pub struct PoissonSolverND<const N: usize> {
    /// One 1D solver per spatial direction.
    solvers: [PoissonSolver1D; N],
    /// FFT planner for the forward and inverse ND transforms.
    fft: HofftPlannerND<N>,
    /// Diagonal of the Kronecker-product mass matrix `M^{(0)} ⊗ … ⊗ M^{(N-1)}`.
    /// Length = `r_0 * … * r_{N-1}`, first direction (x) varies fastest.
    periodic_mass_matrix: DVector<f64>,
    /// Coefficient of the mass term α (shared across directions).
    alpha: f64,
    /// Coefficient of the stiffness term β (shared across directions).
    beta: f64,
    /// Number of elements per direction.
    ns: [usize; N],
    /// Polynomial order per direction.
    rs: [usize; N],
    /// Domain lengths per direction.
    ls: [f64; N],
    /// Left boundary positions per direction.
    xls: [f64; N],
    /// Element sizes per direction (`ls[d] / ns[d]`).
    hs: [f64; N],
    /// DOF counts per direction (`ns[d] * rs[d]`).
    ndofs: [usize; N],
    /// Total number of DOFs (`∏_d ndofs[d]`).
    total: usize,
    /// Total number of local nodes per element (`∏_d rs[d]`).
    r_total: usize,
    /// Total number of elements (`∏_d ns[d]`).
    n_total: usize,
    /// Column-major strides for the DOF tensor (`ndofs`).
    strides: [usize; N],
    /// Column-major strides for the local r-block (`rs`).
    local_strides: [usize; N],
    /// Column-major strides for the element grid (`ns`).
    n_strides: [usize; N],
    /// Number of interpolation nodes per direction (`rs[d] + 1`).
    counts: [usize; N],
    /// Column-major strides for the interpolation-node tensor (`counts`).
    count_strides: [usize; N],
    /// Total number of interpolation nodes per element patch (`∏_d counts[d]`).
    local_total: usize,
}

impl<const N: usize> PoissonSolverND<N> {
    /// Construct the ND solver from per-direction parameters.
    ///
    /// - `ls[d]`: domain length in direction `d`.
    /// - `xls[d]`: position of the left boundary in direction `d`.
    /// - `ns[d]`: number of elements in direction `d`.
    /// - `rs[d]`: polynomial order in direction `d`.
    /// - `alpha`, `beta`: coefficients of `α M + β K`.
    pub fn new(
        ls: [f64; N],
        xls: [f64; N],
        ns: [usize; N],
        rs: [usize; N],
        alpha: f64,
        beta: f64,
    ) -> Self {
        let solvers = std::array::from_fn(|d| {
            PoissonSolver1D::new(
                ls[d],
                xls[d],
                ns[d],
                rs[d],
                0.0,
                1.0,
                BoundaryCondition::Periodic,
            )
        });

        // Build the diagonal of M^{(0)} ⊗ … ⊗ M^{(N-1)} iteratively.
        // Each 1D mass matrix is diagonal (GLL quadrature), so the Kronecker product
        // is also diagonal: entry (i_0,…,i_{N-1}) = ∏_d M^{(d)}[i_d, i_d].
        let periodic_mass_matrix =
            (1..N).fold(solvers[0].periodic_mass_matrix.clone(), |acc, d| {
                let m_d = &solvers[d].periodic_mass_matrix;
                DVector::from_iterator(
                    acc.len() * m_d.len(),
                    m_d.iter().flat_map(|&w| acc.iter().map(move |&v| v * w)),
                )
            });

        let ns: [usize; N] = std::array::from_fn(|d| solvers[d].n);
        let rs: [usize; N] = std::array::from_fn(|d| solvers[d].r);
        let ls: [f64; N] = std::array::from_fn(|d| solvers[d].l);
        let ndofs: [usize; N] = std::array::from_fn(|d| ns[d] * rs[d]);
        let total = ndofs.iter().product();
        let r_total = rs.iter().product();
        let n_total = ns.iter().product();
        let strides = Self::compute_strides(&ndofs);
        let local_strides = Self::compute_strides(&rs);
        let n_strides = Self::compute_strides(&ns);
        let counts: [usize; N] = std::array::from_fn(|d| rs[d] + 1);
        let count_strides = Self::compute_strides(&counts);
        let local_total = counts.iter().product();
        let hs: [f64; N] = std::array::from_fn(|d| ls[d] / (ns[d] as f64));
        let fft = HofftPlannerND::new(ns, rs, ls, xls);

        Self {
            solvers,
            fft,
            periodic_mass_matrix,
            alpha,
            beta,
            ns,
            rs,
            ls,
            xls,
            hs,
            ndofs,
            total,
            r_total,
            n_total,
            strides,
            local_strides,
            n_strides,
            counts,
            count_strides,
            local_total,
        }
    }

    /// Return the tensor-product grid as a `Vec<[f64; N]>` of length
    /// `total_ndof = ∏_d (n_d * r_d)`.
    ///
    /// Point `p` has coordinates `result[p][d]` for `d in 0..N`.
    /// The first direction (x) varies fastest.
    pub fn get_x(&self) -> Vec<[f64; N]> {
        self.fft.get_x()
    }

    /// Solve `(α M + β K) u = f` in place using the ND HoFFT algorithm.
    ///
    /// On input, `f` contains the strong-form RHS at the tensor-product GLL grid
    /// returned by [`PoissonSolverND::get_x`].
    /// On output, `f` contains the FEM solution `u` at the same grid.
    ///
    pub fn solve_in_place(&self, f: &mut [f64]) {
        // Build g with mass scaling (parallel, no unsafe)
        let mut g: Vec<Complex<f64>> = f
            .par_iter()
            .enumerate()
            .map(|(p, &fp)| {
                let lj = (0..N).fold(0, |acc, d| {
                    acc + ((p / self.strides[d]) % self.ndofs[d] / self.ns[d])
                        * self.local_strides[d]
                });
                Complex::new(self.periodic_mass_matrix[lj] * fp, 0.0)
            })
            .collect();

        self.fft.forward(&mut g);

        // Apply ND symbol pseudo-inverse, parallel over Fourier modes.
        // SAFETY: different k_flat map to disjoint cells in g —
        //   index = Σ_d [k_d + j_d·n_d]·strides[d] is injective in (k_flat, j_flat),
        //   so sets for distinct k_flat are disjoint.
        const PINV_TOL: f64 = 1e-12;
        {
            let ptr = g.as_mut_ptr() as usize;
            (0..self.n_total).into_par_iter().for_each(|k_flat| {
                let base = (0..N).fold(0, |acc, d| {
                    acc + (k_flat / self.n_strides[d]) % self.ns[d] * self.strides[d]
                });
                let p = ptr as *mut Complex<f64>;

                // Gather the r_total values belonging to this Fourier mode
                let mut v = DVector::<Complex<f64>>::from_fn(self.r_total, |j_flat, _| {
                    let off = (0..N).fold(0, |acc, d| {
                        acc + (j_flat / self.local_strides[d]) % self.rs[d]
                            * self.ns[d]
                            * self.strides[d]
                    });
                    unsafe { *p.add(base + off) }
                });

                // Step a: apply Q†_d for each direction
                for d in 0..N {
                    let k_d = (k_flat / self.n_strides[d]) % self.ns[d];
                    let r_d = self.rs[d];
                    let step = self.local_strides[d] - 1;
                    for s in 0..self.r_total / r_d {
                        let bj = self.flat_base_local(s, d);
                        let x = self.solvers[d].eigenvectors[k_d]
                            .ad_mul(&v.rows_with_step(bj, r_d, step));
                        v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                    }
                }

                // Step b: pointwise divide by ND eigenvalue λ = α + Σ_d β λ^{(d)}_{k_d, j_d}
                for j_flat in 0..self.r_total {
                    let lambda = self.alpha
                        + (0..N)
                            .map(|d| {
                                let k_d = (k_flat / self.n_strides[d]) % self.ns[d];
                                let j_d = (j_flat / self.local_strides[d]) % self.rs[d];
                                self.beta * self.solvers[d].eigenvalues[k_d][j_d]
                            })
                            .sum::<f64>();
                    if lambda.abs() > PINV_TOL {
                        v[j_flat] /= lambda;
                    } else {
                        v[j_flat] = Complex::new(0.0, 0.0);
                    }
                }

                // Step c: apply Q_d for each direction
                for d in 0..N {
                    let k_d = (k_flat / self.n_strides[d]) % self.ns[d];
                    let r_d = self.rs[d];
                    let step = self.local_strides[d] - 1;
                    for s in 0..self.r_total / r_d {
                        let bj = self.flat_base_local(s, d);
                        let x =
                            &self.solvers[d].eigenvectors[k_d] * v.rows_with_step(bj, r_d, step);
                        v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                    }
                }

                // Scatter back
                for j_flat in 0..self.r_total {
                    let off = (0..N).fold(0, |acc, d| {
                        acc + (j_flat / self.local_strides[d]) % self.rs[d]
                            * self.ns[d]
                            * self.strides[d]
                    });
                    unsafe { *p.add(base + off) = v[j_flat] };
                }
            });
        }

        self.fft.inverse(&mut g);

        for i in 0..self.total {
            f[i] = g[i].re;
        }
    }

    /// Evaluate the interpolant of `u` at a batch of positions, returning one value per point.
    ///
    /// ## Optimisations applied
    ///
    /// 1. **Sum factorisation**: after gathering, each point is evaluated by N sequential
    ///    axis contractions of sizes `(r+1)^N, (r+1)^{N-1}, …` — O((r+1)^N / r) instead of the
    ///    naive N·(r+1)^N.
    /// 2. **Shared gather**: points sharing the same element tuple `(k_0,…,k_{N-1})`
    ///    read the `(r_0+1)·…·(r_{N-1}+1)` scattered DOF values into a contiguous array **once**,
    ///    then all such points evaluate from the cache-hot local patch.
    /// 3. **Rayon**: element groups are processed in parallel.
    pub fn eval(&self, xs: &[[f64; N]], u: &[f64]) -> Vec<f64> {
        // Step 1: compute element index for each point (parallel).
        let k: Vec<usize> = xs.par_iter().map(|x| self.element_index(x)).collect();

        // Step 2: group point indices by element index.
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &k_flat) in k.iter().enumerate() {
            groups.entry(k_flat).or_default().push(i);
        }

        // Step 3: parallel over each element — gather once, sum-factorise per point.
        // Reference coordinates and basis values are computed for each point inside the group.
        let partial: Vec<Vec<(usize, f64)>> = groups
            .into_par_iter()
            .filter_map(|(k_flat, indices)| {
                let ks: [usize; N] =
                    std::array::from_fn(|d| (k_flat / self.n_strides[d]) % self.ns[d]);

                // Opt 3: gather u into a contiguous local patch (once per element group).
                let u_local: Vec<f64> = (0..self.local_total)
                    .map(|p| u[self.flat_dof(p, &ks)])
                    .collect();

                // Opt 2: sum-factorised eval for each point in this element.
                let result: Vec<(usize, f64)> = indices
                    .iter()
                    .map(|&i| {
                        let phi: [Vec<f64>; N] = std::array::from_fn(|d| {
                            let y = (xs[i][d] - self.xls[d]).rem_euclid(self.ls[d]) + self.xls[d];
                            let xi = (y - self.xls[d] - (ks[d] as f64) * self.hs[d]) / self.hs[d];
                            lagrangian_interpolation_all(
                                &self.solvers[d].lobatto_basis.points(0),
                                &self.solvers[d].bary_weights,
                                xi,
                            )
                        });
                        // Contract dim 0 from u_local (contiguous reads).
                        let c0 = self.counts[0];
                        let rest0 = self.local_total / c0;
                        let mut v = vec![0.0_f64; rest0];
                        for s in 0..rest0 {
                            let base = s * c0;
                            for j in 0..c0 {
                                v[s] += phi[0][j] * u_local[base + j];
                            }
                        }
                        // Contract remaining dims on the progressively smaller v.
                        for d in 1..N {
                            let c_d = self.counts[d];
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
                        (i, v[0])
                    })
                    .collect();
                Some(result)
            })
            .collect();

        // Assemble output in original point order.
        let mut out = vec![0.0_f64; xs.len()];
        for group in partial {
            for (i, val) in group {
                out[i] = val;
            }
        }
        out
    }

    /// Evaluate `u` at the convected (shifted) DOF nodes.
    ///
    /// - `lambda`: flow map `[f64; N] -> [f64; N]` — takes a grid node and returns
    ///   the shifted position at which `u` is sampled.
    /// - `u`: the solution vector on the tensor-product GLL grid.
    ///
    /// Returns a `Vec<f64>` of length equal to the number of DOFs: entry `p`
    /// is `u(lambda(x_p))`, i.e. the interpolated value of `u` at the image
    /// of the `p`-th DOF node under the flow map.
    pub fn convect<F>(&self, lambda: F, u: &[f64]) -> Vec<f64>
    where
        F: Fn([f64; N]) -> [f64; N] + Sync,
    {
        // Build the full tensor-product grid as a Vec<[f64; N]>.
        let xs = self.get_x();

        // Apply the flow map to every node, then interpolate u at those positions.
        let shifted: Vec<[f64; N]> = xs.par_iter().map(|&pt| lambda(pt)).collect();
        self.eval(&shifted, u)
    }

    // Flat start index of the `s`-th fiber along direction `d` in the local r-block.
    fn flat_base_local(&self, s: usize, d: usize) -> usize {
        let mut base = 0usize;
        let mut rem = s;
        for d2 in 0..N {
            if d2 == d {
                continue;
            }
            base += (rem % self.rs[d2]) * self.local_strides[d2];
            rem /= self.rs[d2];
        }
        base
    }

    // Column-major strides for a tensor-product array of shape `sizes`.
    // st[0] = 1 (direction 0 / x varies fastest), st[d] = st[d-1] * sizes[d-1].
    fn compute_strides(sizes: &[usize; N]) -> [usize; N] {
        let mut st = [1usize; N];
        for d in 1..N {
            st[d] = st[d - 1] * sizes[d - 1];
        }
        st
    }

    // Global flat DOF index for local patch node `p` in element `ks`.
    // `p` ranges over `0..local_total`; the last local node in each direction
    // (j == rs[d]) wraps to the next element via periodicity.
    fn flat_dof(&self, p: usize, ks: &[usize; N]) -> usize {
        (0..N).fold(0, |acc, d| {
            let j = (p / self.count_strides[d]) % self.counts[d];
            let raw = if j < self.rs[d] {
                j * self.ns[d] + ks[d]
            } else {
                (ks[d] + 1) % self.ns[d]
            };
            acc + raw * self.strides[d]
        })
    }

    // Returns the flat element index for a point `x` in the tensor-product mesh.
    fn element_index(&self, x: &[f64; N]) -> usize {
        (0..N).fold(0, |acc, d| {
            let y = (x[d] - self.xls[d]).rem_euclid(self.ls[d]);
            let k_d = ((y / self.hs[d]) as usize).min(self.ns[d] - 1);
            acc + k_d * self.n_strides[d]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Complex;

    #[test]
    fn test_symbol_eigendecomposition() {
        const ORTHO_TOL: f64 = 1e-10;
        const RECON_TOL: f64 = 1e-8;

        for &r in &[1, 2, 3, 4, 5, 6] {
            for &n in &[4, 16, 32, 64, 128] {
                let solver =
                    PoissonSolver1D::new(1.0, 0.0, n, r, 1.0, 1.0, BoundaryCondition::Periodic);
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

                    // 3.  V =symbol^-1 * M * V * Λ
                    let lambda_diag = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                        Complex::new(solver.eigenvalues[k][i], 0.0)
                    }));
                    let lhs = v.clone();
                    let rhs = &solver.symbol_pseudo_inv[k] * &m_c * v * &lambda_diag;
                    let recon_err = (&lhs - &rhs).norm() / rhs.norm();
                    assert!(
                        recon_err < RECON_TOL,
                        "r={r} n={n} k={k}: V ≠  symbol^-1 · M·V·Λ (relative err={recon_err:.2e})"
                    );
                }
            }
        }
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
            let solver = PoissonSolverND::<1>::new([1.0], [0.0], [4], [3], 1.0, 0.0);
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
            let solver = PoissonSolverND::<2>::new([1.0; 2], [0.0; 2], [4; 2], [2; 2], 1.0, 0.0);
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
            let solver = PoissonSolverND::<3>::new([1.0; 3], [0.0; 3], [4; 3], [2; 3], 1.0, 0.0);
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
