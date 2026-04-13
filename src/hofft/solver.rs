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

use super::planner::{Extension, HofftPlanner, HofftPlannerND};
use crate::basis::*;
use rayon::prelude::*;
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
/// The DOF layout in `get_x` / `solve_in_place` is x-increasing:
/// flat index `k * r + j` corresponds to local node `j` of element `k`.
pub struct PoissonSolver1D {
    /// Polynomial order per element (number of interior+boundary GLL nodes per element).
    r: usize,

    /// HoFFT planner: handles extension packing, forward/inverse FFT, eval, and mode fill-back.
    pub planner: HofftPlanner,

    /// Pseudo-inverse of each symbol matrix `S(k) = α h M_loc + β/h K_loc(e^{iφk})`,
    /// pre-computed via eigendecomposition. Length = `n_freq`.
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

        let extension = match bc {
            BoundaryCondition::Periodic => Extension::Periodic,
            BoundaryCondition::Dirichlet => Extension::Odd,
            BoundaryCondition::Neumann => Extension::Even,
        };
        let planner = HofftPlanner::new(n, r, l, xl, extension);

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
        let mass_matrix = lobatto_basis.mass_matrix();
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

        Self {
            r,
            planner,
            symbol_pseudo_inv,
            eigenvalues,
            eigenvectors,
            periodic_mass_matrix,
            bc,
        }
    }

    /// GLL nodes in x-increasing order on `[xl, xl+l]`.
    ///
    /// - [`BoundaryCondition::Periodic`]:  `n*r` nodes — left endpoint included, right excluded.
    /// - [`BoundaryCondition::Dirichlet`]: `n*r - 1` nodes — both endpoints excluded.
    /// - [`BoundaryCondition::Neumann`]:   `n*r + 1` nodes — both endpoints included.
    pub fn get_x(&self) -> Vec<f64> {
        self.planner.get_x()
    }

    /// Solve `(α M + β K) u = f` in place.
    ///
    /// On input, `f` contains the strong-form right-hand side evaluated at
    /// the GLL nodes returned by [`PoissonSolver1D::get_x`].
    /// On output, `f` contains the FEM solution `u` at the same nodes.
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

        // Step 2: symbol solve — for each independent mode k, apply the r×r pseudo-inverse.
        let n_freq = self.symbol_pseudo_inv.len();
        let mut g_k = vec![Complex::new(0.0_f64, 0.0); self.r];
        let mut col = DVector::<Complex<f64>>::zeros(self.r);
        let mut result = DVector::<Complex<f64>>::zeros(self.r);
        for k in 0..n_freq {
            self.planner.get_values(k, f, &mut g_k);
            col.iter_mut().zip(g_k.iter()).for_each(|(c, &g)| *c = g);
            self.symbol_pseudo_inv[k].mul_to(&col, &mut result);
            self.planner.set_values(k, f, result.as_slice());
        }

        // Step 3: recover symmetric modes, inverse FFT.
        self.planner.inverse(f);
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
    /// FFT planner for the forward and inverse ND transforms, eval, and convect.
    pub fft: HofftPlannerND<N>,
    /// Diagonal of the Kronecker-product mass matrix `M^{(0)} ⊗ … ⊗ M^{(N-1)}`.
    /// Length = `r_0 * … * r_{N-1}`, first direction (x) varies fastest.
    periodic_mass_matrix: DVector<f64>,
    /// Coefficient of the mass term α (shared across directions).
    alpha: f64,
    /// Coefficient of the stiffness term β (shared across directions).
    beta: f64,
    /// Polynomial order per direction.
    rs: [usize; N],
    /// Total number of local nodes per element (`∏_d rs[d]`).
    r_total: usize,
    /// Column-major strides for the local r-block (`rs`).
    local_strides: [usize; N],
    /// Number of independent Fourier modes per direction: `n_d` (Periodic) or `n_d+1` (Odd/Even).
    n_freqs: [usize; N],
    /// Total number of independent mode tuples (`∏_d n_freqs[d]`).
    n_freq_total: usize,
    /// Column-major strides for the mode-tuple tensor (`n_freqs`).
    n_freq_strides: [usize; N],
    /// DOF index offset per direction for mass scaling: 1 for Dirichlet, 0 otherwise.
    j_offsets: [usize; N],
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
        bc: [BoundaryCondition; N],
        alpha: f64,
        beta: f64,
    ) -> Self {
        let solvers = std::array::from_fn(|d| {
            PoissonSolver1D::new(ls[d], xls[d], ns[d], rs[d], 0.0, 1.0, bc[d])
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

        let r_total = rs.iter().product();
        let local_strides = Self::compute_strides(&rs);

        let extension: [Extension; N] = std::array::from_fn(|d| match bc[d] {
            BoundaryCondition::Periodic => Extension::Periodic,
            BoundaryCondition::Dirichlet => Extension::Odd,
            BoundaryCondition::Neumann => Extension::Even,
        });

        let fft = HofftPlannerND::new(ns, rs, ls, xls, extension);

        // Number of independent Fourier modes per direction.
        let n_freqs: [usize; N] = std::array::from_fn(|d| solvers[d].symbol_pseudo_inv.len());
        let n_freq_total = n_freqs.iter().product();
        let n_freq_strides = Self::compute_strides(&n_freqs);

        // DOF index offset for mass scaling: Dirichlet DOFs start at j=1.
        let j_offsets: [usize; N] = std::array::from_fn(|d| {
            if bc[d] == BoundaryCondition::Dirichlet {
                1
            } else {
                0
            }
        });

        Self {
            solvers,
            fft,
            periodic_mass_matrix,
            alpha,
            beta,
            rs,
            r_total,
            local_strides,
            n_freqs,
            n_freq_total,
            n_freq_strides,
            j_offsets,
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
    pub fn solve_in_place(&self, f: &mut [Complex<f64>]) {
        // Step 1: mass-scale in place, then forward FFT.
        // Mass weight for DOF p in direction d: j_d = (dof_d + j_offset_d) % r_d.
        f.par_iter_mut().enumerate().for_each(|(p, fp)| {
            let lj = (0..N).fold(0, |acc, d| {
                let dof_d = (p / self.fft.strides[d]) % self.fft.ndofs[d];
                acc + ((dof_d + self.j_offsets[d]) % self.rs[d]) * self.local_strides[d]
            });
            *fp *= self.periodic_mass_matrix[lj];
        });

        self.fft.forward(f);

        // Step 2: symbol solve — iterate over all independent mode tuples,
        // apply the r_total × r_total pseudo-inverse via sum-factored eigendecomposition.
        // SAFETY: different k_flat values map to disjoint positions in f (each compact
        // entry belongs to exactly one independent mode), so concurrent reads/writes are sound.
        let ptr = f.as_mut_ptr() as usize;
        (0..self.n_freq_total).into_par_iter().for_each(|k_flat| {
            let ks: [usize; N] =
                std::array::from_fn(|d| (k_flat / self.n_freq_strides[d]) % self.n_freqs[d]);

            let mut g_k = vec![Complex::new(0.0_f64, 0.0); self.r_total];
            {
                let f_ro = unsafe {
                    std::slice::from_raw_parts(ptr as *const Complex<f64>, self.fft.total)
                };
                self.fft.get_values(&ks, f_ro, &mut g_k);
            }

            let mut v = DVector::<Complex<f64>>::from_column_slice(&g_k);

            // Apply Q†_d for each direction (sum factorisation).
            for d in 0..N {
                let r_d = self.rs[d];
                let step = self.local_strides[d] - 1;
                for s in 0..self.r_total / r_d {
                    let bj = self.flat_base_local(s, d);
                    let x = self.solvers[d].eigenvectors[ks[d]]
                        .ad_mul(&v.rows_with_step(bj, r_d, step));
                    v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                }
            }

            // Pointwise divide by ND eigenvalue λ = α + Σ_d β λ^{(d)}_{k_d, j_d}.
            const PINV_TOL: f64 = 1e-12;
            for j_flat in 0..self.r_total {
                let lambda = self.alpha
                    + (0..N)
                        .map(|d| {
                            let j_d = (j_flat / self.local_strides[d]) % self.rs[d];
                            self.beta * self.solvers[d].eigenvalues[ks[d]][j_d]
                        })
                        .sum::<f64>();
                if lambda.abs() > PINV_TOL {
                    v[j_flat] /= lambda;
                } else {
                    v[j_flat] = Complex::new(0.0, 0.0);
                }
            }

            // Apply Q_d for each direction.
            for d in 0..N {
                let r_d = self.rs[d];
                let step = self.local_strides[d] - 1;
                for s in 0..self.r_total / r_d {
                    let bj = self.flat_base_local(s, d);
                    let x =
                        &self.solvers[d].eigenvectors[ks[d]] * v.rows_with_step(bj, r_d, step);
                    v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                }
            }

            g_k.iter_mut().zip(v.iter()).for_each(|(dst, &src)| *dst = src);
            {
                let f_rw = unsafe {
                    std::slice::from_raw_parts_mut(ptr as *mut Complex<f64>, self.fft.total)
                };
                self.fft.set_values(&ks, f_rw, &g_k);
            }
        });

        // Step 3: inverse FFT.
        self.fft.inverse(f);
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
            let solver = PoissonSolverND::<1>::new(
                [1.0],
                [0.0],
                [4],
                [3],
                [BoundaryCondition::Periodic; 1],
                1.0,
                0.0,
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
            let solver = PoissonSolverND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [4; 2],
                [2; 2],
                [BoundaryCondition::Periodic; 2],
                1.0,
                0.0,
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
            let solver = PoissonSolverND::<3>::new(
                [1.0; 3],
                [0.0; 3],
                [4; 3],
                [2; 3],
                [BoundaryCondition::Periodic; 3],
                1.0,
                0.0,
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
