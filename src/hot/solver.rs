//! High-order Tensor (HoT) solver for the periodic Laplace equation.
//!
//! Solves `(α M + β K) u = f` on `[0, L]^N` with periodic boundary conditions
//! using a **single spectral element** of polynomial order *r* per direction
//! (i.e. `n = 1` everywhere).  Because there is only one element per direction
//! the only Fourier mode is `k = 0`, so all matrices are real — no FFT is needed.
//!
//! ## Algorithm
//!
//! The solve is a pure real-arithmetic tensor-product diagonalisation:
//!
//! 1. **Mass scaling** — multiply `f` by the diagonal mass matrix (Kronecker
//!    product of the 1D diagonal masses).
//! 2. **Apply Q^T_d** for each direction — transform each fiber using the
//!    transpose of the M-orthonormal eigenvector matrix.
//! 3. **Divide by eigenvalue** — pointwise divide by
//!    `α + Σ_d β λ^{(d)}_{j_d}`.
//! 4. **Apply Q_d** for each direction — back-transform.
//!
//! Steps 1 and 3 are trivially parallel; steps 2 and 4 are parallelised over
//! independent fibers using Rayon with a small amount of localized `unsafe` code.

use crate::basis::*;
use crate::utilities::{barycentric_weights, lagrangian_interpolation_all};
use nalgebra::*;
use rayon::prelude::*;

/// 1D HoT Laplace solver: a single element of polynomial order *r* on `[0, L]`.
///
/// Solves `(α M + β K) u = f` with periodic boundary conditions using a single
/// GLL spectral element.  Because `n = 1` the only Fourier mode is `k = 0`, so
/// the symbol is a real symmetric *r×r* matrix and the solve reduces to a
/// generalised eigendecomposition.
pub struct Hot1DLaplaceSolver {
    /// Domain length.
    l: f64,
    /// The domain is [xl, xl+l].
    xl: f64,
    /// Polynomial order (number of interior + left-endpoint GLL nodes).
    r: usize,
    /// GLL Basis for the single element.
    lobatto_basis: Basis,
    /// Pseudo-inverse of `(α M + β K)` for the single periodic element, size *r×r*, real symmetric pre-computed via eigendecomposition.
    symbol_pseudo_inv: DMatrix<f64>,
    /// Eigenvalues of the mass-normalised symbol `M^{-1/2} symbol M^{-1/2}`.
    eigenvalues: DVector<f64>,
    /// M-orthonormal generalised eigenvectors: `symbol V = M_diag V Λ`.
    eigenvectors: DMatrix<f64>,
    /// Diagonal of the periodic local mass matrix (length *r*).
    /// DOF 0 collects contributions from both sides of the periodic boundary.
    periodic_mass_matrix: DVector<f64>,
    /// Barycentric weights for the `r+1` GLL nodes on `[0, L]` (including `x_r = L`).
    bary_weights: Vec<f64>,
}

impl Hot1DLaplaceSolver {
    /// Construct the 1D HoT solver.
    ///
    /// - `l`: domain length.
    /// - `xl`: position of the left boundary.
    /// - `r`: polynomial order.
    /// - `alpha`, `beta`: coefficients of `α M + β K`.
    pub fn new(l: f64, xl: f64, r: usize, alpha: f64, beta: f64) -> Self {
        let lobatto_basis = Basis::new(vec![(r + 1, true)]);
        let mass_matrix = lobatto_basis.mass_matrix();
        let diff_matrix = lobatto_basis.diff_matrix(0);
        let stiffness_matrix = diff_matrix.transpose() * &mass_matrix * &diff_matrix;
        let h = l; // single element spans the whole domain

        // Periodic diagonal mass: DOF 0 accumulates contributions from both sides.
        let periodic_mass_matrix = DVector::from_fn(r, |i, _| {
            if i == 0 {
                h * 2.0 * mass_matrix[(0, 0)]
            } else {
                h * mass_matrix[(i, i)]
            }
        });

        // Symbol at k=0 (cos=1, exp=1): same formula as HoFFT with n=1.
        let mut symbol = DMatrix::<f64>::zeros(r, r);

        symbol[(0, 0)] = 2.0 * alpha * h * mass_matrix[(0, 0)]
            + 2.0 * beta * (stiffness_matrix[(0, 0)] + stiffness_matrix[(0, r)]) / h;

        for i in 1..r {
            let val = beta * (stiffness_matrix[(0, i)] + stiffness_matrix[(r, i)]) / h;
            symbol[(0, i)] = val;
            symbol[(i, 0)] = val;
        }

        for i in 1..r {
            for j in 1..r {
                symbol[(i, j)] =
                    alpha * h * mass_matrix[(i, j)] + beta * stiffness_matrix[(i, j)] / h;
            }
        }

        // Generalised eigendecomposition via mass normalisation:
        //   M^{-1/2} symbol M^{-1/2} W = W Λ   ⟹   eigenvectors V = M^{-1/2} W
        let mass_sqrt_inv = DMatrix::from_diagonal(&periodic_mass_matrix.map(|v| 1.0 / v.sqrt()));
        let eig = SymmetricEigen::new(&mass_sqrt_inv * &symbol * &mass_sqrt_inv);
        let eigenvalues = eig.eigenvalues.clone();
        let eigenvectors = &mass_sqrt_inv * eig.eigenvectors;

        // Pseudo-inverse: V diag(1/λ) V^T  (zero out near-zero eigenvalues)
        const PINV_TOL: f64 = 1e-12;
        let lambda_pinv = eigenvalues.map(|v| if v.abs() > PINV_TOL { 1.0 / v } else { 0.0 });
        let symbol_pseudo_inv =
            &eigenvectors * DMatrix::from_diagonal(&lambda_pinv) * eigenvectors.transpose();

        let bary_weights = barycentric_weights(&lobatto_basis.points[0]);

        Self {
            l,
            xl,
            r,
            lobatto_basis,
            symbol_pseudo_inv,
            eigenvalues,
            eigenvectors,
            periodic_mass_matrix,
            bary_weights,
        }
    }

    /// GLL nodes on `[xl, xl+l]`, length *r*.
    ///
    /// The *r*-th node (`x = xl+l`) is identified with `x = xl` by periodicity and
    /// is not included.
    pub fn get_x(&self) -> DVector<f64> {
        DVector::from_fn(self.r, |i, _| {
            self.l * self.lobatto_basis.points[0][i] + self.xl
        })
    }

    /// Solve `(α M + β K) u = f` in place.
    ///
    /// On input, `f` contains the strong-form RHS at the GLL nodes.
    /// On output, `f` contains the FEM solution `u`.
    pub fn solve_in_place(&self, f: &mut DVector<f64>) {
        let g = DVector::from_fn(self.r, |i, _| self.periodic_mass_matrix[i] * f[i]);
        f.copy_from(&(&self.symbol_pseudo_inv * &g));
    }

    /// Evaluate the interpolant of `u` at position `x`.
    ///
    /// `x` may lie anywhere on the real line; it is first folded into `[0, L)` by
    /// periodicity.  The degree-*r* Lagrange interpolant is evaluated using the
    /// precomputed nodes and barycentric weights; `phi[r]` (at `x_r = L`) is folded
    /// into DOF 0 since `u(L) = u(0)` by periodicity.
    pub fn eval(&self, x: f64, u: &DVector<f64>) -> f64 {
        let xi = (x - self.xl).rem_euclid(self.l) / self.l;
        let phi =
            lagrangian_interpolation_all(&self.lobatto_basis.points[0], &self.bary_weights, xi);
        let mut val = u[0] * (phi[0] + phi[self.r]);
        for i in 1..self.r {
            val += u[i] * phi[i];
        }
        val
    }
}

/// N-dimensional HoT Laplace solver: one element per direction, polynomial order *r_d*.
///
/// Extends [`Hot1DLaplaceSolver`] to `N` spatial dimensions using a tensor-product
/// eigendecomposition.  Each direction is solved with its own 1D eigenvector matrix
/// `Q_d`, and the solve is:
///
/// ```text
/// f ← M_ND · f
/// for d in 0..N: f ← (Q_d^T ⊗ I_others) · f
/// f ← diag(α + Σ_d β λ^{(d)}_{j_d})^{-1} · f
/// for d in 0..N: f ← (Q_d ⊗ I_others) · f
/// ```
pub struct HotNDLaplaceSolver<const N: usize> {
    /// One 1D solver per spatial direction.
    pub solvers: [Hot1DLaplaceSolver; N],
    /// Coefficient of the mass term α.
    pub alpha: f64,
    /// Coefficient of the stiffness term β.
    pub beta: f64,
    /// Polynomial order per direction.
    pub rs: [usize; N],
    /// Domain lengths per direction.
    pub ls: [f64; N],
    /// Left boundary positions per direction.
    pub xls: [f64; N],
    /// Total number of DOFs (`∏_d rs[d]`).
    pub r_total: usize,
    /// Column-major strides for the DOF tensor (`rs`).
    pub local_strides: [usize; N],
}

impl<const N: usize> HotNDLaplaceSolver<N> {
    /// Construct the ND HoT solver.
    ///
    /// - `ls[d]`: domain length in direction `d`.
    /// - `xls[d]`: position of the left boundary in direction `d`.
    /// - `rs[d]`: polynomial order in direction `d`.
    /// - `alpha`, `beta`: coefficients of `α M + β K`.
    pub fn new(ls: [f64; N], xls: [f64; N], rs: [usize; N], alpha: f64, beta: f64) -> Self {
        let solvers =
            std::array::from_fn(|d| Hot1DLaplaceSolver::new(ls[d], xls[d], rs[d], 0.0, 1.0));
        let r_total = rs.iter().product();
        let local_strides = Self::compute_strides(&rs);
        Self {
            solvers,
            alpha,
            beta,
            rs,
            ls,
            xls,
            r_total,
            local_strides,
        }
    }

    /// Return the tensor-product grid as a `Vec<[f64; N]>` of length `∏_d r_d`.
    ///
    /// Point `p` has coordinates `result[p][d]` for `d in 0..N`.
    /// The first direction (x) varies fastest.
    pub fn get_x(&self) -> Vec<[f64; N]> {
        let x1d: [DVector<f64>; N] = std::array::from_fn(|d| self.solvers[d].get_x());
        (0..self.r_total)
            .map(|p| std::array::from_fn(|d| x1d[d][(p / self.local_strides[d]) % self.rs[d]]))
            .collect()
    }

    /// Solve `(α M + β K) u = f` in place using the ND HoT algorithm.
    ///
    /// On input, `f` contains the strong-form RHS at the tensor-product GLL grid
    /// returned by [`HotNDLaplaceSolver::get_x`].
    /// On output, `f` contains the FEM solution `u` at the same grid.
    ///
    /// # Parallelism
    /// - Mass scaling and eigenvalue division: trivially parallel (`par_iter_mut`).
    /// - Eigenvector transforms (steps a and c): parallel over independent fibers
    ///   via Rayon; fiber disjointness is guaranteed by the tensor-product structure
    ///   of `flat_base`, enabling safe use of raw-pointer aliasing.
    pub fn solve_in_place(&self, f: &mut Vec<f64>) {
        // Step 0: multiply by diagonal ND mass matrix (trivially parallel)
        f.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, fi)| {
                *fi *= (0..N)
                    .map(|d| {
                        self.solvers[d].periodic_mass_matrix
                            [(i / self.local_strides[d]) % self.rs[d]]
                    })
                    .product::<f64>();
            });

        // Step a: apply Q^T_d for each direction.
        // For a fixed d the r_total/r_d fibers are non-overlapping → parallel with unsafe.
        for d in 0..N {
            let r_d = self.rs[d];
            let stride = self.local_strides[d];
            let eigvecs = &self.solvers[d].eigenvectors;
            // SAFETY: flat_base produces disjoint index sets for distinct s values,
            // so concurrent reads and writes never alias.
            let ptr = f.as_mut_ptr() as usize;
            (0..self.r_total / r_d).into_par_iter().for_each(|s| {
                let bj = self.flat_base(s, d);
                let p = ptr as *mut f64;
                let fiber = DVector::from_fn(r_d, |k, _| unsafe { *p.add(bj + k * stride) });
                let x = eigvecs.tr_mul(&fiber);
                for k in 0..r_d {
                    unsafe { *p.add(bj + k * stride) = x[k] };
                }
            });
        }

        // Step b: divide by ND eigenvalue (pseudo-inverse: skip near-zero modes)
        const PINV_TOL: f64 = 1e-12;
        f.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(j_flat, fj)| {
                let lambda = self.alpha
                    + (0..N)
                        .map(|d| {
                            let j_d = (j_flat / self.local_strides[d]) % self.rs[d];
                            self.beta * self.solvers[d].eigenvalues[j_d]
                        })
                        .sum::<f64>();
                if lambda.abs() > PINV_TOL {
                    *fj /= lambda;
                } else {
                    *fj = 0.0;
                }
            });

        // Step c: apply Q_d for each direction (same structure as step a)
        for d in 0..N {
            let r_d = self.rs[d];
            let stride = self.local_strides[d];
            let eigvecs = &self.solvers[d].eigenvectors;
            // SAFETY: same non-overlap argument as step a.
            let ptr = f.as_mut_ptr() as usize;
            (0..self.r_total / r_d).into_par_iter().for_each(|s| {
                let bj = self.flat_base(s, d);
                let p = ptr as *mut f64;
                let fiber = DVector::from_fn(r_d, |k, _| unsafe { *p.add(bj + k * stride) });
                let x = eigvecs * &fiber;
                for k in 0..r_d {
                    unsafe { *p.add(bj + k * stride) = x[k] };
                }
            });
        }
    }

    /// Evaluate the interpolant of `u` at a batch of positions, returning one value per point.
    ///
    /// For each point, per-direction basis values are computed with the periodic fold into DOF 0.
    /// The value is obtained by **sum factorisation**: N sequential axis contractions of total cost
    /// `r^N + r^{N-1} + …` against the contiguous `u` array (no gather step: HoT uses a single
    /// element so `u` is already a dense tensor-product array with dim 0 varying fastest).
    /// Points are processed in parallel with Rayon.
    pub fn eval(&self, xs: &[[f64; N]], u: &Vec<f64>) -> Vec<f64> {
        let u_slice = u.as_slice();

        xs.par_iter()
            .map(|x| {
                // Per-direction basis values with periodic fold into DOF 0.
                let phi: [Vec<f64>; N] = std::array::from_fn(|d| {
                    let xi = (x[d] - self.xls[d]).rem_euclid(self.ls[d]) / self.ls[d];
                    let phi_1d = lagrangian_interpolation_all(
                        &self.solvers[d].lobatto_basis.points[0],
                        &self.solvers[d].bary_weights,
                        xi,
                    );
                    let r_d = self.rs[d];
                    let mut phi_d = phi_1d[..r_d].to_vec();
                    phi_d[0] += phi_1d[r_d];
                    phi_d
                });

                // Sum-factorised eval: u has dim 0 varying fastest (stride = 1).
                // Contract dim 0 from u_slice.
                let c0 = self.rs[0];
                let rest0 = u_slice.len() / c0;
                let mut v = vec![0.0_f64; rest0];
                for s in 0..rest0 {
                    let base = s * c0;
                    for j in 0..c0 {
                        v[s] += phi[0][j] * u_slice[base + j];
                    }
                }
                // Contract remaining dims on the progressively smaller v.
                for d in 1..N {
                    let c_d = self.rs[d];
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
    // st[0] = 1 (direction 0 / x varies fastest), st[d] = st[d-1] * sizes[d-1].
    fn compute_strides(sizes: &[usize; N]) -> [usize; N] {
        let mut st = [1usize; N];
        for d in 1..N {
            st[d] = st[d - 1] * sizes[d - 1];
        }
        st
    }

    // Flat start index of the `s`-th fiber along direction `d` in the DOF tensor.
    // SAFETY invariant: for a fixed `d`, distinct values of `s` yield disjoint index sets.
    fn flat_base(&self, s: usize, d: usize) -> usize {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_hot1d_eigendecomposition() {
        const ORTHO_TOL: f64 = 1e-10;
        const RECON_TOL: f64 = 1e-10;

        for &r in &[2, 3, 4, 5, 6] {
            let solver = Hot1DLaplaceSolver::new(1.0, 0.0, r, 1.0, 1.0);
            let m = DMatrix::from_diagonal(&solver.periodic_mass_matrix);
            let id = DMatrix::<f64>::identity(r, r);
            let v = &solver.eigenvectors;

            // V^T M V = I  (M-orthonormality)
            let vtmv = v.transpose() * &m * v;
            let err = (&vtmv - &id).norm();
            assert!(err < ORTHO_TOL, "r={r}: V^T M V ≠ I  (err={err:.2e})");

            //  V = symbol^{-1} * M * V * Λ  (generalised eigenproblem)
            let lambda = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| solver.eigenvalues[i]));
            let lhs = v.clone();
            let rhs = &solver.symbol_pseudo_inv * &m * v * &lambda;
            let rel = (&lhs - &rhs).norm() / rhs.norm();
            assert!(
                rel < RECON_TOL,
                "r={r}: V ≠  symbol^-1 · M·V·Λ  (rel={rel:.2e})"
            );
        }
    }
}
