use crate::calculator::*;
use nalgebra::*;
use rayon::prelude::*;

fn compute_strides<const N: usize>(sizes: &[usize; N]) -> [usize; N] {
    let mut st = [1usize; N];
    for d in (1..N).rev() {
        st[d - 1] = st[d] * sizes[d];
    }
    st
}

fn flat_base<const N: usize>(
    s: usize,
    d: usize,
    sizes: &[usize; N],
    strides: &[usize; N],
) -> usize {
    let mut base = 0usize;
    let mut rem = s;
    for d2 in (0..N).rev() {
        if d2 == d {
            continue;
        }
        base += (rem % sizes[d2]) * strides[d2];
        rem /= sizes[d2];
    }
    base
}

/// High-order Tensor 1D Laplace solver (n=1, polynomial order r).
///
/// Solves (α M + β K) u = f on [0, L] with periodic BCs using a single
/// spectral element. Because n=1 the only Fourier mode is k=0, so all
/// matrices are real — no FFT is needed.
pub struct Hot1DLaplaceSolver {
    pub l: f64,
    pub r: usize,
    pub lobatto_calculator: Calculator,
    /// (α M + β K) for the single periodic element, r×r real symmetric.
    pub symbol: DMatrix<f64>,
    pub symbol_pseudo_inv: DMatrix<f64>,
    /// Eigenvalues of the M-normalised symbol (M^{-1/2} symbol M^{-1/2}).
    pub eigenvalues: DVector<f64>,
    /// M-orthonormal generalised eigenvectors: symbol V = M_diag V Λ.
    pub eigenvectors: DMatrix<f64>,
    /// Diagonal of the periodic local mass matrix (length r).
    pub periodic_mass_matrix: DVector<f64>,
    pub alpha: f64,
    pub beta: f64,
}

impl Hot1DLaplaceSolver {
    pub fn new(l: f64, r: usize, alpha: f64, beta: f64) -> Self {
        let lobatto_calculator = Calculator::new(vec![(r + 1, true)]);
        let mass_matrix = lobatto_calculator.mass_matrix();
        let diff_matrix = lobatto_calculator.diff_matrix(0);
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

        // Generalised eigendecomposition via mass-normalisation:
        //   (M^{-1/2} symbol M^{-1/2}) W = W Λ  →  eigenvectors V = M^{-1/2} W
        let mass_sqrt_inv = DMatrix::from_diagonal(&periodic_mass_matrix.map(|v| 1.0 / v.sqrt()));
        let eig = SymmetricEigen::new(&mass_sqrt_inv * &symbol * &mass_sqrt_inv);
        let eigenvalues = eig.eigenvalues.clone();
        let eigenvectors = &mass_sqrt_inv * eig.eigenvectors;

        // Pseudo-inverse: V diag(1/λ) V^T  (zero out near-zero eigenvalues)
        const PINV_TOL: f64 = 1e-12;
        let lambda_pinv = eigenvalues.map(|v| if v.abs() > PINV_TOL { 1.0 / v } else { 0.0 });
        let symbol_pseudo_inv =
            &eigenvectors * DMatrix::from_diagonal(&lambda_pinv) * eigenvectors.transpose();

        Self {
            l,
            r,
            lobatto_calculator,
            symbol,
            symbol_pseudo_inv,
            eigenvalues,
            eigenvectors,
            periodic_mass_matrix,
            alpha,
            beta,
        }
    }

    /// GLL nodes on [0, L].  The r-th node (x=L) is identified with x=0 by periodicity.
    pub fn get_x(&self) -> DVector<f64> {
        DVector::from_fn(self.r, |i, _| self.l * self.lobatto_calculator.points[0][i])
    }

    /// Solve (α M + β K) u = f in place.
    /// Input:  strong-form RHS evaluated at GLL nodes.
    /// Output: FEM solution u at the same nodes.
    pub fn solve_in_place(&self, f: &mut DVector<f64>) {
        let g = DVector::from_fn(self.r, |i, _| self.periodic_mass_matrix[i] * f[i]);
        f.copy_from(&(&self.symbol_pseudo_inv * &g));
    }
}

/// N-dimensional HOT Laplace solver (one element per direction, order r_d).
pub struct HotNDLaplaceSolver<const N: usize> {
    pub solvers: [Hot1DLaplaceSolver; N],
    pub alpha: f64,
    pub beta: f64,
}

impl<const N: usize> HotNDLaplaceSolver<N> {
    /// - `ls[d]`: domain length in direction d
    /// - `rs[d]`: polynomial order in direction d
    pub fn new(ls: [f64; N], rs: [usize; N], alpha: f64, beta: f64) -> Self {
        let solvers = std::array::from_fn(|d| Hot1DLaplaceSolver::new(ls[d], rs[d], 0.0, 1.0));
        Self {
            solvers,
            alpha,
            beta,
        }
    }

    /// Tensor-product grid: N coordinate vectors, each of length ∏_d r_d.
    pub fn get_x(&self) -> [DVector<f64>; N] {
        let x1d: [DVector<f64>; N] = std::array::from_fn(|d| self.solvers[d].get_x());
        let ndofs: [usize; N] = std::array::from_fn(|d| x1d[d].len());
        let total: usize = ndofs.iter().product();
        let strides = compute_strides(&ndofs);

        std::array::from_fn(|d| DVector::from_fn(total, |p, _| x1d[d][(p / strides[d]) % ndofs[d]]))
    }

    pub fn solve_in_place(&self, f: &mut DVector<f64>) {
        let rs: [usize; N] = std::array::from_fn(|d| self.solvers[d].r);
        let r_total: usize = rs.iter().product();
        let local_strides = compute_strides(&rs);

        // Step 0: multiply by diagonal ND mass matrix (trivially parallel)
        f.as_mut_slice().par_iter_mut().enumerate().for_each(|(i, fi)| {
            *fi *= (0..N)
                .map(|d| self.solvers[d].periodic_mass_matrix[(i / local_strides[d]) % rs[d]])
                .product::<f64>();
        });

        // Step a: apply Q^T_d for each direction.
        // For a fixed d the r_total/r_d fibers are non-overlapping → parallel with unsafe.
        for d in 0..N {
            let r_d = rs[d];
            let stride = local_strides[d];
            let eigvecs = &self.solvers[d].eigenvectors;
            // SAFETY: flat_base produces disjoint index sets for distinct s values,
            // so concurrent reads and writes never alias.
            let ptr = f.as_mut_ptr() as usize;
            (0..r_total / r_d).into_par_iter().for_each(|s| {
                let bj = flat_base(s, d, &rs, &local_strides);
                let p = ptr as *mut f64;
                let fiber = DVector::from_fn(r_d, |k, _| unsafe { *p.add(bj + k * stride) });
                let x = eigvecs.tr_mul(&fiber);
                for k in 0..r_d {
                    unsafe { *p.add(bj + k * stride) = x[k] };
                }
            });
        }

        // Step b: divide by ND eigenvalue (trivially parallel)
        f.as_mut_slice().par_iter_mut().enumerate().for_each(|(j_flat, fj)| {
            let lambda = self.alpha
                + (0..N)
                    .map(|d| {
                        let j_d = (j_flat / local_strides[d]) % rs[d];
                        self.beta * self.solvers[d].eigenvalues[j_d]
                    })
                    .sum::<f64>();
            *fj /= lambda;
        });

        // Step c: apply Q_d for each direction (same structure as step a)
        for d in 0..N {
            let r_d = rs[d];
            let stride = local_strides[d];
            let eigvecs = &self.solvers[d].eigenvectors;
            // SAFETY: same non-overlap argument as step a.
            let ptr = f.as_mut_ptr() as usize;
            (0..r_total / r_d).into_par_iter().for_each(|s| {
                let bj = flat_base(s, d, &rs, &local_strides);
                let p = ptr as *mut f64;
                let fiber = DVector::from_fn(r_d, |k, _| unsafe { *p.add(bj + k * stride) });
                let x = eigvecs * &fiber;
                for k in 0..r_d {
                    unsafe { *p.add(bj + k * stride) = x[k] };
                }
            });
        }
    }
}
