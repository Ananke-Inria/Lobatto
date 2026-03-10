use crate::calculator::*;
use std::f64::consts::PI;
use std::sync::Arc;

use nalgebra::*;
use rustfft::*;

pub struct HoFFT1DLaplaceSolver {
    pub l: f64,   //Length of the domain
    pub n: usize, // Number of elements in the domain
    pub r: usize, //polynomial order of the local approximation method (greater than one)

    pub fw_fft: Arc<dyn Fft<f64>>, //FFT engine
    pub i_fft: Arc<dyn Fft<f64>>,  //IFFT engine

    pub lobatto_calculator: Calculator,
    pub symbol: Vec<DMatrix<Complex<f64>>>,
    pub symbol_pseudo_inv: Vec<DMatrix<Complex<f64>>>,
    // Eigendecomposition of each symbol[k] (valid for Hermitian positive definite symbols):
    // symbol[k] = eigenvectors[k] * diag(eigenvalues[k]) * eigenvectors[k]†
    pub eigenvalues: Vec<DVector<f64>>,
    pub eigenvectors: Vec<DMatrix<Complex<f64>>>,
    pub periodic_mass_matrix: DVector<f64>,
    pub alpha: f64,
    pub beta: f64,
}

impl HoFFT1DLaplaceSolver {
    pub fn new(l: f64, n: usize, r: usize, alpha: f64, beta: f64) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fw_fft = planner.plan_fft_forward(n);
        let i_fft = planner.plan_fft_inverse(n);

        let lobatto_calculator = Calculator::new(vec![(r + 1, true)]);

        let mut symbol = vec![DMatrix::<Complex::<f64>>::zeros(r, r); n];
        let mass_matrix = lobatto_calculator.mass_matrix();
        let periodic_mass_matrix = DVector::from_fn(r, |i, _| {
            let h = l / (n as f64);
            if i == 0 {
                h * 2.0 * mass_matrix[(0, 0)]
            } else {
                h * mass_matrix[(i, i)]
            }
        });
        let diff_matrix = lobatto_calculator.diff_matrix(0);

        //Construct the stiffness matrix
        let stiffness_matrix: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> =
            diff_matrix.transpose() * mass_matrix.clone() * diff_matrix;

        //Conpute the symbol of the operator for each frequency
        for k in 0..n {
            let h = l / (n as f64);
            let phase = -2.0 * PI * (k as f64) / (n as f64);
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

        //Eigenvectors and eigenvalues of alpha I + beta M^{-1/2} K M^{-1/2}
        let eigs: Vec<SymmetricEigen<Complex<f64>, Dyn>> = symbol
            .iter()
            .map(|s| SymmetricEigen::new(&mass_sqrt_inv_c * s * &mass_sqrt_inv_c))
            .collect();
        let eigenvalues: Vec<DVector<f64>> = eigs.iter().map(|e| e.eigenvalues.clone()).collect();

        //Eigenvectors of alpha I + beta M^{-1} K
        let eigenvectors: Vec<DMatrix<Complex<f64>>> = eigs
            .iter()
            .map(|e| &mass_sqrt_inv_c * e.eigenvectors.clone())
            .collect();

        // Pseudoinverse of  alpha M + beta K via eigendecomposition: S^+ = Q * diag(1/λ if |λ|>tol else 0) * Q†
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
            l,
            n,
            r,
            fw_fft,
            i_fft,
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

    pub fn get_x(&self) -> DVector<f64> {
        let h = self.l / (self.n as f64);
        let mut x = DVector::<f64>::zeros(self.n * self.r);
        for k in 0..self.n {
            for j in 0..self.r {
                x[j * self.n + k] = h * ((k as f64) + self.lobatto_calculator.points[0][j]);
            }
        }
        x
    }

    pub fn solve_in_place(&self, f: &mut DVector<f64>) {
        let norm = self.n as f64;

        if self.r == 1 {
            let scale = self.periodic_mass_matrix[0];
            let mut g: Vec<Complex<f64>> =
                f.iter().map(|&v| Complex::new(scale * v, 0.0)).collect();

            self.fw_fft.process(&mut g);

            for k in 0..self.n {
                g[k] *= self.symbol_pseudo_inv[k][(0, 0)];
            }

            self.i_fft.process(&mut g);

            for i in 0..self.n {
                f[i] = g[i].re / norm;
            }
            return;
        }

        let mut g = DVector::<Complex<f64>>::zeros(f.len());

        for j in 0..self.r {
            for k in 0..self.n {
                let idx = j * self.n + k;
                g[idx] = Complex {
                    re: self.periodic_mass_matrix[j] * f[idx],
                    im: 0.0,
                };
            }
        }

        self.fw_fft.process(g.as_mut_slice());

        for k in 0..self.n {
            let x = &self.symbol_pseudo_inv[k] * g.rows_with_step(k, self.r, self.n - 1);
            g.rows_with_step_mut(k, self.r, self.n - 1)
                .set_column(0, &x);
        }

        self.i_fft.process(g.as_mut_slice());

        for i in 0..f.len() {
            f[i] = g[i].re / norm;
        }
    }
}

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

/// N-dimensional HoFFT Laplace solver.
/// Stores one `HoFFT1DLaplaceSolver` per spatial direction.
/// N is a compile-time constant (const generic).
pub struct HoFFTNDLaplaceSolver<const N: usize> {
    pub solvers: [HoFFT1DLaplaceSolver; N],
    /// Diagonal of the local ND mass matrix M^{(0)} ⊗ ... ⊗ M^{(N-1)}.
    /// Length = r_0 * ... * r_{N-1}, last direction varies fastest.
    pub periodic_mass_matrix: DVector<f64>,
    pub alpha: f64,
    pub beta: f64,
}

impl<const N: usize> HoFFTNDLaplaceSolver<N> {
    /// Construct the ND solver from per-direction parameters.
    /// - `ls[d]`  : domain length in direction d
    /// - `ns[d]`  : number of elements in direction d
    /// - `rs[d]`  : polynomial order in direction d
    /// - `alpha`  : coefficient of the mass term (shared across directions)
    /// - `beta`   : coefficient of the stiffness term (shared across directions)
    pub fn new(ls: [f64; N], ns: [usize; N], rs: [usize; N], alpha: f64, beta: f64) -> Self {
        let solvers =
            std::array::from_fn(|d| HoFFT1DLaplaceSolver::new(ls[d], ns[d], rs[d], 0.0, 1.0));

        // Build the diagonal of M^{(0)} ⊗ ... ⊗ M^{(N-1)} iteratively.
        // Each 1D mass matrix is diagonal (GLL quadrature), so the Kronecker product
        // is also diagonal: entry (i_0,...,i_{N-1}) = prod_d M^{(d)}[i_d, i_d].
        let periodic_mass_matrix =
            (1..N).fold(solvers[0].periodic_mass_matrix.clone(), |acc, d| {
                let m_d = &solvers[d].periodic_mass_matrix;
                DVector::from_iterator(
                    acc.len() * m_d.len(),
                    acc.iter().flat_map(|&v| m_d.iter().map(move |&w| v * w)),
                )
            });

        Self {
            solvers,
            periodic_mass_matrix,
            alpha,
            beta,
        }
    }

    /// Return the tensor-product grid as N coordinate vectors, each of length
    /// total_ndof = ∏_d (n_d * r_d). Point `p` has coordinates (result[0][p], …, result[N-1][p]).
    /// The last direction varies fastest
    pub fn get_x(&self) -> [DVector<f64>; N] {
        let x1d: [DVector<f64>; N] = std::array::from_fn(|d| self.solvers[d].get_x());
        let ndofs: [usize; N] = std::array::from_fn(|d| x1d[d].len());
        let total: usize = ndofs.iter().product();
        let strides = compute_strides(&ndofs);

        std::array::from_fn(|d| {
            DVector::from_fn(total, |p, _| x1d[d][(p / strides[d]) % ndofs[d]])
        })
    }

    pub fn solve_in_place(&self, f: &mut DVector<f64>) {
        let ndofs: [usize; N] = std::array::from_fn(|d| self.solvers[d].n * self.solvers[d].r);
        let rs: [usize; N] = std::array::from_fn(|d| self.solvers[d].r);
        let ns: [usize; N] = std::array::from_fn(|d| self.solvers[d].n);
        let total: usize = ndofs.iter().product();
        let r_total: usize = rs.iter().product();
        let n_total: usize = ns.iter().product();
        let strides = compute_strides(&ndofs);
        let local_strides = compute_strides(&rs);
        let n_strides = compute_strides(&ns);

        // local mass matrix index from global flat index (j-component only)
        let local_j = |p: usize| -> usize {
            (0..N).fold(0, |acc, d| {
                acc + ((p / strides[d]) % ndofs[d] / ns[d]) * local_strides[d]
            })
        };
        // offset in g for local DOF j_flat within a Fourier mode
        let joffset = |j_flat: usize| -> usize {
            (0..N).fold(0, |acc, d| {
                acc + (j_flat / local_strides[d]) % rs[d] * ns[d] * strides[d]
            })
        };

        let mut g: Vec<Complex<f64>> = (0..total)
            .map(|p| Complex::new(self.periodic_mass_matrix[local_j(p)] * f[p], 0.0))
            .collect();

        // FFT direction by direction
        for d in 0..N {
            let nd = ndofs[d];
            let stride = strides[d];
            let mut buf = vec![Complex::<f64>::new(0.0, 0.0); nd];
            for s in 0..total / nd {
                let base = flat_base(s, d, &ndofs, &strides);
                for m in 0..nd {
                    buf[m] = g[base + m * stride];
                }
                self.solvers[d].fw_fft.process(&mut buf);
                for m in 0..nd {
                    g[base + m * stride] = buf[m];
                }
            }
        }

        // Apply ND symbol pseudo-inverse
        for k_flat in 0..n_total {
            let base = (0..N).fold(0, |acc, d| {
                acc + (k_flat / n_strides[d]) % ns[d] * strides[d]
            });
            let mut v = DVector::<Complex<f64>>::zeros(r_total);
            for j_flat in 0..r_total {
                v[j_flat] = g[base + joffset(j_flat)];
            }

            // Step a: apply Q†_d for each direction
            for d in 0..N {
                let k_d = (k_flat / n_strides[d]) % ns[d];
                let r_d = rs[d];
                let step = local_strides[d] - 1;
                for s in 0..r_total / r_d {
                    let bj = flat_base(s, d, &rs, &local_strides);
                    let x = self.solvers[d].eigenvectors[k_d].ad_mul(&v.rows_with_step(bj, r_d, step));
                    v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                }
            }

            // Step b: pointwise divide by ND eigenvalue
            for j_flat in 0..r_total {
                let lambda = self.alpha
                    + (0..N)
                        .map(|d| {
                            let k_d = (k_flat / n_strides[d]) % ns[d];
                            let j_d = (j_flat / local_strides[d]) % rs[d];
                            self.beta * self.solvers[d].eigenvalues[k_d][j_d]
                        })
                        .sum::<f64>();
                v[j_flat] /= lambda;
            }

            // Step c: apply Q_d for each direction
            for d in 0..N {
                let k_d = (k_flat / n_strides[d]) % ns[d];
                let r_d = rs[d];
                let step = local_strides[d] - 1;
                for s in 0..r_total / r_d {
                    let bj = flat_base(s, d, &rs, &local_strides);
                    let x = &self.solvers[d].eigenvectors[k_d] * v.rows_with_step(bj, r_d, step);
                    v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                }
            }

            for j_flat in 0..r_total {
                g[base + joffset(j_flat)] = v[j_flat];
            }
        }

        // IFFT direction by direction
        for d in 0..N {
            let nd = ndofs[d];
            let norm = self.solvers[d].n as f64;
            let stride = strides[d];
            let mut buf = vec![Complex::<f64>::new(0.0, 0.0); nd];
            for s in 0..total / nd {
                let base = flat_base(s, d, &ndofs, &strides);
                for m in 0..nd {
                    buf[m] = g[base + m * stride];
                }
                self.solvers[d].i_fft.process(&mut buf);
                for m in 0..nd {
                    g[base + m * stride] = buf[m] / norm;
                }
            }
        }

        for i in 0..total {
            f[i] = g[i].re;
        }
    }
}
