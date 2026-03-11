//! High-order FFT (HoFFT) solver for the periodic Laplace equation.
//!
//! Solves (α M + β K) u = f` on `[0, L]^N` with periodic boundary conditions
//! using a spectral-element discretisation: *n* elements of polynomial order *r*
//! per direction, so each direction has `n·r` DOFs.
//!
//! ## Algorithm (1D outline)
//!
//! 1. **Mass scaling** — multiply the load vector `f` by the periodic diagonal
//!    mass matrix `M` to convert from strong-form RHS to FEM RHS.
//! 2. **Forward FFT** — apply a length-*n* FFT along each direction to decouple
//!    the *n* Fourier modes.
//! 3. **Symbol solve** — for each Fourier mode `k`, apply the pseudo-inverse of
//!    the local symbol matrix `S(k)` (an *r×r* Hermitian system) via its
//!    pre-computed eigendecomposition.
//! 4. **Inverse FFT** — transform back to physical space.
//!
//! The ND solver (`HoFFTNDLaplaceSolver`) extends this by applying steps 2 and 4
//! direction by direction (sum factorisation) and solving the full `r^N × r^N`
//! block-diagonal system in step 3 using tensor-product eigendecompositions.
//! Steps 2 and 4 are parallelised with Rayon over independent fibers; step 3 is
//! parallelised over independent Fourier modes.

use crate::calculator::*;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::Arc;

use nalgebra::*;
use rustfft::*;

/// 1D HoFFT Laplace solver: $n$ elements of polynomial order $r$ in $\[0,L\]$.
///
/// Solves $$ \alpha u - \beta \Delta u = f $$ with periodic boundary conditions using:
/// - a GLL spectral-element discretisation with *n* elements of order *r*,
/// - FFT diagonalisation over the *n* element-to-element coupling modes,
/// - per-mode *r×r* Hermitian eigendecomposition for the local solve.
///
/// The DOF layout in `get_x` / `solve_in_place` is *interleaved by mode*:
/// flat index `j * n + k` corresponds to local node `j` of element `k`.
pub struct HoFFT1DLaplaceSolver {
    /// Domain length.
    pub l: f64,
    /// Number of elements.
    pub n: usize,
    /// Polynomial order per element (number of interior+boundary GLL nodes per element).
    pub r: usize,

    /// Forward FFT engine (length *n*).
    pub fw_fft: Arc<dyn Fft<f64>>,
    /// Inverse FFT engine (length *n*).
    pub i_fft: Arc<dyn Fft<f64>>,

    /// GLL quadrature calculator for a single element of order *r*.
    pub lobatto_calculator: Calculator,
    /// Per-Fourier-mode symbol matrices `S(k)`, size *r×r*, Hermitian.
    /// `symbol[k] = α h M_loc + β/h K_loc(e^{2πi k/n})`.
    pub symbol: Vec<DMatrix<Complex<f64>>>,
    /// Pseudo-inverse of each symbol matrix, pre-computed via eigendecomposition.
    pub symbol_pseudo_inv: Vec<DMatrix<Complex<f64>>>,
    /// Real eigenvalues of the mass-normalised symbol for each Fourier mode `k`.
    /// `eigenvalues[k][j]` = λ_j of `M^{-1/2} S(k) M^{-1/2}`.
    pub eigenvalues: Vec<DVector<f64>>,
    /// M-orthonormal eigenvectors: `S(k) V(k) = M V(k) diag(λ)`.
    /// `eigenvectors[k]` is the *r×r* complex unitary matrix.
    pub eigenvectors: Vec<DMatrix<Complex<f64>>>,
    /// Diagonal of the periodic local mass matrix (length *r*).
    /// DOF 0 collects contributions from both sides of the periodic boundary.
    pub periodic_mass_matrix: DVector<f64>,
    /// Coefficient of the mass term α.
    pub alpha: f64,
    /// Coefficient of the stiffness term β.
    pub beta: f64,
}

impl HoFFT1DLaplaceSolver {
    /// Construct the 1D HoFFT solver.
    ///
    /// - `l`: domain length.
    /// - `n`: number of elements.
    /// - `r`: polynomial order (GLL nodes per element, excluding the shared endpoint).
    /// - `alpha`, `beta`: operator coefficients for `α M + β K`.
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

        //Compute the symbol of the operator for each frequency
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

    /// GLL nodes on `[0, L]`, length `n * r`, ordered by local node then element:
    /// `x[j * n + k]` = position of local node `j` in element `k`.
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

    /// Solve `(α M + β K) u = f` in place.
    ///
    /// On input, `f` contains the strong-form right-hand side evaluated at
    /// the GLL nodes returned by [`HoFFT1DLaplaceSolver::get_x`].
    /// On output, `f` contains the FEM solution `u` at the same nodes.
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

// Return the column-major strides for a tensor-product array of shape `sizes`.
// st[0] = 1 (direction 0 / x varies fastest), st[d] = st[d-1] * sizes[d-1].
fn compute_strides<const N: usize>(sizes: &[usize; N]) -> [usize; N] {
    let mut st = [1usize; N];
    for d in 1..N {
        st[d] = st[d - 1] * sizes[d - 1];
    }
    st
}

// Compute the flat base index of the `s`-th fiber along direction `d`.
//
// A "fiber" along direction `d` is the set of indices obtained by varying only
// the `d`-th coordinate.  This function returns the flat index at which that
// fiber starts (i.e. where the `d`-th coordinate is 0).
//
// For a given `d`, successive values of `s` in `0..total/sizes[d]` produce
// *disjoint* index sets — a key safety invariant for the parallel unsafe code.
fn flat_base<const N: usize>(
    s: usize,
    d: usize,
    sizes: &[usize; N],
    strides: &[usize; N],
) -> usize {
    let mut base = 0usize;
    let mut rem = s;
    for d2 in 0..N {
        if d2 == d {
            continue;
        }
        base += (rem % sizes[d2]) * strides[d2];
        rem /= sizes[d2];
    }
    base
}

/// N-dimensional HoFFT Laplace solver.
///
/// Extends [`HoFFT1DLaplaceSolver`] to `N` spatial dimensions using sum
/// factorisation: FFTs are applied direction by direction, and the symbol
/// pseudo-inverse is applied to each `(r_0 × … × r_{N-1})`-dimensional block
/// independently for each Fourier mode combination.
///
/// `N` is a compile-time constant (const generic), so the compiler can
/// unroll the per-direction loops.
pub struct HoFFTNDLaplaceSolver<const N: usize> {
    /// One 1D solver per spatial direction.
    pub solvers: [HoFFT1DLaplaceSolver; N],
    /// Diagonal of the Kronecker-product mass matrix `M^{(0)} ⊗ … ⊗ M^{(N-1)}`.
    /// Length = `r_0 * … * r_{N-1}`, first direction (x) varies fastest.
    pub periodic_mass_matrix: DVector<f64>,
    /// Coefficient of the mass term α (shared across directions).
    pub alpha: f64,
    /// Coefficient of the stiffness term β (shared across directions).
    pub beta: f64,
}

impl<const N: usize> HoFFTNDLaplaceSolver<N> {
    /// Construct the ND solver from per-direction parameters.
    ///
    /// - `ls[d]`: domain length in direction `d`.
    /// - `ns[d]`: number of elements in direction `d`.
    /// - `rs[d]`: polynomial order in direction `d`.
    /// - `alpha`, `beta`: coefficients of `α M + β K`.
    pub fn new(ls: [f64; N], ns: [usize; N], rs: [usize; N], alpha: f64, beta: f64) -> Self {
        let solvers =
            std::array::from_fn(|d| HoFFT1DLaplaceSolver::new(ls[d], ns[d], rs[d], 0.0, 1.0));

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

        Self {
            solvers,
            periodic_mass_matrix,
            alpha,
            beta,
        }
    }

    /// Return the tensor-product grid as `N` coordinate vectors, each of length
    /// `total_ndof = ∏_d (n_d * r_d)`.
    ///
    /// Point `p` has coordinates `(result[0][p], …, result[N-1][p])`.
    /// The first direction (x) varies fastest.
    pub fn get_x(&self) -> [DVector<f64>; N] {
        let x1d: [DVector<f64>; N] = std::array::from_fn(|d| self.solvers[d].get_x());
        let ndofs: [usize; N] = std::array::from_fn(|d| x1d[d].len());
        let total: usize = ndofs.iter().product();
        let strides = compute_strides(&ndofs);

        std::array::from_fn(|d| DVector::from_fn(total, |p, _| x1d[d][(p / strides[d]) % ndofs[d]]))
    }

    /// Solve `(α M + β K) u = f` in place using the ND HoFFT algorithm.
    ///
    /// On input, `f` contains the strong-form RHS at the tensor-product GLL grid
    /// returned by [`HoFFTNDLaplaceSolver::get_x`].
    /// On output, `f` contains the FEM solution `u` at the same grid.
    ///
    /// # Parallelism
    /// - Mass scaling: trivially parallel (`par_iter_mut`).
    /// - Forward/inverse FFT: parallel over independent fibers (unsafe raw-pointer
    ///   aliasing, safe by the disjoint-fiber invariant of `flat_base`).
    /// - Symbol solve: parallel over independent Fourier modes (unsafe, safe by
    ///   injectivity of the global index formula).
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

        // Build g with mass scaling (parallel, no unsafe)
        let mut g: Vec<Complex<f64>> = f
            .as_slice()
            .par_iter()
            .enumerate()
            .map(|(p, &fp)| {
                let lj = (0..N).fold(0, |acc, d| {
                    acc + ((p / strides[d]) % ndofs[d] / ns[d]) * local_strides[d]
                });
                Complex::new(self.periodic_mass_matrix[lj] * fp, 0.0)
            })
            .collect();

        // Forward FFT direction by direction.
        // SAFETY: flat_base yields disjoint strided slices of g for distinct s values.
        for d in 0..N {
            let nd = ndofs[d];
            let stride = strides[d];
            let fw_fft = &self.solvers[d].fw_fft;
            let ptr = g.as_mut_ptr() as usize;
            (0..total / nd).into_par_iter().for_each(|s| {
                let base = flat_base(s, d, &ndofs, &strides);
                let p = ptr as *mut Complex<f64>;
                let mut buf: Vec<Complex<f64>> = (0..nd)
                    .map(|m| unsafe { *p.add(base + m * stride) })
                    .collect();
                fw_fft.process(&mut buf);
                for m in 0..nd {
                    unsafe { *p.add(base + m * stride) = buf[m] };
                }
            });
        }

        // Apply ND symbol pseudo-inverse, parallel over Fourier modes.
        // SAFETY: different k_flat map to disjoint cells in g —
        //   index = Σ_d [k_d + j_d·n_d]·strides[d] is injective in (k_flat, j_flat),
        //   so sets for distinct k_flat are disjoint.
        {
            let ptr = g.as_mut_ptr() as usize;
            (0..n_total).into_par_iter().for_each(|k_flat| {
                let base = (0..N).fold(0, |acc, d| {
                    acc + (k_flat / n_strides[d]) % ns[d] * strides[d]
                });
                let p = ptr as *mut Complex<f64>;

                // Gather the r_total values belonging to this Fourier mode
                let mut v = DVector::<Complex<f64>>::from_fn(r_total, |j_flat, _| {
                    let off = (0..N).fold(0, |acc, d| {
                        acc + (j_flat / local_strides[d]) % rs[d] * ns[d] * strides[d]
                    });
                    unsafe { *p.add(base + off) }
                });

                // Step a: apply Q†_d for each direction
                for d in 0..N {
                    let k_d = (k_flat / n_strides[d]) % ns[d];
                    let r_d = rs[d];
                    let step = local_strides[d] - 1;
                    for s in 0..r_total / r_d {
                        let bj = flat_base(s, d, &rs, &local_strides);
                        let x = self.solvers[d].eigenvectors[k_d]
                            .ad_mul(&v.rows_with_step(bj, r_d, step));
                        v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                    }
                }

                // Step b: pointwise divide by ND eigenvalue λ = α + Σ_d β λ^{(d)}_{k_d, j_d}
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
                        let x =
                            &self.solvers[d].eigenvectors[k_d] * v.rows_with_step(bj, r_d, step);
                        v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                    }
                }

                // Scatter back
                for j_flat in 0..r_total {
                    let off = (0..N).fold(0, |acc, d| {
                        acc + (j_flat / local_strides[d]) % rs[d] * ns[d] * strides[d]
                    });
                    unsafe { *p.add(base + off) = v[j_flat] };
                }
            });
        }

        // Inverse FFT direction by direction.
        // SAFETY: same disjoint-fiber argument as forward FFT.
        for d in 0..N {
            let nd = ndofs[d];
            let norm = self.solvers[d].n as f64;
            let stride = strides[d];
            let i_fft = &self.solvers[d].i_fft;
            let ptr = g.as_mut_ptr() as usize;
            (0..total / nd).into_par_iter().for_each(|s| {
                let base = flat_base(s, d, &ndofs, &strides);
                let p = ptr as *mut Complex<f64>;
                let mut buf: Vec<Complex<f64>> = (0..nd)
                    .map(|m| unsafe { *p.add(base + m * stride) })
                    .collect();
                i_fft.process(&mut buf);
                for m in 0..nd {
                    unsafe { *p.add(base + m * stride) = buf[m] / norm };
                }
            });
        }

        for i in 0..total {
            f[i] = g[i].re;
        }
    }
}
