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
    pub mass_matrix: DMatrix<f64>,
    pub stiffness_matrix: DMatrix<f64>,
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

        Self {
            l,
            n,
            r,
            fw_fft,
            i_fft,
            lobatto_calculator,
            symbol,
            mass_matrix,
            stiffness_matrix,
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
        let mut g = DVector::<Complex<f64>>::zeros(f.len());
        let h = self.l / (self.n as f64);

        // Vertex DOF (j=0) is assembled from two elements
        for k in 0..self.n {
            g[k] = Complex {
                re: 2.0 * h * self.mass_matrix[(0, 0)] * f[k],
                im: 0.0,
            };
        }
        for j in 1..self.r {
            for k in 0..self.n {
                let idx = j * self.n + k;
                g[idx] = Complex {
                    re: h * self.mass_matrix[(j, j)] * f[idx],
                    im: 0.0,
                };
            }
        }

        self.fw_fft.process(g.as_mut_slice());

        for k in 0..self.n {
            let svd = SVD::new(self.symbol[k].clone(), true, true);
            let x = svd
                .solve(&g.rows_with_step(k, self.r, self.n - 1), 1e-14)
                .unwrap();

            g.rows_with_step_mut(k, self.r, self.n - 1)
                .set_column(0, &x);
        }

        self.i_fft.process(g.as_mut_slice());

        let norm = self.n as f64;
        for i in 0..f.len() {
            f[i] = g[i].re / norm;
        }
    }
}
