use crate::quadrature::*;
use nalgebra::*;

/// 1D Interpolation at 'p' of the ith basis function. The lagrangian basis are built upon
/// the list of points given by 'pos'. The index i should be between 0 and n-1.
pub fn lagrangian_interpolation(pos: &Vec<f64>, p: f64, i: usize) -> f64 {
    let mut s = 1.0;
    let mut d = 1.0;

    for r in 0..pos.len() {
        if r != i {
            d *= pos[i] - pos[r];
        }
    }

    for r in 0..pos.len() {
        if r != i {
            s *= p - pos[r];
        }
    }

    debug_assert!(d != 0.0);

    s / d
}

// 1D Interpolation at 'p' of the derivative of the ith basis function. The lagrangian basis are built upon
// the list of points given by 'pos'. The index i should be between 0 and n-1.
pub fn lagrangian_derivative_interpolation(pos: &Vec<f64>, p: f64, i: usize) -> f64 {
    let mut s = 0.0;
    let mut d = 1.0;

    for r in 0..pos.len() {
        if r != i {
            d *= pos[i] - pos[r];
        }
    }

    for r1 in 0..pos.len() {
        if r1 != i {
            let mut c = 1.0;
            for r2 in 0..pos.len() {
                if r2 != r1 && r2 != i {
                    c *= p - pos[r2];
                }
            }
            s += c;
        }
    }

    debug_assert!(d != 0.0);

    s / d
}

// Operator that can interpolate a function or its derivative represented on a Lagrangian basis on Gauss or Gauss-Lobatto points
// on another Lagrangian basis on Gauss or Gauss-Lobatto points
// Note: at the moment only functions represented on the same set of points are implemented
#[derive(Debug)]
pub struct Calculator {
    pub basis: Vec<(usize, bool)>,

    pub weights: Vec<Vec<f64>>,
    pub points: Vec<Vec<f64>>,

    pub diff_matrices: Vec<DMatrix<f64>>,

    pub n_points: usize,
    pub n_points_acc: Vec<usize>,

    pub step: Vec<usize>,
    pub start: Vec<Vec<usize>>,
}

impl Calculator {
    fn get_multi_index(n_points_acc: &Vec<usize>, mut i: usize) -> Vec<usize> {
        let n = n_points_acc.len();

        let mut multi_index = vec![0; n];

        for k in (1..n).rev() {
            multi_index[k] = i / n_points_acc[k - 1];
            i -= n_points_acc[k - 1] * multi_index[k];
        }
        multi_index[0] = i;

        multi_index
    }

    fn get_index(n_points_acc: &Vec<usize>, mult_i: Vec<usize>) -> usize {
        let n = n_points_acc.len();
        let mut i = 0;

        for k in 1..n {
            i += mult_i[k] * n_points_acc[k - 1];
        }

        i += mult_i[0];

        i
    }

    pub fn new(basis: Vec<(usize, bool)>) -> Self {
        let n = basis.len();

        let mut weights = vec![vec![1.0]; n];
        let mut points = vec![vec![0.5]; n];

        let mut diff_matrices = vec![nalgebra::DMatrix::<f64>::zeros(1, 1); n];

        for k in 0..n {
            assert!(
                (basis[k].0 > 1 && basis[k].1 == true) || (basis[k].0 > 0 && basis[k].1 == false),
                "not enough points to define the quadrature rule"
            );

            if basis[k].1 == true {
                (points[k], weights[k]) = gauss_lobatto_on_unit_interval(basis[k].0);
            } else {
                (points[k], weights[k]) = gauss_on_unit_interval(basis[k].0);
            }

            diff_matrices[k].resize_mut(basis[k].0, basis[k].0, 0.0);

            for i in 0..basis[k].0 {
                for j in 0..basis[k].0 {
                    diff_matrices[k][(j, i)] =
                        lagrangian_derivative_interpolation(&points[k], points[k][j], i);
                }
            }
        }

        let mut step = vec![1; n];
        let mut start = vec![Vec::<usize>::new(); n];

        let mut n_points = 1;
        let mut n_points_acc = vec![basis[0].0; n];

        for k in 0..n {
            n_points *= basis[k].0;
        }

        for k in 1..n {
            n_points_acc[k] = basis[k].0 * n_points_acc[k - 1];
        }

        for k in 1..n {
            step[k] = step[k - 1] * basis[k].0;
        }

        for k in 0..n {
            step[k] -= 1;
        }

        for k in 0..n {
            for l in 0..n_points {
                let multi_index = Self::get_multi_index(&n_points_acc, l);

                if multi_index[k] == 0 {
                    start[k].push(l);
                }
            }
        }

        Self {
            basis,
            weights,
            points,
            diff_matrices,
            n_points,
            n_points_acc,
            step,
            start,
        }
    }

    pub fn diff(&self, dim: usize, sol: &DVector<f64>, sol_diff: &mut DVector<f64>) {
        assert!(
            dim < self.basis.len(),
            "Differentiation is not defined in this direction"
        );

        assert!(
            self.n_points == sol.len(),
            "The input vector does not have the adequate size"
        );

        assert!(
            self.n_points == sol_diff.len(),
            "The ouput vector does not have the adequate size"
        );

        let step = self.step[dim];
        let start = &self.start[dim];

        let diff_matrix = &self.diff_matrices[dim];
        let n = self.basis[dim].0;

        for k in start {
            let sol_view = sol.rows_with_step(*k, n, step);

            let mut sol_diff_view = sol_diff.rows_with_step_mut(*k, n, step);

            diff_matrix.mul_to(&sol_view, &mut sol_diff_view);

            // for j in 0..n {
            //     for i in 0..n {
            //         sol_diff[k + (step + 1) * j] += sol[k + (step + 1) * i] * diff_matrix[(j, i)];
            //     }
            // }
        }
    }
}
