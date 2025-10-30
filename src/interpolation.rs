use crate::quadrature::*;

/// 1D Interpolation at 'p' of the ith basis function. The lagrangian basis are built upon
/// the list of points given by 'pos'. The index i should be between 0 and n-1.
/// if two values are identical in the list of points the behavior is undetermined
pub fn lagrangian_interpolation(pos: &Vec<f64>, p: f64, i: usize) -> f64 {
    let mut s = 1.0;

    for r in 0..pos.len() {
        if r != i {
            s *= (p - pos[r]) / (pos[i] - pos[r]);
        }
    }

    s
}

// 1D Interpolation at 'p' of the derivative of the ith basis function. The lagrangian basis are built upon
// the list of points given by 'pos'. The index i should be between 0 and n-1.
// if two values are identical in the list of points the behavior is undertemined
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

    s / d
}

// Operator that can interpolate a function or its derivative represented on a Lagrangian basis on Gauss or Gauss-Lobatto points
// on another Lagrangian basis on Gauss or Gauss-Lobatto points
// Note: at the moment only functions represented on the same set of points are implemented
#[derive(Debug)]
pub struct Interpolator {
    pub dim: Vec<usize>,
    pub weights: Vec<Vec<f64>>,
    pub points: Vec<Vec<f64>>,
    pub derivative_interpolation: Vec<Vec<Vec<f64>>>,

    pub n_points: usize,
    pub n_points_acc: Vec<usize>,
    pub iterator_dim_dec: Vec<usize>,
    pub iterator_dim_init: Vec<Vec<usize>>,
}

impl Interpolator {
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

    pub fn new(dim: Vec<usize>) -> Self {
        let n = dim.len();

        let mut weights = vec![vec![1.0]; n];
        let mut points = vec![vec![0.5]; n];
        let mut derivative_interpolation = vec![vec![vec![0.0]]; n];

        for k in 0..n {
            assert!(
                dim[k] != 0,
                "There should be at elast one point in all dimension"
            );

            if dim[k] > 1 {
                (points[k], weights[k]) = normalized_gauss_lobatto(dim[k]);

                derivative_interpolation[k] = vec![vec![0.0; dim[k]]; dim[k]];

                for i in 0..dim[k] {
                    for j in 0..dim[k] {
                        derivative_interpolation[k][i][j] =
                            lagrangian_derivative_interpolation(&points[k], points[k][j], i);
                    }
                }
            }
        }

        let mut iterator_dim_dec = vec![1; n];
        let mut iterator_dim_init = vec![Vec::<usize>::new(); n];
        let mut n_points = 1;
        let mut n_points_acc = vec![dim[0]; n];

        for k in 0..n {
            n_points *= dim[k];
        }

        for k in 1..n {
            n_points_acc[k] = dim[k] * n_points_acc[k - 1];
        }

        for k in 1..n {
            iterator_dim_dec[k] = iterator_dim_dec[k - 1] * dim[k];
        }

        for k in 0..n {
            for l in 0..n_points {
                let multi_index = Self::get_multi_index(&n_points_acc, l);

                if multi_index[k] == 0 {
                    iterator_dim_init[k].push(l);
                }
            }
        }

        Self {
            dim,
            weights,
            points,
            derivative_interpolation,
            n_points,
            n_points_acc,
            iterator_dim_dec,
            iterator_dim_init,
        }
    }

    pub fn derivative_interpolation(&self, dim: usize, sol: &Vec<f64>, sol_der: &mut Vec<f64>) {
        assert!(dim < self.dim.len(), "Derivative is not defined");

        let dec = self.iterator_dim_dec[dim];
        let init_indices = &self.iterator_dim_init[dim];
        let der_interpol = &self.derivative_interpolation[dim];
        let n = self.dim[dim];

        for k in init_indices {
            for j in 0..n {
                for i in 0..n {
                    sol_der[k + dec * j] += sol[k + dec * i] * der_interpol[i][j];
                }
            }
        }
    }
}
