//! Multi-dimensional GLL differentiation and mass-matrix operators.
//!
//! The [`Calculator`] struct assembles, for a tensor-product grid of
//! Gauss or Gauss-Lobatto points, the 1D differentiation matrices and the
//! diagonal mass matrix in each direction.  It exposes [`Calculator::diff`]
//! to apply the sum-factorised differentiation operator along any axis, and
//! [`Calculator::mass_matrix`] / [`Calculator::diff_matrix`] to return the
//! corresponding dense matrices (useful for building the stiffness matrix in
//! the HoFFT and HoT solvers).

use std::vec;

use crate::lagrangian_interpolation::lagrangian_derivative_interpolation;
use crate::quadrature::*;
use nalgebra::*;

/// Multi-dimensional spectral differentiation operator on a tensor-product GLL/Gauss grid.
///
/// Each direction `k` is described by a pair `(n_k, is_lobatto)`:
/// - `n_k`: number of quadrature points in direction `k`.
/// - `is_lobatto`: if `true`, use Gauss-Lobatto points; otherwise use Gauss points.
///
/// The global DOF ordering is *row-major* (last direction varies fastest):
/// flat index = `i_0 * (n_1 * n_2 * …) + i_1 * (n_2 * …) + … + i_{d-1}`.
///
/// # Usage
/// ```rust
/// use quadratix::calculator::Calculator;
/// // 2D Q4 element: 5×5 GLL nodes
/// let calc = Calculator::new(vec![(5, true), (5, true)]);
/// ```
#[derive(Debug)]
pub struct Calculator {
    /// Per-direction description: `(n_points, is_lobatto)`.
    pub basis: Vec<(usize, bool)>,

    /// Quadrature weights, one vector per direction.
    pub weights: Vec<Vec<f64>>,
    /// Quadrature nodes on [0, 1], one vector per direction.
    pub points: Vec<Vec<f64>>,

    /// 1D differentiation matrices D, one per direction.
    /// `diff_matrices[k][j, i] = φ_i'(ξ_j)` where `ξ_j` are the nodes of direction `k`.
    pub diff_matrices: Vec<DMatrix<f64>>,

    /// Total number of DOFs: `∏_k n_k`.
    pub n_points: usize,
    /// Accumulated product strides: `n_points_acc[k] = n_0 * n_1 * … * n_k`.
    pub n_points_acc: Vec<usize>,

    /// Stride step used by [`Calculator::diff`] for direction `k`.
    pub step: Vec<usize>,
    /// Starting flat indices of fibers in direction `k` (indices where `i_k == 0`).
    pub start: Vec<Vec<usize>>,
}

impl Calculator {
    // Convert a flat index to a multi-index `(i_0, i_1, …, i_{d-1})`.
    fn _get_multi_index(n_points_acc: &Vec<usize>, index: usize) -> Vec<usize> {
        let n = n_points_acc.len();

        let mut multi_index = vec![0; n];
        let mut i = index;

        for k in (1..n).rev() {
            multi_index[k] = i / n_points_acc[k - 1];
            i -= n_points_acc[k - 1] * multi_index[k];
        }
        multi_index[0] = i;

        multi_index
    }

    /// Convert a flat index to a multi-index `(i_0, i_1, …, i_{d-1})`.
    pub fn get_multi_index(&self, index: usize) -> Vec<usize> {
        Self::_get_multi_index(&self.n_points_acc, index)
    }

    /// Convert a multi-index `(i_0, i_1, …, i_{d-1})` to a flat index.
    pub fn get_index(&self, multi_index: &Vec<usize>) -> usize {
        let n = self.n_points_acc.len();
        let mut i = 0;

        for k in 1..n {
            i += multi_index[k] * self.n_points_acc[k - 1];
        }

        i += multi_index[0];

        i
    }

    /// Build a `Calculator` from a list of per-direction descriptions.
    ///
    /// Each entry `(n, is_lobatto)` specifies the number of quadrature points
    /// and whether to use Gauss-Lobatto (`true`) or Gauss (`false`) nodes.
    /// Lobatto rules require `n ≥ 2`; Gauss rules require `n ≥ 1`.
    pub fn new(basis: Vec<(usize, bool)>) -> Self {
        let n = basis.len();

        let mut weights = vec![vec![1.0]; n];
        let mut points = vec![vec![0.5]; n];

        let mut diff_matrices = vec![DMatrix::<f64>::zeros(1, 1); n];

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

            // D_{ji} = (phi_i)'(xi_j)
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
                let multi_index = Self::_get_multi_index(&n_points_acc, l);

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

    /// Apply the spectral differentiation operator along direction `dim` in place.
    ///
    /// Computes `sol_diff = D_dim · sol` where `D_dim` is the 1D differentiation
    /// matrix in direction `dim`, applied to every fiber of the tensor-product grid
    /// via sum factorisation.
    ///
    /// # Panics
    /// Panics if `dim ≥ number of directions`, or if `sol`/`sol_diff` have the
    /// wrong length.
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
        }
    }

    /// Return the diagonal mass matrix as a dense `n_points × n_points` matrix.
    ///
    /// Entry `(k, k)` equals the tensor-product quadrature weight at DOF `k`:
    /// `w_{i_0} * w_{i_1} * … * w_{i_{d-1}}`.  Off-diagonal entries are zero.
    pub fn mass_matrix(&self) -> DMatrix<f64> {
        let mut mass_matrix = DMatrix::<f64>::zeros(self.n_points, self.n_points);

        for k in 0..self.n_points {
            let multi_index = self.get_multi_index(k);

            let mut weight = 1.0;
            for l in 0..self.basis.len() {
                weight *= self.weights[l][multi_index[l]];
            }

            mass_matrix[(k, k)] = weight;
        }

        mass_matrix
    }

    /// Return the full differentiation matrix along direction `dim` as a dense
    /// `n_points × n_points` matrix.
    ///
    /// This is the block-diagonal lift of the 1D matrix `D_dim` to the full
    /// tensor-product space, assembled via the same fiber iteration used by
    /// [`Calculator::diff`].  Useful for constructing the stiffness matrix
    /// `K = D^T M D`.
    pub fn diff_matrix(&self, dim: usize) -> DMatrix<f64> {
        let mut diff_matrix = DMatrix::<f64>::zeros(self.n_points, self.n_points);

        let step = self.step[dim];
        let start = &self.start[dim];

        let n = self.basis[dim].0;

        for k in start {
            diff_matrix
                .view_with_steps_mut((*k, *k), (n, n), (step, step))
                .copy_from(&self.diff_matrices[dim]);
        }

        diff_matrix
    }
}
