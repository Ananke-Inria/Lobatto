//! Multi-dimensional GLL differentiation and mass-matrix operators.
//!
//! [`Basis`] assembles, for a tensor-product grid of Gauss or Gauss–Lobatto–Legendre
//! nodes, the per-direction **differentiation matrices** and the **diagonal mass
//! matrix**.
//!
//! ## Differentiation matrix
//!
//! For direction $k$ with $n_k$ nodes $\xi_0 < \cdots < \xi_{n_k-1}$, the 1D
//! differentiation matrix $D^{(k)}$ is defined by
//! $$D^{(k)}_{ji} = L_i'(\xi_j), \quad i,j = 0, \ldots, n_k - 1,$$
//! where $L_i$ is the $i$-th Lagrange basis polynomial.
//! [`Basis::diff`] applies the resulting operator fibre-by-fibre over the full
//! tensor-product grid; [`Basis::diff_matrix`] returns it as a dense matrix.
//!
//! ## Mass matrix
//!
//! The mass matrix is diagonal with entries
//! $$M_{pp} = \prod_{k=0}^{d-1} w^{(k)}_{i_k(p)},$$
//! where $w^{(k)}_j$ is the $j$-th quadrature weight in direction $k$ and
//! $(i_0(p), \ldots, i_{d-1}(p))$ is the multi-index of DOF $p$.
//!
//! ## DOF ordering
//!
//! The global DOF ordering is **row-major** (last direction varies fastest):
//! $$p = i_0 \cdot (n_1 n_2 \cdots) + i_1 \cdot (n_2 \cdots) + \cdots + i_{d-1}.$$

use crate::quadrature::*;
use crate::utilities::lagrangian_derivative_interpolation;
use nalgebra::*;

/// Multi-dimensional spectral differentiation and mass-matrix operator on a
/// tensor-product GLL or GL grid.
///
/// Each direction $k$ is described by a pair `(n_k, is_lobatto)`:
/// - `n_k`: number of quadrature nodes in direction $k$.
/// - `is_lobatto`: `true` for Gauss–Lobatto–Legendre nodes, `false` for Gauss nodes.
///
/// # Example
/// ```rust
/// use quadratix::basis::Basis;
/// // 2D Q4 element: 5×5 GLL nodes
/// let basis = Basis::new(vec![(5, true), (5, true)]);
/// ```
#[derive(Debug)]
pub struct Basis {
    /// Per-direction description: `(n_points, is_lobatto)`.
    basis: Vec<(usize, bool)>,

    /// Quadrature weights on $[0,1]$, one vector per direction.
    weights: Vec<Vec<f64>>,
    /// Quadrature nodes on $[0,1]$, one vector per direction.
    points: Vec<Vec<f64>>,

    /// 1D differentiation matrices, one per direction.
    /// `diff_matrices[k][(j, i)]` $= L_i'(\xi_j)$ where $\xi_j$ are the nodes
    /// of direction $k$.
    diff_matrices: Vec<DMatrix<f64>>,

    /// Total number of DOFs: $\prod_k n_k$.
    n_points: usize,
    /// Prefix products of point counts: `n_points_acc[k]` $= n_0 \cdots n_k$.
    n_points_acc: Vec<usize>,

    /// Stride step used by [`Basis::diff`] for direction $k$.
    step: Vec<usize>,
    /// Starting flat indices of fibres in direction $k$ (all $p$ with $i_k(p) = 0$).
    start: Vec<Vec<usize>>,
}

impl Basis {
    // Convert a flat DOF index to the multi-index $(i_0, \ldots, i_{d-1})$.
    fn flat_to_multi(n_points_acc: &[usize], index: usize) -> Vec<usize> {
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

    /// Convert a flat DOF index to the multi-index $(i_0, \ldots, i_{d-1})$.
    pub fn get_multi_index(&self, index: usize) -> Vec<usize> {
        Self::flat_to_multi(&self.n_points_acc, index)
    }

    /// Convert a multi-index $(i_0, \ldots, i_{d-1})$ to a flat DOF index.
    pub fn get_index(&self, multi_index: &[usize]) -> usize {
        let n = self.n_points_acc.len();
        let mut i = multi_index[0];
        for k in 1..n {
            i += multi_index[k] * self.n_points_acc[k - 1];
        }
        i
    }

    /// Build a `Basis` from a list of per-direction descriptions `(n, is_lobatto)`.
    ///
    /// `n` is the number of quadrature nodes and `is_lobatto` selects GLL (`true`)
    /// or GL (`false`) nodes on $[0,1]$.
    ///
    /// # Panics
    /// Panics if a GLL direction has fewer than 2 nodes, or a GL direction has 0 nodes.
    pub fn new(basis: Vec<(usize, bool)>) -> Self {
        let n = basis.len();

        let mut weights = vec![vec![1.0]; n];
        let mut points = vec![vec![0.5]; n];
        let mut diff_matrices = vec![DMatrix::<f64>::zeros(1, 1); n];

        for k in 0..n {
            assert!(
                (basis[k].0 > 1 && basis[k].1) || (basis[k].0 > 0 && !basis[k].1),
                "not enough points to define the quadrature rule"
            );

            if basis[k].1 {
                (points[k], weights[k]) = gauss_lobatto_on_unit_interval(basis[k].0);
            } else {
                (points[k], weights[k]) = gauss_on_unit_interval(basis[k].0);
            }

            diff_matrices[k].resize_mut(basis[k].0, basis[k].0, 0.0);

            // D_{ji} = L_i'(ξ_j)
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
                let multi_index = Self::flat_to_multi(&n_points_acc, l);
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

    /// Return the quadrature nodes on $[0,1]$ for direction `dim`.
    pub fn points(&self, dim: usize) -> &[f64] {
        &self.points[dim]
    }

    /// Return the raw 1D differentiation matrix for direction `dim`.
    ///
    /// Entry $(j, i)$ equals $L_i'(\xi_j)$.  For the full tensor-product lift
    /// see [`Basis::diff_matrix`].
    pub fn diff_matrix_1d(&self, dim: usize) -> &DMatrix<f64> {
        &self.diff_matrices[dim]
    }

    /// Apply the differentiation operator along direction `dim` in place.
    ///
    /// Computes $\mathtt{sol\_diff} \leftarrow D^{(\mathrm{dim})} \cdot \mathtt{sol}$,
    /// where $D^{(\mathrm{dim})}$ is applied fibre-by-fibre over the tensor-product grid:
    /// for each combination of indices $(i_0, \ldots, \widehat{i_{\mathrm{dim}}}, \ldots, i_{d-1})$,
    /// the length-$n_{\mathrm{dim}}$ fibre is multiplied by $D^{(\mathrm{dim})}$.
    ///
    /// # Panics
    /// Panics if `dim ≥ d`, or if `sol`/`sol_diff` have the wrong length.
    pub fn diff(&self, dim: usize, sol: &[f64], sol_diff: &mut [f64]) {
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
            "The output vector does not have the adequate size"
        );

        let step = self.step[dim];
        let start = &self.start[dim];
        let diff_matrix = &self.diff_matrices[dim];
        let n = self.basis[dim].0;

        let sol_na = DVectorView::from_slice(sol, sol.len());
        let mut sol_diff_na = DVectorViewMut::from_slice(sol_diff, sol_diff.len());

        for k in start {
            let sol_view = sol_na.rows_with_step(*k, n, step);
            let mut sol_diff_view = sol_diff_na.rows_with_step_mut(*k, n, step);
            diff_matrix.mul_to(&sol_view, &mut sol_diff_view);
        }
    }

    /// Return the diagonal mass matrix as a dense $N_\mathrm{dof} \times N_\mathrm{dof}$ matrix.
    ///
    /// Entry $(p, p)$ equals the tensor-product quadrature weight at DOF $p$:
    /// $$M_{pp} = \prod_{k=0}^{d-1} w^{(k)}_{i_k(p)}.$$
    /// All off-diagonal entries are zero.
    pub fn mass_matrix(&self) -> DMatrix<f64> {
        let mut mass_matrix = DMatrix::<f64>::zeros(self.n_points, self.n_points);
        for k in 0..self.n_points {
            let multi_index = self.get_multi_index(k);
            let weight = (0..self.basis.len())
                .map(|l| self.weights[l][multi_index[l]])
                .product();
            mass_matrix[(k, k)] = weight;
        }
        mass_matrix
    }

    /// Return the full tensor-product differentiation matrix along direction `dim` as a
    /// dense $N_\mathrm{dof} \times N_\mathrm{dof}$ matrix.
    ///
    /// This is the block-diagonal lift of $D^{(\mathrm{dim})}$ to the full tensor-product
    /// space, assembled via the same fibre iteration as [`Basis::diff`].
    /// Useful for building the stiffness matrix $K = D^T M D$.
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
