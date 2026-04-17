//! Multi-dimensional Lagrangian basis on the unit hypercube $\[0,1\]^d$ on tensor-product Gauss and Gauss–Lobatto grids. Interpolations points are collocated with quadrature points.
//!
//! ## 1D Lagrangian basis
//!
//! In each direction $k$, choose $n_k$ distinct nodes
//! $\xi_0^{(k)} < \xi_1^{(k)} < \cdots < \xi_{n_k-1}^{(k)}$,
//! either the **Gauss–Legendre** (GL) or the **Gauss–Lobatto–Legendre** (GLL) points on
//! $\[0,1\]$ (see [`crate::quadrature`]).
//! The associated **Lagrange basis polynomials** $L_0^{(k)}, \ldots, L_{n_k-1}^{(k)}$
//! are the unique degree-$(n_k-1)$ polynomials satisfying the interpolation condition
//! $$L_i^{(k)}(\xi_j^{(k)}) = \delta_{ij}, \quad i,j = 0, \ldots, n_k-1.$$
//!
//! ## ND Lagrangian basis by tensor-product   
//!
//! In $d$ dimensions with nodes $\xi_{i_k}^{(k)}$ in direction $k$, the
//! **tensor-product Lagrange basis** is
//! $$\Phi_p \big(\xi^{(0)}, \ldots, \xi^{(d-1)} \big)
//!   = \prod_{k=0}^{d-1} L_{i_k(p)}^{(k)}(\xi^{(k)}),$$
//! where $p$ is a flat index and $(i_0(p), \ldots, i_{d-1}(p))$ its multi-index.
//! Any function $f$ represented by the basis reads
//! $$f(\boldsymbol{\xi}) = \sum_p f_p ~ \Phi_p(\boldsymbol{\xi}),
//!   \quad f_p = f \big(\xi_{i_0(p)}^{(0)}, \ldots, \xi_{i_{d-1}(p)}^{(d-1)}\big).$$
//! Note that the flat index is obtained using an ordering where lower direction number varies fastest:
//! $$p = i_0 + n_0 ~ i_1 + n_0 ~n_1 ~ i_2 + \cdots $$
//! giving a total of $N_\mathrm{dof} = \prod_{k=0}^{d-1} n_k$ degrees of freedom.
//!
//! [`CollocationBasis::points`] returns the list of quadrature/interpolation points;
//!
//! ## Differentiation matrix
//!
//! For each direction $k$ the entries of the matrix $D^{(k)}$  are
//! $$
//! D^{(k)}_{ji} = \big( \frac{d}{d\xi} L_i^{(k)} \big)(\xi_j^{(k)}), \quad i,j = 0, \ldots, n_k - 1.
//! $$
//! Differentiating a function $f$ with respect to $\xi^{(k)}$ is achieved by
//! applying the 1D matrix $D^{(k)}$ on each fiber (i.e. the set of DOFs with the same indices in all directions except $k$). If
//! $
//! f(\boldsymbol{\xi}) = \sum_p f_p ~ \Phi_p(\boldsymbol{\xi}) $ then
//! $$
//!   \big( \frac{\partial f}{\partial \xi^{(k)}} \big) (\boldsymbol{\xi})  = \sum_p  g_p ~ \Phi _p  (\boldsymbol{\xi})
//! $$
//! where $g_p$ is given by matrix/vector product with the 1D differentiation matrices  $D^{(k)}$.
//!
//! [`CollocationBasis::diff`] applies this fibre-by-fibre over the full grid in place;
//! [`CollocationBasis::diff_matrix`] returns the corresponding $N_\mathrm{dof} \times N_\mathrm{dof}$ matrix.
//!
//! ## Quadrature weights
//!
//! The quadrature weights $w_i^{(k)}$ associated with the nodes $\xi_i^{(k)}$ define a
//! quadrature on the unit hypercube of dimension $d$ with weight $w_p$ for each flat index $p$ with multi-index $(i_0, \ldots, i_{d-1})$, given by
//! $$
//! w_p = \prod_{k=0}^{d-1} w^{(k)}_ {i_k(p)}.
//! $$
//! [`CollocationBasis::weights`] returns the weights as a `Vec<f64>`;
//! [`CollocationBasis::weights_matrix`] returns the weights as the diagonal of a full dense $N_\mathrm{dof} \times N_\mathrm{dof}$ matrix.

use crate::quadrature::*;
use crate::utilities::{
    barycentric_weights, lagrangian_derivative_interpolation, lagrangian_interpolation,
};
use nalgebra::*;
use rayon::prelude::*;

/// Selects the quadrature node family for one direction of a [`CollocationBasis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gauss {
    /// Gauss–Lobatto–Legendre nodes (include the endpoints; requires ≥ 2 nodes).
    Lobatto,
    /// Gauss–Legendre nodes (open; requires ≥ 1 node).
    Legendre,
}

///  Multi-dimensional GLL/GL operations on tensor-product grids.
///
/// Each direction $k$ is described by a pair `(n_k, node_type)`:
/// - `n_k`: number of quadrature nodes in direction $k$.
/// - `node_type`: [`Gauss::Lobatto`] for GLL nodes, [`Gauss::Legendre`] for GL nodes.
///
/// # Example
/// ```rust
/// use lobatto::collocation::{CollocationBasis, Gauss};
/// // 2D Q4 element: 5×5 GLL nodes on [0,1]^2
/// let basis = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto)]);
///
/// // DOF values: f(x, y) = x^2 + y^2 at the collocation points
/// let u: Vec<f64> = basis.points().iter().map(|p| p[0].powi(2) + p[1].powi(2)).collect();
///
/// // Differentiate along x
/// let mut du_dx = vec![0.0; u.len()];
/// basis.diff(0, &u, &mut du_dx);
///
/// // Evaluate the interpolant at an arbitrary point
/// let vals = basis.eval(&[vec![0.3, 0.7]], &u);
/// let expected = 0.3f64.powi(2) + 0.7f64.powi(2);
/// assert!((vals[0] - expected).abs() < 1e-13);
///
/// // Evaluate the interpolant derivative at an arbitrary point
/// let vals = basis.eval(&[vec![0.7, 0.3]], &du_dx);
/// let expected = 2.0*0.7;
/// assert!((vals[0] - expected).abs() < 1e-13);
/// ```
#[derive(Debug)]
pub struct CollocationBasis {
    /// Per-direction description: `(n_points, node_type)`.
    basis: Vec<(usize, Gauss)>,

    /// 1D quadrature weights on $\[0,1\]$, one vector per direction.
    weights_1d: Vec<Vec<f64>>,

    /// 1D quadrature nodes on $\[0,1\]$, one vector per direction.
    points_1d: Vec<Vec<f64>>,

    /// Tensor-product quadrature weights (one per flat DOF index).
    weights: Vec<f64>,

    /// Tensor-product quadrature nodes (one ND point per flat DOF index).
    points: Vec<Vec<f64>>,

    /// 1D differentiation matrices, one per direction.
    /// `diff_matrices[k][(j, i)]` $= L_i'(\xi_j)$ where $\xi_j$ are the nodes
    /// of direction $k$.
    diff_matrices: Vec<DMatrix<f64>>,

    /// Total number of DOFs: $\prod_k n_k$.
    n_points: usize,
    /// Prefix products of point counts: `n_points_acc[k]` $= n_0 \cdots n_k$.
    n_points_acc: Vec<usize>,

    /// Stride step used by [`CollocationBasis::diff`] for direction $k$.
    step: Vec<usize>,
    /// Starting flat indices of fibres in direction $k$ (all $p$ with $i_k(p) = 0$).
    start: Vec<Vec<usize>>,
}

impl CollocationBasis {
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

    /// Build a [`CollocationBasis`] from a list of per-direction descriptions `(n, node_type)`.
    ///
    /// `n` is the number of quadrature nodes and `node_type` selects [`Gauss::Lobatto`] (GLL)
    /// or [`Gauss::Legendre`] (GL) nodes on $\[0,1\]$.
    ///
    /// # Panics
    /// Panics if a GLL direction has fewer than 2 nodes, or a GL direction has 0 nodes.
    pub fn new(basis: Vec<(usize, Gauss)>) -> Self {
        let n = basis.len();

        let mut weights_1d = vec![vec![1.0]; n];
        let mut points_1d = vec![vec![0.5]; n];
        let mut diff_matrices = vec![DMatrix::<f64>::zeros(1, 1); n];

        for k in 0..n {
            assert!(
                (basis[k].0 > 1 && basis[k].1 == Gauss::Lobatto)
                    || (basis[k].0 > 0 && basis[k].1 == Gauss::Legendre),
                "not enough points to define the quadrature rule"
            );

            if basis[k].1 == Gauss::Lobatto {
                (points_1d[k], weights_1d[k]) = gauss_lobatto_on_unit_interval(basis[k].0);
            } else {
                (points_1d[k], weights_1d[k]) = gauss_legendre_on_unit_interval(basis[k].0);
            }

            diff_matrices[k].resize_mut(basis[k].0, basis[k].0, 0.0);

            // D_{ji} = L_i'(ξ_j)
            for i in 0..basis[k].0 {
                for j in 0..basis[k].0 {
                    diff_matrices[k][(j, i)] =
                        lagrangian_derivative_interpolation(&points_1d[k], points_1d[k][j], i);
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

        // Precompute tensor-product weights and points.
        let weights = (0..n_points)
            .map(|p| {
                let mi = Self::flat_to_multi(&n_points_acc, p);
                (0..n).map(|k| weights_1d[k][mi[k]]).product()
            })
            .collect();

        let points = (0..n_points)
            .map(|p| {
                let mi = Self::flat_to_multi(&n_points_acc, p);
                (0..n).map(|k| points_1d[k][mi[k]]).collect()
            })
            .collect();

        Self {
            basis,
            weights_1d,
            points_1d,
            weights,
            points,
            diff_matrices,
            n_points,
            n_points_acc,
            step,
            start,
        }
    }

    /// Return the number of spatial dimensions $d$.
    pub fn ndim(&self) -> usize {
        self.basis.len()
    }

    /// Return the total number of DOFs $N_\mathrm{dof} = \prod_{k=0}^{d-1} n_k$.
    pub fn n_dof(&self) -> usize {
        self.n_points
    }

    /// Return the number of quadrature nodes in direction `dim`.
    pub fn n_nodes(&self, dim: usize) -> usize {
        self.basis[dim].0
    }

    /// Return the node type ([`Gauss::Lobatto`] or [`Gauss::Legendre`]) for direction `dim`.
    pub fn node_type(&self, dim: usize) -> Gauss {
        self.basis[dim].1
    }

    /// Return the tensor-product quadrature nodes.
    ///
    /// Entry $p$ is the ND point $(\xi^{(0)}_{i_0(p)}, \ldots, \xi^{(d-1)}_{i_{d-1}(p)})$.
    pub fn points(&self) -> &[Vec<f64>] {
        &self.points
    }

    /// Return the 1D quadrature nodes on $\[0,1\]$ for direction `dim`.
    pub fn points_1d(&self, dim: usize) -> &[f64] {
        &self.points_1d[dim]
    }

    /// Return the 1D quadrature weights on $\[0,1\]$ for direction `dim`.
    pub fn weights_1d(&self, dim: usize) -> &[f64] {
        &self.weights_1d[dim]
    }

    /// Return the raw 1D differentiation matrix for direction `dim`.
    ///
    /// Entry $(j, i)$ equals $L_i'(\xi_j)$.  For the full tensor-product lift
    /// see [`CollocationBasis::diff_matrix`].
    pub fn diff_matrix_1d(&self, dim: usize) -> &DMatrix<f64> {
        &self.diff_matrices[dim]
    }

    /// Apply the differentiation operator along direction `dim` in place.
    /// The matrix $D^{(\mathrm{dim})}$ is applied fibre-by-fibre over the tensor-product grid:
    /// for each combination of indices $(i_0, \ldots, {i_{\mathrm{dim}}}, \ldots, i_{d-1})$,
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

    /// Return the tensor-product quadrature weights.
    ///
    /// Entry $p$ equals $w_p = \prod_{k=0}^{d-1} w^{(k)}_{i_k(p)}$.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Return the quadrature weights as a dense $N_\mathrm{dof} \times N_\mathrm{dof}$ matrix.
    ///
    /// All off-diagonal entries are zero; diagonal entries are given by [`weights`](Self::weights).
    pub fn weights_matrix(&self) -> DMatrix<f64> {
        let diag = self.weights();
        let mut weights_matrix = DMatrix::<f64>::zeros(self.n_points, self.n_points);
        for (p, w) in diag.iter().enumerate() {
            weights_matrix[(p, p)] = *w;
        }
        weights_matrix
    }

    /// Return the full tensor-product differentiation matrix along direction `dim` as a
    /// dense $N_\mathrm{dof} \times N_\mathrm{dof}$ matrix assembled via
    /// the same operation as [`CollocationBasis::diff`].
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

    /// Evaluate the collocation interpolant of `u` at arbitrary points in $\[0,1\]^d$.
    ///
    /// `u` holds the DOF values at the collocation points (flat, direction-0 fastest).
    /// Each query point must have `d` coordinates. Uses sum factorisation:
    /// $$u_h(\xi) = \sum_{j_0,\ldots,j_{d-1}} u_{j_0,\ldots,j_{d-1}}
    ///              \prod_{k=0}^{d-1} L^{(k)}_{j_k}(\xi^{(k)})$$
    ///
    /// # Panics
    /// Panics if any coordinate lies outside $[0, 1]$.
    ///
    /// # Example
    /// ```rust
    /// use lobatto::collocation::{CollocationBasis, Gauss};
    ///
    /// // 2D Q2 basis: 3×3 GLL nodes on [0,1]^2
    /// let basis = CollocationBasis::new(vec![(3, Gauss::Lobatto), (3, Gauss::Lobatto)]);
    ///
    /// // f(x, y) = x^2 + y^2 — degree 2, reproduced exactly by the basis
    /// let u: Vec<f64> = basis.points().iter().map(|p| p[0].powi(2) + p[1].powi(2)).collect();
    ///
    /// let query = vec![vec![0.3, 0.7], vec![0.0, 1.0]];
    /// let vals = basis.eval(&query, &u);
    ///
    /// assert!((vals[0] - (0.3f64.powi(2) + 0.7f64.powi(2))).abs() < 1e-14);
    /// assert!((vals[1] - (0.0f64.powi(2) + 1.0f64.powi(2))).abs() < 1e-14);
    /// ```
    pub fn eval(&self, xs: &[Vec<f64>], u: &[f64]) -> Vec<f64> {
        let ndim = self.basis.len();
        xs.par_iter()
            .map(|x| {
                assert_eq!(x.len(), ndim, "point dimension mismatch");

                // Per-direction Lagrange basis values via barycentric interpolation.
                let phi: Vec<Vec<f64>> = (0..ndim)
                    .map(|d| {
                        assert!(
                            (0.0..=1.0).contains(&x[d]),
                            "x[{d}] = {} is outside [0, 1]",
                            x[d]
                        );
                        let bary = barycentric_weights(&self.points_1d[d]);
                        lagrangian_interpolation(&self.points_1d[d], &bary, x[d])
                    })
                    .collect();

                // Sum-factorised evaluation: contract one direction at a time,
                // direction 0 varies fastest.
                let mut v = u.to_vec();
                for d in 0..ndim {
                    let n_d = self.basis[d].0;
                    let rest = v.len() / n_d;
                    let mut v_next = vec![0.0_f64; rest];
                    for s in 0..rest {
                        for j in 0..n_d {
                            v_next[s] += phi[d][j] * v[s * n_d + j];
                        }
                    }
                    v = v_next;
                }
                v[0]
            })
            .collect()
    }
}
