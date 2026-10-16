//! Grid geometry: [`Extension`], [`Grid1D`] and [`GridND`].
//!
//! A grid describes a multi-element Gauss-Lobatto discretisation of an
//! arbitrary interval `[x_l, x_l + L]` (1D) or tensor-product domain
//! $\prod_d [x_{l,d}, x_{l,d} + L_d]$ (ND): number of elements, polynomial
//! order, domain, and the [`Extension`] that encodes how the discretisation
//! behaves at the boundary. The `[0, 1]` reference element used for the
//! Gauss-Lobatto nodes and basis evaluation is an internal detail only.

use lobatto::collocation::{CollocationBasis, Gauss};
use lobatto::utilities::barycentric_weights;
use rayon::prelude::*;

// ── Extension ────────────────────────────────────────────────────────────────

/// Boundary extension of a [`Grid1D`].
///
/// Encodes how the field represented on the grid extends beyond the physical
/// domain `[x_l, x_l + L]`, which in turn fixes the DOF layout and count:
///
/// | Variant    | BC          | `N_dof` |
/// |------------|-------------|------------------|
/// | `Periodic` | Periodic    | `n·r`            |
/// | `Odd`      | Dirichlet   | `n·r - 1`        |
/// | `Even`     | Neumann     | `n·r + 1`        |
///
/// where `n` is the number of elements and `r` the polynomial order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension {
    /// Periodic identification: `u(x_l) = u(x_l + L)`; no reflection needed.
    Periodic,
    /// Odd (anti-symmetric) reflection — enforces homogeneous Dirichlet values at both endpoints.
    Odd,
    /// Even (symmetric) reflection — enforces zero normal derivative (Neumann) at both endpoints.
    Even,
}

// ── 1D grid ─────────────────────────────────────────────────────────────────

/// 1D multi-element Gauss-Lobatto grid on `[x_l, x_l + L]`.
///
/// `n` elements of polynomial order `r` (`r + 1` Gauss-Lobatto nodes per
/// element, endpoints shared between neighbours), with an [`Extension`] that
/// fixes the boundary behaviour and the DOF layout.
///
/// ## Data layout
///
/// The **physical DOF buffer** (length `N_dof`) is in `x`-increasing order:
///
/// | Extension   | Flat index of node `j` in element `k` |
/// |-------------|---------------------------------------|
/// | `Periodic`  | `k·r + j`                             |
/// | `Odd`       | `k·r + j - 1` (left boundary excluded) |
/// | `Even`      | `k·r + j`, plus the right endpoint at `n·r` |
#[derive(Debug, Clone, PartialEq)]
pub struct Grid1D {
    /// Number of elements `n`.
    pub(crate) n: usize,
    /// Polynomial order `r` (nodes per element, excluding the shared endpoint).
    pub(crate) r: usize,
    /// Domain length `L`.
    pub(crate) l: f64,
    /// Left boundary position `x_l`.
    pub(crate) xl: f64,
    /// Number of physical DOFs `N_dof`.
    pub(crate) n_dof: usize,
    /// Boundary extension.
    pub(crate) extension: Extension,
    /// Gauss-Lobatto nodes on the `[0, 1]` reference element (length `r + 1`).
    pub(crate) gauss_lobatto_points: Vec<f64>,
    /// Gauss-Lobatto quadrature weights on the `[0, 1]` reference element (length `r + 1`).
    pub(crate) gauss_lobatto_weights: Vec<f64>,
    /// Barycentric weights for the `r + 1` Gauss-Lobatto nodes.
    pub(crate) bary_weights: Vec<f64>,
    /// Reference differentiation matrix on `[0, 1]`, `(r + 1) × (r + 1)`
    /// column-major: entry `j + (r + 1)·i` is `ℓ_i'(ξ_j)`.
    pub(crate) diff_matrix: Vec<f64>,
}

impl Grid1D {
    /// Create a grid of `n` elements of order `r` on `[x_l, x_l + L]`.
    pub fn new(n: usize, r: usize, l: f64, xl: f64, extension: Extension) -> Self {
        let n_dof = match extension {
            Extension::Periodic => r * n,
            Extension::Odd => r * n - 1,
            Extension::Even => r * n + 1,
        };
        let basis = CollocationBasis::new(vec![(r + 1, Gauss::Lobatto)]);
        let gauss_lobatto_points = basis.points_1d(0).to_vec();
        let gauss_lobatto_weights = basis.weights_1d(0).to_vec();
        let bary_weights = barycentric_weights(&gauss_lobatto_points);
        let diff_matrix = basis.diff_matrix_1d(0).as_slice().to_vec();
        Self {
            n,
            r,
            l,
            xl,
            n_dof,
            extension,
            gauss_lobatto_points,
            gauss_lobatto_weights,
            bary_weights,
            diff_matrix,
        }
    }

    /// Number of elements `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Polynomial order `r`.
    pub fn r(&self) -> usize {
        self.r
    }

    /// Domain length `L`.
    pub fn l(&self) -> f64 {
        self.l
    }

    /// Left boundary position `x_l`.
    pub fn xl(&self) -> f64 {
        self.xl
    }

    /// Element width `h = L/n`.
    pub fn h(&self) -> f64 {
        self.l / (self.n as f64)
    }

    /// Number of physical DOFs `N_dof`.
    pub fn n_dof(&self) -> usize {
        self.n_dof
    }

    /// Boundary extension.
    pub fn extension(&self) -> Extension {
        self.extension
    }

    /// Gauss-Lobatto nodes on the `[0, 1]` reference element (length `r + 1`).
    pub fn ref_points(&self) -> &[f64] {
        &self.gauss_lobatto_points
    }

    /// Gauss-Lobatto quadrature weights on the `[0, 1]` reference element (length `r + 1`).
    pub fn ref_weights(&self) -> &[f64] {
        &self.gauss_lobatto_weights
    }

    /// Diagonal of the lumped mass matrix: one quadrature weight per stored DOF.
    ///
    /// The reference weights $\hat w_j$ are scaled by the element width
    /// `h = L/n` and assembled over the elements, so that
    /// $\sum_i w_i\, u_i \approx \int u\,dx$ for a field `u` given by its
    /// stored DOFs (length `N_dof`, in the order of [`get_x`](Self::get_x)):
    ///
    /// | Node | Weight |
    /// |------|--------|
    /// | interior to an element | `h·ŵ_j` |
    /// | shared by two elements, periodic seam included | sum of the two end weights |
    /// | stored boundary endpoint (`Even`) | its single end weight |
    /// | unstored boundary endpoint (`Odd`) | dropped |
    pub fn mass_weights(&self) -> Vec<f64> {
        let h = self.h();
        let mut w = vec![0.0; self.n_dof];
        for k in 0..self.n {
            for (j, &wj) in self.gauss_lobatto_weights.iter().enumerate() {
                let i = self.eval_dof_index(k, j);
                if i < self.n_dof {
                    w[i] += h * wj;
                }
            }
        }
        w
    }

    /// Gauss-Lobatto nodes in `x`-increasing order on `[x_l, x_l + L]`.
    ///
    /// - `Periodic`: `n·r` nodes — left endpoint included, right excluded.
    /// - `Odd`:      `n·r - 1` nodes — both boundary endpoints excluded.
    /// - `Even`:     `n·r + 1` nodes — both boundary endpoints included.
    pub fn get_x(&self) -> Vec<f64> {
        let h = self.h();
        let mut x = Vec::with_capacity(self.n_dof);

        match self.extension {
            Extension::Periodic => {
                for k in 0..self.n {
                    for j in 0..self.r {
                        x.push(self.xl + h * ((k as f64) + self.gauss_lobatto_points[j]));
                    }
                }
                x
            }
            Extension::Odd => {
                // Skip xl (k=0,j=0) and xl+l (k=n-1,j=r): both are Dirichlet zeros.
                for j in 1..self.r {
                    x.push(self.xl + h * self.gauss_lobatto_points[j]);
                }
                for k in 1..self.n {
                    for j in 0..self.r {
                        x.push(self.xl + h * ((k as f64) + self.gauss_lobatto_points[j]));
                    }
                }
                x
            }
            Extension::Even => {
                // Element 0: all r+1 nodes; elements 1..n: skip shared left endpoint.
                for j in 0..=self.r {
                    x.push(self.xl + h * self.gauss_lobatto_points[j]);
                }
                for k in 1..self.n {
                    for j in 1..=self.r {
                        x.push(self.xl + h * ((k as f64) + self.gauss_lobatto_points[j]));
                    }
                }
                x
            }
        }
    }

    /// Fold `x` into `[x_l, x_l + L]` according to the extension, returning the
    /// folded coordinate and the sign of the extended interpolant there
    /// (`-1` on odd-reflected copies, `+1` otherwise).
    pub fn fold_extended(&self, x: f64) -> (f64, f64) {
        match self.extension {
            Extension::Periodic => ((x - self.xl).rem_euclid(self.l) + self.xl, 1.0),
            Extension::Odd | Extension::Even => {
                let t = (x - self.xl).rem_euclid(2.0 * self.l);
                if t <= self.l {
                    (self.xl + t, 1.0)
                } else {
                    let sign = match self.extension {
                        Extension::Odd => -1.0,
                        _ => 1.0,
                    };
                    (self.xl + (2.0 * self.l - t), sign)
                }
            }
        }
    }

    /// Returns the flat DOF index for local node `j` (0..=r) of element `k` (0..n).
    ///
    /// - `Periodic`: right endpoint (j==r) wraps to next element's j=0.
    /// - `Odd`: left boundary (k=0,j=0) and right boundary (k=n-1,j=r) are Dirichlet zeros,
    ///   returned as `n_dof` (sentinel for "zero, not stored").
    ///   Otherwise: flat index `k*r + j - 1`.
    /// - `Even`: flat index `k*r + j`.
    pub fn eval_dof_index(&self, k: usize, j: usize) -> usize {
        match self.extension {
            Extension::Periodic => {
                if j < self.r {
                    k * self.r + j
                } else {
                    ((k + 1) % self.n) * self.r
                }
            }
            Extension::Odd => {
                if (k == 0 && j == 0) || (k == self.n - 1 && j == self.r) {
                    self.n_dof // sentinel: zero boundary
                } else {
                    k * self.r + j - 1
                }
            }
            Extension::Even => k * self.r + j,
        }
    }

    /// ∂u/∂x at the Gauss–Lobatto points; a node shared by
    /// two elements (the periodic seam included) takes the average of the
    /// two; an `Even` boundary node its one element's value; the unstored
    /// `Odd` boundary zeros enter as zeros.
    ///
    /// `u` and `out` are stored DOFs (length `N_dof`, in the order of
    /// [`get_x`](Self::get_x)). Non-assembled: no global matrix is formed;
    /// each element gathers its `r + 1` nodal values, applies the reference
    /// differentiation matrix scaled by `1/h`, and adds the result into
    /// `out`, with weight `1/2` at the nodes it shares with a neighbour.
    /// Cost `n·(r + 1)²`.
    ///
    /// # Panics
    ///
    /// If `u` or `out` does not have length `N_dof`.
    pub fn derivative(&self, u: &[f64], out: &mut [f64]) {
        assert_eq!(u.len(), self.n_dof, "u length differs from N_dof");
        assert_eq!(out.len(), self.n_dof, "out length differs from N_dof");
        let (n, r) = (self.n, self.r);
        let inv_h = 1.0 / self.h();
        let mut ul = vec![0.0; r + 1];
        out.fill(0.0);
        for k in 0..n {
            for (j, v) in ul.iter_mut().enumerate() {
                let i = self.eval_dof_index(k, j);
                *v = if i < self.n_dof { u[i] } else { 0.0 };
            }
            for j in 0..=r {
                let i = self.eval_dof_index(k, j);
                if i == self.n_dof {
                    continue;
                }
                // The element end nodes are shared, except the domain
                // endpoints of a non-periodic grid.
                let shared = (j == 0 && (k > 0 || self.extension == Extension::Periodic))
                    || (j == r && (k < n - 1 || self.extension == Extension::Periodic));
                let weight = if shared { 0.5 * inv_h } else { inv_h };
                let du: f64 = (0..=r).map(|m| self.diff_matrix[j + (r + 1) * m] * ul[m]).sum();
                out[i] += weight * du;
            }
        }
    }
}

// ── ND grid ─────────────────────────────────────────────────────────────────

/// `N`-dimensional tensor-product Gauss-Lobatto grid.
///
/// One [`Grid1D`] per spatial direction, each with an independent
/// [`Extension`]. The DOF tensor is stored flat in column-major order
/// (direction 0 varies fastest) with `total = ∏_d n_dof_d` entries.
#[derive(Debug, Clone, PartialEq)]
pub struct GridND<const N: usize> {
    /// One 1D grid per spatial direction.
    pub(crate) grids: [Grid1D; N],
    /// DOF counts per direction: `ndofs[d] = grids[d].n_dof()`.
    pub(crate) ndofs: [usize; N],
    /// Total DOFs: `∏_d ndofs[d]`.
    pub(crate) total: usize,
    /// Column-major strides for the DOF tensor.
    pub(crate) strides: [usize; N],
}

impl<const N: usize> GridND<N> {
    /// Create an ND grid for a tensor-product domain.
    ///
    /// - `ns[d]`:   number of elements in direction `d`.
    /// - `rs[d]`:   polynomial order (local nodes per element) in direction `d`.
    /// - `ls[d]`:   domain length in direction `d`.
    /// - `xls[d]`:  left boundary in direction `d`.
    /// - `exts[d]`: [`Extension`] for direction `d`.
    pub fn new(
        ns: [usize; N],
        rs: [usize; N],
        ls: [f64; N],
        xls: [f64; N],
        exts: [Extension; N],
    ) -> Self {
        Self::from_grids(std::array::from_fn(|d| {
            Grid1D::new(ns[d], rs[d], ls[d], xls[d], exts[d])
        }))
    }

    /// Create an ND grid from per-direction 1D grids.
    pub fn from_grids(grids: [Grid1D; N]) -> Self {
        let ndofs: [usize; N] = std::array::from_fn(|d| grids[d].n_dof);
        let total = ndofs.iter().product();
        let strides = nd_col_major_strides(&ndofs);
        Self {
            grids,
            ndofs,
            total,
            strides,
        }
    }

    /// The 1D grid of direction `d`.
    pub fn axis(&self, d: usize) -> &Grid1D {
        &self.grids[d]
    }

    /// Per-direction 1D grids.
    pub fn grids(&self) -> &[Grid1D; N] {
        &self.grids
    }

    /// DOF counts per direction.
    pub fn ndofs(&self) -> [usize; N] {
        self.ndofs
    }

    /// Total DOFs: `∏_d ndofs[d]`.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Column-major strides for the DOF tensor (direction 0 varies fastest).
    pub fn strides(&self) -> [usize; N] {
        self.strides
    }

    /// Diagonal of the tensor-product lumped mass matrix (length `total`).
    ///
    /// Entry `p` is the product of the per-direction
    /// [`Grid1D::mass_weights`] at the multi-index of `p`, in the column-major
    /// DOF order (direction 0 fastest), so that $\sum_p w_p\, u_p$
    /// approximates the integral of `u` over the box.
    pub fn mass_weights(&self) -> Vec<f64> {
        let w1d: [Vec<f64>; N] = std::array::from_fn(|d| self.grids[d].mass_weights());
        (0..self.total)
            .map(|p| {
                (0..N)
                    .map(|d| w1d[d][(p / self.strides[d]) % self.ndofs[d]])
                    .product()
            })
            .collect()
    }

    /// Tensor-product Gauss-Lobatto grid in x-increasing column-major order.
    ///
    /// Returns `Vec<[f64; N]>` of length `total = ∏_d ndofs[d]`.
    /// `result[p][d]` is the coordinate of point `p` in direction `d`.
    /// Direction 0 varies fastest; within each direction nodes are x-increasing.
    pub fn get_x(&self) -> Vec<[f64; N]> {
        let x1d: [Vec<f64>; N] = std::array::from_fn(|d| self.grids[d].get_x());
        (0..self.total)
            .map(|p| std::array::from_fn(|d| x1d[d][(p / self.strides[d]) % self.ndofs[d]]))
            .collect()
    }

    /// ∂u/∂x_d of a nodal field `u` (column-major, direction 0 fastest),
    /// written into `out` (same length): `Grid1D::derivative` of direction
    /// `d` along every fibre of that direction.
    ///
    /// Parallel over the fibres, without `unsafe`. Direction 0 has contiguous
    /// fibres and is differentiated in place in `out`. For `d > 0` the fibres
    /// are strided: each one is gathered and differentiated into a contiguous
    /// row of a temporary (fibre-major, length `total`), which is then
    /// transposed into `out` one contiguous run of direction-`d` index at a
    /// time.
    ///
    /// # Panics
    ///
    /// If `d ≥ N`, or `u` or `out` does not have length `total`.
    pub fn derivative(&self, d: usize, u: &[f64], out: &mut [f64]) {
        assert_eq!(u.len(), self.total, "u length differs from total");
        assert_eq!(out.len(), self.total, "out length differs from total");
        let g = &self.grids[d];
        let (m, s) = (self.ndofs[d], self.strides[d]);
        if s == 1 {
            out.par_chunks_mut(m)
                .zip(u.par_chunks(m))
                .for_each(|(o, f)| g.derivative(f, o));
            return;
        }
        let mut tmp = vec![0.0; self.total];
        tmp.par_chunks_mut(m)
            .enumerate()
            .for_each_init(
                || vec![0.0; m],
                |fibre, (f, o)| {
                    let base = nd_fiber_base(f, d, &self.ndofs, &self.strides);
                    for (j, v) in fibre.iter_mut().enumerate() {
                        *v = u[base + s * j];
                    }
                    g.derivative(fibre, o);
                },
            );
        // Run c = j + m·i_high of out is out[i_low + s·c], i_low = 0..s.
        out.par_chunks_mut(s).enumerate().for_each(|(c, o)| {
            let (j, i_high) = (c % m, c / m);
            for (i_low, v) in o.iter_mut().enumerate() {
                *v = tmp[(i_low + s * i_high) * m + j];
            }
        });
    }
}

// ── ND helpers ───────────────────────────────────────────────────────────────

/// Compute column-major strides for a fixed-size shape array.
///
/// Returns `st` with `st[0] = 1` and `st[d] = st[d-1] * sizes[d-1]` for `d ≥ 1`,
/// so that flat index `Σ_d i_d * st[d]` maps the multi-index `(i_0, …, i_{N-1})`
/// with direction 0 varying fastest.
pub fn nd_col_major_strides<const N: usize>(sizes: &[usize; N]) -> [usize; N] {
    let mut st = [1usize; N];
    for d in 1..N {
        st[d] = st[d - 1] * sizes[d - 1];
    }
    st
}

/// Flat start index of the `s`-th fibre along direction `d` of a
/// column-major tensor of shape `sizes` with strides `strides` (as returned
/// by [`GridND::strides`]).
///
/// `s ∈ 0..total / sizes[d]` enumerates the index combinations of the other
/// directions in column-major order (direction 0 fastest, `d` skipped); the
/// fibre is `base + j·strides[d]` for `j ∈ 0..sizes[d]`. For a fixed `d`,
/// distinct `s` give disjoint fibres covering the whole tensor.
///
/// Closed form: the directions before `d` form one index
/// `s % strides[d]` (stride 1), those after `d` one index `s / strides[d]`
/// (stride `strides[d]·sizes[d]`), so the cost is independent of `N`.
/// Requires column-major strides, `strides[d] = Π_{d' < d} sizes[d']`
/// (checked in debug builds).
pub fn nd_fiber_base(s: usize, d: usize, sizes: &[usize], strides: &[usize]) -> usize {
    debug_assert_eq!(strides[d], sizes[..d].iter().product::<usize>());
    let sd = strides[d];
    s % sd + sd * sizes[d] * (s / sd)
}
