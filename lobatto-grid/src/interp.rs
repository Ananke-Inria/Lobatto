//! Interpolation of grid DOF vectors: point evaluation, extension folding,
//! convection, and precomputed evaluation plans.
//!
//! This module gathers the interpolation routines of [`Grid1D`] (1D) and
//! [`GridND`] (ND), plus [`EvalPlan`]: a precomputed plan for the
//! configuration where the same set of evaluation points (typically a fixed
//! flow map `lambda` in [`GridND::convect`]) is used many times with varying
//! DOF data `u`.

use crate::grid::{nd_col_major_strides, Extension, Grid1D, GridND};
use lobatto::utilities::{lagrangian_interpolation, lagrangian_interpolation_into};
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::{AddAssign, Mul, MulAssign};

/// Scalar types the interpolation routines operate on.
///
/// Interpolation never mixes real and imaginary parts — every operation is
/// "scalar × f64" or "scalar += scalar" — so the routines are generic over the
/// DOF scalar. Implemented in particular by `f64` (real problems, half the
/// memory traffic and flops) and `Complex<f64>`.
pub trait Scalar:
    Copy + Default + Send + Sync + AddAssign<Self> + Mul<f64, Output = Self> + MulAssign<f64>
{
}

impl<T> Scalar for T where
    T: Copy + Default + Send + Sync + AddAssign<Self> + Mul<f64, Output = Self> + MulAssign<f64>
{
}

// ─── 1D interpolation ─────────────────────────────────────────────────────────

impl Grid1D {
    /// Evaluate the interpolant of `u` at a single position `x`.
    ///
    /// `u` must be in the x-increasing layout matching [`get_x`](Self::get_x).
    /// `x` must lie in `[xl, xl+l]` for **every** extension; use
    /// [`eval_extended_at`](Self::eval_extended_at) to evaluate the underlying
    /// extension at positions outside the domain.
    ///
    /// Batch variant with a caller-provided output buffer: [`eval`](Self::eval).
    pub fn eval_at<T: Scalar>(&self, x: f64, u: &[T]) -> T {
        assert!(
            x >= self.xl && x <= self.xl + self.l,
            "eval: x={x} outside [{}, {}]",
            self.xl,
            self.xl + self.l
        );
        let h = self.l / (self.n as f64);
        let k = (((x - self.xl) / h) as usize).min(self.n - 1);
        let xi = (x - self.xl - (k as f64) * h) / h;
        let phi = lagrangian_interpolation(&self.gauss_lobatto_points, &self.bary_weights, xi);

        let mut val = T::default();
        match self.extension {
            Extension::Periodic => {
                for j in 0..self.r {
                    val += u[k * self.r + j] * phi[j];
                }
                val += u[((k + 1) % self.n) * self.r] * phi[self.r];
            }
            Extension::Odd => {
                // Boundary zeros (k=0,j=0) and (k=n-1,j=r) not stored; flat index k*r+j−1.
                for j in 0..=self.r {
                    let v = if (k == 0 && j == 0) || (k == self.n - 1 && j == self.r) {
                        T::default()
                    } else {
                        u[k * self.r + j - 1]
                    };
                    val += v * phi[j];
                }
            }
            Extension::Even => {
                for j in 0..=self.r {
                    val += u[k * self.r + j] * phi[j];
                }
            }
        }
        val
    }

    /// Evaluate the interpolant of `u` at a batch of positions.
    ///
    /// - `xs`: query points (length `m`); every `xs[i]` must lie in `[xl, xl+l]`,
    ///   for **every** extension; use [`eval_extended`](Self::eval_extended) to
    ///   evaluate the underlying extension at positions outside the domain.
    /// - `u`: DOF vector in the x-increasing layout matching [`get_x`](Self::get_x).
    /// - `fu`: output buffer of length `m`; `fu[i]` receives the interpolated value at `xs[i]`.
    pub fn eval<T: Scalar>(&self, xs: &[f64], u: &[T], fu: &mut [T]) {
        debug_assert_eq!(fu.len(), xs.len());
        fu.par_iter_mut()
            .zip(xs.par_iter())
            .for_each(|(f, &x)| *f = self.eval_at(x, u));
    }

    /// Evaluate the extension of the interpolant of `u` at a single position `x`.
    ///
    /// Unlike [`eval_at`](Self::eval_at), `x` may lie anywhere: it is folded back into
    /// `[xl, xl+l]` according to the extension — periodic wrap (`Periodic`),
    /// even reflection (`Even`/Neumann), or odd reflection with a sign flip
    /// (`Odd`/Dirichlet) — so the result is consistent with the underlying
    /// extended representation.
    ///
    /// Batch variant with a caller-provided output buffer:
    /// [`eval_extended`](Self::eval_extended).
    pub fn eval_extended_at<T: Scalar>(&self, x: f64, u: &[T]) -> T {
        let (y, sign) = self.fold_extended(x);
        self.eval_at(y, u) * sign
    }

    /// Evaluate the extension of the interpolant of `u` at a batch of positions.
    ///
    /// Unlike [`eval`](Self::eval), positions may lie anywhere: each is folded
    /// back into `[xl, xl+l]` according to the extension — periodic wrap
    /// (`Periodic`), even reflection (`Even`/Neumann), or odd reflection with a
    /// sign flip (`Odd`/Dirichlet) — so the result is consistent with the
    /// underlying extended representation.
    ///
    /// - `xs`: query points (length `m`).
    /// - `u`: DOF vector in the x-increasing layout matching [`get_x`](Self::get_x).
    /// - `fu`: output buffer of length `m`; `fu[i]` receives the value at `xs[i]`.
    pub fn eval_extended<T: Scalar>(&self, xs: &[f64], u: &[T], fu: &mut [T]) {
        debug_assert_eq!(fu.len(), xs.len());
        fu.par_iter_mut()
            .zip(xs.par_iter())
            .for_each(|(f, &x)| *f = self.eval_extended_at(x, u));
    }

    /// Apply a flow map `lambda` to every grid node and interpolate `u` at the resulting positions.
    ///
    /// Positions outside `[xl, xl+l]` are handled by [`eval_extended`](Self::eval_extended),
    /// i.e. folded back according to the extension.
    ///
    /// - `u`: DOF vector in the x-increasing layout matching [`get_x`](Self::get_x).
    /// - `out`: output buffer of the same length as `u`; receives the convected values.
    pub fn convect<T: Scalar, F>(&self, lambda: F, u: &[T], out: &mut [T])
    where
        F: Fn(f64) -> f64 + Sync,
    {
        debug_assert_eq!(out.len(), u.len());
        let xs = self.get_x();
        out.par_iter_mut()
            .zip(xs.par_iter())
            .for_each(|(o, &x)| *o = self.eval_extended_at(lambda(x), u));
    }
}

// ─── ND geometry helpers ──────────────────────────────────────────────────────

/// Per-element, per-point geometry shared by [`GridND::eval`] and the plan
/// construction/application: local node counts, strides, offsets into the flat
/// basis-value scratch buffer.
struct EvalGeometry<const N: usize> {
    ns: [usize; N],
    ls: [f64; N],
    xls: [f64; N],
    hs: [f64; N],
    counts: [usize; N],
    count_strides: [usize; N],
    local_total: usize,
    n_strides: [usize; N],
    phi_off: [usize; N],
    phi_len: usize,
}

impl<const N: usize> EvalGeometry<N> {
    fn new(grid: &GridND<N>) -> Self {
        let ns: [usize; N] = std::array::from_fn(|d| grid.grids[d].n);
        let rs: [usize; N] = std::array::from_fn(|d| grid.grids[d].r);
        let ls: [f64; N] = std::array::from_fn(|d| grid.grids[d].l);
        let xls: [f64; N] = std::array::from_fn(|d| grid.grids[d].xl);
        let hs: [f64; N] = std::array::from_fn(|d| ls[d] / ns[d] as f64);
        let counts: [usize; N] = std::array::from_fn(|d| rs[d] + 1);
        let count_strides = nd_col_major_strides(&counts);
        let local_total: usize = counts.iter().product();

        // DOF strides in the "periodic-equivalent" layout (n_d * r_d per direction).
        // For Odd/Even, the actual layout differs but the same formula holds because
        // the eval indexing accounts for BCs via the per-direction eval logic.
        let n_strides = nd_col_major_strides(&ns);

        let mut phi_off = [0usize; N];
        for d in 1..N {
            phi_off[d] = phi_off[d - 1] + counts[d - 1];
        }
        let phi_len: usize = counts.iter().sum();

        Self {
            ns,
            ls,
            xls,
            hs,
            counts,
            count_strides,
            local_total,
            n_strides,
            phi_off,
            phi_len,
        }
    }

    /// Flat element index of point `x`, asserting each coordinate is in-domain.
    fn elem_index(&self, x: &[f64; N]) -> usize {
        (0..N).fold(0, |acc, d| {
            assert!(
                x[d] >= self.xls[d] && x[d] <= self.xls[d] + self.ls[d],
                "eval: x[{d}]={} outside [{}, {}]",
                x[d],
                self.xls[d],
                self.xls[d] + self.ls[d]
            );
            let k_d = (((x[d] - self.xls[d]) / self.hs[d]) as usize).min(self.ns[d] - 1);
            acc + k_d * self.n_strides[d]
        })
    }

    /// Per-direction element indices of flat element index `k_flat`.
    fn elem_multi_index(&self, k_flat: usize) -> [usize; N] {
        std::array::from_fn(|d| (k_flat / self.n_strides[d]) % self.ns[d])
    }

    /// Local coordinate of `x` in element `ks`, normalised to `[0, 1]` per direction.
    fn local_coords(&self, x: &[f64; N], ks: &[usize; N]) -> [f64; N] {
        std::array::from_fn(|d| (x[d] - self.xls[d] - (ks[d] as f64) * self.hs[d]) / self.hs[d])
    }
}

impl<const N: usize> GridND<N> {
    /// Gather `u` into a contiguous local patch for element `ks`.
    ///
    /// Each direction contributes `dof_d * stride_d`; if any direction returns
    /// the sentinel (`n_dof` = Dirichlet zero), the value is 0.
    fn gather_local<T: Scalar>(
        &self,
        geo: &EvalGeometry<N>,
        ks: &[usize; N],
        u: &[T],
    ) -> Vec<T> {
        (0..geo.local_total)
            .map(|p| {
                let mut global = 0usize;
                let mut is_zero = false;
                for (d, &k_d) in ks.iter().enumerate() {
                    let j = (p / geo.count_strides[d]) % geo.counts[d];
                    let dof_d = self.grids[d].eval_dof_index(k_d, j);
                    if dof_d == self.grids[d].n_dof {
                        is_zero = true;
                        break;
                    }
                    global += dof_d * self.strides[d];
                }
                if is_zero {
                    T::default()
                } else {
                    u[global]
                }
            })
            .collect()
    }

    /// Fill the flat basis-value buffer `phi` for local coordinates `xi`.
    fn fill_phi(&self, geo: &EvalGeometry<N>, xi: &[f64; N], phi: &mut [f64]) {
        for d in 0..N {
            lagrangian_interpolation_into(
                &self.grids[d].gauss_lobatto_points,
                &self.grids[d].bary_weights,
                xi[d],
                &mut phi[geo.phi_off[d]..geo.phi_off[d] + geo.counts[d]],
            );
        }
    }
}

/// Sum-factored contraction of the local patch `u_local` with the basis values
/// in `phi` (direction `d` at offset `phi_off[d]`), using `v` as scratch of
/// length `local_total / counts[N-1]`.
///
/// The slowest-varying direction is contracted first: every stage is then a
/// j-outer / s-inner loop over contiguous memory with independent
/// accumulations across `s` — SIMD-friendly and free of the per-point FMA
/// dependency chains a j-inner loop would create.
///
/// Stages run in place in `v`: stage `d` writes `v[0..len]` (its j=0 chunk)
/// while reading only `v[j*len + s]` with `j >= 1`, which are untouched.
fn contract<T: Scalar, const N: usize>(
    geo: &EvalGeometry<N>,
    phi: &[f64],
    u_local: &[T],
    v: &mut [T],
) -> T {
    // First stage (d = N-1) reads from u_local into v.
    let c_last = geo.counts[N - 1];
    let off_last = geo.phi_off[N - 1];
    let mut len = geo.local_total / c_last;
    let w0 = phi[off_last];
    for s in 0..len {
        v[s] = u_local[s] * w0;
    }
    for j in 1..c_last {
        let w = phi[off_last + j];
        let base = j * len;
        for s in 0..len {
            let t = u_local[base + s] * w;
            v[s] += t;
        }
    }
    // Remaining stages (d = N-2 .. 0) run in place in v.
    for d in (0..N - 1).rev() {
        let c_d = geo.counts[d];
        let off = geo.phi_off[d];
        len /= c_d;
        let w0 = phi[off];
        for x in v[..len].iter_mut() {
            *x *= w0;
        }
        for j in 1..c_d {
            let w = phi[off + j];
            let base = j * len;
            for s in 0..len {
                let t = v[base + s] * w;
                v[s] += t;
            }
        }
    }
    v[0]
}

// ─── ND interpolation ─────────────────────────────────────────────────────────

impl<const N: usize> GridND<N> {
    /// Evaluate the ND interpolant of `u` at a batch of positions.
    ///
    /// - `xs`: query points (length `m`); each `xs[i]` is a coordinate array `[f64; N]`.
    ///   Every coordinate must lie inside its direction's domain `[xl, xl+l]`,
    ///   for **every** extension; use [`eval_extended`](Self::eval_extended) to
    ///   evaluate the underlying extension at positions outside the domain.
    /// - `u`: DOF vector in the column-major tensor-product Gauss-Lobatto layout from [`get_x`](Self::get_x).
    /// - `fu`: output buffer of length `m`; `fu[i]` receives the interpolated value at `xs[i]`.
    ///
    /// Uses sum factorisation and groups points by element for efficiency.
    /// For repeated evaluation at the same points with varying `u`, build an
    /// [`EvalPlan`] once with [`eval_plan`](Self::eval_plan) and apply it with
    /// [`eval_with_plan`](Self::eval_with_plan).
    pub fn eval<T: Scalar>(&self, xs: &[[f64; N]], u: &[T], fu: &mut [T]) {
        debug_assert_eq!(u.len(), self.total);
        debug_assert_eq!(fu.len(), xs.len());

        let geo = EvalGeometry::new(self);

        // Step 1: compute flat element index for each point.
        let elem_indices: Vec<usize> = xs.par_iter().map(|x| geo.elem_index(x)).collect();

        // Step 2: group point indices by element.
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &k_flat) in elem_indices.iter().enumerate() {
            groups.entry(k_flat).or_default().push(i);
        }

        // Step 3: parallel over elements — gather once, sum-factorise per point.
        // Results are written straight into `fu`.
        // SAFETY: point indices are unique across all groups, so parallel
        // writes to fu[i] never overlap.
        let fu_ptr = fu.as_mut_ptr() as usize;
        groups.into_par_iter().for_each(|(k_flat, indices)| {
            let ks = geo.elem_multi_index(k_flat);
            let u_local = self.gather_local(&geo, &ks, u);

            // Per-thread scratch: basis values (phi) and contraction buffer (v),
            // reused across all points handled by the same worker.
            indices.par_iter().for_each_init(
                || {
                    (
                        vec![0.0_f64; geo.phi_len],
                        vec![T::default(); geo.local_total / geo.counts[N - 1]],
                    )
                },
                |(phi, v), &i| {
                    let xi = geo.local_coords(&xs[i], &ks);
                    self.fill_phi(&geo, &xi, phi);
                    let val = contract(&geo, phi, &u_local, v);
                    let fu_ptr = fu_ptr as *mut T;
                    unsafe { *fu_ptr.add(i) = val };
                },
            );
        });
    }

    /// Evaluate the extension of the ND interpolant of `u` at a batch of positions.
    ///
    /// Unlike [`eval`](Self::eval), coordinates may lie anywhere: each direction is
    /// folded back into its domain according to that direction's extension —
    /// periodic wrap (`Periodic`), even reflection (`Even`/Neumann), or odd
    /// reflection with a sign flip (`Odd`/Dirichlet) — so the result is
    /// consistent with the underlying extended representation.
    pub fn eval_extended<T: Scalar>(&self, xs: &[[f64; N]], u: &[T], fu: &mut [T]) {
        self.eval_extended_iter(xs.par_iter().copied(), u, fu);
    }

    /// Fold a point into the domain direction by direction, returning the folded
    /// point and the sign of the extended interpolant there
    /// (`-1` on odd-reflected copies, `+1` otherwise).
    fn fold_point(&self, x: [f64; N]) -> ([f64; N], f64) {
        let mut sign = 1.0_f64;
        let y: [f64; N] = std::array::from_fn(|d| {
            let (y_d, s_d) = self.grids[d].fold_extended(x[d]);
            sign *= s_d;
            y_d
        });
        (y, sign)
    }

    /// `true` iff some direction uses the `Odd` extension, i.e. reflection
    /// signs can differ from `+1`.
    fn has_odd(&self) -> bool {
        (0..N).any(|d| matches!(self.grids[d].extension, Extension::Odd))
    }

    /// Shared core of [`eval_extended`](Self::eval_extended) and
    /// [`convect`](Self::convect): fold a stream of points into the domain in a
    /// single parallel pass, evaluate, and apply the reflection signs.
    ///
    /// This pass runs before (not inside) `eval`, so its parallelism does not
    /// nest with eval's. The sign is `-1` only on odd-reflected copies, so when
    /// no direction is `Odd` the sign vector is not built and no sign pass runs.
    fn eval_extended_iter<T: Scalar, I>(&self, pts: I, u: &[T], fu: &mut [T])
    where
        I: IndexedParallelIterator<Item = [f64; N]>,
    {
        if self.has_odd() {
            let (ys, signs): (Vec<[f64; N]>, Vec<f64>) = pts.map(|x| self.fold_point(x)).unzip();
            self.eval(&ys, u, fu);
            fu.par_iter_mut()
                .zip(signs.par_iter())
                .for_each(|(v, &s)| *v *= s);
        } else {
            let ys: Vec<[f64; N]> = pts.map(|x| self.fold_point(x).0).collect();
            self.eval(&ys, u, fu);
        }
    }

    /// Apply a flow map `lambda` to every DOF node and interpolate `u` at the resulting positions.
    ///
    /// Positions outside the domain are handled by [`eval_extended`](Self::eval_extended),
    /// i.e. folded back according to each direction's extension.
    ///
    /// - `u`: DOF vector in the column-major tensor-product layout from [`get_x`](Self::get_x).
    /// - `out`: output buffer of the same length as `u`; receives the convected values.
    ///
    /// When the same `lambda` is applied many times with varying `u`, build an
    /// [`EvalPlan`] once with [`convect_plan`](Self::convect_plan) and apply it
    /// with [`eval_with_plan`](Self::eval_with_plan).
    pub fn convect<T: Scalar, F>(&self, lambda: F, u: &[T], out: &mut [T])
    where
        F: Fn([f64; N]) -> [f64; N] + Sync,
    {
        debug_assert_eq!(out.len(), u.len());
        // Node coordinates are computed from the flat index (as in `get_x`), so
        // the grid, the flow map and the domain fold fuse into a single pass
        // without materialising the grid or the shifted positions.
        let x1d: [Vec<f64>; N] = std::array::from_fn(|d| self.grids[d].get_x());
        self.eval_extended_iter(
            (0..self.total).into_par_iter().map(|p| {
                lambda(std::array::from_fn(|d| {
                    x1d[d][(p / self.strides[d]) % self.ndofs[d]]
                }))
            }),
            u,
            out,
        );
    }
}

// ─── Precomputed evaluation plans ─────────────────────────────────────────────

/// Sentinel in gather tables for "Dirichlet zero, not stored".
const ZERO_DOF: u32 = u32::MAX;

/// Points assigned to one element: a range into the plan's grouped point
/// arrays, plus the element's precomputed DOF gather table.
struct ElemGroup {
    /// Range into `EvalPlan::point_ids` / `EvalPlan::xi`.
    start: usize,
    end: usize,
    /// Global DOF index per local node ([`ZERO_DOF`] = Dirichlet zero).
    gather: Vec<u32>,
}

/// Precomputed interpolation plan: fixed evaluation points, varying DOF data.
///
/// Built once by [`GridND::eval_plan`], [`GridND::eval_extended_plan`] or
/// [`GridND::convect_plan`], then applied any number of times with
/// [`GridND::eval_with_plan`]. All position-dependent work — flow map,
/// extension folding, element search, grouping, DOF gather indices, local
/// coordinates — is done at construction; the apply is pure gather + contraction.
///
/// Memory: ~`(8 + 8·N)` bytes per point (grouped order) plus one `u32` gather
/// table of `∏_d (r_d + 1)` entries per occupied element, plus `8` bytes per
/// point for reflection signs when some direction is `Odd`.
///
/// The plan stores only positions and indices, so the same plan can be applied
/// to real (`f64`) and complex (`Complex<f64>`) fields interchangeably.
///
/// ```rust
/// use lobatto_grid::{Extension, GridND};
///
/// let grid = GridND::<2>::new([4; 2], [2; 2], [1.0; 2], [0.0; 2], [Extension::Periodic; 2]);
/// let x = grid.get_x();
/// let u: Vec<f64> = x.iter().map(|&[a, b]| a + b).collect();
///
/// // Build the plan once for a fixed flow map ...
/// let plan = grid.convect_plan(|[a, b]| [a + 0.1, b]);
///
/// // ... and reuse it for every new field `u` (real or complex).
/// let mut out = vec![0.0_f64; u.len()];
/// grid.eval_with_plan(&plan, &u, &mut out);
/// ```
pub struct EvalPlan<const N: usize> {
    /// Number of query points.
    m: usize,
    /// `GridND::total` of the grid the plan was built for (compatibility check).
    total: usize,
    /// Reflection signs in original point order (`None` when all are +1).
    signs: Option<Vec<f64>>,
    /// Occupied elements with their point ranges and gather tables.
    groups: Vec<ElemGroup>,
    /// Original point index of each grouped point.
    point_ids: Vec<u32>,
    /// Local (element-normalised) coordinates, in grouped order.
    xi: Vec<[f64; N]>,
}

impl<const N: usize> GridND<N> {
    /// Build an [`EvalPlan`] for a fixed batch of in-domain positions.
    ///
    /// Same domain requirements as [`eval`](Self::eval); use
    /// [`eval_extended_plan`](Self::eval_extended_plan) for arbitrary positions.
    pub fn eval_plan(&self, xs: &[[f64; N]]) -> EvalPlan<N> {
        self.build_plan(xs.to_vec(), None)
    }

    /// Build an [`EvalPlan`] for a fixed batch of arbitrary positions.
    ///
    /// Positions are folded into the domain according to each direction's
    /// extension (as in [`eval_extended`](Self::eval_extended)); reflection
    /// signs are precomputed and applied on every
    /// [`eval_with_plan`](Self::eval_with_plan).
    pub fn eval_extended_plan(&self, xs: &[[f64; N]]) -> EvalPlan<N> {
        if self.has_odd() {
            let (ys, signs): (Vec<[f64; N]>, Vec<f64>) =
                xs.par_iter().map(|&x| self.fold_point(x)).unzip();
            self.build_plan(ys, Some(signs))
        } else {
            let ys: Vec<[f64; N]> = xs.par_iter().map(|&x| self.fold_point(x).0).collect();
            self.build_plan(ys, None)
        }
    }

    /// Build an [`EvalPlan`] for a fixed flow map `lambda` over the DOF nodes.
    ///
    /// `eval_with_plan(&convect_plan(lambda), u, out)` computes the same values
    /// as [`convect`](Self::convect)`(lambda, u, out)`, but all position
    /// processing is done once here instead of on every call.
    pub fn convect_plan<F>(&self, lambda: F) -> EvalPlan<N>
    where
        F: Fn([f64; N]) -> [f64; N] + Sync,
    {
        let x1d: [Vec<f64>; N] = std::array::from_fn(|d| self.grids[d].get_x());
        let pts: Vec<[f64; N]> = (0..self.total)
            .into_par_iter()
            .map(|p| {
                lambda(std::array::from_fn(|d| {
                    x1d[d][(p / self.strides[d]) % self.ndofs[d]]
                }))
            })
            .collect();
        self.eval_extended_plan(&pts)
    }

    /// Shared plan construction from already-folded in-domain positions.
    fn build_plan(&self, ys: Vec<[f64; N]>, signs: Option<Vec<f64>>) -> EvalPlan<N> {
        let m = ys.len();
        assert!(
            m < ZERO_DOF as usize && self.total < ZERO_DOF as usize,
            "eval_plan: point/DOF counts must fit in u32"
        );
        let geo = EvalGeometry::new(self);

        let elem_indices: Vec<usize> = ys.par_iter().map(|y| geo.elem_index(y)).collect();

        let mut by_elem: HashMap<usize, Vec<u32>> = HashMap::new();
        for (i, &k_flat) in elem_indices.iter().enumerate() {
            by_elem.entry(k_flat).or_default().push(i as u32);
        }

        let mut groups = Vec::with_capacity(by_elem.len());
        let mut point_ids = Vec::with_capacity(m);
        let mut xi = Vec::with_capacity(m);
        for (k_flat, ids) in by_elem {
            let ks = geo.elem_multi_index(k_flat);

            // Precompute the DOF gather table for this element.
            let gather: Vec<u32> = (0..geo.local_total)
                .map(|p| {
                    let mut global = 0usize;
                    for (d, &k_d) in ks.iter().enumerate() {
                        let j = (p / geo.count_strides[d]) % geo.counts[d];
                        let dof_d = self.grids[d].eval_dof_index(k_d, j);
                        if dof_d == self.grids[d].n_dof {
                            return ZERO_DOF;
                        }
                        global += dof_d * self.strides[d];
                    }
                    global as u32
                })
                .collect();

            let start = point_ids.len();
            for &i in &ids {
                xi.push(geo.local_coords(&ys[i as usize], &ks));
                point_ids.push(i);
            }
            groups.push(ElemGroup {
                start,
                end: point_ids.len(),
                gather,
            });
        }

        EvalPlan {
            m,
            total: self.total,
            signs,
            groups,
            point_ids,
            xi,
        }
    }

    /// Interpolate `u` at the positions precomputed in `plan`.
    ///
    /// - `plan`: built by [`eval_plan`](Self::eval_plan),
    ///   [`eval_extended_plan`](Self::eval_extended_plan) or
    ///   [`convect_plan`](Self::convect_plan) **on this grid**.
    /// - `u`: DOF vector in the column-major tensor-product layout from [`get_x`](Self::get_x).
    /// - `fu`: output buffer; `fu[i]` receives the value at the plan's `i`-th position.
    pub fn eval_with_plan<T: Scalar>(&self, plan: &EvalPlan<N>, u: &[T], fu: &mut [T]) {
        assert_eq!(
            plan.total, self.total,
            "eval_with_plan: plan was built for a different grid"
        );
        assert_eq!(u.len(), self.total);
        assert_eq!(fu.len(), plan.m);

        let geo = EvalGeometry::new(self);

        // SAFETY: `plan.point_ids` is a permutation of 0..m, and each group owns
        // a disjoint slice of it, so parallel writes to fu[i] never overlap.
        let fu_ptr = fu.as_mut_ptr() as usize;
        plan.groups.par_iter().for_each(|g| {
            let u_local: Vec<T> = g
                .gather
                .iter()
                .map(|&idx| {
                    if idx == ZERO_DOF {
                        T::default()
                    } else {
                        u[idx as usize]
                    }
                })
                .collect();

            let ids = &plan.point_ids[g.start..g.end];
            let xis = &plan.xi[g.start..g.end];
            // Points within a group are processed serially: the outer loop over
            // groups already saturates the thread pool, and per-group scratch
            // buffers are reused across all of the group's points.
            let mut phi = vec![0.0_f64; geo.phi_len];
            let mut v = vec![T::default(); geo.local_total / geo.counts[N - 1]];
            for (&i, xi_i) in ids.iter().zip(xis.iter()) {
                self.fill_phi(&geo, xi_i, &mut phi);
                let val = contract(&geo, &phi, &u_local, &mut v);
                let fu_ptr = fu_ptr as *mut T;
                unsafe { *fu_ptr.add(i as usize) = val };
            }
        });

        if let Some(signs) = &plan.signs {
            fu.par_iter_mut()
                .zip(signs.par_iter())
                .for_each(|(v, &s)| *v *= s);
        }
    }
}
