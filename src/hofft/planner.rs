//! HoFFT transform planners.
//!
//! This module provides [`HofftPlanner`] (1D) and [`HofftPlannerND`] (N-dimensional),
//! which encapsulate only the **forward and inverse FFT steps** of the HoFFT algorithm
//! (steps 2 and 4 from [`hofft::solver`]).
//!
//! //! Algorithm based on
//!
//! - \[1\] Caforio, F. and Imperiale, S. (2019). *High-order discrete fourier transform for the solution of the Poisson equation*. SIAM Journal of Scientific Computing, 41(5), 2747-2771
//!
//!
//! # Data layout
//!
//! Both planners share the same layout convention as [`hofft::solver`]:
//!
//! - **1D** (`HofftPlanner`): flat buffer of length `r × n`.
//!   Entry `g[j * n + k]` corresponds to local node `j ∈ [0, r)`, element `k ∈ [0, n)`.
//! - **ND** (`HofftPlannerND`): flat buffer of length `∏_d (n_d × r_d)`, column-major
//!   (direction 0 varies fastest).
//!   Entry `g[Σ_d (j_d * n_d + k_d) * stride_d]` for local node `j_d`, element `k_d`.

use crate::basis::Basis;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

// ── 1D ──────────────────────────────────────────────────────────────────────

/// 1D HoFFT transform planner.
///
/// Plans and applies `r` independent forward/inverse FFTs of length `n`,
/// corresponding to steps 2 and 4 of the HoFFT algorithm along a single direction.
///
/// The buffer layout is rows-major: `r` consecutive blocks of `n` elements each,
/// matching the DOF ordering `g[j * n + k]` used by [`solver::PoissonSolver1D`].
pub struct HofftPlanner {
    /// FFT length (number of elements).
    pub n: usize,
    /// Number of local nodes per element (polynomial order).
    pub r: usize,
    /// Domain length.
    pub l: f64,
    /// Left boundary of the domain.
    pub xl: f64,
    /// GLL quadrature nodes on `[0, 1]` for a single element (length `r + 1`).
    gll_points: Vec<f64>,
    /// Forward FFT engine (length `n`).
    pub(crate) fw_fft: Arc<dyn Fft<f64>>,
    /// Inverse FFT engine (length `n`).
    pub(crate) i_fft: Arc<dyn Fft<f64>>,
}

impl HofftPlanner {
    /// Create a planner for `r` independent FFTs of length `n` on the domain `[xl, xl+l]`.
    pub fn new(n: usize, r: usize, l: f64, xl: f64) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fw_fft = planner.plan_fft_forward(n);
        let i_fft = planner.plan_fft_inverse(n);
        let gll_points = Basis::new(vec![(r + 1, true)]).points[0].clone();
        Self {
            n,
            r,
            l,
            xl,
            gll_points,
            fw_fft,
            i_fft,
        }
    }

    /// GLL nodes on `[xl, xl+l]`, length `n * r`, ordered by local node then element:
    /// `x[j * n + k]` = position of local node `j` in element `k`.
    pub fn get_x(&self) -> Vec<f64> {
        let h = self.l / (self.n as f64);
        let mut x = vec![0.0_f64; self.n * self.r];
        for k in 0..self.n {
            for j in 0..self.r {
                x[j * self.n + k] = h * ((k as f64) + self.gll_points[j]) + self.xl;
            }
        }
        x
    }

    /// Apply `r` forward FFTs of length `n` to `g`.
    ///
    /// `g` must have length `r * n`. The buffer is interpreted as `r` consecutive
    /// rows of `n` elements; each row is transformed independently in place.
    /// No normalisation is applied.
    pub fn forward(&self, g: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.r * self.n);
        self.fw_fft.process(g);
    }

    /// Apply `r` inverse FFTs of length `n` to `g`, then normalise by `1/n`.
    ///
    /// `g` must have length `r * n`. The buffer is interpreted as `r` consecutive
    /// rows of `n` elements; each row is transformed independently in place.
    pub fn inverse(&self, g: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.r * self.n);
        let norm = self.n as f64;
        self.i_fft.process(g);
        for v in g.iter_mut() {
            *v /= norm;
        }
    }
}

// ── ND ──────────────────────────────────────────────────────────────────────

/// N-dimensional HoFFT transform planner.
///
/// Applies forward/inverse FFTs direction by direction (sum factorisation),
/// corresponding to steps 2 and 4 of the ND HoFFT algorithm described in
/// [`solver::PoissonSolverND`].
///
/// Per-direction transforms are parallelised over independent fibers with Rayon.
pub struct HofftPlannerND<const N: usize> {
    /// One 1D planner per spatial direction.
    planners: [HofftPlanner; N],
    /// FFT lengths per direction (`ns[d]` = number of elements in direction `d`).
    ns: [usize; N],
    /// DOF counts per direction (`ndofs[d] = ns[d] * rs[d]`).
    ndofs: [usize; N],
    /// Total number of DOFs (`∏_d ndofs[d]`).
    total: usize,
    /// Column-major strides for the DOF tensor.
    strides: [usize; N],
}

impl<const N: usize> HofftPlannerND<N> {
    /// Create a planner for a tensor-product domain with `ns[d]` elements and
    /// `rs[d]` local nodes per element on `[xls[d], xls[d]+ls[d]]` in direction `d`.
    pub fn new(ns: [usize; N], rs: [usize; N], ls: [f64; N], xls: [f64; N]) -> Self {
        let planners = std::array::from_fn(|d| HofftPlanner::new(ns[d], rs[d], ls[d], xls[d]));
        let ndofs: [usize; N] = std::array::from_fn(|d| ns[d] * rs[d]);
        let total = ndofs.iter().product();
        let strides = Self::compute_strides(&ndofs);
        Self {
            planners,
            ns,
            ndofs,
            total,
            strides,
        }
    }

    /// Tensor-product GLL grid as `Vec<[f64; N]>` of length `total = ∏_d (ns[d] * rs[d])`.
    ///
    /// `result[p][d]` is the coordinate of point `p` in direction `d`.
    /// Direction 0 varies fastest.
    pub fn get_x(&self) -> Vec<[f64; N]> {
        let x1d: [Vec<f64>; N] = std::array::from_fn(|d| self.planners[d].get_x());
        (0..self.total)
            .map(|p| std::array::from_fn(|d| x1d[d][(p / self.strides[d]) % self.ndofs[d]]))
            .collect()
    }

    /// Apply forward FFT direction by direction.
    ///
    /// `g` must have length `total = ∏_d (ns[d] * rs[d])`.
    /// No normalisation is applied.
    ///
    /// # Safety
    /// Fibers along each direction are disjoint strided slices; parallel writes
    /// for distinct fiber indices do not alias.
    pub fn forward(&self, g: &mut Vec<Complex<f64>>) {
        debug_assert_eq!(g.len(), self.total);
        for d in 0..N {
            let nd = self.ndofs[d];
            let stride = self.strides[d];
            let fw_fft = &self.planners[d].fw_fft;
            let ptr = g.as_mut_ptr() as usize;
            // SAFETY: flat_base yields disjoint strided slices for distinct s values.
            (0..self.total / nd).into_par_iter().for_each(|s| {
                let base = self.flat_base(s, d);
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
    }

    /// Apply inverse FFT direction by direction, normalising by `1/ns[d]` per direction.
    ///
    /// `g` must have length `total = ∏_d (ns[d] * rs[d])`.
    ///
    /// # Safety
    /// Same disjoint-fiber argument as [`forward`](Self::forward).
    pub fn inverse(&self, g: &mut Vec<Complex<f64>>) {
        debug_assert_eq!(g.len(), self.total);
        for d in 0..N {
            let nd = self.ndofs[d];
            let norm = self.ns[d] as f64;
            let stride = self.strides[d];
            let i_fft = &self.planners[d].i_fft;
            let ptr = g.as_mut_ptr() as usize;
            // SAFETY: flat_base yields disjoint strided slices for distinct s values.
            (0..self.total / nd).into_par_iter().for_each(|s| {
                let base = self.flat_base(s, d);
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
    }

    // Flat start index of the s-th fiber along direction d in the DOF tensor.
    fn flat_base(&self, s: usize, d: usize) -> usize {
        let mut base = 0usize;
        let mut rem = s;
        for d2 in 0..N {
            if d2 == d {
                continue;
            }
            base += (rem % self.ndofs[d2]) * self.strides[d2];
            rem /= self.ndofs[d2];
        }
        base
    }

    // Column-major strides for a tensor-product array of shape `sizes`.
    // st[0] = 1 (direction 0 varies fastest), st[d] = st[d-1] * sizes[d-1].
    fn compute_strides(sizes: &[usize; N]) -> [usize; N] {
        let mut st = [1usize; N];
        for d in 1..N {
            st[d] = st[d - 1] * sizes[d - 1];
        }
        st
    }
}
