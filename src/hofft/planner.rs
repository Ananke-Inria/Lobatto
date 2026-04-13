//! HoFFT transform planners.
//!
//! This module provides [`HofftPlanner`] (1D) and [`HofftPlannerND`] (N-dimensional),
//! which encapsulate only the **forward and inverse FFT steps** of the HoFFT algorithm
//! (steps 2 and 4 from [`hofft::solver`]).
//!
//! Algorithm based on
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
use crate::utilities::{barycentric_weights, lagrangian_interpolation_all};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

// ── Extension ────────────────────────────────────────────────────────────────

/// Extension type for a [`HofftPlanner`].
///
/// Controls the number of required dof n_dof to handle non-periodic boundary conditions
/// using the periodic FFT machinery (FFT of size n_fft).
/// The number of elements is n while the polynomial order of the shape functions is r
///
/// | Variant    | BC        | `n_fft` | `n_dof`       |
/// |------------|-----------|---------|---------------|
/// | `Periodic` | Periodic  | `n`     | `n * r`       |
/// | `Odd`      | Dirichlet | `2n`    | `n * r − 1`   |
/// | `Even`     | Neumann   | `2n`    | `n * r + 1`   |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension {
    /// Standard periodic FFT; no extension needed.
    Periodic,
    /// Odd (anti-symmetric) extension — enforces zero Dirichlet boundary values.
    Odd,
    /// Even (symmetric) extension — enforces zero-derivative Neumann boundary values.
    Even,
}

// ── 1D ──────────────────────────────────────────────────────────────────────

/// 1D HoFFT Raw planner.
///
/// Plans and applies `r` independent forward/inverse FFTs of length `n`,
/// The buffer layout is: `r` consecutive blocks of `n` elements each.
pub struct RawPlanner {
    /// FFT length (equivalent to number of elements).
    pub n: usize,
    /// Polynomial order.
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

impl RawPlanner {
    /// Create a planner for `r` independent FFTs of length `n` on the domain `[xl, xl+l]`.
    pub fn new(n: usize, r: usize, l: f64, xl: f64) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fw_fft = planner.plan_fft_forward(n);
        let i_fft = planner.plan_fft_inverse(n);
        let gll_points = Basis::new(vec![(r + 1, true)]).points(0).to_vec();
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

/// 1D HoFFT planner with odd/even extension support.
///
/// Physical DOF layout: x-increasing
/// - `Periodic`: `n_dof = n * r`.
/// - `Odd`:      `n_dof = n * r − 1`.
/// - `Even`:     `n_dof = n * r + 1`.
pub struct HofftPlanner {
    /// Number of physical elements.
    pub n: usize,
    /// FFT length: `n` for `Periodic`, `2n` for `Odd`/`Even`.
    pub n_fft: usize,
    /// Number of dofs
    pub n_dof: usize,
    /// Polynomial order.
    pub r: usize,
    /// Physical domain length.
    pub l: f64,
    /// Left boundary of the domain.
    pub xl: f64,
    /// Extension type.
    pub extension: Extension,
    /// GLL node positions on `[0, 1]` (length `r + 1`).
    gll_points: Vec<f64>,
    /// Barycentric weights for Lagrange interpolation on the `r+1` GLL nodes.
    bary_weights: Vec<f64>,
    /// Forward FFT engine (length `n_fft`).
    fw_fft: Arc<dyn Fft<f64>>,
    /// Inverse FFT engine (length `n_fft`).
    i_fft: Arc<dyn Fft<f64>>,
    /// Precomputed unpacking map: `unpack_table[j*n_fft+k] = (packed_idx, coeff)`.
    /// Avoids repeated `exp()` calls in `inverse()` and `get_values()`.
    pub(crate) unpack_table: Vec<(usize, Complex<f64>)>,
    /// Precomputed canonicality: `canonical_table[j*n_fft+k] = true` iff the entry
    /// is independently stored (not derived by symmetry or a forced zero).
    pub(crate) canonical_table: Vec<bool>,
}

impl HofftPlanner {
    /// Create a planner for HoFFT of functions describe by piecewise polynomial of order `r` on `n` physical elements
    /// on the domain `[xl, xl+l]` with the given [`Extension`].
    pub fn new(n: usize, r: usize, l: f64, xl: f64, extension: Extension) -> Self {
        let n_fft = match extension {
            Extension::Periodic => n,
            Extension::Odd | Extension::Even => 2 * n,
        };

        let n_dof = match extension {
            Extension::Periodic => r * n,
            Extension::Odd => r * n - 1,
            Extension::Even => r * n + 1,
        };

        let mut fft_planner = FftPlanner::<f64>::new();
        let fw_fft = fft_planner.plan_fft_forward(n_fft);
        let i_fft = fft_planner.plan_fft_inverse(n_fft);
        let gll_points = Basis::new(vec![(r + 1, true)]).points(0).to_vec();
        let bary_weights = barycentric_weights(&gll_points);
        let mut s = Self {
            n,
            n_fft,
            n_dof,
            r,
            l,
            xl,
            extension,
            gll_points,
            bary_weights,
            fw_fft,
            i_fft,
            unpack_table: vec![],
            canonical_table: vec![],
        };
        let table_size = s.r * s.n_fft;
        s.unpack_table = (0..table_size).map(|i| s.k_unpacking(i)).collect();
        s.canonical_table = (0..table_size)
            .map(|i| {
                let (c, _) = s.k_unpacking(i);
                s.k_packing(c) == i
            })
            .collect();
        s
    }

    /// Number of independent Fourier modes for the symbol solve.
    ///
    /// - `Periodic`: `n`.
    /// - `Odd` / `Even`: `n + 1` (DC through Nyquist).
    pub fn n_dof(&self) -> usize {
        match self.extension {
            Extension::Periodic => self.n,
            Extension::Odd => self.n - 1,
            Extension::Even => self.n + 1,
        }
    }

    /// Extract the r values for Fourier mode `k` from the packed buffer `g` into `g_k`.
    ///
    /// `k` may be any mode in `0..n_fft`; modes in the symmetric half (`k > n`) are
    /// recovered via Lemma 3.2 and the cross-row symmetries.
    pub fn get_values(&self, k: usize, g: &[Complex<f64>], g_k: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.n_dof);
        debug_assert_eq!(g_k.len(), self.r);
        for j in 0..self.r {
            let (i_g, coeff) = self.unpack_table[j * self.n_fft + k];
            g_k[j] = coeff * g[i_g];
        }
    }

    /// Write the r values for Fourier mode `k` from `g_k` into the packed buffer `g`.
    ///
    /// Only the independently stored entries are written; `k` must be in `0..=n`
    /// (the symmetric half is derived and must not be stored directly).
    pub fn set_values(&self, k: usize, g: &mut [Complex<f64>], g_k: &[Complex<f64>]) {
        assert!(k <= self.n);
        debug_assert_eq!(g.len(), self.n_dof);
        debug_assert_eq!(g_k.len(), self.r);

        match self.extension {
            Extension::Periodic => {
                assert!(k < self.n);
                for j in 0..self.r {
                    g[j * self.n + k] = g_k[j];
                }
            }
            Extension::Odd => {
                if k >= 1 && k < self.n {
                    // Chunk 1: all j rows stored directly.
                    for j in 0..self.r {
                        g[j * (self.n - 1) + (k - 1)] = g_k[j];
                    }
                } else if k == 0 {
                    // Chunk 2: j=1..⌊(r−1)/2⌋ only (j=0 and j=r/2 are zero, rest derived).
                    let base = self.r * (self.n - 1);
                    for j in 1..=(self.r - 1) / 2 {
                        g[base + (j - 1)] = g_k[j];
                    }
                } else {
                    // k == n — Chunk 3: j=1..⌊r/2⌋ only (j=0 zero, rest derived).
                    let base = self.r * (self.n - 1) + (self.r - 1) / 2;
                    for j in 1..=self.r / 2 {
                        g[base + (j - 1)] = g_k[j];
                    }
                }
            }
            Extension::Even => {
                if k >= 1 && k < self.n {
                    // Chunk 1: all j rows stored directly.
                    for j in 0..self.r {
                        g[j * (self.n - 1) + (k - 1)] = g_k[j];
                    }
                } else if k == 0 {
                    // Chunk 2: j=0..⌊r/2⌋ (rest derived).
                    let base = self.r * (self.n - 1);
                    for j in 0..=self.r / 2 {
                        g[base + j] = g_k[j];
                    }
                } else {
                    // k == n — Chunk 3: j=0..⌊(r−1)/2⌋ (j=r/2 zero if r even, rest derived).
                    let base = self.r * (self.n - 1) + self.r / 2 + 1;
                    for j in 0..=(self.r - 1) / 2 {
                        g[base + j] = g_k[j];
                    }
                }
            }
        }
    }

    //
    // Layout (same for Odd and Even):
    //   Chunk 1  [0 .. r*(n−1))          : modes k=1..n−1 for ALL j=0..r−1 ->  recovers k=n+1..2n−1 frequency from these.
    //   Chunk 2  [r*(n−1) .. r*(n−1)+c)  : mode k=0 for a reduced set of j rows.
    //   Chunk 3  [r*(n−1)+c .. n_dof)    : mode k=n for a reduced set of j rows.
    //
    // Odd   — k=0: j=1..⌊(r−1)/2⌋ (j=0 zero, j=r/2 zero if r even, rest = −row_{r−j})
    //       — k=n: j=1..⌊r/2⌋     (j=0 zero, rest = row_{r−j})
    //   sizes: r*(n−1) + ⌊(r−1)/2⌋ + ⌊r/2⌋ = r*n − 1 ✓
    //
    // Even  — k=0: j=0..⌊r/2⌋          (rest = row_{r−j})
    //       — k=n: j=0..⌊(r−1)/2⌋      (j=r/2 zero if r even, rest = −row_{r−j})
    //   sizes: r*(n−1) + (⌊r/2⌋+1) + (⌊(r−1)/2⌋+1) = r*n + 1 ✓
    /// Maps a compact packed index `i` (0..n_dof) to its corresponding index in the
    /// full `r × n_fft` buffer.  Used during the forward pass to extract independent modes.
    ///
    /// Layout (Odd and Even):
    ///   Chunk 1 `[0, r*(n−1))`:        modes k=1..n−1, all j → buf[j*n_fft + k]
    ///   Chunk 2 `[r*(n−1), ..)`:       mode k=0, independent j rows only
    ///   Chunk 3 `[.., n_dof)`:         mode k=n, independent j rows only
    fn k_packing(&self, i: usize) -> usize {
        match self.extension {
            Extension::Periodic => i,
            Extension::Odd => {
                let c1 = self.r * (self.n - 1);
                let c2 = (self.r - 1) / 2; // number of stored k=0 entries
                if i < c1 {
                    // Chunk 1: j*(n-1) + (k-1)  →  j*n_fft + k
                    let j = i / (self.n - 1);
                    let k = i % (self.n - 1) + 1;
                    j * self.n_fft + k
                } else if i < c1 + c2 {
                    // Chunk 2: k=0, j=1..⌊(r−1)/2⌋
                    let j = i - c1 + 1;
                    j * self.n_fft
                } else {
                    // Chunk 3: k=n, j=1..⌊r/2⌋
                    let j = i - c1 - c2 + 1;
                    j * self.n_fft + self.n
                }
            }
            Extension::Even => {
                let c1 = self.r * (self.n - 1);
                let c2 = self.r / 2 + 1; // number of stored k=0 entries
                if i < c1 {
                    // Chunk 1: j*(n-1) + (k-1)  →  j*n_fft + k
                    let j = i / (self.n - 1);
                    let k = i % (self.n - 1) + 1;
                    j * self.n_fft + k
                } else if i < c1 + c2 {
                    // Chunk 2: k=0, j=0..⌊r/2⌋
                    let j = i - c1;
                    j * self.n_fft
                } else {
                    // Chunk 3: k=n, j=0..⌊(r−1)/2⌋
                    let j = i - c1 - c2;
                    j * self.n_fft + self.n
                }
            }
        }
    }

    /// Maps a full buffer index `i` (0..r*n_fft) to `(packed_index, coefficient)` such that
    /// `buf[i] = coefficient * g[packed_index]`.  Used during the inverse pass to expand the
    /// full buffer from the compact packed representation.
    ///
    /// The coefficient is `Complex<f64>` to accommodate the complex phase from Lemma 3.2.
    /// Zero entries (e.g. j=0 for Odd k=0) return index 0 with coefficient 0.
    fn k_unpacking(&self, i: usize) -> (usize, Complex<f64>) {
        let one = Complex::new(1.0_f64, 0.0);
        let zero_coeff = Complex::new(0.0_f64, 0.0);
        let minus_one = Complex::new(-1.0_f64, 0.0);

        match self.extension {
            Extension::Periodic => (i, one),
            Extension::Odd => {
                let j = i / self.n_fft;
                let k = i % self.n_fft;
                let c1 = self.r * (self.n - 1);
                let c2 = (self.r - 1) / 2;

                if k >= 1 && k < self.n {
                    // Chunk 1: directly stored.
                    (j * (self.n - 1) + (k - 1), one)
                } else if k == 0 {
                    // Low-frequency row; anti-symmetric: F̂_{0,r−j} = −F̂_{0,j}.
                    if j == 0 {
                        return (0, zero_coeff);
                    } // always zero
                    if self.r % 2 == 0 && j == self.r / 2 {
                        return (0, zero_coeff);
                    } // zero

                    if j <= (self.r - 1) / 2 {
                        (c1 + (j - 1), one)
                    } else {
                        // j > (r-1)/2: anti-symmetry from stored j' = r-j
                        (c1 + (self.r - j - 1), minus_one)
                    }
                } else if k == self.n {
                    // High-frequency row; symmetric: F̂_{n,r−j} = F̂_{n,j}.
                    if j == 0 {
                        return (0, zero_coeff);
                    } // always zero
                    let j_stored = j.min(self.r - j);
                    (c1 + c2 + (j_stored - 1), one)
                } else {
                    // k ∈ n+1..2n−1: Lemma 3.2 (Odd, sign = −1).
                    //   j=0: F̂_{2n−k, 0} = −F̂_{k, 0}           → coefficient −1 (no phase)
                    //   j>0: F̂_{2n−k, j} = −e^{−iπkm/n}·F̂_{km, r−j}
                    let km = self.n_fft - k; // mirror mode ∈ 1..n-1
                    if j == 0 {
                        let packed = km - 1; // row 0 of mode km in Chunk 1
                        (packed, minus_one)
                    } else {
                        let phase =
                            Complex::new(0.0_f64, -PI * (km as f64) / (self.n as f64)).exp();
                        let packed = (self.r - j) * (self.n - 1) + (km - 1);
                        (packed, -phase)
                    }
                }
            }
            Extension::Even => {
                let j = i / self.n_fft;
                let k = i % self.n_fft;
                let c1 = self.r * (self.n - 1);
                let c2 = self.r / 2 + 1;

                if k >= 1 && k < self.n {
                    // Chunk 1: directly stored.
                    (j * (self.n - 1) + (k - 1), one)
                } else if k == 0 {
                    // Low-frequency row; symmetric: F̂_{0,r−j} = F̂_{0,j}.
                    let j_stored = j.min(self.r - j);
                    (c1 + j_stored, one)
                } else if k == self.n {
                    // High-frequency row; anti-symmetric: F̂_{n,r−j} = −F̂_{n,j}.
                    if self.r % 2 == 0 && j == self.r / 2 {
                        return (0, zero_coeff);
                    } // zero
                    if j <= (self.r - 1) / 2 {
                        (c1 + c2 + j, one)
                    } else {
                        // j > (r-1)/2: anti-symmetry from stored j' = r-j
                        (c1 + c2 + (self.r - j), minus_one)
                    }
                } else {
                    // k ∈ n+1..2n−1: Lemma 3.2 (Even, sign = +1).
                    //   j=0: F̂_{2n−k, 0} = F̂_{k, 0}             → coefficient +1 (no phase)
                    //   j>0: F̂_{2n−k, j} = e^{−iπkm/n}·F̂_{km, r−j}
                    let km = self.n_fft - k;
                    if j == 0 {
                        let packed = km - 1; // row 0 of mode km in Chunk 1
                        (packed, one)
                    } else {
                        let phase =
                            Complex::new(0.0_f64, -PI * (km as f64) / (self.n as f64)).exp();
                        let packed = (self.r - j) * (self.n - 1) + (km - 1);
                        (packed, phase)
                    }
                }
            }
        }
    }

    /// GLL nodes in x-increasing order on `[xl, xl+l]`, matching
    /// [`solver::PoissonSolver1D::get_x`].
    ///
    /// - `Periodic`: `n * r` nodes — left endpoint included, right excluded.
    /// - `Odd`:      `n * r − 1` nodes — both boundary endpoints excluded.
    /// - `Even`:     `n * r + 1` nodes — both boundary endpoints included.
    pub fn get_x(&self) -> Vec<f64> {
        let h = self.l / (self.n as f64);
        let mut x = Vec::with_capacity(self.n_dof);

        match self.extension {
            Extension::Periodic => {
                for k in 0..self.n {
                    for j in 0..self.r {
                        x.push(self.xl + h * ((k as f64) + self.gll_points[j]));
                    }
                }
                x
            }
            Extension::Odd => {
                // Skip xl (k=0,j=0) and xl+l (k=n-1,j=r): both are Dirichlet zeros.
                for j in 1..self.r {
                    x.push(self.xl + h * self.gll_points[j]);
                }
                for k in 1..self.n {
                    for j in 0..self.r {
                        x.push(self.xl + h * ((k as f64) + self.gll_points[j]));
                    }
                }
                x
            }
            Extension::Even => {
                // Element 0: all r+1 nodes; elements 1..n: skip shared left endpoint.
                for j in 0..=self.r {
                    x.push(self.xl + h * self.gll_points[j]);
                }
                for k in 1..self.n {
                    for j in 1..=self.r {
                        x.push(self.xl + h * ((k as f64) + self.gll_points[j]));
                    }
                }
                x
            }
        }
    }

    /// Evaluate the interpolant of `u` at position `x`.
    ///
    /// `u` must be in the x-increasing layout matching [`get_x`](Self::get_x).
    /// - `Periodic`: `x` is folded into `[xl, xl+l)` before lookup.
    /// - `Odd` (Dirichlet) / `Even` (Neumann): `x` must lie in `[xl, xl+l]`.
    pub fn eval(&self, x: f64, u: &[Complex<f64>]) -> Complex<f64> {
        let y = match self.extension {
            Extension::Periodic => (x - self.xl).rem_euclid(self.l) + self.xl,
            _ => {
                assert!(
                    x >= self.xl && x <= self.xl + self.l,
                    "eval: x={x} outside [{}, {}]",
                    self.xl,
                    self.xl + self.l
                );
                x
            }
        };
        let h = self.l / (self.n as f64);
        let k = (((y - self.xl) / h) as usize).min(self.n - 1);
        let xi = (y - self.xl - (k as f64) * h) / h;
        let phi = lagrangian_interpolation_all(&self.gll_points, &self.bary_weights, xi);

        let mut val = Complex::new(0.0_f64, 0.0);
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
                        Complex::default()
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

    /// Apply a flow map `lambda` to every grid node and interpolate `u` at the resulting positions.
    ///
    /// Returns a `Vec<Complex<f64>>` of the same length as `u`.
    pub fn convect<F>(&self, lambda: F, u: &[Complex<f64>]) -> Vec<Complex<f64>>
    where
        F: Fn(f64) -> f64,
    {
        self.get_x()
            .iter()
            .map(|&x| self.eval(lambda(x), u))
            .collect()
    }

    /// Returns the flat DOF index for local node `j` (0..=r) of element `k` (0..n).
    ///
    /// Used by `HofftPlannerND::eval` to gather local patch values.
    ///
    /// - `Periodic`: right endpoint (j==r) wraps to next element's j=0.
    /// - `Odd`:      left boundary (k=0,j=0) and right boundary (k=n-1,j=r) are Dirichlet zeros,
    ///               returned as `n_dof` (sentinel for "zero, not stored").
    ///               Otherwise: flat index `k*r + j - 1`.
    /// - `Even`:     flat index `k*r + j`.
    pub(crate) fn eval_dof_index(&self, k: usize, j: usize) -> usize {
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

    /// Extend 'f' if need be in a buffer of size 'r * n_fft' then apply the forward FFT
    /// after dof re-ordering,  return then the `n_dof` independent Fourier modes.
    /// The input `f` must be in x-increasing layout with the length matching the extension
    /// type (see [`Extension`]).  The input is  complex-valued for general purposes.
    /// The output is packed.
    pub fn forward(&self, f: &mut [Complex<f64>]) {
        debug_assert_eq!(f.len(), self.n_dof);

        let mut buf = vec![Complex::<f64>::default(); self.r * self.n_fft];

        match self.extension {
            Extension::Periodic => {
                for k in 0..self.n {
                    for j in 0..self.r {
                        buf[j * self.n_fft + k] = f[k * self.r + j];
                    }
                }
            }
            Extension::Odd => {
                // Odd (anti-symmetric) extension on 2n elements.
                // First half  k ∈ [0, n): j=0, k=0 → 0 (left Dirichlet); else → f[k*r+j−1]
                // Second half k ∈ [n,2n): j=0, k=n → 0 (right Dirichlet); else → −f[(2n−k)*r−j−1]
                for j in 0..self.r {
                    for k in 0..self.n_fft {
                        buf[j * self.n_fft + k] = if k < self.n {
                            if k == 0 && j == 0 {
                                Complex::default()
                            } else {
                                f[k * self.r + j - 1]
                            }
                        } else if k == self.n && j == 0 {
                            Complex::default()
                        } else {
                            -f[(self.n_fft - k) * self.r - j - 1]
                        };
                    }
                }
            }
            Extension::Even => {
                // Even extension on 2n elements.
                //   First half  k ∈ [0, n): f[k*r+j].
                //   Second half k ∈ [n,2n): f[(2n−k)*r−j].
                for j in 0..self.r {
                    for k in 0..self.n_fft {
                        buf[j * self.n_fft + k] = if k < self.n {
                            f[k * self.r + j]
                        } else {
                            f[(self.n_fft - k) * self.r - j]
                        };
                    }
                }
            }
        }

        self.fw_fft.process(&mut buf);

        // Compact extraction using symmetry properties.
        match self.extension {
            //Do not use k_packing here for efficiency
            Extension::Periodic => {
                for j in 0..self.r {
                    f[j * self.n..j * self.n + self.n]
                        .copy_from_slice(&buf[j * self.n_fft..j * self.n_fft + self.n]);
                }
            }
            Extension::Odd | Extension::Even => {
                for i in 0..self.n_dof {
                    f[i] = buf[self.k_packing(i)];
                }
            }
        }
    }

    /// Inverse HoFFT, the input layout should de packed.
    /// Return the dofs in x-increasing layout as complex values.
    ///
    /// The input `g` must have length `n_dof` (matching the output of [`forward`](Self::forward)).  
    pub fn inverse(&self, g: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.n_dof);

        let mut buf = vec![Complex::<f64>::default(); self.r * self.n_fft];

        let norm = self.n_fft as f64;

        match self.extension {
            Extension::Periodic => {
                buf.copy_from_slice(g);
            }
            Extension::Odd | Extension::Even => {
                for i in 0..self.r * self.n_fft {
                    let (i_g, coeff_g) = self.unpack_table[i];
                    buf[i] = coeff_g * g[i_g];
                }
            }
        }

        self.i_fft.process(&mut buf);

        match self.extension {
            Extension::Periodic => {
                for k in 0..self.n {
                    for j in 0..self.r {
                        g[k * self.r + j] = buf[j * self.n + k] / norm;
                    }
                }
            }
            Extension::Odd => {
                for j in 0..self.r {
                    for k in 0..self.n {
                        if k == 0 && j == 0 {
                            continue;
                        }
                        g[k * self.r + j - 1] = buf[j * self.n_fft + k] / norm;
                    }
                }
            }
            Extension::Even => {
                for j in 0..self.r {
                    for k in 0..self.n {
                        g[k * self.r + j] = buf[j * self.n_fft + k] / norm;
                    }
                }
                g[self.n * self.r] = buf[self.n] / norm;
            }
        }
    }
}

// ── ND ──────────────────────────────────────────────────────────────────────

/// N-dimensional HoFFT transform planner.
///
/// Applies forward/inverse FFTs direction by direction (sum factorisation),
/// corresponding to steps 2 and 4 of the ND HoFFT algorithm.  Each direction
/// has its own [`HofftPlanner`] with an independent [`Extension`] type, enabling
/// mixed periodic/Dirichlet/Neumann boundary conditions.
///
/// Both the physical and frequency buffers have the same length `total = ∏_d ndofs[d]`
/// (the compact layout produced by each 1D [`HofftPlanner`] preserves the DOF count).
/// [`forward`](Self::forward) and [`inverse`](Self::inverse) both operate in place.
pub struct HofftPlannerND<const N: usize> {
    /// One 1D planner per spatial direction.
    pub planners: [HofftPlanner; N],
    /// Physical (= frequency) DOF counts per direction: `ndofs[d] = planners[d].n_dof`.
    pub ndofs: [usize; N],
    /// Total DOFs: `∏_d ndofs[d]`.
    pub total: usize,
    /// Column-major strides for the DOF tensor.
    pub strides: [usize; N],
    /// Polynomial orders per direction: `rs[d] = planners[d].r`.
    rs: [usize; N],
    /// Column-major strides for the `r`-tensor (used in `get_values`/`set_values`).
    r_strides: [usize; N],
    /// Total local nodes per element: `∏_d rs[d]`.
    r_total: usize,
    /// Full-buffer sizes per direction: `full_sizes[d] = rs[d] * planners[d].n_fft`.
    full_sizes: [usize; N],
    /// Column-major strides for the full buffer (used in `k_packing`/`k_unpacking`).
    full_strides: [usize; N],
}

impl<const N: usize> HofftPlannerND<N> {
    /// Create a planner for a tensor-product domain.
    ///
    /// - `ns[d]`:   number of physical elements in direction `d`.
    /// - `rs[d]`:   polynomial order (local nodes per element) in direction `d`.
    /// - `ls[d]`:   domain length in direction `d`.
    /// - `xls[d]`:  left boundary in direction `d`.
    /// - `exts[d]`: [`Extension`] type for direction `d`.
    pub fn new(
        ns: [usize; N],
        rs: [usize; N],
        ls: [f64; N],
        xls: [f64; N],
        exts: [Extension; N],
    ) -> Self {
        let planners: [HofftPlanner; N] =
            std::array::from_fn(|d| HofftPlanner::new(ns[d], rs[d], ls[d], xls[d], exts[d]));
        let ndofs: [usize; N] = std::array::from_fn(|d| planners[d].n_dof);
        let total = ndofs.iter().product();
        let strides = nd_col_major_strides(&ndofs);
        let rs_arr: [usize; N] = std::array::from_fn(|d| planners[d].r);
        let r_strides = nd_col_major_strides(&rs_arr);
        let r_total = rs_arr.iter().product();
        let full_sizes: [usize; N] =
            std::array::from_fn(|d| planners[d].r * planners[d].n_fft);
        let full_strides = nd_col_major_strides(&full_sizes);
        Self {
            planners,
            ndofs,
            total,
            strides,
            rs: rs_arr,
            r_strides,
            r_total,
            full_sizes,
            full_strides,
        }
    }

    /// Tensor-product GLL grid in x-increasing column-major order.
    ///
    /// Returns `Vec<[f64; N]>` of length `total = ∏_d ndofs[d]`.
    /// `result[p][d]` is the coordinate of point `p` in direction `d`.
    /// Direction 0 varies fastest; within each direction nodes are x-increasing.
    pub fn get_x(&self) -> Vec<[f64; N]> {
        let x1d: [Vec<f64>; N] = std::array::from_fn(|d| self.planners[d].get_x());
        (0..self.total)
            .map(|p| std::array::from_fn(|d| x1d[d][(p / self.strides[d]) % self.ndofs[d]]))
            .collect()
    }

    /// Maps a compact ND flat index `i ∈ [0, total)` to the corresponding index
    /// in the full buffer `∏_d (r_d × n_fft_d)`.
    ///
    /// Each direction is handled independently: the per-direction compact index
    /// `c_d` is extracted, the 1D [`HofftPlanner::k_packing`] maps it to the
    /// full 1D index `f_d`, and the result is assembled using full-buffer strides.
    pub fn k_packing(&self, i: usize) -> usize {
        (0..N).fold(0, |acc, d| {
            let c_d = (i / self.strides[d]) % self.ndofs[d];
            acc + self.planners[d].k_packing(c_d) * self.full_strides[d]
        })
    }

    /// Maps a full-buffer ND flat index `i ∈ [0, ∏_d (r_d × n_fft_d))` to
    /// `(compact_index, coefficient)` such that `full_buf[i] = coefficient * compact_buf[compact_index]`.
    ///
    /// Each direction is handled independently via the 1D [`HofftPlanner::k_unpacking`];
    /// the overall coefficient is the product of the per-direction coefficients.
    pub fn k_unpacking(&self, i: usize) -> (usize, Complex<f64>) {
        let mut compact_idx = 0usize;
        let mut coeff = Complex::new(1.0_f64, 0.0);
        for d in 0..N {
            let f_d = (i / self.full_strides[d]) % self.full_sizes[d];
            let (c_d, coeff_d) = self.planners[d].unpack_table[f_d];
            compact_idx += c_d * self.strides[d];
            coeff *= coeff_d;
        }
        (compact_idx, coeff)
    }

    /// Extract the `∏_d r_d` local-node values for ND Fourier mode `ks` from the
    /// compact buffer `g` into `g_k`.
    ///
    /// `ks[d]` may be any mode in `0..n_fft_d`; modes in the symmetric half are
    /// recovered via the per-direction [`HofftPlanner::k_unpacking`], with the
    /// overall coefficient being the product of the per-direction coefficients.
    ///
    /// `g_k` must have length `∏_d r_d`, indexed column-major (direction 0 fastest).
    pub fn get_values(&self, ks: &[usize; N], g: &[Complex<f64>], g_k: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.total);
        debug_assert_eq!(g_k.len(), self.r_total);

        for j_flat in 0..self.r_total {
            let mut compact_idx = 0usize;
            let mut coeff = Complex::new(1.0_f64, 0.0);
            for d in 0..N {
                let j_d = (j_flat / self.r_strides[d]) % self.rs[d];
                let (c_d, coeff_d) =
                    self.planners[d].unpack_table[j_d * self.planners[d].n_fft + ks[d]];
                compact_idx += c_d * self.strides[d];
                coeff *= coeff_d;
            }
            g_k[j_flat] = coeff * g[compact_idx];
        }
    }

    /// Write the `∏_d r_d` local-node values for ND Fourier mode `ks` from `g_k`
    /// into the compact buffer `g`.
    ///
    /// Only the independently stored entries are written: for each local-node
    /// combination `(j_0, …, j_{N-1})`, the entry is stored only if, for every
    /// direction `d`, the 1D compact index `c_d` obtained from `k_unpacking` is
    /// canonical (i.e., `k_packing(c_d) == j_d * n_fft_d + ks[d]`).
    /// Derived entries (symmetric mirrors, zeros) are skipped.
    ///
    /// `ks[d]` must be an independent mode in each direction (`ks[d] ≤ n_d` for
    /// Odd/Even, `ks[d] < n_d` for Periodic).
    pub fn set_values(&self, ks: &[usize; N], g: &mut [Complex<f64>], g_k: &[Complex<f64>]) {
        debug_assert_eq!(g.len(), self.total);
        debug_assert_eq!(g_k.len(), self.r_total);

        for j_flat in 0..self.r_total {
            let mut canonical = true;
            let mut compact_idx = 0usize;
            for d in 0..N {
                let j_d = (j_flat / self.r_strides[d]) % self.rs[d];
                let full_1d = j_d * self.planners[d].n_fft + ks[d];
                let (c_d, _) = self.planners[d].unpack_table[full_1d];
                if !self.planners[d].canonical_table[full_1d] {
                    canonical = false;
                    break;
                }
                compact_idx += c_d * self.strides[d];
            }
            if canonical {
                g[compact_idx] = g_k[j_flat];
            }
        }
    }

    /// Evaluate the ND interpolant of `u` at a batch of positions.
    ///
    /// - `xs`: query points (length `m`); each `xs[i]` is a coordinate array `[f64; N]`.
    /// - `u`: DOF vector in the column-major tensor-product GLL layout from [`get_x`](Self::get_x).
    /// - `fu`: output buffer of length `m`; `fu[i]` receives the interpolated value at `xs[i]`.
    ///
    /// Uses sum factorisation and groups points by element for efficiency.
    pub fn eval(&self, xs: &[[f64; N]], u: &[Complex<f64>], fu: &mut [Complex<f64>]) {
        debug_assert_eq!(u.len(), self.total);
        debug_assert_eq!(fu.len(), xs.len());

        let ns: [usize; N] = std::array::from_fn(|d| self.planners[d].n);
        let rs: [usize; N] = std::array::from_fn(|d| self.planners[d].r);
        let ls: [f64; N] = std::array::from_fn(|d| self.planners[d].l);
        let xls: [f64; N] = std::array::from_fn(|d| self.planners[d].xl);
        let hs: [f64; N] = std::array::from_fn(|d| ls[d] / ns[d] as f64);
        let counts: [usize; N] = std::array::from_fn(|d| rs[d] + 1);
        let count_strides = nd_col_major_strides(&counts);
        let local_total: usize = counts.iter().product();

        // DOF strides in the "periodic-equivalent" layout (n_d * r_d per direction).
        // For Odd/Even, the actual layout differs but the same formula holds because
        // the eval indexing below accounts for BCs via the per-direction eval logic.
        // We use the planner's x-increasing DOF strides directly from ndofs.
        let n_strides = nd_col_major_strides(&ns);

        // Step 1: compute flat element index for each point.
        let elem_indices: Vec<usize> = xs
            .par_iter()
            .map(|x| {
                (0..N).fold(0, |acc, d| {
                    let y = (x[d] - xls[d]).rem_euclid(ls[d]);
                    let k_d = ((y / hs[d]) as usize).min(ns[d] - 1);
                    acc + k_d * n_strides[d]
                })
            })
            .collect();

        // Step 2: group point indices by element.
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &k_flat) in elem_indices.iter().enumerate() {
            groups.entry(k_flat).or_default().push(i);
        }

        // Step 3: parallel over elements — gather once, sum-factorise per point.
        let partial: Vec<Vec<(usize, Complex<f64>)>> = groups
            .into_par_iter()
            .map(|(k_flat, indices)| {
                let ks: [usize; N] = std::array::from_fn(|d| (k_flat / n_strides[d]) % ns[d]);

                // Gather u into contiguous local patch (once per element group).
                let u_local: Vec<Complex<f64>> = (0..local_total)
                    .map(|p| {
                        // Compute global DOF flat index for local node p.
                        // Each direction contributes dof_d * stride_d; if any direction
                        // returns the sentinel (n_dof = Dirichlet zero), the value is 0.
                        let mut global = 0usize;
                        let mut is_zero = false;
                        for d in 0..N {
                            let j = (p / count_strides[d]) % counts[d];
                            let dof_d = self.planners[d].eval_dof_index(ks[d], j);
                            if dof_d == self.planners[d].n_dof {
                                is_zero = true;
                                break;
                            }
                            global += dof_d * self.strides[d];
                        }
                        if is_zero {
                            Complex::default()
                        } else {
                            u[global]
                        }
                    })
                    .collect();

                let result: Vec<(usize, Complex<f64>)> = indices
                    .iter()
                    .map(|&i| {
                        let phi: [Vec<f64>; N] = std::array::from_fn(|d| {
                            let y = match self.planners[d].extension {
                                Extension::Periodic => {
                                    (xs[i][d] - xls[d]).rem_euclid(ls[d]) + xls[d]
                                }
                                _ => xs[i][d],
                            };
                            let xi = (y - xls[d] - (ks[d] as f64) * hs[d]) / hs[d];
                            lagrangian_interpolation_all(
                                &self.planners[d].gll_points,
                                &self.planners[d].bary_weights,
                                xi,
                            )
                        });

                        // Sum-factored contraction: start from dimension 0.
                        let c0 = counts[0];
                        let rest0 = local_total / c0;
                        let mut v: Vec<Complex<f64>> = (0..rest0)
                            .map(|s| {
                                (0..c0)
                                    .map(|j| phi[0][j] * u_local[s * c0 + j])
                                    .sum::<Complex<f64>>()
                            })
                            .collect();

                        for d in 1..N {
                            let c_d = counts[d];
                            let next = v.len() / c_d;
                            v = (0..next)
                                .map(|s| {
                                    (0..c_d)
                                        .map(|j| phi[d][j] * v[s * c_d + j])
                                        .sum::<Complex<f64>>()
                                })
                                .collect();
                        }
                        (i, v[0])
                    })
                    .collect();
                result
            })
            .collect();

        for group in partial {
            for (i, val) in group {
                fu[i] = val;
            }
        }
    }

    /// Apply a flow map `lambda` to every DOF node and interpolate `u` at the resulting positions.
    ///
    /// Returns a `Vec<Complex<f64>>` of length equal to `u`.
    pub fn convect<F>(&self, lambda: F, u: &[Complex<f64>]) -> Vec<Complex<f64>>
    where
        F: Fn([f64; N]) -> [f64; N] + Sync,
    {
        let xs = self.get_x();
        let shifted: Vec<[f64; N]> = xs.par_iter().map(|&pt| lambda(pt)).collect();
        let mut out = vec![Complex::new(0.0_f64, 0.0); u.len()];
        self.eval(&shifted, u, &mut out);
        out
    }

    /// Apply forward HoFFT direction by direction, in place.
    ///
    /// `f` must have length `total`. Each direction is transformed independently
    /// using the corresponding 1D [`HofftPlanner`]; the compact layout preserves
    /// the buffer length, so no reallocation occurs.
    pub fn forward(&self, f: &mut [Complex<f64>]) {
        debug_assert_eq!(f.len(), self.total);
        let sizes = self.ndofs.as_slice();
        let strides = self.strides.as_slice();

        for d in 0..N {
            let dim_d = self.ndofs[d];
            let n_fibers = self.total / dim_d;
            let stride_d = self.strides[d];
            // Cast to usize (Copy + Send) to share the pointer across threads.
            // SAFETY: fibers at distinct `s` access non-overlapping indices, so
            // parallel mutable access is sound.
            let ptr = f.as_mut_ptr() as usize;
            (0..n_fibers).into_par_iter().for_each(|s| {
                let base = nd_fiber_base(s, d, sizes, strides);
                let ptr = ptr as *mut Complex<f64>;
                let mut fiber: Vec<Complex<f64>> = (0..dim_d)
                    .map(|m| unsafe { *ptr.add(base + m * stride_d) })
                    .collect();
                self.planners[d].forward(fiber.as_mut_slice());
                for m in 0..dim_d {
                    unsafe { *ptr.add(base + m * stride_d) = fiber[m] };
                }
            });
        }
    }

    /// Apply inverse HoFFT direction by direction, in place.
    ///
    /// `g` must have length `total`. Each direction is transformed independently
    /// using the corresponding 1D [`HofftPlanner`].
    pub fn inverse(&self, g: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.total);
        let sizes = self.ndofs.as_slice();
        let strides = self.strides.as_slice();

        for d in 0..N {
            let dim_d = self.ndofs[d];
            let n_fibers = self.total / dim_d;
            let stride_d = self.strides[d];
            // SAFETY: same non-overlapping fiber argument as in `forward`.
            let ptr = g.as_mut_ptr() as usize;
            (0..n_fibers).into_par_iter().for_each(|s| {
                let base = nd_fiber_base(s, d, sizes, strides);
                let ptr = ptr as *mut Complex<f64>;
                let mut fiber: Vec<Complex<f64>> = (0..dim_d)
                    .map(|m| unsafe { *ptr.add(base + m * stride_d) })
                    .collect();
                self.planners[d].inverse(fiber.as_mut_slice());
                for m in 0..dim_d {
                    unsafe { *ptr.add(base + m * stride_d) = fiber[m] };
                }
            });
        }
    }
}

// ── ND helpers ───────────────────────────────────────────────────────────────

// Column-major strides for a fixed-size shape array.
fn nd_col_major_strides<const N: usize>(sizes: &[usize; N]) -> [usize; N] {
    let mut st = [1usize; N];
    for d in 1..N {
        st[d] = st[d - 1] * sizes[d - 1];
    }
    st
}

// Start offset of the s-th fiber along direction `d` in a tensor with given sizes/strides.
//
// `s` runs over [0, total/sizes[d]) by enumerating all index combinations for
// dimensions other than `d` in column-major order (d'=0 varies fastest, skipping d).
fn nd_fiber_base(s: usize, d: usize, sizes: &[usize], strides: &[usize]) -> usize {
    let mut base = 0usize;
    let mut rem = s;
    for d2 in 0..sizes.len() {
        if d2 == d {
            continue;
        }
        base += (rem % sizes[d2]) * strides[d2];
        rem /= sizes[d2];
    }
    base
}
