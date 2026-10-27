//! HoFFT transform engine for Gauss-Lobatto spectral-element grids.
//!
//! Implements the forward and inverse **High-order FFT (HoFFT)** transforms of
//! Caforio & Imperiale (2019) \[1\] for multi-element Gauss-Lobatto grids.
//!
//! For `n` Gauss-Lobatto finite-elements of polynomial order `r` on an interval, the global DOF
//! vector has length `N_dof` (see table below) and can be decomposed as
//! `r` "rows" of `n` nodes (local node index `j` × element index `k`).  The key
//! observation is that, for the screened Poisson operator, the inter-element
//! coupling along each row has a (near-)circulant structure, so a single FFT of
//! length `n_fft` simultaneously decouples all `n_fft` Fourier
//! modes.  After the transform each mode `k` is independent and corresponds to
//! an `r × r` linear system — the **local symbol** $S(k)$.  The full ND
//! solve applies this direction by direction via sum factorisation; see
//! [`crate::solver`].
//!
//! ## Key types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Engine`] | 1D HoFFT engine: forward/inverse transform, DOF layout. |
//! | [`EngineND`] | ND HoFFT engine: applies [`Engine`] direction by direction. |
//! | [`Extension`] | Boundary extension (re-exported from [`lobatto_grid`]); selects the FFT extension strategy. |
//!
//! Each engine holds the [`lobatto_grid`] grid it is built on, exposed via
//! [`Engine::grid`] / [`EngineND::grid`]; interpolation and convection of DOF
//! vectors are provided by [`lobatto_grid::Grid1D`] / [`lobatto_grid::GridND`]
//! directly.
//!
//! ## Reference
//!
//! \[1\] Caforio, F. and Imperiale, S. (2019). *High-order discrete Fourier transform
//! for the solution of the Poisson equation*. SIAM Journal on Scientific Computing,
//! 41(5), A2747–A2771.
//!
//! ## Extension strategy
//!
//! Non-periodic BCs are handled by extending the `n`-element physical domain to a
//! `2n`-element periodic domain with odd or even symmetry, applying a single FFT of
//! length `2n`, then retaining only the `N_dof` independent Fourier
//! coefficients (Lemma 3.2 of \[1\]):
//!
//! | Variant    | `n_fft` | `N_dof` |
//! |------------|------------------|------------------|
//! | `Periodic` | `n`              | `n·r`            |
//! | `Odd`      | `2n`             | `n·r - 1`        |
//! | `Even`     | `2n`             | `n·r + 1`        |
//!
//! ## Data layout
//!
//! The **physical DOF buffer** (length `N_dof`) is in `x`-increasing order:
//!
//! | Extension   | Flat index of node `j` in element `k` |
//! |-------------|---------------------------------------|
//! | `Periodic`  | `k·r + j`                             |
//! | `Odd`       | `k·r + j - 1` (left boundary excluded) |
//! | `Even`      | `k·r + j`, plus the right endpoint at `n·r` |
//!
//! The **internal transform buffer** stores `r` rows of `n_fft` complex
//! values: entry `g[j*n_fft + k]` is local node `j`, Fourier mode
//! `k`.  After [`Engine::forward`] this is compacted to `N_dof`
//! independent coefficients by discarding modes determined by the spectral symmetry.

use lobatto::collocation::{CollocationBasis, Gauss};
use lobatto_grid::grid::{nd_col_major_strides, nd_fiber_base, nd_multi_indices};
use lobatto_grid::{Grid1D, GridND};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

/// Boundary extension (re-exported from [`lobatto_grid`]).
///
/// Selects the FFT extension strategy of an [`Engine`]: non-periodic boundary
/// conditions are handled by extending the `n`-element physical domain to a
/// `2n`-element periodic domain with the appropriate symmetry, then applying a
/// single FFT of length `n_fft` and discarding the redundant Fourier
/// coefficients.
///
/// | Variant    | BC          | `n_fft` | `N_dof` |
/// |------------|-------------|------------------|------------------|
/// | `Periodic` | Periodic    | `n`              | `n·r`            |
/// | `Odd`      | Dirichlet   | `2n`             | `n·r - 1`        |
/// | `Even`     | Neumann     | `2n`             | `n·r + 1`        |
pub use lobatto_grid::Extension;

// ── 1D ──────────────────────────────────────────────────────────────────────

/// Low-level 1D HoFFT engine (periodic, no extension).
///
/// Applies `r` independent forward/inverse FFTs of length `n` to the buffer
/// layout $[r \text{ rows} \times n \text{ columns}]$, where entry `g[j * n + k]`
/// corresponds to local node `j` of element `k`.
///
/// This is used internally by [`crate::solver::Poisson`] for the periodic case and
/// can be reused for advection or other operations that require direct access to
/// the per-mode data without the extension/packing overhead of [`Engine`].
pub struct RawEngine {
    /// Number of elements `n` (= FFT length).
    n: usize,
    /// Polynomial order `r` (nodes per element, excluding shared endpoint).
    r: usize,
    /// Domain length `L`.
    l: f64,
    /// Left boundary position `x_l`.
    xl: f64,
    /// Gauss-Lobatto nodes on `[0, 1]` for a single element (length `r + 1`).
    gauss_lobatto_points: Vec<f64>,
    /// Forward FFT engine of length `n`.
    fw_fft: Arc<dyn Fft<f64>>,
    /// Inverse FFT engine of length `n`.
    i_fft: Arc<dyn Fft<f64>>,
}

impl RawEngine {
    /// Create a periodic 1D HoFFT engine for `r` independent FFTs of length `n` on `[x_l, x_l + L]`.
    pub fn new(n: usize, r: usize, l: f64, xl: f64) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fw_fft = planner.plan_fft_forward(n);
        let i_fft = planner.plan_fft_inverse(n);
        let gauss_lobatto_points = CollocationBasis::new(vec![(r + 1, Gauss::Lobatto)])
            .points_1d(0)
            .to_vec();
        Self {
            n,
            r,
            l,
            xl,
            gauss_lobatto_points,
            fw_fft,
            i_fft,
        }
    }

    /// Return the Gauss-Lobatto nodes on `[x_l, x_l + L]`, length `n·r`, ordered by local
    /// node then element: `x[j * n + k]` is the position of local node `j` in element `k`.
    pub fn get_x(&self) -> Vec<f64> {
        let h = self.l / (self.n as f64);
        let mut x = vec![0.0_f64; self.n * self.r];
        for k in 0..self.n {
            for j in 0..self.r {
                x[j * self.n + k] = h * ((k as f64) + self.gauss_lobatto_points[j]) + self.xl;
            }
        }
        x
    }

    /// Apply `r` forward FFTs of length `n` to `g` in place.
    ///
    /// `g` must have length `r·n`.  The buffer is treated as `r` consecutive
    /// rows of `n` elements; each row is transformed independently.
    /// No normalisation is applied ($\hat{g}_k = \sum_j g_j e^{-2\pi i jk/n}$).
    pub fn forward(&self, g: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.r * self.n);
        self.fw_fft.process(g);
    }

    /// Apply `r` inverse FFTs of length `n` to `g` in place, then normalise by `1/n`.
    ///
    /// `g` must have length `r·n`.  The buffer is treated as `r` consecutive
    /// rows; each row is transformed independently
    /// ($g_j = \frac{1}{n}\sum_k \hat{g}_k e^{2\pi i jk/n}$).
    pub fn inverse(&self, g: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.r * self.n);
        let norm = self.n as f64;
        self.i_fft.process(g);
        for v in g.iter_mut() {
            *v /= norm;
        }
    }
}

/// 1D HoFFT engine with odd/even extension support.
///
/// Handles extension, forward/inverse FFT, and the packing/unpacking of the
/// compact DOF buffer.  All physical DOFs are stored in **`x`-increasing order**:
///
/// | Extension | `N_dof` | Layout |
/// |-----------|-----------------|--------|
/// | `Periodic` | `n·r` | flat index `k·r + j`: local node `j` of element `k` |
/// | `Odd` (Dirichlet) | `n·r - 1` | same, but left boundary node excluded; flat index `k·r + j - 1` |
/// | `Even` (Neumann)  | `n·r + 1` | same as Periodic plus the right boundary node |
///
/// ## Packing and unpacking
///
/// After the forward FFT on the extended buffer of length `r·n_fft`,
/// the spectral symmetry (Lemma 3.2 of \[1\]) means only `N_dof` independent
/// Fourier coefficients need to be stored.  Two precomputed tables encode the bijection
/// between the compact and full representations:
///
/// - **Packing table**: maps compact index $i \in [0, N_\mathrm{dof})$ → full-buffer index.
///   Used by [`forward`](Self::forward) to extract independent coefficients.
/// - **Unpacking table**: maps full-buffer index `i` → `(compact_index, phase_coefficient)`.
///   Used by [`inverse`](Self::inverse) to reconstruct the full buffer from the compact one.
pub struct Engine {
    /// Grid geometry: elements, order, domain, extension, DOF layout.
    grid: Grid1D,
    /// Number of physical elements `n`.
    n: usize,
    /// FFT length: `n` for `Periodic`, `2n` for `Odd`/`Even`.
    n_fft: usize,
    /// Number of physical DOFs `N_dof`.
    n_dof: usize,
    /// Polynomial order `r`.
    r: usize,
    /// Extension type.
    extension: Extension,
    /// Forward FFT engine of length `n_fft`.
    fw_fft: Arc<dyn Fft<f64>>,
    /// Inverse FFT engine of length `n_fft`.
    i_fft: Arc<dyn Fft<f64>>,
    /// Unpacking table: `unpack_table[j * n_fft + k] = (packed_idx, coeff)` such that
    /// `full_buf[j * n_fft + k] = coeff * compact_buf[packed_idx]`.
    /// Encodes the symmetry relations from Lemma 3.2 of \[1\].
    unpack_table: Vec<(usize, Complex<f64>)>,
    /// Canonicality table: `canonical_table[j * n_fft + k] = true` iff entry `(j, k)`
    /// is an independently stored coefficient (not derived by symmetry or forced zero).
    canonical_table: Vec<bool>,
    /// Packing table: `packing_table[i] = full_buffer_index` for compact index `i`.
    /// Used by `forward` to copy independent coefficients into the compact buffer.
    /// Empty for `Periodic` (trivial mapping).
    packing_table: Vec<usize>,
}

impl Engine {
    /// Create a 1D HoFFT engine for `n` elements of order `r` on `[x_l, x_l + L]`.
    ///
    /// Precomputes the forward/inverse FFT plans of length `n_fft`
    /// and the packing/unpacking tables for the given [`Extension`] type.
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
        let grid = Grid1D::new(n, r, l, xl, extension);
        debug_assert_eq!(grid.n_dof(), n_dof);
        let mut s = Self {
            grid,
            n,
            n_fft,
            n_dof,
            r,
            extension,
            fw_fft,
            i_fft,
            unpack_table: vec![],
            canonical_table: vec![],
            packing_table: vec![],
        };
        let table_size = s.r * s.n_fft;
        s.unpack_table = (0..table_size).map(|i| s.k_unpacking(i)).collect();
        s.canonical_table = (0..table_size)
            .map(|i| {
                let (c, _) = s.k_unpacking(i);
                s.k_packing(c) == i
            })
            .collect();
        if extension != Extension::Periodic {
            s.packing_table = (0..n_dof).map(|i| s.k_packing(i)).collect();
        }
        s
    }

    /// Extract the `r` values for Fourier mode `k` from the packed buffer `g` into `g_k`.
    ///
    /// `k` may be any mode in `0..n_fft`; modes in the
    /// symmetric half (`k > n`) are recovered via Lemma 3.2 of \[1\] and the
    /// cross-row symmetries of the packed representation.
    pub fn get_values(&self, k: usize, g: &[Complex<f64>], g_k: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.n_dof);
        debug_assert_eq!(g_k.len(), self.r);
        for (j, v) in g_k.iter_mut().enumerate() {
            let (i_g, coeff) = self.unpack_table[j * self.n_fft + k];
            *v = coeff * g[i_g];
        }
    }

    /// Write the `r` values for Fourier mode `k` from `g_k` into the packed buffer `g`.
    ///
    /// Only independently stored entries are written; the symmetric mirror modes
    /// (`k > n`) are derived and must not be set directly.
    /// `k` must be in `0..=n`.
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
                    let m = self.r / 2;
                    g[base..=base + m].copy_from_slice(&g_k[..=m]);
                } else {
                    // k == n — Chunk 3: j=0..⌊(r−1)/2⌋ (j=r/2 zero if r even, rest derived).
                    let base = self.r * (self.n - 1) + self.r / 2 + 1;
                    let m = (self.r - 1) / 2;
                    g[base..=base + m].copy_from_slice(&g_k[..=m]);
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
                    if self.r.is_multiple_of(2) && j == self.r / 2 {
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
                    if self.r.is_multiple_of(2) && j == self.r / 2 {
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

    /// The [`Grid1D`] this engine is built on.
    ///
    /// Interpolation and convection of DOF vectors ([`Grid1D::eval`],
    /// [`Grid1D::eval_extended`], [`Grid1D::convect`], …) are provided by the
    /// grid directly.
    pub fn grid(&self) -> &Grid1D {
        &self.grid
    }

    /// Gauss-Lobatto nodes in x-increasing order on `[xl, xl+l]`, matching
    /// [`crate::solver::Poisson::get_x`].
    ///
    /// - `Periodic`: `n * r` nodes — left endpoint included, right excluded.
    /// - `Odd`:      `n * r − 1` nodes — both boundary endpoints excluded.
    /// - `Even`:     `n * r + 1` nodes — both boundary endpoints included.
    pub fn get_x(&self) -> Vec<f64> {
        self.grid.get_x()
    }

    /// Forward HoFFT: extend, reorder, FFT, and compact in place.
    ///
    /// On input, `f` holds the physical DOFs in `x`-increasing order, length `N_dof`.
    /// The function:
    /// 1. Builds the extended buffer of length `r·n_fft` (odd or even
    ///    extension, or identity for Periodic).
    /// 2. Applies the forward FFT of length `n_fft` along each of the `r` rows.
    /// 3. Compacts the result to the `N_dof` independent Fourier coefficients
    ///    using the packing table.
    ///
    /// On output, `f` contains the packed frequency-domain representation.
    pub fn forward(&self, f: &mut [Complex<f64>]) {
        let (buf_len, scratch_len) = self.work_lens();
        let zero = Complex::<f64>::default();
        self.forward_with(f, &mut vec![zero; buf_len], &mut vec![zero; scratch_len]);
    }

    /// Lengths of the work buffers of [`forward_with`](Self::forward_with) and
    /// [`inverse_with`](Self::inverse_with): the extended buffer `r·n_fft`, and
    /// the in-place FFT scratch (the larger of the forward and inverse plans').
    pub(crate) fn work_lens(&self) -> (usize, usize) {
        let scratch = self.fw_fft.get_inplace_scratch_len().max(self.i_fft.get_inplace_scratch_len());
        (self.r * self.n_fft, scratch)
    }

    /// [`forward`](Self::forward) with the extended buffer and the FFT
    /// scratch supplied by the caller (lengths from
    /// [`work_lens`](Self::work_lens)): no allocation. Both are overwritten.
    pub(crate) fn forward_with(
        &self,
        f: &mut [Complex<f64>],
        buf: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
    ) {
        debug_assert_eq!(f.len(), self.n_dof);

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

        self.fw_fft.process_with_scratch(buf, scratch);

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
                    f[i] = buf[self.packing_table[i]];
                }
            }
        }
    }

    /// Inverse HoFFT: unpack, inverse FFT, reorder in place.
    ///
    /// On input, `g` holds the packed frequency-domain representation of length
    /// `N_dof` (as produced by [`forward`](Self::forward)).
    /// The function:
    /// 1. Expands to the full `r·n_fft` buffer using the unpacking table
    ///    (applies symmetry coefficients from Lemma 3.2 of \[1\]).
    /// 2. Applies the inverse FFT of length `n_fft` and normalises by `1/n_fft`.
    /// 3. Reorders to the `x`-increasing physical DOF layout.
    ///
    /// On output, `g` contains the physical DOFs in `x`-increasing order.
    pub fn inverse(&self, g: &mut [Complex<f64>]) {
        let (buf_len, scratch_len) = self.work_lens();
        let zero = Complex::<f64>::default();
        self.inverse_with(g, &mut vec![zero; buf_len], &mut vec![zero; scratch_len]);
    }

    /// [`inverse`](Self::inverse) with the extended buffer and the FFT
    /// scratch supplied by the caller (lengths from
    /// [`work_lens`](Self::work_lens)): no allocation. Both are overwritten.
    pub(crate) fn inverse_with(
        &self,
        g: &mut [Complex<f64>],
        buf: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
    ) {
        debug_assert_eq!(g.len(), self.n_dof);

        let norm = self.n_fft as f64;

        match self.extension {
            Extension::Periodic => {
                buf.copy_from_slice(g);
            }
            Extension::Odd | Extension::Even => {
                for (i, b) in buf.iter_mut().enumerate() {
                    let (i_g, coeff_g) = self.unpack_table[i];
                    *b = coeff_g * g[i_g];
                }
            }
        }

        self.i_fft.process_with_scratch(buf, scratch);

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

/// `N`-dimensional HoFFT transform engine.
///
/// Applies forward/inverse FFTs direction by direction (sum factorisation),
/// corresponding to the transform steps of the ND HoFFT algorithm.  Each direction
/// has its own 1D [`Engine`] with an independent [`Extension`] type, enabling
/// mixed periodic/Dirichlet/Neumann boundary conditions.
///
/// Both the physical and frequency buffers have the same length
/// `total = ∏_d ndofs[d]` (the compact layout produced by each 1D [`Engine`]
/// preserves the DOF count per direction).
/// [`forward`](Self::forward) and [`inverse`](Self::inverse) both operate in place.
pub struct EngineND<const N: usize> {
    /// One 1D engine per spatial direction.
    planners: [Engine; N],
    /// Grid geometry for the tensor-product domain.
    grid: GridND<N>,
    /// Physical (= frequency) DOF counts per direction: `ndofs[d] = planners[d].n_dof`.
    pub(crate) ndofs: [usize; N],
    /// Total DOFs: `∏_d ndofs[d]`.
    pub(crate) total: usize,
    /// Column-major strides for the DOF tensor.
    pub(crate) strides: [usize; N],
    /// Polynomial orders per direction: `rs[d] = planners[d].r`.
    pub(crate) rs: [usize; N],
    /// Column-major strides for the `r`-tensor (used in `get_values`/`set_values`).
    pub(crate) r_strides: [usize; N],
    /// Total local nodes per element: `∏_d rs[d]`.
    pub(crate) r_total: usize,
    /// Full-buffer sizes per direction: `full_sizes[d] = rs[d] * planners[d].n_fft`.
    full_sizes: [usize; N],
    /// Column-major strides for the full buffer (used in `k_packing`/`k_unpacking`).
    full_strides: [usize; N],
    /// Per direction, the `get_values` entries of each independent mode `k`
    /// (`0..n_d` for `Periodic`, `0..=n_d` otherwise), contiguous in `j`:
    /// `mode_entries[d][k * r_d + j] = (c · strides[d], coeff)`.
    mode_entries: [Vec<(usize, Complex<f64>)>; N],
    /// Per direction and independent mode `k`, the independently stored rows
    /// `(j, c · strides[d])`.
    mode_canonical: [Vec<Vec<(usize, usize)>>; N],
}

/// Compact-buffer addresses of the `∏_d r_d` local-node values of one ND
/// mode tuple, filled by [`EngineND::mode_indices`].
pub(crate) struct ModeIndices {
    /// Compact index of local node `j_flat`.
    pub(crate) offsets: Vec<usize>,
    /// Coefficient of local node `j_flat`: value `j_flat` is
    /// `coeffs[j_flat] * g[offsets[j_flat]]`, as in [`EngineND::get_values`].
    pub(crate) coeffs: Vec<Complex<f64>>,
    /// The independently stored entries `(j_flat, compact index)`, the ones
    /// [`EngineND::set_values`] writes.
    pub(crate) canonical: Vec<(usize, usize)>,
}

impl<const N: usize> EngineND<N> {
    /// Create an ND HoFFT engine for a tensor-product domain.
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
        let planners: [Engine; N] =
            std::array::from_fn(|d| Engine::new(ns[d], rs[d], ls[d], xls[d], exts[d]));
        let grid = GridND::new(ns, rs, ls, xls, exts);
        let ndofs: [usize; N] = std::array::from_fn(|d| planners[d].n_dof);
        let total = ndofs.iter().product();
        let strides = nd_col_major_strides(&ndofs);
        let rs_arr: [usize; N] = std::array::from_fn(|d| planners[d].r);
        let r_strides = nd_col_major_strides(&rs_arr);
        let r_total = rs_arr.iter().product();
        let full_sizes: [usize; N] = std::array::from_fn(|d| planners[d].r * planners[d].n_fft);
        let full_strides = nd_col_major_strides(&full_sizes);
        let independent_modes = |p: &Engine| if p.extension == Extension::Periodic { p.n } else { p.n + 1 };
        let mode_entries = std::array::from_fn(|d| {
            let p = &planners[d];
            (0..independent_modes(p))
                .flat_map(|k| {
                    (0..p.r).map(move |j| {
                        let (c, coeff) = p.unpack_table[j * p.n_fft + k];
                        (c * strides[d], coeff)
                    })
                })
                .collect()
        });
        let mode_canonical = std::array::from_fn(|d| {
            let p = &planners[d];
            (0..independent_modes(p))
                .map(|k| {
                    (0..p.r)
                        .filter(|&j| p.canonical_table[j * p.n_fft + k])
                        .map(|j| (j, p.unpack_table[j * p.n_fft + k].0 * strides[d]))
                        .collect()
                })
                .collect()
        });
        Self {
            planners,
            grid,
            ndofs,
            total,
            strides,
            rs: rs_arr,
            r_strides,
            r_total,
            full_sizes,
            full_strides,
            mode_entries,
            mode_canonical,
        }
    }

    /// An empty [`ModeIndices`] sized for this engine.
    pub(crate) fn mode_indices_buffer(&self) -> ModeIndices {
        ModeIndices {
            offsets: vec![0; self.r_total],
            coeffs: vec![Complex::new(0.0, 0.0); self.r_total],
            canonical: Vec::with_capacity(2 * self.r_total),
        }
    }

    /// Fill `m` with the compact-buffer addresses of the independent ND mode
    /// tuple `ks`: the same entries and coefficients as
    /// [`get_values`](Self::get_values) / [`set_values`](Self::set_values),
    /// bit for bit.
    pub(crate) fn mode_indices(&self, ks: &[usize; N], m: &mut ModeIndices) {
        // Directions combined one after the other: after direction d, the
        // first ∏_{e ≤ d} r_e entries hold the partial sums and products.
        m.offsets[0] = 0;
        m.coeffs[0] = Complex::new(1.0, 0.0);
        let mut len = 1;
        for ((&r, entries), &k) in self.rs.iter().zip(&self.mode_entries).zip(ks) {
            let entries = &entries[k * r..(k + 1) * r];
            // Block b is block 0 shifted by entry b; block 0 is rewritten in
            // place last.
            for (b, &(o, c)) in entries.iter().enumerate().rev() {
                for a in 0..len {
                    m.offsets[b * len + a] = m.offsets[a] + o;
                    m.coeffs[b * len + a] = m.coeffs[a] * c;
                }
            }
            len *= r;
        }

        m.canonical.clear();
        m.canonical.push((0, 0));
        for ((canonical, &k), &r_stride) in self.mode_canonical.iter().zip(ks).zip(&self.r_strides) {
            let prev = m.canonical.len();
            for i in 0..prev {
                let (j_flat, o) = m.canonical[i];
                for &(j, oj) in &canonical[k] {
                    m.canonical.push((j_flat + j * r_stride, o + oj));
                }
            }
            m.canonical.drain(..prev);
        }
    }

    /// The [`GridND`] this engine is built on.
    ///
    /// Interpolation and convection of DOF vectors ([`GridND::eval`],
    /// [`GridND::eval_extended`], [`GridND::convect`], the
    /// [`lobatto_grid::EvalPlan`] API, …) are provided by the grid directly.
    pub fn grid(&self) -> &GridND<N> {
        &self.grid
    }

    /// Tensor-product Gauss-Lobatto grid in x-increasing column-major order.
    ///
    /// Returns `Vec<[f64; N]>` of length `total = ∏_d ndofs[d]`.
    /// `result[p][d]` is the coordinate of point `p` in direction `d`.
    /// Direction 0 varies fastest; within each direction nodes are x-increasing.
    pub fn get_x(&self) -> Vec<[f64; N]> {
        self.grid.get_x()
    }

    /// Maps a compact ND flat index `i ∈ [0, total)` to the corresponding index
    /// in the full buffer `∏_d (r_d × n_fft_d)`.
    ///
    /// Each direction is handled independently: the per-direction compact index
    /// `c_d` is extracted, the 1D `k_packing` maps it to the
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
    /// Each direction is handled independently via the 1D `k_unpacking`;
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
    /// recovered via the per-direction `k_unpacking`, with the
    /// overall coefficient being the product of the per-direction coefficients.
    ///
    /// `g_k` must have length `∏_d r_d`, indexed column-major (direction 0 fastest).
    pub fn get_values(&self, ks: &[usize; N], g: &[Complex<f64>], g_k: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.total);
        debug_assert_eq!(g_k.len(), self.r_total);

        for (v, js) in g_k.iter_mut().zip(nd_multi_indices(self.rs)) {
            let mut compact_idx = 0usize;
            let mut coeff = Complex::new(1.0_f64, 0.0);
            for (d, (&k_d, &j_d)) in ks.iter().zip(&js).enumerate() {
                let (c_d, coeff_d) =
                    self.planners[d].unpack_table[j_d * self.planners[d].n_fft + k_d];
                compact_idx += c_d * self.strides[d];
                coeff *= coeff_d;
            }
            *v = coeff * g[compact_idx];
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

        for (&v, js) in g_k.iter().zip(nd_multi_indices(self.rs)) {
            let mut canonical = true;
            let mut compact_idx = 0usize;
            for (d, (&k_d, &j_d)) in ks.iter().zip(&js).enumerate() {
                let full_1d = j_d * self.planners[d].n_fft + k_d;
                let (c_d, _) = self.planners[d].unpack_table[full_1d];
                if !self.planners[d].canonical_table[full_1d] {
                    canonical = false;
                    break;
                }
                compact_idx += c_d * self.strides[d];
            }
            if canonical {
                g[compact_idx] = v;
            }
        }
    }

    /// Apply forward HoFFT direction by direction, in place.
    ///
    /// `f` must have length `total`. Each direction is transformed independently
    /// using the corresponding 1D [`Engine`]; the compact layout preserves
    /// the buffer length, so no reallocation occurs.
    pub fn forward(&self, f: &mut [Complex<f64>]) {
        self.sweep(f, true);
    }

    /// Apply inverse HoFFT direction by direction, in place.
    ///
    /// `g` must have length `total`. Each direction is transformed independently
    /// using the corresponding 1D [`Engine`].
    pub fn inverse(&self, g: &mut [Complex<f64>]) {
        self.sweep(g, false);
    }

    /// The 1D forward (`forward = true`) or inverse transform of every fibre,
    /// direction by direction, fibres in parallel.
    // Hot path: per-task buffers only (`for_each_init`), no allocation per
    // fibre; a per-fibre `Vec` here made it five times slower.
    fn sweep(&self, f: &mut [Complex<f64>], forward: bool) {
        debug_assert_eq!(f.len(), self.total);
        let zero = Complex::<f64>::default();
        for d in 0..N {
            let (dim_d, stride_d) = (self.ndofs[d], self.strides[d]);
            let engine = &self.planners[d];
            let (buf_len, scratch_len) = engine.work_lens();
            let transform = |fiber: &mut [Complex<f64>], buf: &mut [Complex<f64>], scratch: &mut [Complex<f64>]| {
                if forward {
                    engine.forward_with(fiber, buf, scratch);
                } else {
                    engine.inverse_with(fiber, buf, scratch);
                }
            };
            if stride_d == 1 {
                // Contiguous fibres: transformed in place, no gather/scatter.
                f.par_chunks_mut(dim_d).for_each_init(
                    || (vec![zero; buf_len], vec![zero; scratch_len]),
                    |(buf, scratch), fiber| transform(fiber, buf, scratch),
                );
                continue;
            }
            // Cast to usize (Copy + Send) to share the pointer across threads.
            // SAFETY: fibers at distinct `s` access non-overlapping indices
            // (tested in `solver::tests`), so parallel mutable access is sound.
            let ptr = f.as_mut_ptr() as usize;
            (0..self.total / dim_d).into_par_iter().for_each_init(
                || (vec![zero; dim_d], vec![zero; buf_len], vec![zero; scratch_len]),
                |(fiber, buf, scratch), s| {
                    let base = nd_fiber_base(s, d, &self.ndofs, &self.strides);
                    let ptr = ptr as *mut Complex<f64>;
                    for (m, v) in fiber.iter_mut().enumerate() {
                        *v = unsafe { *ptr.add(base + m * stride_d) };
                    }
                    transform(fiber, buf, scratch);
                    for (m, &v) in fiber.iter().enumerate() {
                        unsafe { *ptr.add(base + m * stride_d) = v };
                    }
                },
            );
        }
    }
}
