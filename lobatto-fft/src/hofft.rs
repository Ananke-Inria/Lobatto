//! High-order FFT (HoFFT) based on Gauss-Lobatto points and corresponding screened Poisson solver.
//!
//! Solves the screened Poisson equation
//! $$(\alpha M + \beta K)\, u = f$$
//! on $[x_l, x_l+L]^N$ using a GLL spectral-element discretisation with
//! **$n$ elements of polynomial order $r$ per direction**.
//!
//! The key idea is that the inter-element coupling pattern has a circulant structure
//! (for periodic BCs) or a near-circulant structure (for Dirichlet/Neumann BCs via
//! odd/even extension), so the $n$ inter-element coupling modes can be decoupled
//! by a single FFT of length $n$ (or $2n$).  After decoupling, each mode $k$
//! requires solving an $r \times r$ Hermitian system — the **local symbol**
//! $S(k)$ — whose eigendecomposition is precomputed.
//!
//! ## Reference
//!
//! \[1\] Caforio, F. and Imperiale, S. (2019). *High-order discrete Fourier transform
//! for the solution of the Poisson equation*. SIAM Journal on Scientific Computing,
//! 41(5), A2747–A2771.
//!
//! ## Boundary conditions
//!
//! Non-periodic boundary conditions are handled by **extending** the physical domain
//! to a $2n$-element periodic domain with the appropriate symmetry, as detailed in
//! the reference:
//!
//! | BC          | Extension | FFT length | $N_\mathrm{dof}$  |
//! |-------------|-----------|-----------|---------------------|
//! | `Periodic`  | none      | $n$       | $n r$               |
//! | `Dirichlet` | Odd       | $2n$      | $n r - 1$           |
//! | `Neumann`   | Even      | $2n$      | $n r + 1$           |
//!
//! The compacted/packed DOF buffer exploits the symmetry of the extended spectrum
//! to store only the independent Fourier coefficients.
//!
//! ## Data layout
//!
//! - **1D** (planner buffer): `r` blocks of `n` elements.
//!   Entry `g[j * n + k]` corresponds to local node $j \in \{0,\ldots,r-1\}$,
//!   element $k \in \{0,\ldots,n-1\}$.
//! - **ND** (planner buffer): flat length $\prod_d n_d r_d$, column-major
//!   (direction 0 varies fastest).
//! - **Physical DOF buffer** (`get_x`, `solve_in_place`): $x$-increasing order,
//!   flat index $k r + j$ for local node $j$ of element $k$ (Periodic and Neumann;
//!   offset by $-1$ for Dirichlet since the left boundary node is excluded).
//!
//! ## 1D algorithm
//!
//! 1. **Mass scaling** — $f \leftarrow M f$ (pointwise, diagonal mass matrix).
//! 2. **Extension + Forward FFT** — build the extended buffer, apply FFT of length
//!    $n_\mathrm{fft}$, compact the independent modes.
//! 3. **Symbol solve** — for each Fourier mode $k \in \{0,\ldots,n_\mathrm{fft}/2\}$,
//!    solve the $r \times r$ Hermitian system $S(k)\, \hat{u}_k = \hat{f}_k$ via the
//!    precomputed eigendecomposition $S(k) = V_k \Lambda_k V_k^\dagger$:
//!    $$\hat{u}_k = V_k \Lambda_k^{-1} V_k^\dagger \hat{f}_k.$$
//! 4. **Inverse FFT + unpack** — expand by symmetry, inverse FFT, reorder to
//!    $x$-increasing layout.
//!
//! The $N$-dimensional solver ([`PoissonSolverND`]) applies steps 2 and 4
//! direction by direction (sum factorisation) and solves an
//! $(r_0 \cdots r_{N-1}) \times (r_0 \cdots r_{N-1})$ block-diagonal system
//! in step 3 using tensor-product eigendecompositions.
//! Steps 2/4 are parallelised over independent fibres; step 3 is parallelised
//! over independent Fourier-mode tuples.

use lobatto::collocation::{CollocationBasis, Gauss};
use lobatto::utilities::{barycentric_weights, lagrangian_interpolation};
use nalgebra::*;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

// ── Extension ────────────────────────────────────────────────────────────────

/// Extension strategy for a [`HofftPlanner`].
///
/// Non-periodic boundary conditions are handled by extending the $n$-element physical
/// domain to a $2n$-element periodic domain with the appropriate symmetry, then
/// applying a single FFT of length $n_\mathrm{fft}$ and discarding the redundant
/// Fourier coefficients.
///
/// | Variant    | BC          | $n_\mathrm{fft}$ | $N_\mathrm{dof}$ |
/// |------------|-------------|------------------|------------------|
/// | `Periodic` | Periodic    | $n$              | $n r$            |
/// | `Odd`      | Dirichlet   | $2n$             | $n r - 1$        |
/// | `Even`     | Neumann     | $2n$             | $n r + 1$        |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension {
    /// Standard periodic FFT; no extension needed.
    Periodic,
    /// Odd (anti-symmetric) extension — enforces homogeneous Dirichlet values at both endpoints.
    Odd,
    /// Even (symmetric) extension — enforces zero normal derivative (Neumann) at both endpoints.
    Even,
}

/// Boundary condition type for the HoFFT solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Periodic: $u(x_l) = u(x_l+L)$.  Uses a plain FFT of length $n$.
    Periodic,
    /// Homogeneous Dirichlet: $u(x_l) = u(x_l+L) = 0$.  Uses an odd extension to length $2n$.
    Dirichlet,
    /// Homogeneous Neumann: $u'(x_l) = u'(x_l+L) = 0$.  Uses an even extension to length $2n$.
    Neumann,
}

impl BoundaryCondition {
    /// Convert to the corresponding [`Extension`] type.
    fn extension(self) -> Extension {
        match self {
            BoundaryCondition::Periodic => Extension::Periodic,
            BoundaryCondition::Dirichlet => Extension::Odd,
            BoundaryCondition::Neumann => Extension::Even,
        }
    }
}

// ── 1D ──────────────────────────────────────────────────────────────────────

/// Low-level 1D HoFFT planner (periodic, no extension).
///
/// Applies $r$ independent forward/inverse FFTs of length $n$ to the buffer
/// layout `[r rows × n columns]` where entry `g[j * n + k]` corresponds to
/// local node $j$ of element $k$.
///
/// This is used internally by [`PoissonSolver1D`] for the periodic case and
/// can be reused for advection or other operations requiring direct access to
/// the per-mode data.
pub struct RawPlanner {
    /// Number of elements $n$ (= FFT length).
    n: usize,
    /// Polynomial order $r$ (nodes per element, excluding shared endpoint).
    r: usize,
    /// Domain length $L$.
    l: f64,
    /// Left boundary position $x_l$.
    xl: f64,
    /// GLL nodes on $[0,1]$ for a single element (length $r+1$).
    gll_points: Vec<f64>,
    /// Forward FFT engine of length $n$.
    fw_fft: Arc<dyn Fft<f64>>,
    /// Inverse FFT engine of length $n$.
    i_fft: Arc<dyn Fft<f64>>,
}

impl RawPlanner {
    /// Create a planner for $r$ independent FFTs of length $n$ on $[x_l, x_l+L]$.
    pub fn new(n: usize, r: usize, l: f64, xl: f64) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fw_fft = planner.plan_fft_forward(n);
        let i_fft = planner.plan_fft_inverse(n);
        let gll_points = CollocationBasis::new(vec![(r + 1, Gauss::Lobatto)]).points_1d(0).to_vec();
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

    /// Return the GLL nodes on $[x_l, x_l+L]$, length $n r$, ordered by local
    /// node then element: `x[j * n + k]` is the position of local node $j$ in element $k$.
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

    /// Apply $r$ forward FFTs of length $n$ to `g` in place.
    ///
    /// `g` must have length $r n$.  The buffer is treated as $r$ consecutive
    /// rows of $n$ elements; each row is transformed independently.
    /// No normalisation is applied ($\hat{g}_k = \sum_j g_j e^{-2\pi i jk/n}$).
    pub fn forward(&self, g: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.r * self.n);
        self.fw_fft.process(g);
    }

    /// Apply $r$ inverse FFTs of length $n$ to `g` in place, then normalise by $1/n$.
    ///
    /// `g` must have length $r n$.  The buffer is treated as $r$ consecutive
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

/// 1D HoFFT planner with odd/even extension support.
///
/// Handles extension, forward/inverse FFT, and the packing/unpacking of the
/// compact DOF buffer.  All physical DOFs are stored in **$x$-increasing order**:
///
/// | Extension | $N_\mathrm{dof}$ | Layout |
/// |-----------|-----------------|--------|
/// | `Periodic` | $n r$ | flat index $k r + j$: local node $j$ of element $k$ |
/// | `Odd` (Dirichlet) | $n r - 1$ | same, but left boundary node excluded; flat index $k r + j - 1$ |
/// | `Even` (Neumann)  | $n r + 1$ | same as Periodic plus the right boundary node |
///
/// ## Packing and unpacking
///
/// After the forward FFT on the extended buffer of length $r \cdot n_\mathrm{fft}$,
/// the spectral symmetry (odd anti-symmetry, even symmetry; Lemma 3.2 of \[1\])
/// means only $N_\mathrm{dof}$ independent coefficients need to be stored.
/// The precomputed tables [`unpack_table`](Self) and [`packing_table`](Self)
/// encode the bijection between the compact and full buffers.
pub struct HofftPlanner {
    /// Number of physical elements $n$.
    n: usize,
    /// FFT length: $n$ for `Periodic`, $2n$ for `Odd`/`Even`.
    n_fft: usize,
    /// Number of physical DOFs $N_\mathrm{dof}$.
    n_dof: usize,
    /// Polynomial order $r$.
    r: usize,
    /// Physical domain length $L$.
    l: f64,
    /// Left boundary position $x_l$.
    xl: f64,
    /// Extension type.
    extension: Extension,
    /// GLL nodes on $[0,1]$ (length $r+1$).
    gll_points: Vec<f64>,
    /// Barycentric weights for the $r+1$ GLL nodes (used in [`eval`](Self::eval)).
    bary_weights: Vec<f64>,
    /// Forward FFT engine of length $n_\mathrm{fft}$.
    fw_fft: Arc<dyn Fft<f64>>,
    /// Inverse FFT engine of length $n_\mathrm{fft}$.
    i_fft: Arc<dyn Fft<f64>>,
    /// Unpacking table: `unpack_table[j * n_fft + k] = (packed_idx, coeff)` such that
    /// `full_buf[j * n_fft + k] = coeff * compact_buf[packed_idx]`.
    /// Encodes the symmetry relations from Lemma 3.2 of \[1\].
    unpack_table: Vec<(usize, Complex<f64>)>,
    /// Canonicality table: `canonical_table[j * n_fft + k] = true` iff entry $(j, k)$
    /// is an independently stored coefficient (not derived by symmetry or forced zero).
    canonical_table: Vec<bool>,
    /// Packing table: `packing_table[i] = full_buffer_index` for compact index $i$.
    /// Used by `forward` to copy independent coefficients into the compact buffer.
    /// Empty for `Periodic` (trivial mapping).
    packing_table: Vec<usize>,
}

impl HofftPlanner {
    /// Create a 1D HoFFT planner for $n$ elements of order $r$ on $[x_l, x_l+L]$.
    ///
    /// Precomputes the forward/inverse FFT plans of length $n_\mathrm{fft}$
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
        let gll_points = CollocationBasis::new(vec![(r + 1, Gauss::Lobatto)]).points_1d(0).to_vec();
        let bary_weights: Vec<f64> = barycentric_weights(&gll_points);
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

    /// Extract the $r$ values for Fourier mode $k$ from the packed buffer `g` into `g_k`.
    ///
    /// `k` may be any mode in $\{0, \ldots, n_\mathrm{fft}-1\}$; modes in the
    /// symmetric half ($k > n$) are recovered via Lemma 3.2 of \[1\] and the
    /// cross-row symmetries of the packed representation.
    pub fn get_values(&self, k: usize, g: &[Complex<f64>], g_k: &mut [Complex<f64>]) {
        debug_assert_eq!(g.len(), self.n_dof);
        debug_assert_eq!(g_k.len(), self.r);
        for j in 0..self.r {
            let (i_g, coeff) = self.unpack_table[j * self.n_fft + k];
            g_k[j] = coeff * g[i_g];
        }
    }

    /// Write the $r$ values for Fourier mode $k$ from `g_k` into the packed buffer `g`.
    ///
    /// Only independently stored entries are written; the symmetric mirror modes
    /// ($k > n$) are derived and must not be set directly.
    /// `k` must be in $\{0, \ldots, n\}$.
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
    /// [`PoissonSolver1D::get_x`].
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
        let phi = lagrangian_interpolation(&self.gll_points, &self.bary_weights, xi);

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

    /// Forward HoFFT: extend, reorder, FFT, and compact in place.
    ///
    /// On input, `f` holds the physical DOFs in $x$-increasing order, length $N_\mathrm{dof}$.
    /// The function:
    /// 1. Builds the extended buffer of length $r \cdot n_\mathrm{fft}$ (odd or even
    ///    extension, or identity for Periodic).
    /// 2. Applies the forward FFT of length $n_\mathrm{fft}$ along each of the $r$ rows.
    /// 3. Compacts the result to the $N_\mathrm{dof}$ independent Fourier coefficients
    ///    using the packing table.
    ///
    /// On output, `f` contains the packed frequency-domain representation.
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
                    f[i] = buf[self.packing_table[i]];
                }
            }
        }
    }

    /// Inverse HoFFT: unpack, inverse FFT, reorder in place.
    ///
    /// On input, `g` holds the packed frequency-domain representation of length
    /// $N_\mathrm{dof}$ (as produced by [`forward`](Self::forward)).
    /// The function:
    /// 1. Expands to the full $r \cdot n_\mathrm{fft}$ buffer using the unpacking table
    ///    (applies symmetry coefficients from Lemma 3.2 of \[1\]).
    /// 2. Applies the inverse FFT of length $n_\mathrm{fft}$ and normalises by $1/n_\mathrm{fft}$.
    /// 3. Reorders to the $x$-increasing physical DOF layout.
    ///
    /// On output, `g` contains the physical DOFs in $x$-increasing order.
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
    planners: [HofftPlanner; N],
    /// Physical (= frequency) DOF counts per direction: `ndofs[d] = planners[d].n_dof`.
    ndofs: [usize; N],
    /// Total DOFs: `∏_d ndofs[d]`.
    total: usize,
    /// Column-major strides for the DOF tensor.
    strides: [usize; N],
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
        let full_sizes: [usize; N] = std::array::from_fn(|d| planners[d].r * planners[d].n_fft);
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
                    .par_iter()
                    .map(|&i| {
                        let phi: [Vec<f64>; N] = std::array::from_fn(|d| {
                            let y = match self.planners[d].extension {
                                Extension::Periodic => {
                                    (xs[i][d] - xls[d]).rem_euclid(ls[d]) + xls[d]
                                }
                                _ => xs[i][d],
                            };
                            let xi = (y - xls[d] - (ks[d] as f64) * hs[d]) / hs[d];
                            lagrangian_interpolation(
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

// ── Solver ──────────────────────────────────────────────────────────────────

/// 1D HoFFT screened Poisson solver.
///
/// Solves $(\alpha M + \beta K)\, u = f$ on $[x_l, x_l+L]$ using a GLL
/// spectral-element discretisation with $n$ elements of polynomial order $r$.
///
/// ## Algorithm
///
/// The inter-element coupling matrix has a block-circulant structure (periodic)
/// or a near-circulant structure (Dirichlet/Neumann via odd/even extension).
/// A single FFT of length $n_\mathrm{fft}$ decouples the $n_\mathrm{fft}$ Fourier
/// modes, reducing the global $nr \times nr$ system to $n_\mathrm{fft}/2 + 1$
/// independent **local symbols** $S(k)$ of size $r \times r$.
///
/// Each $S(k)$ is a Hermitian matrix assembled from the periodic element matrices
/// modulated by the inter-element phase $e^{-2\pi i k / n}$:
/// $$S(k)_{ij} = \alpha h M_{ij} + \frac{\beta}{h}\bigl(K_{ij} + K_{ri}\,e^{-2\pi ik/n}\bigr), \quad i,j = 1,\ldots,r-1,$$
/// with appropriate modifications for the $(0,0)$ entry that couples the two sides
/// of an element interface.
///
/// The generalised eigendecomposition $S(k) V_k = M V_k \Lambda_k$,
/// precomputed at build time, yields the per-mode solve
/// $$\hat{u}_k = V_k \Lambda_k^{-1} V_k^\dagger \hat{f}_k.$$
///
/// Physical DOF layout: $x$-increasing, flat index $k r + j$ for local node $j$
/// of element $k$ (with offset adjustments for Dirichlet; see [`HofftPlanner`]).
pub struct PoissonSolver1D {
    /// Polynomial order $r$ per element.
    r: usize,

    /// HoFFT planner: handles extension, packing, forward/inverse FFT, and interpolation.
    pub planner: HofftPlanner,

    /// Real eigenvalues of the mass-normalised symbol for each independent Fourier mode $k$:
    /// $\lambda_j(k)$ of $M^{-1/2} S(k) M^{-1/2}$, length $r$.
    eigenvalues: Vec<DVector<f64>>,
    /// $M$-orthonormal eigenvectors $V_k$ satisfying $S(k) V_k = M V_k \Lambda_k$;
    /// `eigenvectors[k]` is the $r \times r$ complex matrix.
    eigenvectors: Vec<DMatrix<Complex<f64>>>,
    /// Diagonal of the **periodic** local mass matrix (length $r$):
    /// $\tilde{M}_{jj} = h w_j$, with DOF 0 accumulating contributions from both sides
    /// of the element interface: $\tilde{M}_{00} = 2h w_0$.
    periodic_mass_matrix: DVector<f64>,
    /// Boundary condition type.
    bc: BoundaryCondition,
}

impl PoissonSolver1D {
    /// Construct the 1D HoFFT solver.
    ///
    /// Assembles the local symbol matrices $S(k)$ for all independent Fourier
    /// modes and precomputes their generalised eigendecompositions.
    ///
    /// # Arguments
    /// - `l`: domain length $L$.
    /// - `xl`: left boundary position $x_l$.
    /// - `n`: number of elements.
    /// - `r`: polynomial order ($r+1$ GLL nodes per element).
    /// - `alpha`, `beta`: operator coefficients.
    /// - `bc`: boundary condition type.
    pub fn new(
        l: f64,
        xl: f64,
        n: usize,
        r: usize,
        alpha: f64,
        beta: f64,
        bc: BoundaryCondition,
    ) -> Self {
        let lobatto_basis = CollocationBasis::new(vec![(r + 1, Gauss::Lobatto)]);

        let h = l / (n as f64);

        let planner = HofftPlanner::new(n, r, l, xl, bc.extension());

        let n_freq = match bc {
            BoundaryCondition::Periodic => n,
            BoundaryCondition::Dirichlet => n + 1,
            BoundaryCondition::Neumann => n + 1,
        };

        let coeff_phase = if bc == BoundaryCondition::Periodic {
            2.0
        } else {
            1.0
        };

        let mut symbol = vec![DMatrix::<Complex::<f64>>::zeros(r, r); n_freq];
        let mass_matrix = lobatto_basis.weights_matrix();
        let periodic_mass_matrix = DVector::from_fn(r, |i, _| {
            if i == 0 {
                h * 2.0 * mass_matrix[(0, 0)]
            } else {
                h * mass_matrix[(i, i)]
            }
        });
        let diff_matrix = lobatto_basis.diff_matrix(0);

        // Construct the stiffness matrix
        let stiffness_matrix: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> =
            diff_matrix.transpose() * mass_matrix.clone() * diff_matrix;

        // Compute the symbol of the operator for each frequency
        for k in 0..n_freq {
            let phase = -coeff_phase * PI * (k as f64) / (n as f64);
            let cos = phase.cos();
            let sin = phase.sin();
            let exp = cos + Complex::<f64>::I * sin;

            for i in 1..r {
                for j in 1..r {
                    symbol[k][(i, j)] = (alpha * h * mass_matrix[(i, j)]
                        + beta * stiffness_matrix[(i, j)] / h)
                        .into();
                }
            }

            symbol[k][(0, 0)] = (2.0 * alpha * h * mass_matrix[(0, 0)]
                + 2.0 * beta * stiffness_matrix[(0, 0)] / h
                + 2.0 * beta * stiffness_matrix[(0, r)] * cos / h)
                .into();

            for i in 1..r {
                symbol[k][(0, i)] =
                    beta * stiffness_matrix[(0, i)] / h + beta * stiffness_matrix[(r, i)] * exp / h;
                symbol[k][(i, 0)] = symbol[k][(0, i)].conj();
            }
        }

        let mass_sqrt_inv_c: DMatrix<Complex<f64>> =
            DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                Complex::new(1.0 / periodic_mass_matrix[i].sqrt(), 0.0)
            }));

        // Generalised eigendecomposition via mass normalisation:
        //   M^{-1/2} S(k) M^{-1/2} W = W Λ   ⟹   eigenvectors V = M^{-1/2} W
        let eigs: Vec<SymmetricEigen<Complex<f64>, Dyn>> = symbol
            .iter()
            .map(|s| SymmetricEigen::new(&mass_sqrt_inv_c * s * &mass_sqrt_inv_c))
            .collect();
        let eigenvalues: Vec<DVector<f64>> = eigs.iter().map(|e| e.eigenvalues.clone()).collect();

        // M-orthonormal eigenvectors of α I + β M^{-1} K
        let eigenvectors: Vec<DMatrix<Complex<f64>>> = eigs
            .iter()
            .map(|e| &mass_sqrt_inv_c * e.eigenvectors.clone())
            .collect();

        Self {
            r,
            planner,
            eigenvalues,
            eigenvectors,
            periodic_mass_matrix,
            bc,
        }
    }

    /// Return the GLL nodes in $x$-increasing order on $[x_l, x_l+L]$.
    ///
    /// - [`BoundaryCondition::Periodic`]:  $n r$ nodes — left endpoint included, right excluded.
    /// - [`BoundaryCondition::Dirichlet`]: $n r - 1$ nodes — both boundary endpoints excluded.
    /// - [`BoundaryCondition::Neumann`]:   $n r + 1$ nodes — both boundary endpoints included.
    pub fn get_x(&self) -> Vec<f64> {
        self.planner.get_x()
    }

    /// Solve $(\alpha M + \beta K)\, u = f$ in place.
    ///
    /// On input, `f` holds the strong-form right-hand side at the GLL nodes
    /// returned by [`get_x`](Self::get_x).
    /// On output, `f` contains the FEM solution $u$ at the same nodes.
    ///
    /// Executes the three-step HoFFT solve:
    /// 1. Mass scaling + forward FFT.
    /// 2. Per-mode symbol solve: $\hat{u}_k = V_k \Lambda_k^{-1} V_k^\dagger \hat{f}_k$.
    /// 3. Inverse FFT.
    pub fn solve_in_place(&self, f: &mut [Complex<f64>]) {
        // Step 1: mass-scale f in place, then forward FFT.
        // Local node index j determines the mass weight:
        //   Periodic / Neumann:  j = idx % r
        //   Dirichlet:           j = (idx + 1) % r  (elem 0 starts at j=1)
        let j_offset = if self.bc == BoundaryCondition::Dirichlet {
            1
        } else {
            0
        };
        for (idx, v) in f.iter_mut().enumerate() {
            *v *= self.periodic_mass_matrix[(idx + j_offset) % self.r];
        }
        self.planner.forward(f);

        // Step 2: symbol solve via eigendecomposition.
        // For each mode k: v = Q†·g_k, pointwise divide by λ, result = Q·v.
        let n_freq = self.eigenvalues.len();
        let mut col = DVector::<Complex<f64>>::zeros(self.r);
        const PINV_TOL: f64 = 1e-12;
        for k in 0..n_freq {
            self.planner.get_values(k, f, col.as_mut_slice());
            // Q† · col
            let mut v = self.eigenvectors[k].ad_mul(&col);
            // Pointwise pseudo-inverse
            for j in 0..self.r {
                let lam = self.eigenvalues[k][j];
                if lam.abs() > PINV_TOL {
                    v[j] /= lam;
                } else {
                    v[j] = Complex::new(0.0, 0.0);
                }
            }
            // Q · v
            col.gemv(
                Complex::new(1.0, 0.0),
                &self.eigenvectors[k],
                &v,
                Complex::new(0.0, 0.0),
            );
            self.planner.set_values(k, f, col.as_slice());
        }

        // Step 3: recover symmetric modes, inverse FFT.
        self.planner.inverse(f);
    }
}

/// $N$-dimensional HoFFT screened Poisson solver.
///
/// Extends [`PoissonSolver1D`] to $N$ spatial dimensions using **sum factorisation**:
/// the forward and inverse FFTs are applied direction by direction, and the
/// per-mode symbol solve is applied independently to each mode tuple
/// $(k_0, \ldots, k_{N-1})$ via a tensor-product eigendecomposition.
///
/// `N` is a compile-time constant (const generic); the compiler can unroll
/// the per-direction loops.
///
/// ## ND algorithm
///
/// 1. **Mass scaling**: $f_p \leftarrow \tilde{M}_p f_p$ for each DOF $p$,
///    where $\tilde{M}_p = \prod_d \tilde{M}^{(d)}_{j_d(p)}$.
/// 2. **Forward ND FFT**: apply [`HofftPlannerND::forward`] direction by direction.
/// 3. **ND symbol solve**: for each mode tuple $(k_0,\ldots,k_{N-1})$, apply the
///    tensor-product pseudo-inverse
///    $$\hat{u}_{k,j} \leftarrow \hat{f}_{k,j} \Big/ \Bigl(\alpha + \beta \sum_d \lambda^{(d)}_{k_d, j_d}\Bigr)$$
///    after projecting onto and back from the per-direction eigenbases $V_{k_d}^{(d)}$.
/// 4. **Inverse ND FFT**: apply [`HofftPlannerND::inverse`] direction by direction.
///
/// Step 3 is parallelised over independent mode tuples with Rayon.
pub struct PoissonSolverND<const N: usize> {
    /// One 1D solver per spatial direction (built with $\alpha=0$, $\beta=1$ to obtain
    /// the per-direction eigenvectors; actual $\alpha$, $\beta$ applied at the ND level).
    solvers: [PoissonSolver1D; N],
    /// ND HoFFT planner for the forward/inverse transforms, interpolation, and advection.
    pub fft: HofftPlannerND<N>,
    /// Diagonal of the Kronecker-product periodic mass matrix
    /// $\tilde{M}^{(0)} \otimes \cdots \otimes \tilde{M}^{(N-1)}$,
    /// length $\prod_d r_d$, direction 0 ($x$) varies fastest.
    periodic_mass_matrix: DVector<f64>,
    /// Coefficient $\alpha$ of the mass term.
    alpha: f64,
    /// Coefficient $\beta$ of the stiffness term.
    beta: f64,
    /// Number of independent Fourier modes per direction:
    /// $n_d$ for `Periodic`, $n_d + 1$ for `Odd`/`Even`.
    n_freqs: [usize; N],
    /// Total number of independent mode tuples: $\prod_d n_\mathrm{freqs}[d]$.
    n_freq_total: usize,
    /// Column-major strides for the mode-tuple tensor.
    n_freq_strides: [usize; N],
    /// Per-direction DOF index offset for mass scaling: 1 for Dirichlet (first DOF
    /// is local node $j=1$), 0 otherwise.
    j_offsets: [usize; N],
}

impl<const N: usize> PoissonSolverND<N> {
    /// Construct the ND HoFFT solver.
    ///
    /// # Arguments
    /// - `ls[d]`: domain length $L_d$ in direction $d$.
    /// - `xls[d]`: left boundary position $x_{l,d}$ in direction $d$.
    /// - `ns[d]`: number of elements in direction $d$.
    /// - `rs[d]`: polynomial order $r_d$ in direction $d$.
    /// - `alpha`, `beta`: operator coefficients.
    /// - `bc[d]`: boundary condition for direction $d$.
    pub fn new(
        ls: [f64; N],
        xls: [f64; N],
        ns: [usize; N],
        rs: [usize; N],
        alpha: f64,
        beta: f64,
        bc: [BoundaryCondition; N],
    ) -> Self {
        let solvers = std::array::from_fn(|d| {
            PoissonSolver1D::new(ls[d], xls[d], ns[d], rs[d], 0.0, 1.0, bc[d])
        });

        // Build the diagonal of M^{(0)} ⊗ … ⊗ M^{(N-1)} iteratively.
        // Each 1D mass matrix is diagonal (GLL quadrature), so the Kronecker product
        // is also diagonal: entry (i_0,…,i_{N-1}) = ∏_d M^{(d)}[i_d, i_d].
        let periodic_mass_matrix =
            (1..N).fold(solvers[0].periodic_mass_matrix.clone(), |acc, d| {
                let m_d = &solvers[d].periodic_mass_matrix;
                DVector::from_iterator(
                    acc.len() * m_d.len(),
                    m_d.iter().flat_map(|&w| acc.iter().map(move |&v| v * w)),
                )
            });

        let extension: [Extension; N] = std::array::from_fn(|d| bc[d].extension());

        let fft = HofftPlannerND::new(ns, rs, ls, xls, extension);

        // Number of independent Fourier modes per direction.
        let n_freqs: [usize; N] = std::array::from_fn(|d| solvers[d].eigenvalues.len());
        let n_freq_total = n_freqs.iter().product();
        let n_freq_strides = nd_col_major_strides(&n_freqs);

        // DOF index offset for mass scaling: Dirichlet DOFs start at j=1.
        let j_offsets: [usize; N] = std::array::from_fn(|d| {
            if bc[d] == BoundaryCondition::Dirichlet {
                1
            } else {
                0
            }
        });

        Self {
            solvers,
            fft,
            periodic_mass_matrix,
            alpha,
            beta,
            n_freqs,
            n_freq_total,
            n_freq_strides,
            j_offsets,
        }
    }

    /// Return the tensor-product GLL grid as a `Vec<[f64; N]>` of length
    /// $N_\mathrm{dof} = \prod_d N_\mathrm{dof}^{(d)}$.
    ///
    /// Entry `p` holds coordinates $(x_0, \ldots, x_{N-1})$ for DOF $p$.
    /// Direction 0 ($x$) varies fastest.
    pub fn get_x(&self) -> Vec<[f64; N]> {
        self.fft.get_x()
    }

    /// Solve $(\alpha M + \beta K)\, u = f$ in place using the ND HoFFT algorithm.
    ///
    /// On input, `f` holds the strong-form right-hand side at the tensor-product
    /// GLL grid returned by [`get_x`](Self::get_x).
    /// On output, `f` contains the FEM solution $u$ at the same grid.
    ///
    /// See the struct-level documentation for the four-step ND algorithm.
    /// Step 3 (symbol solve) is parallelised over independent mode tuples with Rayon.
    pub fn solve_in_place(&self, f: &mut [Complex<f64>]) {
        // Step 1: mass-scale in place, then forward FFT.
        // Mass weight for DOF p in direction d: j_d = (dof_d + j_offset_d) % r_d.
        let rs = &self.fft.rs;
        let r_strides = &self.fft.r_strides;
        let r_total = self.fft.r_total;

        f.par_iter_mut().enumerate().for_each(|(p, fp)| {
            let lj = (0..N).fold(0, |acc, d| {
                let dof_d = (p / self.fft.strides[d]) % self.fft.ndofs[d];
                acc + ((dof_d + self.j_offsets[d]) % rs[d]) * r_strides[d]
            });
            *fp *= self.periodic_mass_matrix[lj];
        });

        self.fft.forward(f);

        // Step 2: symbol solve — independent mode tuples in parallel.
        // SAFETY: distinct k_flat values write to disjoint positions in f.
        let ptr = f.as_mut_ptr() as usize;
        (0..self.n_freq_total).into_par_iter().for_each(|k_flat| {
            let ks: [usize; N] =
                std::array::from_fn(|d| (k_flat / self.n_freq_strides[d]) % self.n_freqs[d]);

            let mut v = DVector::<Complex<f64>>::zeros(r_total);

            {
                let f_ro = unsafe {
                    std::slice::from_raw_parts(ptr as *const Complex<f64>, self.fft.total)
                };
                self.fft.get_values(&ks, f_ro, v.as_mut_slice());
            }

            // Apply Q†_d for each direction (sum factorisation).
            for d in 0..N {
                let r_d = rs[d];
                let step = r_strides[d] - 1;
                for s in 0..r_total / r_d {
                    let bj = nd_fiber_base(s, d, rs, r_strides);
                    let x = self.solvers[d].eigenvectors[ks[d]]
                        .ad_mul(&v.rows_with_step(bj, r_d, step));
                    v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                }
            }

            // Pointwise divide by ND eigenvalue λ = α + Σ_d β λ^{(d)}_{k_d, j_d}.
            const PINV_TOL: f64 = 1e-12;
            for j_flat in 0..r_total {
                let lambda = self.alpha
                    + (0..N)
                        .map(|d| {
                            let j_d = (j_flat / r_strides[d]) % rs[d];
                            self.beta * self.solvers[d].eigenvalues[ks[d]][j_d]
                        })
                        .sum::<f64>();
                if lambda.abs() > PINV_TOL {
                    v[j_flat] /= lambda;
                } else {
                    v[j_flat] = Complex::new(0.0, 0.0);
                }
            }

            // Apply Q_d for each direction.
            for d in 0..N {
                let r_d = rs[d];
                let step = r_strides[d] - 1;
                for s in 0..r_total / r_d {
                    let bj = nd_fiber_base(s, d, rs, r_strides);
                    let x = &self.solvers[d].eigenvectors[ks[d]] * v.rows_with_step(bj, r_d, step);
                    v.rows_with_step_mut(bj, r_d, step).set_column(0, &x);
                }
            }

            {
                let f_rw = unsafe {
                    std::slice::from_raw_parts_mut(ptr as *mut Complex<f64>, self.fft.total)
                };
                self.fft.set_values(&ks, f_rw, v.as_slice());
            }
        });

        // Step 3: inverse FFT.
        self.fft.inverse(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Complex;

    #[test]
    fn test_symbol_eigendecomposition() {
        const ORTHO_TOL: f64 = 1e-10;
        const RECON_TOL: f64 = 1e-8;

        for &r in &[1, 2, 3, 4, 5, 6] {
            for &n in &[4, 16, 32, 64, 128] {
                let solver =
                    PoissonSolver1D::new(1.0, 0.0, n, r, 1.0, 1.0, BoundaryCondition::Periodic);
                let id = DMatrix::<Complex<f64>>::identity(r, r);

                let m_c = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                    Complex::new(solver.periodic_mass_matrix[i], 0.0)
                }));
                let m_inv_c = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                    Complex::new(1.0 / solver.periodic_mass_matrix[i], 0.0)
                }));

                for k in 0..n {
                    let v = &solver.eigenvectors[k];

                    // 1. V† M V = I
                    let vtmv = v.adjoint() * &m_c * v;
                    let ortho_err = (&vtmv - &id).norm();
                    assert!(
                        ortho_err < ORTHO_TOL,
                        "r={r} n={n} k={k}: eigenvectors not M-orthonormal (err={ortho_err:.2e})"
                    );

                    // 2. V V† = M^{-1}
                    let vvt = v * v.adjoint();
                    let vvt_err = (&vvt - &m_inv_c).norm();
                    assert!(
                        vvt_err < ORTHO_TOL,
                        "r={r} n={n} k={k}: V V† ≠ M⁻¹ (err={vvt_err:.2e})"
                    );

                    // 3. Check Q† M Q = I (already covered above) and eigenvalues are real.
                    //    Also verify round-trip: Q diag(1/λ) Q† M V Λ = V
                    let lambda_diag = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                        Complex::new(solver.eigenvalues[k][i], 0.0)
                    }));
                    let lambda_inv = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                        let lam = solver.eigenvalues[k][i];
                        Complex::new(if lam.abs() > 1e-12 { 1.0 / lam } else { 0.0 }, 0.0)
                    }));
                    let rhs = v * &lambda_inv * v.adjoint() * &m_c * v * &lambda_diag;
                    let recon_err = (v - &rhs).norm() / v.norm().max(1.0);
                    assert!(
                        recon_err < RECON_TOL,
                        "r={r} n={n} k={k}: eigendecomposition round-trip failed (err={recon_err:.2e})"
                    );
                }
            }
        }
    }

    /// Check that with beta=0 the tensorial structure recovers the inverse of the local mass matrix.
    ///
    /// V_ND = V_0^{(0)} ⊗ ... ⊗ V_{N-1}^{(0)} (eigenvectors at k=0 for each direction)
    /// satisfies V_ND * V_ND† = tilde_M_ND^{-1} = diag(1 / periodic_mass_matrix).
    #[test]
    fn test_nd_mass_matrix_inverse_tensorial() {
        const TOL: f64 = 1e-12;

        let kron = |a: &DMatrix<Complex<f64>>, b: &DMatrix<Complex<f64>>| {
            let (ra, ca) = (a.nrows(), a.ncols());
            let (rb, cb) = (b.nrows(), b.ncols());
            DMatrix::from_fn(ra * rb, ca * cb, |i, j| {
                a[(i / rb, j / cb)] * b[(i % rb, j % cb)]
            })
        };

        // 1D
        {
            let solver = PoissonSolverND::<1>::new(
                [1.0],
                [0.0],
                [4],
                [3],
                1.0,
                0.0,
                [BoundaryCondition::Periodic; 1],
            );
            let v = &solver.solvers[0].eigenvectors[0];
            let vvt = v * v.adjoint();
            let r = v.nrows();
            for i in 0..r {
                for j in 0..r {
                    let expected = if i == j {
                        Complex::new(1.0 / solver.solvers[0].periodic_mass_matrix[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                    assert!((vvt[(i, j)] - expected).norm() < TOL, "1D [{i},{j}]");
                }
            }
        }

        // 2D: V_ND = V_0 ⊗ V_1
        {
            let solver = PoissonSolverND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [4; 2],
                [2; 2],
                1.0,
                0.0,
                [BoundaryCondition::Periodic; 2],
            );
            let v_nd = kron(
                &solver.solvers[0].eigenvectors[0],
                &solver.solvers[1].eigenvectors[0],
            );
            let vvt = &v_nd * v_nd.adjoint();
            let r_total = v_nd.nrows();
            for i in 0..r_total {
                for j in 0..r_total {
                    let expected = if i == j {
                        Complex::new(1.0 / solver.periodic_mass_matrix[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                    assert!((vvt[(i, j)] - expected).norm() < TOL, "2D [{i},{j}]");
                }
            }
        }

        // 3D: V_ND = V_0 ⊗ V_1 ⊗ V_2
        {
            let solver = PoissonSolverND::<3>::new(
                [1.0; 3],
                [0.0; 3],
                [4; 3],
                [2; 3],
                1.0,
                0.0,
                [BoundaryCondition::Periodic; 3],
            );
            let v_nd = kron(
                &kron(
                    &solver.solvers[0].eigenvectors[0],
                    &solver.solvers[1].eigenvectors[0],
                ),
                &solver.solvers[2].eigenvectors[0],
            );
            let vvt = &v_nd * v_nd.adjoint();
            let r_total = v_nd.nrows();
            for i in 0..r_total {
                for j in 0..r_total {
                    let expected = if i == j {
                        Complex::new(1.0 / solver.periodic_mass_matrix[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                    let scale = expected.norm().max(1.0_f64);
                    assert!(
                        (vvt[(i, j)] - expected).norm() / scale < TOL,
                        "3D [{i},{j}]"
                    );
                }
            }
        }
    }
}
