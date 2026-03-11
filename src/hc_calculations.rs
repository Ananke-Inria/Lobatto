//! Hard-coded spectral differentiation operators for Q3 and Q4 Gauss-Lobatto elements.
//!
//! These functions implement the action of the 1D spectral differentiation
//! matrix **D** on a vector of nodal values, exploiting the structure of the
//! Gauss-Lobatto nodes to reduce the flop count compared to a general
//! matrix–vector product.  Multi-dimensional operators (Q3×Q3, Q3×Q3×Q3,
//! Q4×Q4, Q4×Q4×Q4) are built by applying the 1D kernel direction by direction
//! (sum-factorisation).
//!
//! ## Two families
//!
//! - **Scalar** — operate on a single element at a time (Array-of-Structures layout).
//! - **Vectorized** — operate on a batch of `N` elements simultaneously,
//!   using a Structure-of-Arrays layout that exposes SIMD parallelism to the
//!   compiler.
//!
//! ## Data layouts
//!
//! **Scalar / AoS** — flat array indexed as node `(i, j, k)` →
//! `i + (r+1)*j + (r+1)²*k` (x fastest, z slowest), where `r+1` is the number
//! of nodes per direction (`r+1 = 4` for Q3, `r+1 = 5` for Q4).
//!
//! **Vectorized / SoA** — for a batch of `N` elements the position of element
//! `e` at node `(i, j, k)` is `N*(i + (r+1)*j + (r+1)²*k) + e`.
//! Equivalently, the batch is stored as `(r+1)³` contiguous blocks of `N`
//! values, one block per node index.

// ----------------------------------------------------------------------------
// Differentiation coefficients for Q3 (4-node Gauss-Lobatto, degree 3)

const DX3: [f64; 5] = [
    1.6180339887498944584,
    0.61803398874989456946,
    8.0901699437494745126,
    3.0901699437494727363,
    2.2360679774997902491,
];

// ============================================================================
// Kernel primitives
// ============================================================================

/// Apply the 1D Q3 differentiation operator to 4 nodes at stride `M`.
///
/// Reads nodes `a = in[0]`, `b = in[M]`, `c = in[2M]`, `d = in[3M]` and
/// writes the derivative values back to the same positions in `out`.
/// The stride `M` controls which direction is differentiated:
///   - `M = 1`  → x-direction (consecutive nodes)
///   - `M = 4`  → y-direction (stride 4 in a 4×4 face)
///   - `M = 16` → z-direction (stride 16 in a 4×4×4 cube)
#[inline(always)]
pub fn _d_q3_q3<const M: usize>(in_q3: &[f64], out_q3: &mut [f64]) {
    _d_q3_q3_vec::<1, M>(in_q3, out_q3);
}

/// Vectorized 1D Q3 differentiation kernel for a batch of `N` elements.
///
/// Input and output use the SoA layout: node group `g` occupies the slice
/// `[g*M*N .. g*M*N + N]` (four groups: g = 0, 1, 2, 3).  The stride `M`
/// selects the differentiation direction, exactly as in `_d_q3_q3`.
///
/// The four output loops are kept separate so the compiler can auto-vectorize
/// each one independently (SIMD over the N-element batch dimension).
pub fn _d_q3_q3_vec<const N: usize, const M: usize>(in_q3: &[f64], out_q3: &mut [f64]) {
    assert!(in_q3.len() >= N * (3 * M + 1));
    assert!(out_q3.len() >= N * (3 * M + 1));

    let a = &in_q3[0..N];
    let b = &in_q3[M * N..M * N + N];
    let c = &in_q3[2 * M * N..2 * M * N + N];
    let d = &in_q3[3 * M * N..3 * M * N + N];

    for i in 0..N {
        out_q3[i] = DX3[2] * b[i] - DX3[3] * c[i] + d[i] - 6.0 * a[i];
    }
    for i in 0..N {
        out_q3[M * N + i] = DX3[4] * c[i] - DX3[0] * a[i] - DX3[1] * d[i];
    }
    for i in 0..N {
        out_q3[2 * M * N + i] = DX3[1] * a[i] - DX3[4] * b[i] + DX3[0] * d[i];
    }
    for i in 0..N {
        out_q3[3 * M * N + i] = DX3[3] * b[i] - DX3[2] * c[i] - a[i] + 6.0 * d[i];
    }
}

// ============================================================================
// Scalar differentiation operators  (single element, AoS layout)
// ============================================================================

// ----------------------------------------------------------------------------
// 1D  —  Q3  (4 dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a single Q3 element (4 dofs, flat layout).
pub fn dx_q3_q3(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 4 && output.len() >= 4);
    _d_q3_q3::<1>(&input[0..4], &mut output[0..4]);
}

// ----------------------------------------------------------------------------
// 2D  —  Q3×Q3  (16 dofs, layout: i + 4*j,  x fastest)
// ----------------------------------------------------------------------------

/// x-derivative on a single Q3×Q3 element (16 dofs).
/// Applies the 1D kernel to each of the 4 x-lines independently.
pub fn dx_q33_q33(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 16 && output.len() >= 16);
    _d_q3_q3::<1>(&input[0..4], &mut output[0..4]);
    _d_q3_q3::<1>(&input[4..8], &mut output[4..8]);
    _d_q3_q3::<1>(&input[8..12], &mut output[8..12]);
    _d_q3_q3::<1>(&input[12..16], &mut output[12..16]);
}

/// y-derivative on a single Q3×Q3 element (16 dofs).
/// The AoS layout (x-fastest) groups all y-node values for each x-position into
/// consecutive 4-element blocks, matching exactly the SoA layout for `_d_q3_q3_vec::<4, 1>`.
pub fn dy_q33_q33(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 16 && output.len() >= 16);
    _d_q3_q3_vec::<4, 1>(input, output);
}

// ----------------------------------------------------------------------------
// 3D  —  Q3×Q3×Q3  (64 dofs, layout: i + 4*j + 16*k,  x fastest)
// ----------------------------------------------------------------------------

/// x-derivative on a single Q3×Q3×Q3 element (64 dofs).
/// Applies the 1D kernel to each of the 16 x-lines (stride M=1).
pub fn dx_q333_q333(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 64 && output.len() >= 64);
    for i in 0..16 {
        let start = 4 * i;
        _d_q3_q3::<1>(&input[start..start + 4], &mut output[start..start + 4]);
    }
}

/// y-derivative on a single Q3×Q3×Q3 element (64 dofs).
/// Applies the 1D kernel to each of the 16 y-columns (stride M=4).
/// A y-column at position (i, k) starts at index `i + 16*k`.
pub fn dy_q333_q333(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 64 && output.len() >= 64);
    for k in 0..4 {
        for i in 0..4 {
            let start = i + 16 * k;
            _d_q3_q3::<4>(&input[start..start + 13], &mut output[start..start + 13]);
        }
    }
}

/// z-derivative on a single Q3×Q3×Q3 element (64 dofs).
/// The AoS layout (z-slowest) stores all 16 (x, y) values for each z-level g in
/// a contiguous block at offset 16*g — exactly the SoA layout for `_d_q3_q3_vec::<16, 1>`.
pub fn dz_q333_q333(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 64 && output.len() >= 64);
    _d_q3_q3_vec::<16, 1>(input, output);
}

// ============================================================================
// Vectorized differentiation operators  (batch of N elements, dx-SoA layout)
//
// dx-SoA layout for a batch of N Q3×Q3×Q3 elements:
//   position of element e at node (i, j, k)  =  N*(i + 4*j + 16*k) + e
//
// The same layout is used for all three directions; only the stride M passed
// to `_d_q3_q3_vec` changes (M=1 for x, M=4 for y, M=16 for z).
// ============================================================================

// ----------------------------------------------------------------------------
// 1D  —  Q3  (4*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q3 elements (dx-SoA layout, 4*N dofs).
pub fn dx_q3_q3_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    _d_q3_q3_vec::<N, 1>(&input[0..N * 4], &mut output[0..N * 4]);
}

// ----------------------------------------------------------------------------
// 2D  —  Q3×Q3  (16*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q3×Q3 elements (dx-SoA layout, 16*N dofs).
/// Processes 4 independent x-line groups of N×4 values.
pub fn dx_q33_q33_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..4 {
        let start = N * 4 * i;
        let end = start + N * 4;
        _d_q3_q3_vec::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

// ----------------------------------------------------------------------------
// 3D  —  Q3×Q3×Q3  (64*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q3×Q3×Q3 elements (dx-SoA layout, 64*N dofs).
/// Processes 16 independent x-line groups of N×4 values (stride M=1).
pub fn dx_q333_q333_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..16 {
        let start = N * 4 * i;
        let end = start + N * 4;
        _d_q3_q3_vec::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

/// y-derivative on a batch of N Q3×Q3×Q3 elements (dx-SoA layout, 64*N dofs).
/// Processes 16 y-columns; column (i, k) starts at `N*(i + 16*k)`,
/// slice length `13*N` (stride M=4 → 3*4+1 = 13 nodes).
pub fn dy_q333_q333_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 64 * N && output.len() >= 64 * N);
    for k in 0..4 {
        for i in 0..4 {
            let start = N * (i + 16 * k);
            _d_q3_q3_vec::<N, 4>(
                &input[start..start + 13 * N],
                &mut output[start..start + 13 * N],
            );
        }
    }
}

/// z-derivative on a batch of N Q3×Q3×Q3 elements (dx-SoA layout, 64*N dofs).
/// Processes 16 z-columns; column (i, j) starts at `N*(i + 4*j)`,
/// slice length `49*N` (stride M=16 → 3*16+1 = 49 nodes).
pub fn dz_q333_q333_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 64 * N && output.len() >= 64 * N);
    for j in 0..4 {
        for i in 0..4 {
            let start = N * (i + 4 * j);
            _d_q3_q3_vec::<N, 16>(
                &input[start..start + 49 * N],
                &mut output[start..start + 49 * N],
            );
        }
    }
}

// ============================================================================
// Q4 operators  (5-node Gauss-Lobatto, degree 4)
// ============================================================================

// ----------------------------------------------------------------------------
// Differentiation coefficients for Q4 (5-node Gauss-Lobatto, degree 4)
// ----------------------------------------------------------------------------
const DX4: [f64; 7] = [
    2.4819805060619613002,
    0.51801949393803370381,
    13.513004977448479949,
    2.673169155390906937,
    1.5275252316519465356,
    2.8203283558848517565,
    3.4914862437758804603 / (16.0 / 3.0), // ≈ 0.6547
];

// ============================================================================
// Kernel primitives
// ============================================================================

/// Apply the 1D Q4 differentiation operator to 5 nodes at stride `M`.
///
/// Reads nodes `a..e` at positions `0, M, 2M, 3M, 4M` and writes the
/// derivative values back to the same positions in `out`.
/// The middle node is internally scaled by `16/3` before use.
/// The stride `M` controls which direction is differentiated:
///   - `M = 1`  → x-direction
///   - `M = 5`  → y-direction (stride 5 in a 5×5 face)
///   - `M = 25` → z-direction (stride 25 in a 5×5×5 cube)
#[inline(always)]
pub fn _d_q4_q4<const M: usize>(in_q4: &[f64], out_q4: &mut [f64]) {
    _d_q4_q4_vec::<1, M>(in_q4, out_q4);
}

/// Vectorized 1D Q4 differentiation kernel for a batch of `N` elements.
///
/// Input and output use the SoA layout: node group `g` occupies the slice
/// `[g*M*N .. g*M*N + N]` (five groups: g = 0..4).  The stride `M` selects
/// the differentiation direction, exactly as in `_d_q4_q4`.
///
/// Loops are kept separate for independent SIMD vectorization.
pub fn _d_q4_q4_vec<const N: usize, const M: usize>(in_q4: &[f64], out_q4: &mut [f64]) {
    assert!(in_q4.len() >= N * (4 * M + 1));
    assert!(out_q4.len() >= N * (4 * M + 1));

    let a = &in_q4[0..N];
    let b = &in_q4[M * N..M * N + N];
    let c = &in_q4[2 * M * N..2 * M * N + N];
    let d = &in_q4[3 * M * N..3 * M * N + N];
    let e = &in_q4[4 * M * N..4 * M * N + N];

    for i in 0..N {
        out_q4[i] = DX4[2] * b[i] - 10.0 * a[i] - (16.0 / 3.0) * c[i] + DX4[5] * d[i] - e[i];
    }
    for i in 0..N {
        out_q4[M * N + i] =
            DX4[1] * e[i] - DX4[0] * a[i] + (16.0 / 3.0) * DX4[6] * c[i] - DX4[4] * d[i];
    }
    for i in 0..N {
        out_q4[2 * M * N + i] = 0.75 * (a[i] - e[i]) + DX4[3] * (d[i] - b[i]);
    }
    for i in 0..N {
        out_q4[3 * M * N + i] =
            DX4[4] * b[i] - DX4[1] * a[i] - (16.0 / 3.0) * DX4[6] * c[i] + DX4[0] * e[i];
    }
    for i in 0..N {
        out_q4[4 * M * N + i] =
            a[i] - DX4[5] * b[i] + (16.0 / 3.0) * c[i] - DX4[2] * d[i] + 10.0 * e[i];
    }
}

// ============================================================================
// Scalar differentiation operators  (single element, AoS layout)
// ============================================================================

// ----------------------------------------------------------------------------
// 1D  —  Q4  (5 dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a single Q4 element (5 dofs, flat layout).
pub fn dx_q4_q4(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 5 && output.len() >= 5);
    _d_q4_q4::<1>(&input[0..5], &mut output[0..5]);
}

// ----------------------------------------------------------------------------
// 2D  —  Q4×Q4  (25 dofs, layout: i + 5*j,  x fastest)
// ----------------------------------------------------------------------------

/// x-derivative on a single Q4×Q4 element (25 dofs).
/// Applies the 1D kernel to each of the 5 x-lines (stride M=1).
pub fn dx_q44_q44(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 25 && output.len() >= 25);
    for i in 0..5 {
        let start = 5 * i;
        _d_q4_q4::<1>(&input[start..start + 5], &mut output[start..start + 5]);
    }
}

/// y-derivative on a single Q4×Q4 element (25 dofs).
/// The AoS layout (x-fastest) already stores all y-node values for a given
/// x-position in consecutive blocks of 5, which matches exactly the SoA
/// layout expected by `_d_q4_q4_vec_bis::<5, 1>`.  We exploit this to
/// process all 5 x-columns simultaneously with SIMD.
pub fn dy_q44_q44(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 25 && output.len() >= 25);
    _d_q4_q4_vec::<5, 1>(input, output);
}

// ----------------------------------------------------------------------------
// 3D  —  Q4×Q4×Q4  (125 dofs, layout: i + 5*j + 25*k,  x fastest)
// ----------------------------------------------------------------------------

/// x-derivative on a single Q4×Q4×Q4 element (125 dofs) — original scalar implementation.
/// Applies the 1D kernel to each of the 25 x-lines (stride M=1).
pub fn dx_q444_q444(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 125 && output.len() >= 125);
    for i in 0..25 {
        let start = 5 * i;
        _d_q4_q4::<1>(&input[start..start + 5], &mut output[start..start + 5]);
    }
}

/// y-derivative on a single Q4×Q4×Q4 element (125 dofs) — original scalar implementation.
/// Applies the 1D kernel to each of the 25 y-columns (stride M=5).
pub fn dy_q444_q444(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 125 && output.len() >= 125);
    for k in 0..5 {
        for i in 0..5 {
            let start = i + 25 * k;
            _d_q4_q4::<5>(&input[start..start + 21], &mut output[start..start + 21]);
        }
    }
}

/// z-derivative on a single Q4×Q4×Q4 element (125 dofs).
/// The AoS layout (z-slowest) stores all 25 (x, y) values for each z-level g in
/// a contiguous block at offset 25*g — this is exactly the SoA layout expected by
/// `_d_q4_q4_vec::<25, 1>`.  No gather/scatter needed, same principle as `dy_q44_q44`.
pub fn dz_q444_q444(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 125 && output.len() >= 125);
    _d_q4_q4_vec::<25, 1>(input, output);
}

// ============================================================================
// Vectorized differentiation operators  (batch of N elements, SoA layout)
//
// SoA layout for a batch of N Q4×Q4×Q4 elements:
//   position of element e at node (i, j, k)  =  N*(i + 5*j + 25*k) + e
//
// The same layout is used for all three directions; only the stride M passed
// to `_d_q4_q4_vec` changes (M=1 for x, M=5 for y, M=25 for z).
// ============================================================================

// ----------------------------------------------------------------------------
// 1D  —  Q4  (5*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q4 elements (SoA layout, 5*N dofs).
pub fn dx_q4_q4_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    _d_q4_q4_vec::<N, 1>(&input[0..N * 5], &mut output[0..N * 5]);
}

// ----------------------------------------------------------------------------
// 2D  —  Q4×Q4  (25*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q4×Q4 elements (SoA layout, 25*N dofs).
/// Processes 5 independent x-line groups of N×5 values.
pub fn dx_q44_q44_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..5 {
        let start = N * 5 * i;
        let end = start + N * 5;
        _d_q4_q4_vec::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

// ----------------------------------------------------------------------------
// 3D  —  Q4×Q4×Q4  (125*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q4×Q4×Q4 elements (SoA layout, 125*N dofs).
/// Processes 25 independent x-line groups of N×5 values (stride M=1).
pub fn dx_q444_q444_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..25 {
        let start = N * 5 * i;
        let end = start + N * 5;
        _d_q4_q4_vec::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

/// y-derivative on a batch of N Q4×Q4×Q4 elements (SoA layout, 125*N dofs).
/// Processes 25 y-columns; column (i, k) starts at `N*(i + 25*k)`,
/// slice length `21*N` (stride M=5 → 4*5+1 = 21 nodes).
pub fn dy_q444_q444_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 125 * N && output.len() >= 125 * N);
    for k in 0..5 {
        for i in 0..5 {
            let start = N * (i + 25 * k);
            _d_q4_q4_vec::<N, 5>(
                &input[start..start + 21 * N],
                &mut output[start..start + 21 * N],
            );
        }
    }
}

/// z-derivative on a batch of N Q4×Q4×Q4 elements (SoA layout, 125*N dofs).
/// Processes 25 z-columns; column (i, j) starts at `N*(i + 5*j)`,
/// slice length `101*N` (stride M=25 → 4*25+1 = 101 nodes).
pub fn dz_q444_q444_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 125 * N && output.len() >= 125 * N);
    for j in 0..5 {
        for i in 0..5 {
            let start = N * (i + 5 * j);
            _d_q4_q4_vec::<N, 25>(
                &input[start..start + 101 * N],
                &mut output[start..start + 101 * N],
            );
        }
    }
}
