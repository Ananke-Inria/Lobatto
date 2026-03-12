//! Hard-coded spectral differentiation operators for Q3 and Q4 Gauss-Lobatto elements.
//!
//! These functions implement the action of the 1D spectral differentiation
//! matrix **D** on a vector of nodal values, exploiting the structure of the
//! Gauss-Lobatto nodes to reduce the flop count compared to a general
//! matrix–vector product.  Multi-dimensional operators (Q3×Q3, Q3×Q3×Q3,
//! Q4×Q4, Q4×Q4×Q4) are built by applying the 1D kernel direction by direction
//! (sum-factorisation).
//!
//! ## Data layout
//!
//! For a batch of `N` elements the position of element `e` at node `(i, j, k)`
//! is `N*(i + (r+1)*j + (r+1)²*k) + e`, where `r+1` is the number of nodes
//! per direction (`r+1 = 4` for Q3, `r+1 = 5` for Q4).
//! Equivalently, the batch is stored as `(r+1)³` contiguous blocks of `N`
//! values, one block per node index.  For `N = 1` this reduces to the standard
//! AoS (scalar) layout.

// ----------------------------------------------------------------------------
// Differentiation coefficients for Q3 (4-node Gauss-Lobatto, degree 3)

const DX3: [f64; 5] = [
    8.09016994374947451263, // |D[0,1]|
    3.09016994374947362445, // |D[0,2]|
    1.61803398874989490253, // |D[1,0]|
    2.23606797749978936096, // |D[1,2]|
    0.61803398874989479150, // |D[1,3]|
];

// ============================================================================
// Kernel primitives
// ============================================================================

/// Vectorized 1D Q3 differentiation kernel for a batch of `N` elements.
///
/// Input and output use the SoA layout: node group `g` occupies the slice
/// `[g*M*N .. g*M*N + N]` (four groups: g = 0, 1, 2, 3).  The stride `M`
/// selects the differentiation direction:
///   - `M = 1`  → x-direction (consecutive nodes)
///   - `M = 4`  → y-direction (stride 4 in a 4×4 face)
///   - `M = 16` → z-direction (stride 16 in a 4×4×4 cube)
///
/// The four output loops are kept separate so the compiler can auto-vectorize
/// each one independently (SIMD over the N-element batch dimension).
pub fn _d_q3_q3<const N: usize, const M: usize>(in_q3: &[f64], out_q3: &mut [f64]) {
    assert!(in_q3.len() >= N * (3 * M + 1));
    assert!(out_q3.len() >= N * (3 * M + 1));

    let a = &in_q3[0..N];
    let b = &in_q3[M * N..M * N + N];
    let c = &in_q3[2 * M * N..2 * M * N + N];
    let d = &in_q3[3 * M * N..3 * M * N + N];

    for i in 0..N {
        out_q3[i] = -6.0 * a[i] + DX3[0] * b[i] - DX3[1] * c[i] + 1.0 * d[i];
    }
    for i in 0..N {
        out_q3[1 * M * N + i] = -DX3[2] * a[i] + DX3[3] * c[i] - DX3[4] * d[i];
    }
    for i in 0..N {
        out_q3[2 * M * N + i] = DX3[4] * a[i] - DX3[3] * b[i] + DX3[2] * d[i];
    }
    for i in 0..N {
        out_q3[3 * M * N + i] = -1.0 * a[i] + DX3[1] * b[i] - DX3[0] * c[i] + 6.0 * d[i];
    }
}

// ============================================================================
// Differentiation operators  (batch of N elements, SoA layout)
//
// SoA layout for a batch of N Q3×Q3×Q3 elements:
//   position of element e at node (i, j, k)  =  N*(i + 4*j + 16*k) + e
//
// The same layout is used for all three directions; only the stride M passed
// to `_d_q3_q3` changes (M=1 for x, M=4 for y, M=16 for z).
// ============================================================================

// ----------------------------------------------------------------------------
// 1D  —  Q3  (4*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q3 elements (SoA layout, 4*N dofs).
pub fn dx_q3_q3<const N: usize>(input: &[f64], output: &mut [f64]) {
    _d_q3_q3::<N, 1>(&input[0..N * 4], &mut output[0..N * 4]);
}

// ----------------------------------------------------------------------------
// 2D  —  Q3×Q3  (16*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q3×Q3 elements (SoA layout, 16*N dofs).
/// Processes 4 independent x-line groups of N×4 values.
pub fn dx_q33_q33<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..4 {
        let start = N * 4 * i;
        let end = start + N * 4;
        _d_q3_q3::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

// ----------------------------------------------------------------------------
// 3D  —  Q3×Q3×Q3  (64*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q3×Q3×Q3 elements (SoA layout, 64*N dofs).
/// Processes 16 independent x-line groups of N×4 values (stride M=1).
pub fn dx_q333_q333<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..16 {
        let start = N * 4 * i;
        let end = start + N * 4;
        _d_q3_q3::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

/// y-derivative on a batch of N Q3×Q3×Q3 elements (SoA layout, 64*N dofs).
/// Processes 16 y-columns; column (i, k) starts at `N*(i + 16*k)`,
/// slice length `13*N` (stride M=4 → 3*4+1 = 13 nodes).
pub fn dy_q333_q333<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 64 * N && output.len() >= 64 * N);
    for k in 0..4 {
        for i in 0..4 {
            let start = N * (i + 16 * k);
            _d_q3_q3::<N, 4>(
                &input[start..start + 13 * N],
                &mut output[start..start + 13 * N],
            );
        }
    }
}

/// z-derivative on a batch of N Q3×Q3×Q3 elements (SoA layout, 64*N dofs).
/// Processes 16 z-columns; column (i, j) starts at `N*(i + 4*j)`,
/// slice length `49*N` (stride M=16 → 3*16+1 = 49 nodes).
pub fn dz_q333_q333<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 64 * N && output.len() >= 64 * N);
    for j in 0..4 {
        for i in 0..4 {
            let start = N * (i + 4 * j);
            _d_q3_q3::<N, 16>(
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
const DX4: [f64; 9] = [
    13.51300497744847639581, // |D[0,1]|
    5.33333333333333126092,  // |D[0,2]|
    2.82032835588485264466,  // |D[0,3]|
    2.48198050606196707335,  // |D[1,0]|
    3.49148624377587779577,  // |D[1,2]|
    1.52752523165194675769,  // |D[1,3]|
    0.51801949393803470301,  // |D[1,4]|
    0.75000000000000033307,  // |D[2,0]|
    2.67316915539090604881,  // |D[2,1]|
];

// ============================================================================
// Kernel primitives
// ============================================================================

/// Vectorized 1D Q4 differentiation kernel for a batch of `N` elements.
///
/// Input and output use the SoA layout: node group `g` occupies the slice
/// `[g*M*N .. g*M*N + N]` (five groups: g = 0..4).  The stride `M` selects
/// the differentiation direction:
///   - `M = 1`  → x-direction
///   - `M = 5`  → y-direction (stride 5 in a 5×5 face)
///   - `M = 25` → z-direction (stride 25 in a 5×5×5 cube)
///
/// Loops are kept separate for independent SIMD vectorization.
pub fn _d_q4_q4<const N: usize, const M: usize>(in_q4: &[f64], out_q4: &mut [f64]) {
    assert!(in_q4.len() >= N * (4 * M + 1));
    assert!(out_q4.len() >= N * (4 * M + 1));

    let a = &in_q4[0..N];
    let b = &in_q4[M * N..M * N + N];
    let c = &in_q4[2 * M * N..2 * M * N + N];
    let d = &in_q4[3 * M * N..3 * M * N + N];
    let e = &in_q4[4 * M * N..4 * M * N + N];

    for i in 0..N {
        out_q4[i] = -10.0 * a[i] + DX4[0] * b[i] - DX4[1] * c[i] + DX4[2] * d[i] - 1.0 * e[i];
    }
    for i in 0..N {
        out_q4[1 * M * N + i] = -DX4[3] * a[i] + DX4[4] * c[i] - DX4[5] * d[i] + DX4[6] * e[i];
    }
    for i in 0..N {
        out_q4[2 * M * N + i] = DX4[7] * a[i] - DX4[8] * b[i] + DX4[8] * d[i] - DX4[7] * e[i];
    }
    for i in 0..N {
        out_q4[3 * M * N + i] = -DX4[6] * a[i] + DX4[5] * b[i] - DX4[4] * c[i] + DX4[3] * e[i];
    }
    for i in 0..N {
        out_q4[4 * M * N + i] =
            1.0 * a[i] - DX4[2] * b[i] + DX4[1] * c[i] - DX4[0] * d[i] + 10.0 * e[i];
    }
}

// ============================================================================
// Differentiation operators  (batch of N elements, SoA layout)
//
// SoA layout for a batch of N Q4×Q4×Q4 elements:
//   position of element e at node (i, j, k)  =  N*(i + 5*j + 25*k) + e
//
// The same layout is used for all three directions; only the stride M passed
// to `_d_q4_q4` changes (M=1 for x, M=5 for y, M=25 for z).
// ============================================================================

// ----------------------------------------------------------------------------
// 1D  —  Q4  (5*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q4 elements (SoA layout, 5*N dofs).
pub fn dx_q4_q4<const N: usize>(input: &[f64], output: &mut [f64]) {
    _d_q4_q4::<N, 1>(&input[0..N * 5], &mut output[0..N * 5]);
}

// ----------------------------------------------------------------------------
// 2D  —  Q4×Q4  (25*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q4×Q4 elements (SoA layout, 25*N dofs).
/// Processes 5 independent x-line groups of N×5 values.
pub fn dx_q44_q44<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..5 {
        let start = N * 5 * i;
        let end = start + N * 5;
        _d_q4_q4::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

// ----------------------------------------------------------------------------
// 3D  —  Q4×Q4×Q4  (125*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q4×Q4×Q4 elements (SoA layout, 125*N dofs).
/// Processes 25 independent x-line groups of N×5 values (stride M=1).
pub fn dx_q444_q444<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..25 {
        let start = N * 5 * i;
        let end = start + N * 5;
        _d_q4_q4::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

/// y-derivative on a batch of N Q4×Q4×Q4 elements (SoA layout, 125*N dofs).
/// Processes 25 y-columns; column (i, k) starts at `N*(i + 25*k)`,
/// slice length `21*N` (stride M=5 → 4*5+1 = 21 nodes).
pub fn dy_q444_q444<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 125 * N && output.len() >= 125 * N);
    for k in 0..5 {
        for i in 0..5 {
            let start = N * (i + 25 * k);
            _d_q4_q4::<N, 5>(
                &input[start..start + 21 * N],
                &mut output[start..start + 21 * N],
            );
        }
    }
}

/// z-derivative on a batch of N Q4×Q4×Q4 elements (SoA layout, 125*N dofs).
/// Processes 25 z-columns; column (i, j) starts at `N*(i + 5*j)`,
/// slice length `101*N` (stride M=25 → 4*25+1 = 101 nodes).
pub fn dz_q444_q444<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 125 * N && output.len() >= 125 * N);
    for j in 0..5 {
        for i in 0..5 {
            let start = N * (i + 5 * j);
            _d_q4_q4::<N, 25>(
                &input[start..start + 101 * N],
                &mut output[start..start + 101 * N],
            );
        }
    }
}

// ============================================================================
// Q5 operators  (6-node Gauss-Lobatto, degree 5)
// ============================================================================

// ----------------------------------------------------------------------------
// Differentiation coefficients for Q5 (6-node Gauss-Lobatto, degree 5)
//
// Generated by `cargo run --bin gen_q5`.
// Unique absolute values of D[i,j] for rows 0,1,2 (rows 3,4,5 follow from
// the antisymmetry D[5-i,5-j] = -D[i,j]).  Integer-valued entries (±1, ±15)
// are inlined directly in the kernel.
// ----------------------------------------------------------------------------
const DX5: [f64; 14] = [
    20.28283187263933839972, // DX5[0]  = |D[0][1]|
    8.07237454061069925615,  // DX5[1]  = |D[0][2]|
    4.48936929635233550329,  // DX5[2]  = |D[0][3]|
    2.69982662838097731139,  // DX5[3]  = |D[0][4]|
    3.57272989667818707815,  // DX5[4]  = |D[1][0]|
    5.04685355485891129490,  // DX5[5]  = |D[1][2]|
    2.30565631707185847432,  // DX5[6]  = |D[1][3]|
    1.30709501485960055689,  // DX5[7]  = |D[1][4]|
    0.47556235596846258007,  // DX5[8]  = |D[1][5]|
    0.96990209570713803444,  // DX5[9]  = |D[2][0]|
    3.44251390566046788777,  // DX5[10] = |D[2][1]|
    3.50592393273573055268,  // DX5[11] = |D[2][3]|
    1.57271334444648158701,  // DX5[12] = |D[2][4]|
    0.53940122166407777904,  // DX5[13] = |D[2][5]|
];

// ============================================================================
// Kernel primitives
// ============================================================================

/// Vectorized 1D Q5 differentiation kernel for a batch of `N` elements.
///
/// Input and output use the SoA layout: node group `g` occupies the slice
/// `[g*M*N .. g*M*N + N]` (six groups: g = 0..5).  The stride `M` selects
/// the differentiation direction:
///   - `M = 1`  → x-direction
///   - `M = 6`  → y-direction (stride 6 in a 6×6 face)
///   - `M = 36` → z-direction (stride 36 in a 6×6×6 cube)
///
/// Rows 3,4,5 are computed via the antisymmetry D[5-i,5-j] = -D[i,j].
/// Loops are kept separate for independent SIMD vectorization.
pub fn _d_q5_q5<const N: usize, const M: usize>(in_q5: &[f64], out_q5: &mut [f64]) {
    assert!(in_q5.len() >= N * (5 * M + 1));
    assert!(out_q5.len() >= N * (5 * M + 1));

    let a = &in_q5[0..N];
    let b = &in_q5[M * N..M * N + N];
    let c = &in_q5[2 * M * N..2 * M * N + N];
    let d = &in_q5[3 * M * N..3 * M * N + N];
    let e = &in_q5[4 * M * N..4 * M * N + N];
    let f = &in_q5[5 * M * N..5 * M * N + N];

    for i in 0..N {
        out_q5[i] =
            -15.0 * a[i] + DX5[0] * b[i] - DX5[1] * c[i] + DX5[2] * d[i] - DX5[3] * e[i] + f[i];
    }
    for i in 0..N {
        out_q5[M * N + i] =
            -DX5[4] * a[i] + DX5[5] * c[i] - DX5[6] * d[i] + DX5[7] * e[i] - DX5[8] * f[i];
    }
    for i in 0..N {
        out_q5[2 * M * N + i] =
            DX5[9] * a[i] - DX5[10] * b[i] + DX5[11] * d[i] - DX5[12] * e[i] + DX5[13] * f[i];
    }
    for i in 0..N {
        out_q5[3 * M * N + i] =
            -DX5[13] * a[i] + DX5[12] * b[i] - DX5[11] * c[i] + DX5[10] * e[i] - DX5[9] * f[i];
    }
    for i in 0..N {
        out_q5[4 * M * N + i] =
            DX5[8] * a[i] - DX5[7] * b[i] + DX5[6] * c[i] - DX5[5] * d[i] + DX5[4] * f[i];
    }
    for i in 0..N {
        out_q5[5 * M * N + i] =
            -a[i] + DX5[3] * b[i] - DX5[2] * c[i] + DX5[1] * d[i] - DX5[0] * e[i] + 15.0 * f[i];
    }
}

// ============================================================================
// Differentiation operators  (batch of N elements, SoA layout)
//
// SoA layout for a batch of N Q5×Q5×Q5 elements:
//   position of element e at node (i, j, k)  =  N*(i + 6*j + 36*k) + e
//
// The same layout is used for all three directions; only the stride M passed
// to `_d_q5_q5` changes (M=1 for x, M=6 for y, M=36 for z).
// ============================================================================

// ----------------------------------------------------------------------------
// 1D  —  Q5  (6*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q5 elements (SoA layout, 6*N dofs).
pub fn dx_q5_q5<const N: usize>(input: &[f64], output: &mut [f64]) {
    _d_q5_q5::<N, 1>(&input[0..N * 6], &mut output[0..N * 6]);
}

// ----------------------------------------------------------------------------
// 2D  —  Q5×Q5  (36*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q5×Q5 elements (SoA layout, 36*N dofs).
/// Processes 6 independent x-line groups of N×6 values.
pub fn dx_q55_q55<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..6 {
        let start = N * 6 * i;
        let end = start + N * 6;
        _d_q5_q5::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

// ----------------------------------------------------------------------------
// 3D  —  Q5×Q5×Q5  (216*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q5×Q5×Q5 elements (SoA layout, 216*N dofs).
/// Processes 36 independent x-line groups of N×6 values (stride M=1).
pub fn dx_q555_q555<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..36 {
        let start = N * 6 * i;
        let end = start + N * 6;
        _d_q5_q5::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

/// y-derivative on a batch of N Q5×Q5×Q5 elements (SoA layout, 216*N dofs).
/// Processes 36 y-columns; column (i, k) starts at `N*(i + 36*k)`,
/// slice length `31*N` (stride M=6 → 5*6+1 = 31 nodes).
pub fn dy_q555_q555<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 216 * N && output.len() >= 216 * N);
    for k in 0..6 {
        for i in 0..6 {
            let start = N * (i + 36 * k);
            _d_q5_q5::<N, 6>(
                &input[start..start + 31 * N],
                &mut output[start..start + 31 * N],
            );
        }
    }
}

/// z-derivative on a batch of N Q5×Q5×Q5 elements (SoA layout, 216*N dofs).
/// Processes 36 z-columns; column (i, j) starts at `N*(i + 6*j)`,
/// slice length `181*N` (stride M=36 → 5*36+1 = 181 nodes).
pub fn dz_q555_q555<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 216 * N && output.len() >= 216 * N);
    for j in 0..6 {
        for i in 0..6 {
            let start = N * (i + 6 * j);
            _d_q5_q5::<N, 36>(
                &input[start..start + 181 * N],
                &mut output[start..start + 181 * N],
            );
        }
    }
}

// ============================================================================
// Q6 operators  (7-node Gauss-Lobatto, degree 6)
// ============================================================================

// ----------------------------------------------------------------------------
// Differentiation coefficients for Q6 (7-node Gauss-Lobatto, degree 6)
//
// Generated by `cargo run --bin gen_q6`.
// Unique absolute values of D[i,j] for rows 0,1,2,3 (rows 4,5,6 follow from
// the antisymmetry D[6-i,6-j] = -D[i,j]).  Integer-valued entries (±1, ±21)
// are inlined directly in the kernel.
// ----------------------------------------------------------------------------
const DX6: [f64; 20] = [
    28.40315320583964009415, // DX6[0]  = |D[0,1]|
    11.33797045109102263893, // DX6[1]  = |D[0,2]|
    6.39999999999999680256,  // DX6[2]  = |D[0,3]|
    4.09992962615348588429,  // DX6[3]  = |D[0,4]|
    2.63474687140487073833,  // DX6[4]  = |D[0,5]|
    4.88585202848857402813,  // DX6[5]  = |D[1,0]|
    6.91165642858857065534,  // DX6[6]  = |D[1,2]|
    3.19721337619673073860,  // DX6[7]  = |D[1,3]|
    1.92267959457742243501,  // DX6[8]  = |D[1,4]|
    1.20449435927157155746,  // DX6[9]  = |D[1,5]|
    0.45322374079089045029,  // DX6[10] = |D[1,6]|
    1.25051333103068440522,  // DX6[11] = |D[2,0]|
    4.43160856633994804099,  // DX6[12] = |D[2,1]|
    4.53339617417199303162,  // DX6[13] = |D[2,3]|
    2.13288380801274835008,  // DX6[14] = |D[2,4]|
    1.23278167103515956526,  // DX6[15] = |D[2,5]|
    0.45219880188514915975,  // DX6[16] = |D[2,6]|
    0.62500000000000022204,  // DX6[17] = |D[3,0]|
    1.81508894253764396787,  // DX6[18] = |D[3,1]|
    4.01393848117750717819,  // DX6[19] = |D[3,2]|
];

// ============================================================================
// Kernel primitives
// ============================================================================

/// Vectorized 1D Q6 differentiation kernel for a batch of `N` elements.
///
/// Input and output use the SoA layout: node group `g` occupies the slice
/// `[g*M*N .. g*M*N + N]` (seven groups: g = 0..6).  The stride `M` selects
/// the differentiation direction:
///   - `M = 1`  → x-direction
///   - `M = 7`  → y-direction (stride 7 in a 7×7 face)
///   - `M = 49` → z-direction (stride 49 in a 7×7×7 cube)
///
/// Rows 4,5,6 are computed via the antisymmetry D[6-i,6-j] = -D[i,j].
/// Loops are kept separate for independent SIMD vectorization.
pub fn _d_q6_q6<const N: usize, const M: usize>(in_q6: &[f64], out_q6: &mut [f64]) {
    assert!(in_q6.len() >= N * (6 * M + 1));
    assert!(out_q6.len() >= N * (6 * M + 1));

    let a = &in_q6[0..N];
    let b = &in_q6[M * N..M * N + N];
    let c = &in_q6[2 * M * N..2 * M * N + N];
    let d = &in_q6[3 * M * N..3 * M * N + N];
    let e = &in_q6[4 * M * N..4 * M * N + N];
    let f = &in_q6[5 * M * N..5 * M * N + N];
    let g = &in_q6[6 * M * N..6 * M * N + N];

    for i in 0..N {
        out_q6[i] = -21.0 * a[i] + DX6[0] * b[i] - DX6[1] * c[i] + DX6[2] * d[i] - DX6[3] * e[i]
            + DX6[4] * f[i]
            - 1.0 * g[i];
    }
    for i in 0..N {
        out_q6[1 * M * N + i] = -DX6[5] * a[i] + DX6[6] * c[i] - DX6[7] * d[i] + DX6[8] * e[i]
            - DX6[9] * f[i]
            + DX6[10] * g[i];
    }
    for i in 0..N {
        out_q6[2 * M * N + i] = DX6[11] * a[i] - DX6[12] * b[i] + DX6[13] * d[i] - DX6[14] * e[i]
            + DX6[15] * f[i]
            - DX6[16] * g[i];
    }
    for i in 0..N {
        out_q6[3 * M * N + i] = -DX6[17] * a[i] + DX6[18] * b[i] - DX6[19] * c[i] + DX6[19] * e[i]
            - DX6[18] * f[i]
            + DX6[17] * g[i];
    }
    for i in 0..N {
        out_q6[4 * M * N + i] = DX6[16] * a[i] - DX6[15] * b[i] + DX6[14] * c[i] - DX6[13] * d[i]
            + DX6[12] * f[i]
            - DX6[11] * g[i];
    }
    for i in 0..N {
        out_q6[5 * M * N + i] = -DX6[10] * a[i] + DX6[9] * b[i] - DX6[8] * c[i] + DX6[7] * d[i]
            - DX6[6] * e[i]
            + DX6[5] * g[i];
    }
    for i in 0..N {
        out_q6[6 * M * N + i] = 1.0 * a[i] - DX6[4] * b[i] + DX6[3] * c[i] - DX6[2] * d[i]
            + DX6[1] * e[i]
            - DX6[0] * f[i]
            + 21.0 * g[i];
    }
}

// ============================================================================
// Differentiation operators  (batch of N elements, SoA layout)
//
// SoA layout for a batch of N Q6×Q6×Q6 elements:
//   position of element e at node (i, j, k)  =  N*(i + 7*j + 49*k) + e
//
// The same layout is used for all three directions; only the stride M passed
// to `_d_q6_q6` changes (M=1 for x, M=7 for y, M=49 for z).
// ============================================================================

// ----------------------------------------------------------------------------
// 1D  —  Q6  (7*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q6 elements (SoA layout, 7*N dofs).
pub fn dx_q6_q6<const N: usize>(input: &[f64], output: &mut [f64]) {
    _d_q6_q6::<N, 1>(&input[0..N * 7], &mut output[0..N * 7]);
}

// ----------------------------------------------------------------------------
// 2D  —  Q6×Q6  (49*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q6×Q6 elements (SoA layout, 49*N dofs).
/// Processes 7 independent x-line groups of N×7 values.
pub fn dx_q66_q66<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..7 {
        let start = N * 7 * i;
        let end = start + N * 7;
        _d_q6_q6::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

// ----------------------------------------------------------------------------
// 3D  —  Q6×Q6×Q6  (343*N dofs)
// ----------------------------------------------------------------------------

/// x-derivative on a batch of N Q6×Q6×Q6 elements (SoA layout, 343*N dofs).
/// Processes 49 independent x-line groups of N×7 values (stride M=1).
pub fn dx_q666_q666<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..49 {
        let start = N * 7 * i;
        let end = start + N * 7;
        _d_q6_q6::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

/// y-derivative on a batch of N Q6×Q6×Q6 elements (SoA layout, 343*N dofs).
/// Processes 49 y-columns; column (i, k) starts at `N*(i + 49*k)`,
/// slice length `43*N` (stride M=7 → 6*7+1 = 43 nodes).
pub fn dy_q666_q666<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 343 * N && output.len() >= 343 * N);
    for k in 0..7 {
        for i in 0..7 {
            let start = N * (i + 49 * k);
            _d_q6_q6::<N, 7>(
                &input[start..start + 43 * N],
                &mut output[start..start + 43 * N],
            );
        }
    }
}

/// z-derivative on a batch of N Q6×Q6×Q6 elements (SoA layout, 343*N dofs).
/// Processes 49 z-columns; column (i, j) starts at `N*(i + 7*j)`,
/// slice length `295*N` (stride M=49 → 6*49+1 = 295 nodes).
pub fn dz_q666_q666<const N: usize>(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 343 * N && output.len() >= 343 * N);
    for j in 0..7 {
        for i in 0..7 {
            let start = N * (i + 7 * j);
            _d_q6_q6::<N, 49>(
                &input[start..start + 295 * N],
                &mut output[start..start + 295 * N],
            );
        }
    }
}
