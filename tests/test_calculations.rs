use gg_lobatto::calculator::*;
use gg_lobatto::hc_calculations::*;
use nalgebra::DVector;

/// Xorshift64 with a fixed seed — same sequence as benchmarks.
fn seeded_vec(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x3f2a1b4c8e7d5690;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f64::from_bits((state >> 12) | 0x3FF0000000000000) * 2.0 - 3.0
        })
        .collect()
}

/// AoS → SoA layout transposition (same as benchmarks).
///
/// Each element contains `groups` consecutive groups of `dofs_per_group` dofs.
/// - AoS index: `j * groups * dofs_per_group + i * dofs_per_group + k`
/// - SoA index: `i * n * dofs_per_group + k * n + j`
fn aos_to_soa(aos: &[f64], n: usize, groups: usize, dofs_per_group: usize) -> Vec<f64> {
    let mut soa = vec![0.0_f64; n * groups * dofs_per_group];
    for j in 0..n {
        for i in 0..groups {
            for k in 0..dofs_per_group {
                soa[i * n * dofs_per_group + k * n + j] =
                    aos[j * groups * dofs_per_group + i * dofs_per_group + k];
            }
        }
    }
    soa
}

/// SoA → AoS (inverse of `aos_to_soa`).
fn soa_to_aos(soa: &[f64], n: usize, groups: usize, dofs_per_group: usize) -> Vec<f64> {
    let mut aos = vec![0.0_f64; n * groups * dofs_per_group];
    for j in 0..n {
        for i in 0..groups {
            for k in 0..dofs_per_group {
                aos[j * groups * dofs_per_group + i * dofs_per_group + k] =
                    soa[i * n * dofs_per_group + k * n + j];
            }
        }
    }
    aos
}

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "index {i}: hc={x:.15e}  ref={y:.15e}  |diff|={:.3e}",
            (x - y).abs()
        );
    }
}

// ── Scalar functions vs GLE Calculator ──────────────────────────────────────

/// x-differentiation on a single Q3 element (4 dofs).
#[test]
fn test_dx_q3_q3() {
    let v_in = seeded_vec(4);
    let mut v_out_hc = vec![0.0_f64; 4];
    dx_q3_q3(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(4);
    let gle = Calculator::new(vec![(4, true)]);
    gle.diff(0, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// x-differentiation on a Q3×Q3 element (16 dofs).
#[test]
fn test_dx_q33_q33() {
    let v_in = seeded_vec(16);
    let mut v_out_hc = vec![0.0_f64; 16];
    dx_q33_q33(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(16);
    let gle = Calculator::new(vec![(4, true), (4, true)]);
    gle.diff(0, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// y-differentiation on a Q3×Q3 element (16 dofs).
#[test]
fn test_dy_q33_q33() {
    let v_in = seeded_vec(16);
    let mut v_out_hc = vec![0.0_f64; 16];
    dy_q33_q33(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(16);
    let gle = Calculator::new(vec![(4, true), (4, true)]);
    gle.diff(1, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// x-differentiation on a Q3×Q3×Q3 element (64 dofs).
#[test]
fn test_dx_q333_q333() {
    let v_in = seeded_vec(64);
    let mut v_out_hc = vec![0.0_f64; 64];
    dx_q333_q333(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(64);
    let gle = Calculator::new(vec![(4, true), (4, true), (4, true)]);
    gle.diff(0, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// y-differentiation on a Q3×Q3×Q3 element (64 dofs).
#[test]
fn test_dy_q333_q333() {
    let v_in = seeded_vec(64);
    let mut v_out_hc = vec![0.0_f64; 64];
    dy_q333_q333(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(64);
    let gle = Calculator::new(vec![(4, true), (4, true), (4, true)]);
    gle.diff(1, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// z-differentiation on a Q3×Q3×Q3 element (64 dofs).
#[test]
fn test_dz_q333_q333() {
    let v_in = seeded_vec(64);
    let mut v_out_hc = vec![0.0_f64; 64];
    dz_q333_q333(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(64);
    let gle = Calculator::new(vec![(4, true), (4, true), (4, true)]);
    gle.diff(2, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

// ── Vectorized (SoA) functions vs scalar equivalents ────────────────────────
//
// The vec functions use SoA layout: for a batch of N elements each with G groups
// of D dofs, the input is organised as G contiguous blocks of N×D values.
// Tests verify vec output matches scalar output after converting layouts.

/// x-differentiation on a batch of N Q3 elements: `dx_q3_q3_vec` vs `dx_q3_q3`.
/// Layout: 1 group × 4 dofs  →  aos_to_soa(n, 1, 4).
#[test]
fn test_dx_q3_q3_vec() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(4 * N);

    let mut v_out_scalar = vec![0.0_f64; 4 * N];
    for i in 0..N {
        let s = 4 * i;
        dx_q3_q3(&v_in_aos[s..s + 4], &mut v_out_scalar[s..s + 4]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 1, 4);
    let mut v_out_soa = vec![0.0_f64; 4 * N];
    dx_q3_q3_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 1, 4);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}

/// x-differentiation on a batch of N Q3×Q3 elements: `dx_q33_q33_vec` vs `dx_q33_q33`.
/// Layout: 4 groups × 4 dofs  →  aos_to_soa(n, 4, 4).
#[test]
fn test_dx_q33_q33_vec() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(16 * N);

    let mut v_out_scalar = vec![0.0_f64; 16 * N];
    for i in 0..N {
        let s = 16 * i;
        dx_q33_q33(&v_in_aos[s..s + 16], &mut v_out_scalar[s..s + 16]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 4, 4);
    let mut v_out_soa = vec![0.0_f64; 16 * N];
    dx_q33_q33_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 4, 4);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}

/// x-differentiation on a batch of N Q3×Q3×Q3 elements: `dx_q333_q333_vec` vs `dx_q333_q333`.
/// Layout: 16 groups × 4 dofs  →  aos_to_soa(n, 16, 4).
#[test]
fn test_dx_q333_q333_vec() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(64 * N);

    let mut v_out_scalar = vec![0.0_f64; 64 * N];
    for i in 0..N {
        let s = 64 * i;
        dx_q333_q333(&v_in_aos[s..s + 64], &mut v_out_scalar[s..s + 64]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 16, 4);
    let mut v_out_soa = vec![0.0_f64; 64 * N];
    dx_q333_q333_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 16, 4);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}

// For dy/dz vec, input is dx-SoA (same aos_to_soa(n, 16, 4)) and output stays in that
// same format (position = N*(i+4j+16k)+e), so soa_to_aos(n, 16, 4) converts it back.

/// y-differentiation on a batch of N Q3×Q3×Q3 elements: `dy_q333_q333_vec` vs `dy_q333_q333`.
#[test]
fn test_dy_q333_q333_vec() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(64 * N);

    let mut v_out_scalar = vec![0.0_f64; 64 * N];
    for i in 0..N {
        let s = 64 * i;
        dy_q333_q333(&v_in_aos[s..s + 64], &mut v_out_scalar[s..s + 64]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 16, 4);
    let mut v_out_soa = vec![0.0_f64; 64 * N];
    dy_q333_q333_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 16, 4);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}

/// z-differentiation on a batch of N Q3×Q3×Q3 elements: `dz_q333_q333_vec` vs `dz_q333_q333`.
#[test]
fn test_dz_q333_q333_vec() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(64 * N);

    let mut v_out_scalar = vec![0.0_f64; 64 * N];
    for i in 0..N {
        let s = 64 * i;
        dz_q333_q333(&v_in_aos[s..s + 64], &mut v_out_scalar[s..s + 64]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 16, 4);
    let mut v_out_soa = vec![0.0_f64; 64 * N];
    dz_q333_q333_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 16, 4);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}

// ── Q4 scalar functions vs GLE Calculator ────────────────────────────────────

/// x-differentiation on a single Q4 element (5 dofs).
#[test]
fn test_dx_q4_q4() {
    let v_in = seeded_vec(5);
    let mut v_out_hc = vec![0.0_f64; 5];
    dx_q4_q4(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(5);
    let gle = Calculator::new(vec![(5, true)]);
    gle.diff(0, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// x-differentiation on a Q4×Q4 element (25 dofs).
#[test]
fn test_dx_q44_q44() {
    let v_in = seeded_vec(25);
    let mut v_out_hc = vec![0.0_f64; 25];
    dx_q44_q44(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(25);
    let gle = Calculator::new(vec![(5, true), (5, true)]);
    gle.diff(0, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// y-differentiation on a Q4×Q4 element (25 dofs).
#[test]
fn test_dy_q44_q44() {
    let v_in = seeded_vec(25);
    let mut v_out_hc = vec![0.0_f64; 25];
    dy_q44_q44(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(25);
    let gle = Calculator::new(vec![(5, true), (5, true)]);
    gle.diff(1, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// x-differentiation on a Q4×Q4×Q4 element (125 dofs).
#[test]
fn test_dx_q444_q444() {
    let v_in = seeded_vec(125);
    let mut v_out_hc = vec![0.0_f64; 125];
    dx_q444_q444(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(125);
    let gle = Calculator::new(vec![(5, true), (5, true), (5, true)]);
    gle.diff(0, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// y-differentiation on a Q4×Q4×Q4 element (125 dofs).
#[test]
fn test_dy_q444_q444() {
    let v_in = seeded_vec(125);
    let mut v_out_hc = vec![0.0_f64; 125];
    dy_q444_q444(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(125);
    let gle = Calculator::new(vec![(5, true), (5, true), (5, true)]);
    gle.diff(1, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

/// z-differentiation on a Q4×Q4×Q4 element (125 dofs).
#[test]
fn test_dz_q444_q444() {
    let v_in = seeded_vec(125);
    let mut v_out_hc = vec![0.0_f64; 125];
    dz_q444_q444(&v_in, &mut v_out_hc);

    let mut v_out_gle = DVector::zeros(125);
    let gle = Calculator::new(vec![(5, true), (5, true), (5, true)]);
    gle.diff(2, &DVector::from_vec(v_in), &mut v_out_gle);

    assert_close(&v_out_hc, v_out_gle.as_slice(), 1e-10);
}

// ── Q4 vectorized (SoA) functions vs scalar equivalents ──────────────────────
//
// SoA layout for Q4×Q4×Q4: position = N*(i + 5*j + 25*k) + e.
// Transposition parameters: 25 groups × 5 dofs  →  aos_to_soa(n, 25, 5).

/// x-differentiation on a batch of N Q4 elements: `dx_q4_q4_vec` vs `dx_q4_q4`.
/// Layout: 1 group × 5 dofs  →  aos_to_soa(n, 1, 5).
#[test]
fn test_dx_q4_q4_vec() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(5 * N);

    let mut v_out_scalar = vec![0.0_f64; 5 * N];
    for i in 0..N {
        let s = 5 * i;
        dx_q4_q4(&v_in_aos[s..s + 5], &mut v_out_scalar[s..s + 5]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 1, 5);
    let mut v_out_soa = vec![0.0_f64; 5 * N];
    dx_q4_q4_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 1, 5);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}

/// x-differentiation on a batch of N Q4×Q4 elements: `dx_q44_q44_vec` vs `dx_q44_q44`.
/// Layout: 5 groups × 5 dofs  →  aos_to_soa(n, 5, 5).
#[test]
fn test_dx_q44_q44_vec() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(25 * N);

    let mut v_out_scalar = vec![0.0_f64; 25 * N];
    for i in 0..N {
        let s = 25 * i;
        dx_q44_q44(&v_in_aos[s..s + 25], &mut v_out_scalar[s..s + 25]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 5, 5);
    let mut v_out_soa = vec![0.0_f64; 25 * N];
    dx_q44_q44_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 5, 5);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}

/// x-differentiation on a batch of N Q4×Q4×Q4 elements: `dx_q444_q444_vec` vs `dx_q444_q444`.
/// Layout: 25 groups × 5 dofs  →  aos_to_soa(n, 25, 5).
#[test]
fn test_dx_q444_q444_vec() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(125 * N);

    let mut v_out_scalar = vec![0.0_f64; 125 * N];
    for i in 0..N {
        let s = 125 * i;
        dx_q444_q444(&v_in_aos[s..s + 125], &mut v_out_scalar[s..s + 125]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 25, 5);
    let mut v_out_soa = vec![0.0_f64; 125 * N];
    dx_q444_q444_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 25, 5);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}

/// y-differentiation on a batch of N Q4×Q4×Q4 elements: `dy_q444_q444_vec` vs `dy_q444_q444`.
#[test]
fn test_dy_q444_q444_vec() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(125 * N);

    let mut v_out_scalar = vec![0.0_f64; 125 * N];
    for i in 0..N {
        let s = 125 * i;
        dy_q444_q444(&v_in_aos[s..s + 125], &mut v_out_scalar[s..s + 125]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 25, 5);
    let mut v_out_soa = vec![0.0_f64; 125 * N];
    dy_q444_q444_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 25, 5);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}

/// z-differentiation on a batch of N Q4×Q4×Q4 elements: `dz_q444_q444_vec` vs `dz_q444_q444`.
#[test]
fn test_dz_q444_q444_vec() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(125 * N);

    let mut v_out_scalar = vec![0.0_f64; 125 * N];
    for i in 0..N {
        let s = 125 * i;
        dz_q444_q444(&v_in_aos[s..s + 125], &mut v_out_scalar[s..s + 125]);
    }

    let v_in_soa = aos_to_soa(&v_in_aos, N, 25, 5);
    let mut v_out_soa = vec![0.0_f64; 125 * N];
    dz_q444_q444_vec::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_vec = soa_to_aos(&v_out_soa, N, 25, 5);

    assert_close(&v_out_scalar, &v_out_vec, 1e-14);
}
