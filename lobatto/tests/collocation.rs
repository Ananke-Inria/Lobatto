use lobatto::collocation::*;
use lobatto::collocation_hc::*;

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

/// AoS → SoA layout transposition.
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

/// Apply GLE `diff(dir)` to each of `n` elements (AoS layout, `dofs` dofs each).
fn gll_batch_ref(
    v_in_aos: &[f64],
    n: usize,
    dofs: usize,
    dir: usize,
    gle: &CollocationBasis,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * dofs];
    for i in 0..n {
        let s = dofs * i;
        gle.diff(dir, &v_in_aos[s..s + dofs], &mut out[s..s + dofs]);
    }
    out
}

// ── N=1 tests vs GLE Basis ─────────────────────────────────────────────
//
// With N=1 the SoA layout is identical to AoS, so the functions can be called
// directly on the plain input vector and compared against the GLE reference.

#[test]
fn test_dx_q3_q3() {
    let v_in = seeded_vec(4);
    let mut v_out = vec![0.0_f64; 4];
    dx_q3_q3::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 4];
    CollocationBasis::new(vec![(4, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dx_q33_q33() {
    let v_in = seeded_vec(16);
    let mut v_out = vec![0.0_f64; 16];
    dx_q33_q33::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 16];
    CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dy_q33_q33() {
    let v_in = seeded_vec(16);
    let mut v_out = vec![0.0_f64; 16];
    dy_q33_q33::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 16];
    CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto)]).diff(1, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dx_q333_q333() {
    let v_in = seeded_vec(64);
    let mut v_out = vec![0.0_f64; 64];
    dx_q333_q333::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 64];
    CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto), (4, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dy_q333_q333() {
    let v_in = seeded_vec(64);
    let mut v_out = vec![0.0_f64; 64];
    dy_q333_q333::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 64];
    CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto), (4, Gauss::Lobatto)]).diff(1, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dz_q333_q333() {
    let v_in = seeded_vec(64);
    let mut v_out = vec![0.0_f64; 64];
    dz_q333_q333::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 64];
    CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto), (4, Gauss::Lobatto)]).diff(2, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dx_q4_q4() {
    let v_in = seeded_vec(5);
    let mut v_out = vec![0.0_f64; 5];
    dx_q4_q4::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 5];
    CollocationBasis::new(vec![(5, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dx_q44_q44() {
    let v_in = seeded_vec(25);
    let mut v_out = vec![0.0_f64; 25];
    dx_q44_q44::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 25];
    CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dy_q44_q44() {
    let v_in = seeded_vec(25);
    let mut v_out = vec![0.0_f64; 25];
    dy_q44_q44::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 25];
    CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto)]).diff(1, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dx_q444_q444() {
    let v_in = seeded_vec(125);
    let mut v_out = vec![0.0_f64; 125];
    dx_q444_q444::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 125];
    CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto), (5, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dy_q444_q444() {
    let v_in = seeded_vec(125);
    let mut v_out = vec![0.0_f64; 125];
    dy_q444_q444::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 125];
    CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto), (5, Gauss::Lobatto)]).diff(1, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dz_q444_q444() {
    let v_in = seeded_vec(125);
    let mut v_out = vec![0.0_f64; 125];
    dz_q444_q444::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 125];
    CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto), (5, Gauss::Lobatto)]).diff(2, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

// ── Batch N=8 tests vs GLE ───────────────────────────────────────────────────
//
// Batch input is prepared in AoS layout, transposed to SoA, processed by the
// generic function, transposed back to AoS, then compared element-wise against
// the GLE reference.

#[test]
fn test_dx_q3_q3_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(4 * N);
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 4, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 1, 4);
    let mut v_out_soa = vec![0.0_f64; 4 * N];
    dx_q3_q3::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 1, 4);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dx_q33_q33_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(16 * N);
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 16, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 4, 4);
    let mut v_out_soa = vec![0.0_f64; 16 * N];
    dx_q33_q33::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 4, 4);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dy_q33_q33_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(16 * N);
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 16, 1, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 4, 4);
    let mut v_out_soa = vec![0.0_f64; 16 * N];
    dy_q33_q33::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 4, 4);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dx_q333_q333_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(64 * N);
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto), (4, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 64, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 16, 4);
    let mut v_out_soa = vec![0.0_f64; 64 * N];
    dx_q333_q333::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 16, 4);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dy_q333_q333_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(64 * N);
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto), (4, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 64, 1, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 16, 4);
    let mut v_out_soa = vec![0.0_f64; 64 * N];
    dy_q333_q333::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 16, 4);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dz_q333_q333_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(64 * N);
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto), (4, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 64, 2, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 16, 4);
    let mut v_out_soa = vec![0.0_f64; 64 * N];
    dz_q333_q333::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 16, 4);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dx_q4_q4_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(5 * N);
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 5, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 1, 5);
    let mut v_out_soa = vec![0.0_f64; 5 * N];
    dx_q4_q4::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 1, 5);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dx_q44_q44_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(25 * N);
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 25, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 5, 5);
    let mut v_out_soa = vec![0.0_f64; 25 * N];
    dx_q44_q44::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 5, 5);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dy_q44_q44_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(25 * N);
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 25, 1, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 5, 5);
    let mut v_out_soa = vec![0.0_f64; 25 * N];
    dy_q44_q44::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 5, 5);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dx_q444_q444_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(125 * N);
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto), (5, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 125, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 25, 5);
    let mut v_out_soa = vec![0.0_f64; 125 * N];
    dx_q444_q444::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 25, 5);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dy_q444_q444_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(125 * N);
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto), (5, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 125, 1, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 25, 5);
    let mut v_out_soa = vec![0.0_f64; 125 * N];
    dy_q444_q444::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 25, 5);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dz_q444_q444_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(125 * N);
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto), (5, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 125, 2, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 25, 5);
    let mut v_out_soa = vec![0.0_f64; 125 * N];
    dz_q444_q444::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 25, 5);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

// ── Q5 N=1 tests vs GLE Basis ──────────────────────────────────────────

#[test]
fn test_dx_q5_q5() {
    let v_in = seeded_vec(6);
    let mut v_out = vec![0.0_f64; 6];
    dx_q5_q5::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 6];
    CollocationBasis::new(vec![(6, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dx_q55_q55() {
    let v_in = seeded_vec(36);
    let mut v_out = vec![0.0_f64; 36];
    dx_q55_q55::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 36];
    CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dy_q55_q55() {
    let v_in = seeded_vec(36);
    let mut v_out = vec![0.0_f64; 36];
    dy_q55_q55::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 36];
    CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto)]).diff(1, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dx_q555_q555() {
    let v_in = seeded_vec(216);
    let mut v_out = vec![0.0_f64; 216];
    dx_q555_q555::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 216];
    CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto), (6, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dy_q555_q555() {
    let v_in = seeded_vec(216);
    let mut v_out = vec![0.0_f64; 216];
    dy_q555_q555::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 216];
    CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto), (6, Gauss::Lobatto)]).diff(1, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dz_q555_q555() {
    let v_in = seeded_vec(216);
    let mut v_out = vec![0.0_f64; 216];
    dz_q555_q555::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 216];
    CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto), (6, Gauss::Lobatto)]).diff(2, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

// ── Q5 batch N=4/8 tests vs GLE ──────────────────────────────────────────────

#[test]
fn test_dx_q5_q5_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(6 * N);
    let gle = CollocationBasis::new(vec![(6, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 6, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 1, 6);
    let mut v_out_soa = vec![0.0_f64; 6 * N];
    dx_q5_q5::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 1, 6);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dy_q55_q55_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(36 * N);
    let gle = CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 36, 1, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 6, 6);
    let mut v_out_soa = vec![0.0_f64; 36 * N];
    dy_q55_q55::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 6, 6);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dx_q555_q555_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(216 * N);
    let gle = CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto), (6, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 216, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 36, 6);
    let mut v_out_soa = vec![0.0_f64; 216 * N];
    dx_q555_q555::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 36, 6);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dy_q555_q555_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(216 * N);
    let gle = CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto), (6, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 216, 1, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 36, 6);
    let mut v_out_soa = vec![0.0_f64; 216 * N];
    dy_q555_q555::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 36, 6);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dz_q555_q555_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(216 * N);
    let gle = CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto), (6, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 216, 2, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 36, 6);
    let mut v_out_soa = vec![0.0_f64; 216 * N];
    dz_q555_q555::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 36, 6);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

// ── Q6 N=1 tests vs GLE Basis ──────────────────────────────────────────

#[test]
fn test_dx_q6_q6() {
    let v_in = seeded_vec(7);
    let mut v_out = vec![0.0_f64; 7];
    dx_q6_q6::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 7];
    CollocationBasis::new(vec![(7, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dx_q66_q66() {
    let v_in = seeded_vec(49);
    let mut v_out = vec![0.0_f64; 49];
    dx_q66_q66::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 49];
    CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dy_q66_q66() {
    let v_in = seeded_vec(49);
    let mut v_out = vec![0.0_f64; 49];
    dy_q66_q66::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 49];
    CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto)]).diff(1, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dx_q666_q666() {
    let v_in = seeded_vec(343);
    let mut v_out = vec![0.0_f64; 343];
    dx_q666_q666::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 343];
    CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto), (7, Gauss::Lobatto)]).diff(0, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dy_q666_q666() {
    let v_in = seeded_vec(343);
    let mut v_out = vec![0.0_f64; 343];
    dy_q666_q666::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 343];
    CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto), (7, Gauss::Lobatto)]).diff(1, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

#[test]
fn test_dz_q666_q666() {
    let v_in = seeded_vec(343);
    let mut v_out = vec![0.0_f64; 343];
    dz_q666_q666::<1>(&v_in, &mut v_out);

    let mut ref_out = vec![0.0_f64; 343];
    CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto), (7, Gauss::Lobatto)]).diff(2, &v_in, &mut ref_out);
    assert_close(&v_out, &ref_out, 1e-10);
}

// ── eval ─────────────────────────────────────────────────────────────────────

#[test]
fn test_eval_q4_3d() {
    // Q4 in 3D: 5 GLL nodes per direction, 5^3 = 125 DOFs.
    let basis = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto), (5, Gauss::Lobatto)]);

    // A polynomial of degree ≤ 4 in each variable — exactly representable in Q4.
    let f = |x: f64, y: f64, z: f64| -> f64 {
        x.powi(4) + 2.0 * x.powi(3) * y + 3.0 * x.powi(2) * z.powi(2)
            + y.powi(4) - y.powi(3) * z + z.powi(4)
    };

    // DOF values: f evaluated at the 125 collocation points.
    let u: Vec<f64> = basis.points().iter().map(|p| f(p[0], p[1], p[2])).collect();

    // 200 random query points in [0, 1]^3 (seeded, reproducible).
    // seeded_vec produces values in [-1, 1); map to [0, 1).
    let xs: Vec<Vec<f64>> = seeded_vec(200 * 3)
        .chunks(3)
        .map(|c| c.iter().map(|&v| (v + 1.0) / 2.0).collect())
        .collect();

    let result = basis.eval(&xs, &u);
    let expected: Vec<f64> = xs.iter().map(|p| f(p[0], p[1], p[2])).collect();

    assert_close(&result, &expected, 1e-10);
}

// ── Q6 batch N=4/8 tests vs GLE ──────────────────────────────────────────────

#[test]
fn test_dx_q6_q6_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(7 * N);
    let gle = CollocationBasis::new(vec![(7, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 7, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 1, 7);
    let mut v_out_soa = vec![0.0_f64; 7 * N];
    dx_q6_q6::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 1, 7);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dy_q66_q66_batch() {
    const N: usize = 8;
    let v_in_aos = seeded_vec(49 * N);
    let gle = CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 49, 1, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 7, 7);
    let mut v_out_soa = vec![0.0_f64; 49 * N];
    dy_q66_q66::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 7, 7);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dx_q666_q666_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(343 * N);
    let gle = CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto), (7, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 343, 0, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 49, 7);
    let mut v_out_soa = vec![0.0_f64; 343 * N];
    dx_q666_q666::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 49, 7);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dy_q666_q666_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(343 * N);
    let gle = CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto), (7, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 343, 1, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 49, 7);
    let mut v_out_soa = vec![0.0_f64; 343 * N];
    dy_q666_q666::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 49, 7);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}

#[test]
fn test_dz_q666_q666_batch() {
    const N: usize = 4;
    let v_in_aos = seeded_vec(343 * N);
    let gle = CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto), (7, Gauss::Lobatto)]);
    let ref_out = gll_batch_ref(&v_in_aos, N, 343, 2, &gle);

    let v_in_soa = aos_to_soa(&v_in_aos, N, 49, 7);
    let mut v_out_soa = vec![0.0_f64; 343 * N];
    dz_q666_q666::<N>(&v_in_soa, &mut v_out_soa);
    let v_out_aos = soa_to_aos(&v_out_soa, N, 49, 7);

    assert_close(&v_out_aos, &ref_out, 1e-10);
}
