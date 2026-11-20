//! Tests of `lobatto::tables` and of the generic rectangular kernels of
//! `lobatto::collocation_hc` (`dx_qn_qm`, …, `dz_t_qnnm_qnnn`).

use lobatto::collocation::{CollocationBasis, Gauss};
use lobatto::collocation_hc::*;
use lobatto::quadrature::{gauss_legendre_on_unit_interval, gauss_lobatto_on_unit_interval};
use lobatto::tables::*;
use lobatto::utilities::{barycentric_weights, lagrangian_derivative_interpolation, lagrangian_interpolation};

const FAMILIES: [Gauss; 2] = [Gauss::Lobatto, Gauss::Legendre];

/// Xorshift64 with a fixed seed, values in `[-1, 1)`.
fn seeded_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut state: u64 = 0x3f2a1b4c8e7d5690 ^ seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f64::from_bits((state >> 12) | 0x3FF0000000000000) * 2.0 - 3.0
        })
        .collect()
}

/// Polynomial `Σ_k c_k x^k` of degree `c.len() - 1` and its derivative.
fn poly(c: &[f64], x: f64) -> (f64, f64) {
    let (mut p, mut dp) = (0.0, 0.0);
    for &ck in c.iter().rev() {
        dp = dp * x + p;
        p = p * x + ck;
    }
    (p, dp)
}

fn apply<const NI: usize, const NO: usize>(m: &[[f64; NI]; NO], u: &[f64; NI]) -> [f64; NO] {
    core::array::from_fn(|j| (0..NI).map(|i| m[j][i] * u[i]).sum())
}

// ============================================================================
// Tables
// ============================================================================

fn check_points_weights<const N: usize>(family: Gauss) {
    let (x, w) = points_weights::<N>(family);
    let (xr, wr) = match family {
        Gauss::Lobatto => gauss_lobatto_on_unit_interval(N),
        Gauss::Legendre => gauss_legendre_on_unit_interval(N),
    };
    assert_eq!(x.as_slice(), xr.as_slice());
    assert_eq!(w.as_slice(), wr.as_slice());
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    assert!(x.windows(2).all(|p| p[0] < p[1]));
    if family == Gauss::Lobatto {
        assert!(x[0].abs() < 1e-15 && (x[N - 1] - 1.0).abs() < 1e-15);
    }
}

#[test]
fn points_weights_match_quadrature() {
    check_points_weights::<1>(Gauss::Legendre);
    for f in FAMILIES {
        check_points_weights::<2>(f);
        check_points_weights::<3>(f);
        check_points_weights::<5>(f);
        check_points_weights::<9>(f);
    }
}

#[test]
fn dyn_versions_match_fixed_size() {
    for f in FAMILIES {
        let (x, w) = points_weights::<5>(f);
        let (xd, wd) = points_weights_dyn(5, f);
        assert_eq!((x.as_slice(), w.as_slice()), (xd.as_slice(), wd.as_slice()));
        for to in FAMILIES {
            let (y, _) = points_weights_dyn(3, to);
            let m = derivative_matrix::<5, 3>(f, to);
            let md = derivative_matrix_dyn(&x, &y);
            for j in 0..3 {
                assert_eq!(m[j].as_slice(), md[j].as_slice());
            }
        }
    }
}

/// Off the nodes `interpolation_row` is `lagrangian_interpolation`; at a node
/// it is the exact unit vector.
#[test]
fn interpolation_row_exact_at_nodes() {
    for f in FAMILIES {
        let (x, _) = points_weights_dyn(6, f);
        let bw = barycentric_weights(&x);
        let mut out = vec![f64::NAN; 6];
        for y in [0.0, 0.123, 0.5, 0.77, 1.0] {
            if x.contains(&y) {
                continue;
            }
            interpolation_row(&x, &bw, y, &mut out);
            assert_eq!(out, lagrangian_interpolation(&x, &bw, y));
        }
        for (k, &xk) in x.iter().enumerate() {
            interpolation_row(&x, &bw, xk, &mut out);
            assert!(out.iter().enumerate().all(|(i, &v)| v == if i == k { 1.0 } else { 0.0 }));
        }
    }
}

/// `CollocationBasis::eval` at the collocation points returns the nodal values
/// exactly (mixed families and counts per direction).
#[test]
fn collocation_eval_exact_at_nodes() {
    let basis = CollocationBasis::new(vec![(4, Gauss::Lobatto), (3, Gauss::Legendre), (5, Gauss::Lobatto)]);
    let u = seeded_vec(basis.n_dof(), 11);
    let vals = basis.eval(basis.points(), &u);
    assert_eq!(vals, u);
}

#[test]
#[should_panic(expected = "at least 2 points")]
fn collocation_basis_lobatto_one_point_panics() {
    let _ = CollocationBasis::new(vec![(1, Gauss::Lobatto)]);
}

#[test]
#[should_panic(expected = "at least 2 points")]
fn points_weights_lobatto_one_point_panics() {
    let _ = points_weights::<1>(Gauss::Lobatto);
}

#[test]
#[should_panic(expected = "at least 1 point")]
fn points_weights_legendre_zero_points_panics() {
    let _ = points_weights::<0>(Gauss::Legendre);
}

/// Interpolation and derivative are exact on polynomials of degree `NI - 1`.
fn check_exactness<const NI: usize, const NO: usize>(from: Gauss, to: Gauss) {
    let mi = interpolation_matrix::<NI, NO>(from, to);
    let md = derivative_matrix::<NI, NO>(from, to);
    let (x, _) = points_weights::<NI>(from);
    let (y, _) = points_weights::<NO>(to);
    let c = seeded_vec(NI, (NI * 31 + NO) as u64);
    let u = x.map(|x| poly(&c, x).0);
    let (vi, vd) = (apply(&mi, &u), apply(&md, &u));
    for j in 0..NO {
        let (p, dp) = poly(&c, y[j]);
        assert!((vi[j] - p).abs() < 1e-13, "interp {NI}->{NO} {from:?}->{to:?}: {} vs {p}", vi[j]);
        assert!((vd[j] - dp).abs() < 1e-11 * (1.0 + dp.abs()), "deriv {NI}->{NO} {from:?}->{to:?}: {} vs {dp}", vd[j]);
    }
}

#[test]
fn interpolation_and_derivative_are_exact() {
    for to in FAMILIES {
        check_exactness::<1, 3>(Gauss::Legendre, to);
    }
    for from in FAMILIES {
        for to in FAMILIES {
            check_exactness::<2, 2>(from, to);
            check_exactness::<3, 5>(from, to);
            check_exactness::<5, 3>(from, to);
            check_exactness::<5, 5>(from, to);
            check_exactness::<7, 4>(from, to);
            check_exactness::<4, 9>(from, to);
            check_exactness::<9, 9>(from, to);
        }
    }
}

fn check_identity<const N: usize>(f: Gauss) {
    let m = interpolation_matrix::<N, N>(f, f);
    for (j, row) in m.iter().enumerate() {
        for (i, &mji) in row.iter().enumerate() {
            assert_eq!(mji, if i == j { 1.0 } else { 0.0 });
        }
    }
}

#[test]
fn same_points_give_exact_identity() {
    check_identity::<1>(Gauss::Legendre);
    for f in FAMILIES {
        check_identity::<2>(f);
        check_identity::<5>(f);
        check_identity::<9>(f);
    }
    // Shared points only: the endpoints of GLL 5 -> GLL 3 and the midpoint.
    let m = interpolation_matrix::<5, 3>(Gauss::Lobatto, Gauss::Lobatto);
    for (j, k) in [(0, 0), (1, 2), (2, 4)] {
        assert_eq!(m[j], core::array::from_fn::<f64, 5, _>(|i| if i == k { 1.0 } else { 0.0 }));
    }
}

fn check_round_trip<const N: usize>() {
    let a = interpolation_matrix::<N, N>(Gauss::Lobatto, Gauss::Legendre);
    let b = interpolation_matrix::<N, N>(Gauss::Legendre, Gauss::Lobatto);
    for (j, bj) in b.iter().enumerate() {
        for i in 0..N {
            let ba: f64 = bj.iter().zip(&a).map(|(bjk, ak)| bjk * ak[i]).sum();
            assert!((ba - if i == j { 1.0 } else { 0.0 }).abs() < 1e-13, "N = {N}: ({j},{i}) = {ba}");
        }
    }
}

#[test]
fn lobatto_legendre_round_trip_is_identity() {
    check_round_trip::<2>();
    check_round_trip::<3>();
    check_round_trip::<5>();
    check_round_trip::<9>();
}

/// `derivative_matrix` and `CollocationBasis::diff_matrix_1d` (both built on
/// `derivative_matrix_dyn`) equal the classical `L_i'(ξ_j)` loop.
fn check_vs_collocation<const NP: usize>() {
    let d = derivative_matrix::<NP, NP>(Gauss::Lobatto, Gauss::Lobatto);
    let basis = CollocationBasis::new(vec![(NP, Gauss::Lobatto)]);
    let dm = basis.diff_matrix_1d(0);
    let (x, _) = gauss_lobatto_on_unit_interval(NP);
    for j in 0..NP {
        for i in 0..NP {
            let classical = lagrangian_derivative_interpolation(&x, x[j], i);
            assert_eq!(d[j][i], classical);
            assert_eq!(dm[(j, i)], classical);
        }
    }
}

#[test]
fn lobatto_derivative_matches_collocation_basis() {
    check_vs_collocation::<2>();
    check_vs_collocation::<4>();
    check_vs_collocation::<5>();
    check_vs_collocation::<9>();
}

// ============================================================================
// Rectangular kernels
// ============================================================================

/// Naive reference on the batched layout: `shape` is the input shape (x
/// fastest), `dir` the swept direction, `N` lanes. Forward maps `NI → NO`
/// along `dir`, transpose `NO → NI`; same summation order as the kernels.
fn naive<const NI: usize, const NO: usize>(
    m: &[[f64; NI]; NO],
    shape: [usize; 3],
    dir: usize,
    transpose: bool,
    n: usize,
    input: &[f64],
) -> Vec<f64> {
    let (g_in, g_out) = if transpose { (NO, NI) } else { (NI, NO) };
    assert_eq!(shape[dir], g_in);
    let mut out_shape = shape;
    out_shape[dir] = g_out;
    let idx = |s: [usize; 3], p: [usize; 3], e: usize| (p[0] + s[0] * (p[1] + s[1] * p[2])) * n + e;
    let mut out = vec![0.0; out_shape.iter().product::<usize>() * n];
    for p2 in 0..out_shape[2] {
        for p1 in 0..out_shape[1] {
            for p0 in 0..out_shape[0] {
                for e in 0..n {
                    let po = [p0, p1, p2];
                    let mut acc = 0.0;
                    // `k` indexes rows or columns of `m` depending on `transpose`.
                    #[allow(clippy::needless_range_loop)]
                    for k in 0..g_in {
                        let mut pi = po;
                        pi[dir] = k;
                        let c = if transpose { m[k][po[dir]] } else { m[po[dir]][k] };
                        acc += c * input[idx(shape, pi, e)];
                    }
                    out[idx(out_shape, po, e)] = acc;
                }
            }
        }
    }
    out
}

/// Runs `kernel` on a NaN-filled output with a sentinel tail, compares it
/// bit for bit with `naive`, and checks the tail is untouched.
fn run<const NI: usize, const NO: usize>(
    m: &[[f64; NI]; NO],
    shape: [usize; 3],
    dir: usize,
    transpose: bool,
    n: usize,
    kernel: impl Fn(&[f64], &mut [f64]),
) {
    let len_in = shape.iter().product::<usize>() * n;
    let input = seeded_vec(len_in, (len_in * 7 + dir) as u64);
    let reference = naive(m, shape, dir, transpose, n, &input);
    let mut output = vec![f64::NAN; reference.len() + 3];
    output[reference.len()..].fill(-7.0);
    kernel(&input, &mut output);
    assert_eq!(&output[reference.len()..], &[-7.0; 3], "wrote past the output, shape {shape:?} dir {dir}");
    for (k, (a, b)) in output.iter().zip(&reference).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "shape {shape:?} dir {dir} T={transpose} N={n}: [{k}] {a} vs {b}");
    }
}

fn mat<const NI: usize, const NO: usize>() -> [[f64; NI]; NO] {
    let v = seeded_vec(NI * NO, (NI * 13 + NO) as u64);
    core::array::from_fn(|j| core::array::from_fn(|i| v[j * NI + i]))
}

fn check_1d<const NI: usize, const NO: usize, const N: usize>() {
    let m = mat::<NI, NO>();
    run(&m, [NI, 1, 1], 0, false, N, |i, o| dx_qn_qm::<NI, NO, N>(&m, i, o));
    run(&m, [NO, 1, 1], 0, true, N, |i, o| dx_t_qm_qn::<NI, NO, N>(&m, i, o));
}

/// `A` is the count of the other direction.
fn check_2d<const NI: usize, const NO: usize, const A: usize, const N: usize>() {
    let m = mat::<NI, NO>();
    run(&m, [NI, A, 1], 0, false, N, |i, o| dx_qnn_qmn::<NI, NO, A, N>(&m, i, o));
    run(&m, [A, NI, 1], 1, false, N, |i, o| dy_qnn_qnm::<NI, NO, A, N>(&m, i, o));
    run(&m, [NO, A, 1], 0, true, N, |i, o| dx_t_qmn_qnn::<NI, NO, A, N>(&m, i, o));
    run(&m, [A, NO, 1], 1, true, N, |i, o| dy_t_qnm_qnn::<NI, NO, A, N>(&m, i, o));
}

/// `A`, `B` are the counts of the other two directions, in increasing order.
fn check_3d<const NI: usize, const NO: usize, const A: usize, const B: usize, const N: usize>() {
    let m = mat::<NI, NO>();
    run(&m, [NI, A, B], 0, false, N, |i, o| dx_qnnn_qmnn::<NI, NO, A, B, N>(&m, i, o));
    run(&m, [A, NI, B], 1, false, N, |i, o| dy_qnnn_qnmn::<NI, NO, A, B, N>(&m, i, o));
    run(&m, [A, B, NI], 2, false, N, |i, o| dz_qnnn_qnnm::<NI, NO, A, B, N>(&m, i, o));
    run(&m, [NO, A, B], 0, true, N, |i, o| dx_t_qmnn_qnnn::<NI, NO, A, B, N>(&m, i, o));
    run(&m, [A, NO, B], 1, true, N, |i, o| dy_t_qnmn_qnnn::<NI, NO, A, B, N>(&m, i, o));
    run(&m, [A, B, NO], 2, true, N, |i, o| dz_t_qnnm_qnnn::<NI, NO, A, B, N>(&m, i, o));
}

macro_rules! for_lanes {
    ($f:ident :: < $($c:literal),* >) => {
        $f::<$($c),*, 1>();
        $f::<$($c),*, 4>();
        $f::<$($c),*, 8>();
    };
}

#[test]
fn rectangular_1d_matches_naive() {
    for_lanes!(check_1d::<5, 3>);
    for_lanes!(check_1d::<3, 5>);
    for_lanes!(check_1d::<1, 4>);
    for_lanes!(check_1d::<4, 4>);
}

#[test]
fn rectangular_2d_matches_naive() {
    for_lanes!(check_2d::<5, 3, 4>);
    for_lanes!(check_2d::<3, 5, 2>);
    for_lanes!(check_2d::<4, 6, 1>);
    for_lanes!(check_2d::<6, 6, 3>);
}

#[test]
fn rectangular_3d_matches_naive() {
    for_lanes!(check_3d::<5, 3, 4, 2>);
    for_lanes!(check_3d::<3, 5, 2, 6>);
    for_lanes!(check_3d::<9, 7, 3, 5>);
    for_lanes!(check_3d::<2, 4, 1, 3>);
}

/// `⟨M x, y⟩ = ⟨x, Mᵀ y⟩` for the 3D kernels, on `(NI, A, B)` / `(NO, A, B)`
/// and the permuted shapes.
fn check_adjoint<const NI: usize, const NO: usize, const A: usize, const B: usize, const N: usize>() {
    let m = mat::<NI, NO>();
    let (li, lo) = (NI * A * B * N, NO * A * B * N);
    let x = seeded_vec(li, 1);
    let y = seeded_vec(lo, 2);
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    type K<'a> = Box<dyn Fn(&[f64], &mut [f64]) + 'a>;
    let pairs: [(K, K); 3] = [
        (Box::new(|i, o| dx_qnnn_qmnn::<NI, NO, A, B, N>(&m, i, o)), Box::new(|i, o| dx_t_qmnn_qnnn::<NI, NO, A, B, N>(&m, i, o))),
        (Box::new(|i, o| dy_qnnn_qnmn::<NI, NO, A, B, N>(&m, i, o)), Box::new(|i, o| dy_t_qnmn_qnnn::<NI, NO, A, B, N>(&m, i, o))),
        (Box::new(|i, o| dz_qnnn_qnnm::<NI, NO, A, B, N>(&m, i, o)), Box::new(|i, o| dz_t_qnnm_qnnn::<NI, NO, A, B, N>(&m, i, o))),
    ];
    for (fwd, tr) in pairs {
        let (mut mx, mut mty) = (vec![0.0; lo], vec![0.0; li]);
        fwd(&x, &mut mx);
        tr(&y, &mut mty);
        let (a, b) = (dot(&mx, &y), dot(&x, &mty));
        assert!((a - b).abs() < 1e-12 * (1.0 + a.abs()), "{a} vs {b}");
    }
}

#[test]
fn rectangular_transpose_is_adjoint() {
    check_adjoint::<5, 3, 4, 2, 4>();
    check_adjoint::<3, 5, 2, 6, 8>();
    check_adjoint::<4, 4, 3, 3, 1>();
}
