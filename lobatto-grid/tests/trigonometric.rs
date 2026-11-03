//! Tests for equidistant trigonometric directions (`Grid1D::trigonometric`):
//! cardinal functions, interpolation against the DFT interpolant, the dense
//! derivative, and mixed ND grids with Gauss–Lobatto directions.

use lobatto_grid::grid::nd_fiber_base;
use lobatto_grid::{Extension, Grid1D, GridND, NodeLayout};
use std::f64::consts::PI;

const L: f64 = 2.7;
const XL: f64 = -0.9;

/// Deterministic pseudo-random numbers in `[0, 1)`.
fn uniform(k: usize) -> f64 {
    (0.5 + k as f64 * 0.618_033_988_749_894_9).fract()
}

/// Real DFT coefficients `(a_k, b_k)`, `k ∈ 0..=n/2`, of nodal data `u`:
/// `c_k = a_k − i b_k = (1/n) Σ_j u_j e^{−2πi jk/n}`.
fn dft(u: &[f64]) -> Vec<(f64, f64)> {
    let n = u.len();
    (0..=n / 2)
        .map(|k| {
            u.iter().enumerate().fold((0.0, 0.0), |(a, b), (j, &uj)| {
                let th = 2.0 * PI * ((j * k) % n) as f64 / n as f64;
                (a + uj * th.cos() / n as f64, b + uj * th.sin() / n as f64)
            })
        })
        .collect()
}

/// The DFT interpolant (Nyquist mode of an even `n` as a cosine) at `x`,
/// and its derivative with the Nyquist mode dropped.
fn dft_interpolant(u: &[f64], x: f64) -> (f64, f64) {
    let n = u.len();
    let c = dft(u);
    let th = 2.0 * PI * (x - XL) / L;
    let (mut p, mut dp) = (c[0].0, 0.0);
    for (k, &(a, b)) in c.iter().enumerate().skip(1) {
        let (s, co) = (k as f64 * th).sin_cos();
        if 2 * k == n {
            p += a * co;
        } else {
            p += 2.0 * (a * co + b * s);
            dp += 2.0 * k as f64 * (b * co - a * s);
        }
    }
    (p, dp * 2.0 * PI / L)
}

fn random_data(n: usize, seed: usize) -> Vec<f64> {
    (0..n).map(|j| 2.0 * uniform(seed + 7 * j) - 1.0).collect()
}

#[test]
fn accessors() {
    let g = Grid1D::trigonometric(10, L, XL);
    assert_eq!(g.layout(), NodeLayout::Equidistant);
    assert_eq!(Grid1D::new(2, 3, L, XL, Extension::Odd).layout(), NodeLayout::GaussLobatto);
    assert_eq!((g.n(), g.r(), g.n_dof(), g.local_count()), (1, 10, 10, 10));
    assert_eq!(g.extension(), Extension::Periodic);
    assert_eq!(g.h(), L);
    for (j, &x) in g.get_x().iter().enumerate() {
        assert!((x - (XL + j as f64 * L / 10.0)).abs() < 1e-15);
        assert_eq!(g.eval_dof_index(0, j), j);
    }
    assert!(g.mass_weights().iter().all(|&w| (w - L / 10.0).abs() < 1e-15));
    assert_eq!(g.fold_extended(XL + L + 0.1).1, 1.0);
}

#[test]
#[should_panic(expected = "at least 2 points")]
fn one_point_panics() {
    Grid1D::trigonometric(1, L, XL);
}

/// `C_j(x_i) = δ_ij`; the weights sum to `L`.
#[test]
fn cardinal_functions_at_nodes() {
    for n in [2, 3, 8, 9, 16, 17] {
        let g = Grid1D::trigonometric(n, L, XL);
        let x = g.get_x();
        for i in 0..n {
            let e: Vec<f64> = (0..n).map(|j| f64::from(u8::from(i == j))).collect();
            for (j, &xj) in x.iter().enumerate() {
                let c = g.eval_at(xj, &e);
                let exact = f64::from(u8::from(i == j));
                assert!((c - exact).abs() < 1e-14, "n={n} C_{i}(x_{j}) = {c}");
            }
            // The right end of the period is node 0.
            assert!((g.eval_at(XL + L, &e) - f64::from(u8::from(i == 0))).abs() < 1e-14);
        }
        let sum: f64 = g.mass_weights().iter().sum();
        assert!((sum - L).abs() < 1e-14, "n={n}: Σw = {sum}");
    }
}

/// Exact for trigonometric polynomials of degree `< n/2`, and for the
/// Nyquist cosine of an even `n`.
#[test]
fn exact_for_trigonometric_polynomials() {
    for n in [4, 7, 8, 15, 16] {
        let g = Grid1D::trigonometric(n, L, XL);
        let coef: Vec<(f64, f64)> = (0..n.div_ceil(2)).map(|k| (uniform(3 * k), uniform(3 * k + 1))).collect();
        let f = |x: f64| {
            let th = 2.0 * PI * (x - XL) / L;
            let mut v: f64 = coef.iter().enumerate().map(|(k, (a, b))| a * (k as f64 * th).cos() + b * (k as f64 * th).sin()).sum();
            if n % 2 == 0 {
                v += 0.7 * (n as f64 * th / 2.0).cos();
            }
            v
        };
        let u: Vec<f64> = g.get_x().iter().map(|&x| f(x)).collect();
        for k in 0..100 {
            let x = XL + L * uniform(k + 1000);
            let e = (g.eval_at(x, &u) - f(x)).abs();
            assert!(e < 1e-13, "n={n} x={x}: error {e:.2e}");
        }
    }
}

/// `eval_at` equals the DFT interpolant at random points, points `1e-12`
/// from a node, and exactly at the nodes.
#[test]
fn matches_dft_interpolant() {
    for n in [8, 9, 64, 65] {
        let g = Grid1D::trigonometric(n, L, XL);
        let u = random_data(n, n);
        let x = g.get_x();
        let mut pts: Vec<f64> = (0..200).map(|k| XL + L * uniform(k + 50)).collect();
        for j in [0, 1, n / 2, n - 1] {
            pts.push(x[j]);
            pts.push((x[j] + 1e-12).min(XL + L));
            pts.push((x[j] - 1e-12).max(XL));
        }
        pts.push(XL + L);
        let mut fu = vec![0.0; pts.len()];
        g.eval(&pts, &u, &mut fu);
        for (&p, &v) in pts.iter().zip(&fu) {
            let (exact, _) = dft_interpolant(&u, p);
            assert!((v - exact).abs() < 1e-13, "n={n} x={p}: {v} vs {exact}");
            assert_eq!(v, g.eval_at(p, &u));
        }
    }
}

/// The derivative is the FFT derivative with the Nyquist mode set to zero,
/// an antisymmetric matrix (to rounding).
#[test]
fn derivative_matches_dft_and_is_antisymmetric() {
    for n in [2, 3, 8, 9, 16, 17, 256, 257] {
        let g = Grid1D::trigonometric(n, L, XL);
        let columns: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let e: Vec<f64> = (0..n).map(|j| f64::from(u8::from(i == j))).collect();
                let mut out = vec![f64::NAN; n];
                g.derivative(&e, &mut out);
                out
            })
            .collect();
        // Entries scale like n/L; the FFT rounding like log n.
        let tol = 1e-15 * n as f64 / L * (n as f64).log2().max(1.0);
        for (i, col) in columns.iter().enumerate() {
            assert!(col[i].abs() < tol, "n={n} diagonal {i}: {:.2e}", col[i]);
            for (j, &v) in col.iter().enumerate() {
                assert!((v + columns[j][i]).abs() < tol, "n={n} ({i},{j}): {:.2e}", v + columns[j][i]);
            }
        }
        let u = random_data(n, 3 * n);
        let mut du = vec![0.0; n];
        g.derivative(&u, &mut du);
        let scale = du.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for (&x, &d) in g.get_x().iter().zip(&du) {
            let (_, exact) = dft_interpolant(&u, x);
            assert!((d - exact).abs() < 1e-13 * scale, "n={n}: {d} vs {exact}");
        }
    }
}

/// `e^{sin θ}` is interpolated and differentiated to rounding at `n = 32`.
#[test]
fn spectral_convergence() {
    let g = Grid1D::trigonometric(32, L, XL);
    let th = |x: f64| 2.0 * PI * (x - XL) / L;
    let u: Vec<f64> = g.get_x().iter().map(|&x| th(x).sin().exp()).collect();
    for k in 0..100 {
        let x = XL + L * uniform(k);
        let e = (g.eval_at(x, &u) - th(x).sin().exp()).abs();
        assert!(e < 1e-13, "interpolation error {e:.2e}");
    }
    let mut du = vec![0.0; 32];
    g.derivative(&u, &mut du);
    for (&x, &d) in g.get_x().iter().zip(&du) {
        let exact = 2.0 * PI / L * th(x).cos() * th(x).sin().exp();
        assert!((d - exact).abs() < 1e-12, "derivative error {:.2e}", (d - exact).abs());
    }
}

fn mixed_2d() -> [GridND<2>; 2] {
    [
        GridND::from_grids([Grid1D::trigonometric(8, L, XL), Grid1D::new(3, 4, 1.5, 0.2, Extension::Odd)]),
        GridND::from_grids([Grid1D::new(2, 3, 1.0, 0.0, Extension::Periodic), Grid1D::trigonometric(9, L, XL)]),
    ]
}

fn mixed_3d() -> GridND<3> {
    GridND::from_grids([
        Grid1D::trigonometric(8, L, XL),
        Grid1D::new(3, 3, 1.5, 0.2, Extension::Odd),
        Grid1D::new(2, 4, 1.0, 0.0, Extension::Periodic),
    ])
}

/// Random points of the box of `grid`.
fn points<const N: usize>(grid: &GridND<N>, m: usize) -> Vec<[f64; N]> {
    (0..m)
        .map(|k| std::array::from_fn(|d| grid.axis(d).xl() + grid.axis(d).l() * uniform(N * k + d)))
        .collect()
}

/// A separable field `Π_d f_d(x_d)` on `grid`, and the tensor product of
/// the 1D interpolants at `x`.
fn separable<const N: usize>(grid: &GridND<N>, x: &[[f64; N]]) -> (Vec<f64>, Vec<f64>) {
    let f = |d: usize, t: f64| (1.3 * t + d as f64).sin() + 0.5;
    let u = grid.get_x().iter().map(|p| (0..N).map(|d| f(d, p[d])).product()).collect();
    let u1d: [Vec<f64>; N] = std::array::from_fn(|d| grid.axis(d).get_x().iter().map(|&t| f(d, t)).collect());
    let exact = x
        .iter()
        .map(|p| (0..N).map(|d| grid.axis(d).eval_at(p[d], &u1d[d])).product())
        .collect();
    (u, exact)
}

/// Mixed ND grids: `eval` is the tensor product of the 1D interpolants and
/// agrees with `eval_with_plan`; `derivative` is the 1D one on every fibre.
fn check_mixed<const N: usize>(grid: &GridND<N>) {
    let x = points(grid, 300);
    let (u, exact) = separable(grid, &x);
    let mut fu = vec![0.0; x.len()];
    grid.eval(&x, &u, &mut fu);
    let mut fp = vec![0.0; x.len()];
    grid.eval_with_plan(&grid.eval_plan(&x), &u, &mut fp);
    for ((&a, &b), &e) in fu.iter().zip(&fp).zip(&exact) {
        assert!((a - e).abs() < 1e-14, "eval {a} vs tensor product {e}");
        assert!((a - b).abs() <= 1e-15, "eval {a} vs plan {b}");
    }

    let (ndofs, strides, total) = (grid.ndofs(), grid.strides(), grid.total());
    let mut du = vec![0.0; total];
    for d in 0..N {
        grid.derivative(d, &u, &mut du);
        for s in 0..total / ndofs[d] {
            let base = nd_fiber_base(s, d, &ndofs, &strides);
            let fibre: Vec<f64> = (0..ndofs[d]).map(|j| u[base + j * strides[d]]).collect();
            let mut d1 = vec![0.0; ndofs[d]];
            grid.axis(d).derivative(&fibre, &mut d1);
            for (j, &v) in d1.iter().enumerate() {
                assert_eq!(du[base + j * strides[d]], v, "d={d} fibre {s} j={j}");
            }
        }
    }
}

#[test]
fn mixed_grids_eval_plan_and_derivative() {
    for g in &mixed_2d() {
        check_mixed(g);
    }
    check_mixed(&mixed_3d());
}

/// `convect_plan` of a constant shift along the trigonometric direction is
/// the exact Fourier translation (Nyquist by its real part).
#[test]
fn constant_shift_is_fourier_translation() {
    let grid = mixed_3d();
    let n = grid.ndofs()[0];
    let u: Vec<f64> = (0..grid.total()).map(|p| (1.7 * p as f64).sin()).collect();
    for c in [0.1, -1.3, 2.0 * L + 0.37] {
        let plan = grid.convect_plan(|[x, y, z]| [x + c, y, z]);
        let mut out = vec![0.0; u.len()];
        grid.eval_with_plan(&plan, &u, &mut out);
        let x0 = grid.axis(0).get_x();
        for q in 0..grid.total() / n {
            let fibre = &u[q * n..(q + 1) * n];
            for (j, &v) in out[q * n..(q + 1) * n].iter().enumerate() {
                let (exact, _) = dft_interpolant(fibre, x0[j] + c);
                assert!((v - exact).abs() < 1e-13, "c={c} fibre {q} j={j}: {v} vs {exact}");
            }
        }
        // Same through the unplanned path.
        let mut direct = vec![0.0; u.len()];
        grid.convect(|[x, y, z]| [x + c, y, z], &u, &mut direct);
        assert_eq!(direct, out);
    }
}
