#![cfg(feature = "fft")]
//! Tests for equidistant trigonometric directions (`NodeLayout::Equidistant`):
//! the DFT engine, manufactured Fourier solutions, mixed ND solves against
//! Gauss–Lobatto reference solves, and the per-direction Fourier multiplier.

use lobatto_grid::fft::hofft::{Engine, EngineND};
use lobatto_grid::fft::solver::{BoundaryCondition, NodeLayout, Poisson, PoissonND};
use lobatto_grid::grid::nd_fiber_base;
use lobatto_grid::{Extension, Grid1D};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

use BoundaryCondition::{Dirichlet, Periodic};
use NodeLayout::{Equidistant, GaussLobatto};

const L: f64 = 2.7;
const XL: f64 = -0.9;

fn max_err(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max)
}

fn random_complex(n: usize, seed: f64) -> Vec<Complex<f64>> {
    (0..n)
        .map(|p| Complex::new((seed * p as f64 + 0.3).sin(), (1.1 * seed * p as f64).cos()))
        .collect()
}

/// Naive DFT `X_k = Σ_j x_j e^{∓2πi jk/n}`.
fn dft(x: &[Complex<f64>], sign: f64) -> Vec<Complex<f64>> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(j, &v)| v * Complex::from_polar(1.0, sign * 2.0 * PI * ((j * k) % n) as f64 / n as f64))
                .sum()
        })
        .collect()
}

/// The Fourier multiplier `g(μ_m)`, `μ_m = (2π m/L)²`, `m = min(k, n − k)`,
/// applied to one fibre by naive DFTs.
fn fourier_multiplier(x: &[Complex<f64>], l: f64, g: impl Fn(f64) -> f64) -> Vec<Complex<f64>> {
    let n = x.len();
    let mut c = dft(x, -1.0);
    for (k, v) in c.iter_mut().enumerate() {
        *v *= g((2.0 * PI * k.min(n - k) as f64 / l).powi(2));
    }
    dft(&c, 1.0).iter().map(|v| v / n as f64).collect()
}

#[test]
fn engine_is_the_dft_on_the_trigonometric_grid() {
    for n in [8, 9] {
        let e = Engine::trigonometric(n, L, XL);
        assert_eq!(e.grid(), &Grid1D::trigonometric(n, L, XL));
        assert_eq!(e.get_x(), Grid1D::trigonometric(n, L, XL).get_x());
        let x = random_complex(n, 1.3);
        let mut f = x.clone();
        e.forward(&mut f);
        assert!(max_err(&f, &dft(&x, -1.0)) < 1e-13);
        e.inverse(&mut f);
        assert!(max_err(&f, &x) < 1e-14);
    }
    let nd = EngineND::<2>::with_layouts([4, 3], [1, 4], [L, 1.0], [XL, 0.0], [Extension::Periodic, Extension::Odd], [Equidistant, GaussLobatto]);
    assert_eq!(nd.grid().axis(0).layout(), Equidistant);
    assert_eq!(nd.grid().ndofs(), [4, 11]);
}

/// The Fourier modes, Nyquist included, are solved exactly:
/// `(α + β μ_m) u = f`.
#[test]
fn manufactured_fourier_modes_1d() {
    let (alpha, beta) = (0.7, 1.3);
    for (n, r) in [(8, 1), (3, 3), (16, 1), (17, 1)] {
        let s = Poisson::with_layout(L, XL, n, r, Periodic, Equidistant);
        let points = n * r;
        assert_eq!(s.grid().layout(), Equidistant);
        assert_eq!(s.get_x(), Grid1D::trigonometric(points, L, XL).get_x());
        let th: Vec<f64> = s.get_x().iter().map(|&x| 2.0 * PI * (x - XL) / L).collect();
        for m in 0..=points / 2 {
            let mu = (2.0 * PI * m as f64 / L).powi(2);
            let modes: Vec<Box<dyn Fn(f64) -> f64>> = if 2 * m == points {
                vec![Box::new(move |t: f64| (m as f64 * t).cos())]
            } else {
                vec![Box::new(move |t: f64| (m as f64 * t).cos()), Box::new(move |t: f64| (m as f64 * t).sin())]
            };
            for u in modes {
                let exact: Vec<Complex<f64>> = th.iter().map(|&t| Complex::from(u(t))).collect();
                let mut f: Vec<Complex<f64>> = exact.iter().map(|v| v * (alpha + beta * mu)).collect();
                s.solve_in_place(&mut f, alpha, beta);
                assert!(max_err(&f, &exact) < 1e-12, "N={points} m={m}: {:.2e}", max_err(&f, &exact));
            }
        }
        assert!((s.mu_max() - (2.0 * PI * (points / 2) as f64 / L).powi(2)).abs() < 1e-9);
    }
}

/// With `α = 0` the constant mode is projected out.
#[test]
fn constant_mode_annihilated() {
    let s = Poisson::with_layout(L, XL, 12, 1, Periodic, Equidistant);
    let f = random_complex(12, 0.7);
    let (mut u1, mut u2) = (f.clone(), f.iter().map(|v| v + 1.0).collect::<Vec<_>>());
    s.solve_in_place(&mut u1, 0.0, 1.0);
    s.solve_in_place(&mut u2, 0.0, 1.0);
    assert!(max_err(&u1, &u2) < 1e-13);
}

/// A separable right-hand side `e_m(x)·g(y, z)` with `e_m` a Fourier mode of
/// the equidistant direction: the 3D solve is `e_m` times the 2D
/// Gauss–Lobatto solve with `α + β μ_m`.
#[test]
fn mixed_3d_matches_shifted_gauss_lobatto_solve() {
    let (alpha, beta) = (Complex::new(0.4, 0.2), 1.1);
    let s = PoissonND::<3>::with_layouts(
        [L, 1.5, 1.0],
        [XL, 0.2, 0.0],
        [10, 3, 2],
        [1, 3, 4],
        [Periodic, Dirichlet, Periodic],
        [Equidistant, GaussLobatto, GaussLobatto],
    );
    let gl = PoissonND::<2>::new([1.5, 1.0], [0.2, 0.0], [3, 2], [3, 4], [Dirichlet, Periodic]);
    let n = 10;
    let x0 = s.grid().axis(0).get_x();
    let g = random_complex(gl.grid().total(), 0.9);
    for m in [0, 1, 3, 5] {
        let e = |x: f64| Complex::from_polar(1.0, 2.0 * PI * m as f64 * (x - XL) / L);
        let mut f: Vec<Complex<f64>> = (0..s.grid().total()).map(|p| e(x0[p % n]) * g[p / n]).collect();
        s.solve_in_place(&mut f, alpha, beta);
        let mut v = g.clone();
        gl.solve_in_place(&mut v, alpha + beta * (2.0 * PI * m as f64 / L).powi(2), beta);
        let exact: Vec<Complex<f64>> = (0..s.grid().total()).map(|p| e(x0[p % n]) * v[p / n]).collect();
        assert!(max_err(&f, &exact) < 1e-12, "m={m}: {:.2e}", max_err(&f, &exact));
    }
}

/// `apply_mu_in_place_dir` along an equidistant direction (contiguous or
/// strided) is the Fourier multiplier on every fibre, and equals
/// `apply_in_place_dirs` with a `g` that keeps only that direction.
#[test]
fn directional_multiplier_is_fourier_multiplier() {
    let (h, beta) = (0.03, 0.8);
    let step = |mu: f64| 1.0 / (1.0 + h * beta * mu);
    for (layouts, d) in [([Equidistant, GaussLobatto], 0), ([GaussLobatto, Equidistant], 1)] {
        let n_trig = 9;
        let (ns, rs) = if d == 0 { ([n_trig, 2], [1, 3]) } else { ([2, 3], [3, 3]) };
        let ls = if d == 0 { [L, 1.0] } else { [1.0, L] };
        let s = PoissonND::<2>::with_layouts(ls, [0.0; 2], ns, rs, [Periodic, Periodic], layouts);
        let (ndofs, strides, total) = (s.grid().ndofs(), s.grid().strides(), s.grid().total());
        assert_eq!(ndofs[d], n_trig);
        let f = random_complex(total, 1.7);
        let mut u = f.clone();
        s.apply_mu_in_place_dir(d, &mut u, |mu| Complex::from(step(mu)));
        let mut v = f.clone();
        s.apply_in_place_dirs(&mut v, |mu| Complex::from(step(mu[d])));
        assert!(max_err(&u, &v) < 1e-13, "d={d}: dir vs dirs {:.2e}", max_err(&u, &v));
        for q in 0..total / ndofs[d] {
            let base = nd_fiber_base(q, d, &ndofs, &strides);
            let fibre: Vec<Complex<f64>> = (0..ndofs[d]).map(|j| f[base + j * strides[d]]).collect();
            let exact = fourier_multiplier(&fibre, L, step);
            let got: Vec<Complex<f64>> = (0..ndofs[d]).map(|j| u[base + j * strides[d]]).collect();
            assert!(max_err(&got, &exact) < 1e-13, "d={d} fibre {q}: {:.2e}", max_err(&got, &exact));
        }
    }
}

/// Real data with real coefficients give a real solution (the symbol is
/// even in `m`), and the real wrapper agrees with the complex solve.
#[test]
fn real_solution_for_real_data() {
    let s = PoissonND::<2>::with_layouts([L, 1.0], [XL, 0.0], [8, 3], [1, 3], [Periodic, Dirichlet], [Equidistant, GaussLobatto]);
    let f: Vec<f64> = (0..s.grid().total()).map(|p| (0.37 * p as f64).sin()).collect();
    let mut c: Vec<Complex<f64>> = f.iter().map(|&v| Complex::from(v)).collect();
    s.solve_in_place(&mut c, 0.5, 1.0);
    let mut r = f.clone();
    s.solve_in_place_real(&mut r, 0.5, 1.0);
    for (a, &b) in c.iter().zip(&r) {
        assert!(a.im.abs() < 1e-14);
        assert_eq!(a.re, b);
    }
}

#[test]
#[should_panic(expected = "an Equidistant layout must be Periodic")]
fn equidistant_dirichlet_panics() {
    Poisson::with_layout(L, XL, 8, 1, Dirichlet, Equidistant);
}

/// The exact translation `u(x − c)` of one fibre by naive DFTs: phase
/// `e^{−ikc}`, Nyquist by its real part.
fn fourier_shift(x: &[Complex<f64>], l: f64, c: f64) -> Vec<Complex<f64>> {
    let n = x.len();
    let mut a = dft(x, -1.0);
    for (k, v) in a.iter_mut().enumerate() {
        let m = if 2 * k < n { k as f64 } else { k as f64 - n as f64 };
        let phase = Complex::from_polar(1.0, -2.0 * PI * m * c / l);
        *v *= if 2 * k == n { Complex::from(phase.re) } else { phase };
    }
    dft(&a, 1.0).iter().map(|v| v / n as f64).collect()
}

/// A shift by a multiple of the spacing is a cyclic roll, any shift is the
/// naive-DFT translation, and shifting back restores the data.
#[test]
fn shift_1d() {
    for n in [8, 9] {
        let s = Poisson::with_layout(L, XL, n, 1, Periodic, Equidistant);
        let f = random_complex(n, 0.8);
        for j in [1, 3, n + 2] {
            let mut u = f.clone();
            s.shift_in_place(&mut u, j as f64 * L / n as f64);
            let rolled: Vec<Complex<f64>> = (0..n).map(|i| f[(i + n - j % n) % n]).collect();
            assert!(max_err(&u, &rolled) < 1e-13, "n={n} j={j}: {:.2e}", max_err(&u, &rolled));
        }
        for c in [0.11, -0.73, 5.0 * L + 0.2] {
            let mut u = f.clone();
            s.shift_in_place(&mut u, c);
            let exact = fourier_shift(&f, L, c);
            assert!(max_err(&u, &exact) < 1e-13, "n={n} c={c}: {:.2e}", max_err(&u, &exact));
            // Even n: the Nyquist mode loses its sine part, so the round trip
            // is exact only for odd n.
            if n % 2 == 1 {
                s.shift_in_place(&mut u, -c);
                assert!(max_err(&u, &f) < 1e-13, "n={n} c={c} round trip");
            }
        }
    }
}

/// A different shift per fibre, along a contiguous and a strided direction:
/// every fibre is the naive-DFT translation, and for real data the result is
/// `GridND::convect` with `x_d ↦ x_d − c`.
#[test]
fn shift_dir_per_fibre() {
    for (layouts, d) in [([Equidistant, GaussLobatto], 0), ([GaussLobatto, Equidistant], 1)] {
        let (ns, rs) = if d == 0 { ([10, 2], [1, 3]) } else { ([2, 3], [3, 3]) };
        let ls = if d == 0 { [L, 1.0] } else { [1.0, L] };
        let xls = if d == 0 { [XL, 0.0] } else { [0.0, XL] };
        let s = PoissonND::<2>::with_layouts(ls, xls, ns, rs, [Periodic, Periodic], layouts);
        let (ndofs, strides, total) = (s.grid().ndofs(), s.grid().strides(), s.grid().total());
        let fibres = total / ndofs[d];
        let c: Vec<f64> = (0..fibres).map(|q| 0.37 * q as f64 - 0.9).collect();
        let f = random_complex(total, 1.9);
        let mut u = f.clone();
        s.shift_in_place_dir(d, &mut u, &c);
        for (q, &cq) in c.iter().enumerate() {
            let base = nd_fiber_base(q, d, &ndofs, &strides);
            let fibre: Vec<Complex<f64>> = (0..ndofs[d]).map(|j| f[base + j * strides[d]]).collect();
            let exact = fourier_shift(&fibre, L, cq);
            let got: Vec<Complex<f64>> = (0..ndofs[d]).map(|j| u[base + j * strides[d]]).collect();
            assert!(max_err(&got, &exact) < 1e-13, "d={d} fibre {q}: {:.2e}", max_err(&got, &exact));
        }

        // Real data against convect: the shift depends on the other
        // coordinate only, so it is constant along each fibre.
        let o = 1 - d;
        let speed = |y: f64| 0.4 + 0.3 * y;
        let x = s.get_x();
        let cs: Vec<f64> = (0..fibres).map(|q| speed(x[nd_fiber_base(q, d, &ndofs, &strides)][o])).collect();
        let fr: Vec<f64> = (0..total).map(|p| (0.61 * p as f64).sin()).collect();
        let mut ur: Vec<Complex<f64>> = fr.iter().map(|&v| Complex::from(v)).collect();
        s.shift_in_place_dir(d, &mut ur, &cs);
        let mut conv = vec![0.0; total];
        s.grid().convect(
            |p| {
                let mut q = p;
                q[d] -= speed(p[o]);
                q
            },
            &fr,
            &mut conv,
        );
        for (a, &b) in ur.iter().zip(&conv) {
            assert!(a.im.abs() < 1e-14 && (a.re - b).abs() < 1e-13, "d={d}: {a} vs {b}");
        }
    }
}

#[test]
#[should_panic(expected = "direction 1: shift needs an Equidistant layout")]
fn shift_along_gauss_lobatto_panics() {
    let s = PoissonND::<2>::with_layouts([L, 1.0], [XL, 0.0], [8, 2], [1, 3], [Periodic, Periodic], [Equidistant, GaussLobatto]);
    let mut f = vec![Complex::default(); s.grid().total()];
    s.shift_in_place_dir(1, &mut f, &[0.0; 8]);
}
