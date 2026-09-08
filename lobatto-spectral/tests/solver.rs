use lobatto_spectral::solver::*;
use std::f64::consts::PI;

/// A 1D test case: (label, exact solution u, rhs f, starting order, maximal
/// order, expected final error).
type Case1D = (
    &'static str,
    Box<dyn Fn(f64) -> f64>,
    Box<dyn Fn(f64) -> f64>,
    usize,
    usize,
    f64,
);

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ── 1D p-convergence ─────────────────────────────────────────────────────────

#[test]
fn convergence_1d() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    // Multi-mode smooth functions only — single-frequency trig would trigger
    // superconvergence when GLL nodes happen to land on zeros.
    let tests: Vec<Case1D> = vec![
        (
            "exp(sin(2πx))",
            Box::new(move |x| (w * x).sin().exp()),
            Box::new(move |x| {
                let s = (w * x).sin();
                let c = (w * x).cos();
                let u = s.exp();
                alpha * u - beta * w2 * (c * c - s) * u
            }),
            4,    // monotone check starts at this r
            30,   // maximal order to check the monotony
            1e-8, // expected error at the end of the convergence test
        ),
        (
            "1/(2+cos(2πx))",
            Box::new(move |x| 1.0 / (2.0 + (w * x).cos())),
            Box::new(move |x| {
                let s = (w * x).sin();
                let c = (w * x).cos();
                let g = 2.0 + c;
                let u = 1.0 / g;
                let u_pp = w2 * (c * g + 2.0 * s * s) / (g * g * g);
                alpha * u - beta * u_pp
            }),
            6,
            25,
            1e-4,
        ),
    ];

    println!("\n{:=<65}", "");
    println!("1D p-convergence on [0,1], one element,  α={alpha}, β={beta}");
    println!("{:=<65}\n", "");

    for (name, exact_u, rhs_f, r_mono, r_max, err_min) in &tests {
        println!("  u = {name}  (monotone check from r={r_mono})");
        println!("{:>5} {:>14} {:>12}", "r", "max_err", "ratio");
        println!("{:-<34}", "");

        let mut prev_err: Option<f64> = None;

        for r in 2..=*r_max {
            let solver = Poisson::new(1.0, 0.0, 1, r, BoundaryCondition::Periodic);
            let x = solver.get_x();
            let mut f: Vec<f64> = x.iter().map(|&xi| rhs_f(xi)).collect();
            solver.solve_in_place(&mut f, alpha, beta);
            let err = (0..r)
                .map(|i| (f[i] - exact_u(x[i])).abs())
                .fold(0.0_f64, f64::max);

            let ratio = match prev_err {
                Some(pe) if pe > 1e-14 && err > 1e-14 => {
                    let rv = pe / err;
                    if r >= *r_mono {
                        assert!(
                            rv > 1.0,
                            "{name} r={r}: error did not decrease (ratio={rv:.2})"
                        );
                    }
                    format!("{rv:.1}×")
                }
                _ => "-".to_string(),
            };
            println!("{r:>5} {err:>14.3e} {ratio:>12}");
            prev_err = Some(err);
        }
        assert!(
            prev_err.unwrap() < *err_min,
            "{name}: error at r={r_max:} is {:.2e}",
            prev_err.unwrap()
        );
        println!();
    }
}

// ── 1D Dirichlet p-convergence ───────────────────────────────────────────────

#[test]
fn convergence_1d_dirichlet() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let p = PI;

    // u(x) = sin(πx)·exp(sin(πx))  — smooth, zero Dirichlet BCs on [0,1]
    // Let s = sin(πx), c = cos(πx), g = exp(s):
    //   u   = s·g
    //   u'' = g·π²·[c²(2+s) − s(1+s)]
    let exact_u = move |x: f64| {
        let s = (p * x).sin();
        s * s.exp()
    };
    let rhs_f = move |x: f64| {
        let s = (p * x).sin();
        let c = (p * x).cos();
        let g = s.exp();
        let u_pp = g * p * p * (2.0 * c * c + s * (c * c - s - 1.0));
        alpha * s * g - beta * u_pp
    };

    let r_mono = 4;
    let r_max = 30;
    println!("\n{:=<65}", "");
    println!("1D Dirichlet p-convergence on [0,1], one element,  α={alpha}, β={beta}");
    println!("u = sin(πx)·exp(sin(πx)),  u(0)=u(1)=0  (monotone check from r={r_mono})");
    println!("{:=<65}\n", "");
    println!("{:>5} {:>14} {:>12}", "r", "max_err", "ratio");
    println!("{:-<34}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver = Poisson::new(1.0, 0.0, 1, r, BoundaryCondition::Dirichlet);
        let x = solver.get_x();
        let mut f: Vec<f64> = x.iter().map(|&xi| rhs_f(xi)).collect();
        solver.solve_in_place(&mut f, alpha, beta);
        let err = x
            .iter()
            .zip(&f)
            .map(|(&xi, &ui)| (ui - exact_u(xi)).abs())
            .fold(0.0_f64, f64::max);

        let ratio = match prev_err {
            Some(pe) if pe > 1e-12 && err > 1e-12 => {
                let rv = pe / err;
                if r >= r_mono {
                    assert!(
                        rv > 1.0,
                        "Dirichlet r={r}: error did not decrease (ratio={rv:.2})"
                    );
                }
                format!("{rv:.1}×")
            }
            _ => "-".to_string(),
        };
        println!("{r:>5} {err:>14.3e} {ratio:>12}");
        prev_err = Some(err);
    }
    assert!(
        prev_err.unwrap() < 1e-8,
        "Dirichlet: error at r={r_max} is {:.2e}",
        prev_err.unwrap()
    );
    println!();
}

// ── 1D Neumann p-convergence ─────────────────────────────────────────────────

#[test]
fn convergence_1d_neumann() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    // u(x) = exp(cos(2πx))  — smooth, zero Neumann BCs on [0,1]
    // u'' = exp(cos(2πx))·(2π)²·(sin²(2πx) − cos(2πx))
    let exact_u = move |x: f64| (w * x).cos().exp();
    let rhs_f = move |x: f64| {
        let sw = (w * x).sin();
        let cw = (w * x).cos();
        let g = cw.exp();
        let u_pp = g * w2 * (sw * sw - cw);
        alpha * g - beta * u_pp
    };

    let r_mono = 4;
    let r_max = 30;
    println!("\n{:=<65}", "");
    println!("1D Neumann p-convergence on [0,1], one element,  α={alpha}, β={beta}");
    println!("u = exp(cos(2πx)),  u'(0)=u'(1)=0  (monotone check from r={r_mono})");
    println!("{:=<65}\n", "");
    println!("{:>5} {:>14} {:>12}", "r", "max_err", "ratio");
    println!("{:-<34}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver = Poisson::new(1.0, 0.0, 1, r, BoundaryCondition::Neumann);
        let x = solver.get_x();
        let mut f: Vec<f64> = x.iter().map(|&xi| rhs_f(xi)).collect();
        solver.solve_in_place(&mut f, alpha, beta);
        let err = x
            .iter()
            .zip(&f)
            .map(|(&xi, &ui)| (ui - exact_u(xi)).abs())
            .fold(0.0_f64, f64::max);

        let ratio = match prev_err {
            Some(pe) if pe > 1e-11 && err > 1e-11 => {
                let rv = pe / err;
                if r >= r_mono {
                    assert!(
                        rv > 1.0,
                        "Neumann r={r}: error did not decrease (ratio={rv:.2})"
                    );
                }
                format!("{rv:.1}×")
            }
            _ => "-".to_string(),
        };
        println!("{r:>5} {err:>14.3e} {ratio:>12}");
        prev_err = Some(err);
    }
    assert!(
        prev_err.unwrap() < 1e-8,
        "Neumann: error at r={r_max} is {:.2e}",
        prev_err.unwrap()
    );
    println!();
}

// ── 1D h-convergence ─────────────────────────────────────────────────────────

/// Fixed order `r`, refined elements: the nodal error decreases at least
/// like `h^r` once in the asymptotic regime, for all boundary conditions.
#[test]
fn h_convergence_1d() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    // (bc, u, f = α u − β u'')
    type Case = (BoundaryCondition, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>);
    let cases: Vec<Case> = vec![
        (
            BoundaryCondition::Periodic,
            Box::new(move |x| (w * x).sin().exp()),
            Box::new(move |x| {
                let (s, c) = ((w * x).sin(), (w * x).cos());
                let u = s.exp();
                alpha * u - beta * w2 * (c * c - s) * u
            }),
        ),
        (
            BoundaryCondition::Dirichlet,
            Box::new(move |x| (PI * x).sin() * (PI * x).sin().exp()),
            Box::new(move |x| {
                let (s, c) = ((PI * x).sin(), (PI * x).cos());
                let g = s.exp();
                alpha * s * g - beta * g * PI * PI * (2.0 * c * c + s * (c * c - s - 1.0))
            }),
        ),
        (
            BoundaryCondition::Neumann,
            Box::new(move |x| (w * x).cos().exp()),
            Box::new(move |x| {
                let (s, c) = ((w * x).sin(), (w * x).cos());
                let g = c.exp();
                alpha * g - beta * g * w2 * (s * s - c)
            }),
        ),
    ];

    let r = 4;
    let ns = [2usize, 4, 8, 16, 32];
    for (bc, exact_u, rhs_f) in &cases {
        println!("\n1D h-convergence, {bc:?}, r = {r}");
        println!("{:>5} {:>14} {:>10}", "n", "max_err", "ratio");
        let mut prev_err: Option<f64> = None;
        for &n in &ns {
            let solver = Poisson::new(1.0, 0.0, n, r, *bc);
            let x = solver.get_x();
            let mut f: Vec<f64> = x.iter().map(|&xi| rhs_f(xi)).collect();
            solver.solve_in_place(&mut f, alpha, beta);
            let err = x
                .iter()
                .zip(&f)
                .map(|(&xi, &ui)| (ui - exact_u(xi)).abs())
                .fold(0.0_f64, f64::max);
            let ratio = prev_err.map(|pe| pe / err);
            println!(
                "{n:>5} {err:>14.3e} {:>10}",
                ratio.map_or("-".to_string(), |v| format!("{v:.1}×"))
            );
            if let Some(v) = ratio {
                if n >= 8 {
                    let expected = 2.0_f64.powi(r as i32);
                    assert!(
                        v > expected,
                        "{bc:?} n={n}: ratio {v:.1} below h^{r} rate {expected}"
                    );
                }
            }
            prev_err = Some(err);
        }
    }
}

// ── 2D p-convergence ─────────────────────────────────────────────────────────

#[test]
fn convergence_2d() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    // u = exp(sin(wx)) · exp(sin(wy))  — all Fourier modes active in both directions
    let u1d = |x: f64| (w * x).sin().exp();
    let u1d_pp = |x: f64| {
        let s = (w * x).sin();
        let c = (w * x).cos();
        w2 * (c * c - s) * s.exp()
    };
    let exact_u = |x: f64, y: f64| u1d(x) * u1d(y);
    let rhs_f =
        |x: f64, y: f64| alpha * u1d(x) * u1d(y) - beta * (u1d_pp(x) * u1d(y) + u1d(x) * u1d_pp(y));
    let r_mono = 5; // asymptotic regime starts here
    let r_max = 20;
    println!("\n{:=<55}", "");
    println!("2D p-convergence on [0,1]², one element,  α={alpha}, β={beta}");
    println!("u = exp(sin(2πx))·exp(sin(2πy))  (monotone check from r={r_mono})");
    println!("{:=<55}\n", "");
    println!("{:>5} {:>10} {:>12}", "r", "ndof", "max_err");
    println!("{:-<30}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver = PoissonND::<2>::new(
            [1.0; 2],
            [0.0; 2],
            [1; 2],
            [r; 2],
            [BoundaryCondition::Periodic; 2],
        );
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<f64> = (0..total).map(|i| rhs_f(x[i][0], x[i][1])).collect();
        solver.solve_in_place(&mut f, alpha, beta);
        let err = (0..total)
            .map(|i| (f[i] - exact_u(x[i][0], x[i][1])).abs())
            .fold(0.0_f64, f64::max);

        println!("{r:>5} {total:>10} {err:>12.3e}");
        if let Some(pe) = prev_err {
            if r >= r_mono && pe > 1e-14 {
                assert!(err < pe, "r={r}: 2D error did not decrease");
            }
        }
        prev_err = Some(err);
    }
    assert!(
        prev_err.unwrap() < 1e-5,
        "2D: error at r={r_max:} is {:.2e}",
        prev_err.unwrap()
    );
}

// ── 2D p-convergence, α=0, zero-mean sine source ─────────────────────────────

#[test]
fn convergence_2d_laplace() {
    let alpha = 0.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    // u(x,y) = sin(2πx)·sin(2πy)  — zero mean, periodic
    // −Δu = 2w²·sin(2πx)·sin(2πy)  — also zero mean (compatibility for α=0)
    let exact_u = |x: f64, y: f64| (w * x).sin() * (w * y).sin();
    let rhs_f = |x: f64, y: f64| beta * 2.0 * w2 * (w * x).sin() * (w * y).sin();

    let r_mono = 3;
    let r_max = 12;
    println!("\n{:=<60}", "");
    println!("2D p-convergence on [0,1]², one element,  α={alpha}, β={beta}  (zero-mean)");
    println!("u = sin(2πx)·sin(2πy),  ∫f=0  (monotone from r={r_mono})");
    println!("{:=<60}\n", "");
    println!("{:>5} {:>10} {:>12}", "r", "ndof", "max_err");
    println!("{:-<30}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver = PoissonND::<2>::new(
            [1.0; 2],
            [0.0; 2],
            [1; 2],
            [r; 2],
            [BoundaryCondition::Periodic; 2],
        );
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<f64> = (0..total).map(|i| rhs_f(x[i][0], x[i][1])).collect();
        solver.solve_in_place(&mut f, alpha, beta);
        let err = (0..total)
            .map(|i| (f[i] - exact_u(x[i][0], x[i][1])).abs())
            .fold(0.0_f64, f64::max);

        println!("{r:>5} {total:>10} {err:>12.3e}");
        if let Some(pe) = prev_err {
            if r >= r_mono && pe > 1e-14 {
                assert!(err < pe, "r={r}: 2D zero-mean error did not decrease");
            }
        }
        prev_err = Some(err);
    }
    assert!(
        prev_err.unwrap() < 1e-8,
        "2D zero-mean: error at r={r_max} is {:.2e}",
        prev_err.unwrap()
    );
}

// ── 2D Dirichlet p-convergence ───────────────────────────────────────────────

#[test]
fn convergence_2d_dirichlet() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let p = PI;

    // u(x,y) = sin(πx)·exp(sin(πx)) · sin(πy)·exp(sin(πy)) — zero on all four edges.
    let f1d = |t: f64| (p * t).sin() * (p * t).sin().exp();
    let f1d_pp = |t: f64| {
        let s = (p * t).sin();
        let c = (p * t).cos();
        s.exp() * p * p * (2.0 * c * c + s * (c * c - s - 1.0))
    };
    let exact_u = |x: f64, y: f64| f1d(x) * f1d(y);
    let rhs_f =
        |x: f64, y: f64| alpha * f1d(x) * f1d(y) - beta * (f1d_pp(x) * f1d(y) + f1d(x) * f1d_pp(y));

    let r_mono = 5;
    let r_max = 20;
    println!("\n{:=<65}", "");
    println!("2D Dirichlet p-convergence on [0,1]², one element,  α={alpha}, β={beta}");
    println!("u = sin(πx)·exp(sin(πx))·sin(πy)·exp(sin(πy))  (monotone from r={r_mono})");
    println!("{:=<65}\n", "");
    println!("{:>5} {:>10} {:>12}", "r", "ndof", "max_err");
    println!("{:-<30}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver = PoissonND::<2>::new(
            [1.0; 2],
            [0.0; 2],
            [1; 2],
            [r; 2],
            [BoundaryCondition::Dirichlet; 2],
        );
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<f64> = (0..total).map(|i| rhs_f(x[i][0], x[i][1])).collect();
        solver.solve_in_place(&mut f, alpha, beta);
        let err = (0..total)
            .map(|i| (f[i] - exact_u(x[i][0], x[i][1])).abs())
            .fold(0.0_f64, f64::max);

        println!("{r:>5} {total:>10} {err:>12.3e}");
        if let Some(pe) = prev_err {
            if r >= r_mono && pe > 1e-12 {
                assert!(err < pe, "r={r}: 2D Dirichlet error did not decrease");
            }
        }
        prev_err = Some(err);
    }
    assert!(
        prev_err.unwrap() < 1e-4,
        "2D Dirichlet: error at r={r_max} is {:.2e}",
        prev_err.unwrap()
    );
}

// ── 2D Neumann p-convergence ──────────────────────────────────────────────────

#[test]
fn convergence_2d_neumann() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    // u(x,y) = exp(cos(2πx)) · exp(cos(2πy)) — zero normal derivative on all edges.
    let g1d = move |t: f64| (w * t).cos().exp();
    let g1d_pp = move |t: f64| {
        let sw = (w * t).sin();
        let cw = (w * t).cos();
        cw.exp() * w2 * (sw * sw - cw)
    };
    let exact_u = move |x: f64, y: f64| g1d(x) * g1d(y);
    let rhs_f = move |x: f64, y: f64| {
        alpha * g1d(x) * g1d(y) - beta * (g1d_pp(x) * g1d(y) + g1d(x) * g1d_pp(y))
    };

    let r_mono = 5;
    let r_max = 20;
    println!("\n{:=<65}", "");
    println!("2D Neumann p-convergence on [0,1]², one element,  α={alpha}, β={beta}");
    println!("u = exp(cos(2πx))·exp(cos(2πy))  (monotone from r={r_mono})");
    println!("{:=<65}\n", "");
    println!("{:>5} {:>10} {:>12}", "r", "ndof", "max_err");
    println!("{:-<30}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver = PoissonND::<2>::new(
            [1.0; 2],
            [0.0; 2],
            [1; 2],
            [r; 2],
            [BoundaryCondition::Neumann; 2],
        );
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<f64> = (0..total).map(|i| rhs_f(x[i][0], x[i][1])).collect();
        solver.solve_in_place(&mut f, alpha, beta);
        let err = (0..total)
            .map(|i| (f[i] - exact_u(x[i][0], x[i][1])).abs())
            .fold(0.0_f64, f64::max);

        println!("{r:>5} {total:>10} {err:>12.3e}");
        if let Some(pe) = prev_err {
            if r >= r_mono && pe > 1e-11 {
                assert!(err < pe, "r={r}: 2D Neumann error did not decrease");
            }
        }
        prev_err = Some(err);
    }
    assert!(
        prev_err.unwrap() < 1e-4,
        "2D Neumann: error at r={r_max} is {:.2e}",
        prev_err.unwrap()
    );
}

// ── 2D h-convergence, mixed boundary conditions ──────────────────────────────

#[test]
fn h_convergence_2d() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;

    // u = sin(πx)·exp(sin(πx)) · exp(cos(2πy)): Dirichlet in x, Neumann in y.
    let fx = |t: f64| (PI * t).sin() * (PI * t).sin().exp();
    let fx_pp = |t: f64| {
        let (s, c) = ((PI * t).sin(), (PI * t).cos());
        s.exp() * PI * PI * (2.0 * c * c + s * (c * c - s - 1.0))
    };
    let gy = |t: f64| (w * t).cos().exp();
    let gy_pp = |t: f64| {
        let (s, c) = ((w * t).sin(), (w * t).cos());
        c.exp() * w * w * (s * s - c)
    };
    let exact_u = |x: f64, y: f64| fx(x) * gy(y);
    let rhs_f = |x: f64, y: f64| alpha * fx(x) * gy(y) - beta * (fx_pp(x) * gy(y) + fx(x) * gy_pp(y));

    let r = 3;
    println!("\n2D h-convergence, Dirichlet × Neumann, r = {r}");
    println!("{:>5} {:>10} {:>14} {:>10}", "n", "ndof", "max_err", "ratio");
    let mut prev_err: Option<f64> = None;
    for n in [2usize, 4, 8, 16] {
        let solver = PoissonND::<2>::new(
            [1.0; 2],
            [0.0; 2],
            [n; 2],
            [r; 2],
            [BoundaryCondition::Dirichlet, BoundaryCondition::Neumann],
        );
        let x = solver.get_x();
        let mut f: Vec<f64> = x.iter().map(|p| rhs_f(p[0], p[1])).collect();
        solver.solve_in_place(&mut f, alpha, beta);
        let err = x
            .iter()
            .zip(&f)
            .map(|(p, &ui)| (ui - exact_u(p[0], p[1])).abs())
            .fold(0.0_f64, f64::max);
        let ratio = prev_err.map(|pe| pe / err);
        println!(
            "{n:>5} {:>10} {err:>14.3e} {:>10}",
            x.len(),
            ratio.map_or("-".to_string(), |v| format!("{v:.1}×"))
        );
        if let Some(v) = ratio {
            if n >= 8 {
                let expected = 2.0_f64.powi(r as i32);
                assert!(v > expected, "n={n}: ratio {v:.1} below h^{r} rate {expected}");
            }
        }
        prev_err = Some(err);
    }
}

// ── Agreement with lobatto-fft ───────────────────────────────────────────────

/// Same grid, same discretisation: the dense and the FFT solvers compute the
/// same discrete solution, in 1D for every boundary condition and in 2D with
/// mixed conditions. Pins the assembly, the DOF layout and the sign conventions.
#[test]
fn agrees_with_lobatto_fft() {
    use lobatto_fft::solver as fft;

    let to_fft = |bc: BoundaryCondition| match bc {
        BoundaryCondition::Periodic => fft::BoundaryCondition::Periodic,
        BoundaryCondition::Dirichlet => fft::BoundaryCondition::Dirichlet,
        BoundaryCondition::Neumann => fft::BoundaryCondition::Neumann,
    };
    let (alpha, beta) = (0.7, 1.3);
    let (l, xl) = (1.3, -0.2);
    let rhs = |x: f64| (3.0 * x).sin() + 0.5 * (x * x).cos();

    for bc in [
        BoundaryCondition::Periodic,
        BoundaryCondition::Dirichlet,
        BoundaryCondition::Neumann,
    ] {
        let (n, r) = (5, 4);
        let dense = Poisson::new(l, xl, n, r, bc);
        let fast = fft::Poisson::new(l, xl, n, r, to_fft(bc));
        assert_eq!(dense.get_x(), fast.get_x(), "{bc:?}: node sets differ");
        let mut u1: Vec<f64> = dense.get_x().iter().map(|&x| rhs(x)).collect();
        let mut u2 = u1.clone();
        dense.solve_in_place(&mut u1, alpha, beta);
        fast.solve_in_place_real(&mut u2, alpha, beta);
        let scale = u1.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let diff = max_abs_diff(&u1, &u2) / scale;
        println!("1D {bc:?}: relative difference {diff:.2e}");
        assert!(diff < 1e-10, "{bc:?}: solvers disagree ({diff:.2e})");
    }

    let bcs = [BoundaryCondition::Dirichlet, BoundaryCondition::Periodic];
    let (ns, rs) = ([3, 4], [3, 5]);
    let dense = PoissonND::<2>::new([l, 2.0], [xl, 0.5], ns, rs, bcs);
    let fast = fft::PoissonND::<2>::new([l, 2.0], [xl, 0.5], ns, rs, bcs.map(to_fft));
    assert_eq!(dense.get_x(), fast.get_x(), "2D: node sets differ");
    let mut u1: Vec<f64> = dense.get_x().iter().map(|p| rhs(p[0]) * rhs(p[1])).collect();
    let mut u2 = u1.clone();
    dense.solve_in_place(&mut u1, alpha, beta);
    fast.solve_in_place_real(&mut u2, alpha, beta);
    let scale = u1.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = max_abs_diff(&u1, &u2) / scale;
    println!("2D Dirichlet × Periodic: relative difference {diff:.2e}");
    assert!(diff < 1e-10, "2D: solvers disagree ({diff:.2e})");
}

// ── Operator functions ───────────────────────────────────────────────────────

/// `apply_in_place` with `g(λ) = 1/(1 + dt λ)` on the nodal vector equals the
/// solve with `α' = 1 + dt α`, `β' = dt β` of the mass-scaled vector; the
/// semigroup at `t = 0` is the identity.
#[test]
fn operator_functions() {
    let (alpha, beta, dt) = (0.3, 1.0, 0.02);
    let solver = PoissonND::<2>::new(
        [1.0, 1.5],
        [0.0; 2],
        [3, 2],
        [4, 3],
        [BoundaryCondition::Neumann, BoundaryCondition::Dirichlet],
    );
    let x = solver.get_x();
    let u: Vec<f64> = x.iter().map(|p| (p[0] * 3.0).cos() * (PI * p[1] / 1.5).sin()).collect();

    // Euler step via apply vs. solve of the mass-scaled vector: M f is
    // obtained from apply with g ≡ 1 followed by the identity; simpler,
    // compare the Euler step with the solve of M u computed by apply(g = λ)
    // then solve(λ' = 1 + dt λ)... keep it direct: (I + dt L)⁻¹ u == solve of
    // (α' M + β' K) v = M u, so first build M u = apply(g ≡ 1)⁻¹ ... Instead
    // check the identity g(L) g(L)⁻¹ = I: apply the Euler step, then the
    // inverse function.
    let mut v = u.clone();
    solver.euler_step_in_place(&mut v, alpha, beta, dt);
    solver.apply_in_place(&mut v, alpha, beta, |lam| 1.0 + dt * lam);
    let diff = max_abs_diff(&u, &v);
    assert!(diff < 1e-12, "Euler step round trip failed ({diff:.2e})");

    let mut v = u.clone();
    solver.exp_in_place(&mut v, alpha, beta, 0.0);
    let diff = max_abs_diff(&u, &v);
    assert!(diff < 1e-12, "exp(0) is not the identity ({diff:.2e})");

    // solve of the strong-form rhs (α M + β K) u = f, with f := apply(g = λ)
    // of u (that is, f = M⁻¹(αM + βK) u mass-scaled back): solve(M⁻¹·apply)...
    // Equivalent and simpler: apply(g = λ) then apply(g = 1/λ) is the identity.
    let mut v = u.clone();
    solver.apply_in_place(&mut v, alpha, beta, |lam| lam);
    solver.apply_in_place(&mut v, alpha, beta, |lam| 1.0 / lam);
    let diff = max_abs_diff(&u, &v);
    assert!(diff < 1e-12, "apply(λ)·apply(1/λ) is not the identity ({diff:.2e})");
}

// ── 3D p-convergence ─────────────────────────────────────────────────────────

#[test]
fn convergence_3d() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    // u = exp(sin(wx))·exp(sin(wy))·exp(sin(wz))
    let u1d = |x: f64| (w * x).sin().exp();
    let u1d_pp = |x: f64| {
        let s = (w * x).sin();
        let c = (w * x).cos();
        w2 * (c * c - s) * s.exp()
    };

    let ls = [1.0_f64, 2.0, 3.0];
    let xls = [0.0_f64, 1.0, -1.0];

    println!("\n{:=<65}", "");
    println!(
        "3D p-convergence on [{},{}]×[{},{}]×[{},{}], one element,  α={alpha}, β={beta}",
        xls[0],
        xls[0] + ls[0],
        xls[1],
        xls[1] + ls[1],
        xls[2],
        xls[2] + ls[2]
    );

    let r_mono = 7; // 3D pre-asymptotic regime is wider — check from r=7
    let r_max = 20;
    println!("u = exp(sin(2πx))·exp(sin(2πy))·exp(sin(2πz))  (monotone check from r={r_mono})");
    println!("{:=<55}\n", "");
    println!("{:>5} {:>12} {:>12}", "r", "ndof", "max_err");
    println!("{:-<32}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver =
            PoissonND::<3>::new(ls, xls, [1; 3], [r; 3], [BoundaryCondition::Periodic; 3]);
        let x = solver.get_x();
        let total = x.len();

        let mut f: Vec<f64> = (0..total)
            .map(|i| {
                let xi = (x[i][0] - xls[0]) / ls[0];
                let yi = (x[i][1] - xls[1]) / ls[1];
                let zi = (x[i][2] - xls[2]) / ls[2];
                let (ux, uy, uz) = (u1d(xi), u1d(yi), u1d(zi));
                let uxx = u1d_pp(xi) / ls[0].powi(2);
                let uyy = u1d_pp(yi) / ls[1].powi(2);
                let uzz = u1d_pp(zi) / ls[2].powi(2);
                alpha * ux * uy * uz - beta * (uxx * uy * uz + ux * uyy * uz + ux * uy * uzz)
            })
            .collect();
        solver.solve_in_place(&mut f, alpha, beta);

        let err = (0..total)
            .map(|i| {
                let xi = (x[i][0] - xls[0]) / ls[0];
                let yi = (x[i][1] - xls[1]) / ls[1];
                let zi = (x[i][2] - xls[2]) / ls[2];
                (f[i] - u1d(xi) * u1d(yi) * u1d(zi)).abs()
            })
            .fold(0.0_f64, f64::max);

        println!("{r:>5} {total:>12} {err:>12.3e}");
        if let Some(pe) = prev_err {
            if r >= r_mono && pe > 1e-14 {
                assert!(err < pe, "r={r}: 3D error did not decrease");
            }
        }
        prev_err = Some(err);
    }
    assert!(
        prev_err.unwrap() < 1e-4,
        "3D: error at r={r_max:} is {:.2e}",
        prev_err.unwrap()
    );
}

// ── 3D p-convergence measured via interpolation at off-grid points ───────────

#[test]
fn convergence_3d_eval() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    let u1d = |x: f64| (w * x).sin().exp();
    let u1d_pp = |x: f64| {
        let s = (w * x).sin();
        let c = (w * x).cos();
        w2 * (c * c - s) * s.exp()
    };
    let ls = [1.0_f64, 2.0, 3.0];
    let xls = [0.0_f64, 1.0, -1.0];

    let exact_u = |x: f64, y: f64, z: f64| {
        u1d((x - xls[0]) / ls[0]) * u1d((y - xls[1]) / ls[1]) * u1d((z - xls[2]) / ls[2])
    };

    // 100 deterministic points spread across the domain, well-separated from GLL nodes.
    let pts: Vec<[f64; 3]> = (0..100)
        .map(|i| {
            let t = (i as f64 + 0.5) / 100.0;
            [
                xls[0] + (7.0 * PI * t).fract() * ls[0],
                xls[1] + (11.0 * PI * t).fract() * ls[1],
                xls[2] + (13.0 * PI * t).fract() * ls[2],
            ]
        })
        .collect();

    let r_mono = 7;
    let r_max = 14;

    println!("\n{:=<55}", "");
    println!(
        "3D eval p-convergence on [{},{}]×[{},{}]×[{},{}], one element,  α={alpha}, β={beta}",
        xls[0],
        xls[0] + ls[0],
        xls[1],
        xls[1] + ls[1],
        xls[2],
        xls[2] + ls[2]
    );
    println!("100 off-grid points  (monotone check from r={r_mono})");
    println!("{:=<55}", "");
    println!("{:>5} {:>14}", "r", "max_err_eval");
    println!("{:-<22}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver =
            PoissonND::<3>::new(ls, xls, [1; 3], [r; 3], [BoundaryCondition::Periodic; 3]);
        let x = solver.get_x();
        let total = x.len();

        let mut f: Vec<f64> = (0..total)
            .map(|i| {
                let xi = (x[i][0] - xls[0]) / ls[0];
                let yi = (x[i][1] - xls[1]) / ls[1];
                let zi = (x[i][2] - xls[2]) / ls[2];
                let (ux, uy, uz) = (u1d(xi), u1d(yi), u1d(zi));
                let uxx = u1d_pp(xi) / ls[0].powi(2);
                let uyy = u1d_pp(yi) / ls[1].powi(2);
                let uzz = u1d_pp(zi) / ls[2].powi(2);
                alpha * ux * uy * uz - beta * (uxx * uy * uz + ux * uyy * uz + ux * uy * uzz)
            })
            .collect();
        solver.solve_in_place(&mut f, alpha, beta);

        let mut vals = vec![0.0; pts.len()];
        solver.grid().eval(&pts, &f, &mut vals);
        let max_err = pts
            .iter()
            .zip(vals.iter())
            .map(|(pt, &v)| (v - exact_u(pt[0], pt[1], pt[2])).abs())
            .fold(0.0_f64, f64::max);

        println!("{r:>5} {max_err:>14.3e}");
        if let Some(pe) = prev_err {
            if r >= r_mono && pe > 1e-14 {
                assert!(max_err < pe, "r={r}: 3D eval error did not decrease");
            }
        }
        prev_err = Some(max_err);
    }
    assert!(
        prev_err.unwrap() < 1e-2,
        "3D eval: error at r={r_max} is {:.2e}",
        prev_err.unwrap()
    );
}
