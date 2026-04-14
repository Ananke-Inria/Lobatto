use nalgebra::Complex;
use lobatto_tensor::hofft::*;
use std::f64::consts::PI;
use std::time::Instant;

/// Solve (alpha - beta * d^2/dx^2) u = f and return the max error against exact_u.
fn compute_error(
    l: f64,
    n: usize,
    r: usize,
    alpha: f64,
    beta: f64,
    exact_u: &dyn Fn(f64) -> f64,
    rhs_f: &dyn Fn(f64) -> f64,
    bc: BoundaryCondition,
) -> f64 {
    let solver = PoissonSolver1D::new(l, 0.0, n, r, alpha, beta, bc);
    let x = solver.get_x();
    let ndof = x.len();

    let mut f: Vec<Complex<f64>> = x.iter().map(|&xi| Complex::new(rhs_f(xi), 0.0)).collect();

    solver.solve_in_place(&mut f);

    let mut max_error = 0.0_f64;
    for i in 0..ndof {
        max_error = max_error.max((f[i].re - exact_u(x[i])).abs());
    }
    max_error
}

#[test]
fn convergence_periodic_1d() {
    let l = 3.8;
    let alpha = 1.0;
    let beta = 1.0;
    let w = 2.0 * PI / l;

    // Each test case: (name, u(x), f(x) = alpha*u - beta*u'')
    let test_functions: Vec<(&str, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>)> = vec![
        // Single Fourier mode
        {
            let c = alpha + beta * w * w;
            (
                "sin(2*pi*x)",
                Box::new(move |x| (w * x).sin()),
                Box::new(move |x| c * (w * x).sin()),
            )
        },
        // Fourier polynomial with multiple modes
        {
            let w2 = w * w;
            (
                "3+2cos(wx)+cos(2wx)-0.5sin(3wx)",
                Box::new(move |x| {
                    3.0 + 2.0 * (w * x).cos() + (2.0 * w * x).cos() - 0.5 * (3.0 * w * x).sin()
                }),
                Box::new(move |x| {
                    alpha
                        * (3.0 + 2.0 * (w * x).cos() + (2.0 * w * x).cos()
                            - 0.5 * (3.0 * w * x).sin())
                        + beta * w2 * 2.0 * (w * x).cos()
                        + beta * 4.0 * w2 * (2.0 * w * x).cos()
                        - beta * 4.5 * w2 * (3.0 * w * x).sin()
                }),
            )
        },
        // Smooth periodic with all Fourier modes active
        // u'' = w^2 * (cos^2(wx) - sin(wx)) * exp(sin(wx))
        {
            (
                "exp(sin(2*pi*x))",
                Box::new(move |x| (w * x).sin().exp()),
                Box::new(move |x| {
                    let s = (w * x).sin();
                    let c = (w * x).cos();
                    let u = s.exp();
                    alpha * u - beta * w * w * (c * c - s) * u
                }),
            )
        },
        // Rational periodic function
        // u = 1/g, g = 2+cos(wx), u'' = w^2*(cos(wx)*g + 2*sin^2(wx)) / g^3
        {
            (
                "1/(2+cos(2*pi*x))",
                Box::new(move |x| 1.0 / (2.0 + (w * x).cos())),
                Box::new(move |x| {
                    let s = (w * x).sin();
                    let c = (w * x).cos();
                    let g = 2.0 + c;
                    let u = 1.0 / g;
                    let u_pp = w * w * (c * g + 2.0 * s * s) / (g * g * g);
                    alpha * u - beta * u_pp
                }),
            )
        },
    ];

    let orders = vec![1, 2, 3, 4];
    let n_values = vec![4, 8, 16, 32, 64, 128, 256];

    println!("\n{:=<80}", "");
    println!("Convergence study: (alpha*M + beta*K) u = f  on [0, {l}]");
    println!("alpha = {alpha}, beta = {beta}");
    println!("{:=<80}\n", "");

    for (name, exact_u, rhs_f) in &test_functions {
        println!("--- Test function: u(x) = {name} ---\n");
        println!(
            "{:<6} {:>10} {:>14} {:>10} {:>14} {:>10}",
            "r", "n", "h", "ndof", "max_error", "rate"
        );
        println!("{:-<68}", "");

        for &r in &orders {
            let mut prev_h: Option<f64> = None;
            let mut prev_err: Option<f64> = None;

            for &n in &n_values {
                let h = l / (n as f64);
                let ndof = n * r;
                let err = compute_error(
                    l,
                    n,
                    r,
                    alpha,
                    beta,
                    exact_u.as_ref(),
                    rhs_f.as_ref(),
                    BoundaryCondition::Periodic,
                );

                let rate = match (prev_h, prev_err) {
                    (Some(ph), Some(pe)) if pe > 0.0 && err > 0.0 => {
                        let rate_val = (pe / err).ln() / (ph / h).ln();
                        assert!(
                            err < 1e-8 || rate_val >= (r as f64 + 1.0) - 1.1,
                            "1D r={r} n={n}: rate {rate_val:.2} < {} but err={err:.2e}",
                            r + 1
                        );
                        format!("{rate_val:.2}")
                    }
                    _ => "-".to_string(),
                };

                println!("{r:<6} {n:>10} {h:>14.2e} {ndof:>10} {err:>14.2e} {rate:>10}");

                prev_h = Some(h);
                prev_err = Some(err);
            }
            println!();
        }
        println!();
    }
}

#[test]
fn convergence_dirichlet_1d() {
    let l = 2.5;
    let alpha = 1.0;
    let beta = 1.0;
    let w = PI / l; // fundamental sine mode on [0, l]

    // Each test function satisfies u(0) = 0 and u(l) = 0.
    let test_functions: Vec<(&str, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>)> = vec![
        // Single Dirichlet eigenfunction: u = sin(πx/l)
        // u'' = -w²·u  →  f = (α + βw²)·u
        {
            let c = alpha + beta * w * w;
            (
                "sin(pi*x/l)",
                Box::new(move |x| (w * x).sin()),
                Box::new(move |x| c * (w * x).sin()),
            )
        },
        // Sine polynomial with multiple Dirichlet modes:
        // u = 2·sin(wx) + sin(2wx) − 0.5·sin(3wx)
        // u'' = −2w²·sin(wx) − 4w²·sin(2wx) + 4.5w²·sin(3wx)
        {
            let w2 = w * w;
            (
                "2*sin(wx)+sin(2wx)-0.5*sin(3wx)",
                Box::new(move |x| {
                    2.0 * (w * x).sin() + (2.0 * w * x).sin() - 0.5 * (3.0 * w * x).sin()
                }),
                Box::new(move |x| {
                    (2.0 * alpha + 2.0 * beta * w2) * (w * x).sin()
                        + (alpha + 4.0 * beta * w2) * (2.0 * w * x).sin()
                        - 0.5 * (alpha + 9.0 * beta * w2) * (3.0 * w * x).sin()
                }),
            )
        },
        // Smooth Dirichlet function: u = exp(sin(wx)) − 1
        // u(0) = u(l) = 0 since sin(0) = sin(π) = 0
        // u'' = w²·(cos²(wx) − sin(wx))·exp(sin(wx))
        {
            (
                "exp(sin(pi*x/l))-1",
                Box::new(move |x| (w * x).sin().exp() - 1.0),
                Box::new(move |x| {
                    let s = (w * x).sin();
                    let c = (w * x).cos();
                    let e = s.exp();
                    alpha * (e - 1.0) - beta * w * w * (c * c - s) * e
                }),
            )
        },
        // Rational Dirichlet function: u = sin(wx) / (2 + cos(2wx))
        // u(0) = u(l) = 0 since sin(0) = sin(π) = 0
        // u'' = w²·[g·(s1·(3c2−2) + 4·s2·c1) + 8·s2²·s1] / g³,  g = 2+cos(2wx)
        {
            (
                "sin(pi*x/l)/(2+cos(2*pi*x/l))",
                Box::new(move |x| {
                    let g = 2.0 + (2.0 * w * x).cos();
                    (w * x).sin() / g
                }),
                Box::new(move |x| {
                    let s1 = (w * x).sin();
                    let c1 = (w * x).cos();
                    let s2 = (2.0 * w * x).sin();
                    let c2 = (2.0 * w * x).cos();
                    let g = 2.0 + c2;
                    let u = s1 / g;
                    let u_pp =
                        w * w * (g * (s1 * (3.0 * c2 - 2.0) + 4.0 * s2 * c1) + 8.0 * s2 * s2 * s1)
                            / (g * g * g);
                    alpha * u - beta * u_pp
                }),
            )
        },
    ];

    let orders = vec![1, 2, 3, 4];
    let n_values = vec![4, 8, 16, 32, 64, 128, 256];

    println!("\n{:=<80}", "");
    println!("Convergence study (Dirichlet): (alpha*M + beta*K) u = f  on [0, {l}]");
    println!("alpha = {alpha}, beta = {beta}");
    println!("{:=<80}\n", "");

    for (name, exact_u, rhs_f) in &test_functions {
        println!("--- Test function: u(x) = {name} ---\n");
        println!(
            "{:<6} {:>10} {:>14} {:>10} {:>14} {:>10}",
            "r", "n", "h", "ndof", "max_error", "rate"
        );
        println!("{:-<68}", "");

        for &r in &orders {
            let mut prev_h: Option<f64> = None;
            let mut prev_err: Option<f64> = None;

            for &n in &n_values {
                let h = l / (n as f64);
                let ndof = n * r - 1;
                let err = compute_error(
                    l,
                    n,
                    r,
                    alpha,
                    beta,
                    exact_u.as_ref(),
                    rhs_f.as_ref(),
                    BoundaryCondition::Dirichlet,
                );

                let rate = match (prev_h, prev_err) {
                    (Some(ph), Some(pe)) if pe > 0.0 && err > 0.0 => {
                        let rate_val = (pe / err).ln() / (ph / h).ln();
                        assert!(
                            err < 1e-8 || rate_val >= (r as f64 + 1.0) - 1.1,
                            "1D Dirichlet r={r} n={n}: rate {rate_val:.2} < {} but err={err:.2e}",
                            r + 1
                        );
                        format!("{rate_val:.2}")
                    }
                    _ => "-".to_string(),
                };

                println!("{r:<6} {n:>10} {h:>14.2e} {ndof:>10} {err:>14.2e} {rate:>10}");

                prev_h = Some(h);
                prev_err = Some(err);
            }
            println!();
        }
        println!();
    }
}

#[test]
fn convergence_neumann_1d() {
    let l = 0.85;
    let alpha = 1.0;
    let beta = 1.0;
    let w = PI / l; // fundamental cosine mode on [0, l]

    // Each test function satisfies u'(0) = 0 and u'(l) = 0.
    let test_functions: Vec<(&str, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>)> = vec![
        // Single Neumann eigenfunction: u = cos(πx/l)
        // u'' = -w²·u  →  f = (α + βw²)·u
        {
            let c = alpha + beta * w * w;
            (
                "cos(pi*x/l)",
                Box::new(move |x| (w * x).cos()),
                Box::new(move |x| c * (w * x).cos()),
            )
        },
        // Cosine polynomial with multiple Neumann modes:
        // u = 2·cos(wx) + cos(2wx) − 0.5·cos(3wx)
        // u'' = −2w²·cos(wx) − 4w²·cos(2wx) + 4.5w²·cos(3wx)
        {
            let w2 = w * w;
            (
                "2*cos(wx)+cos(2wx)-0.5*cos(3wx)",
                Box::new(move |x| {
                    2.0 * (w * x).cos() + (2.0 * w * x).cos() - 0.5 * (3.0 * w * x).cos()
                }),
                Box::new(move |x| {
                    (2.0 * alpha + 2.0 * beta * w2) * (w * x).cos()
                        + (alpha + 4.0 * beta * w2) * (2.0 * w * x).cos()
                        - 0.5 * (alpha + 9.0 * beta * w2) * (3.0 * w * x).cos()
                }),
            )
        },
        // Smooth Neumann function: u = exp(cos(wx))
        // u'(0) = u'(l) = 0 since sin(0) = sin(π) = 0
        // u'' = w²·(sin²(wx) − cos(wx))·exp(cos(wx))
        {
            (
                "exp(cos(pi*x/l))",
                Box::new(move |x| (w * x).cos().exp()),
                Box::new(move |x| {
                    let s = (w * x).sin();
                    let c = (w * x).cos();
                    let e = c.exp();
                    alpha * e - beta * w * w * (s * s - c) * e
                }),
            )
        },
        // Rational Neumann function: u = cos(wx) / (2 + cos(2wx))
        // u'(0) = u'(l) = 0 since sin(0) = sin(π) = 0
        // u'' = w²·[g·(c1·(3c2−2) − 4·s1·s2) + 8·c1·s2²] / g³,  g = 2+cos(2wx)
        {
            (
                "cos(pi*x/l)/(2+cos(2*pi*x/l))",
                Box::new(move |x| {
                    let g = 2.0 + (2.0 * w * x).cos();
                    (w * x).cos() / g
                }),
                Box::new(move |x| {
                    let s1 = (w * x).sin();
                    let c1 = (w * x).cos();
                    let s2 = (2.0 * w * x).sin();
                    let c2 = (2.0 * w * x).cos();
                    let g = 2.0 + c2;
                    let u = c1 / g;
                    let u_pp =
                        w * w * (g * (c1 * (3.0 * c2 - 2.0) - 4.0 * s1 * s2) + 8.0 * c1 * s2 * s2)
                            / (g * g * g);
                    alpha * u - beta * u_pp
                }),
            )
        },
    ];

    let orders = vec![1, 2, 3, 4];
    let n_values = vec![4, 8, 16, 32, 64, 128, 256];

    println!("\n{:=<80}", "");
    println!("Convergence study (Neumann): (alpha*M + beta*K) u = f  on [0, {l}]");
    println!("alpha = {alpha}, beta = {beta}");
    println!("{:=<80}\n", "");

    for (name, exact_u, rhs_f) in &test_functions {
        println!("--- Test function: u(x) = {name} ---\n");
        println!(
            "{:<6} {:>10} {:>14} {:>10} {:>14} {:>10}",
            "r", "n", "h", "ndof", "max_error", "rate"
        );
        println!("{:-<68}", "");

        for &r in &orders {
            let mut prev_h: Option<f64> = None;
            let mut prev_err: Option<f64> = None;

            for &n in &n_values {
                let h = l / (n as f64);
                let ndof = n * r + 1;
                let err = compute_error(
                    l,
                    n,
                    r,
                    alpha,
                    beta,
                    exact_u.as_ref(),
                    rhs_f.as_ref(),
                    BoundaryCondition::Neumann,
                );

                let rate = match (prev_h, prev_err) {
                    (Some(ph), Some(pe)) if pe > 0.0 && err > 0.0 => {
                        let rate_val = (pe / err).ln() / (ph / h).ln();
                        assert!(
                            err < 1e-8 || rate_val >= (r as f64 + 1.0) - 1.1,
                            "1D Neumann r={r} n={n}: rate {rate_val:.2} < {} but err={err:.2e}",
                            r + 1
                        );
                        format!("{rate_val:.2}")
                    }
                    _ => "-".to_string(),
                };

                println!("{r:<6} {n:>10} {h:>14.2e} {ndof:>10} {err:>14.2e} {rate:>10}");

                prev_h = Some(h);
                prev_err = Some(err);
            }
            println!();
        }
        println!();
    }
}

/// 1D Periodic eval convergence: max error over 100 off-grid points via `eval`.
///
/// u(t) = exp(sin(2π·t)),  t = (x−xl)/l,  on [0.7, 2.2].
#[test]
fn convergence_periodic_1d_eval() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let r = 5_usize;
    let n_values = [4, 8, 16, 32, 64, 128];
    let l = 1.5_f64;
    let xl = 0.7_f64;
    let w = 2.0 * PI;
    let u = |t: f64| (w * t).sin().exp();
    let u_pp = |t: f64| {
        let (s, c) = ((w * t).sin(), (w * t).cos());
        w * w * (c * c - s) * s.exp()
    };
    let exact = |x: f64| u((x - xl) / l);
    let pts: Vec<f64> = (0..100)
        .map(|i| xl + (7.0 * PI * (i as f64 + 0.5) / 100.0).fract() * l)
        .collect();

    println!("\n{:=<55}", "");
    println!("1D eval convergence — Periodic (r={r}), 100 off-grid pts");
    println!("domain: [{xl}, {}]", xl + l);
    println!("{:=<55}", "");
    println!("{:>8} {:>14} {:>10}", "n", "max_err_eval", "rate");
    println!("{:-<36}", "");

    let (mut prev_h, mut prev_err) = (None::<f64>, None::<f64>);
    for &n in &n_values {
        let h = l / n as f64;
        let solver = PoissonSolver1D::new(l, xl, n, r, alpha, beta, BoundaryCondition::Periodic);
        let x = solver.get_x();
        let mut f: Vec<Complex<f64>> = x
            .iter()
            .map(|&xi| {
                let t = (xi - xl) / l;
                Complex::new(alpha * u(t) - beta * u_pp(t) / l.powi(2), 0.0)
            })
            .collect();
        solver.solve_in_place(&mut f);

        let max_err = pts
            .iter()
            .map(|&p| (solver.planner.eval(p, &f).re - exact(p)).abs())
            .fold(0.0_f64, f64::max);

        let rate = match (prev_h, prev_err) {
            (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                let rv = (pe / max_err).ln() / (ph / h).ln();
                assert!(
                    max_err < 1e-8 || rv >= r as f64 - 1.0,
                    "1D eval Periodic r={r} n={n}: rate {rv:.2} < {}, err={max_err:.2e}",
                    r + 1
                );
                format!("{rv:.2}")
            }
            _ => "-".to_string(),
        };
        println!("{n:>8} {max_err:>14.2e} {rate:>10}");
        prev_h = Some(h);
        prev_err = Some(max_err);
    }
}

/// 1D Dirichlet eval convergence: max error over 100 off-grid points via `eval`.
///
/// u(t) = exp(sin(π·t)) − 1,  u(0) = u(1) = 0,  on [0.3, 2.8].
#[test]
fn convergence_dirichlet_1d_eval() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let r = 5_usize;
    let n_values = [4, 8, 16, 32, 64, 128];
    let l = 2.5_f64;
    let xl = 0.3_f64;
    let w = PI;
    let u = |t: f64| (w * t).sin().exp() - 1.0;
    let u_pp = |t: f64| {
        let (s, c) = ((w * t).sin(), (w * t).cos());
        w * w * (c * c - s) * s.exp()
    };
    let exact = |x: f64| u((x - xl) / l);
    let pts: Vec<f64> = (0..100)
        .map(|i| xl + (7.0 * PI * (i as f64 + 0.5) / 100.0).fract() * l)
        .collect();

    println!("\n{:=<55}", "");
    println!("1D eval convergence — Dirichlet (r={r}), 100 off-grid pts");
    println!("domain: [{xl}, {}]", xl + l);
    println!("{:=<55}", "");
    println!("{:>8} {:>14} {:>10}", "n", "max_err_eval", "rate");
    println!("{:-<36}", "");

    let (mut prev_h, mut prev_err) = (None::<f64>, None::<f64>);
    for &n in &n_values {
        let h = l / n as f64;
        let solver = PoissonSolver1D::new(l, xl, n, r, alpha, beta, BoundaryCondition::Dirichlet);
        let x = solver.get_x();
        let mut f: Vec<Complex<f64>> = x
            .iter()
            .map(|&xi| {
                let t = (xi - xl) / l;
                Complex::new(alpha * u(t) - beta * u_pp(t) / l.powi(2), 0.0)
            })
            .collect();
        solver.solve_in_place(&mut f);

        let max_err = pts
            .iter()
            .map(|&p| (solver.planner.eval(p, &f).re - exact(p)).abs())
            .fold(0.0_f64, f64::max);

        let rate = match (prev_h, prev_err) {
            (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                let rv = (pe / max_err).ln() / (ph / h).ln();
                assert!(
                    max_err < 1e-8 || rv >= r as f64 - 1.0,
                    "1D eval Dirichlet r={r} n={n}: rate {rv:.2} < {}, err={max_err:.2e}",
                    r + 1
                );
                format!("{rv:.2}")
            }
            _ => "-".to_string(),
        };
        println!("{n:>8} {max_err:>14.2e} {rate:>10}");
        prev_h = Some(h);
        prev_err = Some(max_err);
    }
}

/// 1D Neumann eval convergence: max error over 100 off-grid points via `eval`.
///
/// u(t) = exp(cos(π·t)),  u'(0) = u'(1) = 0,  on [0.1, 0.95].
#[test]
fn convergence_neumann_1d_eval() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let r = 5_usize;
    let n_values = [4, 8, 16, 32, 64, 128];
    let l = 0.85_f64;
    let xl = 0.1_f64;
    let w = PI;
    let u = |t: f64| (w * t).cos().exp();
    let u_pp = |t: f64| {
        let (s, c) = ((w * t).sin(), (w * t).cos());
        w * w * (s * s - c) * c.exp()
    };
    let exact = |x: f64| u((x - xl) / l);
    let pts: Vec<f64> = (0..100)
        .map(|i| xl + (7.0 * PI * (i as f64 + 0.5) / 100.0).fract() * l)
        .collect();

    println!("\n{:=<55}", "");
    println!("1D eval convergence — Neumann (r={r}), 100 off-grid pts");
    println!("domain: [{xl}, {}]", xl + l);
    println!("{:=<55}", "");
    println!("{:>8} {:>14} {:>10}", "n", "max_err_eval", "rate");
    println!("{:-<36}", "");

    let (mut prev_h, mut prev_err) = (None::<f64>, None::<f64>);
    for &n in &n_values {
        let h = l / n as f64;
        let solver = PoissonSolver1D::new(l, xl, n, r, alpha, beta, BoundaryCondition::Neumann);
        let x = solver.get_x();
        let mut f: Vec<Complex<f64>> = x
            .iter()
            .map(|&xi| {
                let t = (xi - xl) / l;
                Complex::new(alpha * u(t) - beta * u_pp(t) / l.powi(2), 0.0)
            })
            .collect();
        solver.solve_in_place(&mut f);

        let max_err = pts
            .iter()
            .map(|&p| (solver.planner.eval(p, &f).re - exact(p)).abs())
            .fold(0.0_f64, f64::max);

        let rate = match (prev_h, prev_err) {
            (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                let rv = (pe / max_err).ln() / (ph / h).ln();
                assert!(
                    max_err < 1e-8 || rv >= r as f64 - 1.0,
                    "1D eval Neumann r={r} n={n}: rate {rv:.2} < {}, err={max_err:.2e}",
                    r + 1
                );
                format!("{rv:.2}")
            }
            _ => "-".to_string(),
        };
        println!("{n:>8} {max_err:>14.2e} {rate:>10}");
        prev_h = Some(h);
        prev_err = Some(max_err);
    }
}

#[test]
fn convergence_2d() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI; // frequency on [0,1]

    // Exact solution: u(x,y) = sin(2πx) * cos(2πy)
    // -Δu = w² * sin(2πx)*cos(2πy) + w² * sin(2πx)*cos(2πy) = 2*w² * u
    // (alpha - beta*Δ) u = (alpha + 2*beta*w²) * u
    let c = alpha + 2.0 * beta * w * w;
    let exact_u = |x: f64, y: f64| (w * x).sin() * (w * y).cos();
    let rhs_f = |x: f64, y: f64| c * (w * x).sin() * (w * y).cos();

    let orders = [2, 3, 4, 5];
    let n_values = [4, 8, 16, 32, 64];

    println!("\n{:=<70}", "");
    println!("2D convergence: (alpha*M + beta*K) u = f  on [0,1]²");
    println!("alpha = {alpha}, beta = {beta}");
    println!("{:=<70}\n", "");
    println!(
        "{:<6} {:>8} {:>14} {:>10} {:>10}",
        "r", "n", "h", "max_err", "rate"
    );
    println!("{:-<52}", "");

    for &r in &orders {
        let mut prev_h: Option<f64> = None;
        let mut prev_err: Option<f64> = None;

        for &n in &n_values {
            let h = 1.0 / n as f64;
            let solver = PoissonSolverND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                alpha,
                beta,
                [BoundaryCondition::Periodic; 2],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f);

            let max_err = (0..total)
                .map(|i| (f[i].re - exact_u(x[i][0], x[i][1])).abs())
                .fold(0.0_f64, f64::max);

            let rate = match (prev_h, prev_err) {
                (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                    let rate_val = (pe / max_err).ln() / (ph / h).ln();
                    assert!(
                        max_err < 1e-8 || rate_val >= (r as f64 + 1.0) - 0.1,
                        "2D r={r} n={n}: rate {rate_val:.2} < {} but err={max_err:.2e}",
                        r + 1
                    );
                    format!("{rate_val:.2}")
                }
                _ => "-".to_string(),
            };

            println!("{r:<6} {n:>8} {h:>14.2e} {max_err:>10.2e} {rate:>10}");
            prev_h = Some(h);
            prev_err = Some(max_err);
        }
        println!();
    }
}

#[test]
fn convergence_2d_dirichlet() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;

    // Exact solution: u(x,y) = sin(πx)·sin(πy) on [0,1]²
    //   Dirichlet in both x and y: u = 0 on all boundaries  ✓
    //   (α − β Δ)u = (α + 2β π²)·u
    let w = PI;
    let c = alpha + 2.0 * beta * w * w;
    let exact_u = |x: f64, y: f64| (w * x).sin() * (w * y).sin();
    let rhs_f = |x: f64, y: f64| c * exact_u(x, y);

    let orders = [2, 3, 4, 5];
    let n_values = [4, 8, 16, 32, 64];

    println!("\n{:=<70}", "");
    println!("2D convergence: Dirichlet x / Dirichlet y  on [0,1]²");
    println!("u = sin(πx)·sin(πy),  α={alpha}, β={beta}");
    println!("{:=<70}\n", "");
    println!(
        "{:<6} {:>8} {:>14} {:>10} {:>10}",
        "r", "n", "h", "max_err", "rate"
    );
    println!("{:-<52}", "");

    for &r in &orders {
        let mut prev_h: Option<f64> = None;
        let mut prev_err: Option<f64> = None;

        for &n in &n_values {
            let h = 1.0 / n as f64;
            let solver = PoissonSolverND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                alpha,
                beta,
                [BoundaryCondition::Dirichlet; 2],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f);

            let max_err = (0..total)
                .map(|i| (f[i].re - exact_u(x[i][0], x[i][1])).abs())
                .fold(0.0_f64, f64::max);

            let rate = match (prev_h, prev_err) {
                (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                    let rv = (pe / max_err).ln() / (ph / h).ln();
                    assert!(
                        max_err < 1e-8 || rv >= (r as f64 + 1.0) - 0.1,
                        "2D Dirichlet r={r} n={n}: rate {rv:.2} < {} but err={max_err:.2e}",
                        r + 1
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };

            println!("{r:<6} {n:>8} {h:>14.2e} {max_err:>10.2e} {rate:>10}");
            prev_h = Some(h);
            prev_err = Some(max_err);
        }
        println!();
    }
}

#[test]
fn convergence_2d_neumann() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;

    // Exact solution: u(x,y) = cos(πx)·cos(πy) on [0,1]²
    //   Neumann in both x and y: u'=0 on all boundaries  ✓
    //   (α − β Δ)u = (α + 2β π²)·u
    let w = PI;
    let c = alpha + 2.0 * beta * w * w;
    let exact_u = |x: f64, y: f64| (w * x).cos() * (w * y).cos();
    let rhs_f = |x: f64, y: f64| c * exact_u(x, y);

    let orders = [2, 3, 4, 5];
    let n_values = [4, 8, 16, 32, 64];

    println!("\n{:=<70}", "");
    println!("2D convergence: Neumann x / Neumann y  on [0,1]²");
    println!("u = cos(πx)·cos(πy),  α={alpha}, β={beta}");
    println!("{:=<70}\n", "");
    println!(
        "{:<6} {:>8} {:>14} {:>10} {:>10}",
        "r", "n", "h", "max_err", "rate"
    );
    println!("{:-<52}", "");

    for &r in &orders {
        let mut prev_h: Option<f64> = None;
        let mut prev_err: Option<f64> = None;

        for &n in &n_values {
            let h = 1.0 / n as f64;
            let solver = PoissonSolverND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                alpha,
                beta,
                [BoundaryCondition::Neumann; 2],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f);

            let max_err = (0..total)
                .map(|i| (f[i].re - exact_u(x[i][0], x[i][1])).abs())
                .fold(0.0_f64, f64::max);

            let rate = match (prev_h, prev_err) {
                (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                    let rv = (pe / max_err).ln() / (ph / h).ln();
                    assert!(
                        max_err < 1e-8 || rv >= (r as f64 + 1.0) - 0.1,
                        "2D Neumann r={r} n={n}: rate {rv:.2} < {} but err={max_err:.2e}",
                        r + 1
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };

            println!("{r:<6} {n:>8} {h:>14.2e} {max_err:>10.2e} {rate:>10}");
            prev_h = Some(h);
            prev_err = Some(max_err);
        }
        println!();
    }
}

#[test]
fn convergence_2d_dirichlet_neumann() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;

    // Exact solution: u(x,y) = sin(πx) · cos(πy)  on [0,1]²
    //   Dirichlet in x: u(0,y) = u(1,y) = 0  ✓
    //   Neumann   in y: ∂u/∂y(x,0) = ∂u/∂y(x,1) = 0  ✓
    //   (α − β Δ)u = (α + 2β π²)·u
    let w = PI;
    let c = alpha + 2.0 * beta * w * w;
    let exact_u = |x: f64, y: f64| (w * x).sin() * (w * y).cos();
    let rhs_f = |x: f64, y: f64| c * exact_u(x, y);

    let orders = [2, 3, 4, 5];
    let n_values = [4, 8, 16, 32, 64];

    println!("\n{:=<70}", "");
    println!("2D convergence: Dirichlet x / Neumann y  on [0,1]²");
    println!("u = sin(πx)·cos(πy),  α={alpha}, β={beta}");
    println!("{:=<70}\n", "");
    println!(
        "{:<6} {:>8} {:>14} {:>10} {:>10}",
        "r", "n", "h", "max_err", "rate"
    );
    println!("{:-<52}", "");

    for &r in &orders {
        let mut prev_h: Option<f64> = None;
        let mut prev_err: Option<f64> = None;

        for &n in &n_values {
            let h = 1.0 / n as f64;
            let solver = PoissonSolverND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                alpha,
                beta,
                [BoundaryCondition::Dirichlet, BoundaryCondition::Neumann],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f);

            let max_err = (0..total)
                .map(|i| (f[i].re - exact_u(x[i][0], x[i][1])).abs())
                .fold(0.0_f64, f64::max);

            let rate = match (prev_h, prev_err) {
                (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                    let rv = (pe / max_err).ln() / (ph / h).ln();
                    assert!(
                        max_err < 1e-8 || rv >= (r as f64 + 1.0) - 0.1,
                        "2D Dirichlet/Neumann r={r} n={n}: rate {rv:.2} < {} but err={max_err:.2e}",
                        r + 1
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };

            println!("{r:<6} {n:>8} {h:>14.2e} {max_err:>10.2e} {rate:>10}");
            prev_h = Some(h);
            prev_err = Some(max_err);
        }
        println!();
    }
}

#[test]
fn convergence_2d_anisotropic() {
    let alpha = 1.0;
    let beta = 1.0;
    let w = 2.0 * PI;

    // u(x,y) = sin(w x)·cos(w y),  (α - β Δ)u = (α + 2β w²)·u
    let c = alpha + 2.0 * beta * w * w;
    let exact_u = |x: f64, y: f64| (w * x).sin() * (w * y).cos();
    let rhs_f = |x: f64, y: f64| c * exact_u(x, y);

    let compute_err = |ns: [usize; 2], rs: [usize; 2]| -> f64 {
        let solver = PoissonSolverND::<2>::new(
            [1.0; 2],
            [0.0; 2],
            ns,
            rs,
            alpha,
            beta,
            [BoundaryCondition::Periodic; 2],
        );
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
            .collect();
        solver.solve_in_place(&mut f);
        (0..total)
            .map(|i| (f[i].re - exact_u(x[i][0], x[i][1])).abs())
            .fold(0.0_f64, f64::max)
    };

    println!("\n{:=<70}", "");
    println!("2D anisotropic convergence on [0,1]²,  α={alpha}, β={beta}");
    println!("{:=<70}\n", "");

    // ── Part 1: anisotropic order (rx ≠ ry), uniform mesh refinement ──────────
    // Convergence rate limited by the coarser direction: min(rx,ry)+1
    println!("-- Anisotropic order, uniform n --");
    println!(
        "{:<8} {:>5} {:>12} {:>10} {:>8}",
        "(rx,ry)", "n", "h", "max_err", "rate"
    );
    println!("{:-<48}", "");

    // (rs, expected_rate = min(rx,ry)+1)
    let aniso_orders: [([usize; 2], usize); 4] =
        [([2, 4], 3), ([4, 2], 3), ([1, 3], 2), ([3, 1], 2)];
    let n_values = [4usize, 8, 16, 32, 64];

    for ([rx, ry], expected_rate) in aniso_orders {
        let mut prev = None::<(f64, f64)>;
        for &n in &n_values {
            let h = 1.0 / n as f64;
            let err = compute_err([n; 2], [rx, ry]);
            let rate = match prev {
                Some((ph, pe)) if pe > 0.0 && err > 0.0 => {
                    let rv = (pe / err).ln() / (ph / h).ln();
                    assert!(
                        err < 1e-8 || rv >= expected_rate as f64 - 0.1,
                        "rx={rx} ry={ry} n={n}: rate {rv:.2} < {expected_rate}, err={err:.2e}"
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };
            println!("({rx},{ry})   {n:>5} {h:>12.2e} {err:>10.2e} {rate:>8}");
            prev = Some((h, err));
        }
        println!();
    }

    // ── Part 2: anisotropic mesh (nx ≠ ny), r=3 fixed ─────────────────────────
    // Refine one direction while the other is fixed fine — rate must be r+1.
    println!("-- Anisotropic mesh (r=3), refining one direction at a time --");
    let r = 3;
    let n_fixed = 256; // fine enough to be negligible
    let n_coarse = [4, 8, 16, 32]; // stop before error saturates due to the fixed direction

    for (is_x_refined, label) in [(true, "refine x (ny=32)"), (false, "refine y (nx=32)")] {
        println!("\n  {label}:");
        println!("{:>6} {:>12} {:>10} {:>8}", "n", "h", "max_err", "rate");
        println!("{:-<40}", "");
        let mut prev = None::<(f64, f64)>;
        for &n in &n_coarse {
            let h = 1.0 / n as f64;
            let ns = if is_x_refined {
                [n, n_fixed]
            } else {
                [n_fixed, n]
            };
            let err = compute_err(ns, [r; 2]);
            let rate = match prev {
                Some((ph, pe)) if pe > 0.0 && err > 0.0 => {
                    let rv = (pe / err).ln() / (ph / h).ln();
                    assert!(
                        err < 1e-8 || rv >= r as f64 + 1.0 - 0.1,
                        "{label} n={n}: rate {rv:.2} < {}, err={err:.2e}",
                        r + 1
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };
            println!("{n:>6} {h:>12.2e} {err:>10.2e} {rate:>8}");
            prev = Some((h, err));
        }
    }
}

#[test]
fn convergence_3d() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    // Each entry: (name, u1d(x), u1d''(x))
    // Exact solution: u(x,y,z) = u1d(x)*u1d(y)*u1d(z)
    // RHS: alpha*u - beta*(u1d''(x)*u1d(y)*u1d(z) + u1d(x)*u1d''(y)*u1d(z) + u1d(x)*u1d(y)*u1d''(z))
    let test_functions: Vec<(&str, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>)> = vec![
        // Fourier polynomial with multiple modes
        (
            "fourier_poly",
            Box::new(move |x| {
                3.0 + 2.0 * (w * x).cos() + (2.0 * w * x).cos() - 0.5 * (3.0 * w * x).sin()
            }),
            Box::new(move |x| {
                -2.0 * w2 * (w * x).cos() - 4.0 * w2 * (2.0 * w * x).cos()
                    + 4.5 * w2 * (3.0 * w * x).sin()
            }),
        ),
        // Smooth periodic: exp(sin(wx))
        (
            "exp(sin(2π . ))",
            Box::new(move |x| (w * x).sin().exp()),
            Box::new(move |x| {
                let s = (w * x).sin();
                let c = (w * x).cos();
                w2 * (c * c - s) * s.exp()
            }),
        ),
        // Rational periodic: 1/(2+cos(wx))
        (
            "1/(2+cos(2π . ))",
            Box::new(move |x| 1.0 / (2.0 + (w * x).cos())),
            Box::new(move |x| {
                let s = (w * x).sin();
                let c = (w * x).cos();
                let g = 2.0 + c;
                w2 * (c * g + 2.0 * s * s) / (g * g * g)
            }),
        ),
    ];

    let orders = [2, 3, 4];
    let n_values = [4, 8, 16, 32, 64];

    let ls = [1.0_f64, 2.0, 3.0];
    let xls = [0.0_f64, 1.0, -1.0];

    println!("\n{:=<70}", "");
    println!(
        "3D convergence: (alpha*M + beta*K) u = f  on [{},{}]×[{},{}]×[{},{}]",
        xls[0],
        xls[0] + ls[0],
        xls[1],
        xls[1] + ls[1],
        xls[2],
        xls[2] + ls[2]
    );
    println!("alpha = {alpha}, beta = {beta}");
    println!("{:=<70}\n", "");

    for (name, u1d, u1d_pp) in &test_functions {
        println!("--- u(x,y,z) = {name}(x) * {name}(y) * {name}(z) ---\n");
        println!(
            "{:<6} {:>8} {:>14} {:>10} {:>10}",
            "r", "n", "h", "max_err", "rate"
        );
        println!("{:-<52}", "");

        for &r in &orders {
            let mut prev_h: Option<f64> = None;
            let mut prev_err: Option<f64> = None;

            for &n in &n_values {
                let h = 1.0 / n as f64;
                let solver = PoissonSolverND::<3>::new(
                    ls,
                    xls,
                    [n; 3],
                    [r; 3],
                    alpha,
                    beta,
                    [BoundaryCondition::Periodic; 3],
                );
                let x = solver.get_x();
                let total = x.len();

                let mut f: Vec<Complex<f64>> = (0..total)
                    .map(|i| {
                        let xi = (x[i][0] - xls[0]) / ls[0];
                        let yi = (x[i][1] - xls[1]) / ls[1];
                        let zi = (x[i][2] - xls[2]) / ls[2];
                        let (ux, uy, uz) = (u1d(xi), u1d(yi), u1d(zi));
                        let uxx = u1d_pp(xi) / ls[0].powi(2);
                        let uyy = u1d_pp(yi) / ls[1].powi(2);
                        let uzz = u1d_pp(zi) / ls[2].powi(2);
                        Complex::new(
                            alpha * ux * uy * uz
                                - beta * (uxx * uy * uz + ux * uyy * uz + ux * uy * uzz),
                            0.0,
                        )
                    })
                    .collect();
                solver.solve_in_place(&mut f);

                let max_err = (0..total)
                    .map(|i| {
                        let xi = (x[i][0] - xls[0]) / ls[0];
                        let yi = (x[i][1] - xls[1]) / ls[1];
                        let zi = (x[i][2] - xls[2]) / ls[2];
                        (f[i].re - u1d(xi) * u1d(yi) * u1d(zi)).abs()
                    })
                    .fold(0.0_f64, f64::max);

                let rate = match (prev_h, prev_err) {
                    (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                        let rate_val = (pe / max_err).ln() / (ph / h).ln();
                        assert!(
                            max_err < 1e-8 || rate_val >= (r as f64 + 1.0) - 0.1,
                            "3D r={r} n={n}: rate {rate_val:.2} < {} but err={max_err:.2e}",
                            r + 1
                        );
                        format!("{rate_val:.2}")
                    }
                    _ => "-".to_string(),
                };

                println!("{r:<6} {n:>8} {h:>14.2e} {max_err:>10.2e} {rate:>10}");
                prev_h = Some(h);
                prev_err = Some(max_err);
            }
            println!();
        }
        println!();
    }
}

#[test]
fn convergence_3d_dirichlet() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;

    // Anisotropic domain: [0.2, 1.7] × [0.5, 3.0] × [-1.0, 1.0]
    let ls = [1.5_f64, 2.5, 2.0];
    let xls = [0.2_f64, 0.5, -1.0];

    // Exact: u = sin(πξ)·sin(πη)·sin(πζ),  ξ_d = (x_d − xl_d) / l_d
    // Dirichlet in all directions: u = 0 on boundaries  ✓
    // (α − β Δ)u = (α + β π²·Σ 1/l_d²)·u
    let c = alpha + beta * PI * PI * (1.0 / (ls[0] * ls[0]) + 1.0 / (ls[1] * ls[1]) + 1.0 / (ls[2] * ls[2]));

    let exact_u = |x: [f64; 3]| {
        (0..3)
            .map(|d| (PI * (x[d] - xls[d]) / ls[d]).sin())
            .product::<f64>()
    };
    let rhs_f = |x: [f64; 3]| c * exact_u(x);

    let orders = [2, 3, 4, 5];
    let n_values = [4, 8, 16, 32, 64];

    println!("\n{:=<70}", "");
    println!(
        "3D Dirichlet: on [{:.1},{:.1}]×[{:.1},{:.1}]×[{:.1},{:.1}]",
        xls[0], xls[0]+ls[0], xls[1], xls[1]+ls[1], xls[2], xls[2]+ls[2]
    );
    println!("u = sin(πξ)·sin(πη)·sin(πζ),  α={alpha}, β={beta}");
    println!("{:=<70}\n", "");
    println!(
        "{:<6} {:>8} {:>14} {:>10} {:>10}",
        "r", "n", "h", "max_err", "rate"
    );
    println!("{:-<52}", "");

    for &r in &orders {
        let mut prev_h: Option<f64> = None;
        let mut prev_err: Option<f64> = None;
        for &n in &n_values {
            let h = 1.0 / n as f64;
            let solver = PoissonSolverND::<3>::new(
                ls, xls, [n; 3], [r; 3], alpha, beta,
                [BoundaryCondition::Dirichlet; 3],
            );
            let x = solver.get_x();
            let total = x.len();
            let mut f: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(rhs_f(x[i]), 0.0)).collect();
            solver.solve_in_place(&mut f);
            let max_err = (0..total)
                .map(|i| (f[i].re - exact_u(x[i])).abs())
                .fold(0.0_f64, f64::max);

            let rate = match (prev_h, prev_err) {
                (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                    let rv = (pe / max_err).ln() / (ph / h).ln();
                    assert!(
                        max_err < 1e-8 || rv >= (r as f64 + 1.0) - 0.1,
                        "3D Dirichlet r={r} n={n}: rate {rv:.2} < {}, err={max_err:.2e}",
                        r + 1
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };
            println!("{r:<6} {n:>8} {h:>14.2e} {max_err:>10.2e} {rate:>10}");
            prev_h = Some(h);
            prev_err = Some(max_err);
        }
        println!();
    }
}

#[test]
fn convergence_3d_neumann() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;

    // Anisotropic domain: [0.2, 1.7] × [0.5, 3.0] × [-1.0, 1.0]
    let ls = [1.5_f64, 2.5, 2.0];
    let xls = [0.2_f64, 0.5, -1.0];

    // Exact: u = cos(πξ)·cos(πη)·cos(πζ),  ξ_d = (x_d − xl_d) / l_d
    // Neumann in all directions: u' = 0 on boundaries  ✓
    // (α − β Δ)u = (α + β π²·Σ 1/l_d²)·u
    let c = alpha + beta * PI * PI * (1.0 / (ls[0] * ls[0]) + 1.0 / (ls[1] * ls[1]) + 1.0 / (ls[2] * ls[2]));

    let exact_u = |x: [f64; 3]| {
        (0..3)
            .map(|d| (PI * (x[d] - xls[d]) / ls[d]).cos())
            .product::<f64>()
    };
    let rhs_f = |x: [f64; 3]| c * exact_u(x);

    let orders = [2, 3, 4, 5];
    let n_values = [4, 8, 16, 32, 64];

    println!("\n{:=<70}", "");
    println!(
        "3D Neumann: on [{:.1},{:.1}]×[{:.1},{:.1}]×[{:.1},{:.1}]",
        xls[0], xls[0]+ls[0], xls[1], xls[1]+ls[1], xls[2], xls[2]+ls[2]
    );
    println!("u = cos(πξ)·cos(πη)·cos(πζ),  α={alpha}, β={beta}");
    println!("{:=<70}\n", "");
    println!(
        "{:<6} {:>8} {:>14} {:>10} {:>10}",
        "r", "n", "h", "max_err", "rate"
    );
    println!("{:-<52}", "");

    for &r in &orders {
        let mut prev_h: Option<f64> = None;
        let mut prev_err: Option<f64> = None;
        for &n in &n_values {
            let h = 1.0 / n as f64;
            let solver = PoissonSolverND::<3>::new(
                ls, xls, [n; 3], [r; 3], alpha, beta,
                [BoundaryCondition::Neumann; 3],
            );
            let x = solver.get_x();
            let total = x.len();
            let mut f: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(rhs_f(x[i]), 0.0)).collect();
            solver.solve_in_place(&mut f);
            let max_err = (0..total)
                .map(|i| (f[i].re - exact_u(x[i])).abs())
                .fold(0.0_f64, f64::max);

            let rate = match (prev_h, prev_err) {
                (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                    let rv = (pe / max_err).ln() / (ph / h).ln();
                    assert!(
                        max_err < 1e-8 || rv >= (r as f64 + 1.0) - 0.1,
                        "3D Neumann r={r} n={n}: rate {rv:.2} < {}, err={max_err:.2e}",
                        r + 1
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };
            println!("{r:<6} {n:>8} {h:>14.2e} {max_err:>10.2e} {rate:>10}");
            prev_h = Some(h);
            prev_err = Some(max_err);
        }
        println!();
    }
}

#[test]
fn convergence_3d_mixed_bc() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;

    // Anisotropic domain: [0.2, 1.7] × [0.5, 3.0] × [-1.0, 1.0]
    let ls = [1.5_f64, 2.5, 2.0];
    let xls = [0.2_f64, 0.5, -1.0];

    // Exact solution using normalised coordinates ξ = (x−xl)/l:
    //   x: Dirichlet  → sin(πξ)         vanishes at 0 and 1
    //   y: Neumann    → cos(πη)         zero derivative at 0 and 1
    //   z: Periodic   → sin(2πζ)        periodic on [−1, 1]
    //
    // (α − β Δ)u = α·u − β·(u_xx + u_yy + u_zz)
    //            = α·u + β·(π/lx)²·u + β·(π/ly)²·u + β·(2π/lz)²·u
    //            = (α + β·π²·(1/lx² + 1/ly² + (2/lz)²))·u
    let wx = PI / ls[0];
    let wy = PI / ls[1];
    let wz = 2.0 * PI / ls[2];
    let c = alpha + beta * (wx * wx + wy * wy + wz * wz);

    let u1d = [
        |xi: f64| (PI * xi).sin(),
        |xi: f64| (PI * xi).cos(),
        |xi: f64| (2.0 * PI * xi).sin(),
    ];
    let u1d_pp = [
        |xi: f64| -(PI * PI) * (PI * xi).sin(),
        |xi: f64| -(PI * PI) * (PI * xi).cos(),
        |xi: f64| -(4.0 * PI * PI) * (2.0 * PI * xi).sin(),
    ];

    let exact_u = |x: [f64; 3]| {
        (0..3)
            .map(|d| u1d[d]((x[d] - xls[d]) / ls[d]))
            .product::<f64>()
    };
    let rhs_f = |x: [f64; 3]| {
        let xi: [f64; 3] = std::array::from_fn(|d| (x[d] - xls[d]) / ls[d]);
        let u: [f64; 3] = std::array::from_fn(|d| u1d[d](xi[d]));
        let uxx: [f64; 3] = std::array::from_fn(|d| u1d_pp[d](xi[d]) / ls[d].powi(2));
        let prod_u: f64 = u.iter().product();
        alpha * prod_u
            - beta
                * (0..3)
                    .map(|d| uxx[d] * (0..3).filter(|&e| e != d).map(|e| u[e]).product::<f64>())
                    .sum::<f64>()
    };

    let bc = [
        BoundaryCondition::Dirichlet,
        BoundaryCondition::Neumann,
        BoundaryCondition::Periodic,
    ];

    let compute_err = |ns: [usize; 3], rs: [usize; 3]| -> f64 {
        let solver = PoissonSolverND::<3>::new(ls, xls, ns, rs, alpha, beta, bc);
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(rhs_f(x[i]), 0.0)).collect();
        solver.solve_in_place(&mut f);
        (0..total)
            .map(|i| (f[i].re - exact_u(x[i])).abs())
            .fold(0.0_f64, f64::max)
    };

    println!("\n{:=<70}", "");
    println!(
        "3D mixed BC: Dirichlet x / Neumann y / Periodic z  on [{:.1},{:.1}]×[{:.1},{:.1}]×[{:.1},{:.1}]",
        xls[0], xls[0]+ls[0], xls[1], xls[1]+ls[1], xls[2], xls[2]+ls[2]
    );
    println!("u = sin(πξ)·cos(πη)·sin(2πζ),  α={alpha}, β={beta}");
    println!("c = {c:.6}  (exact eigenvalue check)");
    println!("{:=<70}\n", "");

    // ── Part 1: isotropic refinement, various orders ───────────────────────────
    let orders = [2, 3, 4, 5];
    let n_values = [4, 8, 16, 32, 64];

    println!("-- Isotropic refinement --");
    println!(
        "{:<6} {:>8} {:>14} {:>10} {:>10}",
        "r", "n", "h", "max_err", "rate"
    );
    println!("{:-<52}", "");

    for &r in &orders {
        let mut prev_h: Option<f64> = None;
        let mut prev_err: Option<f64> = None;
        for &n in &n_values {
            let h = 1.0 / n as f64;
            let err = compute_err([n; 3], [r; 3]);
            let rate = match (prev_h, prev_err) {
                (Some(ph), Some(pe)) if pe > 0.0 && err > 0.0 => {
                    let rv = (pe / err).ln() / (ph / h).ln();
                    assert!(
                        err < 1e-8 || rv >= (r as f64 + 1.0) - 0.1,
                        "3D mixed r={r} n={n}: rate {rv:.2} < {}, err={err:.2e}",
                        r + 1
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };
            println!("{r:<6} {n:>8} {h:>14.2e} {err:>10.2e} {rate:>10}");
            prev_h = Some(h);
            prev_err = Some(err);
        }
        println!();
    }

    // ── Part 2: anisotropic order (rx ≠ ry ≠ rz) ─────────────────────────────
    // Rate limited by min(r_d)+1.
    println!("\n-- Anisotropic order, uniform n --");
    println!(
        "{:<14} {:>5} {:>12} {:>10} {:>8}",
        "(rx,ry,rz)", "n", "h", "max_err", "rate"
    );
    println!("{:-<54}", "");

    let aniso_orders: [([usize; 3], usize); 3] = [([2, 4, 3], 3), ([4, 2, 3], 3), ([3, 3, 2], 3)];

    for ([rx, ry, rz], expected_rate) in aniso_orders {
        let mut prev = None::<(f64, f64)>;
        for &n in &n_values {
            let h = 1.0 / n as f64;
            let err = compute_err([n; 3], [rx, ry, rz]);
            let rate = match prev {
                Some((ph, pe)) if pe > 0.0 && err > 0.0 => {
                    let rv = (pe / err).ln() / (ph / h).ln();
                    assert!(
                        err < 1e-8 || rv >= expected_rate as f64 - 0.1,
                        "({rx},{ry},{rz}) n={n}: rate {rv:.2} < {expected_rate}, err={err:.2e}"
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };
            println!("({rx},{ry},{rz})   {n:>5} {h:>12.2e} {err:>10.2e} {rate:>8}");
            prev = Some((h, err));
        }
        println!();
    }

    // ── Part 3: anisotropic mesh, r=3, refine one direction at a time ─────────
    println!("\n-- Anisotropic mesh (r=3), refining one direction at a time --");
    let r = 3;
    let n_fine = 64;
    let n_coarse = [4, 8, 16, 32];

    for (d_refine, label) in [(0, "refine x"), (1, "refine y"), (2, "refine z")] {
        println!("\n  {label} (other dirs n={n_fine}):");
        println!("{:>6} {:>12} {:>10} {:>8}", "n", "h", "max_err", "rate");
        println!("{:-<40}", "");
        let mut prev = None::<(f64, f64)>;
        for &n in &n_coarse {
            let h = 1.0 / n as f64;
            let ns = std::array::from_fn(|d| if d == d_refine { n } else { n_fine });
            let err = compute_err(ns, [r; 3]);
            let rate = match prev {
                Some((ph, pe)) if pe > 0.0 && err > 0.0 => {
                    let rv = (pe / err).ln() / (ph / h).ln();
                    assert!(
                        err < 1e-8 || rv >= r as f64 + 1.0 - 0.1,
                        "{label} n={n}: rate {rv:.2} < {}, err={err:.2e}",
                        r + 1
                    );
                    format!("{rv:.2}")
                }
                _ => "-".to_string(),
            };
            println!("{n:>6} {h:>12.2e} {err:>10.2e} {rate:>8}");
            prev = Some((h, err));
        }
    }
}

#[test]
fn convergence_periodic_2d() {
    let alpha = 0.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;

    // u(x,y) = sin(2πx)·sin(2πy)  — zero mean, periodic
    // −Δu = 2w²·sin(2πx)·sin(2πy)  — also zero mean (compatibility for α=0)
    let exact_u = |x: f64, y: f64| (w * x).sin() * (w * y).sin();
    let rhs_f = |x: f64, y: f64| beta * 2.0 * w * w * (w * x).sin() * (w * y).sin();

    let orders = [1, 2, 3, 4];
    let n_values = [4, 8, 16, 32, 64];

    println!("\n{:=<70}", "");
    println!("2D zero-mean convergence: (alpha*M + beta*K) u = f  on [0,1]²");
    println!("alpha = {alpha}, beta = {beta}");
    println!("u = sin(2πx)·sin(2πy),  ∫f=0  (compatibility for α=0)");
    println!("{:=<70}\n", "");
    println!(
        "{:<6} {:>8} {:>14} {:>10} {:>10}",
        "r", "n", "h", "max_err", "rate"
    );
    println!("{:-<52}", "");

    for &r in &orders {
        let mut prev_h: Option<f64> = None;
        let mut prev_err: Option<f64> = None;

        for &n in &n_values {
            let h = 1.0 / n as f64;
            let solver = PoissonSolverND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                alpha,
                beta,
                [BoundaryCondition::Periodic; 2],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f);

            let max_err = (0..total)
                .map(|i| (f[i].re - exact_u(x[i][0], x[i][1])).abs())
                .fold(0.0_f64, f64::max);

            let rate = match (prev_h, prev_err) {
                (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                    let rate_val = (pe / max_err).ln() / (ph / h).ln();
                    assert!(
                        max_err < 1e-5 || rate_val >= (r as f64 - 1.0) - 0.1,
                        "2D zero-mean r={r} n={n}: rate {rate_val:.2} < {} but err={max_err:.2e}",
                        r + 1
                    );
                    format!("{rate_val:.2}")
                }
                _ => "-".to_string(),
            };

            println!("{r:<6} {n:>8} {h:>14.2e} {max_err:>10.2e} {rate:>10}");
            prev_h = Some(h);
            prev_err = Some(max_err);
        }
        println!();
    }
}

/// 3D convergence measured at 100 deterministic off-grid points via `eval`.
///
/// Checks h-convergence (r=3 fixed, n increasing) using the same tensor-product
/// test function as `convergence_3d`.  Error is the max over 100 pseudo-random
/// points that do NOT coincide with any GLL node.
#[test]
fn convergence_3d_eval() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    let ls = [1.0_f64, 2.0, 3.0];
    let xls = [0.0_f64, 1.0, -1.0];

    let u1d = |x: f64| (w * x).sin().exp();
    let u1d_pp = |x: f64| {
        let s = (w * x).sin();
        let c = (w * x).cos();
        w2 * (c * c - s) * s.exp()
    };
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

    let r = 5;
    let n_values = [4, 8, 16, 32, 64, 128];

    println!("\n{:=<55}", "");
    println!("3D eval convergence (HoFFT, r={r}), 100 off-grid points");
    println!(
        "domain: [{},{}]×[{},{}]×[{},{}]",
        xls[0],
        xls[0] + ls[0],
        xls[1],
        xls[1] + ls[1],
        xls[2],
        xls[2] + ls[2]
    );
    println!("{:=<55}", "");
    println!("{:>8} {:>14} {:>10}", "n", "max_err_eval", "rate");
    println!("{:-<36}", "");

    let mut prev_h: Option<f64> = None;
    let mut prev_err: Option<f64> = None;

    for &n in &n_values {
        let h = 1.0 / n as f64;
        let solver = PoissonSolverND::<3>::new(
            ls,
            xls,
            [n; 3],
            [r; 3],
            alpha,
            beta,
            [BoundaryCondition::Periodic; 3],
        );
        let x = solver.get_x();
        let total = x.len();

        let mut f: Vec<Complex<f64>> = (0..total)
            .map(|i| {
                let xi = (x[i][0] - xls[0]) / ls[0];
                let yi = (x[i][1] - xls[1]) / ls[1];
                let zi = (x[i][2] - xls[2]) / ls[2];
                let (ux, uy, uz) = (u1d(xi), u1d(yi), u1d(zi));
                let uxx = u1d_pp(xi) / ls[0].powi(2);
                let uyy = u1d_pp(yi) / ls[1].powi(2);
                let uzz = u1d_pp(zi) / ls[2].powi(2);
                Complex::new(
                    alpha * ux * uy * uz - beta * (uxx * uy * uz + ux * uyy * uz + ux * uy * uzz),
                    0.0,
                )
            })
            .collect();
        solver.solve_in_place(&mut f);

        let mut vals = vec![Complex::new(0.0_f64, 0.0); pts.len()];
        solver.fft.eval(&pts, &f, &mut vals);
        let max_err = pts
            .iter()
            .zip(vals.iter())
            .map(|(pt, v)| (v.re - exact_u(pt[0], pt[1], pt[2])).abs())
            .fold(0.0_f64, f64::max);

        let rate = match (prev_h, prev_err) {
            (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                let rv = (pe / max_err).ln() / (ph / h).ln();
                assert!(
                    max_err < 1e-8 || rv >= r as f64 - 1.0,
                    "3D eval r={r} n={n}: rate {rv:.2} < {}, err={max_err:.2e}",
                    r + 1
                );
                format!("{rv:.2}")
            }
            _ => "-".to_string(),
        };

        println!("{n:>8} {max_err:>14.2e} {rate:>10}");
        prev_h = Some(h);
        prev_err = Some(max_err);
    }
}

// ── 4D convection exactness test ──────────────────────────────────────────────

#[test]
fn convect_4d_q6() {
    // Q6 elements (r=6), 4 elements per direction, 4D periodic domain [0,1]^4.
    //
    // u     = Q3 function  → exactly represented in Q6 space
    // φ_d   = Q2 shift vanishing at boundaries: φ(t) = t + 0.3·t·(1−t)
    //         φ(0)=0, φ(1)=1, monotone on [0,1]  → periodicity preserved
    // u∘φ   = Q3 ∘ Q2 = Q6  → also exactly represented
    //
    // Expected max error: machine precision (~1e-13).
    let r = 6;
    let n = 6;

    let solver = PoissonSolverND::<4>::new(
        [1.0; 4],
        [0.0; 4],
        [n; 4],
        [r; 4],
        1.0,
        1.0,
        [BoundaryCondition::Periodic; 4],
    );
    let x = solver.get_x();
    let total = x.len();

    // Q3 factor: q(t) = t²·(t−1),  q(0)=q(1)=0  →  periodic value ✓
    // The periodic boundary wrap in flat_dof maps the right endpoint of the last
    // element to DOF 0 (position 0), so q must satisfy q(0)=q(1) for exact repr.
    let q = |t: f64| t * t * (t - 1.0);

    // Build u at all DOF nodes.
    let u: Vec<f64> = (0..total)
        .map(|p| (0..4_usize).map(|d| q(x[p][d])).product::<f64>())
        .collect();

    // Q2 flow map vanishing at both endpoints.
    let phi = |t: f64| t + 0.3 * t * (1.0 - t);
    let lambda = |pt: [f64; 4]| pt.map(phi);

    // Exact convected values at each DOF node (computed directly).
    let exact: Vec<f64> = (0..total)
        .map(|p| (0..4_usize).map(|d| q(phi(x[p][d]))).product::<f64>())
        .collect();

    println!("\n{:=<62}", "");
    println!("4D convect — r={r}, n={n}, DOFs = {total}  ({}^4)", n * r);
    println!("u = Q3,  φ = t+0.3t(1−t) (Q2, bdry-preserving)  →  u∘φ = Q6");
    println!("{:=<62}\n", "");

    let u_cx: Vec<Complex<f64>> = u.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let t0 = Instant::now();
    let result = solver.fft.convect(lambda, &u_cx);
    let elapsed = t0.elapsed();

    let max_err = result
        .iter()
        .zip(exact.iter())
        .map(|(a, &b)| (a.re - b).abs())
        .fold(0.0_f64, f64::max);

    println!("  max error : {max_err:.3e}");
    println!("  wall time : {:.1} ms", elapsed.as_secs_f64() * 1e3);

    assert!(
        max_err < 1e-10,
        "4D Q6 convect: max error {max_err:.2e} > 1e-10"
    );
}
