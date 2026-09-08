use lobatto_fft::solver::*;
use nalgebra::Complex;
use std::cell::RefCell;
use std::f64::consts::PI;

/// A test case: (name, exact solution u(x), right-hand side f(x)).
type TestCase<'a> = (&'a str, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>);

/// Solve (alpha - beta * d^2/dx^2) u = f and return the max error against exact_u.
#[allow(clippy::too_many_arguments)]
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
    let solver = Poisson::new(l, 0.0, n, r, bc);
    let x = solver.get_x();
    let ndof = x.len();

    let mut f: Vec<Complex<f64>> = x.iter().map(|&xi| Complex::new(rhs_f(xi), 0.0)).collect();

    solver.solve_in_place(&mut f, alpha, beta);

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
    let test_functions: Vec<TestCase> = vec![
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
    let test_functions: Vec<TestCase> = vec![
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
    let test_functions: Vec<TestCase> = vec![
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
            let solver = PoissonND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                [BoundaryCondition::Periodic; 2],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f, alpha, beta);

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
            let solver = PoissonND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                [BoundaryCondition::Dirichlet; 2],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f, alpha, beta);

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
            let solver = PoissonND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                [BoundaryCondition::Neumann; 2],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f, alpha, beta);

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
            let solver = PoissonND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                [BoundaryCondition::Dirichlet, BoundaryCondition::Neumann],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f, alpha, beta);

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
        let solver = PoissonND::<2>::new(
            [1.0; 2],
            [0.0; 2],
            ns,
            rs,
            [BoundaryCondition::Periodic; 2],
        );
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
            .collect();
        solver.solve_in_place(&mut f, alpha, beta);
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
    let test_functions: Vec<TestCase> = vec![
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
                let solver = PoissonND::<3>::new(
                    ls,
                    xls,
                    [n; 3],
                    [r; 3],
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
                solver.solve_in_place(&mut f, alpha, beta);

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
    let c = alpha
        + beta * PI * PI * (1.0 / (ls[0] * ls[0]) + 1.0 / (ls[1] * ls[1]) + 1.0 / (ls[2] * ls[2]));

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
        xls[0],
        xls[0] + ls[0],
        xls[1],
        xls[1] + ls[1],
        xls[2],
        xls[2] + ls[2]
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
            let solver = PoissonND::<3>::new(
                ls,
                xls,
                [n; 3],
                [r; 3],
                [BoundaryCondition::Dirichlet; 3],
            );
            let x = solver.get_x();
            let total = x.len();
            let mut f: Vec<Complex<f64>> =
                (0..total).map(|i| Complex::new(rhs_f(x[i]), 0.0)).collect();
            solver.solve_in_place(&mut f, alpha, beta);
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
    let c = alpha
        + beta * PI * PI * (1.0 / (ls[0] * ls[0]) + 1.0 / (ls[1] * ls[1]) + 1.0 / (ls[2] * ls[2]));

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
        xls[0],
        xls[0] + ls[0],
        xls[1],
        xls[1] + ls[1],
        xls[2],
        xls[2] + ls[2]
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
            let solver = PoissonND::<3>::new(
                ls,
                xls,
                [n; 3],
                [r; 3],
                [BoundaryCondition::Neumann; 3],
            );
            let x = solver.get_x();
            let total = x.len();
            let mut f: Vec<Complex<f64>> =
                (0..total).map(|i| Complex::new(rhs_f(x[i]), 0.0)).collect();
            solver.solve_in_place(&mut f, alpha, beta);
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
        let solver = PoissonND::<3>::new(ls, xls, ns, rs, bc);
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(rhs_f(x[i]), 0.0)).collect();
        solver.solve_in_place(&mut f, alpha, beta);
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
            let solver = PoissonND::<2>::new(
                [1.0; 2],
                [0.0; 2],
                [n; 2],
                [r; 2],
                [BoundaryCondition::Periodic; 2],
            );
            let x = solver.get_x();
            let total = x.len();

            let mut f: Vec<Complex<f64>> = (0..total)
                .map(|i| Complex::new(rhs_f(x[i][0], x[i][1]), 0.0))
                .collect();
            solver.solve_in_place(&mut f, alpha, beta);

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





/// `solve_in_place_real` must reproduce the real part of the complex solve
/// exactly (the imaginary parts stay identically zero through the pipeline).
#[test]
fn solve_in_place_real_matches_complex() {
    // 1D, Dirichlet.
    let solver = Poisson::new(1.0, 0.0, 8, 3, BoundaryCondition::Dirichlet);
    let x = solver.get_x();
    let f_re: Vec<f64> = x.iter().map(|&xi| (2.7 * xi + 0.3).sin()).collect();
    let mut f_cx: Vec<Complex<f64>> = f_re.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut f_real = f_re.clone();
    solver.solve_in_place(&mut f_cx, 1.0, 1.0);
    solver.solve_in_place_real(&mut f_real, 1.0, 1.0);
    for (a, b) in f_real.iter().zip(f_cx.iter()) {
        assert_eq!(*a, b.re);
    }

    // 2D, mixed BCs.
    let solver = PoissonND::<2>::new(
        [1.0; 2],
        [0.0; 2],
        [6; 2],
        [3; 2],
        [BoundaryCondition::Periodic, BoundaryCondition::Neumann],
    );
    let x = solver.get_x();
    let f_re: Vec<f64> = x
        .iter()
        .map(|&[a, b]| (2.1 * a).sin() * (1.3 * b + 0.2).cos())
        .collect();
    let mut f_cx: Vec<Complex<f64>> = f_re.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut f_real = f_re.clone();
    solver.solve_in_place(&mut f_cx, 1.0, 1.0);
    solver.solve_in_place_real(&mut f_real, 1.0, 1.0);
    for (a, b) in f_real.iter().zip(f_cx.iter()) {
        assert_eq!(*a, b.re);
    }
}

// ── Complex coefficients ─────────────────────────────────────────────────────

/// Scaling both coefficients by a complex scalar `c` must scale the discrete
/// solution by exactly `1/c`: `(cα M + cβ K) u = f  ⟺  u = u₀ / c`. This is an
/// exact algebraic identity of the discrete operator, so it validates the
/// complex division and the pseudo-inverse scaling in every solve path
/// independently of any discretisation error.
#[test]
fn complex_coefficient_scaling_is_exact() {
    const TOL: f64 = 1e-13;
    let c = Complex::new(0.7, -1.9);

    // 1D, all BCs.
    for bc in [
        BoundaryCondition::Periodic,
        BoundaryCondition::Dirichlet,
        BoundaryCondition::Neumann,
    ] {
        let (n, r) = (12, 4);
        let solver = Poisson::new(2.3, -0.4, n, r, bc);
        let x = solver.get_x();
        let f: Vec<Complex<f64>> = x
            .iter()
            .map(|&xi| Complex::new((2.7 * xi + 0.3).sin(), (1.1 * xi).cos()))
            .collect();
        let mut u0 = f.clone();
        let mut u1 = f.clone();
        solver.solve_in_place(&mut u0, 1.5, 0.8);
        solver.solve_in_place(&mut u1, c * 1.5, c * 0.8);
        let scale = u0.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
        for (a, b) in u0.iter().zip(&u1) {
            assert!(
                (a / c - b).norm() <= TOL * scale,
                "1D {bc:?}: |u0/c - u1| = {:.2e}",
                (a / c - b).norm()
            );
        }
    }

    // 2D, mixed BCs.
    let bcs = [BoundaryCondition::Periodic, BoundaryCondition::Neumann];
    let solver = PoissonND::<2>::new([1.0, 1.7], [0.0, 0.2], [6, 5], [3, 4], bcs);
    let x = solver.get_x();
    let f: Vec<Complex<f64>> = x
        .iter()
        .map(|&[a, b]| Complex::new((2.1 * a).sin() * (1.3 * b + 0.2).cos(), (a * b).sin()))
        .collect();
    let mut u0 = f.clone();
    let mut u1 = f;
    solver.solve_in_place(&mut u0, 1.5, 0.8);
    solver.solve_in_place(&mut u1, c * 1.5, c * 0.8);
    let scale = u0.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
    for (a, b) in u0.iter().zip(&u1) {
        assert!(
            (a / c - b).norm() <= TOL * scale,
            "2D: |u0/c - u1| = {:.2e}",
            (a / c - b).norm()
        );
    }
}

/// Damped Helmholtz `(-ω² + iωc) u - u'' = f` with a manufactured solution:
/// spectral convergence towards the exact solution with a genuinely complex
/// `α` and real `β`, 1D (all BCs) and 3D mixed.
#[test]
fn convergence_complex_alpha() {
    let omega = 2.0;
    let damping = 0.7;
    let alpha = Complex::new(-omega * omega, omega * damping);
    let beta = 1.0;

    // 1D: u = sin(2πx) (Periodic, Dirichlet) or cos(2πx) (Neumann) on [0, 1],
    // f = (α + β w²) u.
    let w = 2.0 * PI;
    let cw = alpha + beta * w * w;
    let u_sin = |x: f64| (w * x).sin();
    let u_cos = |x: f64| (w * x).cos();
    let cases: [(BoundaryCondition, &dyn Fn(f64) -> f64); 3] = [
        (BoundaryCondition::Periodic, &u_sin),
        (BoundaryCondition::Dirichlet, &u_sin),
        (BoundaryCondition::Neumann, &u_cos),
    ];
    for (bc, u1d) in cases {
        let mut prev: Option<f64> = None;
        for n in [4, 8, 16, 32] {
            let r = 4;
            let solver = Poisson::new(1.0, 0.0, n, r, bc);
            let x = solver.get_x();
            let mut f: Vec<Complex<f64>> = x.iter().map(|&xi| cw * u1d(xi)).collect();
            solver.solve_in_place(&mut f, alpha, beta);
            let err = x
                .iter()
                .zip(&f)
                .map(|(&xi, u)| (u - u1d(xi)).norm())
                .fold(0.0f64, f64::max);
            println!("1D {bc:?} complex α: n={n} r={r} err={err:.2e}");
            if let Some(pe) = prev {
                let rate = (pe / err).log2();
                assert!(
                    err < 1e-10 || rate >= (r as f64 + 1.0) - 0.1,
                    "1D {bc:?} n={n}: rate {rate:.2} < {}, err={err:.2e}",
                    r + 1
                );
            }
            prev = Some(err);
        }
    }

    // 3D mixed: u = sin(πx)·cos(πy)·sin(2πz), f = α u − β Δu.
    let bcs = [
        BoundaryCondition::Dirichlet,
        BoundaryCondition::Neumann,
        BoundaryCondition::Periodic,
    ];
    let exact = |p: [f64; 3]| (PI * p[0]).sin() * (PI * p[1]).cos() * (2.0 * PI * p[2]).sin();
    let lap_coeff = -(PI * PI + PI * PI + 4.0 * PI * PI);
    let c3 = alpha - beta * lap_coeff;
    let mut prev: Option<f64> = None;
    for n in [4, 8, 16] {
        let r = 4;
        let solver = PoissonND::<3>::new([1.0; 3], [0.0; 3], [n; 3], [r; 3], bcs);
        let x = solver.get_x();
        let mut f: Vec<Complex<f64>> = x.iter().map(|&p| c3 * exact(p)).collect();
        solver.solve_in_place(&mut f, alpha, beta);
        let err = x
            .iter()
            .zip(&f)
            .map(|(&p, u)| (u - exact(p)).norm())
            .fold(0.0f64, f64::max);
        println!("3D mixed complex α: n={n} r={r} err={err:.2e}");
        if let Some(pe) = prev {
            let rate = (pe / err).log2();
            assert!(
                err < 1e-10 || rate >= (r as f64 + 1.0) - 0.1,
                "3D n={n}: rate {rate:.2} < {}, err={err:.2e}",
                r + 1
            );
        }
        prev = Some(err);
    }
}

/// Real coefficients passed as `f64` and as `Complex<f64>` give identical solves.
#[test]
fn f64_and_complex_real_coefficients_agree() {
    let solver = Poisson::new(1.0, 0.0, 8, 3, BoundaryCondition::Dirichlet);
    let x = solver.get_x();
    let f: Vec<Complex<f64>> = x.iter().map(|&xi| Complex::new((3.0 * xi).sin(), 0.0)).collect();
    let (mut ua, mut ub) = (f.clone(), f);
    solver.solve_in_place(&mut ua, 2.0, 0.5);
    solver.solve_in_place(&mut ub, Complex::new(2.0, 0.0), Complex::new(0.5, 0.0));
    assert_eq!(ua, ub);
}

// ── Operator functions ───────────────────────────────────────────────────────

const ALL_BC: [BoundaryCondition; 3] = [
    BoundaryCondition::Periodic,
    BoundaryCondition::Dirichlet,
    BoundaryCondition::Neumann,
];

fn max_norm(v: &[Complex<f64>]) -> f64 {
    v.iter().map(|z| z.norm()).fold(0.0, f64::max)
}

fn max_diff(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max)
}

/// Deterministic pseudo-random complex vector.
fn pseudo_random(len: usize, seed: u64) -> Vec<Complex<f64>> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..len).map(|_| Complex::new(next(), next())).collect()
}

/// Diagonal of the assembled mass matrix, recovered from a `β = 0` solve:
/// `(1·M)⁻¹ 1 = (1/M_ii)_i`.
fn mass_diagonal_1d(n: usize, r: usize, bc: BoundaryCondition) -> Vec<f64> {
    let s = Poisson::new(1.0, 0.0, n, r, bc);
    let mut ones = vec![Complex::new(1.0, 0.0); s.get_x().len()];
    s.solve_in_place(&mut ones, 1.0, 0.0);
    ones.iter().map(|z| 1.0 / z.re).collect()
}

fn mass_diagonal_2d(ns: [usize; 2], rs: [usize; 2], bc: [BoundaryCondition; 2]) -> Vec<f64> {
    let s = PoissonND::<2>::new([1.0; 2], [0.0; 2], ns, rs, bc);
    let mut ones = vec![Complex::new(1.0, 0.0); s.get_x().len()];
    s.solve_in_place(&mut ones, 1.0, 0.0);
    ones.iter().map(|z| 1.0 / z.re).collect()
}

/// `apply_in_place(u, 1/λ)` = `(αM + βK)⁻¹ M u` = `solve_in_place(M u)`, 1D and
/// 2D, all BCs, complex `α`.
#[test]
fn apply_reproduces_solve() {
    const TOL: f64 = 1e-12;
    let alpha = Complex::new(0.8, -1.3);
    let beta = Complex::new(1.0, 0.4);
    let inv = |l: Complex<f64>| 1.0 / l;

    for bc in ALL_BC {
        let (n, r) = (10, 4);
        let solver = Poisson::new(1.0, 0.0, n, r, bc);
        let m = mass_diagonal_1d(n, r, bc);
        let u = pseudo_random(m.len(), 11);
        let mut via_apply = u.clone();
        solver.apply_in_place(&mut via_apply, alpha, beta, inv);
        let mut via_solve: Vec<Complex<f64>> = u.iter().zip(&m).map(|(z, &w)| z * w).collect();
        solver.solve_in_place(&mut via_solve, alpha, beta);
        let err = max_diff(&via_apply, &via_solve) / max_norm(&via_solve);
        assert!(err < TOL, "1D {bc:?}: apply vs solve rel err {err:.2e}");

        // Real wrapper (real coefficients): real part of the complex apply.
        let mut f_re: Vec<f64> = u.iter().map(|z| z.re).collect();
        let mut f_cx: Vec<Complex<f64>> = f_re.iter().map(|&v| Complex::new(v, 0.0)).collect();
        solver.apply_in_place_real(&mut f_re, 0.8, 1.0, inv);
        solver.apply_in_place(&mut f_cx, 0.8, 1.0, inv);
        for (a, b) in f_re.iter().zip(&f_cx) {
            assert_eq!(*a, b.re);
        }
    }

    for bc in [
        [BoundaryCondition::Periodic; 2],
        [BoundaryCondition::Dirichlet, BoundaryCondition::Neumann],
        [BoundaryCondition::Neumann, BoundaryCondition::Periodic],
    ] {
        let (ns, rs) = ([5, 6], [3, 4]);
        let solver = PoissonND::<2>::new([1.0; 2], [0.0; 2], ns, rs, bc);
        let m = mass_diagonal_2d(ns, rs, bc);
        let u = pseudo_random(m.len(), 13);
        let mut via_apply = u.clone();
        solver.apply_in_place(&mut via_apply, alpha, beta, inv);
        let mut via_solve: Vec<Complex<f64>> = u.iter().zip(&m).map(|(z, &w)| z * w).collect();
        solver.solve_in_place(&mut via_solve, alpha, beta);
        let err = max_diff(&via_apply, &via_solve) / max_norm(&via_solve);
        assert!(err < TOL, "2D {bc:?}: apply vs solve rel err {err:.2e}");

        let mut f_re: Vec<f64> = u.iter().map(|z| z.re).collect();
        let mut f_cx: Vec<Complex<f64>> = f_re.iter().map(|&v| Complex::new(v, 0.0)).collect();
        solver.apply_in_place_real(&mut f_re, 0.8, 1.0, inv);
        solver.apply_in_place(&mut f_cx, 0.8, 1.0, inv);
        for (a, b) in f_re.iter().zip(&f_cx) {
            assert_eq!(*a, b.re);
        }
    }
}

/// Dense matrix of a 1D operator function obtained column by column
/// (`out[j]` is column `j`).
fn dense_apply_1d(
    solver: &Poisson,
    alpha: f64,
    beta: f64,
    g: impl Fn(Complex<f64>) -> Complex<f64>,
) -> Vec<Vec<Complex<f64>>> {
    let n = solver.get_x().len();
    (0..n)
        .map(|i| {
            let mut e = vec![Complex::new(0.0, 0.0); n];
            e[i] = Complex::new(1.0, 0.0);
            solver.apply_in_place(&mut e, alpha, beta, &g);
            e
        })
        .collect()
}

/// Directionally split Euler step = Kronecker product `S_0 ⊗ S_1` of the 1D
/// steps, and = the 1D steps applied fibre by fibre. Periodic (`n = 4`,
/// `r = 3`, `dt = 0.05 h²`) and mixed Dirichlet/Neumann with different
/// `n_d`, `r_d`.
#[test]
fn split_step_is_kronecker_product() {
    const TOL: f64 = 1e-13;
    for (ns, rs, bc) in [
        ([4, 4], [3, 3], [BoundaryCondition::Periodic; 2]),
        ([5, 3], [2, 4], [BoundaryCondition::Dirichlet, BoundaryCondition::Neumann]),
        ([3, 4], [3, 2], [BoundaryCondition::Neumann, BoundaryCondition::Periodic]),
    ] {
        let h = 1.0 / ns[0] as f64;
        let dt = 0.05 * h * h;
        let solver = PoissonND::<2>::new([1.0; 2], [0.0; 2], ns, rs, bc);
        let s1: [Vec<Vec<Complex<f64>>>; 2] = std::array::from_fn(|d| {
            let p = Poisson::new(1.0, 0.0, ns[d], rs[d], bc[d]);
            dense_apply_1d(&p, 0.0, 1.0, |l| 1.0 / (1.0 + dt * l))
        });
        let (m0, m1) = (s1[0].len(), s1[1].len());
        let total = m0 * m1;
        assert_eq!(total, solver.get_x().len());

        // Columns of the ND operator vs the Kronecker product (direction 0 fastest).
        let mut worst = 0.0f64;
        for q in 0..total {
            let (j0, j1) = (q % m0, q / m0);
            let mut e = vec![Complex::new(0.0, 0.0); total];
            e[q] = Complex::new(1.0, 0.0);
            solver.euler_split_step_in_place(&mut e, 0.0, 1.0, dt);
            for (p, &ep) in e.iter().enumerate() {
                let (i0, i1) = (p % m0, p / m0);
                let kron = s1[0][j0][i0] * s1[1][j1][i1];
                worst = worst.max((ep - kron).norm());
            }
        }
        assert!(worst < TOL, "{bc:?}: split vs Kronecker max diff {worst:.2e}");

        // Fibre-by-fibre application of the dense 1D operators to a random vector.
        let u = pseudo_random(total, 17);
        let mut split = u.clone();
        solver.euler_split_step_in_place(&mut split, 0.0, 1.0, dt);
        let mut seq = u.clone();
        for i1 in 0..m1 {
            let fib: Vec<Complex<f64>> = (0..m0).map(|i0| seq[i0 + m0 * i1]).collect();
            for i0 in 0..m0 {
                seq[i0 + m0 * i1] = (0..m0).map(|j0| s1[0][j0][i0] * fib[j0]).sum();
            }
        }
        for i0 in 0..m0 {
            let fib: Vec<Complex<f64>> = (0..m1).map(|i1| seq[i0 + m0 * i1]).collect();
            for i1 in 0..m1 {
                seq[i0 + m0 * i1] = (0..m1).map(|j1| s1[1][j1][i1] * fib[j1]).sum();
            }
        }
        let err = max_diff(&split, &seq) / max_norm(&seq);
        assert!(err < TOL, "{bc:?}: split vs fibrewise rel err {err:.2e}");

        // apply_in_place_dirs with the explicit closure equals the convenience method.
        let mut dirs = u.clone();
        solver.apply_in_place_dirs(&mut dirs, |mu| {
            Complex::from(mu.iter().map(|&m| 1.0 / (1.0 + dt * m)).product::<f64>())
        });
        let err = max_diff(&dirs, &split) / max_norm(&split);
        assert!(err < TOL, "{bc:?}: apply_in_place_dirs vs euler_split rel err {err:.2e}");
    }
}

/// `exp(-t L)` in ND is the Kronecker product of the 1D exponentials:
/// `apply(exp(-tλ))` = `exp(-tα) · apply_dirs(Π_d exp(-tβ μ_d))`, complex `α`, `β`.
#[test]
fn exp_is_kronecker_product() {
    const TOL: f64 = 1e-14;
    let alpha = Complex::new(0.3, -0.7);
    let beta = Complex::new(1.0, 0.2);
    let t = 2e-3;
    let solver = PoissonND::<3>::new(
        [1.0; 3],
        [0.0; 3],
        [3, 4, 2],
        [2, 3, 3],
        [
            BoundaryCondition::Periodic,
            BoundaryCondition::Dirichlet,
            BoundaryCondition::Neumann,
        ],
    );
    let u = pseudo_random(solver.get_x().len(), 19);
    let mut a = u.clone();
    solver.exp_in_place(&mut a, alpha, beta, t);
    let mut b = u.clone();
    solver.apply_in_place_dirs(&mut b, |mu| {
        mu.iter().map(|&m| (-t * beta * m).exp()).product::<Complex<f64>>() * (-t * alpha).exp()
    });
    let err = max_diff(&a, &b) / max_norm(&a);
    assert!(err < TOL, "exp vs Kronecker exp rel err {err:.2e}");

    // Euler convenience = closure form.
    let mut c = u.clone();
    solver.euler_step_in_place(&mut c, alpha, beta, t);
    let mut d = u.clone();
    solver.apply_in_place(&mut d, alpha, beta, |l| 1.0 / (1.0 + t * l));
    assert_eq!(c, d);
}

/// Neumann on `[0, 1]` = periodic on `[0, 2)` applied to the even extension:
/// an Euler step of the Neumann solver (`n = 8`, `r = 4`) matches the periodic
/// step with `2n` elements on the mirrored data.
#[test]
fn neumann_is_even_periodic_extension() {
    const TOL: f64 = 1e-14;
    let (n, r) = (8, 4);
    let dt = 3e-3;
    let neu = Poisson::new(1.0, 0.0, n, r, BoundaryCondition::Neumann);
    let per = Poisson::new(2.0, 0.0, 2 * n, r, BoundaryCondition::Periodic);
    let x_n = neu.get_x();
    let x_p = per.get_x();
    let nr = n * r;
    assert_eq!(x_n.len(), nr + 1);
    assert_eq!(x_p.len(), 2 * nr);

    // Even extension: node p > nr of the periodic grid mirrors node 2nr - p.
    let mirror = |p: usize| if p <= nr { p } else { 2 * nr - p };
    for (p, &xp) in x_p.iter().enumerate() {
        let xm = if p <= nr { xp } else { 2.0 - xp };
        assert!((xm - x_n[mirror(p)]).abs() < 1e-14);
    }

    let u: Vec<Complex<f64>> = x_n
        .iter()
        .map(|&x| Complex::new((3.1 * x).cos() + 0.4 * (x * x - x).exp(), 0.2 * x))
        .collect();
    let mut u_neu = u.clone();
    neu.euler_step_in_place(&mut u_neu, 0.0, 1.0, dt);
    let mut u_ext: Vec<Complex<f64>> = (0..2 * nr).map(|p| u[mirror(p)]).collect();
    per.euler_step_in_place(&mut u_ext, 0.0, 1.0, dt);

    let err = max_diff(&u_neu, &u_ext[..=nr]) / max_norm(&u_neu);
    assert!(err < TOL, "Neumann vs even periodic extension rel err {err:.2e}");
}

// ── Accuracy of the symbol eigenvalues ───────────────────────────────────────
//
// For each Fourier mode k the solver diagonalises an r×r Hermitian symbol whose
// largest eigenvalue is μ_max ~ (r·n)².  A backward-stable eigensolver returns
// every eigenvalue with absolute error ≈ ε·μ_max, including the lowest one,
// μ0(k) ≈ (πk)², which carries the physical solution.  The relative error on
// μ0 therefore grows like n², and every solve inherits an error floor
//
//     ε · μ_max / |α + β μ0|.
//
// The tests below compare against exact values (an eigenvalue known to double
// precision, single-mode analytic solutions).  Residual checks and
// solve/apply round trips do not see the problem and are not used.

/// Polynomial order used by the eigenvalue accuracy tests.
const ORDER: usize = 3;

/// Per-mode eigenvalues `μ[k][j]` of the stiffness symbol of a 1D solver.
///
/// `apply_mu_in_place` evaluates its closure once per `(k, j)`, mode after
/// mode; recording the arguments recovers the spectrum.
fn symbol_eigenvalues(solver: &Poisson) -> Vec<Vec<f64>> {
    let mut f = vec![Complex::new(0.0, 0.0); solver.get_x().len()];
    let seen = RefCell::new(Vec::new());
    solver.apply_mu_in_place(&mut f, |mu| {
        seen.borrow_mut().push(mu);
        Complex::new(0.0, 0.0)
    });
    seen.into_inner().chunks(ORDER).map(<[f64]>::to_vec).collect()
}

/// Max nodal error of the solution of `(α - d²/dx²) u = f` on `[0, 1]`.
fn solve_error_1d(
    n: usize,
    bc: BoundaryCondition,
    alpha: f64,
    exact: impl Fn(f64) -> f64,
    rhs: impl Fn(f64) -> f64,
) -> f64 {
    let solver = Poisson::new(1.0, 0.0, n, ORDER, bc);
    let x = solver.get_x();
    let mut u: Vec<f64> = x.iter().map(|&xi| rhs(xi)).collect();
    solver.solve_in_place_real(&mut u, alpha, 1.0);
    x.iter()
        .zip(&u)
        .map(|(&xi, &ui)| (ui - exact(xi)).abs())
        .fold(0.0, f64::max)
}

/// The lowest eigenvalue of mode `k = 1` is the discrete eigenvalue of
/// `sin(πx)` (Dirichlet), `cos(πx)` (Neumann) or `e^{2iπx}` (periodic) on
/// `[0, 1]`.  Its dispersion error is `O((π/n)^{2r})`, below 1e-16 for
/// `n ≥ 256`, so `π²` (resp. `4π²`) is an exact double-precision reference.
#[test]
fn lowest_symbol_eigenvalue() {
    let cases = [
        (BoundaryCondition::Dirichlet, PI * PI),
        (BoundaryCondition::Neumann, PI * PI),
        (BoundaryCondition::Periodic, 4.0 * PI * PI),
    ];
    let mut worst = 0.0_f64;
    for (bc, exact) in cases {
        println!("{bc:?}: relative error of the lowest eigenvalue of mode k = 1");
        for n in [256, 1024, 4096, 16384] {
            let solver = Poisson::new(1.0, 0.0, n, ORDER, bc);
            let mu0 = symbol_eigenvalues(&solver)[1]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let rel = ((mu0 - exact) / exact).abs();
            println!("  n = {n:>5}   mu_max = {:.1e}   error = {rel:.1e}", solver.mu_max());
            worst = worst.max(rel);
        }
    }
    assert!(worst < 1e-13, "lowest eigenvalue relative error {worst:.1e}");
}

/// Solve `-u'' + u = f` for a single Fourier mode, where the discretisation
/// error is negligible: the nodal error is the round-off floor of the solve
/// and grows with `n`.
#[test]
fn solve_floor_1d() {
    type Case = (&'static str, BoundaryCondition, fn(f64) -> f64);
    let cases: [Case; 3] = [
        ("Dirichlet, u = sin(pi x)", BoundaryCondition::Dirichlet, |x| (PI * x).sin()),
        ("Neumann,   u = cos(pi x)", BoundaryCondition::Neumann, |x| (PI * x).cos()),
        ("Periodic,  u = sin(2 pi x)", BoundaryCondition::Periodic, |x| (2.0 * PI * x).sin()),
    ];
    let mut worst = 0.0_f64;
    for (name, bc, u) in cases {
        // -u'' + u = (1 + w²) u with w = π or 2π.
        let w = if bc == BoundaryCondition::Periodic { 2.0 * PI } else { PI };
        let f = move |x: f64| (1.0 + w * w) * u(x);
        println!("{name}");
        for n in [1024, 4096, 16384] {
            let err = solve_error_1d(n, bc, 1.0, u, f);
            println!("  n = {n:>5}   max nodal error = {err:.1e}");
            worst = worst.max(err);
        }
    }
    assert!(worst < 1e-12, "1D solve error {worst:.1e}");
}

/// Same as `solve_floor_1d` in 2D: the floor is set by `Σ_d μ_max^{(d)}`.
#[test]
fn solve_floor_2d() {
    let n = 1024;
    let solver = PoissonND::<2>::new(
        [1.0; 2],
        [0.0; 2],
        [n; 2],
        [ORDER; 2],
        [BoundaryCondition::Dirichlet; 2],
    );
    let u = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let pts = solver.get_x();
    // -Δu + u = (1 + 2π²) u
    let mut f: Vec<f64> = pts.iter().map(|&[x, y]| (1.0 + 2.0 * PI * PI) * u(x, y)).collect();
    solver.solve_in_place_real(&mut f, 1.0, 1.0);
    let err = pts
        .iter()
        .zip(&f)
        .map(|(&[x, y], &ui)| (ui - u(x, y)).abs())
        .fold(0.0, f64::max);
    println!(
        "2D Dirichlet, {n}x{n} elements, sum mu_max = {:.1e}: max nodal error = {err:.1e}",
        solver.mu_max()
    );
    assert!(err < 1e-12, "2D solve error {err:.1e}");
}

/// Near a resonance the error floor is amplified by `1/|α + μ0|`.
///
/// With `α = -π²(1 - δ)` the mode `sin(πx)` is divided by `λ0 = π² δ`, and
/// the solve reproduces `u = sin(πx)` from `f = λ0 u` with relative error
/// `(error on μ0) / λ0`.
#[test]
fn near_resonance() {
    let delta = 1e-6;
    let alpha = -PI * PI * (1.0 - delta);
    let lambda0 = PI * PI * delta;
    let u = |x: f64| (PI * x).sin();
    let mut worst = 0.0_f64;
    println!("alpha = -pi^2 (1 - {delta:.0e}), lambda0 = {lambda0:.2e}");
    for n in [256, 1024, 4096] {
        let err = solve_error_1d(n, BoundaryCondition::Dirichlet, alpha, u, |x| lambda0 * u(x));
        println!("  n = {n:>5}   relative error = {err:.1e}");
        worst = worst.max(err);
    }
    assert!(worst < 1e-8, "near-resonance relative error {worst:.1e}");
}
