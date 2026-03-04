use gg_lobatto::hofft::*;
use nalgebra::DVector;
use std::f64::consts::PI;

/// Solve (alpha - beta * d^2/dx^2) u = f and return the max error against exact_u.
fn compute_error(
    l: f64,
    n: usize,
    r: usize,
    alpha: f64,
    beta: f64,
    exact_u: &dyn Fn(f64) -> f64,
    rhs_f: &dyn Fn(f64) -> f64,
) -> f64 {
    let solver = HoFFT1DLaplaceSolver::new(l, n, r, alpha, beta);
    let x = solver.get_x();
    let ndof = n * r;

    let mut f = DVector::<f64>::zeros(ndof);
    for i in 0..ndof {
        f[i] = rhs_f(x[i]);
    }

    solver.solve_in_place(&mut f);

    let mut max_error = 0.0_f64;
    for i in 0..ndof {
        max_error = max_error.max((f[i] - exact_u(x[i])).abs());
    }
    max_error
}

#[test]
fn convergence_curves() {
    let l = 1.0;
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
    let n_values = vec![4, 8, 16, 32, 64, 128];

    println!("\n{:=<80}", "");
    println!("Convergence study: (alpha*M + beta*K) u = f  on [0, {l}]");
    println!("alpha = {alpha}, beta = {beta}");
    println!("{:=<80}\n", "");

    for (name, exact_u, rhs_f) in &test_functions {
        println!("--- Test function: u(x) = {name} ---\n");
        println!(
            "{:<6} {:>10} {:>14} {:>14} {:>10}",
            "r", "n", "h", "max_error", "rate"
        );
        println!("{:-<58}", "");

        for &r in &orders {
            let mut prev_h: Option<f64> = None;
            let mut prev_err: Option<f64> = None;

            for &n in &n_values {
                let h = l / (n as f64);
                let err = compute_error(l, n, r, alpha, beta, exact_u.as_ref(), rhs_f.as_ref());

                let rate = match (prev_h, prev_err) {
                    (Some(ph), Some(pe)) if pe > 0.0 && err > 0.0 => {
                        let r = (pe / err).ln() / (ph / h).ln();
                        format!("{r:.2}")
                    }
                    _ => "-".to_string(),
                };

                println!("{r:<6} {n:>10} {h:>14.2e} {err:>14.2e} {rate:>10}");

                prev_h = Some(h);
                prev_err = Some(err);
            }
            println!();
        }
        println!();
    }
}
