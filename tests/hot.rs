use gg_lobatto::hot::*;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

// ── Eigendecomposition ───────────────────────────────────────────────────────

#[test]
fn test_hot1d_eigendecomposition() {
    const ORTHO_TOL: f64 = 1e-10;
    const RECON_TOL: f64 = 1e-10;
    const PINV_TOL: f64 = 1e-8;

    for &r in &[2, 3, 4, 5, 6] {
        let solver = Hot1DLaplaceSolver::new(1.0, r, 1.0, 1.0);
        let m = DMatrix::from_diagonal(&solver.periodic_mass_matrix);
        let id = DMatrix::<f64>::identity(r, r);
        let v = &solver.eigenvectors;

        // V^T M V = I  (M-orthonormality)
        let vtmv = v.transpose() * &m * v;
        let err = (&vtmv - &id).norm();
        assert!(err < ORTHO_TOL, "r={r}: V^T M V ≠ I  (err={err:.2e})");

        // symbol * V = M * V * Λ  (generalised eigenproblem)
        let lambda = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| solver.eigenvalues[i]));
        let lhs = &solver.symbol * v;
        let rhs = &m * v * &lambda;
        let rel = (&lhs - &rhs).norm() / rhs.norm();
        assert!(rel < RECON_TOL, "r={r}: symbol·V ≠ M·V·Λ  (rel={rel:.2e})");

        // symbol * symbol_pseudo_inv ≈ I  (when alpha=1 pseudo-inv = inv)
        let prod = &solver.symbol * &solver.symbol_pseudo_inv;
        let pinv_err = (&prod - &id).norm();
        assert!(
            pinv_err < PINV_TOL,
            "r={r}: symbol · symbol_pseudo_inv ≠ I  (err={pinv_err:.2e})"
        );
    }
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
    let tests: Vec<(
        &str,
        Box<dyn Fn(f64) -> f64>,
        Box<dyn Fn(f64) -> f64>,
        usize,
        usize,
        f64,
    )> = vec![
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
    println!("HOT 1D p-convergence on [0,1],  α={alpha}, β={beta}");
    println!("{:=<65}\n", "");

    for (name, exact_u, rhs_f, r_mono, r_max, err_min) in &tests {
        println!("  u = {name}  (monotone check from r={r_mono})");
        println!("{:>5} {:>14} {:>12}", "r", "max_err", "ratio");
        println!("{:-<34}", "");

        let mut prev_err: Option<f64> = None;

        for r in 2..=*r_max {
            let solver = Hot1DLaplaceSolver::new(1.0, r, alpha, beta);
            let x = solver.get_x();
            let mut f = DVector::from_fn(r, |i, _| rhs_f(x[i]));
            solver.solve_in_place(&mut f);
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
    println!("HOT 2D p-convergence on [0,1]²,  α={alpha}, β={beta}");
    println!("u = exp(sin(2πx))·exp(sin(2πy))  (monotone check from r={r_mono})");
    println!("{:=<55}\n", "");
    println!("{:>5} {:>10} {:>12}", "r", "ndof", "max_err");
    println!("{:-<30}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver = HotNDLaplaceSolver::<2>::new([1.0; 2], [r; 2], alpha, beta);
        let x = solver.get_x();
        let total = x[0].len();
        let mut f = DVector::from_fn(total, |i, _| rhs_f(x[0][i], x[1][i]));
        solver.solve_in_place(&mut f);
        let err = (0..total)
            .map(|i| (f[i] - exact_u(x[0][i], x[1][i])).abs())
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

// ── 3D p-convergence ─────────────────────────────────────────────────────────

#[test]
fn convergence_3d() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;
    let w2 = w * w;

    // u = exp(sin(wx))·exp(sin(wy))·exp(sin(wz))
    // -Δu  = Σ_d (-u_d''(xd)) · Π_{d'≠d} u_d'(xd')
    // u_d'' = w²·(cos²(w·xd) - sin(w·xd))·exp(sin(w·xd))
    let u1d = |x: f64| (w * x).sin().exp();
    let u1d_pp = |x: f64| {
        let s = (w * x).sin();
        let c = (w * x).cos();
        w2 * (c * c - s) * s.exp()
    };

    println!("\n{:=<65}", "");
    println!("HOT 3D p-convergence on [0,1]³,  α={alpha}, β={beta}");

    let r_mono = 7; // 3D pre-asymptotic regime is wider — check from r=7
    let r_max = 20;
    println!("u = exp(sin(2πx))·exp(sin(2πy))·exp(sin(2πz))  (monotone check from r={r_mono})");
    println!("{:=<55}\n", "");
    println!("{:>5} {:>12} {:>12}", "r", "ndof", "max_err");
    println!("{:-<32}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver = HotNDLaplaceSolver::<3>::new([1.0; 3], [r; 3], alpha, beta);
        let x = solver.get_x();
        let total = x[0].len();

        let mut f = DVector::from_fn(total, |i, _| {
            let (xi, yi, zi) = (x[0][i], x[1][i], x[2][i]);
            let (ux, uy, uz) = (u1d(xi), u1d(yi), u1d(zi));
            let (uxx, uyy, uzz) = (u1d_pp(xi), u1d_pp(yi), u1d_pp(zi));
            alpha * ux * uy * uz - beta * (uxx * uy * uz + ux * uyy * uz + ux * uy * uzz)
        });
        solver.solve_in_place(&mut f);

        let err = (0..total)
            .map(|i| (f[i] - u1d(x[0][i]) * u1d(x[1][i]) * u1d(x[2][i])).abs())
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
