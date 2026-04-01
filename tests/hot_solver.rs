use nalgebra::DVector;
use quadratix::hot::*;
use std::f64::consts::PI;

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
            let solver = Hot1DLaplaceSolver::new(1.0, 0.0, r, alpha, beta);
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
        let solver = HotNDLaplaceSolver::<2>::new([1.0; 2], [0.0; 2], [r; 2], alpha, beta);
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<f64> = (0..total).map(|i| rhs_f(x[i][0], x[i][1])).collect();
        solver.solve_in_place(&mut f);
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
    println!("HOT 2D p-convergence on [0,1]²,  α={alpha}, β={beta}  (zero-mean)");
    println!("u = sin(2πx)·sin(2πy),  ∫f=0  (monotone from r={r_mono})");
    println!("{:=<60}\n", "");
    println!("{:>5} {:>10} {:>12}", "r", "ndof", "max_err");
    println!("{:-<30}", "");

    let mut prev_err: Option<f64> = None;
    for r in 2..=r_max {
        let solver = HotNDLaplaceSolver::<2>::new([1.0; 2], [0.0; 2], [r; 2], alpha, beta);
        let x = solver.get_x();
        let total = x.len();
        let mut f: Vec<f64> = (0..total).map(|i| rhs_f(x[i][0], x[i][1])).collect();
        solver.solve_in_place(&mut f);
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

    let ls = [1.0_f64, 2.0, 3.0];
    let xls = [0.0_f64, 1.0, -1.0];

    println!("\n{:=<65}", "");
    println!(
        "HOT 3D p-convergence on [{},{}]×[{},{}]×[{},{}],  α={alpha}, β={beta}",
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
        let solver = HotNDLaplaceSolver::<3>::new(ls, xls, [r; 3], alpha, beta);
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
        solver.solve_in_place(&mut f);

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

// ── 3D p-convergence measured via eval at off-grid points ────────────────────

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
        "HOT 3D eval p-convergence on [{},{}]×[{},{}]×[{},{}],  α={alpha}, β={beta}",
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
        let solver = HotNDLaplaceSolver::<3>::new(ls, xls, [r; 3], alpha, beta);
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
        solver.solve_in_place(&mut f);

        let vals = solver.eval(&pts, &f);
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
