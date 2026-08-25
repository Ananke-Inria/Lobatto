//! Example: solve the 5-D periodic Laplace problem
//!
//!   (α M + β K) u = f   on [0, 1]^5
//!
//! using two methods:
//!   - HoFFT:    n=10 elements × r=4 polynomial order  → 40 DOF/direction
//!   - HoFFT-1:  n=1  element  × r=20                  → 20 DOF/direction
//!
//! Exact solution: u = exp(sin(2π x_1)) · … · exp(sin(2π x_5))
//!
//! Run with:
//!   cargo run --example laplace_5d --release

use lobatto_fft::solver::BoundaryCondition;
use lobatto_fft::solver::PoissonND;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    let alpha = 1.0;
    let beta = 1.0;
    let w = 2.0 * PI;
    let w2 = w * w;

    // u = ∏_d u1d(x_d),  u1d(x) = exp(sin(wx))
    // -Δu = Σ_d (-u1d''(x_d)) · ∏_{d'≠d} u1d(x_{d'})
    // u1d'' = w² (cos²(wx) - sin(wx)) exp(sin(wx))
    let u1d = |x: f64| (w * x).sin().exp();
    let u1d_pp = |x: f64| {
        let s = (w * x).sin();
        let c = (w * x).cos();
        w2 * (c * c - s) * s.exp()
    };

    // 1000000 deterministic off-grid evaluation points in [0,1)^5
    let eval_hofft_pts: Vec<[f64; 5]> = (0..1000000)
        .map(|i| {
            let t = (i as f64 + 0.5) / 1000000.0;
            [
                (7.0 * PI * t).fract(),
                (11.0 * PI * t).fract(),
                (13.0 * PI * t).fract(),
                (17.0 * PI * t).fract(),
                (19.0 * PI * t).fract(),
            ]
        })
        .collect();

    // 1000 deterministic off-grid evaluation points in [0,1)^5
    let eval_hot_pts: Vec<[f64; 5]> = (0..1000)
        .map(|i| {
            let t = (i as f64 + 0.5) / 1000.0;
            [
                (7.0 * PI * t).fract(),
                (11.0 * PI * t).fract(),
                (13.0 * PI * t).fract(),
                (17.0 * PI * t).fract(),
                (19.0 * PI * t).fract(),
            ]
        })
        .collect();

    // HoFFT parameters
    let n = 10; // elements per direction
    let r = 4; // polynomial order per direction
    let ndof_1d: usize = n * r;

    // HoFFT-1 / HoT parameters: single element, same DOF/direction
    let r_hot = 20_usize;

    let ndof_hofft: usize = ndof_1d.pow(5);
    let ndof_1: usize = r_hot.pow(5); // HoFFT-1 and HoT share the same DOF count

    println!("╔═════════════════════════════════════════════════════╗");
    println!("║    5-D Laplace solver comparison: HoFFT / HoFFT-1   ║");
    println!("╚═════════════════════════════════════════════════════╝");
    println!();
    println!("  HoFFT   : n={n}×r={r} → {ndof_1d} DOF/direction, {ndof_hofft} total");
    println!("  HoFFT-1 : n=1×r={r_hot} → {r_hot} DOF/direction, {ndof_1} total");
    println!();

    // ── HoFFT ─────────────────────────────────────────────────────────────────
    println!("── HoFFT (n={n}, r={r}) ──────────────────────────────────────────");

    let t = Instant::now();
    let solver_hofft = PoissonND::<5>::new(
        [1.0; 5],
        [0.0; 5],
        [n; 5],
        [r; 5],
        [BoundaryCondition::Periodic; 5],
    );
    let hofft_setup_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Setup time : {hofft_setup_ms:.1} ms");

    let t = Instant::now();
    let x_hofft = solver_hofft.get_x();
    let mut f_hofft: Vec<rustfft::num_complex::Complex<f64>> = (0..ndof_hofft)
        .into_par_iter()
        .map(|i| {
            let [x0, x1, x2, x3, x4] = x_hofft[i];
            let u = u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4);
            rustfft::num_complex::Complex::new(
                alpha * u
                    - beta
                        * (u1d_pp(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d_pp(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d_pp(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d(x2) * u1d_pp(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d_pp(x4)),
                0.0,
            )
        })
        .collect();
    let hofft_rhs_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  RHS build  : {hofft_rhs_ms:.1} ms");

    let t = Instant::now();
    solver_hofft.solve_in_place(&mut f_hofft, alpha, beta);
    let hofft_solve_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Solve time : {hofft_solve_ms:.1} ms");

    let hofft_err = (0..ndof_hofft)
        .map(|i| {
            let [x0, x1, x2, x3, x4] = x_hofft[i];
            let u_ex = u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4);
            (f_hofft[i].re - u_ex).abs()
        })
        .fold(0.0_f64, f64::max);
    println!("  Max error  : {hofft_err:.3e}");

    let t = Instant::now();
    let mut hofft_vals =
        vec![rustfft::num_complex::Complex::new(0.0_f64, 0.0); eval_hofft_pts.len()];
    solver_hofft
        .grid()
        .eval(&eval_hofft_pts, &f_hofft, &mut hofft_vals);
    let hofft_eval_err = eval_hofft_pts
        .iter()
        .zip(hofft_vals.iter())
        .map(|(&[p0, p1, p2, p3, p4], v)| {
            (v.re - u1d(p0) * u1d(p1) * u1d(p2) * u1d(p3) * u1d(p4)).abs()
        })
        .fold(0.0_f64, f64::max);
    let hofft_eval_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Eval error : {hofft_eval_err:.3e}  ({hofft_eval_ms:.2} ms for 1000000 pts)");
    println!(
        "  Total time : {:.1} ms",
        hofft_setup_ms + hofft_rhs_ms + hofft_solve_ms
    );
    println!();

    // ── HoFFT-1 (n=1, r=r_hot — equivalent to HoT) ───────────────────────────
    println!("── HoFFT-1 (n=1, r={r_hot}) ────────────────────────────────────────");

    let t = Instant::now();
    let solver_hofft1 = PoissonND::<5>::new(
        [1.0; 5],
        [0.0; 5],
        [1; 5],
        [r_hot; 5],
        [BoundaryCondition::Periodic; 5],
    );
    let hofft1_setup_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Setup time : {hofft1_setup_ms:.1} ms");

    let t = Instant::now();
    let x_hofft1 = solver_hofft1.get_x();
    let mut f_hofft1: Vec<rustfft::num_complex::Complex<f64>> = (0..ndof_1)
        .into_par_iter()
        .map(|i| {
            let [x0, x1, x2, x3, x4] = x_hofft1[i];
            let u = u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4);
            rustfft::num_complex::Complex::new(
                alpha * u
                    - beta
                        * (u1d_pp(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d_pp(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d_pp(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d(x2) * u1d_pp(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d_pp(x4)),
                0.0,
            )
        })
        .collect();
    let hofft1_rhs_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  RHS build  : {hofft1_rhs_ms:.1} ms");

    let t = Instant::now();
    solver_hofft1.solve_in_place(&mut f_hofft1, alpha, beta);
    let hofft1_solve_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Solve time : {hofft1_solve_ms:.1} ms");

    let hofft1_err = (0..ndof_1)
        .map(|i| {
            let [x0, x1, x2, x3, x4] = x_hofft1[i];
            let u_ex = u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4);
            (f_hofft1[i].re - u_ex).abs()
        })
        .fold(0.0_f64, f64::max);
    println!("  Max error  : {hofft1_err:.3e}");

    let t = Instant::now();
    let mut hofft1_vals =
        vec![rustfft::num_complex::Complex::new(0.0_f64, 0.0); eval_hot_pts.len()];
    solver_hofft1
        .grid()
        .eval(&eval_hot_pts, &f_hofft1, &mut hofft1_vals);
    let hofft1_eval_err = eval_hot_pts
        .iter()
        .zip(hofft1_vals.iter())
        .map(|(&[p0, p1, p2, p3, p4], v)| {
            (v.re - u1d(p0) * u1d(p1) * u1d(p2) * u1d(p3) * u1d(p4)).abs()
        })
        .fold(0.0_f64, f64::max);
    let hofft1_eval_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Eval error : {hofft1_eval_err:.3e}  ({hofft1_eval_ms:.2} ms for 1000 pts)");
    println!(
        "  Total time : {:.1} ms",
        hofft1_setup_ms + hofft1_rhs_ms + hofft1_solve_ms
    );
    println!();

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("── Summary ────────────────────────────────────────────────────────");
    println!(
        "  {:>10}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>14}",
        "method", "setup (ms)", "rhs (ms)", "solve (ms)", "max err", "eval err", "eval (ms/1000)"
    );
    println!(
        "  {:->10}  {:->12}  {:->12}  {:->12}  {:->12}  {:->12}  {:->14}",
        "", "", "", "", "", "", ""
    );
    println!(
        "  {:>10}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.3e}  {:>12.3e}  {:>14.2}",
        "HoFFT",
        hofft_setup_ms,
        hofft_rhs_ms,
        hofft_solve_ms,
        hofft_err,
        hofft_eval_err,
        hofft_eval_ms / 1000.0
    );
    println!(
        "  {:>10}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.3e}  {:>12.3e}  {:>14.2}",
        "HoFFT-1",
        hofft1_setup_ms,
        hofft1_rhs_ms,
        hofft1_solve_ms,
        hofft1_err,
        hofft1_eval_err,
        hofft1_eval_ms
    );
}
