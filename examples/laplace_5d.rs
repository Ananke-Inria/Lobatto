//! Example: solve the 5-D periodic Laplace problem
//!
//!   (α M + β K) u = f   on [0, 1]^5
//!
//! using two methods with the same number of DOFs per direction:
//!   - HoFFT: n=8 elements × r=5 polynomial order  → 40 DOF/direction
//!   - HOT:   r_hot=40 (single element, spectral)   → 40 DOF/direction
//!
//! Exact solution: u = exp(sin(2π x_1)) · … · exp(sin(2π x_5))
//!
//! Run with:
//!   cargo run --example laplace_5d --release

use gg_lobatto::hofft::HoFFTNDLaplaceSolver;
use gg_lobatto::hot::HotNDLaplaceSolver;
use nalgebra::DVector;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
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

    // HoFFT parameters
    let n = 8; // elements per direction
    let r = 5; // polynomial order per direction
    let ndof_1d: usize = n * r;

    // HOT parameters: same DOF/direction
    let r_hot = ndof_1d;

    let ndof_hofft: usize = ndof_1d.pow(5);
    let ndof_hot: usize = r_hot.pow(5);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║      5-D Laplace solver comparison: HoFFT vs HoT         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  DOF per direction : {ndof_1d}  (HoFFT: n={n}×r={r},  HOT: r_hot={r_hot})");
    println!("  Total DOF         : {ndof_hofft}");
    println!();

    // ── HoFFT ─────────────────────────────────────────────────────────────────
    println!("── HoFFT (n={n}, r={r}) ──────────────────────────────────────────");

    let t = Instant::now();
    let solver_hofft = HoFFTNDLaplaceSolver::<5>::new([1.0; 5], [n; 5], [r; 5], alpha, beta);
    let hofft_setup_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Setup time : {hofft_setup_ms:.1} ms");

    let t = Instant::now();
    let x_hofft = solver_hofft.get_x();
    let mut f_hofft = DVector::from_vec(
        (0..ndof_hofft)
            .into_par_iter()
            .map(|i| {
                let (x0, x1, x2, x3, x4) = (
                    x_hofft[0][i],
                    x_hofft[1][i],
                    x_hofft[2][i],
                    x_hofft[3][i],
                    x_hofft[4][i],
                );
                let u = u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4);
                alpha * u
                    - beta
                        * (u1d_pp(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d_pp(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d_pp(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d(x2) * u1d_pp(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d_pp(x4))
            })
            .collect::<Vec<_>>(),
    );
    let hofft_rhs_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  RHS build  : {hofft_rhs_ms:.1} ms");

    let t = Instant::now();
    solver_hofft.solve_in_place(&mut f_hofft);
    let hofft_solve_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Solve time : {hofft_solve_ms:.1} ms");

    let hofft_err = (0..ndof_hofft)
        .map(|i| {
            let u_ex = u1d(x_hofft[0][i])
                * u1d(x_hofft[1][i])
                * u1d(x_hofft[2][i])
                * u1d(x_hofft[3][i])
                * u1d(x_hofft[4][i]);
            (f_hofft[i] - u_ex).abs()
        })
        .fold(0.0_f64, f64::max);
    println!("  Max error  : {hofft_err:.3e}");
    println!(
        "  Total time : {:.1} ms",
        hofft_setup_ms + hofft_rhs_ms + hofft_solve_ms
    );
    println!();

    // ── HoT ───────────────────────────────────────────────────────────────────
    println!("── HoT (r_hot={r_hot}, single element) ────────────────────────────");

    let t = Instant::now();
    let solver_hot = HotNDLaplaceSolver::<5>::new([1.0; 5], [r_hot; 5], alpha, beta);
    let hot_setup_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Setup time : {hot_setup_ms:.1} ms");

    let t = Instant::now();
    let x_hot = solver_hot.get_x();
    let mut f_hot = DVector::from_vec(
        (0..ndof_hot)
            .into_par_iter()
            .map(|i| {
                let (x0, x1, x2, x3, x4) = (
                    x_hot[0][i],
                    x_hot[1][i],
                    x_hot[2][i],
                    x_hot[3][i],
                    x_hot[4][i],
                );
                let u = u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4);
                alpha * u
                    - beta
                        * (u1d_pp(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d_pp(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d_pp(x2) * u1d(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d(x2) * u1d_pp(x3) * u1d(x4)
                            + u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d_pp(x4))
            })
            .collect::<Vec<_>>(),
    );
    let hot_rhs_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  RHS build  : {hot_rhs_ms:.1} ms");

    let t = Instant::now();
    solver_hot.solve_in_place(&mut f_hot);
    let hot_solve_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Solve time : {hot_solve_ms:.1} ms");

    let hot_err = (0..ndof_hot)
        .map(|i| {
            let u_ex = u1d(x_hot[0][i])
                * u1d(x_hot[1][i])
                * u1d(x_hot[2][i])
                * u1d(x_hot[3][i])
                * u1d(x_hot[4][i]);
            (f_hot[i] - u_ex).abs()
        })
        .fold(0.0_f64, f64::max);
    println!("  Max error  : {hot_err:.3e}");
    println!(
        "  Total time : {:.1} ms",
        hot_setup_ms + hot_rhs_ms + hot_solve_ms
    );
    println!();

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("── Summary ────────────────────────────────────────────────────────");
    println!(
        "  {:>10}  {:>12}  {:>12}  {:>12}  {:>12}",
        "method", "setup (ms)", "rhs (ms)", "solve (ms)", "max err"
    );
    println!("  {:->10}  {:->12}  {:->12}  {:->12}  {:->12}", "", "", "", "", "");
    println!(
        "  {:>10}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.3e}",
        "HoFFT", hofft_setup_ms, hofft_rhs_ms, hofft_solve_ms, hofft_err
    );
    println!(
        "  {:>10}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.3e}",
        "HoT", hot_setup_ms, hot_rhs_ms, hot_solve_ms, hot_err
    );
}
