//! Example: solve the 5-D periodic Laplace problem
//!
//!   (α M + β K) u = f   on [0, 1]^5
//!
//! with known exact solution  u = ∏_{d=1}^{5} sin(2π x_d).
//!
//! Run with:
//!   cargo run --example laplace_5d --release

use gg_lobatto::hofft::HoFFTNDLaplaceSolver;
use nalgebra::DVector;
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    // Problem parameters
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let n = 8; // elements per direction
    let r = 5; // polynomial order per direction

    let w = 2.0 * PI;
    // u = ∏ sin(w xd)  =>  -Δu = 5w² u  =>  (α + 5βw²) u = rhs
    let coeff = alpha + 5.0 * beta * w * w;
    let u1d = |x: f64| (w * x).sin();

    let ndof_1d: usize = n * r;
    let ndof = ndof_1d.pow(5);

    println!("=== 5-D HoFFT Laplace solver ===");
    println!("  n = {n}  (elements per direction)");
    println!("  r = {r}  (polynomial order)");
    println!("  DOF per direction = {ndof_1d}");
    println!("  Total DOF         = {ndof}");
    println!();

    // ── Solver setup ──────────────────────────────────────────────────────────
    let t_setup = Instant::now();
    let solver = HoFFTNDLaplaceSolver::<5>::new([1.0; 5], [n; 5], [r; 5], alpha, beta);
    let setup_ms = t_setup.elapsed().as_secs_f64() * 1e3;
    println!("  Setup time : {setup_ms:.1} ms");

    // ── Build right-hand side ─────────────────────────────────────────────────
    let t_rhs = Instant::now();
    let x = solver.get_x();
    let mut f = DVector::<f64>::from_fn(ndof, |i, _| {
        coeff * u1d(x[0][i]) * u1d(x[1][i]) * u1d(x[2][i]) * u1d(x[3][i]) * u1d(x[4][i])
    });
    let rhs_ms = t_rhs.elapsed().as_secs_f64() * 1e3;
    println!("  RHS build  : {rhs_ms:.1} ms");

    // ── Solve ─────────────────────────────────────────────────────────────────
    let t_solve = Instant::now();
    solver.solve_in_place(&mut f);
    let solve_ms = t_solve.elapsed().as_secs_f64() * 1e3;
    println!("  Solve time : {solve_ms:.1} ms");

    // ── Error ─────────────────────────────────────────────────────────────────
    let max_err = (0..ndof)
        .map(|i| {
            let u_ex = u1d(x[0][i]) * u1d(x[1][i]) * u1d(x[2][i]) * u1d(x[3][i]) * u1d(x[4][i]);
            (f[i] - u_ex).abs()
        })
        .fold(0.0_f64, f64::max);

    println!();
    println!("  Max error  : {max_err:.3e}");
    println!("  Total time : {:.1} ms", setup_ms + rhs_ms + solve_ms);
}
