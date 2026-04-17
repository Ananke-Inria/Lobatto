//! Example: solve the 5-D periodic Laplace problem
//!
//!   (α M + β K) u = f   on [0, 1]^5
//!
//! using the HoT solver (single spectral element per direction).
//!
//! Exact solution: u = exp(sin(2π x_1)) · … · exp(sin(2π x_5))
//!
//! Run with:
//!   cargo run --example laplace_5d --release

use lobatto_spectral::solver::{BoundaryCondition, PoissonND};
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

    // 1000 deterministic off-grid evaluation points in [0,1)^5
    let eval_pts: Vec<[f64; 5]> = (0..1000)
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

    let r = 20_usize;
    let ndof: usize = r.pow(5);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         5-D Laplace solver: HoT                          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  HoT : r={r} → {r} DOF/direction, {ndof} total");
    println!();

    println!("── HoT (r={r}, single element) ──────────────────────────────");

    let t = Instant::now();
    let solver = PoissonND::<5>::new(
        [1.0; 5],
        [0.0; 5],
        [r; 5],
        alpha,
        beta,
        [BoundaryCondition::Periodic; 5],
    );
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Setup time : {setup_ms:.1} ms");

    let t = Instant::now();
    let x = solver.get_x();
    let mut f: Vec<f64> = (0..ndof)
        .into_par_iter()
        .map(|i| {
            let [x0, x1, x2, x3, x4] = x[i];
            let u = u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4);
            alpha * u
                - beta
                    * (u1d_pp(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                        + u1d(x0) * u1d_pp(x1) * u1d(x2) * u1d(x3) * u1d(x4)
                        + u1d(x0) * u1d(x1) * u1d_pp(x2) * u1d(x3) * u1d(x4)
                        + u1d(x0) * u1d(x1) * u1d(x2) * u1d_pp(x3) * u1d(x4)
                        + u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d_pp(x4))
        })
        .collect();
    let rhs_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  RHS build  : {rhs_ms:.1} ms");

    let t = Instant::now();
    solver.solve_in_place(&mut f);
    let solve_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Solve time : {solve_ms:.1} ms");

    let node_err = (0..ndof)
        .map(|i| {
            let [x0, x1, x2, x3, x4] = x[i];
            let u_ex = u1d(x0) * u1d(x1) * u1d(x2) * u1d(x3) * u1d(x4);
            (f[i] - u_ex).abs()
        })
        .fold(0.0_f64, f64::max);
    println!("  Max error  : {node_err:.3e}");

    let t = Instant::now();
    let vals = solver.eval(&eval_pts, &f);
    let eval_err = eval_pts
        .iter()
        .zip(vals.iter())
        .map(|(&[p0, p1, p2, p3, p4], &v)| {
            (v - u1d(p0) * u1d(p1) * u1d(p2) * u1d(p3) * u1d(p4)).abs()
        })
        .fold(0.0_f64, f64::max);
    let eval_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  Eval error : {eval_err:.3e}  ({eval_ms:.2} ms for 1000 pts)");
    println!("  Total time : {:.1} ms", setup_ms + rhs_ms + solve_ms);
}
