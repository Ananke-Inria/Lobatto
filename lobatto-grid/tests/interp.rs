//! Tests for the interpolation routines:
//! `eval`, `eval_extended`, `convect`, and the precomputed `EvalPlan`.

use lobatto_grid::{Extension, Grid1D, GridND};
use num_complex::Complex;
use std::f64::consts::PI;
use std::time::Instant;

/// Shared 1D interpolation-convergence driver: sample `u` at the grid nodes
/// and measure the max interpolation error over 100 off-grid points.
///
/// Nodal interpolation of a smooth function converges at rate ~r+1; the
/// assertion requires rate ≥ r−1 (or error already at rounding level).
fn convergence_1d(ext: Extension, label: &str, l: f64, xl: f64, u: impl Fn(f64) -> f64) {
    let r = 5_usize;
    let n_values = [4, 8, 16, 32, 64, 128];
    let exact = |x: f64| u((x - xl) / l);
    let pts: Vec<f64> = (0..100)
        .map(|i| xl + (7.0 * PI * (i as f64 + 0.5) / 100.0).fract() * l)
        .collect();

    println!("\n{:=<55}", "");
    println!("1D eval convergence — {label} (r={r}), 100 off-grid pts");
    println!("domain: [{xl}, {}]", xl + l);
    println!("{:=<55}", "");
    println!("{:>8} {:>14} {:>10}", "n", "max_err_eval", "rate");
    println!("{:-<36}", "");

    let (mut prev_h, mut prev_err) = (None::<f64>, None::<f64>);
    for &n in &n_values {
        let h = l / n as f64;
        let grid = Grid1D::new(n, r, l, xl, ext);
        let f: Vec<f64> = grid.get_x().iter().map(|&xi| exact(xi)).collect();

        let max_err = pts
            .iter()
            .map(|&p| (grid.eval_at(p, &f) - exact(p)).abs())
            .fold(0.0_f64, f64::max);

        let rate = match (prev_h, prev_err) {
            (Some(ph), Some(pe)) if pe > 0.0 && max_err > 0.0 => {
                let rv = (pe / max_err).ln() / (ph / h).ln();
                assert!(
                    max_err < 1e-8 || rv >= r as f64 - 1.0,
                    "1D eval {label} r={r} n={n}: rate {rv:.2} < {}, err={max_err:.2e}",
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

/// 1D Periodic eval convergence: u(t) = exp(sin(2π·t)) on [0.7, 2.2].
#[test]
fn convergence_periodic_1d_eval() {
    let w = 2.0 * PI;
    convergence_1d(Extension::Periodic, "Periodic", 1.5, 0.7, |t| {
        (w * t).sin().exp()
    });
}

/// 1D Dirichlet eval convergence: u(t) = exp(sin(π·t)) − 1, u(0) = u(1) = 0,
/// on [0.3, 2.8].
#[test]
fn convergence_dirichlet_1d_eval() {
    let w = PI;
    convergence_1d(Extension::Odd, "Dirichlet", 2.5, 0.3, |t| {
        (w * t).sin().exp() - 1.0
    });
}

/// 1D Neumann eval convergence: u(t) = exp(cos(π·t)), u'(0) = u'(1) = 0,
/// on [0.1, 0.95].
#[test]
fn convergence_neumann_1d_eval() {
    let w = PI;
    convergence_1d(Extension::Even, "Neumann", 0.85, 0.1, |t| {
        (w * t).cos().exp()
    });
}

/// 3D interpolation convergence measured at 100 deterministic off-grid points
/// via `eval`.
///
/// Checks h-convergence (r=5 fixed, n increasing) for the nodal interpolant of
/// a smooth tensor-product function.  Error is the max over 100 pseudo-random
/// points that do NOT coincide with any Gauss-Lobatto node.
#[test]
fn convergence_3d_eval() {
    let w = 2.0 * PI;

    let ls = [1.0_f64, 2.0, 3.0];
    let xls = [0.0_f64, 1.0, -1.0];

    let u1d = |x: f64| (w * x).sin().exp();
    let exact_u = |x: f64, y: f64, z: f64| {
        u1d((x - xls[0]) / ls[0]) * u1d((y - xls[1]) / ls[1]) * u1d((z - xls[2]) / ls[2])
    };

    // 100 deterministic points spread across the domain, well-separated from Gauss-Lobatto nodes.
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
    let n_values = [4, 8, 16, 32, 64];

    println!("\n{:=<55}", "");
    println!("3D eval convergence (r={r}), 100 off-grid points");
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
        let grid = GridND::<3>::new([n; 3], [r; 3], ls, xls, [Extension::Periodic; 3]);
        let x = grid.get_x();

        let f: Vec<f64> = x.iter().map(|p| exact_u(p[0], p[1], p[2])).collect();

        let mut vals = vec![0.0_f64; pts.len()];
        grid.eval(&pts, &f, &mut vals);
        let max_err = pts
            .iter()
            .zip(vals.iter())
            .map(|(pt, v)| (v - exact_u(pt[0], pt[1], pt[2])).abs())
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
    // Q6 elements (r=6), 6 elements per direction, 4D periodic domain [0,1]^4.
    //
    // u     = Q3 function  → exactly represented in Q6 space
    // φ_d   = Q2 shift vanishing at boundaries: φ(t) = t + 0.3·t·(1−t)
    //         φ(0)=0, φ(1)=1, monotone on [0,1]  → periodicity preserved
    // u∘φ   = Q3 ∘ Q2 = Q6  → also exactly represented
    //
    // Expected max error: machine precision (~1e-13).
    let r = 6;
    let n = 6;

    let grid = GridND::<4>::new([n; 4], [r; 4], [1.0; 4], [0.0; 4], [Extension::Periodic; 4]);
    let x = grid.get_x();
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
    let mut result = vec![Complex::new(0.0_f64, 0.0); u_cx.len()];
    grid.convect(lambda, &u_cx, &mut result);
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

// ── eval domain guard and eval_extended ───────────────────────────────────────

mod eval_extended {
    use lobatto_grid::{Extension, Grid1D, GridND};
    use num_complex::Complex;
    use std::f64::consts::PI;

    /// Reference fold: map `x` into `[xl, xl+l]` per the extension, with the
    /// sign of the extended interpolant (−1 on odd-reflected copies).
    fn fold(x: f64, xl: f64, l: f64, ext: Extension) -> (f64, f64) {
        match ext {
            Extension::Periodic => ((x - xl).rem_euclid(l) + xl, 1.0),
            Extension::Even | Extension::Odd => {
                let t = (x - xl).rem_euclid(2.0 * l);
                if t <= l {
                    (xl + t, 1.0)
                } else {
                    let s = if matches!(ext, Extension::Odd) { -1.0 } else { 1.0 };
                    (xl + (2.0 * l - t), s)
                }
            }
        }
    }

    fn nodal_u(grid: &Grid1D, xl: f64, l: f64) -> Vec<Complex<f64>> {
        grid.get_x()
            .iter()
            .map(|&x| Complex::new((1.7 * PI * (x - xl) / l).sin().exp(), 0.0))
            .collect()
    }

    #[test]
    fn extended_1d_matches_folded_eval() {
        let (n, r, l, xl) = (8_usize, 4_usize, 2.5_f64, -1.2_f64);
        for ext in [Extension::Periodic, Extension::Even, Extension::Odd] {
            let grid = Grid1D::new(n, r, l, xl, ext);
            let u = nodal_u(&grid, xl, l);
            for i in 0..200 {
                let x = -9.0 + 0.093 * i as f64; // well outside [-1.2, 1.3]
                let (y, sign) = fold(x, xl, l, ext);
                let got = grid.eval_extended_at(x, &u);
                let expected = grid.eval_at(y, &u) * sign;
                assert!(
                    (got - expected).norm() < 1e-12,
                    "{ext:?}: eval_extended({x}) = {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn eval_1d_at_exact_boundaries() {
        let (n, r, l, xl) = (8_usize, 4_usize, 2.5_f64, -1.2_f64);
        for ext in [Extension::Periodic, Extension::Even, Extension::Odd] {
            let grid = Grid1D::new(n, r, l, xl, ext);
            let u = nodal_u(&grid, xl, l);
            // Both endpoints are in-domain and must not panic or blow up.
            for x in [xl, xl + l] {
                let v = grid.eval_at(x, &u);
                assert!(v.norm() < 1e3, "{ext:?}: eval({x}) = {v} blew up");
            }
        }
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn eval_1d_out_of_domain_panics() {
        let grid = Grid1D::new(8, 4, 2.5, -1.2, Extension::Even);
        let u = nodal_u(&grid, -1.2, 2.5);
        grid.eval_at(1.31, &u);
    }

    #[test]
    fn extended_nd_matches_folded_eval() {
        let (n, r) = (6_usize, 4_usize);
        let ls = [2.0_f64, 3.0];
        let xls = [-1.0_f64, 0.5];
        let exts = [Extension::Even, Extension::Odd];
        let grid = GridND::<2>::new([n; 2], [r; 2], ls, xls, exts);
        let u: Vec<Complex<f64>> = grid
            .get_x()
            .iter()
            .map(|p| {
                let s = (1.3 * PI * (p[0] - xls[0]) / ls[0]).cos()
                    * (0.8 * PI * (p[1] - xls[1]) / ls[1]).sin();
                Complex::new(s, 0.0)
            })
            .collect();

        // Query points far outside the domain (as a flow map φ⁻¹ can produce).
        let pts: Vec<[f64; 2]> = (0..100)
            .map(|i| [-20.0 + 0.41 * i as f64, 25.0 - 0.47 * i as f64])
            .collect();
        let (folded, signs): (Vec<[f64; 2]>, Vec<f64>) = pts
            .iter()
            .map(|p| {
                let (y0, s0) = fold(p[0], xls[0], ls[0], exts[0]);
                let (y1, s1) = fold(p[1], xls[1], ls[1], exts[1]);
                ([y0, y1], s0 * s1)
            })
            .unzip();

        let mut got = vec![Complex::new(0.0_f64, 0.0); pts.len()];
        grid.eval_extended(&pts, &u, &mut got);
        let mut expected = vec![Complex::new(0.0_f64, 0.0); pts.len()];
        grid.eval(&folded, &u, &mut expected);

        for i in 0..pts.len() {
            let e = expected[i] * signs[i];
            assert!(
                (got[i] - e).norm() < 1e-12,
                "point {:?}: eval_extended = {}, expected {}",
                pts[i],
                got[i],
                e
            );
        }
    }

    /// Regression: an in-domain point exactly on the right boundary used to be
    /// folded to k=0 while ξ was computed from the raw coordinate (ξ = n),
    /// extrapolating the Lagrange basis far outside the element.
    #[test]
    fn eval_nd_at_exact_right_boundary() {
        let ls = [2.0_f64, 3.0];
        let xls = [-1.0_f64, 0.5];
        let grid = GridND::<2>::new([6; 2], [4; 2], ls, xls, [Extension::Even; 2]);
        let x = grid.get_x();
        let u: Vec<Complex<f64>> = x
            .iter()
            .map(|p| Complex::new((p[0] + 0.3 * p[1]).sin(), 0.0))
            .collect();

        // The top-right corner node is a stored Neumann DOF: eval there must
        // reproduce the nodal value exactly.
        let corner = [xls[0] + ls[0], xls[1] + ls[1]];
        let idx = x
            .iter()
            .position(|p| p[0] == corner[0] && p[1] == corner[1])
            .expect("corner node not found in grid");
        let mut out = vec![Complex::new(0.0_f64, 0.0); 1];
        grid.eval(&[corner], &u, &mut out);
        assert!(
            (out[0] - u[idx]).norm() < 1e-12,
            "corner eval {} != nodal value {}",
            out[0],
            u[idx]
        );
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn eval_nd_out_of_domain_panics() {
        let grid = GridND::<2>::new([6; 2], [4; 2], [2.0, 3.0], [-1.0, 0.5], [Extension::Even; 2]);
        let u = vec![Complex::new(0.0_f64, 0.0); 25 * 25];
        let mut out = vec![Complex::new(0.0_f64, 0.0); 1];
        grid.eval(&[[-1.5, 1.0]], &u, &mut out);
    }
}

mod eval_plan {
    use lobatto_grid::{Extension, GridND};
    use num_complex::Complex;

    /// Mirror of `convect_4d_q6`: u = Q3, φ = Q2 boundary-preserving shift, so
    /// u∘φ = Q6 is exactly represented at r=6. The planned path must reproduce
    /// both the exact values and the direct `convect` output.
    #[test]
    fn convect_plan_4d_matches_convect_and_exact() {
        let r = 6;
        let n = 6;

        let grid = GridND::<4>::new([n; 4], [r; 4], [1.0; 4], [0.0; 4], [Extension::Periodic; 4]);
        let x = grid.get_x();
        let total = x.len();

        let q = |t: f64| t * t * (t - 1.0);
        let u: Vec<Complex<f64>> = (0..total)
            .map(|p| Complex::new((0..4_usize).map(|d| q(x[p][d])).product::<f64>(), 0.0))
            .collect();

        let phi = |t: f64| t + 0.3 * t * (1.0 - t);
        let lambda = |pt: [f64; 4]| pt.map(phi);

        let exact: Vec<f64> = (0..total)
            .map(|p| (0..4_usize).map(|d| q(phi(x[p][d]))).product::<f64>())
            .collect();

        let mut direct = vec![Complex::new(0.0_f64, 0.0); total];
        grid.convect(lambda, &u, &mut direct);

        let plan = grid.convect_plan(lambda);
        let mut planned = vec![Complex::new(0.0_f64, 0.0); total];
        grid.eval_with_plan(&plan, &u, &mut planned);

        let max_err_exact = planned
            .iter()
            .zip(exact.iter())
            .map(|(a, &b)| (a.re - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err_exact < 1e-12,
            "planned convect vs exact: {max_err_exact:.3e}"
        );

        let max_diff = planned
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-14,
            "planned convect vs direct convect: {max_diff:.3e}"
        );
    }

    /// Dirichlet (odd extension) case: the plan precomputes reflection signs and
    /// Dirichlet-zero gather entries; out-of-domain points must match
    /// `eval_extended`, and in-domain points must match `eval`.
    #[test]
    fn extended_plan_2d_dirichlet_matches_eval_extended() {
        let grid = GridND::<2>::new([6; 2], [4; 2], [1.0; 2], [0.0; 2], [Extension::Odd; 2]);
        let x = grid.get_x();

        let u: Vec<Complex<f64>> = x
            .iter()
            .map(|&[a, b]| Complex::new((3.1 * a).sin() * (2.3 * b + 0.4).cos(), 0.0))
            .collect();

        // Deterministic points well outside [0,1]^2 (both reflection parities).
        let pts: Vec<[f64; 2]> = (0..500)
            .map(|i| {
                let t = i as f64 / 499.0;
                [-2.3 + 5.1 * t, 3.4 - 6.2 * t * t]
            })
            .collect();

        let mut direct = vec![Complex::new(0.0_f64, 0.0); pts.len()];
        grid.eval_extended(&pts, &u, &mut direct);

        let plan = grid.eval_extended_plan(&pts);
        let mut planned = vec![Complex::new(0.0_f64, 0.0); pts.len()];
        grid.eval_with_plan(&plan, &u, &mut planned);

        let max_diff = planned
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-14, "planned vs eval_extended: {max_diff:.3e}");

        // In-domain points through the plain `eval_plan` constructor.
        let pts_in: Vec<[f64; 2]> = (0..300)
            .map(|i| {
                let t = i as f64 / 299.0;
                [0.999 * t, 1.0 - 0.998 * t]
            })
            .collect();
        let mut direct_in = vec![Complex::new(0.0_f64, 0.0); pts_in.len()];
        grid.eval(&pts_in, &u, &mut direct_in);

        let plan_in = grid.eval_plan(&pts_in);
        let mut planned_in = vec![Complex::new(0.0_f64, 0.0); pts_in.len()];
        grid.eval_with_plan(&plan_in, &u, &mut planned_in);

        let max_diff_in = planned_in
            .iter()
            .zip(direct_in.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_diff_in < 1e-14, "planned vs eval: {max_diff_in:.3e}");
    }

    /// The f64 path must produce exactly the real part of the Complex path,
    /// including reflection signs (Dirichlet/odd) and through a shared plan.
    #[test]
    fn real_field_matches_complex() {
        let grid = GridND::<2>::new([6; 2], [4; 2], [1.0; 2], [0.0; 2], [Extension::Odd; 2]);
        let x = grid.get_x();

        let u_re: Vec<f64> = x
            .iter()
            .map(|&[a, b]| (3.1 * a).sin() * (2.3 * b + 0.4).cos())
            .collect();
        let u_cx: Vec<Complex<f64>> = u_re.iter().map(|&v| Complex::new(v, 0.0)).collect();

        let lambda = |[a, b]: [f64; 2]| [a + 0.37, b - 1.21];

        // Direct convect, both scalar types.
        let mut out_re = vec![0.0_f64; u_re.len()];
        grid.convect(lambda, &u_re, &mut out_re);
        let mut out_cx = vec![Complex::new(0.0_f64, 0.0); u_cx.len()];
        grid.convect(lambda, &u_cx, &mut out_cx);

        let max_diff = out_re
            .iter()
            .zip(out_cx.iter())
            .map(|(a, b)| (a - b.re).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-14, "real vs complex convect: {max_diff:.3e}");

        // One shared plan applied to both scalar types.
        let plan = grid.convect_plan(lambda);
        let mut plan_re = vec![0.0_f64; u_re.len()];
        grid.eval_with_plan(&plan, &u_re, &mut plan_re);
        let mut plan_cx = vec![Complex::new(0.0_f64, 0.0); u_cx.len()];
        grid.eval_with_plan(&plan, &u_cx, &mut plan_cx);

        let max_diff_plan = plan_re
            .iter()
            .zip(plan_cx.iter())
            .map(|(a, b)| (a - b.re).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff_plan < 1e-14,
            "real vs complex plan apply: {max_diff_plan:.3e}"
        );

        let max_diff_direct = plan_re
            .iter()
            .zip(out_re.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff_direct < 1e-14,
            "planned vs direct real convect: {max_diff_direct:.3e}"
        );
    }
}

// ── one-element periodic shift = permutation ─────────────────────────────────

/// Shifting the flow map by exactly one element width (per direction, periodic
/// setting) sends every DOF node onto another DOF node, so the convected
/// vector must be a permutation of `u` — for *arbitrary* nodal values, not
/// just resolved fields. Checks both `convect` and the plan path.
///
/// The shifted coordinate is not bitwise equal to the target node (floating
/// point), so basis values are 1/0 only up to rounding; tolerance 1e-11.
#[test]
fn convect_one_element_shift_is_permutation() {
    let n = 4;
    let r = 3;
    let ndof_1d = n * r; // 12 per direction, periodic
    let h = 1.0 / n as f64;

    let grid = GridND::<4>::new([n; 4], [r; 4], [1.0; 4], [0.0; 4], [Extension::Periodic; 4]);
    let total = ndof_1d.pow(4);

    // Arbitrary (pseudo-random, deterministic) nodal values.
    let mut state = 88172645463325252_u64;
    let u: Vec<f64> = (0..total)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        })
        .collect();

    // Shift one element to the left: u_out(x) = u(x - h).
    let lambda = |pt: [f64; 4]| pt.map(|t| t - h);

    // Expected: per-direction node index m_d -> (m_d - r) mod ndof_1d.
    let expected: Vec<f64> = (0..total)
        .map(|p| {
            let q = (0..4).fold(0, |acc, d| {
                let stride = ndof_1d.pow(d as u32);
                let m_d = (p / stride) % ndof_1d;
                acc + ((m_d + ndof_1d - r) % ndof_1d) * stride
            });
            u[q]
        })
        .collect();

    let mut direct = vec![0.0_f64; total];
    grid.convect(lambda, &u, &mut direct);
    let max_err = direct
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-11, "convect shift permutation: {max_err:.3e}");

    let plan = grid.convect_plan(lambda);
    let mut planned = vec![0.0_f64; total];
    grid.eval_with_plan(&plan, &u, &mut planned);
    let max_err_plan = planned
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err_plan < 1e-11,
        "planned shift permutation: {max_err_plan:.3e}"
    );
}
