use gg_lobatto::hofft::*;
use nalgebra::Complex;
use nalgebra::{DMatrix, DVector};
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
fn test_symbol_eigendecomposition() {
    const ORTHO_TOL: f64 = 1e-10;
    const RECON_TOL: f64 = 1e-10;
    const PINV_TOL: f64 = 1e-8;

    for &r in &[1, 2, 3, 4, 5, 6] {
        for &n in &[4, 16, 32, 64, 128] {
            let solver = HoFFT1DLaplaceSolver::new(1.0, n, r, 1.0, 1.0);
            let id = DMatrix::<Complex<f64>>::identity(r, r);

            // M and M^-1 as complex diagonal matrices
            let m_c = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                Complex::new(solver.periodic_mass_matrix[i], 0.0)
            }));

            let m_inv_c = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                Complex::new(1.0 / solver.periodic_mass_matrix[i], 0.0)
            }));

            for k in 0..n {
                // V : physical eigenvectors (M-orthonormal)
                let v = &solver.eigenvectors[k];

                // 1. V is M-orthonormal: V† M V = I
                let vtmv = v.adjoint() * &m_c * v;
                let ortho_err = (&vtmv - &id).norm();
                assert!(
                    ortho_err < ORTHO_TOL,
                    "r={r} n={n} k={k}: eigenvectors not M-orthonormal (err={ortho_err:.2e})"
                );

                // 2. V V† = M^{-1}
                let vvt = v * v.adjoint();
                let vvt_err = (&vvt - &m_inv_c).norm();
                assert!(
                    vvt_err < ORTHO_TOL,
                    "r={r} n={n} k={k}: V V† ≠ M⁻¹ (err={vvt_err:.2e})"
                );

                // 3. Generalized eigenvalue problem: symbol * V = M * V * Λ
                let lambda_diag = DMatrix::from_diagonal(&DVector::from_fn(r, |i, _| {
                    Complex::new(solver.eigenvalues[k][i], 0.0)
                }));
                let lhs = &solver.symbol[k] * v;
                let rhs = &m_c * v * &lambda_diag;
                let recon_err = (&lhs - &rhs).norm() / rhs.norm();
                assert!(
                    recon_err < RECON_TOL,
                    "r={r} n={n} k={k}: symbol*V ≠ M*V*Λ (relative err={recon_err:.2e})"
                );

                // 4. Pseudo-inverse: symbol * symbol_pseudo_inv = I (when alpha = 1 the pseudo-inverse is the inverse)
                let pinv_prod = &solver.symbol[k] * &solver.symbol_pseudo_inv[k];
                let pinv_err = (&pinv_prod - &id).norm();
                assert!(
                    pinv_err < PINV_TOL,
                    "r={r} n={n} k={k}: symbol * symbol_pseudo_inv ≠ I (err={pinv_err:.2e})"
                );
            }
        }
    }
}

#[test]
fn test_nd_fft_roundtrip() {
    const TOL: f64 = 1e-12;

    // 1D
    let solver = HoFFTNDLaplaceSolver::<1>::new([1.0], [8], [3], 1.0, 0.0);
    let x = solver.get_x();
    let f_orig: DVector<f64> = DVector::from_fn(x[0].len(), |i, _| (x[0][i] * PI).sin());
    let mut f = f_orig.clone();
    solver.solve_in_place(&mut f);
    let err: f64 = (&f - &f_orig).amax();
    assert!(err < TOL, "1D round-trip error = {err:.2e}");

    // 2D
    let solver = HoFFTNDLaplaceSolver::<2>::new([1.0; 2], [4; 2], [2; 2], 1.0, 0.0);
    let x = solver.get_x();
    let total = x[0].len();
    let f_orig: DVector<f64> =
        DVector::from_fn(total, |i, _| (x[0][i] * PI).sin() * (x[1][i] * PI).cos());
    let mut f = f_orig.clone();
    solver.solve_in_place(&mut f);
    let err: f64 = (&f - &f_orig).amax();
    assert!(err < TOL, "2D round-trip error = {err:.2e}");

    // 3D
    let solver = HoFFTNDLaplaceSolver::<3>::new([1.0; 3], [4; 3], [2; 3], 1.0, 0.0);
    let x = solver.get_x();
    let total = x[0].len();
    let f_orig: DVector<f64> = DVector::from_fn(total, |i, _| {
        (x[0][i] * PI).sin() * (x[1][i] * PI).cos() * (x[2][i] * 2.0 * PI).sin()
    });
    let mut f = f_orig.clone();
    solver.solve_in_place(&mut f);
    let err: f64 = (&f - &f_orig).amax();
    assert!(err < TOL, "3D round-trip error = {err:.2e}");
}

#[test]
fn convergence_1d() {
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
                let err = compute_error(l, n, r, alpha, beta, exact_u.as_ref(), rhs_f.as_ref());

                let rate = match (prev_h, prev_err) {
                    (Some(ph), Some(pe)) if pe > 0.0 && err > 0.0 => {
                        let rate_val = (pe / err).ln() / (ph / h).ln();
                        assert!(
                            err < 1e-8 || rate_val >= (r as f64 + 1.0) - 0.1,
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

    let orders = [1, 2, 3, 4, 5];
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
            let solver = HoFFTNDLaplaceSolver::<2>::new([1.0; 2], [n; 2], [r; 2], alpha, beta);
            let x = solver.get_x();
            let total = x[0].len();

            let mut f = DVector::<f64>::from_fn(total, |i, _| rhs_f(x[0][i], x[1][i]));
            solver.solve_in_place(&mut f);

            let max_err = (0..total)
                .map(|i| (f[i] - exact_u(x[0][i], x[1][i])).abs())
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
fn convergence_2d_anisotropic() {
    let alpha = 1.0;
    let beta = 1.0;
    let w = 2.0 * PI;

    // u(x,y) = sin(w x)·cos(w y),  (α - β Δ)u = (α + 2β w²)·u
    let c = alpha + 2.0 * beta * w * w;
    let exact_u = |x: f64, y: f64| (w * x).sin() * (w * y).cos();
    let rhs_f = |x: f64, y: f64| c * exact_u(x, y);

    let compute_err = |ns: [usize; 2], rs: [usize; 2]| -> f64 {
        let solver = HoFFTNDLaplaceSolver::<2>::new([1.0; 2], ns, rs, alpha, beta);
        let x = solver.get_x();
        let total = x[0].len();
        let mut f = DVector::<f64>::from_fn(total, |i, _| rhs_f(x[0][i], x[1][i]));
        solver.solve_in_place(&mut f);
        (0..total)
            .map(|i| (f[i] - exact_u(x[0][i], x[1][i])).abs())
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

    println!("\n{:=<70}", "");
    println!("3D convergence: (alpha*M + beta*K) u = f  on [0,1]³");
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
                let solver = HoFFTNDLaplaceSolver::<3>::new([1.0; 3], [n; 3], [r; 3], alpha, beta);
                let x = solver.get_x();
                let total = x[0].len();

                let mut f = DVector::<f64>::from_fn(total, |i, _| {
                    let (xi, yi, zi) = (x[0][i], x[1][i], x[2][i]);
                    let (ux, uy, uz) = (u1d(xi), u1d(yi), u1d(zi));
                    let (uxx, uyy, uzz) = (u1d_pp(xi), u1d_pp(yi), u1d_pp(zi));
                    alpha * ux * uy * uz - beta * (uxx * uy * uz + ux * uyy * uz + ux * uy * uzz)
                });
                solver.solve_in_place(&mut f);

                let max_err = (0..total)
                    .map(|i| (f[i] - u1d(x[0][i]) * u1d(x[1][i]) * u1d(x[2][i])).abs())
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

/// Check that with beta=0 the tensorial structure recovers the inverse of the local mass matrix.
///
/// V_ND = V_0^{(0)} ⊗ ... ⊗ V_{N-1}^{(0)} (eigenvectors at k=0 for each direction)
/// satisfies V_ND * V_ND† = tilde_M_ND^{-1} = diag(1 / periodic_mass_matrix).
#[test]
fn test_nd_mass_matrix_inverse_tensorial() {
    const TOL: f64 = 1e-12;

    // Helper: build Kronecker product of two matrices A ⊗ B
    let kron = |a: &DMatrix<Complex<f64>>, b: &DMatrix<Complex<f64>>| {
        let (ra, ca) = (a.nrows(), a.ncols());
        let (rb, cb) = (b.nrows(), b.ncols());
        DMatrix::from_fn(ra * rb, ca * cb, |i, j| {
            a[(i / rb, j / cb)] * b[(i % rb, j % cb)]
        })
    };

    // 1D
    {
        let solver = HoFFTNDLaplaceSolver::<1>::new([1.0], [4], [3], 1.0, 0.0);
        let v = &solver.solvers[0].eigenvectors[0];
        let vvt = v * v.adjoint();
        let r = v.nrows();
        for i in 0..r {
            for j in 0..r {
                let expected = if i == j {
                    Complex::new(1.0 / solver.solvers[0].periodic_mass_matrix[i], 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                assert!((vvt[(i, j)] - expected).norm() < TOL, "1D [{i},{j}]");
            }
        }
    }

    // 2D: V_ND = V_0 ⊗ V_1
    {
        let solver = HoFFTNDLaplaceSolver::<2>::new([1.0; 2], [4; 2], [2; 2], 1.0, 0.0);
        let v_nd = kron(
            &solver.solvers[0].eigenvectors[0],
            &solver.solvers[1].eigenvectors[0],
        );
        let vvt = &v_nd * v_nd.adjoint();
        let r_total = v_nd.nrows();
        for i in 0..r_total {
            for j in 0..r_total {
                let expected = if i == j {
                    Complex::new(1.0 / solver.periodic_mass_matrix[i], 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                assert!((vvt[(i, j)] - expected).norm() < TOL, "2D [{i},{j}]");
            }
        }
    }

    // 3D: V_ND = V_0 ⊗ V_1 ⊗ V_2
    {
        let solver = HoFFTNDLaplaceSolver::<3>::new([1.0; 3], [4; 3], [2; 3], 1.0, 0.0);
        let v_nd = kron(
            &kron(
                &solver.solvers[0].eigenvectors[0],
                &solver.solvers[1].eigenvectors[0],
            ),
            &solver.solvers[2].eigenvectors[0],
        );
        let vvt = &v_nd * v_nd.adjoint();
        let r_total = v_nd.nrows();
        for i in 0..r_total {
            for j in 0..r_total {
                let expected = if i == j {
                    Complex::new(1.0 / solver.periodic_mass_matrix[i], 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                let scale = expected.norm().max(1.0);
                assert!(
                    (vvt[(i, j)] - expected).norm() / scale < TOL,
                    "3D [{i},{j}]"
                );
            }
        }
    }
}
