use nalgebra::Complex;
use quadratix::hofft::*;
use std::f64::consts::PI;

// ── Lemma 3.2 symmetry check (Even extension) ────────────────────────────────

/// Verify Lemma 3.2 of Caforio & Imperiale (2019) for the Even extension.
///
/// Setup: n=3 physical elements, r=3 local nodes, Extension::Even.
/// Physical DOFs: n·r+1 = 10, extended FFT length: 2n = 6.
///
/// A complex function f is packed into the 2n-element periodic buffer:
///   k < n : buf[j·n_fft + k] = f[k·r + j]
///   k ≥ n : buf[j·n_fft + k] = f[(2n−k)·r − j]
///
/// Lemma 3.2 predicts that after the forward FFT the full 2n-mode buffer satisfies:
///   j = 0:  F̂_{2n−k, 0}  =  F̂_{k, 0}
///   j > 0:  F̂_{2n−k, j}  =  e^{−iπk/n} · F̂_{k, r−j}
///
///
/// Moreover on can show the following symmetry properties
///
/// j > 0: F̂_{0, r-j}  =  F̂_{0, j}
/// j > 0: F̂_{N, r-j}  =  -F̂_{N, j}
///
/// The test prints the extended physical function, the full FFT coefficients, and
/// the symmetry residuals, then asserts they are below machine-precision tolerance.
#[test]
fn even_extension_symmetry() {
    const TOL: f64 = 1e-10;

    let n: usize = 3;
    let r: usize = 4;
    let l = 1.0_f64;
    let xl = 0.0_f64;

    // Physical planner (Even extension): 10 DOFs, n_fft = 6, n_freq = n+1 = 4.
    let planner = HofftPlanner::new(n, r, l, xl, Extension::Even);
    let x = planner.get_x(); // 10 x-increasing nodes

    // Generic complex function with no special symmetry properties:
    //   Re f(x) = x² − 0.7x + 1.3         (quadratic, not even or odd about any point)
    //   Im f(x) = sin(1.5πx) + 0.4cos(πx)  (non-Neumann: Im f'(0) ≠ 0, Im f'(L) ≠ 0)
    // This prevents accidental Hermitian FFT symmetry, real-FFT structure,
    // or any other property that would make the lemma trivially satisfied.
    let f: Vec<Complex<f64>> = x
        .iter()
        .map(|&xi| {
            Complex::new(
                xi * xi - 0.7 * xi + 1.3,
                (1.5 * PI * xi).sin() + 0.4 * (PI * xi).cos(),
            )
        })
        .collect();

    // ── Step 1: pack the even extension into the full r × n_fft buffer ────────
    //
    // Layout: buf[j * n_fft + k], j ∈ [0,r), k ∈ [0, n_fft=2n).
    // Even packing mirrors the physical DOFs about x = L:
    //   k <  n: buf[j·n_fft + k] = f[k·r + j]
    //   k >= n: buf[j·n_fft + k] = f[(n_fft−k)·r − j]
    let n_fft = 2 * n; // = 4
    let mut buf = vec![Complex::new(0.0_f64, 0.0); r * n_fft];
    for j in 0..r {
        for k in 0..n_fft {
            buf[j * n_fft + k] = if k < n {
                f[k * r + j]
            } else {
                f[(n_fft - k) * r - j]
            };
        }
    }

    // Print physical DOFs and the extended periodic function.
    println!("\n{:=<70}", "");
    println!("Lemma 3.2 check — Even extension  (n={n}, r={r}, n_fft={n_fft})");
    println!("{:=<70}", "");
    println!("\nPhysical DOFs (x-increasing, {} nodes):", x.len());
    println!(
        "{:>8}  {:>12}  {:>22}  {:>22}",
        "idx", "x", "Re f(x)", "Im f(x)"
    );
    for (i, (&xi, fi)) in x.iter().zip(f.iter()).enumerate() {
        println!("{:>8}  {:>12.6}  {:>22.10}  {:>22.10}", i, xi, fi.re, fi.im);
    }

    println!("\nEven-extended periodic buffer (mode-major: row j, col k):");
    println!("{:>6}  {:>4}  {:>22}  {:>22}", "j", "k", "Re", "Im");
    for j in 0..r {
        for k in 0..n_fft {
            let v = buf[j * n_fft + k];
            println!("{:>6}  {:>4}  {:>22.10}  {:>22.10}", j, k, v.re, v.im);
        }
    }

    // ── Step 2: apply r independent forward FFTs of length n_fft ─────────────
    //
    // RawPlanner with n = n_fft (= 2n_phys) applies r in-place FFTs, each of
    // length n_fft, laid out as r consecutive rows of n_fft entries.
    let raw = RawPlanner::new(n_fft, r, l, xl); // r rows, each of length n_fft
    raw.forward(&mut buf);

    // ── Step 3: print the full frequency-domain buffer ────────────────────────
    println!("\nFull FFT coefficients F̂[j·n_fft + k]  (before compaction):");
    println!(
        "{:>6}  {:>4}  {:>22}  {:>22}  {:>12}",
        "j", "k", "Re", "Im", "|F̂|"
    );
    for j in 0..r {
        for k in 0..n_fft {
            let v = buf[j * n_fft + k];
            println!(
                "{:>6}  {:>4}  {:>22.10}  {:>22.10}  {:>12.6}",
                j,
                k,
                v.re,
                v.im,
                v.norm()
            );
        }
    }

    // ── Step 4: verify Lemma 3.2 ─────────────────────────────────────────────
    //
    // For Even extension, Lemma 3.2 states for k ∈ {1,…,n−1}:
    //   j = 0:  F̂_{2n−k, 0}  =  F̂_{k, 0}
    //   j > 0:  F̂_{2n−k, j}  =  e^{−iπk/n} · F̂_{k, r−j}
    println!("\nLemma 3.2 symmetry residuals  (should be < {TOL:.0e}):");
    println!(
        "{:>6}  {:>4}  {:>22}  {:>22}",
        "j", "k", "predicted F̂_{2n−k,j}", "residual"
    );
    let mut max_residual = 0.0_f64;

    for k in 1..n {
        let km = n_fft - k; // mirror index 2n − k
        let phase = Complex::new(0.0_f64, -PI * (k as f64) / (n as f64)).exp();

        // j = 0: predicted = F̂_{k, 0}
        {
            let predicted = buf[0 * n_fft + k]; // F̂_{k, 0}
            let actual = buf[0 * n_fft + km]; // F̂_{2n−k, 0}
            let residual = (actual - predicted).norm();
            max_residual = max_residual.max(residual);
            println!(
                "{:>6}  {:>4}  {:>22.10}+{:.10}i  {:>22.6e}",
                0, k, predicted.re, predicted.im, residual
            );
        }

        // j > 0: predicted = phase · F̂_{k, r−j}
        for j in 1..r {
            let predicted = phase * buf[(r - j) * n_fft + k];
            let actual = buf[j * n_fft + km];
            let residual = (actual - predicted).norm();
            max_residual = max_residual.max(residual);
            println!(
                "{:>6}  {:>4}  {:>22.10}+{:.10}i  {:>22.6e}",
                j, k, predicted.re, predicted.im, residual
            );
        }
    }

    println!("\nMax Lemma 3.2 residual: {max_residual:.4e}");
    assert!(
        max_residual < TOL,
        "Lemma 3.2 symmetry violated: max residual = {max_residual:.4e} ≥ {TOL:.0e}"
    );

    // ── Step 5: verify low-frequency / high-frequency symmetries ─────────────────────
    //
    // Low-frequency (k = 0):       F̂_{0, r−j}  =   F̂_{0, j}   for j ∈ {1,…,r−1}
    // High-frequency (k = n):    F̂_{n, r−j}  =  −F̂_{n, j}   for j ∈ {1,…,r−1}
    //
    // Special case when r is even and j = r/2: F̂_{n, j} = 0  (self-negating).
    println!("\nLow-frequency (k=0) cross-row residuals  F̂_{{0,r−j}} − F̂_{{0,j}}  (should be < {TOL:.0e}):");
    println!(
        "{:>6}  {:>22}  {:>22}  {:>14}",
        "j", "F̂_{0,j}", "F̂_{0,r-j}", "residual"
    );
    let mut max_dc = 0.0_f64;
    for j in 1..r {
        let fwd = buf[j * n_fft]; // F̂_{0, j}
        let rev = buf[(r - j) * n_fft]; // F̂_{0, r−j}
        let res = (rev - fwd).norm();
        max_dc = max_dc.max(res);
        println!(
            "{:>6}  {:>10.6}+{:>10.6}i  {:>10.6}+{:>10.6}i  {:>14.6e}",
            j, fwd.re, fwd.im, rev.re, rev.im, res
        );
    }
    println!("Max low-frequency residual: {max_dc:.4e}");
    assert!(
        max_dc < TOL,
        "Low-frequency cross-row symmetry violated: {max_dc:.4e}"
    );

    println!("\nHigh-frequency (k=n={n}) cross-row residuals  F̂_{{n,r−j}} + F̂_{{n,j}}  (should be < {TOL:.0e}):");
    println!(
        "{:>6}  {:>22}  {:>22}  {:>14}",
        "j", "F̂_{n,j}", "F̂_{n,r-j}", "residual"
    );
    let mut max_nyq = 0.0_f64;
    for j in 1..r {
        let fwd = buf[j * n_fft + n]; // F̂_{n, j}
        let rev = buf[(r - j) * n_fft + n]; // F̂_{n, r−j}
        let res = (rev + fwd).norm(); // should be zero
        max_nyq = max_nyq.max(res);
        println!(
            "{:>6}  {:>10.6}+{:>10.6}i  {:>10.6}+{:>10.6}i  {:>14.6e}",
            j, fwd.re, fwd.im, rev.re, rev.im, res
        );
    }
    println!("Max high-frequency residual: {max_nyq:.4e}");
    assert!(
        max_nyq < TOL,
        "High-frequency cross-row symmetry violated: {max_nyq:.4e}"
    );
}

// ── Lemma 3.2 symmetry check (Odd extension) ─────────────────────────────────

/// Verify Lemma 3.2 and low-frequency/high-frequency symmetries for the Odd extension.
///
/// Setup: n=3 physical elements, r=4 local nodes, Extension::Odd.
/// Physical DOFs: n·r−1 = 11 (both Dirichlet endpoints excluded).
/// Extended FFT length: 2n = 6.
///
/// The Odd packing is anti-symmetric about x = L:
///   k < n, (k,j) ≠ (0,0) : buf[j·n_fft + k] =  f[k·r + j − 1]
///   k = 0, j = 0          :                    0  (left Dirichlet zero)
///   k = n, j = 0          :                    0  (right Dirichlet zero)
///   otherwise k ≥ n       : buf[j·n_fft + k] = −f[(2n−k)·r − j − 1]
///
/// Lemma 3.2 (Odd, sign flip vs. Even):
///   j = 0:  F̂_{2n−k, 0}  =  −F̂_{k, 0}
///   j > 0:  F̂_{2n−k, j}  =  −e^{−iπk/n} · F̂_{k, r−j}
///
/// Additional symmetries (opposite roles to Even):
///   Low-frequency (k = 0): F̂_{0, 0} = 0,  F̂_{0, r−j} = −F̂_{0, j}  for j > 0
///   High-frequency (k = n): F̂_{n, 0} = 0,  F̂_{n, r−j} =  F̂_{n, j}  for j > 0
///
/// Special case r even, j = r/2: low-frequency mode F̂_{0, r/2} = 0.
#[test]
fn odd_extension_symmetry() {
    const TOL: f64 = 1e-10;

    let n: usize = 3;
    let r: usize = 4;
    let l = 1.0_f64;
    let xl = 0.0_f64;

    // Physical planner (Odd extension): n·r−1 = 11 DOFs, n_fft = 6, n_freq = n+1 = 4.
    let planner = HofftPlanner::new(n, r, l, xl, Extension::Odd);
    let x = planner.get_x();

    // Same generic complex function as the Even test — no special symmetry.
    let f: Vec<Complex<f64>> = x
        .iter()
        .map(|&xi| {
            Complex::new(
                xi * xi - 0.7 * xi + 1.3,
                (1.5 * PI * xi).sin() + 0.4 * (PI * xi).cos(),
            )
        })
        .collect();

    // ── Step 1: pack the Odd extension into the full r × n_fft buffer ─────────
    let n_fft = 2 * n;
    let mut buf = vec![Complex::new(0.0_f64, 0.0); r * n_fft];
    for j in 0..r {
        for k in 0..n_fft {
            buf[j * n_fft + k] = if k < n {
                if k == 0 && j == 0 {
                    Complex::default() // left Dirichlet zero
                } else {
                    f[k * r + j - 1]
                }
            } else if k == n && j == 0 {
                Complex::default() // right Dirichlet zero
            } else {
                -f[(n_fft - k) * r - j - 1]
            };
        }
    }

    // Print physical DOFs.
    println!("\n{:=<70}", "");
    println!("Lemma 3.2 check — Odd extension  (n={n}, r={r}, n_fft={n_fft})");
    println!("{:=<70}", "");
    println!("\nPhysical DOFs (x-increasing, {} nodes):", x.len());
    println!(
        "{:>8}  {:>12}  {:>22}  {:>22}",
        "idx", "x", "Re f(x)", "Im f(x)"
    );
    for (i, (&xi, fi)) in x.iter().zip(f.iter()).enumerate() {
        println!("{:>8}  {:>12.6}  {:>22.10}  {:>22.10}", i, xi, fi.re, fi.im);
    }

    println!("\nOdd-extended periodic buffer (mode-major: row j, col k):");
    println!("{:>6}  {:>4}  {:>22}  {:>22}", "j", "k", "Re", "Im");
    for j in 0..r {
        for k in 0..n_fft {
            let v = buf[j * n_fft + k];
            println!("{:>6}  {:>4}  {:>22.10}  {:>22.10}", j, k, v.re, v.im);
        }
    }

    // ── Step 2: apply r independent forward FFTs of length n_fft ─────────────
    let raw = RawPlanner::new(n_fft, r, l, xl);
    raw.forward(&mut buf);

    println!("\nFull FFT coefficients F̂[j·n_fft + k]:");
    println!(
        "{:>6}  {:>4}  {:>22}  {:>22}  {:>12}",
        "j", "k", "Re", "Im", "|F̂|"
    );
    for j in 0..r {
        for k in 0..n_fft {
            let v = buf[j * n_fft + k];
            println!(
                "{:>6}  {:>4}  {:>22.10}  {:>22.10}  {:>12.6}",
                j,
                k,
                v.re,
                v.im,
                v.norm()
            );
        }
    }

    // ── Step 3: verify Lemma 3.2 (Odd: sign = −1) ────────────────────────────
    println!("\nLemma 3.2 (Odd) residuals  (should be < {TOL:.0e}):");
    println!(
        "{:>6}  {:>4}  {:>22}  {:>14}",
        "j", "k", "predicted F̂_{2n−k,j}", "residual"
    );
    let mut max_lem = 0.0_f64;

    for k in 1..n {
        let km = n_fft - k;
        let phase = Complex::new(0.0_f64, -PI * (k as f64) / (n as f64)).exp();

        // j = 0: F̂_{2n−k, 0} = −F̂_{k, 0}
        {
            let predicted = -buf[k]; // −F̂_{k, 0}
            let actual = buf[km]; //  F̂_{2n−k, 0}
            let res = (actual - predicted).norm();
            max_lem = max_lem.max(res);
            println!(
                "{:>6}  {:>4}  {:>10.6}+{:>10.6}i  {:>14.6e}",
                0, k, predicted.re, predicted.im, res
            );
        }
        // j > 0: F̂_{2n−k, j} = −e^{−iπk/n} · F̂_{k, r−j}
        for j in 1..r {
            let predicted = -phase * buf[(r - j) * n_fft + k];
            let actual = buf[j * n_fft + km];
            let res = (actual - predicted).norm();
            max_lem = max_lem.max(res);
            println!(
                "{:>6}  {:>4}  {:>10.6}+{:>10.6}i  {:>14.6e}",
                j, k, predicted.re, predicted.im, res
            );
        }
    }
    println!("Max Lemma 3.2 (Odd) residual: {max_lem:.4e}");
    assert!(max_lem < TOL, "Lemma 3.2 (Odd) violated: {max_lem:.4e}");

    // ── Step 4: verify low-frequency / high-frequency cross-row symmetries ─────────────────────
    //
    // Low-frequency (k=0): F̂_{0,0} = 0  and  F̂_{0, r−j} = −F̂_{0, j}  for j > 0.
    //   Anti-symmetry: the j-row and (r−j)-row of the Odd buffer contain the same
    //   physical values but opposite sign → their sums cancel / negate.
    //   Special case r even, j = r/2: F̂_{0, r/2} = 0.
    //
    // High-frequency (k=n): F̂_{n,0} = 0  and  F̂_{n, r−j} = F̂_{n, j}  for j > 0.
    //   The alternating (−1)^k factor combined with the Odd sign flip restores
    //   symmetry between row j and row (r−j).
    println!("\nLow-frequency (k=0): F̂_{{0,0}} = 0  and  F̂_{{0,r−j}} = −F̂_{{0,j}}  (residuals < {TOL:.0e}):");
    println!(
        "{:>6}  {:>22}  {:>22}  {:>14}",
        "j", "F̂_{0,j}", "F̂_{0,r-j}", "residual"
    );
    let mut max_dc = 0.0_f64;

    // j = 0: must be exactly zero
    {
        let v = buf[0]; // F̂_{0, 0}
        let res = v.norm();
        max_dc = max_dc.max(res);
        println!(
            "{:>6}  {:>10.6}+{:>10.6}i  {:>22}  {:>14.6e}",
            0, v.re, v.im, "(zero)", res
        );
    }
    for j in 1..r {
        let fwd = buf[j * n_fft]; // F̂_{0, j}
        let rev = buf[(r - j) * n_fft]; // F̂_{0, r−j}
        let res = (rev + fwd).norm(); // should be zero  (rev = −fwd)
        max_dc = max_dc.max(res);
        println!(
            "{:>6}  {:>10.6}+{:>10.6}i  {:>10.6}+{:>10.6}i  {:>14.6e}",
            j, fwd.re, fwd.im, rev.re, rev.im, res
        );
    }
    println!("Max low-frequency residual: {max_dc:.4e}");
    assert!(
        max_dc < TOL,
        "Low-frequency cross-row symmetry (Odd) violated: {max_dc:.4e}"
    );

    println!("\nHigh-frequency (k=n={n}): F̂_{{n,0}} = 0  and  F̂_{{n,r−j}} = F̂_{{n,j}}  (residuals < {TOL:.0e}):");
    println!(
        "{:>6}  {:>22}  {:>22}  {:>14}",
        "j", "F̂_{n,j}", "F̂_{n,r-j}", "residual"
    );
    let mut max_nyq = 0.0_f64;

    // j = 0: must be exactly zero
    {
        let v = buf[n]; // F̂_{n, 0}
        let res = v.norm();
        max_nyq = max_nyq.max(res);
        println!(
            "{:>6}  {:>10.6}+{:>10.6}i  {:>22}  {:>14.6e}",
            0, v.re, v.im, "(zero)", res
        );
    }
    for j in 1..r {
        let fwd = buf[j * n_fft + n]; // F̂_{n, j}
        let rev = buf[(r - j) * n_fft + n]; // F̂_{n, r−j}
        let res = (rev - fwd).norm(); // should be zero  (rev = fwd)
        max_nyq = max_nyq.max(res);
        println!(
            "{:>6}  {:>10.6}+{:>10.6}i  {:>10.6}+{:>10.6}i  {:>14.6e}",
            j, fwd.re, fwd.im, rev.re, rev.im, res
        );
    }
    println!("Max high-frequency residual: {max_nyq:.4e}");
    assert!(
        max_nyq < TOL,
        "High-frequency cross-row symmetry (Odd) violated: {max_nyq:.4e}"
    );
}

// ── FFT round-trip ────────────────────────────────────────────────────────────

fn roundtrip_error_1d(planner: &HofftPlanner, f_orig: &[Complex<f64>]) -> f64 {
    let mut g = f_orig.to_vec();
    planner.forward(&mut g);
    planner.inverse(&mut g);
    g.iter()
        .zip(f_orig.iter())
        .map(|(c, v)| (c - v).norm())
        .fold(0.0_f64, f64::max)
}

fn roundtrip_error_nd<const N: usize>(planner: &HofftPlannerND<N>, f_orig: &[Complex<f64>]) -> f64 {
    let mut g = f_orig.to_vec();
    planner.forward(&mut g);
    planner.inverse(&mut g);
    g.iter()
        .zip(f_orig.iter())
        .map(|(c, v)| (c - v).norm())
        .fold(0.0_f64, f64::max)
}

// Generic complex test function with no special symmetry:
//   Re f(x) = x² − 0.7x + 1.3
//   Im f(x) = sin(1.5πx) + 0.4cos(πx)
fn cplx(x: f64) -> Complex<f64> {
    Complex::new(
        x * x - 0.7 * x + 1.3,
        (1.5 * PI * x).sin() + 0.4 * (PI * x).cos(),
    )
}

#[test]
fn test_hofft_planner_roundtrip() {
    const TOL: f64 = 1e-12;

    // Periodic
    let p = HofftPlanner::new(8, 3, 1.0, 0.0, Extension::Periodic);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|&x| cplx(x)).collect();
    let err = roundtrip_error_1d(&p, &f);
    assert!(err < TOL, "HofftPlanner Periodic round-trip error = {err:.2e}");

    // Odd (Dirichlet) — imaginary part must vanish at boundaries for odd extension.
    // Use the same generic complex function; the extension enforces the BC internally.
    let p = HofftPlanner::new(8, 3, 1.0, 0.0, Extension::Odd);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|&x| cplx(x)).collect();
    let err = roundtrip_error_1d(&p, &f);
    assert!(err < TOL, "HofftPlanner Odd round-trip error = {err:.2e}");

    // Even (Neumann)
    let p = HofftPlanner::new(8, 3, 1.0, 0.0, Extension::Even);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|&x| cplx(x)).collect();
    let err = roundtrip_error_1d(&p, &f);
    assert!(err < TOL, "HofftPlanner Even round-trip error = {err:.2e}");
}

#[test]
fn test_nd_fft_roundtrip() {
    const TOL: f64 = 1e-12;

    // ── HofftPlannerND::<1> ───────────────────────────────────────────────────

    // Periodic
    let p = HofftPlannerND::<1>::new([8], [3], [1.0], [0.0], [Extension::Periodic]);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|pt| cplx(pt[0])).collect();
    let err = roundtrip_error_nd(&p, &f);
    assert!(err < TOL, "ND<1> Periodic round-trip error = {err:.2e}");

    // Odd (Dirichlet)
    let p = HofftPlannerND::<1>::new([8], [3], [1.0], [0.0], [Extension::Odd]);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|pt| cplx(pt[0])).collect();
    let err = roundtrip_error_nd(&p, &f);
    assert!(err < TOL, "ND<1> Odd round-trip error = {err:.2e}");

    // Even (Neumann)
    let p = HofftPlannerND::<1>::new([8], [3], [1.0], [0.0], [Extension::Even]);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|pt| cplx(pt[0])).collect();
    let err = roundtrip_error_nd(&p, &f);
    assert!(err < TOL, "ND<1> Even round-trip error = {err:.2e}");

    // ── HofftPlannerND::<2> ───────────────────────────────────────────────────

    // Periodic
    let p = HofftPlannerND::<2>::new([4; 2], [2; 2], [1.0; 2], [0.0; 2], [Extension::Periodic; 2]);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|pt| cplx(pt[0]) * cplx(pt[1])).collect();
    let err = roundtrip_error_nd(&p, &f);
    assert!(err < TOL, "ND<2> Periodic round-trip error = {err:.2e}");

    // Mixed Odd / Even
    let p = HofftPlannerND::<2>::new([4; 2], [2; 2], [1.0; 2], [0.0; 2], [Extension::Odd, Extension::Even]);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|pt| cplx(pt[0]) * cplx(pt[1])).collect();
    let err = roundtrip_error_nd(&p, &f);
    assert!(err < TOL, "ND<2> Odd/Even round-trip error = {err:.2e}");

    // ── HofftPlannerND::<3> ───────────────────────────────────────────────────

    // Periodic
    let p = HofftPlannerND::<3>::new([4; 3], [2; 3], [1.0; 3], [0.0; 3], [Extension::Periodic; 3]);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|pt| cplx(pt[0]) * cplx(pt[1]) * cplx(pt[2])).collect();
    let err = roundtrip_error_nd(&p, &f);
    assert!(err < TOL, "ND<3> Periodic round-trip error = {err:.2e}");

    // All Odd
    let p = HofftPlannerND::<3>::new([4; 3], [2; 3], [1.0; 3], [0.0; 3], [Extension::Odd; 3]);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|pt| cplx(pt[0]) * cplx(pt[1]) * cplx(pt[2])).collect();
    let err = roundtrip_error_nd(&p, &f);
    assert!(err < TOL, "ND<3> Odd round-trip error = {err:.2e}");

    // All Even
    let p = HofftPlannerND::<3>::new([4; 3], [2; 3], [1.0; 3], [0.0; 3], [Extension::Even; 3]);
    let f: Vec<Complex<f64>> = p.get_x().iter().map(|pt| cplx(pt[0]) * cplx(pt[1]) * cplx(pt[2])).collect();
    let err = roundtrip_error_nd(&p, &f);
    assert!(err < TOL, "ND<3> Even round-trip error = {err:.2e}");
}
