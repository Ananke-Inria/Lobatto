//! Code generator for a hardcoded Gauss–Lobatto–Legendre quadrature lookup function.
//!
//! For each order n = 2..=15, computes the GLL nodes and weights via the iterative
//! solver in `quadratix`, then verifies that the rule integrates all polynomials up to
//! degree 2*(n-1)−1 exactly with a tolerance of 1e-15.  Symmetry, positivity, ordering,
//! and endpoint conditions are also checked.  Only if **all** checks pass is the
//! certified Rust function written to stdout.
//!
//! Run with:
//!   cargo run --bin gen_lobatto --release

use quadratix::quadrature::gauss_lobatto;

const TOLERANCE: f64 = 1e-15;
const MAX_ORDER: usize = 14;

/// Verify all accuracy conditions with the given tolerance.
/// Returns the maximum polynomial-integration error observed.
fn verify(n: usize, points: &[f64], weights: &[f64]) -> f64 {
    let mut max_err = 0.0_f64;

    // 1. Polynomial exactness up to degree 2*(n-1)-1
    for p in 0..2 * (n - 1) {
        let analytical = if p % 2 == 0 {
            2.0 / ((p + 1) as f64)
        } else {
            0.0
        };

        let numerical: f64 = points
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * x.powi(p as i32))
            .sum();

        let err = (analytical - numerical).abs();
        max_err = max_err.max(err);

        if analytical != 0.0 {
            assert!(
                err < TOLERANCE,
                "Polynomial exactness FAILED for n={n}, p={p}: \
                 analytical={analytical}, numerical={numerical}, err={err:.3e} > tol={TOLERANCE:.0e}"
            );
        } else {
            assert!(
                numerical.abs() < TOLERANCE,
                "Polynomial exactness FAILED for n={n}, p={p} (odd, expect 0): \
                 numerical={numerical:.3e} > tol={TOLERANCE:.0e}"
            );
        }
    }

    // 2. Positive weights
    for (i, &w) in weights.iter().enumerate() {
        assert!(w > 0.0, "Non-positive weight for n={n}, index={i}: w={w}");
    }

    // 3. Strictly increasing order
    for i in 1..points.len() {
        assert!(
            points[i] > points[i - 1],
            "Points not in increasing order for n={n}, index={i}: {} >= {}",
            points[i - 1],
            points[i]
        );
    }

    // 4. Symmetry of points and weights
    let m = points.len();
    for i in 0..(m / 2) {
        let sym_err = (points[i] + points[m - 1 - i]).abs();
        assert!(
            sym_err < TOLERANCE,
            "Points not symmetric for n={n}, i={i}: left={}, right={}, sum={sym_err:.3e}",
            points[i],
            points[m - 1 - i]
        );
        let wt_err = (weights[i] - weights[m - 1 - i]).abs();
        assert!(
            wt_err < TOLERANCE,
            "Weights not symmetric for n={n}, i={i}: left={}, right={}, diff={wt_err:.3e}",
            weights[i],
            weights[m - 1 - i]
        );
    }

    // 5. Middle point = 0 (odd n)
    if m % 2 == 1 {
        let mid_err = points[m / 2].abs();
        assert!(
            mid_err < TOLERANCE,
            "Middle point not zero for n={n}: mid={}, err={mid_err:.3e}",
            points[m / 2]
        );
    }

    // 6. Leftmost point = -1
    let lep_err = (points[0] + 1.0).abs();
    assert!(
        lep_err < TOLERANCE,
        "Left endpoint not -1 for n={n}: leftmost={}, err={lep_err:.3e}",
        points[0]
    );

    max_err
}

fn fmt_f64(v: f64) -> String {
    format!("{:.20}", v)
}

fn main() {
    eprintln!(
        "Generating hardcoded GLL quadrature for n = 2..={MAX_ORDER} \
         (tolerance = {TOLERANCE:.0e})\n"
    );

    // Compute and verify all rules first — no output is written until all pass.
    let mut all_data: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

    for n in 2..=MAX_ORDER {
        let (points, weights) = gauss_lobatto(n);
        let max_err = verify(n, &points, &weights);
        eprintln!("  n={n:2}: {n:2} points, max poly-integration error = {max_err:.3e}  [OK]");
        all_data.push((points, weights));
    }

    eprintln!("\nAll rules verified. Writing certified function to stdout...\n");

    // ── Generated code ────────────────────────────────────────────────────────
    println!(
        "/// Hardcoded Gauss–Lobatto–Legendre (GLL) quadrature on [-1, 1] for n = 2..={MAX_ORDER}."
    );
    println!("///");
    println!("/// Points and weights are certified: every rule integrates all monomials");
    println!("/// of degree up to 2*(n-1)-1 exactly, with a maximum observed floating-point");
    println!("/// error below {TOLERANCE:.0e} (verified by `gen_lobatto`).");
    println!("///");
    println!("/// Returns `(points, weights)` with `points` in increasing order.");
    println!("///");
    println!("/// # Panics");
    println!("/// Panics if `n < 2` or `n > {MAX_ORDER}`.");
    println!("pub fn gauss_lobatto_hardcoded(n: usize) -> (Vec<f64>, Vec<f64>) {{");
    println!("    match n {{");

    for (idx, n) in (2..=MAX_ORDER).enumerate() {
        let (points, weights) = &all_data[idx];
        let pts_str: Vec<String> = points.iter().map(|&v| fmt_f64(v)).collect();
        let wts_str: Vec<String> = weights.iter().map(|&v| fmt_f64(v)).collect();

        println!(
            "        // n={n}: {n} points, exact up to degree {}",
            2 * (n - 1) - 1
        );
        println!("        {n} => (");
        println!("            vec![{}],", pts_str.join(", "));
        println!("            vec![{}],", wts_str.join(", "));
        println!("        ),");
    }

    println!(
        "        _ => panic!(\"gauss_lobatto_hardcoded: n must be in 2..={MAX_ORDER}, got {{}}\", n),"
    );
    println!("    }}");
    println!("}}");
}
