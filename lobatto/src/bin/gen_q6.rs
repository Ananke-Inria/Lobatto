//! Code generator for the Q6 (7-node Gauss-Lobatto) hard-coded differentiation kernel.
//!
//! Computes the 7×7 GLL differentiation matrix on \[0,1\] via `CollocationBasis`,
//! identifies the unique coefficients exploiting the antisymmetry
//! D[6-i, 6-j] = -D[i, j], and prints the corresponding `_d_q6_q6` Rust kernel.
//!
//! Run with:
//!   cargo run --bin gen_q6 --release

use lobatto::collocation::{CollocationBasis, Gauss};

fn main() {
    let n = 7usize; // Q6: 7 nodes
    let calc = CollocationBasis::new(vec![(n, Gauss::Lobatto)]);
    let pts = calc.points_1d(0);

    // 7×7 differentiation matrix from the Basis: d[(row, col)]
    let d = calc.diff_matrix_1d(0);

    // ── Print nodes ──────────────────────────────────────────────────────────
    println!("// Q6 GLL nodes on [0,1]:");
    for (i, x) in pts.iter().enumerate() {
        println!("//   x[{i}] = {x:.20}");
    }
    println!();

    // ── Print full matrix ────────────────────────────────────────────────────
    println!("// Full 7×7 differentiation matrix D[(row, col)]:");
    for row in 0..n {
        print!("//  row {row}: [");
        for col in 0..n {
            if col > 0 {
                print!(", ");
            }
            print!("{:+.16}", d[(row, col)]);
        }
        println!("]");
    }
    println!();

    // ── Verify antisymmetry D[6-i, 6-j] = -D[i, j] ─────────────────────────
    println!("// Antisymmetry check  max|D[(6-i, 6-j)] + D[(i, j)]|:");
    let mut max_err = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            max_err = max_err.max((d[(n - 1 - i, n - 1 - j)] + d[(i, j)]).abs());
        }
    }
    println!("//   {max_err:.3e}");
    println!();

    // ── Unique coefficients (rows 0..n_unique — upper half + middle) ─────────
    let n_unique = (n + 1) / 2; // = 4 for n=7
    println!("// Unique entries (rows 0..{n_unique}):");
    let labels = [
        ["D00", "D01", "D02", "D03", "D04", "D05", "D06"],
        ["D10", "D11", "D12", "D13", "D14", "D15", "D16"],
        ["D20", "D21", "D22", "D23", "D24", "D25", "D26"],
        ["D30", "D31", "D32", "D33", "D34", "D35", "D36"],
    ];
    for row in 0..n_unique {
        for col in 0..n {
            println!("//   {} = {:+.20}", labels[row][col], d[(row, col)]);
        }
    }
    println!();

    // ── Generated kernel (comment form) ─────────────────────────────────────
    println!("// ── Generated Rust kernel ───────────────────────────────────────────");
    println!("// Rows 4,5,6 use antisymmetry: out[6-k] = -out[k](a↔g, b↔f, c↔e)");
    println!();

    let node_names = ["a", "b", "c", "d", "e", "f", "g"];
    let n_half = n / 2; // = 3 for n=7
    for row in 0..n {
        let coeffs: Vec<f64> = if row < n_half {
            (0..n).map(|col| d[(row, col)]).collect()
        } else {
            (0..n).map(|col| -d[(n - 1 - row, n - 1 - col)]).collect()
        };

        print!("// out[{row}*M*N+i] =");
        let mut first = true;
        for col in 0..n {
            let v = coeffs[col];
            if v.abs() < 1e-14 {
                continue;
            }
            let av = v.abs();
            if first && v > 0.0 {
                print!(" {av:.16} * {}[i]", node_names[col]);
            } else {
                let sign = if v > 0.0 { " +" } else { " -" };
                print!("{sign} {av:.16} * {}[i]", node_names[col]);
            }
            first = false;
        }
        println!(";");
    }
    println!();

    // ── Compact unique constants (skip near-integers) ────────────────────────
    println!("// Compact unique absolute values for DX6 (rows 0..{n_unique}, skip near-integers):");
    let mut seen: Vec<f64> = Vec::new();
    for row in 0..n_unique {
        for col in 0..n {
            let av = d[(row, col)].abs();
            if av < 1e-12 {
                continue;
            }
            let is_near_int = (av.round() - av).abs() < 1e-10 && av.round() <= 30.0;
            if is_near_int {
                continue;
            }
            if seen.iter().any(|&s| (s - av).abs() < 1e-12) {
                continue;
            }
            seen.push(av);
            println!(
                "//   {av:.20}  (from D[({row},{col})] = {:+.16})",
                d[(row, col)]
            );
        }
    }
    println!();

    // ── Print DX6 array and full kernel ─────────────────────────────────────
    println!("const DX6: [f64; {}] = [", seen.len());
    for v in &seen {
        println!("    {v:.20},");
    }
    println!("];");
    println!();

    let fmt_coeff = |v: f64| -> String {
        let av = v.abs();
        if av < 1e-12 {
            return "0.0".to_string();
        }
        if (av.round() - av).abs() < 1e-10 && av.round() <= 30.0 {
            return format!("{:.1}", av);
        }
        if let Some(idx) = seen.iter().position(|&s| (s - av).abs() < 1e-12) {
            return format!("DX6[{idx}]");
        }
        format!("{av:.20}")
    };

    println!(
        "pub fn _d_q6_q6<const N: usize, const M: usize>(in_q6: &[f64], out_q6: &mut [f64]) {{"
    );
    println!("    assert!(in_q6.len() >= N * (6 * M + 1));");
    println!("    assert!(out_q6.len() >= N * (6 * M + 1));");
    println!();
    println!("    let a = &in_q6[0..N];");
    println!("    let b = &in_q6[M * N..M * N + N];");
    println!("    let c = &in_q6[2 * M * N..2 * M * N + N];");
    println!("    let d = &in_q6[3 * M * N..3 * M * N + N];");
    println!("    let e = &in_q6[4 * M * N..4 * M * N + N];");
    println!("    let f = &in_q6[5 * M * N..5 * M * N + N];");
    println!("    let g = &in_q6[6 * M * N..6 * M * N + N];");
    println!();

    for row in 0..n {
        let coeffs: Vec<f64> = if row < n_half {
            (0..n).map(|col| d[(row, col)]).collect()
        } else {
            (0..n).map(|col| -d[(n - 1 - row, n - 1 - col)]).collect()
        };

        let offset = if row == 0 {
            "i".to_string()
        } else {
            format!("{row} * M * N + i")
        };
        print!("    for i in 0..N {{ out_q6[{offset}] =");
        let mut first = true;
        for col in 0..n {
            let v = coeffs[col];
            if v.abs() < 1e-14 {
                continue;
            }
            let coeff_str = fmt_coeff(v);
            let node = node_names[col];
            if first && v > 0.0 {
                print!(" {coeff_str} * {node}[i]");
            } else {
                let sign = if v > 0.0 { " +" } else { " -" };
                print!("{sign} {coeff_str} * {node}[i]");
            }
            first = false;
        }
        println!("; }}");
    }
    println!("}}");

    // ── Transpose kernel ─────────────────────────────────────────────────────
    println!();

    let dt = d.transpose();

    let fmt_coeff_t = |v: f64| -> String {
        let av = v.abs();
        if av < 1e-12 { return "0.0".to_string(); }
        if (av.round() - av).abs() < 1e-10 && av.round() <= 30.0 {
            return format!("{:.1}", av);
        }
        if let Some(idx) = seen.iter().position(|&s| (s - av).abs() < 1e-12) {
            return format!("DX6[{idx}]");
        }
        format!("{av:.20}")
    };

    println!("fn _d_t_q6_q6<const N: usize, const M: usize>(in_q6: &[f64], out_q6: &mut [f64]) {{");
    println!("    let a = &in_q6[0..N];");
    println!("    let b = &in_q6[M * N..M * N + N];");
    println!("    let c = &in_q6[2 * M * N..2 * M * N + N];");
    println!("    let d = &in_q6[3 * M * N..3 * M * N + N];");
    println!("    let e = &in_q6[4 * M * N..4 * M * N + N];");
    println!("    let f = &in_q6[5 * M * N..5 * M * N + N];");
    println!("    let g = &in_q6[6 * M * N..6 * M * N + N];");
    println!();

    for row in 0..n {
        let coeffs: Vec<f64> = if row < n_half {
            (0..n).map(|col| dt[(row, col)]).collect()
        } else {
            (0..n).map(|col| -dt[(n - 1 - row, n - 1 - col)]).collect()
        };
        let offset = if row == 0 { "i".to_string() } else { format!("{row} * M * N + i") };
        print!("    for i in 0..N {{ out_q6[{offset}] =");
        let mut first = true;
        for col in 0..n {
            let v = coeffs[col];
            if v.abs() < 1e-14 { continue; }
            let coeff_str = fmt_coeff_t(v);
            let node = node_names[col];
            if first && v > 0.0 {
                print!(" {coeff_str} * {node}[i]");
            } else {
                let sign = if v > 0.0 { " +" } else { " -" };
                print!("{sign} {coeff_str} * {node}[i]");
            }
            first = false;
        }
        println!("; }}");
    }
    println!("}}");
}
