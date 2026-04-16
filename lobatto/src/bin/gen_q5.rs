//! Code generator for the Q5 (6-node Gauss-Lobatto) hard-coded differentiation kernel.
//!
//! Computes the 6×6 GLL differentiation matrix on \[0,1\] via `CollocationBasis`,
//! identifies the unique coefficients exploiting the antisymmetry
//! D[5-i, 5-j] = -D[i, j], and prints the corresponding `_d_q5_q5` Rust kernel.
//!
//! Run with:
//!   cargo run --bin gen_q5 --release

use lobatto::collocation::{CollocationBasis, Gauss};

fn main() {
    let n = 6usize; // Q5: 6 nodes
    let calc = CollocationBasis::new(vec![(n, Gauss::Lobatto)]);
    let pts = calc.points_1d(0);

    // 6×6 differentiation matrix from the Basis: d[(row, col)]
    let d = calc.diff_matrix_1d(0);

    // ── Print nodes ──────────────────────────────────────────────────────────
    println!("// Q5 GLL nodes on [0,1]:");
    for (i, x) in pts.iter().enumerate() {
        println!("//   x[{i}] = {x:.20}");
    }
    println!();

    // ── Print full matrix ────────────────────────────────────────────────────
    println!("// Full 6×6 differentiation matrix D[(row, col)]:");
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

    // ── Verify antisymmetry D[5-i, 5-j] = -D[i, j] ─────────────────────────
    println!("// Antisymmetry check  max|D[(5-i, 5-j)] + D[(i, j)]|:");
    let mut max_err = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            max_err = max_err.max((d[(n - 1 - i, n - 1 - j)] + d[(i, j)]).abs());
        }
    }
    println!("//   {max_err:.3e}");
    println!();

    // ── Unique coefficients (rows 0, 1, 2 — upper half) ─────────────────────
    println!("// Unique entries (rows 0,1,2):");
    let labels = [
        ["D00", "D01", "D02", "D03", "D04", "D05"],
        ["D10", "D11", "D12", "D13", "D14", "D15"],
        ["D20", "D21", "D22", "D23", "D24", "D25"],
    ];
    for row in 0..3 {
        for col in 0..n {
            println!("//   {} = {:+.20}", labels[row][col], d[(row, col)]);
        }
    }
    println!();

    // ── Generated kernel (comment form) ─────────────────────────────────────
    println!("// ── Generated Rust kernel ───────────────────────────────────────────");
    println!("// Rows 3,4,5 use antisymmetry: out[5-k] = -out[k](a↔f, b↔e, c↔d)");
    println!();

    let node_names = ["a", "b", "c", "d", "e", "f"];
    for row in 0..n {
        let coeffs: Vec<f64> = if row < 3 {
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
    println!("// Compact unique absolute values for DX5 (rows 0,1,2, skip near-integers):");
    let mut seen: Vec<f64> = Vec::new();
    for row in 0..3 {
        for col in 0..n {
            let av = d[(row, col)].abs();
            if av < 1e-12 {
                continue;
            }
            let is_near_int = (av.round() - av).abs() < 1e-10 && av.round() <= 20.0;
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

    // ── Print DX5 array and full kernel ─────────────────────────────────────
    println!("const DX5: [f64; {}] = [", seen.len());
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
        if (av.round() - av).abs() < 1e-10 && av.round() <= 20.0 {
            return format!("{:.1}", av);
        }
        if let Some(idx) = seen.iter().position(|&s| (s - av).abs() < 1e-12) {
            return format!("DX5[{idx}]");
        }
        format!("{av:.20}")
    };

    println!(
        "pub fn _d_q5_q5<const N: usize, const M: usize>(in_q5: &[f64], out_q5: &mut [f64]) {{"
    );
    println!("    assert!(in_q5.len() >= N * (5 * M + 1));");
    println!("    assert!(out_q5.len() >= N * (5 * M + 1));");
    println!();
    println!("    let a = &in_q5[0..N];");
    println!("    let b = &in_q5[M * N..M * N + N];");
    println!("    let c = &in_q5[2 * M * N..2 * M * N + N];");
    println!("    let d = &in_q5[3 * M * N..3 * M * N + N];");
    println!("    let e = &in_q5[4 * M * N..4 * M * N + N];");
    println!("    let f = &in_q5[5 * M * N..5 * M * N + N];");
    println!();

    for row in 0..n {
        let coeffs: Vec<f64> = if row < 3 {
            (0..n).map(|col| d[(row, col)]).collect()
        } else {
            (0..n).map(|col| -d[(n - 1 - row, n - 1 - col)]).collect()
        };

        let offset = if row == 0 {
            "i".to_string()
        } else {
            format!("{row} * M * N + i")
        };
        print!("    for i in 0..N {{ out_q5[{offset}] =");
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
}
