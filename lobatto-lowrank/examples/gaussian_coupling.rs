//! A coupled Gaussian on a 2D + 2D split: build from many separable terms,
//! compress, evaluate.
//!
//! The field on `[-1, 1]² × [-1, 1]²` is
//!
//! ```text
//! f(x, y) = exp(-(|x|² + |y|²) / 2a²) · exp(-|x - y|² / 2s²),
//! ```
//!
//! not separable because of the coupling across the cut. The coupling is the
//! convolution of two Gaussians of width `s / √2`,
//!
//! ```text
//! exp(-|x - y|² / 2s²) = 2 / (π s²) ∫ exp(-|x - z|² / s²) exp(-|z - y|² / s²) dz,
//! ```
//!
//! and the trapezoid rule on `z_q` (spacing `h`, `27²` points on
//! `[-2.6, 2.6]²`) turns it into a sum of separable products, one term per
//! quadrature point:
//!
//! ```text
//! f ≈ Σ_q c · [g(x) e^{-|x - z_q|²/s²}] ⊗ [g(y) e^{-|y - z_q|²/s²}],   c = 2h² / (π s²).
//! ```
//!
//! Both quadrature errors (aliasing ~ `e^{-31}`, truncation of the `z` range
//! ~ `1e-9`) are far below the interpolation error, so the pointwise error
//! against the exact formula measures the whole chain: sampling, compression
//! and interpolation.
//!
//! - Small grids, `21 × 21 + 21 × 21` nodes (`21^4 = 194481` on the full
//!   grid): all 729 terms compressed at once, the singular values checked
//!   against a dense SVD of the weighted unfolding.
//! - Large grids, `81 × 81 + 81 × 81` and `161 × 161 + 161 × 161` nodes
//!   (`81^4 = 43046721` and `161^4 = 671898241` on the full grid, which is
//!   never formed): terms added in batches of 64 with a compression after
//!   each batch.
//!
//! Run with `cargo run --release --example gaussian_coupling`.

use std::f64::consts::PI;
use std::time::Instant;

use lobatto_grid::{Extension, GridND};
use lobatto_lowrank::{LowRankSum, Truncation};
use nalgebra::DMatrix;

type Sum = LowRankSum<2, 2>;

/// Width of the marginal Gaussians.
const A: f64 = 0.4;
/// Width of the coupling.
const S: f64 = 0.5;
/// Quadrature spacing in `z`.
const HZ: f64 = 0.2;
/// Quadrature range `[-ZMAX, ZMAX]²`.
const ZMAX: f64 = 2.6;
/// Relative tolerance of every compression.
const TOL: f64 = 1e-10;

/// The exact field at `(x, y)`, `p = [x₁, x₂, y₁, y₂]`; its maximum is 1.
fn exact(p: [f64; 4]) -> f64 {
    let (x2, y2) = (p[0] * p[0] + p[1] * p[1], p[2] * p[2] + p[3] * p[3]);
    let d2 = (p[0] - p[2]).powi(2) + (p[1] - p[3]).powi(2);
    (-(x2 + y2) / (2.0 * A * A) - d2 / (2.0 * S * S)).exp()
}

/// `n × n` elements of order 4 on `[-1, 1]²`, Neumann: `(4n + 1)²` nodes.
fn grid(n: usize) -> GridND<2> {
    GridND::new([n, n], [4, 4], [2.0; 2], [-1.0; 2], [Extension::Even; 2])
}

/// The quadrature points `z_q`.
fn quadrature_points() -> Vec<[f64; 2]> {
    let k = (2.0 * ZMAX / HZ).round() as usize;
    let z = |i: usize| -ZMAX + i as f64 * HZ;
    (0..=k).flat_map(|i| (0..=k).map(move |j| [z(i), z(j)])).collect()
}

/// The term of quadrature point `z`.
fn term(n: usize, z: [f64; 2]) -> Sum {
    let factor = move |x: [f64; 2]| {
        let r2 = x[0] * x[0] + x[1] * x[1];
        let d2 = (x[0] - z[0]).powi(2) + (x[1] - z[1]).powi(2);
        (-r2 / (2.0 * A * A) - d2 / (S * S)).exp()
    };
    let mut t = Sum::new_from_fn(grid(n), grid(n), factor, factor);
    t.scale(HZ * HZ * 2.0 / (PI * S * S));
    t
}

/// Deterministic uniform points of `[-1, 1]⁴`.
fn random_points(count: usize) -> Vec<[f64; 4]> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        2.0 * ((state >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    (0..count).map(|_| [next(), next(), next(), next()]).collect()
}

/// Max pointwise error against the exact field, and the evaluation time.
fn max_error(s: &Sum, xs: &[[f64; 4]]) -> (f64, f64) {
    let mut out = vec![0.0; xs.len()];
    let t = Instant::now();
    s.get_values(xs, &mut out);
    let time = t.elapsed().as_secs_f64();
    let err = xs
        .iter()
        .zip(&out)
        .map(|(&p, v)| (v - exact(p)).abs())
        .fold(0.0, f64::max);
    (err, time)
}

fn print_decay(s: &Sum) {
    let a = s.coefficients();
    let ks: Vec<usize> = (0..a.len()).step_by(5).chain([a.len() - 1]).collect();
    let line: Vec<String> = ks.iter().map(|&k| format!("σ{k}/σ0 = {:.1e}", a[k] / a[0])).collect();
    println!("  decay        {}", line.join(", "));
}

fn print_ranks(s: &Sum) {
    let line: Vec<String> = [1e-2, 1e-4, 1e-6, 1e-8]
        .iter()
        .map(|&tol| format!("{tol:.0e} → {}", s.clone().compress(Truncation::RelativeTol(tol)).kept))
        .collect();
    println!("  rank by tol  {}", line.join(", "));
}

fn main() {
    let zs = quadrature_points();
    let xs = random_points(100_000);
    println!(
        "coupled Gaussian, a = {A}, s = {S}: {} quadrature terms, compression tolerance {TOL:.0e}",
        zs.len()
    );

    // ── Small grids: one-shot compression, checked against a dense SVD ──────
    let n = 5;
    let mut s = Sum::new(grid(n), grid(n));
    let t = Instant::now();
    for &z in &zs {
        s.add_assign(&term(n, z));
    }
    let t_build = t.elapsed().as_secs_f64();
    let (nl, nr) = (s.n_left(), s.n_right());

    // Dense reference: singular values of W_L^{1/2} P W_R^{1/2}.
    let mut p = DMatrix::from_vec(nl, nr, s.assemble());
    for (j, mut col) in p.column_iter_mut().enumerate() {
        for (i, x) in col.iter_mut().enumerate() {
            *x *= s.sqrt_w_left()[i] * s.sqrt_w_right()[j];
        }
    }
    let dense = p.svd(false, false).singular_values;

    let t = Instant::now();
    let c = s.compress(Truncation::RelativeTol(TOL));
    let t_compress = t.elapsed().as_secs_f64();
    let sigma: Vec<f64> = s.coefficients().iter().chain(&c.discarded).copied().collect();
    let sv_diff = sigma
        .iter()
        .zip(dense.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / dense[0];
    let (err, t_eval) = max_error(&s, &xs);

    let (l, r) = (s.left().ndofs(), s.right().ndofs());
    println!(
        "\n── {} × {} + {} × {} nodes, {}^4 = {} on the full grid ──",
        l[0],
        l[1],
        r[0],
        r[1],
        l[0],
        nl * nr
    );
    println!(
        "  terms        {} → {} in one compression ({:.1} ms build, {:.1} ms compress)",
        zs.len(),
        c.kept,
        t_build * 1e3,
        t_compress * 1e3
    );
    println!("  vs dense SVD max |σ_k − σ_k^dense| / σ0 = {sv_diff:.1e}");
    print_decay(&s);
    print_ranks(&s);
    println!(
        "  accuracy     max error {err:.1e} at {} random points ({:.0} ms)",
        xs.len(),
        t_eval * 1e3
    );

    // ── Large grids: batched build, full grid never formed ──────────────────
    for n in [20, 40] {
        let batch = 64;
        let mut s = Sum::new(grid(n), grid(n));
        let (mut peak, mut t_compress) = (0, 0.0);
        let t = Instant::now();
        for (q, &z) in zs.iter().enumerate() {
            s.add_assign(&term(n, z));
            peak = peak.max(s.len());
            if (q + 1) % batch == 0 || q + 1 == zs.len() {
                let tc = Instant::now();
                s.compress(Truncation::RelativeTol(TOL));
                t_compress += tc.elapsed().as_secs_f64();
            }
        }
        let t_total = t.elapsed().as_secs_f64();
        let (nl, nr) = (s.n_left(), s.n_right());
        let (err, t_eval) = max_error(&s, &xs);
        let low_rank_mb = 8.0 * (s.len() * (nl + nr + 1)) as f64 / 1e6;
        let full_mb = 8.0 * (nl * nr) as f64 / 1e6;

        let (l, r) = (s.left().ndofs(), s.right().ndofs());
        println!(
            "\n── {} × {} + {} × {} nodes, {}^4 = {} on the full grid ──",
            l[0],
            l[1],
            r[0],
            r[1],
            l[0],
            nl * nr
        );
        println!(
            "  terms        {} added in batches of {batch} (at most {peak} at once) → {}",
            zs.len(),
            s.len()
        );
        println!(
            "  time         {:.2} s total, of which {:.2} s in {} compressions",
            t_total,
            t_compress,
            zs.len().div_ceil(batch)
        );
        print_decay(&s);
        print_ranks(&s);
        println!(
            "  accuracy     max error {err:.1e} at {} random points ({:.0} ms)",
            xs.len(),
            t_eval * 1e3
        );
        println!("  memory       {low_rank_mb:.1} MB low-rank vs {full_mb:.0} MB for the full field");
    }
}
