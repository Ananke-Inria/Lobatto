use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lobatto_grid::{Extension, GridND};
use num_complex::Complex;
use std::hint::black_box;

/// Benchmark `GridND::<4>::eval` alone (interpolation only) on a ~10M-DOF
/// 4D grid.
///
/// The grid size is fixed at (n*r)^4 = 56^4 ≈ 9.83M DOFs, with 10M query
/// points. Query points are pseudo-random in-domain positions, so no
/// extension folding is involved — this measures the pure interpolation cost
/// (element lookup, grouping, gather, sum-factored contraction).
fn bench_eval_4d(c: &mut Criterion) {
    let r = 4;
    let n = 14; // (n*r)^4 = 56^4 ≈ 9.83M DOFs

    let grid = GridND::<4>::new([n; 4], [r; 4], [1.0; 4], [0.0; 4], [Extension::Periodic; 4]);
    let x = grid.get_x();
    let total = x.len();

    // Smooth field at the DOF nodes.
    let q = |t: f64| t * t * (t - 1.0);
    let u: Vec<Complex<f64>> = (0..total)
        .map(|p| Complex::new((0..4).map(|d| q(x[p][d])).product::<f64>(), 0.0))
        .collect();

    // Deterministic pseudo-random in-domain query points (xorshift64*).
    let mut state = 0x9e3779b97f4a7c15_u64;
    let mut next_unit = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state.wrapping_mul(0x2545f4914f6cdd1d) >> 11) as f64 / (1u64 << 53) as f64
    };

    let m = 10_000_000;

    let mut group = c.benchmark_group("hofft_eval_4d");
    group.sample_size(10);

    let xs: Vec<[f64; 4]> = (0..m).map(|_| std::array::from_fn(|_| next_unit())).collect();
    let mut fu = vec![Complex::new(0.0_f64, 0.0); m];
    group.throughput(Throughput::Elements(m as u64));

    group.bench_function(BenchmarkId::from_parameter(format!("m={m}")), |b| {
        b.iter(|| {
            grid.eval(black_box(&xs), black_box(&u), &mut fu);
            black_box(fu[0])
        })
    });

    group.finish();
}

/// Benchmark `GridND::<4>::convect` on the same ~10M-DOF 4D grid.
///
/// The flow map shifts every node out of the domain, so the periodic folding
/// path of `eval_extended` is exercised; m = n_dof ≈ 9.83M by construction.
fn bench_convect_4d(c: &mut Criterion) {
    let r = 4;
    let n = 14; // (n*r)^4 = 56^4 ≈ 9.83M DOFs

    let grid = GridND::<4>::new([n; 4], [r; 4], [1.0; 4], [0.0; 4], [Extension::Periodic; 4]);
    let x = grid.get_x();
    let total = x.len();

    // Smooth field at the DOF nodes.
    let q = |t: f64| t * t * (t - 1.0);
    let u: Vec<Complex<f64>> = (0..total)
        .map(|p| Complex::new((0..4).map(|d| q(x[p][d])).product::<f64>(), 0.0))
        .collect();

    let lambda = |pt: [f64; 4]| pt.map(|t| t + 0.13);

    let mut out = vec![Complex::new(0.0_f64, 0.0); total];

    let mut group = c.benchmark_group("hofft_convect_4d");
    group.sample_size(10);
    group.throughput(Throughput::Elements(total as u64));

    group.bench_function(BenchmarkId::from_parameter(format!("m={total}")), |b| {
        b.iter(|| {
            grid.convect(black_box(lambda), black_box(&u), &mut out);
            black_box(out[0])
        })
    });

    group.finish();
}

/// Benchmark `GridND::<4>::eval_with_plan` — the repeated-`convect`
/// configuration: the plan (flow map, folding, element grouping, gather
/// tables, local coordinates) is built once, only the apply is measured.
fn bench_convect_plan_4d(c: &mut Criterion) {
    let r = 4;
    let n = 14; // (n*r)^4 = 56^4 ≈ 9.83M DOFs

    let grid = GridND::<4>::new([n; 4], [r; 4], [1.0; 4], [0.0; 4], [Extension::Periodic; 4]);
    let x = grid.get_x();
    let total = x.len();

    // Smooth field at the DOF nodes.
    let q = |t: f64| t * t * (t - 1.0);
    let u: Vec<Complex<f64>> = (0..total)
        .map(|p| Complex::new((0..4).map(|d| q(x[p][d])).product::<f64>(), 0.0))
        .collect();

    let lambda = |pt: [f64; 4]| pt.map(|t| t + 0.13);

    let t0 = std::time::Instant::now();
    let plan = grid.convect_plan(lambda);
    println!("convect_plan construction: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    let mut out = vec![Complex::new(0.0_f64, 0.0); total];

    let mut group = c.benchmark_group("hofft_convect_plan_4d");
    group.sample_size(10);
    group.throughput(Throughput::Elements(total as u64));

    group.bench_function(BenchmarkId::from_parameter(format!("m={total}")), |b| {
        b.iter(|| {
            grid.eval_with_plan(black_box(&plan), black_box(&u), &mut out);
            black_box(out[0])
        })
    });

    group.finish();
}

/// Same as `bench_convect_plan_4d` but with a real (`f64`) field: half the
/// memory traffic and flops in the contraction.
fn bench_convect_plan_4d_real(c: &mut Criterion) {
    let r = 4;
    let n = 14; // (n*r)^4 = 56^4 ≈ 9.83M DOFs

    let grid = GridND::<4>::new([n; 4], [r; 4], [1.0; 4], [0.0; 4], [Extension::Periodic; 4]);
    let x = grid.get_x();
    let total = x.len();

    let q = |t: f64| t * t * (t - 1.0);
    let u: Vec<f64> = (0..total)
        .map(|p| (0..4).map(|d| q(x[p][d])).product::<f64>())
        .collect();

    let lambda = |pt: [f64; 4]| pt.map(|t| t + 0.13);
    let plan = grid.convect_plan(lambda);

    let mut out = vec![0.0_f64; total];

    let mut group = c.benchmark_group("hofft_convect_plan_4d_real");
    group.sample_size(10);
    group.throughput(Throughput::Elements(total as u64));

    group.bench_function(BenchmarkId::from_parameter(format!("m={total}")), |b| {
        b.iter(|| {
            grid.eval_with_plan(black_box(&plan), black_box(&u), &mut out);
            black_box(out[0])
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_eval_4d,
    bench_convect_4d,
    bench_convect_plan_4d,
    bench_convect_plan_4d_real
);
criterion_main!(benches);
