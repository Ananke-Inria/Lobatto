//! Benchmarks of `lobatto-lowrank` on 2D + 2D sums: the compression as a
//! whole, its steps one by one, and the algebra / evaluation operations.
//!
//! Factors are deterministic pseudo-random values on Neumann grids of order 4.
//! Cases: `81 × 81 + 81 × 81` and `161 × 161 + 161 × 161` nodes with
//! `M = 32 → r = 8` (the plan's sizes), and `161 × 161 + 161 × 161` with
//! `M = 158 → r = 94` (the peak of the batched build in the
//! `gaussian_coupling` example).
//!
//! Run with `cargo bench --bench compress`. Group names are stable
//! identifiers: renaming them discards the Criterion baseline history.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::{diag::DiagMut, MatMut, MatRef, Par};
use lobatto_grid::{Extension, GridND};
use lobatto_lowrank::{LowRankSum, Truncation};
use nalgebra::{DMatrix, DVector};

/// `(n, M, r)`: `n × n` elements of order 4 per side, `M` terms, kept rank `r`.
const CASES: &[(usize, usize, usize)] = &[(20, 32, 8), (40, 32, 8), (40, 158, 94)];

/// `n × n` elements of order 4 on `[-1, 1]²`: `(4n + 1)²` nodes.
fn grid(n: usize) -> GridND<2> {
    GridND::new([n, n], [4, 4], [2.0; 2], [-1.0; 2], [Extension::Even; 2])
}

/// Deterministic pseudo-random value at node `x` for term `k` (sine hash).
fn hash(x: [f64; 2], k: usize, salt: f64) -> f64 {
    let h = x[0] * 12.9898 + x[1] * 78.233 + k as f64 * 37.719 + salt;
    (h.sin() * 43_758.545_3).fract() - 0.5
}

/// `m` terms with pseudo-random factors.
fn random_sum(n: usize, m: usize) -> LowRankSum<2, 2> {
    let mut s = LowRankSum::new(grid(n), grid(n));
    for k in 0..m {
        let mut t = LowRankSum::new_from_fn(grid(n), grid(n), |x| hash(x, k, 0.0), |y| hash(y, k, 1.0));
        t.scale(1.0 + k as f64 / m as f64);
        s.add_assign(&t);
    }
    s
}

fn id(n: usize, m: usize, r: usize) -> String {
    let d = 4 * n + 1;
    format!("{d}x{d}+{d}x{d}_m{m}_r{r}")
}

/// `compress(Rank(r))` as a whole.
fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("lowrank_compress");
    group.sample_size(10);
    for &(n, m, r) in CASES {
        let s = random_sum(n, m);
        group.bench_function(BenchmarkId::from_parameter(id(n, m, r)), |b| {
            b.iter_batched(
                || s.clone(),
                |mut t| black_box(t.compress(Truncation::Rank(r))),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// The steps of `compress`, replicated on the public factor matrices (left
/// side for the QR steps; the right side costs the same).
fn bench_compress_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("lowrank_compress_steps");
    group.sample_size(10);
    for &(n, m, r) in CASES {
        let s = random_sum(n, m);
        let p = id(n, m, r);
        let sw = DVector::from_column_slice(s.sqrt_w_left());
        let mut scaled = s.rho().clone();
        for mut col in scaled.column_iter_mut() {
            col.component_mul_assign(&sw);
        }
        let qr = scaled.clone().qr();
        let q = qr.q();
        let a = DVector::from_column_slice(s.coefficients());
        let mut core: DMatrix<f64> = qr.r();
        for (mut col, &ak) in core.column_iter_mut().zip(a.iter()) {
            col *= ak;
        }
        let core = &core * qr.r().transpose();
        let u = core.clone().svd(true, false).u.unwrap();

        group.bench_function(BenchmarkId::new("scale", &p), |b| {
            b.iter_batched(
                || s.rho().clone(),
                |mut x| {
                    for mut col in x.column_iter_mut() {
                        col.component_mul_assign(&sw);
                    }
                    x
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_function(BenchmarkId::new("qr", &p), |b| {
            b.iter_batched(|| scaled.clone(), |x| x.qr(), BatchSize::LargeInput)
        });
        group.bench_function(BenchmarkId::new("form_q", &p), |b| b.iter(|| qr.q()));
        // nalgebra's SVD, kept for comparison; `compress` uses faer's (below).
        group.bench_function(BenchmarkId::new("svd_core", &p), |b| {
            b.iter_batched(|| core.clone(), |x| x.svd(true, true), BatchSize::SmallInput)
        });
        group.bench_function(BenchmarkId::new("svd_core_faer", &p), |b| {
            let k = core.nrows().min(core.ncols());
            let thin = ComputeSvdVectors::Thin;
            b.iter(|| {
                let (mut u, mut v) = (DMatrix::zeros(core.nrows(), k), DMatrix::zeros(core.ncols(), k));
                let mut sigma = DVector::zeros(k);
                let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
                    core.nrows(),
                    core.ncols(),
                    thin,
                    thin,
                    Par::Seq,
                    Default::default(),
                ));
                svd::svd(
                    MatRef::from_column_major_slice(core.as_slice(), core.nrows(), core.ncols()),
                    DiagMut::from_slice_mut(sigma.as_mut_slice()),
                    Some(MatMut::from_column_major_slice_mut(u.as_mut_slice(), core.nrows(), k)),
                    Some(MatMut::from_column_major_slice_mut(v.as_mut_slice(), core.ncols(), k)),
                    Par::Seq,
                    MemStack::new(&mut mem),
                    Default::default(),
                )
                .unwrap();
                (u, sigma, v)
            })
        });
        group.bench_function(BenchmarkId::new("q_times_u", &p), |b| {
            b.iter(|| &q * u.columns(0, r))
        });
    }
    group.finish();
}

/// Algebra and evaluation: Gram matrix, inner product, addition, batch point
/// evaluation (2000 points).
fn bench_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("lowrank_ops");
    group.sample_size(10);
    let xs: Vec<[f64; 4]> = (0..2000)
        .map(|i| {
            let t = i as f64;
            [(0.37 * t).sin(), (0.53 * t).cos(), (0.71 * t).sin(), (0.97 * t).cos()]
        })
        .collect();
    for &(n, m, r) in CASES {
        let (s, o) = (random_sum(n, m), random_sum(n, m));
        let p = id(n, m, r);
        group.bench_function(BenchmarkId::new("gram_left", &p), |b| b.iter(|| s.gram_left()));
        group.bench_function(BenchmarkId::new("inner", &p), |b| b.iter(|| s.inner(&o)));
        group.bench_function(BenchmarkId::new("add", &p), |b| b.iter(|| s.add(&o)));
        group.bench_function(BenchmarkId::new("get_values", &p), |b| {
            let mut out = vec![0.0; xs.len()];
            b.iter(|| s.get_values(&xs, &mut out))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compress, bench_compress_steps, bench_ops);
criterion_main!(benches);
