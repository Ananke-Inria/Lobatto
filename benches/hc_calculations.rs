use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use gg_lobatto::calculator::*;
use gg_lobatto::hc_calculations::*;
use nalgebra::DVector;

/// Xorshift64 with a fixed seed — produces the same sequence every run,
/// no external crate required.
fn seeded_vec(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x3f2a1b4c8e7d5690;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // map to (-1, 1)
            f64::from_bits((state >> 12) | 0x3FF0000000000000) * 2.0 - 3.0
        })
        .collect()
}

/// Transpose a batch of elements from AoS to SoA layout.
///
/// Each element contains `groups` consecutive groups of `dofs_per_group` dofs.
/// - AoS index: `j * groups * dofs_per_group + i * dofs_per_group + k`
/// - SoA index: `i * n * dofs_per_group + k * n + j`
fn aos_to_soa(aos: &[f64], n: usize, groups: usize, dofs_per_group: usize) -> Vec<f64> {
    let mut soa = vec![0.0_f64; n * groups * dofs_per_group];
    for j in 0..n {
        for i in 0..groups {
            for k in 0..dofs_per_group {
                soa[i * n * dofs_per_group + k * n + j] =
                    aos[j * groups * dofs_per_group + i * dofs_per_group + k];
            }
        }
    }
    soa
}

// x-differentiation on a single Q3×Q3 element (16 dofs): HC vs GLE matrix.
fn bench_dx_q33(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q33");
    group.throughput(Throughput::Elements(16));

    let v_in_hc = seeded_vec(16);
    let mut v_out_hc = vec![0.0_f64; 16];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q33_q33::<1>(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(16));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 16]);
    let gle = Calculator::new(vec![(4, true), (4, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// x-differentiation on a single Q3×Q3×Q3 element (64 dofs): HC vs GLE matrix.
fn bench_dx_q333(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q333");
    group.throughput(Throughput::Elements(64));

    let v_in_hc = seeded_vec(64);
    let mut v_out_hc = vec![0.0_f64; 64];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q333_q333::<1>(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(64));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 64]);
    let gle = Calculator::new(vec![(4, true), (4, true), (4, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// x-differentiation on a single Q4×Q4 element (25 dofs): HC vs GLE matrix.
fn bench_dx_q44(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q44");
    group.throughput(Throughput::Elements(25));

    let v_in_hc = seeded_vec(25);
    let mut v_out_hc = vec![0.0_f64; 25];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q44_q44::<1>(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(25));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 25]);
    let gle = Calculator::new(vec![(5, true), (5, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// x-differentiation on a single Q4×Q4×Q4 element (125 dofs): HC vs GLE matrix.
fn bench_dx_q444(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q444");
    group.throughput(Throughput::Elements(125));

    let v_in_hc = seeded_vec(125);
    let mut v_out_hc = vec![0.0_f64; 125];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q444_q444::<1>(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(125));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 125]);
    let gle = Calculator::new(vec![(5, true), (5, true), (5, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16 for dx_q3.
fn bench_dx_q3_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q3_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 4));
            let v_in = aos_to_soa(&seeded_vec(4 * $n), $n, 1, 4);
            let mut v_out = vec![0.0_f64; 4 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_q3_q3::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(1);
    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16 for dx_q33.
fn bench_dx_q33_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q33_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 16));
            let v_in = aos_to_soa(&seeded_vec(16 * $n), $n, 4, 4);
            let mut v_out = vec![0.0_f64; 16 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_q33_q33::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(1);
    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16 for dx_q4.
fn bench_dx_q4_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q4_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 5));
            let v_in = aos_to_soa(&seeded_vec(5 * $n), $n, 1, 5);
            let mut v_out = vec![0.0_f64; 5 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_q4_q4::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(1);
    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q44.
fn bench_dx_q44_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q44_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 25));
            let v_in = aos_to_soa(&seeded_vec(25 * $n), $n, 5, 5);
            let mut v_out = vec![0.0_f64; 25 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_q44_q44::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(1);
    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// x-differentiation on a single Q5×Q5 element (36 dofs): HC vs GLE matrix.
fn bench_dx_q55(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q55");
    group.throughput(Throughput::Elements(36));

    let v_in_hc = seeded_vec(36);
    let mut v_out_hc = vec![0.0_f64; 36];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q55_q55::<1>(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(36));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 36]);
    let gle = Calculator::new(vec![(6, true), (6, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// x-differentiation on a single Q5×Q5×Q5 element (216 dofs): HC vs GLE matrix.
fn bench_dx_q555(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q555");
    group.throughput(Throughput::Elements(216));

    let v_in_hc = seeded_vec(216);
    let mut v_out_hc = vec![0.0_f64; 216];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q555_q555::<1>(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(216));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 216]);
    let gle = Calculator::new(vec![(6, true), (6, true), (6, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q5.
fn bench_dx_q5_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q5_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 6));
            let v_in = aos_to_soa(&seeded_vec(6 * $n), $n, 1, 6);
            let mut v_out = vec![0.0_f64; 6 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_q5_q5::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(1);
    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q55.
fn bench_dx_q55_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q55_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 36));
            let v_in = aos_to_soa(&seeded_vec(36 * $n), $n, 6, 6);
            let mut v_out = vec![0.0_f64; 36 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_q55_q55::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(1);
    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// x-differentiation on a single Q6×Q6 element (49 dofs): HC vs GLE matrix.
fn bench_dx_q66(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q66");
    group.throughput(Throughput::Elements(49));

    let v_in_hc = seeded_vec(49);
    let mut v_out_hc = vec![0.0_f64; 49];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q66_q66::<1>(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(49));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 49]);
    let gle = Calculator::new(vec![(7, true), (7, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// x-differentiation on a single Q6×Q6×Q6 element (343 dofs): HC vs GLE matrix.
fn bench_dx_q666(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q666");
    group.throughput(Throughput::Elements(343));

    let v_in_hc = seeded_vec(343);
    let mut v_out_hc = vec![0.0_f64; 343];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q666_q666::<1>(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(343));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 343]);
    let gle = Calculator::new(vec![(7, true), (7, true), (7, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q6.
fn bench_dx_q6_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q6_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 7));
            let v_in = aos_to_soa(&seeded_vec(7 * $n), $n, 1, 7);
            let mut v_out = vec![0.0_f64; 7 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_q6_q6::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(1);
    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q66.
fn bench_dx_q66_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q66_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 49));
            let v_in = aos_to_soa(&seeded_vec(49 * $n), $n, 7, 7);
            let mut v_out = vec![0.0_f64; 49 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_q66_q66::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(1);
    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

criterion_group!(
    benches,
    bench_dx_q33,
    bench_dx_q333,
    bench_dx_q44,
    bench_dx_q444,
    bench_dx_q55,
    bench_dx_q555,
    bench_dx_q66,
    bench_dx_q666,
    bench_dx_q3_scaling,
    bench_dx_q33_scaling,
    bench_dx_q4_scaling,
    bench_dx_q44_scaling,
    bench_dx_q5_scaling,
    bench_dx_q55_scaling,
    bench_dx_q6_scaling,
    bench_dx_q66_scaling,
);
criterion_main!(benches);
