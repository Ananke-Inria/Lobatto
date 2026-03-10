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

/// Transpose a batch of elements from scalar AoS to vectorized SoA layout.
///
/// Each element contains `groups` consecutive groups of `dofs_per_group` dofs.
/// - AoS (scalar) index: `j * groups * dofs_per_group + i * dofs_per_group + k`
/// - SoA (vec)    index: `i * n * dofs_per_group + k * n + j`
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

// x-differentiation on a single Q3×Q3 element (16 dofs): HC hardcoded operator vs GLE matrix.
fn bench_dx_q33(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q33");
    group.throughput(Throughput::Elements(16));

    let v_in_hc = seeded_vec(16);
    let mut v_out_hc = vec![0.0_f64; 16];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q33_q33(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(16));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 16]);
    let gle = Calculator::new(vec![(4, true), (4, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// x-differentiation on a single Q3×Q3×Q3 element (64 dofs): HC hardcoded operator vs GLE matrix.
fn bench_dx_q333(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q333");
    group.throughput(Throughput::Elements(64));

    let v_in_hc = seeded_vec(64);
    let mut v_out_hc = vec![0.0_f64; 64];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q333_q333(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(64));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 64]);
    let gle = Calculator::new(vec![(4, true), (4, true), (4, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// x-differentiation on a batch of 8 Q3 elements: scalar AoS loop vs vectorized SoA (N=8).
// Scalar layout : AoS — [a0,b0,c0,d0, a1,b1,c1,d1, ...]
// Vectorized layout : SoA — [a0..a7, b0..b7, c0..c7, d0..d7]
fn bench_dx_q3_batch(c: &mut Criterion) {
    const N: usize = 8;
    let mut group = c.benchmark_group("dx_q3_batch8");
    group.throughput(Throughput::Elements(N as u64));

    let v_in_scalar = seeded_vec(4 * N);
    let mut v_out_scalar = vec![0.0_f64; 4 * N];
    group.bench_function("scalar", |b| {
        b.iter(|| {
            for i in 0..N {
                let start = 4 * i;
                dx_q3_q3(
                    black_box(&v_in_scalar[start..start + 4]),
                    &mut v_out_scalar[start..start + 4],
                );
            }
        })
    });

    let v_in_vec = aos_to_soa(&v_in_scalar, N, 1, 4);
    let mut v_out_vec = vec![0.0_f64; 4 * N];
    group.bench_function("vec", |b| {
        b.iter(|| dx_q3_q3_vec::<N>(black_box(&v_in_vec), &mut v_out_vec))
    });

    group.finish();
}

// x-differentiation on a batch of 8 Q3×Q3×Q3 elements (64 dofs each): scalar vs vectorized (N=8).
fn bench_dx_q333_batch(c: &mut Criterion) {
    const N: usize = 8;
    let mut group = c.benchmark_group("dx_q333_batch8");
    group.throughput(Throughput::Elements((64 * N) as u64));

    let v_in_scalar = seeded_vec(64 * N);
    let mut v_out_scalar = vec![0.0_f64; 64 * N];
    group.bench_function("scalar", |b| {
        b.iter(|| {
            for i in 0..N {
                let start = 64 * i;
                dx_q333_q333(
                    black_box(&v_in_scalar[start..start + 64]),
                    &mut v_out_scalar[start..start + 64],
                );
            }
        })
    });

    let v_in_vec = aos_to_soa(&v_in_scalar, N, 16, 4);
    let mut v_out_vec = vec![0.0_f64; 64 * N];
    group.bench_function("vec", |b| {
        b.iter(|| dx_q333_q333_vec::<N>(black_box(&v_in_vec), &mut v_out_vec))
    });

    group.finish();
}

// Sweep batch size N = 2, 4, 8, 16 to find where vectorization of dx_q3 becomes beneficial.
fn bench_dx_q3_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q3_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n));

            let v_in_s = seeded_vec(4 * $n);
            let mut v_out_s = vec![0.0_f64; 4 * $n];
            group.bench_function(BenchmarkId::new("scalar", $n), |b| {
                b.iter(|| {
                    for i in 0..$n {
                        let start = 4 * i;
                        dx_q3_q3(
                            black_box(&v_in_s[start..start + 4]),
                            &mut v_out_s[start..start + 4],
                        );
                    }
                })
            });

            let v_in_v = aos_to_soa(&v_in_s, $n, 1, 4);
            let mut v_out_v = vec![0.0_f64; 4 * $n];
            group.bench_function(BenchmarkId::new("vec", $n), |b| {
                b.iter(|| dx_q3_q3_vec::<$n>(black_box(&v_in_v), &mut v_out_v))
            });
        }};
    }

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);

    group.finish();
}

// x-differentiation on a batch of 8 Q3×Q3 elements (16 dofs each): scalar vs vectorized (N=8).
fn bench_dx_q33_batch(c: &mut Criterion) {
    const N: usize = 8;
    let mut group = c.benchmark_group("dx_q33_batch8");
    group.throughput(Throughput::Elements((16 * N) as u64));

    let v_in_scalar = seeded_vec(16 * N);
    let mut v_out_scalar = vec![0.0_f64; 16 * N];
    group.bench_function("scalar", |b| {
        b.iter(|| {
            for i in 0..N {
                let start = 16 * i;
                dx_q33_q33(
                    black_box(&v_in_scalar[start..start + 16]),
                    &mut v_out_scalar[start..start + 16],
                );
            }
        })
    });

    let v_in_vec = aos_to_soa(&v_in_scalar, N, 4, 4);
    let mut v_out_vec = vec![0.0_f64; 16 * N];
    group.bench_function("vec", |b| {
        b.iter(|| dx_q33_q33_vec::<N>(black_box(&v_in_vec), &mut v_out_vec))
    });

    group.finish();
}

// x-differentiation on a batch of 8 Q4×Q4 elements (25 dofs each): scalar vs vectorized (N=8).
fn bench_dx_q44_batch(c: &mut Criterion) {
    const N: usize = 8;
    let mut group = c.benchmark_group("dx_q44_batch8");
    group.throughput(Throughput::Elements((25 * N) as u64));

    let v_in_scalar = seeded_vec(25 * N);
    let mut v_out_scalar = vec![0.0_f64; 25 * N];
    group.bench_function("scalar", |b| {
        b.iter(|| {
            for i in 0..N {
                let start = 25 * i;
                dx_q44_q44(
                    black_box(&v_in_scalar[start..start + 25]),
                    &mut v_out_scalar[start..start + 25],
                );
            }
        })
    });

    let v_in_vec = aos_to_soa(&v_in_scalar, N, 5, 5);
    let mut v_out_vec = vec![0.0_f64; 25 * N];
    group.bench_function("vec", |b| {
        b.iter(|| dx_q44_q44_vec::<N>(black_box(&v_in_vec), &mut v_out_vec))
    });

    group.finish();
}

// Sweep batch size N = 2, 4, 8, 16 to find where vectorization of dx_q4 becomes beneficial.
fn bench_dx_q4_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q4_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n));

            let v_in_s = seeded_vec(5 * $n);
            let mut v_out_s = vec![0.0_f64; 5 * $n];
            group.bench_function(BenchmarkId::new("scalar", $n), |b| {
                b.iter(|| {
                    for i in 0..$n {
                        let start = 5 * i;
                        dx_q4_q4(
                            black_box(&v_in_s[start..start + 5]),
                            &mut v_out_s[start..start + 5],
                        );
                    }
                })
            });

            let v_in_v = aos_to_soa(&v_in_s, $n, 1, 5);
            let mut v_out_v = vec![0.0_f64; 5 * $n];
            group.bench_function(BenchmarkId::new("vec", $n), |b| {
                b.iter(|| dx_q4_q4_vec::<$n>(black_box(&v_in_v), &mut v_out_v))
            });
        }};
    }

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// x-differentiation on a single Q4×Q4 element (25 dofs): HC hardcoded operator vs GLE matrix.
fn bench_dx_q44(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q44");
    group.throughput(Throughput::Elements(25));

    let v_in_hc = seeded_vec(25);
    let mut v_out_hc = vec![0.0_f64; 25];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q44_q44(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(25));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 25]);
    let gle = Calculator::new(vec![(5, true), (5, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// x-differentiation on a single Q4×Q4×Q4 element (125 dofs): HC hardcoded operator vs GLE matrix.
fn bench_dx_q444(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q444");
    group.throughput(Throughput::Elements(125));

    let v_in_hc = seeded_vec(125);
    let mut v_out_hc = vec![0.0_f64; 125];
    group.bench_function("hc", |b| {
        b.iter(|| dx_q444_q444(black_box(&v_in_hc), &mut v_out_hc))
    });

    let v_in_gle: DVector<f64> = DVector::from_vec(seeded_vec(125));
    let mut v_out_gle: DVector<f64> = DVector::from_vec(vec![0.0; 125]);
    let gle = Calculator::new(vec![(5, true), (5, true), (5, true)]);
    group.bench_function("gle", |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    group.finish();
}

// x-differentiation on a batch of 8 Q4 elements: scalar AoS loop vs vectorized SoA (N=8).
fn bench_dx_q4_batch(c: &mut Criterion) {
    const N: usize = 8;
    let mut group = c.benchmark_group("dx_q4_batch8");
    group.throughput(Throughput::Elements(N as u64));

    let v_in_scalar = seeded_vec(5 * N);
    let mut v_out_scalar = vec![0.0_f64; 5 * N];
    group.bench_function("scalar", |b| {
        b.iter(|| {
            for i in 0..N {
                let start = 5 * i;
                dx_q4_q4(
                    black_box(&v_in_scalar[start..start + 5]),
                    &mut v_out_scalar[start..start + 5],
                );
            }
        })
    });

    let v_in_vec = aos_to_soa(&v_in_scalar, N, 1, 5);
    let mut v_out_vec = vec![0.0_f64; 5 * N];
    group.bench_function("vec", |b| {
        b.iter(|| dx_q4_q4_vec::<N>(black_box(&v_in_vec), &mut v_out_vec))
    });

    group.finish();
}

// x-differentiation on a batch of 8 Q4×Q4×Q4 elements (125 dofs each): scalar vs vectorized (N=8).
fn bench_dx_q444_batch(c: &mut Criterion) {
    const N: usize = 8;
    let mut group = c.benchmark_group("dx_q444_batch8");
    group.throughput(Throughput::Elements((125 * N) as u64));

    let v_in_scalar = seeded_vec(125 * N);
    let mut v_out_scalar = vec![0.0_f64; 125 * N];
    group.bench_function("scalar", |b| {
        b.iter(|| {
            for i in 0..N {
                let start = 125 * i;
                dx_q444_q444(
                    black_box(&v_in_scalar[start..start + 125]),
                    &mut v_out_scalar[start..start + 125],
                );
            }
        })
    });

    let v_in_vec = aos_to_soa(&v_in_scalar, N, 25, 5);
    let mut v_out_vec = vec![0.0_f64; 125 * N];
    group.bench_function("vec", |b| {
        b.iter(|| dx_q444_q444_vec::<N>(black_box(&v_in_vec), &mut v_out_vec))
    });

    group.finish();
}

// Sweep batch size N = 2, 4, 8, 16, 32 to find where vectorization of dx_q44 becomes beneficial.
fn bench_dx_q44_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q44_batch_scaling");

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 25));

            let v_in_s = seeded_vec(25 * $n);
            let mut v_out_s = vec![0.0_f64; 25 * $n];
            group.bench_function(BenchmarkId::new("scalar", $n), |b| {
                b.iter(|| {
                    for i in 0..$n {
                        let start = 25 * i;
                        dx_q44_q44(
                            black_box(&v_in_s[start..start + 25]),
                            &mut v_out_s[start..start + 25],
                        );
                    }
                })
            });

            let v_in_v = aos_to_soa(&v_in_s, $n, 5, 5);
            let mut v_out_v = vec![0.0_f64; 25 * $n];
            group.bench_function(BenchmarkId::new("vec", $n), |b| {
                b.iter(|| dx_q44_q44_vec::<$n>(black_box(&v_in_v), &mut v_out_v))
            });
        }};
    }

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
    bench_dx_q3_batch,
    bench_dx_q33_batch,
    bench_dx_q333_batch,
    bench_dx_q3_scaling,
    bench_dx_q44,
    bench_dx_q444,
    bench_dx_q4_batch,
    bench_dx_q44_batch,
    bench_dx_q444_batch,
    bench_dx_q4_scaling,
    bench_dx_q44_scaling,
);
criterion_main!(benches);
