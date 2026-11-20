use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use lobatto::collocation::*;
use lobatto::collocation_hc::*;
use lobatto::quadrature::gauss_lobatto_on_unit_interval;
use lobatto::utilities::lagrangian_derivative_interpolation;
use std::time::Duration;

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

// Sweep batch size N = 1, 2, 4, 8, 16 for dx_q3.
// At N=1 a Basis::diff reference is included.
fn bench_dx_q3(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q3_batch");

    // N=1: hc vs Basis::diff
    group.throughput(Throughput::Elements(4));
    let v_in_1 = seeded_vec(4);
    let mut v_out_1 = vec![0.0_f64; 4];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_q3_q3::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(4);
    let mut v_out_gle: Vec<f64> = vec![0.0; 4];
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff", 1), |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

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

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16 for dx_q33.
// At N=1 a Basis::diff reference is included.
fn bench_dx_q33(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q33_batch");

    // N=1: hc vs Basis::diff
    group.throughput(Throughput::Elements(16));
    let v_in_1 = seeded_vec(16);
    let mut v_out_1 = vec![0.0_f64; 16];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_q33_q33::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(16);
    let mut v_out_gle: Vec<f64> = vec![0.0; 16];
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff", 1), |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

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

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q4.
// At N=1 a Basis::diff reference is included.
fn bench_dx_q4(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q4_batch");

    // N=1: hc vs Basis::diff
    group.throughput(Throughput::Elements(5));
    let v_in_1 = seeded_vec(5);
    let mut v_out_1 = vec![0.0_f64; 5];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_q4_q4::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(5);
    let mut v_out_gle: Vec<f64> = vec![0.0; 5];
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff", 1), |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

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

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q44.
// At N=1 a Basis::diff reference is included.
fn bench_dx_q44(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q44_batch");

    // N=1: hc vs Basis::diff
    group.throughput(Throughput::Elements(25));
    let v_in_1 = seeded_vec(25);
    let mut v_out_1 = vec![0.0_f64; 25];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_q44_q44::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(25);
    let mut v_out_gle: Vec<f64> = vec![0.0; 25];
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff", 1), |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

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

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q5.
// At N=1 a Basis::diff reference is included.
fn bench_dx_q5(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q5_batch");

    // N=1: hc vs Basis::diff
    group.throughput(Throughput::Elements(6));
    let v_in_1 = seeded_vec(6);
    let mut v_out_1 = vec![0.0_f64; 6];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_q5_q5::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(6);
    let mut v_out_gle: Vec<f64> = vec![0.0; 6];
    let gle = CollocationBasis::new(vec![(6, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff", 1), |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

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

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q55.
// At N=1 a Basis::diff reference is included.
fn bench_dx_q55(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q55_batch");

    // N=1: hc vs Basis::diff
    group.throughput(Throughput::Elements(36));
    let v_in_1 = seeded_vec(36);
    let mut v_out_1 = vec![0.0_f64; 36];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_q55_q55::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(36);
    let mut v_out_gle: Vec<f64> = vec![0.0; 36];
    let gle = CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff", 1), |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

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

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q6.
// At N=1 a Basis::diff reference is included.
fn bench_dx_q6(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q6_batch");

    // N=1: hc vs Basis::diff
    group.throughput(Throughput::Elements(7));
    let v_in_1 = seeded_vec(7);
    let mut v_out_1 = vec![0.0_f64; 7];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_q6_q6::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(7);
    let mut v_out_gle: Vec<f64> = vec![0.0; 7];
    let gle = CollocationBasis::new(vec![(7, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff", 1), |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

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

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

// Sweep batch size N = 1, 2, 4, 8, 16, 32 for dx_q66.
// At N=1 a Basis::diff reference is included.
fn bench_dx_q66(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_q66_batch");

    // N=1: hc vs Basis::diff
    group.throughput(Throughput::Elements(49));
    let v_in_1 = seeded_vec(49);
    let mut v_out_1 = vec![0.0_f64; 49];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_q66_q66::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(49);
    let mut v_out_gle: Vec<f64> = vec![0.0; 49];
    let gle = CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff", 1), |b| {
        b.iter(|| gle.diff(0, black_box(&v_in_gle), &mut v_out_gle))
    });

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

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);

    group.finish();
}

fn bench_dx_t_q3(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_t_q3_batch");

    group.throughput(Throughput::Elements(4));
    let v_in_1 = seeded_vec(4);
    let mut v_out_1 = vec![0.0_f64; 4];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_t_q3_q3::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(4);
    let mut v_out_gle: Vec<f64> = vec![0.0; 4];
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff_transpose", 1), |b| {
        b.iter(|| gle.diff_transpose(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 4));
            let v_in = aos_to_soa(&seeded_vec(4 * $n), $n, 1, 4);
            let mut v_out = vec![0.0_f64; 4 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_t_q3_q3::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);

    group.finish();
}

fn bench_dx_t_q33(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_t_q33_batch");

    group.throughput(Throughput::Elements(16));
    let v_in_1 = seeded_vec(16);
    let mut v_out_1 = vec![0.0_f64; 16];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_t_q33_q33::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(16);
    let mut v_out_gle: Vec<f64> = vec![0.0; 16];
    let gle = CollocationBasis::new(vec![(4, Gauss::Lobatto), (4, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff_transpose", 1), |b| {
        b.iter(|| gle.diff_transpose(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 16));
            let v_in = aos_to_soa(&seeded_vec(16 * $n), $n, 4, 4);
            let mut v_out = vec![0.0_f64; 16 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_t_q33_q33::<$n>(black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_n!(2);
    bench_n!(4);
    bench_n!(8);
    bench_n!(16);

    group.finish();
}

fn bench_dx_t_q4(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_t_q4_batch");

    group.throughput(Throughput::Elements(5));
    let v_in_1 = seeded_vec(5);
    let mut v_out_1 = vec![0.0_f64; 5];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_t_q4_q4::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(5);
    let mut v_out_gle: Vec<f64> = vec![0.0; 5];
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff_transpose", 1), |b| {
        b.iter(|| gle.diff_transpose(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 5));
            let v_in = aos_to_soa(&seeded_vec(5 * $n), $n, 1, 5);
            let mut v_out = vec![0.0_f64; 5 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_t_q4_q4::<$n>(black_box(&v_in), &mut v_out))
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

fn bench_dx_t_q44(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_t_q44_batch");

    group.throughput(Throughput::Elements(25));
    let v_in_1 = seeded_vec(25);
    let mut v_out_1 = vec![0.0_f64; 25];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_t_q44_q44::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(25);
    let mut v_out_gle: Vec<f64> = vec![0.0; 25];
    let gle = CollocationBasis::new(vec![(5, Gauss::Lobatto), (5, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff_transpose", 1), |b| {
        b.iter(|| gle.diff_transpose(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 25));
            let v_in = aos_to_soa(&seeded_vec(25 * $n), $n, 5, 5);
            let mut v_out = vec![0.0_f64; 25 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_t_q44_q44::<$n>(black_box(&v_in), &mut v_out))
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

fn bench_dx_t_q5(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_t_q5_batch");

    group.throughput(Throughput::Elements(6));
    let v_in_1 = seeded_vec(6);
    let mut v_out_1 = vec![0.0_f64; 6];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_t_q5_q5::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(6);
    let mut v_out_gle: Vec<f64> = vec![0.0; 6];
    let gle = CollocationBasis::new(vec![(6, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff_transpose", 1), |b| {
        b.iter(|| gle.diff_transpose(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 6));
            let v_in = aos_to_soa(&seeded_vec(6 * $n), $n, 1, 6);
            let mut v_out = vec![0.0_f64; 6 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_t_q5_q5::<$n>(black_box(&v_in), &mut v_out))
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

fn bench_dx_t_q55(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_t_q55_batch");

    group.throughput(Throughput::Elements(36));
    let v_in_1 = seeded_vec(36);
    let mut v_out_1 = vec![0.0_f64; 36];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_t_q55_q55::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(36);
    let mut v_out_gle: Vec<f64> = vec![0.0; 36];
    let gle = CollocationBasis::new(vec![(6, Gauss::Lobatto), (6, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff_transpose", 1), |b| {
        b.iter(|| gle.diff_transpose(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 36));
            let v_in = aos_to_soa(&seeded_vec(36 * $n), $n, 6, 6);
            let mut v_out = vec![0.0_f64; 36 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_t_q55_q55::<$n>(black_box(&v_in), &mut v_out))
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

fn bench_dx_t_q6(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_t_q6_batch");

    group.throughput(Throughput::Elements(7));
    let v_in_1 = seeded_vec(7);
    let mut v_out_1 = vec![0.0_f64; 7];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_t_q6_q6::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(7);
    let mut v_out_gle: Vec<f64> = vec![0.0; 7];
    let gle = CollocationBasis::new(vec![(7, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff_transpose", 1), |b| {
        b.iter(|| gle.diff_transpose(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 7));
            let v_in = aos_to_soa(&seeded_vec(7 * $n), $n, 1, 7);
            let mut v_out = vec![0.0_f64; 7 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_t_q6_q6::<$n>(black_box(&v_in), &mut v_out))
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

fn bench_dx_t_q66(c: &mut Criterion) {
    let mut group = c.benchmark_group("dx_t_q66_batch");

    group.throughput(Throughput::Elements(49));
    let v_in_1 = seeded_vec(49);
    let mut v_out_1 = vec![0.0_f64; 49];
    group.bench_function(BenchmarkId::new("hc", 1), |b| {
        b.iter(|| dx_t_q66_q66::<1>(black_box(&v_in_1), &mut v_out_1))
    });
    let v_in_gle: Vec<f64> = seeded_vec(49);
    let mut v_out_gle: Vec<f64> = vec![0.0; 49];
    let gle = CollocationBasis::new(vec![(7, Gauss::Lobatto), (7, Gauss::Lobatto)]);
    group.bench_function(BenchmarkId::new("Basis::diff_transpose", 1), |b| {
        b.iter(|| gle.diff_transpose(0, black_box(&v_in_gle), &mut v_out_gle))
    });

    macro_rules! bench_n {
        ($n:literal) => {{
            group.throughput(Throughput::Elements($n * 49));
            let v_in = aos_to_soa(&seeded_vec(49 * $n), $n, 7, 7);
            let mut v_out = vec![0.0_f64; 49 * $n];
            group.bench_function(BenchmarkId::new("hc", $n), |b| {
                b.iter(|| dx_t_q66_q66::<$n>(black_box(&v_in), &mut v_out))
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

fn bench_eval(c: &mut Criterion) {
    const N_PTS: usize = 1000;

    let mut group = c.benchmark_group("eval");
    group.throughput(Throughput::Elements(N_PTS as u64));

    // Query points in [0, 1)^d — seeded_vec gives (-1, 1), shift to (0, 1).
    let raw: Vec<f64> = {
        let mut state: u64 = 0xdeadbeefcafe1234;
        (0..N_PTS * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (f64::from_bits((state >> 12) | 0x3FF0000000000000) - 1.0) * 0.98 + 0.01
            })
            .collect()
    };
    let xs: Vec<Vec<f64>> = raw.chunks(3).map(|c| c.to_vec()).collect();

    for n_nodes in [4usize, 5, 6, 7] {
        let basis = CollocationBasis::new(vec![(n_nodes, Gauss::Lobatto); 3]);
        let u = seeded_vec(n_nodes.pow(3));
        let id = BenchmarkId::new("Q_3d", format!("n={n_nodes}"));
        group.bench_function(id, |b| {
            b.iter(|| basis.eval(black_box(&xs), black_box(&u)))
        });
    }

    group.finish();
}

// ── Generic-order kernels ──────────────────────────────────────────────

/// D[j][i] = L_i'(ξ_j) on the GLL nodes of [0, 1].
fn gll_diff_matrix<const NP: usize>() -> [[f64; NP]; NP] {
    let (x, _) = gauss_lobatto_on_unit_interval(NP);
    core::array::from_fn(|j| {
        core::array::from_fn(|i| lagrangian_derivative_interpolation(&x, x[j], i))
    })
}

// Generic vs hardcoded 3D kernels, NP = 4..7 nodes, batch N = 4, 8.
// dx (stride 1), dz (stride NP², runtime in the generic kernel) and dz_t.
fn bench_generic_vs_hc_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("generic_vs_hc_3d");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    macro_rules! bench_np_n {
        ($np:literal, $n:literal, $dx:ident, $dz:ident, $dz_t:ident) => {{
            let dofs = $np * $np * $np * $n;
            group.throughput(Throughput::Elements(dofs as u64));
            let d = gll_diff_matrix::<$np>();
            let v_in = seeded_vec(dofs);
            let mut v_out = vec![0.0_f64; dofs];
            let p = format!("np{}_n{}", $np, $n);
            group.bench_function(BenchmarkId::new("hc_dx", &p), |b| {
                b.iter(|| $dx::<$n>(black_box(&v_in), &mut v_out))
            });
            group.bench_function(BenchmarkId::new("generic_dx", &p), |b| {
                b.iter(|| dx_qnnn_qnnn::<$np, $n>(black_box(&d), black_box(&v_in), &mut v_out))
            });
            group.bench_function(BenchmarkId::new("hc_dz", &p), |b| {
                b.iter(|| $dz::<$n>(black_box(&v_in), &mut v_out))
            });
            group.bench_function(BenchmarkId::new("generic_dz", &p), |b| {
                b.iter(|| dz_qnnn_qnnn::<$np, $n>(black_box(&d), black_box(&v_in), &mut v_out))
            });
            group.bench_function(BenchmarkId::new("hc_dz_t", &p), |b| {
                b.iter(|| $dz_t::<$n>(black_box(&v_in), &mut v_out))
            });
            group.bench_function(BenchmarkId::new("generic_dz_t", &p), |b| {
                b.iter(|| dz_t_qnnn_qnnn::<$np, $n>(black_box(&d), black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_np_n!(4, 4, dx_q333_q333, dz_q333_q333, dz_t_q333_q333);
    bench_np_n!(4, 8, dx_q333_q333, dz_q333_q333, dz_t_q333_q333);
    bench_np_n!(5, 4, dx_q444_q444, dz_q444_q444, dz_t_q444_q444);
    bench_np_n!(5, 8, dx_q444_q444, dz_q444_q444, dz_t_q444_q444);
    bench_np_n!(6, 4, dx_q555_q555, dz_q555_q555, dz_t_q555_q555);
    bench_np_n!(6, 8, dx_q555_q555, dz_q555_q555, dz_t_q555_q555);
    bench_np_n!(7, 4, dx_q666_q666, dz_q666_q666, dz_t_q666_q666);
    bench_np_n!(7, 8, dx_q666_q666, dz_q666_q666, dz_t_q666_q666);

    group.finish();
}

// Generic 3D kernels for the orders without a hardcoded version
// (NP = 2, 3, 8, 9 nodes), batch N = 4, 8.
fn bench_generic_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("generic_3d");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    macro_rules! bench_np_n {
        ($np:literal, $n:literal) => {{
            let dofs = $np * $np * $np * $n;
            group.throughput(Throughput::Elements(dofs as u64));
            let d = gll_diff_matrix::<$np>();
            let v_in = seeded_vec(dofs);
            let mut v_out = vec![0.0_f64; dofs];
            let p = format!("np{}_n{}", $np, $n);
            group.bench_function(BenchmarkId::new("dx", &p), |b| {
                b.iter(|| dx_qnnn_qnnn::<$np, $n>(black_box(&d), black_box(&v_in), &mut v_out))
            });
            group.bench_function(BenchmarkId::new("dz_t", &p), |b| {
                b.iter(|| dz_t_qnnn_qnnn::<$np, $n>(black_box(&d), black_box(&v_in), &mut v_out))
            });
        }};
    }

    bench_np_n!(2, 4);
    bench_np_n!(2, 8);
    bench_np_n!(3, 4);
    bench_np_n!(3, 8);
    bench_np_n!(8, 4);
    bench_np_n!(8, 8);
    bench_np_n!(9, 4);
    bench_np_n!(9, 8);

    group.finish();
}

criterion_group!(
    benches,
    bench_dx_q3,
    bench_dx_q33,
    bench_dx_t_q3,
    bench_dx_t_q33,
    bench_dx_q4,
    bench_dx_q44,
    bench_dx_t_q4,
    bench_dx_t_q44,
    bench_dx_q5,
    bench_dx_q55,
    bench_dx_t_q5,
    bench_dx_t_q55,
    bench_dx_q6,
    bench_dx_q66,
    bench_dx_t_q6,
    bench_dx_t_q66,
    bench_eval,
    bench_generic_vs_hc_3d,
    bench_generic_3d,
);
criterion_main!(benches);
