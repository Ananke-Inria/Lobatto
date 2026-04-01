use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra::DVector;
use quadratix::hot::*;
use std::f64::consts::PI;
use std::hint::black_box;

/// Benchmark `solve_in_place` for the problem (M + K) u = f on [0, 1]
/// using u(x) = exp(sin(2*pi*x)) as the test function.
/// Sweeps over polynomial order r (single element, so ndof = r).
/// The 1D solver does not use Rayon, so no thread-count loop.
fn bench_hot_solve_in_place(c: &mut Criterion) {
    let l = 1.0_f64;
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI / l;

    let rhs_f = |x: f64| {
        let s = (w * x).sin();
        let cv = (w * x).cos();
        let u = s.exp();
        alpha * u - beta * w * w * (cv * cv - s) * u
    };

    let r_values: &[usize] = &[4, 8, 16, 32, 64];

    let mut group = c.benchmark_group("hot_solve_in_place");

    for &r in r_values {
        group.throughput(Throughput::Elements(r as u64));

        let solver = Hot1DLaplaceSolver::new(l, 0.0, r, alpha, beta);
        let x = solver.get_x();

        let f_init = DVector::from_fn(r, |i, _| rhs_f(x[i]));

        group.bench_with_input(
            BenchmarkId::from_parameter(r),
            &(solver, f_init),
            |b, (solver, f_init)| {
                b.iter(|| {
                    let mut f = black_box(f_init.clone());
                    solver.solve_in_place(&mut f);
                    f
                })
            },
        );
    }

    group.finish();
}

fn bench_hot_solve_in_place_2d(c: &mut Criterion) {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;

    let r_values: &[usize] = &[4, 8, 16, 32, 64];
    let thread_counts: &[usize] = &[1, 4, 8];

    let mut group = c.benchmark_group("hot_solve_in_place_2d");

    for &r in r_values {
        let ndof = r * r;
        group.throughput(Throughput::Elements(ndof as u64));

        let solver = HotNDLaplaceSolver::<2>::new([1.0; 2], [0.0; 2], [r; 2], alpha, beta);
        let x = solver.get_x();

        let f_init: Vec<f64> = (0..ndof)
            .map(|i| {
                let (xi, yi) = (x[i][0], x[i][1]);
                let rhs_1d = |t: f64| {
                    let s = (w * t).sin();
                    let cv = (w * t).cos();
                    let u = s.exp();
                    alpha * u - beta * w * w * (cv * cv - s) * u
                };
                let ux = (w * xi).sin().exp();
                let uy = (w * yi).sin().exp();
                rhs_1d(xi) * uy + ux * rhs_1d(yi) - alpha * ux * uy
            })
            .collect();

        for &threads in thread_counts {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            group.bench_function(BenchmarkId::new(format!("t={threads}"), r), |b| {
                b.iter(|| {
                    let mut f = black_box(f_init.clone());
                    pool.install(|| solver.solve_in_place(&mut f));
                    f
                })
            });
        }
    }

    group.finish();
}

fn bench_hot_solve_in_place_3d(c: &mut Criterion) {
    let alpha = 1.0_f64;
    let beta = 1.0_f64;
    let w = 2.0 * PI;

    let r_values: &[usize] = &[4, 8, 16, 32, 64];
    let thread_counts: &[usize] = &[1, 4, 8];

    let mut group = c.benchmark_group("hot_solve_in_place_3d");

    for &r in r_values {
        let ndof = r * r * r;
        group.throughput(Throughput::Elements(ndof as u64));

        let solver = HotNDLaplaceSolver::<3>::new([1.0; 3], [0.0; 3], [r; 3], alpha, beta);
        let x = solver.get_x();

        let f_init: Vec<f64> = (0..ndof)
            .map(|i| {
                let (xi, yi, zi) = (x[i][0], x[i][1], x[i][2]);
                let rhs_1d = |t: f64| {
                    let s = (w * t).sin();
                    let cv = (w * t).cos();
                    let u = s.exp();
                    alpha * u - beta * w * w * (cv * cv - s) * u
                };
                let (ux, uy, uz) = (
                    (w * xi).sin().exp(),
                    (w * yi).sin().exp(),
                    (w * zi).sin().exp(),
                );
                rhs_1d(xi) * uy * uz + ux * rhs_1d(yi) * uz + ux * uy * rhs_1d(zi)
                    - 2.0 * alpha * ux * uy * uz
            })
            .collect();

        for &threads in thread_counts {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            group.bench_function(BenchmarkId::new(format!("t={threads}"), r), |b| {
                b.iter(|| {
                    let mut f = black_box(f_init.clone());
                    pool.install(|| solver.solve_in_place(&mut f));
                    f
                })
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hot_solve_in_place,
    bench_hot_solve_in_place_2d,
    bench_hot_solve_in_place_3d
);
criterion_main!(benches);
