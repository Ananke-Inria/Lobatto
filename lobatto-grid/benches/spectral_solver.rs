use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lobatto_grid::spectral::solver::*;
use std::f64::consts::PI;
use std::hint::black_box;

/// Right-hand side of (M + K) u = f on [0, 1] for u(x) = exp(sin(2πx)).
fn rhs_1d(x: f64) -> f64 {
    let w = 2.0 * PI;
    let s = (w * x).sin();
    let c = (w * x).cos();
    let u = s.exp();
    u - w * w * (c * c - s) * u
}

fn u_1d(x: f64) -> f64 {
    (2.0 * PI * x).sin().exp()
}

/// `(n, r)` pairs swept by every benchmark: `n·r` unknowns per direction.
const CASES: &[(usize, usize)] = &[(1, 8), (1, 16), (2, 8), (4, 8), (1, 64), (8, 8)];

/// Benchmark `solve_in_place` for (M + K) u = f on [0, 1].
/// The 1D solver does not use Rayon, so no thread-count loop.
fn bench_solve_in_place(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_solve_in_place");

    for &(n, r) in CASES {
        let solver = Poisson::new(1.0, 0.0, n, r, BoundaryCondition::Periodic);
        let x = solver.get_x();
        group.throughput(Throughput::Elements(x.len() as u64));
        let f_init: Vec<f64> = x.iter().map(|&xi| rhs_1d(xi)).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{n}_r{r}")),
            &(solver, f_init),
            |b, (solver, f_init)| {
                b.iter(|| {
                    let mut f = black_box(f_init.clone());
                    solver.solve_in_place(&mut f, 1.0, 1.0);
                    f
                })
            },
        );
    }

    group.finish();
}

fn bench_solve_in_place_2d(c: &mut Criterion) {
    let thread_counts: &[usize] = &[1, 4, 8];
    let mut group = c.benchmark_group("hot_solve_in_place_2d");

    for &(n, r) in CASES {
        let solver = PoissonND::<2>::new(
            [1.0; 2],
            [0.0; 2],
            [n; 2],
            [r; 2],
            [BoundaryCondition::Periodic; 2],
        );
        let x = solver.get_x();
        group.throughput(Throughput::Elements(x.len() as u64));

        // (M + K) u = f for u = u_1d(x) u_1d(y).
        let f_init: Vec<f64> = x
            .iter()
            .map(|&[xi, yi]| rhs_1d(xi) * u_1d(yi) + u_1d(xi) * rhs_1d(yi) - u_1d(xi) * u_1d(yi))
            .collect();

        for &threads in thread_counts {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            group.bench_function(
                BenchmarkId::new(format!("t={threads}"), format!("n{n}_r{r}")),
                |b| {
                    b.iter(|| {
                        let mut f = black_box(f_init.clone());
                        pool.install(|| solver.solve_in_place(&mut f, 1.0, 1.0));
                        f
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_solve_in_place_3d(c: &mut Criterion) {
    let thread_counts: &[usize] = &[1, 4, 8];
    let mut group = c.benchmark_group("hot_solve_in_place_3d");

    for &(n, r) in CASES {
        let solver = PoissonND::<3>::new(
            [1.0; 3],
            [0.0; 3],
            [n; 3],
            [r; 3],
            [BoundaryCondition::Periodic; 3],
        );
        let x = solver.get_x();
        group.throughput(Throughput::Elements(x.len() as u64));

        let f_init: Vec<f64> = x
            .iter()
            .map(|&[xi, yi, zi]| {
                let (ux, uy, uz) = (u_1d(xi), u_1d(yi), u_1d(zi));
                rhs_1d(xi) * uy * uz + ux * rhs_1d(yi) * uz + ux * uy * rhs_1d(zi)
                    - 2.0 * ux * uy * uz
            })
            .collect();

        for &threads in thread_counts {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            group.bench_function(
                BenchmarkId::new(format!("t={threads}"), format!("n{n}_r{r}")),
                |b| {
                    b.iter(|| {
                        let mut f = black_box(f_init.clone());
                        pool.install(|| solver.solve_in_place(&mut f, 1.0, 1.0));
                        f
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_solve_in_place,
    bench_solve_in_place_2d,
    bench_solve_in_place_3d
);
criterion_main!(benches);
