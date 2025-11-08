use criterion::{criterion_group, criterion_main, Criterion};

use gg_lobatto::hc_interpolation::*;
use gg_lobatto::interpolation::*;
use nalgebra::DVector;

fn bench_hc_interpolation(c: &mut Criterion) {
    let v_in = vec![1.0; 16];
    let mut v_out = vec![0.0; 16];

    c.bench_function("hc_interpolation", |b| {
        b.iter(|| {
            dx_q33_q33(&v_in, &mut v_out);
        })
    });
}

fn bench_gle_interpolation(c: &mut Criterion) {
    let v_in: DVector<f64> = DVector::from_vec(vec![1.0; 16]);
    let mut v_out: DVector<f64> = DVector::from_vec(vec![0.0; 16]);

    let gle = Calculator::new(vec![(4, true), (4, true)]);

    c.bench_function("gle_interpolation", |b| {
        b.iter(|| {
            gle.diff(0, &v_in, &mut v_out);
        })
    });
}

fn bench_hc_interpolation_3d(c: &mut Criterion) {
    let v_in = vec![1.0; 64 * 8];
    let mut v_out = vec![0.0; 64 * 8];

    c.bench_function("hc_interpolation_3d", |b| {
        b.iter(|| {
            for i in 0..8 {
                let start = 64 * i;
                let end = start + 64;
                dx_q333_q333(&v_in[start..end], &mut v_out[start..end]);
            }
        })
    });
}

fn bench_hc_interpolation_3d_vec(c: &mut Criterion) {
    let v_in = vec![1.0; 64 * 8];
    let mut v_out = vec![0.0; 64 * 8];

    c.bench_function("hc_interpolation_3d_vec", |b| {
        b.iter(|| {
            dx_q333_q333_vec::<8>(&v_in, &mut v_out);
        })
    });
}

fn bench_hc_interpolation_1d(c: &mut Criterion) {
    let v_in = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 6.0, 30.0, 5.0, 2.0,
    ];

    let mut v_out = vec![0.0; 4 * 8];

    c.bench_function("hc_interpolation_1d", |b| {
        b.iter(|| {
            for i in 0..8 {
                let start = 4 * i;
                let end = start + 4;
                d_q3_q3::<1>(&v_in[start..end], &mut v_out[start..end]);
            }
        })
    });
}

fn bench_hc_interpolation_1d_vec(c: &mut Criterion) {
    let v_in = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 6.0, 30.0, 5.0, 2.0,
    ];

    let mut v_out = vec![0.0; 4 * 8];

    c.bench_function("hc_interpolation_1d_vec", |b| {
        b.iter(|| {
            d_q3_q3_vec::<8, 1>(&v_in, &mut v_out);
        })
    });
}

criterion_group!(
    benches,
    bench_hc_interpolation,
    bench_gle_interpolation,
    bench_hc_interpolation_1d,
    bench_hc_interpolation_1d_vec,
    bench_hc_interpolation_3d,
    bench_hc_interpolation_3d_vec
);

criterion_main!(benches);
