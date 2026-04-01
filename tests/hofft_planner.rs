use nalgebra::Complex;
use quadratix::hofft::*;
use std::f64::consts::PI;

// ── FFT round-trip ────────────────────────────────────────────────────────────

#[test]
fn test_nd_fft_roundtrip() {
    const TOL: f64 = 1e-12;

    fn roundtrip_error<const N: usize>(planner: &HofftPlannerND<N>, f_orig: &[f64]) -> f64 {
        let mut g: Vec<Complex<f64>> = f_orig.iter().map(|&v| Complex::new(v, 0.0)).collect();
        planner.forward(&mut g);
        planner.inverse(&mut g);
        g.iter()
            .zip(f_orig.iter())
            .map(|(c, &v)| (c.re - v).abs())
            .fold(0.0_f64, f64::max)
    }

    // 1D
    let planner = HofftPlannerND::<1>::new([8], [3], [1.0], [0.0]);
    let x = planner.get_x();
    let f_orig: Vec<f64> = x.iter().map(|p| (p[0] * PI).sin()).collect();
    let err = roundtrip_error(&planner, &f_orig);
    assert!(err < TOL, "1D round-trip error = {err:.2e}");

    // 2D
    let planner = HofftPlannerND::<2>::new([4; 2], [2; 2], [1.0; 2], [0.0; 2]);
    let x = planner.get_x();
    let f_orig: Vec<f64> = x
        .iter()
        .map(|p| (p[0] * PI).sin() * (p[1] * PI).cos())
        .collect();
    let err = roundtrip_error(&planner, &f_orig);
    assert!(err < TOL, "2D round-trip error = {err:.2e}");

    // 3D
    let planner = HofftPlannerND::<3>::new([4; 3], [2; 3], [1.0; 3], [0.0; 3]);
    let x = planner.get_x();
    let f_orig: Vec<f64> = x
        .iter()
        .map(|p| (p[0] * PI).sin() * (p[1] * PI).cos() * (p[2] * 2.0 * PI).sin())
        .collect();
    let err = roundtrip_error(&planner, &f_orig);
    assert!(err < TOL, "3D round-trip error = {err:.2e}");
}
