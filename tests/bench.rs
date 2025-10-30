#![feature(test)]

mod bench {
    extern crate test;

    use gg_lobatto::hc_interpolation::*;
    use gg_lobatto::interpolation::*;
    use gg_lobatto::quadrature::*;

    #[bench]
    fn test_efficiency(b: &mut test::Bencher) {
        b.iter(|| {
            gauss_lobatto(5);
        });
    }

    #[bench]
    fn test_efficiency_hc_interpolation(b: &mut test::Bencher) {
        let v_in = vec![1.0; 16];
        let mut v_out = vec![0.0; 16];

        b.iter(|| {
            dx_q33_q33(&v_in, &mut v_out);
        });
    }

    #[bench]
    fn test_efficiency_gle(b: &mut test::Bencher) {
        let v_in: Vec<f64> = vec![1.0; 16];
        let mut v_out: Vec<f64> = vec![0.0; 16];

        let gle = Interpolator::new(vec![4, 4]);

        b.iter(|| {
            gle.derivative_interpolation(0, &v_in, &mut v_out);
        });
    }

    #[bench]
    fn test_efficiency_gle_3d(b: &mut test::Bencher) {
        let v_in: Vec<f64> = vec![1.0; 64];
        let mut v_out: Vec<f64> = vec![0.0; 64];

        let gle = Interpolator::new(vec![4, 4, 4]);

        b.iter(|| {
            gle.derivative_interpolation(0, &v_in, &mut v_out);
        });
    }

    #[bench]
    fn test_efficiency_hc_interpolation_3d(b: &mut test::Bencher) {
        let v_in: Vec<f64> = vec![1.0; 64];
        let mut v_out: Vec<f64> = vec![0.0; 64];

        b.iter(|| {
            dx_q333_q333(&v_in, &mut v_out);
        });
    }

    #[bench]
    fn test_efficiency_hc_interpolation_3d_vec(b: &mut test::Bencher) {
        let v_in = vec![1.0; 64 * 8];
        let mut v_out = vec![0.0; 64 * 8];

        b.iter(|| {
            dx_q333_q333_vec::<8>(&v_in, &mut v_out);
        });
    }

    #[bench]
    fn test_efficiency_hc_interpolation_1d(b: &mut test::Bencher) {
        let v_in = vec![1.0; 4];
        let mut v_out = vec![0.0; 4];

        b.iter(|| {
            d_q3_q3::<1>(&v_in, &mut v_out);
        });
    }

    #[bench]
    fn test_efficiency_hc_interpolation_1d_vec(b: &mut test::Bencher) {
        let v_in = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 6.0, 30.0, 5.0, 2.0,
        ];
        let mut v_out = vec![0.0; 4 * 8];

        b.iter(|| {
            d_q3_q3_vec::<8, 1>(&v_in, &mut v_out);
        });
    }
}
