#![feature(test)]
#![feature(portable_simd)]

#[cfg(test)]
mod tests {
    extern crate test;
    use std::simd::*;

    use gg_lobatto::hc_interpolation::*;
    use gg_lobatto::interpolation::*;
    use gg_lobatto::quadrature::*;

    #[test]
    fn it_works() {
        let (points, weights) = gauss(4);

        println!("Points: {:?}", points);
        println!("Weights: {:?}", weights);

        let sum: f64 = weights.iter().sum();

        println!("The sum is {:?}", sum);

        let (points, weights) = gauss_lobatto(5);

        println!("Points: {:?}", points);
        println!("Weights: {:?}", weights);

        let sum: f64 = weights.iter().sum();

        println!("The sum is {:?}", sum);

        let (points, weights) = gauss_lobatto(6);

        println!("Points: {:?}", points);
        println!("Weights: {:?}", weights);

        let sum: f64 = weights.iter().sum();

        println!("The sum is {:?}", sum);

        let (points, weights) = gauss_lobatto(10);

        println!("Points: {:?}", points);
        println!("Weights: {:?}", weights);

        let sum: f64 = weights.iter().sum();

        println!("The sum is {:?}", sum);
    }

    #[bench]
    fn test_efficiency(b: &mut test::Bencher) {
        b.iter(|| {
            gauss_lobatto(5);
            gauss_lobatto(5);
            gauss_lobatto(5);
            gauss_lobatto(5);
        });
    }

    #[test]
    fn test_gle() {
        let gle = GaussLobattoElement::new(vec![4, 4]);

        println!("gle: {:?}", gle);

        let v_in: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.3, 2.5, 7.8, 9.5, 1.2, 3.5, 4.6, 4.9, 12.0, 15.0,
        ];

        let mut v_out: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        gle.derivative_interpolation(0, &v_in, &mut v_out);

        println!("{:?}", v_out);
    }

    #[test]
    fn test_hc_interpolation() {
        let v_in: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.3, 2.5, 7.8, 9.5, 1.2, 3.5, 4.6, 4.9, 12.0, 15.0,
        ];

        let mut v_out: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        dx_q33_q33(&v_in, &mut v_out);

        println!("{:?}", v_out);
    }

    #[bench]
    fn test_efficiency_hc_interpolation_simd(b: &mut test::Bencher) {
        //let in_simd = f64x8::from_array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        //let out_simd = f64x8::from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v_in = vec![1.0; 16 * 8];
        let mut v_out = vec![0.0; 16 * 8];

        b.iter(|| {
            dx_q33_q33_simd(&v_in, &mut v_out);
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

        let gle = GaussLobattoElement::new(vec![4, 4]);

        b.iter(|| {
            gle.derivative_interpolation(0, &v_in, &mut v_out);
        });
    }

    #[bench]
    fn test_efficiency_gle_3d(b: &mut test::Bencher) {
        let v_in: Vec<f64> = vec![1.0; 64];
        let mut v_out: Vec<f64> = vec![0.0; 64];

        let gle = GaussLobattoElement::new(vec![4, 4, 4]);

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
    fn test_efficiency_hc_interpolation_3d_simd(b: &mut test::Bencher) {
        let v_in = vec![1.0; 64 * 8];
        let mut v_out = vec![0.0; 64 * 8];

        b.iter(|| {
            dx_q333_q333_simd(&v_in, &mut v_out);
        });
    }
}
