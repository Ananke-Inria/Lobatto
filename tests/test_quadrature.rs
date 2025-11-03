use gg_lobatto::hc_interpolation::*;
use gg_lobatto::hofft::*;
use gg_lobatto::interpolation::*;
use gg_lobatto::quadrature::*;

#[test]
fn it_works() {
    let (points, weights) = gauss(4);

    let sum: f64 = weights.iter().sum();

    println!("Points: {:?}", points);
    println!("Weights: {:?}", weights);
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

#[test]
fn test_gle() {
    let gle = Interpolator::new(vec![4, 4]);

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

#[test]
fn symbols() {
    let symbol = symbol_1d_lo(1.0, 0.1, 1.0, 0);
}
