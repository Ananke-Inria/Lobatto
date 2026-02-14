use gg_lobatto::calculator::*;
use gg_lobatto::hc_calculations::*;
use nalgebra::DVector;

#[test]
fn test_gle() {
    let gle = Calculator::new(vec![(4, true), (4, true)]);

    //println!("gle: {:?}", gle);

    let v_in: DVector<f64> = DVector::from_vec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.3, 2.5, 7.8, 9.5, 1.2, 3.5, 4.6, 4.9, 12.0, 15.0,
    ]);

    let mut v_out: DVector<f64> = DVector::from_vec(vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]);

    gle.diff(0, &v_in, &mut v_out);

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
