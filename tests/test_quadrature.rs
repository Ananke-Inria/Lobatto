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
