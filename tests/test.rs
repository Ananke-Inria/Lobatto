extern crate gg_lobatto;

#[cfg(test)]
mod tests {
    use gg_lobatto::gauss_lobatto_quad;
    use gg_lobatto::gauss_quad;
    use gg_lobatto::theta_root_legendre;

    #[test]
    fn it_works() {
        let n = 5;

        let (points, weights) = gauss_quad(n);

        println!("Points: {:?}", points);
        println!("Weights: {:?}", weights);

        let (theta_root, p_at_roots, dp_at_roots) = theta_root_legendre(n);

        let x_root: Vec<f64> = theta_root.iter().map(|theta| theta.cos()).collect();

        println!("Root in theta: {:?}", theta_root);
        println!("P at roots: {:?}", p_at_roots);
        println!("dP at roots: {:?}", dp_at_roots);
        println!("Root in x: {:?}", x_root);

        let (points, weights) = gauss_quad(4);

        println!("Points: {:?}", points);
        println!("Weights: {:?}", weights);

        let sum: f64 = weights.iter().sum();

        println!("The sum is {:?}", sum);

        let (points, weights) = gauss_lobatto_quad(5);

        println!("Points: {:?}", points);
        println!("Weights: {:?}", weights);

        let sum: f64 = weights.iter().sum();

        println!("The sum is {:?}", sum);

        let (points, weights) = gauss_lobatto_quad(6);

        println!("Points: {:?}", points);
        println!("Weights: {:?}", weights);

        let sum: f64 = weights.iter().sum();

        println!("The sum is {:?}", sum);

        let (points, weights) = gauss_lobatto_quad(7);

        println!("Points: {:?}", points);
        println!("Weights: {:?}", weights);

        let sum: f64 = weights.iter().sum();

        println!("The sum is {:?}", sum);
    }
}
