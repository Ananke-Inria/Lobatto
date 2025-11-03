use gg_lobatto::quadrature::*;

#[test]
fn test_gauss_quadrature() {
    let tolerance = 1e-14; // Tolerance to check wether a value is zero or not
    let n_tested_quad = 100; // Number of tested quadratures

    for n in 1..n_tested_quad {
        let (points, weights) = gauss(n);

        //Test if the integral of the monomials x^p are well integrated up to order p = 2n-1
        for p in 0..2 * n {
            let analytical = if p % 2 == 0 {
                2.0 / ((p + 1) as f64)
            } else {
                0.0
            };

            let numerical: f64 = points
                .iter()
                .zip(weights.iter())
                .map(|(&x, &w)| w * x.powi(p as i32))
                .sum();

            assert!(
                (analytical - numerical).abs() < tolerance,
                "Failed for n={}, p={}: analytical={}, numerical={}",
                n,
                p,
                analytical,
                numerical
            );
        }

        //Test if the quadrature weights are positive
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                w > 0.0,
                "Non-positive weight for n={}, index={}: weight={}",
                n,
                i,
                w
            );
        }

        //Test if the quadrature points are symmetric
        let m = points.len();
        for i in 0..(m / 2) {
            let left = points[i];
            let right = points[m - 1 - i];
            let left_weight = weights[i];
            let right_weight = weights[m - 1 - i];

            assert!(
                (left + right).abs() < tolerance,
                "Points not symmetric for n={}, i={}: left={}, right={}",
                n,
                i,
                left,
                right
            );

            assert!(
                (left_weight - right_weight).abs() < tolerance,
                "Weights not symmetric for n={}, i={}: left={}, right={}",
                n,
                i,
                left_weight,
                right_weight
            );
        }

        // If there's a middle point (odd number of points), it should be zero
        if m % 2 == 1 {
            let mid = points[m / 2];
            assert!(
                mid.abs() < tolerance,
                "Middle point not zero for n={}: mid={}",
                n,
                mid
            );
        }
    }
}

#[test]
fn test_gauss_lobatto_quadrature() {
    let tolerance = 1e-14; // Tolerance to check wether a value is zero or not
    let n_tested_quad = 100; // Number of tested quadratures

    for n in 2..n_tested_quad {
        let (points, weights) = gauss_lobatto(n);

        //Test if the integral of the monomials x^p are well integrated up to order p = 2n-1
        for p in 0..2 * (n - 1) {
            let analytical = if p % 2 == 0 {
                2.0 / ((p + 1) as f64)
            } else {
                0.0
            };

            let numerical: f64 = points
                .iter()
                .zip(weights.iter())
                .map(|(&x, &w)| w * x.powi(p as i32))
                .sum();

            assert!(
                (analytical - numerical).abs() < tolerance,
                "Failed for n={}, p={}: analytical={}, numerical={}",
                n,
                p,
                analytical,
                numerical
            );
        }

        //Test if the quadrature weights are positive
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                w > 0.0,
                "Non-positive weight for n={}, index={}: weight={}",
                n,
                i,
                w
            );
        }

        //Test if the quadrature points are symmetric
        let m = points.len();
        for i in 0..(m / 2) {
            let left = points[i];
            let right = points[m - 1 - i];
            let left_weight = weights[i];
            let right_weight = weights[m - 1 - i];

            assert!(
                (left + right).abs() < tolerance,
                "Points not symmetric for n={}, i={}: left={}, right={}",
                n,
                i,
                left,
                right
            );

            assert!(
                (left_weight - right_weight).abs() < tolerance,
                "Weights not symmetric for n={}, i={}: left={}, right={}",
                n,
                i,
                left_weight,
                right_weight
            );
        }

        // If there's a middle point (odd number of points), it should be zero
        if m % 2 == 1 {
            let mid = points[m / 2];
            assert!(
                mid.abs() < tolerance,
                "Middle point not zero for n={}: mid={}",
                n,
                mid
            );
        }

        // Check that the left point is -1 (
        let leftmost = points[0];

        assert!(
            (leftmost + 1.0).abs() < tolerance,
            "Left endpoint not -1 for n={}: leftmost={}",
            n,
            leftmost
        );
    }
}
