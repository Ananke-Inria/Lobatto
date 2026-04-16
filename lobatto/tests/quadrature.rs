use lobatto::quadrature::*;
use lobatto::quadrature_hc::gauss_lobatto_hardcoded;

#[test]
fn test_gauss_quadrature() {
    let tolerance = 1e-14; // Tolerance to check wether a value is zero or not
    let n_tested_quad = 100; // Number of tested quadratures

    for n in 1..n_tested_quad {
        let (points, weights) = gauss_legendre(n);

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

            if analytical != 0.0 {
                assert!(
                    (analytical - numerical).abs() < tolerance,
                    "Failed for n={}, p={}: analytical={}, numerical={}",
                    n,
                    p,
                    analytical,
                    numerical
                );
            } else {
                assert!(
                    (numerical).abs() < tolerance,
                    "Failed for n={}, p={}: analytical={}, numerical={}",
                    n,
                    p,
                    analytical,
                    numerical
                );
            }
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

        //Test if the quadrature points are in increasing order
        for i in 1..points.len() {
            assert!(
                points[i] > points[i - 1],
                "Points not in increasing order for n={}, index={}: {} > {}",
                n,
                i,
                points[i - 1],
                points[i]
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

            if analytical != 0.0 {
                assert!(
                    (analytical - numerical).abs() < tolerance,
                    "Failed for n={}, p={}: analytical={}, numerical={}",
                    n,
                    p,
                    analytical,
                    numerical
                );
            } else {
                assert!(
                    (numerical).abs() < tolerance,
                    "Failed for n={}, p={}: analytical={}, numerical={}",
                    n,
                    p,
                    analytical,
                    numerical
                );
            }
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

        //Test if the quadrature points are in increasing order
        for i in 1..points.len() {
            assert!(
                points[i] > points[i - 1],
                "Points not in increasing order for n={}, index={}: {} > {}",
                n,
                i,
                points[i - 1],
                points[i]
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

#[test]
fn test_gauss_lobatto_hardcoded_accuracy() {
    let tolerance = 1e-15;

    for n in 2..=14usize {
        let (points, weights) = gauss_lobatto_hardcoded(n);

        // Polynomial exactness up to degree 2*(n-1)-1
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

            if analytical != 0.0 {
                assert!(
                    (analytical - numerical).abs() < tolerance,
                    "Failed for n={}, p={}: analytical={}, numerical={}",
                    n,
                    p,
                    analytical,
                    numerical
                );
            } else {
                assert!(
                    numerical.abs() < tolerance,
                    "Failed for n={}, p={}: analytical={}, numerical={}",
                    n,
                    p,
                    analytical,
                    numerical
                );
            }
        }

        // Positive weights
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                w > 0.0,
                "Non-positive weight for n={}, index={}: weight={}",
                n,
                i,
                w
            );
        }

        // Strictly increasing order
        for i in 1..points.len() {
            assert!(
                points[i] > points[i - 1],
                "Points not in increasing order for n={}, index={}: {} > {}",
                n,
                i,
                points[i - 1],
                points[i]
            );
        }

        // Symmetry of points and weights
        let m = points.len();
        for i in 0..(m / 2) {
            assert!(
                (points[i] + points[m - 1 - i]).abs() < tolerance,
                "Points not symmetric for n={}, i={}: left={}, right={}",
                n,
                i,
                points[i],
                points[m - 1 - i]
            );
            assert!(
                (weights[i] - weights[m - 1 - i]).abs() < tolerance,
                "Weights not symmetric for n={}, i={}: left={}, right={}",
                n,
                i,
                weights[i],
                weights[m - 1 - i]
            );
        }

        // Middle point = 0 for odd number of points
        if m % 2 == 1 {
            assert!(
                points[m / 2].abs() < tolerance,
                "Middle point not zero for n={}: mid={}",
                n,
                points[m / 2]
            );
        }

        // Leftmost point = -1
        assert!(
            (points[0] + 1.0).abs() < tolerance,
            "Left endpoint not -1 for n={}: leftmost={}",
            n,
            points[0]
        );
    }
}

#[test]
fn test_gauss_lobatto_hardcoded_matches_computed() {
    // Hardcoded values must be bit-for-bit identical to what gauss_lobatto() computes.
    let tolerance = 0.0;

    for n in 2..=14usize {
        let (pts_hc, wts_hc) = gauss_lobatto_hardcoded(n);
        let (pts_gl, wts_gl) = gauss_lobatto(n);

        assert_eq!(
            pts_hc.len(),
            pts_gl.len(),
            "Point count mismatch for n={}",
            n
        );

        for i in 0..pts_hc.len() {
            assert!(
                (pts_hc[i] - pts_gl[i]).abs() <= tolerance,
                "Point mismatch for n={}, i={}: hardcoded={}, computed={}",
                n,
                i,
                pts_hc[i],
                pts_gl[i]
            );
            assert!(
                (wts_hc[i] - wts_gl[i]).abs() <= tolerance,
                "Weight mismatch for n={}, i={}: hardcoded={}, computed={}",
                n,
                i,
                wts_hc[i],
                wts_gl[i]
            );
        }
    }
}
