// Algorithms based on the article
//
// [1]  ON COMPUTING THE POINTS AND WEIGHTS FOR GAUSS–LEGENDRE QUADRATURE
//                              PAUL N. SWARZTRAUBER
//      SIAM Journal of Scientific Computing, 2002, Vol. 24, No. 3, pp. 945–954
// and
//
//  [2] Accurate Computation of Weights in Classical Gauss–Christoffel Quadrature Rules
//                                  E. Yakimiw
// Journal of Computational Physics, 1996, Vol. 129, No. 2, pp. 406-430
//

pub fn gauss(n_quad_points: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(
        n_quad_points >= 1,
        "Gauss quadratures have at least one point"
    );

    // Exact expressions for order 1, 2, 3
    match n_quad_points {
        1 => {
            return (vec![0.0], vec![2.0]);
        }
        2 => {
            return (
                vec![-(1.0 / 3.0 as f64).sqrt(), (1.0 / 3.0 as f64).sqrt()],
                vec![1.0, 1.0],
            );
        }
        3 => {
            return (
                vec![-(3.0 / 5.0 as f64).sqrt(), 0.0, (3.0 / 5.0 as f64).sqrt()],
                vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
            );
        }
        _ => {
            return compute_gauss_quadrature(n_quad_points);
        }
    }
}

pub fn gauss_lobatto(n_quad_points: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(
        n_quad_points >= 2,
        "Gauss-Lobatto quadratures have at least two points"
    );

    // Exact expressions for order 1, 2, 3
    match n_quad_points {
        2 => {
            return (vec![-1.0, 1.0], vec![1.0, 1.0]);
        }
        3 => {
            return (vec![-1.0, 0.0, 1.0], vec![1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]);
        }
        4 => {
            return (
                vec![
                    -1.0,
                    -(5.0 as f64).sqrt() / 5.0,
                    (5.0 as f64).sqrt() / 5.0,
                    1.0,
                ],
                vec![1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0],
            );
        }
        _ => {
            return compute_gauss_lobatto_quadrature(n_quad_points);
        }
    }
}

pub fn gauss_lobatto_on_unit_interval(n_quad_points: usize) -> (Vec<f64>, Vec<f64>) {
    let mut points: Vec<f64>;
    let mut weights: Vec<f64>;

    (points, weights) = gauss_lobatto(n_quad_points);

    points.iter_mut().for_each(|x| *x = 0.5 * (*x + 1.0));
    weights.iter_mut().for_each(|w| *w *= 0.5);

    return (points, weights);
}

pub fn gauss_on_unit_interval(n_quad_points: usize) -> (Vec<f64>, Vec<f64>) {
    let mut points: Vec<f64>;
    let mut weights: Vec<f64>;

    (points, weights) = gauss(n_quad_points);

    points.iter_mut().for_each(|x| *x = 0.5 * (*x + 1.0));
    weights.iter_mut().for_each(|w| *w *= 0.5);

    return (points, weights);
}

// Computes the fourier coefficients of the legendre  polynomial Pn
// and of its first and second derivative as defined by equation (2) of [1].
// The recurrence of equation (6) of [1] is implemented.
// The chosen normalizaton is to have the leading coefficient equals to one.
fn legendre_fourier(n: usize) -> Vec<f64> {
    let mut ank_vec = vec![0 as f64; n + 1];

    ank_vec[n] = 1.0;

    let mut t1 = 1;
    let mut t2 = n;
    let mut t3 = 1;
    let mut t4 = 2 * n - 1;

    //Implement formula (8) of [1]
    for j in (0..n).rev() {
        if n % 2 == j % 2 {
            ank_vec[j] = ((t1 * t2) as f64) * ank_vec[j + 2] / ((t3 * t4) as f64);

            t1 += 2;
            t2 -= 1;
            t3 += 1;
            t4 -= 2;
        }
    }

    //Correcting factor for the lowest order term (missing in [1] equation (8))
    ank_vec[0] *= 0.5 as f64;

    return ank_vec;
}

fn fourier_value(theta: f64, ank_vec: &Vec<f64>) -> f64 {
    let m = ank_vec.len();
    let mut v = 0 as f64;

    for k in 0..m {
        let ank = ank_vec[k];

        if ank != 0.0 {
            v += ank * (k as f64 * theta).cos();
        }
    }

    return v;
}

fn fourier_d_value(theta: f64, ank_vec: &Vec<f64>) -> f64 {
    let m = ank_vec.len();
    let mut dv = 0 as f64;

    for k in 0..m {
        let ank = ank_vec[k];
        if ank != 0.0 {
            dv -= (k as f64) * ank * (k as f64 * theta).sin();
        }
    }

    return dv;
}

fn fourier_dd_value(theta: f64, ank_vec: &Vec<f64>) -> f64 {
    let m = ank_vec.len();
    let mut ddv = 0 as f64;

    for k in 0..m {
        let ank = ank_vec[k];

        if ank != 0.0 {
            ddv -= ((k * k) as f64) * ank * (k as f64 * theta).cos();
        }
    }

    return ddv;
}

// Return the roots of the legendre polynomial or its derivative of order n using the decompositon (2) of [1]. More precisely, the roots of  P_n (cos \theta) or P_n' (cos \theta) are returned.
// The roots are symmetric with respect to the origin so only the roots in [0,pi/2] are returned
fn theta_root_legendre_or_d_legendre(n: usize, d_legendre: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    const EPS: f64 = std::f64::EPSILON;
    const PI: f64 = core::f64::consts::PI;

    let ank_vec = legendre_fourier(n);

    let n_roots;

    if d_legendre == false {
        n_roots = n / 2;
    } else {
        n_roots = (n - 1) / 2;
    }

    let mut roots = vec![0 as f64; n_roots];
    let mut p_at_roots = vec![0 as f64; n_roots];
    let mut dp_at_roots = vec![0 as f64; n_roots];

    let dtheta = (0.5 * PI) as f64 / (n_roots + 1) as f64;

    roots[0] = dtheta;

    for m in 0..n_roots {
        if m == 0 {
            roots[0] = dtheta;
        } else {
            roots[m] = roots[m - 1] + dtheta;
        }

        let mut count = 0 as usize;
        let mut residue = 1 as f64;

        loop {
            count += 1;

            let old_residue = residue;

            if d_legendre {
                let d_value = fourier_d_value(roots[m], &ank_vec);
                let dd_value = fourier_dd_value(roots[m], &ank_vec);

                residue = -d_value / dd_value;
            } else {
                let value = fourier_value(roots[m], &ank_vec);
                let d_value = fourier_d_value(roots[m], &ank_vec);

                residue = -value / d_value;
            }

            roots[m] += residue;

            if residue.abs() < EPS * roots[m].abs()
                || (count == 5 && (old_residue.abs() < residue.abs()))
                || count == 100
            {
                dp_at_roots[m] = fourier_d_value(roots[m], &ank_vec);
                p_at_roots[m] = fourier_value(roots[m], &ank_vec);

                break;
            }
        }
    }

    //Add the root in theta = pi/2
    if d_legendre == false && n % 2 == 1 {
        roots.push(0.5 * PI);
        p_at_roots.push(fourier_value(0.5 * PI, &ank_vec));
        dp_at_roots.push(fourier_d_value(0.5 * PI, &ank_vec));
    }

    if d_legendre == true && n % 2 == 0 {
        roots.push(0.5 * PI);
        p_at_roots.push(fourier_value(0.5 * PI, &ank_vec));
        dp_at_roots.push(fourier_d_value(0.5 * PI, &ank_vec));
    }

    return (roots, p_at_roots, dp_at_roots);
}

fn get_normalized_coefficient(n: usize) -> f64 {
    let ank_vec = legendre_fourier(n);
    const PI: f64 = core::f64::consts::PI;

    return 1.0 / fourier_value(PI, &ank_vec);
}

fn theta_root_legendre(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    return theta_root_legendre_or_d_legendre(n, false);
}

fn theta_root_d_legendre(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    return theta_root_legendre_or_d_legendre(n, true);
}

fn compute_gauss_lobatto_quadrature(n_quad_points: usize) -> (Vec<f64>, Vec<f64>) {
    let (root_theta, p_at_root_theta, dp_at_root_theta) = theta_root_d_legendre(n_quad_points - 1);

    let mut weights = vec![0.0; root_theta.len()];

    //Formulae (A.1.20) of [2] with the first term in the series
    let normalized_coeff = get_normalized_coefficient(n_quad_points - 1);
    let nbar = n_quad_points as f64 * (n_quad_points - 1) as f64;

    for k in 0..root_theta.len() {
        let f2 = (dp_at_root_theta[k] / p_at_root_theta[k]).powi(2);
        let c2 = 1.0 - root_theta[k].cos().powi(2);

        weights[k] =
            2.0 * nbar / (normalized_coeff * p_at_root_theta[k] * (nbar + 0.5 * f2 * c2)).powi(2);
    }

    let mut points: Vec<f64> = root_theta.iter().map(|theta| theta.cos()).collect();

    //Insert the quadrature point x=1
    points.insert(0, 1.0);
    weights.insert(0, 2.0 / nbar);

    let m = points.len();

    if n_quad_points % 2 == 1 {
        points[m - 1] = 0.0 as f64;

        for n in 0..m - 1 {
            points.push(-points[m - n - 2]);
            weights.push(weights[m - n - 2]);
        }
    } else {
        for n in 0..m {
            points.push(-points[m - n - 1]);
            weights.push(weights[m - n - 1]);
        }
    }

    points.reverse();

    return (points, weights);
}

fn compute_gauss_quadrature(n_quad_points: usize) -> (Vec<f64>, Vec<f64>) {
    let (root_theta, p_at_root_theta, dp_at_root_theta) = theta_root_legendre(n_quad_points);

    let mut weights = vec![0 as f64; root_theta.len()];

    //Formula (17) of [1]
    let normalized_coeff = get_normalized_coefficient(n_quad_points);
    for k in 0..root_theta.len() {
        let tmp = normalized_coeff
            * (dp_at_root_theta[k]
                + p_at_root_theta[k] * root_theta[k].cos() / root_theta[k].sin());
        weights[k] = 2.0 / (tmp * tmp);
    }

    let mut points: Vec<f64> = root_theta.iter().map(|theta| theta.cos()).collect();

    let m = points.len();

    if n_quad_points % 2 == 1 {
        points[m - 1] = 0.0 as f64;

        for n in 0..m - 1 {
            points.push(-points[m - n - 2]);
            weights.push(weights[m - n - 2]);
        }
    } else {
        for n in 0..m {
            points.push(-points[m - n - 1]);
            weights.push(weights[m - n - 1]);
        }
    }

    points.reverse();

    return (points, weights);
}
