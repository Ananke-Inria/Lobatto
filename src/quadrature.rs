//! Gauss and Gauss-Lobatto quadrature rules on [-1, 1] and [0, 1].
//!
//! Algorithms are based on:
//!
//! - \[1\] Swarztrauber, P. N. (2002). *On computing the points and weights for
//!   Gauss–Legendre quadrature*. SIAM Journal of Scientific Computing, 24(3), 945–954.
//! - \[2\] Yakimiw, E. (1996). *Accurate computation of weights in classical
//!   Gauss–Christoffel quadrature rules*. Journal of Computational Physics, 129(2), 406–430.

/// Compute `n_quad_points`-point Gauss-Legendre quadrature on [-1, 1].
///
/// Returns `(points, weights)` with `points` sorted in increasing order.
/// Exact closed-form expressions are used for 1, 2, and 3 points; the general
/// case uses Newton's method on the Fourier representation of Legendre polynomials.
///
/// # Panics
/// Panics if `n_quad_points == 0`.
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

/// Compute `n_quad_points`-point Gauss-Lobatto-Legendre (GLL) quadrature on [-1, 1].
///
/// GLL quadrature includes the endpoints ±1 as nodes, which makes it well-suited
/// for finite-element assembly (shared boundary DOFs). Returns `(points, weights)`
/// with `points` sorted in increasing order.
/// Exact closed-form expressions are used for 2, 3, and 4 points; the general
/// case uses Newton's method on the derivative of the Legendre polynomial.
///
/// The nodes $x_1 = -1 < x_2 < \cdots < x_{n-1} < x_n = 1$ are the endpoints
/// together with the roots of $P_{n-1}'$, the derivative of the Legendre polynomial
/// of degree $n-1$.  The corresponding weights are:
///
/// $$w_i = \frac{2}{n(n-1) [P_{n-1}(x_i)]^2}, \quad i = 1, \dots, n$$
///
/// where the endpoint weights simplify to $w_1 = w_n = \dfrac{2}{n(n-1)}$
/// since $P_{n-1}(\pm 1) = (\pm 1)^{n-1}$.
///
/// # Panics
/// Panics if `n_quad_points < 2` (GLL requires at least the two endpoints).
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

/// GLL quadrature on the unit interval [0, 1].
///
/// Equivalent to [`gauss_lobatto`] with the affine change of variable `x = (t+1)/2`.
/// Returns `(points, weights)` with `points ∈ [0, 1]` and `sum(weights) == 1`.
pub fn gauss_lobatto_on_unit_interval(n_quad_points: usize) -> (Vec<f64>, Vec<f64>) {
    let mut points: Vec<f64>;
    let mut weights: Vec<f64>;

    (points, weights) = gauss_lobatto(n_quad_points);

    points.iter_mut().for_each(|x| *x = 0.5 * (*x + 1.0));
    weights.iter_mut().for_each(|w| *w *= 0.5);

    return (points, weights);
}

/// Gauss-Legendre quadrature on the unit interval [0, 1].
///
/// Equivalent to [`gauss`] with the affine change of variable `x = (t+1)/2`.
/// Returns `(points, weights)` with `points ∈ (0, 1)` and `sum(weights) == 1`.
pub fn gauss_on_unit_interval(n_quad_points: usize) -> (Vec<f64>, Vec<f64>) {
    let mut points: Vec<f64>;
    let mut weights: Vec<f64>;

    (points, weights) = gauss(n_quad_points);

    points.iter_mut().for_each(|x| *x = 0.5 * (*x + 1.0));
    weights.iter_mut().for_each(|w| *w *= 0.5);

    return (points, weights);
}

// Computes the Fourier coefficients of the Legendre polynomial P_n and its
// first and second derivatives, using the Fourier–cosine representation
//   P_n(cos θ) = Σ_k a_{n,k} cos(k θ)
// defined by equation (2) of [1].  The recurrence of equation (6)/(8) of [1]
// is implemented.  The normalisation convention is that the leading coefficient
// equals one.
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

// Evaluate P_n(cos θ) from its Fourier coefficients `ank_vec`.
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

// Evaluate d/dθ P_n(cos θ) from its Fourier coefficients.
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

// Evaluate d²/dθ² P_n(cos θ) from its Fourier coefficients.
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

// Return the roots (in θ-space) of P_n(cos θ) or P_n'(cos θ), along with the
// values of P_n and P_n' at those roots.  Only roots in [0, π/2] are returned
// (the full set is recovered by symmetry).  Newton's method is applied in θ-space.
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

// Return the normalisation constant 1/P_n(cos π) = 1/P_n(-1) used by the
// weight formulae of [1] and [2].
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

// Build the full GLL quadrature rule for n_quad_points ≥ 5 using formula (A.1.20) of [2].
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

// Build the full Gauss-Legendre quadrature rule for n_quad_points ≥ 4 using formula (17) of [1].
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
