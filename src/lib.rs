//! This is my first rust crate

// Algorithm based on the article
//
// [1]  ON COMPUTING THE POINTS AND WEIGHTS FOR GAUSS–LEGENDRE QUADRATURE
//                              PAUL N. SWARZTRAUBER
//      SIAM Journal of Scientific Computing, 2002, Vol. 24, No. 3, pp. 945–954

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

    //Correcting factor for the lowest order term (missing in [1])
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

// Return the roots of the legendre polynomial of order n using the decompositon (2) of [1]
// The roots are symmetric with respect to the origin so only the roots in [0,pi/2] are returned
fn theta_root_legendre(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    const EPS: f64 = std::f64::EPSILON;
    const PI: f64 = core::f64::consts::PI;

    let ank_vec = legendre_fourier(n);

    let n_roots = n / 2;

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

        let mut count = 0u32;

        let mut residue = 1 as f64;

        loop {
            count += 1;

            let value = fourier_value(roots[m], &ank_vec);
            let d_value = fourier_d_value(roots[m], &ank_vec);
            let old_residue = residue;

            residue = -value / d_value;
            roots[m] += residue;
            dp_at_roots[m] = d_value;
            p_at_roots[m] = value;

            if residue.abs() < EPS * roots[m].abs() {
                break;
            }

            if count == 5 && (old_residue.abs() < residue.abs()) {
                break;
            }

            if count == 100 {
                break;
            }
        }
    }

    //Add the root in theta = pi/2
    if n % 2 == 1 {
        roots.push(0.5 * PI);
        p_at_roots.push(fourier_value(0.5 * PI, &ank_vec));
        dp_at_roots.push(fourier_d_value(0.5 * PI, &ank_vec));
    }

    return (roots, p_at_roots, dp_at_roots);
}

// Return the roots of the derivative of legendre polynomial of order n using the decompositon (2) of [1]
// The roots are symmetric with respect to the origin so only the roots in (0,pi/2] are returned
// Note that for even n, the derivative of the legendre polynomial of order n as a root at theta = pi/2.
fn theta_root_d_legendre(n: usize) -> Vec<f64> {
    let ank_vec = legendre_fourier(n);

    const EPS: f64 = std::f64::EPSILON;
    const PI: f64 = core::f64::consts::PI;

    let n_roots = (n - 1) / 2;

    let mut roots = vec![0 as f64; n_roots];

    let dtheta = (0.5 * PI) as f64 / (n_roots + 1) as f64;

    roots[0] = dtheta;

    for m in 0..n_roots {
        if m == 0 {
            roots[0] = dtheta;
        } else {
            roots[m] = roots[m - 1] + dtheta;
        }

        let mut count = 0u32;

        let mut residue = 1 as f64;

        loop {
            count += 1;

            let d_value = fourier_d_value(roots[m], &ank_vec);
            let dd_value = fourier_dd_value(roots[m], &ank_vec);
            let old_residue = residue;

            residue = -d_value / dd_value;
            roots[m] += residue;

            if residue.abs() < EPS * roots[m].abs() {
                break;
            }

            if count == 5 && (old_residue.abs() < residue.abs()) {
                break;
            }

            if count == 100 {
                break;
            }
        }
    }

    return roots;
}

fn gauss_lobatto_quad(n_quad: usize) -> (Vec<f64>, Vec<f64>) {
    let mut points = vec![0 as f64; n_quad + 1];
    let mut weights = vec![0 as f64; n_quad + 1];

    // Exact expressions for order 1, 2, 3
    match n_quad {
        1 => {
            points[0] = 0.0;
            points[1] = 1.0;
            weights[0] = 0.5;
            weights[1] = 0.5;
            return (points, weights);
        }
        2 => {
            points[0] = 0.0;
            points[1] = 0.5;
            points[2] = 1.0;
            weights[0] = 1.0 / 6.0;
            weights[1] = 2.0 / 3.0;
            weights[2] = weights[0];
            return (points, weights);
        }
        3 => {
            points[0] = 0.0;
            points[1] = (5.0 - (5.0f64).sqrt()) / 10.0;
            points[2] = 1.0 - points[1];
            points[3] = 1.0;
            weights[0] = 1.0 / 12.0;
            weights[1] = 5.0 / 12.0;
            weights[2] = weights[1];
            weights[3] = weights[0];
            return (points, weights);
        }
        _ => {
            return (points, weights);
        }
    }
}

fn gauss_quad(n_quad: usize) -> (Vec<f64>, Vec<f64>) {
    let mut points = vec![0 as f64; n_quad];
    let mut weights = vec![0 as f64; n_quad];

    // Exact expressions for order 1, 2, 3
    match n_quad {
        1 => {
            points[0] = 0.5;
            weights[0] = 1.0;
            return (points, weights);
        }
        2 => {
            points[0] = (3.0 - (3.0f64).sqrt()) / 6.0;
            points[1] = 1.0 - points[0];
            weights[0] = 0.5;
            weights[1] = 0.5;
            return (points, weights);
        }
        3 => {
            points[0] = 0.5 - (15.0f64).sqrt() / 10.0;
            points[1] = 0.5;
            points[2] = 1.0 - points[0];
            weights[0] = 5.0 / 18.0;
            weights[1] = 4.0 / 9.0;
            weights[2] = weights[0];
            return (points, weights);
        }
        _ => {
            let (root_theta, p_at_root_theta, dp_at_root_theta) = theta_root_legendre(n_quad);

            let m = root_theta.len();
            let mut weight_theta = vec![0 as f64; m];

            //Formulae (17) of [1]
            for k in 0..m {
                let tmp = dp_at_root_theta[k]
                    + p_at_root_theta[k] * root_theta[k].cos() / root_theta[k].sin();
                weight_theta[k] = 1.0 / (tmp * tmp);
            }

            return (points, weights);
        }
    }
}

// fn compute_gauss_formulas(nb_points_gauss: usize, points: &mut Vec<f64>, weights: &mut Vec<f64>, gauss_lobatto: bool) {

//     let mut theta = vec![0.0; nb_points_gauss];
//     let mut coef = vec![0.0; nb_points_quadrature];
//     let mut deriv_coef = vec![0.0; nb_points_quadrature];
//     let mut second_deriv_coef = vec![0.0; nb_points_quadrature];

//     let mut zero;
//     let mut zprev;
//     let mut zlast;
//     let mut zhold;
//     let mut pb;
//     let mut dpb;
//     let mut ddpb;
//     let mut dcor;
//     let mut sgnd;
//     let mut nix;

//     // For order greater than 3, numerical computation
//     let parity_nb_points = nb_points_quadrature % 2;
//     let half_nb_points = nb_points_gauss / 2;
//     let nhalf = (nb_points_quadrature + 1) / 2;
//     let mut cmax = pio2 / 5.0;

//     let mut nb_iter;
//     let mut test_convergence_newton;
//     let mut residu;
//     let mut residu_prec;

//     let mut dtheta;
//     if gauss_lobatto && parity_nb_points == 1 {
//         dtheta = pio2 / (nhalf - 1) as f64;
//     } else {
//         dtheta = pio2 / nhalf as f64;
//     }

//     let dthalf = dtheta / 2.0;

//     // Compute Fourier coefficients
//     let mut cz = 0.0;
//     compute_fourier_coef(nb_points_quadrature, &mut cz, &mut coef, &mut deriv_coef, &mut second_deriv_coef);

//     // Iterative procedure to compute points and weights
//     zero = 0.0;
//     zprev = 0.0;
//     zlast = 0.0;
//     zhold = 0.0;
//     pb = 0.0;
//     dpb = 0.0;
//     ddpb = 0.0;
//     dcor = 0.0;
//     sgnd = 0.0;
//     nix = 0;

//     let mut each_point_not_computed = true;

//     while each_point_not_computed {
//         nb_iter = 0;
//         test_convergence_newton = true;
//         residu = 1.0;
//         residu_prec = 1.0;

//         while test_convergence_newton && nb_iter < 100 {
//             nb_iter += 1;
//             zlast = zero;
//             let (pbi, dpbi, ddpbi) = compute_legendre_pol_and_derivative(nb_points_quadrature, zero, cz, &coef, &deriv_coef, &second_deriv_coef);

//             pb = pbi;
//             dpb = dpbi;
//             ddpb = ddpbi;

//             if gauss_lobatto {
//                 dcor = dpb / ddpb;
//             } else {
//                 dcor = pb / dpb;
//             }

//             sgnd = 1.0;
//             if dcor != 0.0 {
//                 sgnd = dcor.signum();
//             }

//             let tmp = f64::min(f64::abs(dcor), cmax);
//             dcor = sgnd * tmp;
//             zero -= dcor;
//             residu_prec = residu;
//             residu = f64::abs(zero - zlast);

//             if f64::abs(zero - zlast) <= EPS * f64::abs(zero) || (nb_iter > 5 && residu_prec < residu) {
//                 test_convergence_newton = false;
//             }
//         }

//         theta[nix] = zero;
//         zhold = zero;

//         if gauss_lobatto {
//             weights[nix] = 1.0 / (pb - dcor * dpb + 0.5 * dcor * dcor * ddpb).powi(2);
//         } else {
//             weights[nix] = (2*nb_points_quadrature + 1) as f64 / (dpb + pb * (zero.cos() / zero.sin())).powi(2);
//         }
//         nix += 1;

//         if nix == nb_points_gauss {
//             each_point_not_computed = false;
//         } else if nix <= nhalf - 1 {
//             zero += zero - zprev;
//         }

//         zprev = zhold;
//     }

//     // Extend points and weights via symmetries
//     if !gauss_lobatto && parity_nb_points != 0 {
//         theta[nhalf - 1] = pio2;
//         let (pbi, dpbi, ddpbi) = compute_legendre_pol_and_derivative(nb_points_quadrature, pio2, cz, &coef, &deriv_coef, &second_deriv_coef);
//         pb = pbi;
//         dpb = dpbi;
//         ddpb = ddpbi;
//         weights[nhalf - 1] = (nb_points_quadrature + nb_points_quadrature + 1) as f64 / (dpb * dpb);
//     }

//     if gauss_lobatto && parity_nb_points == 0 {
//         theta[nhalf - 1] = pio2;
//         let (pbi, dpbi, ddpbi) = compute_legendre_pol_and_derivative(nb_points_quadrature, pio2, cz, &coef, &deriv_coef, &second_deriv_coef);
//         pb = pbi;
//         dpb = dpbi;
//         ddpb = ddpbi;
//         weights[nhalf - 1] = 1.0 / (pb * pb);
//     }

//     if gauss_lobatto {
//         for i in (nhalf-1)..=0 {

//             theta[i + 1] = theta[i];
//             weights[i + 1] = weights[i];
//         }

//         theta[0] = 0.0;
//         let (pbi, dpbi, ddpbi) = compute_legendre_pol_and_derivative(nb_points_quadrature, theta[0], cz, &coef, &deriv_coef, &second_deriv_coef);
//         pb = pbi;
//         dpb = dpbi;
//         ddpb = ddpbi;
//         weights[0] = 1.0 / (pb * pb);
//     }

//     let mut sum = 0.0;
//     for i in 0..nb_points_gauss {
//         sum += weights[i];
//     }

//     for i in 0..half_nb_points {
//         weights[nb_points_gauss - i - 1] = weights[i];
//         theta[nb_points_gauss - i - 1] = pi - theta[i];
//     }

//     for i in 0..nb_points_gauss {
//         points[i] = 0.5 + 0.5 * theta[i].cos();
//     }

//     // Reverse the arrays
//     for i in 0..nb_points_gauss / 2 {
//         points.swap(i, nb_points_gauss - i - 1);
//         weights.swap(i, nb_points_gauss - i - 1);
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

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

        // let nb_points_gauss = 6;
        // let mut points = vec![0.0; nb_points_gauss];
        // let mut weights = vec![0.0; nb_points_gauss];
        // let gauss_lobatto = false;
        // compute_gauss_formulas(nb_points_gauss, &mut points, &mut weights, gauss_lobatto);

        // println!("Points: {:?}", points);
        // println!("Weights: {:?}", weights);
    }
}
