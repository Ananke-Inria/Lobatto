//! Tests for the element-wise nodal derivatives `Grid1D::derivative` and
//! `GridND::derivative`.

use lobatto_grid::{Extension, Grid1D, GridND};

fn derivative(g: &Grid1D, u: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; g.n_dof()];
    g.derivative(u, &mut out);
    out
}

#[test]
fn exact_for_polynomials_of_degree_r() {
    // Degree r: one polynomial on every element, so both one-sided
    // derivatives at a shared node agree and their average is exact.
    let (xl, l) = (-0.4, 2.5);
    for (n, r) in [(1, 2), (3, 4), (5, 6)] {
        let even = Grid1D::new(n, r, l, xl, Extension::Even);
        let x = even.get_x();
        let u: Vec<f64> = x.iter().map(|&x| (x - 0.3).powi(r as i32)).collect();
        for (d, &x) in derivative(&even, &u).iter().zip(&x) {
            let exact = r as f64 * (x - 0.3).powi(r as i32 - 1);
            assert!((d - exact).abs() < 1e-10 * (1.0 + exact.abs()), "Even n={n} r={r}");
        }

        // Vanishing at both ends, so the unstored Odd zeros are its values.
        let odd = Grid1D::new(n, r, l, xl, Extension::Odd);
        let x = odd.get_x();
        let f = |x: f64| (x - xl) * (xl + l - x) * (x + 1.0).powi(r as i32 - 2);
        let df = |x: f64| {
            (xl + l - 2.0 * x + xl) * (x + 1.0).powi(r as i32 - 2)
                + (x - xl) * (xl + l - x) * (r as f64 - 2.0) * (x + 1.0).powi(r as i32 - 3)
        };
        let u: Vec<f64> = x.iter().map(|&x| f(x)).collect();
        for (d, &x) in derivative(&odd, &u).iter().zip(&x) {
            assert!((d - df(x)).abs() < 1e-10 * (1.0 + df(x).abs()), "Odd n={n} r={r}");
        }
    }
}

#[test]
fn spectral_accuracy_on_periodic_including_the_seam() {
    let (xl, l) = (0.7, 3.0);
    let k = 2.0 * std::f64::consts::PI / l;
    for n in [1, 4] {
        let errs: Vec<f64> = [6, 10, 14]
            .iter()
            .map(|&r| {
                let g = Grid1D::new(n, r, l, xl, Extension::Periodic);
                let x = g.get_x();
                let u: Vec<f64> = x.iter().map(|&x| (k * (x - xl)).sin()).collect();
                derivative(&g, &u)
                    .iter()
                    .zip(&x)
                    .map(|(d, &x)| (d - k * (k * (x - xl)).cos()).abs())
                    .fold(0.0, f64::max)
            })
            .collect();
        // Spectral: each step of 4 in r gains well over two orders.
        assert!(errs[1] < 1e-2 * errs[0] && errs[2] < 1e-2 * errs[1], "n={n} {errs:?}");
        assert!(errs[2] < 1e-7, "n={n} {errs:?}");
    }
}

#[test]
fn shared_node_averages_one_sided_derivatives() {
    // |x - 1| on two elements of [0, 2]: slope -1 on the left element, +1 on
    // the right one, 0 (the average) at the shared node x = 1.
    let g = Grid1D::new(2, 4, 2.0, 0.0, Extension::Even);
    let x = g.get_x();
    let u: Vec<f64> = x.iter().map(|&x| (x - 1.0).abs()).collect();
    for (d, &x) in derivative(&g, &u).iter().zip(&x) {
        let exact = if x < 1.0 { -1.0 } else if x > 1.0 { 1.0 } else { 0.0 };
        assert!((d - exact).abs() < 1e-12, "x={x} d={d}");
    }
}

#[test]
fn odd_matches_even_with_zero_end_values() {
    // The Odd boundary zeros enter as zeros: same result as the Even grid
    // with the two endpoint values set to zero, restricted to the interior.
    let (n, r, l, xl) = (3, 5, 1.5, 0.2);
    let odd = Grid1D::new(n, r, l, xl, Extension::Odd);
    let even = Grid1D::new(n, r, l, xl, Extension::Even);
    let u: Vec<f64> = odd.get_x().iter().map(|&x| (3.0 * x).cos() + x).collect();
    let mut ue = vec![0.0; even.n_dof()];
    ue[1..even.n_dof() - 1].copy_from_slice(&u);
    let de = derivative(&even, &ue);
    for (a, b) in derivative(&odd, &u).iter().zip(&de[1..even.n_dof() - 1]) {
        assert!((a - b).abs() < 1e-12 * (1.0 + b.abs()));
    }
}

fn grid_3d() -> GridND<3> {
    GridND::new(
        [2, 3, 2],
        [3, 2, 4],
        [1.0, 2.0, 0.5],
        [0.0, -1.0, 3.0],
        [Extension::Periodic, Extension::Odd, Extension::Even],
    )
}

#[test]
fn nd_matches_1d_derivative_on_every_fibre() {
    let g = grid_3d();
    let (nd, st) = (g.ndofs(), g.strides());
    let u: Vec<f64> = (0..g.total()).map(|p| ((p * 7919) % 101) as f64 / 101.0 - 0.5).collect();
    for d in 0..3 {
        let out = {
            let mut out = vec![f64::NAN; g.total()];
            g.derivative(d, &u, &mut out);
            out
        };
        // Reference: walk every fibre of direction d by its base point.
        for base in (0..g.total()).filter(|p| (p / st[d]) % nd[d] == 0) {
            let fibre: Vec<f64> = (0..nd[d]).map(|j| u[base + st[d] * j]).collect();
            let df = derivative(g.axis(d), &fibre);
            for j in 0..nd[d] {
                assert_eq!(out[base + st[d] * j], df[j], "d={d} base={base} j={j}");
            }
        }
    }
}

#[test]
fn nd_exact_for_separable_polynomials() {
    // Degree ≤ r_d per direction on an all-Even grid: exact derivatives.
    let g = GridND::<3>::new(
        [2, 3, 2],
        [3, 2, 4],
        [1.0, 2.0, 0.5],
        [0.0, -1.0, 3.0],
        [Extension::Even; 3],
    );
    let x = g.get_x();
    let u: Vec<f64> = x.iter().map(|x| x[0].powi(3) * (x[1] + 0.5).powi(2) * x[2].powi(4)).collect();
    let exact: [fn(&[f64; 3]) -> f64; 3] = [
        |x| 3.0 * x[0].powi(2) * (x[1] + 0.5).powi(2) * x[2].powi(4),
        |x| x[0].powi(3) * 2.0 * (x[1] + 0.5) * x[2].powi(4),
        |x| x[0].powi(3) * (x[1] + 0.5).powi(2) * 4.0 * x[2].powi(3),
    ];
    for (d, e) in exact.iter().enumerate() {
        let mut out = vec![0.0; g.total()];
        g.derivative(d, &u, &mut out);
        for (o, x) in out.iter().zip(&x) {
            assert!((o - e(x)).abs() < 1e-9 * (1.0 + e(x).abs()), "d={d} x={x:?}");
        }
    }
}

#[test]
#[should_panic(expected = "u length differs from total")]
fn nd_panics_on_wrong_input_length() {
    let g = grid_3d();
    g.derivative(1, &[0.0; 5], &mut vec![0.0; g.total()]);
}

#[test]
#[should_panic(expected = "out length differs from N_dof")]
fn panics_on_wrong_output_length() {
    let g = Grid1D::new(3, 4, 1.0, 0.0, Extension::Periodic);
    g.derivative(&vec![0.0; g.n_dof()], &mut [0.0; 3]);
}
