//! Tests for the lumped mass weights: `Grid1D::mass_weights` and
//! `GridND::mass_weights`.

use lobatto_grid::{Extension, Grid1D, GridND};

const EXTS: [Extension; 3] = [Extension::Periodic, Extension::Odd, Extension::Even];

/// Sum of the weights of the two domain endpoints, i.e. what an `Odd` grid drops.
fn end_weights(g: &Grid1D) -> f64 {
    let w = g.ref_weights();
    g.h() * (w[0] + w[g.r()])
}

#[test]
fn length_matches_n_dof() {
    for ext in EXTS {
        let g = Grid1D::new(5, 4, 2.0, -1.0, ext);
        assert_eq!(g.mass_weights().len(), g.n_dof(), "{ext:?}");
    }
}

#[test]
fn sum_is_domain_length() {
    for ext in EXTS {
        for (n, r) in [(1, 2), (3, 4), (7, 6)] {
            let g = Grid1D::new(n, r, 2.5, -0.3, ext);
            let s: f64 = g.mass_weights().iter().sum();
            let expected = match ext {
                Extension::Odd => 2.5 - end_weights(&g),
                _ => 2.5,
            };
            assert!((s - expected).abs() < 1e-13, "{ext:?} n={n} r={r}: {s} vs {expected}");
        }
    }
}

#[test]
fn integrates_polynomials_exactly() {
    // Gauss-Lobatto with r+1 nodes is exact up to degree 2r-1 per element;
    // x² on [x_l, x_l + L] needs r ≥ 2.
    let (l, xl): (f64, f64) = (1.5, 0.25);
    let exact = ((xl + l).powi(3) - xl.powi(3)) / 3.0;
    for r in 2..=6 {
        let g = Grid1D::new(4, r, l, xl, Extension::Even);
        let q: f64 = g
            .mass_weights()
            .iter()
            .zip(g.get_x())
            .map(|(w, x)| w * x * x)
            .sum();
        assert!((q - exact).abs() < 1e-13, "r={r}: {q} vs {exact}");
    }
}

#[test]
fn shared_node_gets_two_end_weights() {
    let (n, r) = (3, 4);
    for ext in EXTS {
        let g = Grid1D::new(n, r, 1.0, 0.0, ext);
        let w = g.mass_weights();
        // The node between elements 0 and 1.
        let i = g.eval_dof_index(1, 0);
        assert!((w[i] - end_weights(&g)).abs() < 1e-15, "{ext:?}");
        // An interior node of element 1.
        let i = g.eval_dof_index(1, 1);
        assert!((w[i] - g.h() * g.ref_weights()[1]).abs() < 1e-15, "{ext:?}");
    }
    // The periodic seam: node 0 is shared by the last and the first element.
    let g = Grid1D::new(n, r, 1.0, 0.0, Extension::Periodic);
    assert!((g.mass_weights()[0] - end_weights(&g)).abs() < 1e-15);
    // A Neumann end keeps a single end weight.
    let g = Grid1D::new(n, r, 1.0, 0.0, Extension::Even);
    assert!((g.mass_weights()[0] - g.h() * g.ref_weights()[0]).abs() < 1e-15);
}

#[test]
fn nd_is_tensor_product() {
    let g = GridND::<3>::new(
        [2, 3, 2],
        [3, 2, 4],
        [1.0, 2.0, 0.5],
        [0.0, -1.0, 3.0],
        [Extension::Periodic, Extension::Odd, Extension::Even],
    );
    let w = g.mass_weights();
    assert_eq!(w.len(), g.total());
    let w1d: Vec<Vec<f64>> = (0..3).map(|d| g.axis(d).mass_weights()).collect();
    let (nd, st) = (g.ndofs(), g.strides());
    for (p, &wp) in w.iter().enumerate() {
        let expected: f64 = (0..3).map(|d| w1d[d][(p / st[d]) % nd[d]]).product();
        assert_eq!(wp, expected, "p={p}");
    }
    // Sum factorises into the product of the 1D sums.
    let s: f64 = w.iter().sum();
    let s1d: f64 = w1d.iter().map(|v| v.iter().sum::<f64>()).product();
    assert!((s - s1d).abs() < 1e-13);
}
