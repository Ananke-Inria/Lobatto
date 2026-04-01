//! Lagrange interpolation utilities.
//!
//! Provides both the direct O(r) single-basis evaluation and the barycentric
//! form that evaluates all r basis values at once in O(r) after an O(r²)
//! weight precomputation — faster and more numerically stable for repeated use.

/// Evaluate the `i`-th Lagrange basis function at `p`.
///
/// The basis is built on the nodes `pos`.  `i` must be in `0..pos.len()`.
pub fn lagrangian_interpolation(pos: &Vec<f64>, p: f64, i: usize) -> f64 {
    let mut res: f64 = 1.0;
    let mut w = 1.0;

    for r in 0..pos.len() {
        if r != i {
            w *= pos[i] - pos[r];
        }
    }

    debug_assert!(w != 0.0);

    for r in 0..pos.len() {
        if r != i {
            res *= p - pos[r];
        }
    }

    res / w
}

/// Evaluate the derivative of the `i`-th Lagrange basis function at `p`.
///
/// The basis is built on the nodes `pos`.  `i` must be in `0..pos.len()`.
pub fn lagrangian_derivative_interpolation(pos: &Vec<f64>, p: f64, i: usize) -> f64 {
    let mut s = 0.0;
    let mut w = 1.0;

    for r in 0..pos.len() {
        if r != i {
            w *= pos[i] - pos[r];
        }
    }

    debug_assert!(w != 0.0);

    for r1 in 0..pos.len() {
        if r1 != i {
            let mut c = 1.0;
            for r2 in 0..pos.len() {
                if r2 != r1 && r2 != i {
                    c *= p - pos[r2];
                }
            }
            s += c;
        }
    }

    s / w
}

/// Compute barycentric weights `w[j] = 1 / Π_{k≠j}(x[j] - x[k])` for nodes `x`.
///
/// Cost: O(r²).  Compute once per node set and reuse across many evaluation points.
pub fn barycentric_weights(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut w = vec![1.0_f64; n];
    for j in 0..n {
        for k in 0..n {
            if k != j {
                w[j] *= x[j] - x[k];
            }
        }

        debug_assert!(w[j] != 0.0);

        w[j] = 1.0 / w[j];
    }
    w
}

/// Evaluate all Lagrange basis functions at `p`.
///
/// Returns a vector of length `pos.len()` where entry `i` is `L_i(p)`.
/// The basis is built on the nodes `pos`; `w` must be the precomputed barycentric
/// weights from [`barycentric_weights`].  Cost: O(r) after the O(r²) precomputation.
pub fn lagrangian_interpolation_all(pos: &Vec<f64>, w: &Vec<f64>, p: f64) -> Vec<f64> {
    let n = pos.len();

    // Build diffs[r] = p - pos[r], tracking the index of the smallest |diff|.
    let mut diffs = vec![0.0_f64; n];
    let mut i_min = 0;
    for r in 0..n {
        diffs[r] = p - pos[r];
        if diffs[r].abs() < diffs[i_min].abs() {
            i_min = r;
        }
    }

    // Product excluding the closest node — never divides by the smallest term.
    let mut prod_skip = 1.0_f64;
    for r in 0..n {
        if r != i_min {
            prod_skip *= diffs[r];
        }
    }

    // L_i(p) = w[i] * Π_{r≠i}(p - pos[r]).
    //   i = i_min : Π_{r≠i} = prod_skip                         (no division)
    //   i ≠ i_min : Π_{r≠i} = prod_skip * diffs[i_min] / diffs[i]
    let mut res = vec![0.0_f64; n];
    for i in 0..n {
        res[i] = if i == i_min {
            w[i] * prod_skip
        } else {
            w[i] * prod_skip * diffs[i_min] / diffs[i]
        };
    }

    res
}
