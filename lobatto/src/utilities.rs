//! Lagrange interpolation in barycentric form.
//!
//! Given $n$ distinct nodes $x_0 < x_1 < \cdots < x_{n-1}$, the $i$-th Lagrange
//! basis polynomial is
//! $$L_i(p) = \prod_{k \ne i} \frac{p - x_k}{x_i - x_k}.$$
//!
//! The **barycentric form** rewrites this as
//! $$L_i(p) = w_i \prod_{k=0}^{n-1}(p - x_k) \Big/ (p - x_i),
//! \quad w_i = \frac{1}{\displaystyle\prod_{k \ne i}(x_i - x_k)},$$
//! allowing all $n$ values $L_0(p), \ldots, L_{n-1}(p)$ to be evaluated in
//! $O(n)$ time once the weights $w_i$ have been precomputed in $O(n^2)$.
//!
//! [`barycentric_weights`] computes the weights; [`lagrangian_interpolation`]
//! evaluates all basis functions simultaneously using a numerically stable
//! product that avoids dividing by the smallest difference term.

/// Evaluate the derivative $L_i'(p)$ of the $i$-th Lagrange basis function at `p`.
///
/// Uses the formula
/// $$L_i'(p) = \frac{1}{\displaystyle\prod_{k \ne i}(x_i - x_k)}
///   \sum_{r \ne i} \prod_{s \ne i,\, s \ne r}(p - x_s).$$
/// `i` must be in `0..pos.len()`.
pub fn lagrangian_derivative_interpolation(pos: &[f64], p: f64, i: usize) -> f64 {
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

/// Precompute barycentric weights $w_i = 1 / \prod_{k \ne i}(x_i - x_k)$ for nodes `x`.
///
/// These weights are independent of the evaluation point and need only be computed
/// once per node set.  Pass them to [`lagrangian_interpolation`] to evaluate
/// all $n$ basis functions at any point in $O(n)$.
///
/// Cost: $O(n^2)$.
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

/// Evaluate all $n$ Lagrange basis functions at `p` using the barycentric form.
///
/// Returns the vector $(L_0(p), \ldots, L_{n-1}(p))$ where
/// $$L_i(p) = w_i \prod_{k=0}^{n-1}(p - x_k) \Big/ (p - x_i).$$
///
/// `pos` are the nodes, `w` the precomputed weights from [`barycentric_weights`].
///
/// **Numerical stability**: the common product $\prod_k(p - x_k)$ is computed
/// with the term nearest to $p$ excluded, avoiding division by the smallest
/// (and most cancellation-prone) factor.
///
/// Cost: $O(n)$ after the $O(n^2)$ weight precomputation.
pub fn lagrangian_interpolation(pos: &[f64], w: &[f64], p: f64) -> Vec<f64> {
    let mut res = vec![0.0_f64; pos.len()];
    lagrangian_interpolation_into(pos, w, p, &mut res);
    res
}

/// Allocation-free variant of [`lagrangian_interpolation`]: writes
/// $(L_0(p), \ldots, L_{n-1}(p))$ into `out` (length `n`).
///
/// Same barycentric algorithm and stability property; `out` doubles as the
/// scratch buffer for the node differences, so no allocation is performed.
pub fn lagrangian_interpolation_into(pos: &[f64], w: &[f64], p: f64, out: &mut [f64]) {
    let n = pos.len();
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(w.len(), n);

    // out[r] = p - pos[r]; i_min is the index of the closest node.
    let mut i_min = 0;
    for r in 0..n {
        out[r] = p - pos[r];
        if out[r].abs() < out[i_min].abs() {
            i_min = r;
        }
    }

    // Common factor: Π_{r ≠ i_min}(p - pos[r])  (never contains the smallest term).
    let mut prod_skip = 1.0_f64;
    for r in 0..n {
        if r != i_min {
            prod_skip *= out[r];
        }
    }

    // L_i(p) = w[i] * Π_{r≠i}(p - pos[r]).
    //   i = i_min : Π_{r≠i} = prod_skip                      (no division by small term)
    //   i ≠ i_min : Π_{r≠i} = prod_skip * diffs[i_min] / diffs[i]
    let d_min = out[i_min];
    for i in 0..n {
        out[i] = if i == i_min {
            w[i] * prod_skip
        } else {
            w[i] * prod_skip * d_min / out[i]
        };
    }
}
