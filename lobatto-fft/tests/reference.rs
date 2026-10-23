//! Regression against a stored solution: the allocation-free hot paths
//! (TODO item 5) must reproduce the pre-optimisation output bit for bit.
//!
//! `tests/data/reference_3d.txt` was written by `write_reference` (ignored)
//! before the optimisation; rerun it with
//! `cargo test --release --test reference -- --ignored` only when the
//! numerics change on purpose. Format: one `re im` pair per line, `{:e}`,
//! which round-trips `f64` exactly.

use lobatto_fft::solver::{BoundaryCondition, PoissonND};
use rustfft::num_complex::Complex;

const FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/reference_3d.txt");

/// Mixed 3D problem, several elements per direction, complex `α`; the
/// result of `solve_in_place` followed by `exp_in_place`, which exercises
/// both transforms and the per-mode step twice.
fn compute() -> Vec<Complex<f64>> {
    let s = PoissonND::<3>::new(
        [1.0, 2.0, 0.5],
        [0.0, -1.0, 0.3],
        [3, 4, 3],
        [4, 3, 2],
        [BoundaryCondition::Periodic, BoundaryCondition::Dirichlet, BoundaryCondition::Neumann],
    );
    let mut u: Vec<Complex<f64>> = s
        .get_x()
        .iter()
        .map(|x| Complex::new((3.0 * x[0]).sin() * x[1].cos() + x[2], x[0] * x[1] - x[2]))
        .collect();
    s.solve_in_place(&mut u, Complex::new(0.7, 0.3), 1.3);
    s.exp_in_place(&mut u, 0.2, 1.1, 0.01);
    u
}

#[test]
#[ignore = "writes the reference file; run only to regenerate it"]
fn write_reference() {
    let text: String = compute().iter().map(|z| format!("{:e} {:e}\n", z.re, z.im)).collect();
    std::fs::write(FILE, text).unwrap();
}

#[test]
fn matches_reference_bit_for_bit() {
    let text = std::fs::read_to_string(FILE).unwrap();
    let reference: Vec<Complex<f64>> = text
        .lines()
        .map(|l| {
            let mut it = l.split_whitespace().map(|x| x.parse::<f64>().unwrap());
            Complex::new(it.next().unwrap(), it.next().unwrap())
        })
        .collect();
    let u = compute();
    assert_eq!(u.len(), reference.len());
    for (i, (a, b)) in u.iter().zip(&reference).enumerate() {
        assert!(a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits(), "DOF {i}: {a} vs {b}");
    }
}
