//! Regression against a stored solution: the allocation-free hot paths
//! (TODO item 5) must reproduce the pre-optimisation output bit for bit.
//!
//! `tests/data/reference_3d.txt` was written by `write_reference` (ignored)
//! before the optimisation; rerun it with
//! `cargo test --release --test reference -- --ignored` only when the
//! numerics change on purpose. Format: one value per line, `{:e}`, which
//! round-trips `f64` exactly.

use lobatto_spectral::solver::{BoundaryCondition, PoissonND};

const FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/reference_3d.txt");

/// Mixed 3D problem, several elements per direction; the result of
/// `solve_in_place` followed by `exp_in_place`, which exercises every sweep
/// twice.
fn compute() -> Vec<f64> {
    let s = PoissonND::<3>::new(
        [1.0, 2.0, 0.5],
        [0.0, -1.0, 0.3],
        [3, 4, 3],
        [4, 3, 2],
        [BoundaryCondition::Periodic, BoundaryCondition::Dirichlet, BoundaryCondition::Neumann],
    );
    let mut u: Vec<f64> = s
        .get_x()
        .iter()
        .map(|x| (3.0 * x[0]).sin() * x[1].cos() + x[2])
        .collect();
    s.solve_in_place(&mut u, 0.7, 1.3);
    s.exp_in_place(&mut u, 0.2, 1.1, 0.01);
    u
}

#[test]
#[ignore = "writes the reference file; run only to regenerate it"]
fn write_reference() {
    let text: String = compute().iter().map(|v| format!("{v:e}\n")).collect();
    std::fs::write(FILE, text).unwrap();
}

#[test]
fn matches_reference_bit_for_bit() {
    let text = std::fs::read_to_string(FILE).unwrap();
    let reference: Vec<f64> = text.lines().map(|l| l.trim().parse().unwrap()).collect();
    let u = compute();
    assert_eq!(u.len(), reference.len());
    for (i, (a, b)) in u.iter().zip(&reference).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "DOF {i}: {a} vs {b}");
    }
}
