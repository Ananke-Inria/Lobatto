//! Tests for the derived traits of `Grid1D` and `GridND`: `Clone`, `PartialEq`.

use lobatto_grid::{Extension, Grid1D, GridND};

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
fn equal_parameters_give_equal_grids() {
    assert_eq!(
        Grid1D::new(4, 3, 1.5, 0.25, Extension::Even),
        Grid1D::new(4, 3, 1.5, 0.25, Extension::Even)
    );
    assert_eq!(grid_3d(), grid_3d());
}

#[test]
fn clone_is_equal() {
    let g = grid_3d();
    let c = g.clone();
    assert_eq!(g, c);
    assert_eq!(g.get_x(), c.get_x());
    assert_eq!(g.mass_weights(), c.mass_weights());
}

#[test]
fn any_parameter_change_breaks_equality() {
    let g = Grid1D::new(4, 3, 1.5, 0.25, Extension::Even);
    assert_ne!(g, Grid1D::new(5, 3, 1.5, 0.25, Extension::Even));
    assert_ne!(g, Grid1D::new(4, 4, 1.5, 0.25, Extension::Even));
    assert_ne!(g, Grid1D::new(4, 3, 2.0, 0.25, Extension::Even));
    assert_ne!(g, Grid1D::new(4, 3, 1.5, 0.0, Extension::Even));
    assert_ne!(g, Grid1D::new(4, 3, 1.5, 0.25, Extension::Periodic));
}
