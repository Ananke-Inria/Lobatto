use crate::interpolation::*;
use nalgebra::*;
use std::f64::consts::PI;

pub fn symbol_1d_lo(rho: f64, h: f64, l: f64, k: usize) -> f64 {
    let symbol = rho - 2.0 * (f64::cos(2.0 * PI * h * k as f64 / l) - 1.0) / (h * h);

    symbol
}

//pub fn symbol_1d(order: usize, rho: f64, h: f64, l: f64, k: usize) -> DMatrixf64 {}
