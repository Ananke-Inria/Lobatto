use std::simd::Simd;
const N_SIMD: usize = 2;
pub type F64xN = Simd<f64, N_SIMD>;

const SQRT_TWENTY_ONE: f64 = 4.582575694955840;

// constexpr Real a = (7.0-sqrt_twentyone)/14.0;
// constexpr Real w[3] ={1 - 3*a + 2*a*a, 4*a - 4*a*a, 2*a*a - a};
// constexpr Real z[2] = {1-a, a};

const DX3: [f64; 5] = [
    1.6180339887498944584,
    0.61803398874989456946,
    8.0901699437494745126,
    3.0901699437494727363,
    2.2360679774997902491,
];

const SIMD_DX3: [Simd<f64, N_SIMD>; 6] = [
    F64xN::from_array([DX3[0]; N_SIMD]),
    F64xN::from_array([DX3[1]; N_SIMD]),
    F64xN::from_array([DX3[2]; N_SIMD]),
    F64xN::from_array([DX3[3]; N_SIMD]),
    F64xN::from_array([DX3[4]; N_SIMD]),
    F64xN::from_array([6.0; N_SIMD]),
];

const DX4: [f64; 7] = [
    2.4819805060619613002,
    0.51801949393803370381,
    13.513004977448479949,
    2.673169155390906937,
    1.5275252316519465356,
    2.8203283558848517565,
    3.4914862437758804603 / (16.0 / 3.0),
];

pub fn d_q3_q3<const N: isize>(in_q3: *const f64, out_q3: *mut f64) {
    unsafe {
        let a = in_q3;
        let b = in_q3.offset(N);
        let c = in_q3.offset(2 * N);
        let d = in_q3.offset(3 * N);

        *out_q3 = DX3[2] * (*b) - DX3[3] * (*c) + (*d) - 6.0 * (*a);
        *out_q3.offset(N) = DX3[4] * (*c) - DX3[0] * (*a) - DX3[1] * (*d);
        *out_q3.offset(2 * N) = DX3[1] * (*a) - DX3[4] * (*b) + DX3[0] * (*d);
        *out_q3.offset(3 * N) = DX3[3] * (*b) - DX3[2] * (*c) - (*a) + 6.0 * (*d);
    }
}

pub fn d_q3_q3_simd<const N: isize>(in_q3: *const F64xN, out_q3: *mut F64xN) {
    unsafe {
        let a = in_q3;
        let b = in_q3.offset(N);
        let c = in_q3.offset(2 * N);
        let d = in_q3.offset(3 * N);

        *out_q3 = SIMD_DX3[2] * (*b) - SIMD_DX3[3] * (*c) + (*d) - SIMD_DX3[5] * (*a);
        *out_q3.offset(N) = SIMD_DX3[4] * (*c) - SIMD_DX3[0] * (*a) - SIMD_DX3[1] * (*d);
        *out_q3.offset(2 * N) = SIMD_DX3[1] * (*a) - SIMD_DX3[4] * (*b) + SIMD_DX3[0] * (*d);
        *out_q3.offset(3 * N) = SIMD_DX3[3] * (*b) - SIMD_DX3[2] * (*c) - (*a) + SIMD_DX3[5] * (*d);
    }
}

pub fn dx_q3_q3(in_q3: *const f64, out_q3: *mut f64) {
    d_q3_q3::<1>(in_q3, out_q3);
}

pub fn dx_q33_q33(in_q3: &Vec<f64>, out_q3: &mut Vec<f64>) {
    let p_in_q3 = in_q3.as_ptr();
    let p_out_q3 = out_q3.as_mut_ptr();

    unsafe {
        d_q3_q3::<1>(p_in_q3, p_out_q3);
        d_q3_q3::<1>(p_in_q3.offset(4), p_out_q3.offset(4));
        d_q3_q3::<1>(p_in_q3.offset(8), p_out_q3.offset(8));
        d_q3_q3::<1>(p_in_q3.offset(12), p_out_q3.offset(12));
    }
}

pub fn dx_q33_q33_simd(in_q3: &Vec<F64xN>, out_q3: &mut Vec<F64xN>) {
    let p_in_q3 = in_q3.as_ptr();
    let p_out_q3 = out_q3.as_mut_ptr();

    unsafe {
        d_q3_q3_simd::<1>(p_in_q3, p_out_q3);
        d_q3_q3_simd::<1>(p_in_q3.offset(4), p_out_q3.offset(4));
        d_q3_q3_simd::<1>(p_in_q3.offset(8), p_out_q3.offset(8));
        d_q3_q3_simd::<1>(p_in_q3.offset(12), p_out_q3.offset(12));
    }
}

pub fn dx_q333_q333(in_q3: &Vec<f64>, out_q3: &mut Vec<f64>) {
    let p_in_q3 = in_q3.as_ptr();
    let p_out_q3 = out_q3.as_mut_ptr();

    unsafe {
        for i in 0..16 {
            d_q3_q3::<1>(p_in_q3.offset(4 * i), p_out_q3.offset(4 * i));
        }
    }
}
