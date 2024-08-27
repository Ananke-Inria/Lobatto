use std::simd::Simd;

const N: usize = 8;
pub type F64xN = Simd<f64, N>;

//const SQRT_TWENTY_ONE: f64 = 4.582575694955840;

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

const SIMD_DX3: [F64xN; 6] = [
    F64xN::from_array([DX3[0]; N]),
    F64xN::from_array([DX3[1]; N]),
    F64xN::from_array([DX3[2]; N]),
    F64xN::from_array([DX3[3]; N]),
    F64xN::from_array([DX3[4]; N]),
    F64xN::from_array([6.0; N]),
];

// const DX4: [f64; 7] = [
//     2.4819805060619613002,
//     0.51801949393803370381,
//     13.513004977448479949,
//     2.673169155390906937,
//     1.5275252316519465356,
//     2.8203283558848517565,
//     3.4914862437758804603 / (16.0 / 3.0),
// ];

pub fn d_q3_q3<const M: isize>(in_q3: *const f64, out_q3: *mut f64) {
    unsafe {
        let a = in_q3;
        let b = in_q3.offset(M);
        let c = in_q3.offset(2 * M);
        let d = in_q3.offset(3 * M);

        *out_q3 = DX3[2] * (*b) - DX3[3] * (*c) + (*d) - 6.0 * (*a);
        *out_q3.offset(M) = DX3[4] * (*c) - DX3[0] * (*a) - DX3[1] * (*d);
        *out_q3.offset(2 * M) = DX3[1] * (*a) - DX3[4] * (*b) + DX3[0] * (*d);
        *out_q3.offset(3 * M) = DX3[3] * (*b) - DX3[2] * (*c) - (*a) + 6.0 * (*d);
    }
}

pub fn d_q3_q3_simd<const M: isize>(input: *const F64xN, output: *mut F64xN) {
    unsafe {
        let a = input;
        let b = input.offset(M);
        let c = input.offset(2 * M);
        let d = input.offset(3 * M);

        *output = SIMD_DX3[2] * (*b) - SIMD_DX3[3] * (*c) + (*d) - SIMD_DX3[5] * (*a);
        *output.offset(M) = SIMD_DX3[4] * (*c) - SIMD_DX3[0] * (*a) - SIMD_DX3[1] * (*d);
        *output.offset(2 * M) = SIMD_DX3[1] * (*a) - SIMD_DX3[4] * (*b) + SIMD_DX3[0] * (*d);
        *output.offset(3 * M) = SIMD_DX3[3] * (*b) - SIMD_DX3[2] * (*c) - (*a) + SIMD_DX3[5] * (*d);
    }
}

pub fn dx_q33_q33(input: &[f64], output: &mut [f64]) {
    let p_in = input.as_ptr();
    let p_out = output.as_mut_ptr();

    unsafe {
        d_q3_q3::<1>(p_in, p_out);
        d_q3_q3::<1>(p_in.offset(4), p_out.offset(4));
        d_q3_q3::<1>(p_in.offset(8), p_out.offset(8));
        d_q3_q3::<1>(p_in.offset(12), p_out.offset(12));
    }
}

pub fn dx_q33_q33_simd(input: &[f64], output: &mut [f64]) {
    let (prefix, in_shorts, suffix) = input.as_simd::<N>();

    assert!(prefix.len() == 0);
    assert!(suffix.len() == 0);

    let (prefix, out_shorts, suffix) = output.as_simd_mut::<N>();

    assert!(prefix.len() == 0);
    assert!(suffix.len() == 0);

    let p_in: *const F64xN = in_shorts.as_ptr();
    let p_out: *mut F64xN = out_shorts.as_mut_ptr();

    unsafe {
        d_q3_q3_simd::<1>(p_in, p_out);
        d_q3_q3_simd::<1>(p_in.offset(4), p_out.offset(4));
        d_q3_q3_simd::<1>(p_in.offset(8), p_out.offset(8));
        d_q3_q3_simd::<1>(p_in.offset(12), p_out.offset(12));
    }
}

pub fn dx_q333_q333(input: &[f64], output: &mut [f64]) {
    let p_in = input.as_ptr();
    let p_out = output.as_mut_ptr();

    unsafe {
        for i in 0..16 {
            d_q3_q3::<1>(p_in.offset(4 * i), p_out.offset(4 * i));
        }
    }
}

pub fn dx_q333_q333_simd(input: &[f64], output: &mut [f64]) {
    let (prefix, in_shorts, suffix) = input.as_simd::<N>();

    assert!(prefix.len() == 0);
    assert!(suffix.len() == 0);

    let (prefix, out_shorts, suffix) = output.as_simd_mut::<N>();

    assert!(prefix.len() == 0);
    assert!(suffix.len() == 0);

    let p_in: *const F64xN = in_shorts.as_ptr();
    let p_out: *mut F64xN = out_shorts.as_mut_ptr();

    unsafe {
        for i in 0..16 {
            d_q3_q3_simd::<1>(p_in.offset(4 * i), p_out.offset(4 * i));
        }
    }
}
