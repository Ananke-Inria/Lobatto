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

// const DX4: [f64; 7] = [
//     2.4819805060619613002,
//     0.51801949393803370381,
//     13.513004977448479949,
//     2.673169155390906937,
//     1.5275252316519465356,
//     2.8203283558848517565,
//     3.4914862437758804603 / (16.0 / 3.0),
// ];

pub fn d_q3_q3<const M: usize>(in_q3: &[f64], out_q3: &mut [f64]) {
    assert!(in_q3.len() > 3 * M);
    assert!(out_q3.len() > 3 * M);

    let a = &in_q3[0];
    let b = &in_q3[M];
    let c = &in_q3[2 * M];
    let d = &in_q3[3 * M];

    out_q3[0] = DX3[2] * (*b) - DX3[3] * (*c) + (*d) - 6.0 * (*a);
    out_q3[M] = DX3[4] * (*c) - DX3[0] * (*a) - DX3[1] * (*d);
    out_q3[2 * M] = DX3[1] * (*a) - DX3[4] * (*b) + DX3[0] * (*d);
    out_q3[3 * M] = DX3[3] * (*b) - DX3[2] * (*c) - (*a) + 6.0 * (*d);
}

pub fn dx_q33_q33(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 16 && output.len() >= 16);

    d_q3_q3::<1>(&input[0..4], &mut output[0..4]);
    d_q3_q3::<1>(&input[4..8], &mut output[4..8]);
    d_q3_q3::<1>(&input[8..12], &mut output[8..12]);
    d_q3_q3::<1>(&input[12..16], &mut output[12..16]);
}

pub fn dx_q333_q333(input: &[f64], output: &mut [f64]) {
    assert!(input.len() >= 64 && output.len() >= 64);
    for i in 0..16 {
        let start = 4 * i;
        let end = start + 4;
        d_q3_q3::<1>(&input[start..end], &mut output[start..end]);
    }
}

#[inline(never)]
pub fn d_q3_q3_vec<const N: usize, const M: usize>(in_q3: &[f64], out_q3: &mut [f64]) {
    assert!(M == 1);
    assert!(in_q3.len() == N * 4);
    assert!(out_q3.len() == N * 4);
    let a = &in_q3[0..N];
    let b = &in_q3[M * N..M * N + N];
    let c = &in_q3[2 * M * N..2 * M * N + N];
    let d = &in_q3[3 * M * N..3 * M * N + N];

    let (a_out, rest) = out_q3.split_at_mut(N);
    let (b_out, rest) = rest.split_at_mut(N);
    let (c_out, d_out) = rest.split_at_mut(N);

    for ((((a_out, a), b), c), d) in a_out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
        *a_out = DX3[2] * (*b) - DX3[3] * (*c) + (*d) - 6.0 * (*a);
    }

    for (((b_out, a), c), d) in b_out.iter_mut().zip(a).zip(c).zip(d) {
        *b_out = DX3[4] * (*c) - DX3[0] * (*a) - DX3[1] * (*d);
    }

    for (((c_out, a), b), d) in c_out.iter_mut().zip(a).zip(b).zip(d) {
        *c_out = DX3[1] * (*a) - DX3[4] * (*b) + DX3[0] * (*d);
    }

    for ((((d_out, a), b), c), d) in d_out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
        *d_out = DX3[3] * (*b) - DX3[2] * (*c) - (*a) + 6.0 * (*d);
    }
}

pub fn dx_q33_q33_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..4 {
        let start = N * 4 * i;
        let end = start + N * 4;
        d_q3_q3_vec::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}

pub fn dx_q333_q333_vec<const N: usize>(input: &[f64], output: &mut [f64]) {
    for i in 0..16 {
        let start = N * 4 * i;
        let end = start + N * 4;
        d_q3_q3_vec::<N, 1>(&input[start..end], &mut output[start..end]);
    }
}
