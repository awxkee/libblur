/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::neon::median::median_3::{SimdU8, load, load8};
use crate::{BlurImage, BlurImageMut, ThreadingPolicy};
use novtb::{ParallelZonedIterator, TbSliceMut};
use std::arch::aarch64::*;

#[inline(always)]
fn load_scalar_7x7<const CN: usize>(rows: [&[u8]; 7], x: usize, width: usize) -> [u8; CN] {
    let x0 = x.saturating_sub(3 * CN);
    let x1 = x.saturating_sub(2 * CN);
    let x2 = x.saturating_sub(CN);
    let x3 = x;
    let x4 = (x + CN).min(width - CN);
    let x5 = (x + 2 * CN).min(width - CN);
    let x6 = (x + 3 * CN).min(width - CN);
    let mut result = [0u8; CN];
    for c in 0..CN {
        let mut vals = [
            rows[0][x0 + c],
            rows[0][x1 + c],
            rows[0][x2 + c],
            rows[0][x3 + c],
            rows[0][x4 + c],
            rows[0][x5 + c],
            rows[0][x6 + c],
            rows[1][x0 + c],
            rows[1][x1 + c],
            rows[1][x2 + c],
            rows[1][x3 + c],
            rows[1][x4 + c],
            rows[1][x5 + c],
            rows[1][x6 + c],
            rows[2][x0 + c],
            rows[2][x1 + c],
            rows[2][x2 + c],
            rows[2][x3 + c],
            rows[2][x4 + c],
            rows[2][x5 + c],
            rows[2][x6 + c],
            rows[3][x0 + c],
            rows[3][x1 + c],
            rows[3][x2 + c],
            rows[3][x3 + c],
            rows[3][x4 + c],
            rows[3][x5 + c],
            rows[3][x6 + c],
            rows[4][x0 + c],
            rows[4][x1 + c],
            rows[4][x2 + c],
            rows[4][x3 + c],
            rows[4][x4 + c],
            rows[4][x5 + c],
            rows[4][x6 + c],
            rows[5][x0 + c],
            rows[5][x1 + c],
            rows[5][x2 + c],
            rows[5][x3 + c],
            rows[5][x4 + c],
            rows[5][x5 + c],
            rows[5][x6 + c],
            rows[6][x0 + c],
            rows[6][x1 + c],
            rows[6][x2 + c],
            rows[6][x3 + c],
            rows[6][x4 + c],
            rows[6][x5 + c],
            rows[6][x6 + c],
        ];
        vals.sort_unstable();
        result[c] = vals[24]; // median of 49
    }
    result
}

#[allow(unused)]
#[inline(always)]
fn median_network_7x7<S: SimdU8>(
    p00: S,
    p01: S,
    p02: S,
    p03: S,
    p04: S,
    p05: S,
    p06: S,
    p07: S,
    p08: S,
    p09: S,
    p10: S,
    p11: S,
    p12: S,
    p13: S,
    p14: S,
    p15: S,
    p16: S,
    p17: S,
    p18: S,
    p19: S,
    p20: S,
    p21: S,
    p22: S,
    p23: S,
    p24: S,
    p25: S,
    p26: S,
    p27: S,
    p28: S,
    p29: S,
    p30: S,
    p31: S,
    p32: S,
    p33: S,
    p34: S,
    p35: S,
    p36: S,
    p37: S,
    p38: S,
    p39: S,
    p40: S,
    p41: S,
    p42: S,
    p43: S,
    p44: S,
    p45: S,
    p46: S,
    p47: S,
    p48: S,
) -> S {
    let mut v = [
        p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17,
        p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35,
        p36, p37, p38, p39, p40, p41, p42, p43, p44, p45, p46, p47, p48,
    ];

    for i in 1..49 {
        let mut j = i;
        while j > 0 {
            let lo = S::min(v[j - 1], v[j]);
            let hi = S::max(v[j - 1], v[j]);
            v[j - 1] = lo;
            v[j] = hi;
            j -= 1;
        }
    }

    v[24]
}

pub(crate) fn median_blur_7x7(
    src: &BlurImage<u8>,
    dst: &mut BlurImageMut<u8>,
    channels: usize,
    threading_policy: ThreadingPolicy,
) {
    let thread_count = threading_policy.thread_count(src.width, src.height) as u32;
    let pool = novtb::ThreadPool::new(thread_count as usize);
    let dst_stride = dst.row_stride() as usize;
    let height = src.height as usize;
    let src_stride = src.row_stride() as usize;
    dst.data
        .borrow_mut()
        .tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(&pool, |y, row| {
            let rows = [
                &src.data.as_ref()[y.saturating_sub(3) * src_stride..],
                &src.data.as_ref()[y.saturating_sub(2) * src_stride..],
                &src.data.as_ref()[y.saturating_sub(1) * src_stride..],
                &src.data.as_ref()[y * src_stride..],
                &src.data.as_ref()[(y + 1).min(height - 1) * src_stride..],
                &src.data.as_ref()[(y + 2).min(height - 1) * src_stride..],
                &src.data.as_ref()[(y + 3).min(height - 1) * src_stride..],
            ];
            unsafe {
                median_blur_7x7_impl(row, rows, src.width as usize * channels, channels);
            }
        });
}

#[inline(always)]
fn write_scalar_7x7(dst: &mut [u8], rows: [&[u8]; 7], x: usize, width: usize, channels: usize) {
    match channels {
        1 => {
            let px = load_scalar_7x7::<1>(rows, x, width);
            dst[x] = px[0];
        }
        3 => {
            let px = load_scalar_7x7::<3>(rows, x, width);
            dst[x] = px[0];
            dst[x + 1] = px[1];
            dst[x + 2] = px[2];
        }
        4 => {
            let px = load_scalar_7x7::<4>(rows, x, width);
            dst[x] = px[0];
            dst[x + 1] = px[1];
            dst[x + 2] = px[2];
            dst[x + 3] = px[3];
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "neon")]
fn median_blur_7x7_impl(dst: &mut [u8], rows: [&[u8]; 7], width: usize, channels: usize) {
    // Left border: 3 pixels need clamped left neighbors
    for bx in 0..3 {
        let x = bx * channels;
        write_scalar_7x7(dst, rows, x, width, channels);
    }

    let mut x = 3 * channels;

    // SIMD 16-wide loop
    while x + 3 * channels + 16 <= width {
        let med = median_network_7x7(
            load(rows[0], x - 3 * channels),
            load(rows[0], x - 2 * channels),
            load(rows[0], x - channels),
            load(rows[0], x),
            load(rows[0], x + channels),
            load(rows[0], x + 2 * channels),
            load(rows[0], x + 3 * channels),
            load(rows[1], x - 3 * channels),
            load(rows[1], x - 2 * channels),
            load(rows[1], x - channels),
            load(rows[1], x),
            load(rows[1], x + channels),
            load(rows[1], x + 2 * channels),
            load(rows[1], x + 3 * channels),
            load(rows[2], x - 3 * channels),
            load(rows[2], x - 2 * channels),
            load(rows[2], x - channels),
            load(rows[2], x),
            load(rows[2], x + channels),
            load(rows[2], x + 2 * channels),
            load(rows[2], x + 3 * channels),
            load(rows[3], x - 3 * channels),
            load(rows[3], x - 2 * channels),
            load(rows[3], x - channels),
            load(rows[3], x),
            load(rows[3], x + channels),
            load(rows[3], x + 2 * channels),
            load(rows[3], x + 3 * channels),
            load(rows[4], x - 3 * channels),
            load(rows[4], x - 2 * channels),
            load(rows[4], x - channels),
            load(rows[4], x),
            load(rows[4], x + channels),
            load(rows[4], x + 2 * channels),
            load(rows[4], x + 3 * channels),
            load(rows[5], x - 3 * channels),
            load(rows[5], x - 2 * channels),
            load(rows[5], x - channels),
            load(rows[5], x),
            load(rows[5], x + channels),
            load(rows[5], x + 2 * channels),
            load(rows[5], x + 3 * channels),
            load(rows[6], x - 3 * channels),
            load(rows[6], x - 2 * channels),
            load(rows[6], x - channels),
            load(rows[6], x),
            load(rows[6], x + channels),
            load(rows[6], x + 2 * channels),
            load(rows[6], x + 3 * channels),
        );
        unsafe {
            vst1q_u8(dst.get_unchecked_mut(x..).as_mut_ptr().cast(), med.0);
        }
        x += 16;
    }

    // SIMD 8-wide loop
    while x + 3 * channels + 8 <= width {
        let med = median_network_7x7(
            load8(rows[0], x - 3 * channels),
            load8(rows[0], x - 2 * channels),
            load8(rows[0], x - channels),
            load8(rows[0], x),
            load8(rows[0], x + channels),
            load8(rows[0], x + 2 * channels),
            load8(rows[0], x + 3 * channels),
            load8(rows[1], x - 3 * channels),
            load8(rows[1], x - 2 * channels),
            load8(rows[1], x - channels),
            load8(rows[1], x),
            load8(rows[1], x + channels),
            load8(rows[1], x + 2 * channels),
            load8(rows[1], x + 3 * channels),
            load8(rows[2], x - 3 * channels),
            load8(rows[2], x - 2 * channels),
            load8(rows[2], x - channels),
            load8(rows[2], x),
            load8(rows[2], x + channels),
            load8(rows[2], x + 2 * channels),
            load8(rows[2], x + 3 * channels),
            load8(rows[3], x - 3 * channels),
            load8(rows[3], x - 2 * channels),
            load8(rows[3], x - channels),
            load8(rows[3], x),
            load8(rows[3], x + channels),
            load8(rows[3], x + 2 * channels),
            load8(rows[3], x + 3 * channels),
            load8(rows[4], x - 3 * channels),
            load8(rows[4], x - 2 * channels),
            load8(rows[4], x - channels),
            load8(rows[4], x),
            load8(rows[4], x + channels),
            load8(rows[4], x + 2 * channels),
            load8(rows[4], x + 3 * channels),
            load8(rows[5], x - 3 * channels),
            load8(rows[5], x - 2 * channels),
            load8(rows[5], x - channels),
            load8(rows[5], x),
            load8(rows[5], x + channels),
            load8(rows[5], x + 2 * channels),
            load8(rows[5], x + 3 * channels),
            load8(rows[6], x - 3 * channels),
            load8(rows[6], x - 2 * channels),
            load8(rows[6], x - channels),
            load8(rows[6], x),
            load8(rows[6], x + channels),
            load8(rows[6], x + 2 * channels),
            load8(rows[6], x + 3 * channels),
        );
        unsafe {
            vst1_u8(
                dst.get_unchecked_mut(x..).as_mut_ptr().cast(),
                vget_low_u8(med.0),
            );
        }
        x += 8;
    }

    // Scalar tail
    while x < width.saturating_sub(3 * channels) {
        write_scalar_7x7(dst, rows, x, width, channels);
        x += channels;
    }

    // Right border: last 3 pixels
    for bx in 0..3 {
        let x = width - (3 - bx) * channels;
        write_scalar_7x7(dst, rows, x, width, channels);
    }
}

#[cfg(test)]
mod tests_7x7 {
    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct ScalarU8x16([u8; 16]);

    impl SimdU8 for ScalarU8x16 {
        fn min(a: Self, b: Self) -> Self {
            let mut r = [0u8; 16];
            for i in 0..16 {
                r[i] = a.0[i].min(b.0[i]);
            }
            ScalarU8x16(r)
        }
        fn max(a: Self, b: Self) -> Self {
            let mut r = [0u8; 16];
            for i in 0..16 {
                r[i] = a.0[i].max(b.0[i]);
            }
            ScalarU8x16(r)
        }
    }

    fn splat(v: u8) -> ScalarU8x16 {
        ScalarU8x16([v; 16])
    }

    fn reference_median_49(mut v: [u8; 49]) -> u8 {
        v.sort_unstable();
        v[24]
    }

    fn check_median_49(vals: [u8; 49]) {
        let expected = reference_median_49(vals);
        let result = median_network_7x7(
            splat(vals[0]),
            splat(vals[1]),
            splat(vals[2]),
            splat(vals[3]),
            splat(vals[4]),
            splat(vals[5]),
            splat(vals[6]),
            splat(vals[7]),
            splat(vals[8]),
            splat(vals[9]),
            splat(vals[10]),
            splat(vals[11]),
            splat(vals[12]),
            splat(vals[13]),
            splat(vals[14]),
            splat(vals[15]),
            splat(vals[16]),
            splat(vals[17]),
            splat(vals[18]),
            splat(vals[19]),
            splat(vals[20]),
            splat(vals[21]),
            splat(vals[22]),
            splat(vals[23]),
            splat(vals[24]),
            splat(vals[25]),
            splat(vals[26]),
            splat(vals[27]),
            splat(vals[28]),
            splat(vals[29]),
            splat(vals[30]),
            splat(vals[31]),
            splat(vals[32]),
            splat(vals[33]),
            splat(vals[34]),
            splat(vals[35]),
            splat(vals[36]),
            splat(vals[37]),
            splat(vals[38]),
            splat(vals[39]),
            splat(vals[40]),
            splat(vals[41]),
            splat(vals[42]),
            splat(vals[43]),
            splat(vals[44]),
            splat(vals[45]),
            splat(vals[46]),
            splat(vals[47]),
            splat(vals[48]),
        );
        assert_eq!(
            result.0[0], expected,
            "median_network_7x7({:?}) = {} but expected {}",
            vals, result.0[0], expected
        );
    }

    #[test]
    fn test_7x7_already_sorted() {
        let vals: [u8; 49] = core::array::from_fn(|i| i as u8);
        check_median_49(vals);
    }

    #[test]
    fn test_7x7_reverse_sorted() {
        let vals: [u8; 49] = core::array::from_fn(|i| (48 - i) as u8);
        check_median_49(vals);
    }

    #[test]
    fn test_7x7_all_same() {
        check_median_49([42u8; 49]);
    }

    #[test]
    fn test_7x7_all_zero() {
        check_median_49([0u8; 49]);
    }

    #[test]
    fn test_7x7_all_max() {
        check_median_49([255u8; 49]);
    }

    #[test]
    fn test_7x7_single_outlier_high() {
        let mut vals = [50u8; 49];
        vals[48] = 255;
        check_median_49(vals);
    }

    #[test]
    fn test_7x7_single_outlier_low() {
        let mut vals = [100u8; 49];
        vals[0] = 0;
        check_median_49(vals);
    }

    #[test]
    fn test_7x7_twenty_four_low_twenty_five_high() {
        let mut vals = [200u8; 49];
        for i in 0..24 {
            vals[i] = 10;
        }
        check_median_49(vals);
    }

    #[test]
    fn test_7x7_twenty_five_low_twenty_four_high() {
        let mut vals = [200u8; 49];
        for i in 0..25 {
            vals[i] = 10;
        }
        check_median_49(vals);
    }

    #[test]
    fn test_7x7_random_patterns() {
        let cases: &[[u8; 49]] = &[
            core::array::from_fn(|i| ((i * 37 + 13) % 256) as u8),
            core::array::from_fn(|i| ((i * 97 + 7) % 256) as u8),
            core::array::from_fn(|i| ((i * 13 + 200) % 256) as u8),
            core::array::from_fn(|i| (255u8).wrapping_sub((i * 5) as u8)),
            core::array::from_fn(|i| if i % 2 == 0 { 0 } else { 255 }),
            core::array::from_fn(|i| if i < 25 { 0 } else { 255 }),
        ];
        for &vals in cases {
            check_median_49(vals);
        }
    }

    #[test]
    fn test_7x7_all_lanes_agree() {
        let mut inputs: [[u8; 16]; 49] = [[0u8; 16]; 49];
        for lane in 0..16usize {
            for slot in 0..49usize {
                inputs[slot][lane] = ((lane * 7 + slot * 11) % 256) as u8;
            }
        }
        let result = median_network_7x7(
            ScalarU8x16(inputs[0]),
            ScalarU8x16(inputs[1]),
            ScalarU8x16(inputs[2]),
            ScalarU8x16(inputs[3]),
            ScalarU8x16(inputs[4]),
            ScalarU8x16(inputs[5]),
            ScalarU8x16(inputs[6]),
            ScalarU8x16(inputs[7]),
            ScalarU8x16(inputs[8]),
            ScalarU8x16(inputs[9]),
            ScalarU8x16(inputs[10]),
            ScalarU8x16(inputs[11]),
            ScalarU8x16(inputs[12]),
            ScalarU8x16(inputs[13]),
            ScalarU8x16(inputs[14]),
            ScalarU8x16(inputs[15]),
            ScalarU8x16(inputs[16]),
            ScalarU8x16(inputs[17]),
            ScalarU8x16(inputs[18]),
            ScalarU8x16(inputs[19]),
            ScalarU8x16(inputs[20]),
            ScalarU8x16(inputs[21]),
            ScalarU8x16(inputs[22]),
            ScalarU8x16(inputs[23]),
            ScalarU8x16(inputs[24]),
            ScalarU8x16(inputs[25]),
            ScalarU8x16(inputs[26]),
            ScalarU8x16(inputs[27]),
            ScalarU8x16(inputs[28]),
            ScalarU8x16(inputs[29]),
            ScalarU8x16(inputs[30]),
            ScalarU8x16(inputs[31]),
            ScalarU8x16(inputs[32]),
            ScalarU8x16(inputs[33]),
            ScalarU8x16(inputs[34]),
            ScalarU8x16(inputs[35]),
            ScalarU8x16(inputs[36]),
            ScalarU8x16(inputs[37]),
            ScalarU8x16(inputs[38]),
            ScalarU8x16(inputs[39]),
            ScalarU8x16(inputs[40]),
            ScalarU8x16(inputs[41]),
            ScalarU8x16(inputs[42]),
            ScalarU8x16(inputs[43]),
            ScalarU8x16(inputs[44]),
            ScalarU8x16(inputs[45]),
            ScalarU8x16(inputs[46]),
            ScalarU8x16(inputs[47]),
            ScalarU8x16(inputs[48]),
        );
        for lane in 0..16usize {
            let vals: [u8; 49] = core::array::from_fn(|s| inputs[s][lane]);
            let expected = reference_median_49(vals);
            assert_eq!(
                result.0[lane], expected,
                "lane {lane}: got {} expected {expected} for {vals:?}",
                result.0[lane]
            );
        }
    }

    #[test]
    fn test_7x7_median_bounds() {
        let vals: [u8; 49] = core::array::from_fn(|i| (i * 5) as u8);
        let mut sorted = vals;
        sorted.sort_unstable();
        let result = median_network_7x7(
            splat(vals[0]),
            splat(vals[1]),
            splat(vals[2]),
            splat(vals[3]),
            splat(vals[4]),
            splat(vals[5]),
            splat(vals[6]),
            splat(vals[7]),
            splat(vals[8]),
            splat(vals[9]),
            splat(vals[10]),
            splat(vals[11]),
            splat(vals[12]),
            splat(vals[13]),
            splat(vals[14]),
            splat(vals[15]),
            splat(vals[16]),
            splat(vals[17]),
            splat(vals[18]),
            splat(vals[19]),
            splat(vals[20]),
            splat(vals[21]),
            splat(vals[22]),
            splat(vals[23]),
            splat(vals[24]),
            splat(vals[25]),
            splat(vals[26]),
            splat(vals[27]),
            splat(vals[28]),
            splat(vals[29]),
            splat(vals[30]),
            splat(vals[31]),
            splat(vals[32]),
            splat(vals[33]),
            splat(vals[34]),
            splat(vals[35]),
            splat(vals[36]),
            splat(vals[37]),
            splat(vals[38]),
            splat(vals[39]),
            splat(vals[40]),
            splat(vals[41]),
            splat(vals[42]),
            splat(vals[43]),
            splat(vals[44]),
            splat(vals[45]),
            splat(vals[46]),
            splat(vals[47]),
            splat(vals[48]),
        )
        .0[0];
        assert!(
            result >= sorted[23] && result <= sorted[25],
            "median {result} out of expected range [{}, {}]",
            sorted[23],
            sorted[25]
        );
    }
}
