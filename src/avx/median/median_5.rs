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
#![allow(clippy::needless_range_loop)]

use crate::avx::median::median_3::{SimdU8, load8, load16, load32};
use crate::{BlurImage, BlurImageMut, ThreadingPolicy};
use novtb::{ParallelZonedIterator, TbSliceMut};
use std::arch::x86_64::{
    _mm_storeu_si64, _mm_storeu_si128, _mm256_castsi256_si128, _mm256_storeu_si256,
};

fn load_scalar_5x5<const CN: usize>(rows: [&[u8]; 5], x: usize, width: usize) -> [u8; CN] {
    let x0 = x.saturating_sub(2 * CN);
    let x1 = x.saturating_sub(CN);
    let x2 = x;
    let x3 = (x + CN).min(width - CN);
    let x4 = (x + 2 * CN).min(width - CN);
    let mut result = [0u8; CN];
    for c in 0..CN {
        unsafe {
            let mut v = [
                *rows[0].get_unchecked(x0 + c),
                *rows[0].get_unchecked(x1 + c),
                *rows[0].get_unchecked(x2 + c),
                *rows[0].get_unchecked(x3 + c),
                *rows[0].get_unchecked(x4 + c),
                *rows[1].get_unchecked(x0 + c),
                *rows[1].get_unchecked(x1 + c),
                *rows[1].get_unchecked(x2 + c),
                *rows[1].get_unchecked(x3 + c),
                *rows[1].get_unchecked(x4 + c),
                *rows[2].get_unchecked(x0 + c),
                *rows[2].get_unchecked(x1 + c),
                *rows[2].get_unchecked(x2 + c),
                *rows[2].get_unchecked(x3 + c),
                *rows[2].get_unchecked(x4 + c),
                *rows[3].get_unchecked(x0 + c),
                *rows[3].get_unchecked(x1 + c),
                *rows[3].get_unchecked(x2 + c),
                *rows[3].get_unchecked(x3 + c),
                *rows[3].get_unchecked(x4 + c),
                *rows[4].get_unchecked(x0 + c),
                *rows[4].get_unchecked(x1 + c),
                *rows[4].get_unchecked(x2 + c),
                *rows[4].get_unchecked(x3 + c),
                *rows[4].get_unchecked(x4 + c),
            ];
            v.sort_unstable();
            result[c] = v[12]; // median of 25
        }
    }
    result
}

#[allow(unused)]
#[inline(always)]
fn median_network_5x5<S: SimdU8>(
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
) -> S {
    let mut v = [
        p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17,
        p18, p19, p20, p21, p22, p23, p24,
    ];

    for i in 1..25 {
        let mut j = i;
        while j > 0 {
            let lo = S::min(v[j - 1], v[j]);
            let hi = S::max(v[j - 1], v[j]);
            v[j - 1] = lo;
            v[j] = hi;
            j -= 1;
        }
    }

    v[12]
}

pub(crate) fn avx_median_blur_5x5(
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
                &src.data.as_ref()[y.saturating_sub(2) * src_stride..],
                &src.data.as_ref()[y.saturating_sub(1) * src_stride..],
                &src.data.as_ref()[y * src_stride..],
                &src.data.as_ref()[(y + 1).min(height - 1) * src_stride..],
                &src.data.as_ref()[(y + 2).min(height - 1) * src_stride..],
            ];
            unsafe {
                median_blur_5x5_impl(row, rows, src.width as usize * channels, channels);
            }
        });
}

#[target_feature(enable = "avx2")]
fn median_blur_5x5_impl(dst: &mut [u8], rows: [&[u8]; 5], width: usize, channels: usize) {
    // Left border: 2 pixels need clamped left neighbors
    for bx in 0..2 {
        let x = bx * channels;
        write_scalar_5x5(dst, rows, x, width, channels);
    }

    let mut x = 2 * channels;

    // SIMD 16-wide loop — safe as long as x + 2*channels + 16 <= width
    while x + 2 * channels + 32 <= width {
        let med = median_network_5x5(
            load32(rows[0], x - 2 * channels),
            load32(rows[0], x - channels),
            load32(rows[0], x),
            load32(rows[0], x + channels),
            load32(rows[0], x + 2 * channels),
            load32(rows[1], x - 2 * channels),
            load32(rows[1], x - channels),
            load32(rows[1], x),
            load32(rows[1], x + channels),
            load32(rows[1], x + 2 * channels),
            load32(rows[2], x - 2 * channels),
            load32(rows[2], x - channels),
            load32(rows[2], x),
            load32(rows[2], x + channels),
            load32(rows[2], x + 2 * channels),
            load32(rows[3], x - 2 * channels),
            load32(rows[3], x - channels),
            load32(rows[3], x),
            load32(rows[3], x + channels),
            load32(rows[3], x + 2 * channels),
            load32(rows[4], x - 2 * channels),
            load32(rows[4], x - channels),
            load32(rows[4], x),
            load32(rows[4], x + channels),
            load32(rows[4], x + 2 * channels),
        );
        unsafe {
            _mm256_storeu_si256(dst.get_unchecked_mut(x..).as_mut_ptr().cast(), med.0);
        }
        x += 32;
    }

    // SIMD 16-wide loop — safe as long as x + 2*channels + 16 <= width
    while x + 2 * channels + 16 <= width {
        let med = median_network_5x5(
            load16(rows[0], x - 2 * channels),
            load16(rows[0], x - channels),
            load16(rows[0], x),
            load16(rows[0], x + channels),
            load16(rows[0], x + 2 * channels),
            load16(rows[1], x - 2 * channels),
            load16(rows[1], x - channels),
            load16(rows[1], x),
            load16(rows[1], x + channels),
            load16(rows[1], x + 2 * channels),
            load16(rows[2], x - 2 * channels),
            load16(rows[2], x - channels),
            load16(rows[2], x),
            load16(rows[2], x + channels),
            load16(rows[2], x + 2 * channels),
            load16(rows[3], x - 2 * channels),
            load16(rows[3], x - channels),
            load16(rows[3], x),
            load16(rows[3], x + channels),
            load16(rows[3], x + 2 * channels),
            load16(rows[4], x - 2 * channels),
            load16(rows[4], x - channels),
            load16(rows[4], x),
            load16(rows[4], x + channels),
            load16(rows[4], x + 2 * channels),
        );
        unsafe {
            _mm_storeu_si128(
                dst.get_unchecked_mut(x..).as_mut_ptr().cast(),
                _mm256_castsi256_si128(med.0),
            );
        }
        x += 16;
    }

    // SIMD 8-wide loop
    while x + 2 * channels + 8 <= width {
        let med = median_network_5x5(
            load8(rows[0], x - 2 * channels),
            load8(rows[0], x - channels),
            load8(rows[0], x),
            load8(rows[0], x + channels),
            load8(rows[0], x + 2 * channels),
            load8(rows[1], x - 2 * channels),
            load8(rows[1], x - channels),
            load8(rows[1], x),
            load8(rows[1], x + channels),
            load8(rows[1], x + 2 * channels),
            load8(rows[2], x - 2 * channels),
            load8(rows[2], x - channels),
            load8(rows[2], x),
            load8(rows[2], x + channels),
            load8(rows[2], x + 2 * channels),
            load8(rows[3], x - 2 * channels),
            load8(rows[3], x - channels),
            load8(rows[3], x),
            load8(rows[3], x + channels),
            load8(rows[3], x + 2 * channels),
            load8(rows[4], x - 2 * channels),
            load8(rows[4], x - channels),
            load8(rows[4], x),
            load8(rows[4], x + channels),
            load8(rows[4], x + 2 * channels),
        );
        unsafe {
            _mm_storeu_si64(
                dst.get_unchecked_mut(x..).as_mut_ptr().cast(),
                _mm256_castsi256_si128(med.0),
            );
        }
        x += 8;
    }

    // Scalar tail (interior pixels that didn't fit in SIMD)
    while x < width.saturating_sub(2 * channels) {
        write_scalar_5x5(dst, rows, x, width, channels);
        x += channels;
    }

    // Right border: last 2 pixels need clamped right neighbors
    for bx in 0..2 {
        let x = width - (2 - bx) * channels;
        write_scalar_5x5(dst, rows, x, width, channels);
    }
}

#[inline(always)]
fn write_scalar_5x5(dst: &mut [u8], rows: [&[u8]; 5], x: usize, width: usize, channels: usize) {
    match channels {
        1 => {
            let px = load_scalar_5x5::<1>(rows, x, width);
            dst[x] = px[0];
        }
        3 => {
            let px = load_scalar_5x5::<3>(rows, x, width);
            dst[x] = px[0];
            dst[x + 1] = px[1];
            dst[x + 2] = px[2];
        }
        4 => {
            let px = load_scalar_5x5::<4>(rows, x, width);
            dst[x] = px[0];
            dst[x + 1] = px[1];
            dst[x + 2] = px[2];
            dst[x + 3] = px[3];
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests_5x5 {
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

    fn reference_median_25(mut v: [u8; 25]) -> u8 {
        v.sort_unstable();
        v[12]
    }

    fn check_median_25(vals: [u8; 25]) {
        let expected = reference_median_25(vals);
        let result = median_network_5x5(
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
        );
        assert_eq!(
            result.0[0], expected,
            "median_network_5x5({:?}) = {} but expected {}",
            vals, result.0[0], expected
        );
    }

    #[test]
    fn test_5x5_already_sorted() {
        let vals: [u8; 25] = core::array::from_fn(|i| i as u8);
        check_median_25(vals);
    }

    #[test]
    fn test_5x5_reverse_sorted() {
        let vals: [u8; 25] = core::array::from_fn(|i| (24 - i) as u8);
        check_median_25(vals);
    }

    #[test]
    fn test_5x5_all_same() {
        check_median_25([42u8; 25]);
    }

    #[test]
    fn test_5x5_all_zero() {
        check_median_25([0u8; 25]);
    }

    #[test]
    fn test_5x5_all_max() {
        check_median_25([255u8; 25]);
    }

    #[test]
    fn test_5x5_single_outlier_high() {
        let mut vals = [50u8; 25];
        vals[24] = 255;
        check_median_25(vals);
    }

    #[test]
    fn test_5x5_single_outlier_low() {
        let mut vals = [100u8; 25];
        vals[0] = 0;
        check_median_25(vals);
    }

    #[test]
    fn test_5x5_twelve_low_thirteen_high() {
        let mut vals = [200u8; 25];
        for i in 0..12 {
            vals[i] = 10;
        }
        check_median_25(vals);
    }

    #[test]
    fn test_5x5_thirteen_low_twelve_high() {
        let mut vals = [200u8; 25];
        for i in 0..13 {
            vals[i] = 10;
        }
        check_median_25(vals);
    }

    #[test]
    fn test_5x5_random_patterns() {
        let cases: &[[u8; 25]] = &[
            core::array::from_fn(|i| ((i * 37 + 13) % 256) as u8),
            core::array::from_fn(|i| ((i * 97 + 7) % 256) as u8),
            core::array::from_fn(|i| ((i * 13 + 200) % 256) as u8),
            core::array::from_fn(|i| (255 - i * 10) as u8),
        ];
        for &vals in cases {
            check_median_25(vals);
        }
    }

    #[test]
    fn test_5x5_all_lanes_agree() {
        let mut inputs: [[u8; 16]; 25] = [[0u8; 16]; 25];
        for lane in 0..16usize {
            for slot in 0..25usize {
                inputs[slot][lane] = ((lane * 7 + slot * 11) % 256) as u8;
            }
        }
        let result = median_network_5x5(
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
        );
        for lane in 0..16usize {
            let vals: [u8; 25] = core::array::from_fn(|s| inputs[s][lane]);
            let expected = reference_median_25(vals);
            assert_eq!(
                result.0[lane], expected,
                "lane {lane}: got {} expected {expected} for {vals:?}",
                result.0[lane]
            );
        }
    }

    #[test]
    fn test_5x5_median_bounds() {
        let vals: [u8; 25] = core::array::from_fn(|i| (i * 10) as u8);
        let mut sorted = vals;
        sorted.sort_unstable();
        let result = median_network_5x5(
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
        )
        .0[0];
        assert!(
            result >= sorted[11] && result <= sorted[13],
            "median {result} out of expected range [{}, {}]",
            sorted[11],
            sorted[13]
        );
    }
}
