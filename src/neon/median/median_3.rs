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

use crate::{BlurImage, BlurImageMut, ThreadingPolicy};
use novtb::{ParallelZonedIterator, TbSliceMut};
use std::arch::aarch64::*;

pub(crate) trait SimdU8: Copy {
    fn min(a: Self, b: Self) -> Self;
    fn max(a: Self, b: Self) -> Self;
}

fn load_scalar_3x3<const CN: usize>(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    x: usize,
    width: usize,
) -> [u8; CN] {
    let x_left = x.saturating_sub(CN);
    let x_right = (x + CN).min(width - CN);
    let mut result = [0u8; CN];
    for c in 0..CN {
        unsafe {
            let mut v = [
                *r0.get_unchecked(x_left + c),
                *r0.get_unchecked(x + c),
                *r0.get_unchecked(x_right + c),
                *r1.get_unchecked(x_left + c),
                *r1.get_unchecked(x + c),
                *r1.get_unchecked(x_right + c),
                *r2.get_unchecked(x_left + c),
                *r2.get_unchecked(x + c),
                *r2.get_unchecked(x_right + c),
            ];
            macro_rules! s {
                ($a:expr, $b:expr) => {
                    if v[$a] > v[$b] {
                        v.swap($a, $b);
                    }
                };
            }

            s!(0, 1);
            s!(3, 4);
            s!(6, 7);
            s!(1, 2);
            s!(4, 5);
            s!(7, 8);
            s!(0, 1);
            s!(3, 4);
            s!(6, 7);
            s!(0, 3);
            s!(3, 6);
            s!(0, 3);
            s!(1, 4);
            s!(4, 7);
            s!(1, 4);
            s!(2, 5);
            s!(5, 8);
            s!(2, 5);
            s!(2, 6);
            s!(2, 4);
            s!(4, 6);
            s!(0, 4);
            s!(4, 8);
            s!(3, 5);
            s!(1, 4);
            s!(4, 5);
            s!(3, 4);

            result[c] = v[4];
        }
    }
    result
}

#[derive(Clone, Copy)]
pub(crate) struct NeonU8(pub(crate) uint8x16_t);

impl SimdU8 for NeonU8 {
    #[inline(always)]
    fn min(a: Self, b: Self) -> Self {
        unsafe { NeonU8(vminq_u8(a.0, b.0)) }
    }
    #[inline(always)]
    fn max(a: Self, b: Self) -> Self {
        unsafe { NeonU8(vmaxq_u8(a.0, b.0)) }
    }
}

#[allow(unused)]
#[inline(always)]
fn median_network<S: SimdU8>(
    mut p0: S,
    mut p1: S,
    mut p2: S,
    mut p3: S,
    mut p4: S,
    mut p5: S,
    mut p6: S,
    mut p7: S,
    mut p8: S,
) -> S {
    macro_rules! coex {
        ($a:ident, $b:ident) => {{
            let lo = S::min($a, $b);
            let hi = S::max($a, $b);
            $a = lo;
            $b = hi;
        }};
    }

    // Sort each row of 3
    coex!(p0, p1);
    coex!(p3, p4);
    coex!(p6, p7);
    coex!(p1, p2);
    coex!(p4, p5);
    coex!(p7, p8);
    coex!(p0, p1);
    coex!(p3, p4);
    coex!(p6, p7);

    // Sort each column - produces sorted columns
    coex!(p0, p3);
    coex!(p3, p6);
    coex!(p0, p3);
    coex!(p1, p4);
    coex!(p4, p7);
    coex!(p1, p4);
    coex!(p2, p5);
    coex!(p5, p8);
    coex!(p2, p5);

    // Extract median into p4
    coex!(p2, p6);
    coex!(p2, p4);
    coex!(p4, p6);
    coex!(p0, p4);
    coex!(p4, p8);
    coex!(p3, p5);
    coex!(p1, p4);
    coex!(p4, p5);
    coex!(p3, p4);

    p4
}

pub(crate) fn median_blur_3x3(
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
            let r0 = &src.data.as_ref()[y.saturating_sub(1) * src_stride..];
            let r1 = &src.data.as_ref()[y * src_stride..];
            let r2 = &src.data.as_ref()[(y + 1).min(height - 1) * src_stride..];
            unsafe {
                median_blur_3x3_impl_16(row, r0, r1, r2, src.width as usize * channels, channels);
            }
        });
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn load(row: &[u8], col: usize) -> NeonU8 {
    unsafe { NeonU8(vld1q_u8(row.get_unchecked(col..).as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn load8(row: &[u8], col: usize) -> NeonU8 {
    unsafe {
        NeonU8(vcombine_u8(
            vld1_u8(row.get_unchecked(col..).as_ptr().cast()),
            vdup_n_u8(0),
        ))
    }
}

#[target_feature(enable = "neon")]
fn median_blur_3x3_impl_16(
    dst: &mut [u8],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    width: usize,
    channels: usize,
) {
    match channels {
        1 => {
            let px = load_scalar_3x3::<1>(r0, r1, r2, 0, width);
            dst[0] = px[0];
        }
        3 => {
            let px = load_scalar_3x3::<3>(r0, r1, r2, 0, width);
            dst[0] = px[0];
            dst[1] = px[1];
            dst[2] = px[2];
        }
        4 => {
            let px = load_scalar_3x3::<4>(r0, r1, r2, 0, width);
            dst[0] = px[0];
            dst[1] = px[1];
            dst[2] = px[2];
            dst[3] = px[3];
        }
        _ => unreachable!(),
    }

    let mut x = channels;
    while x + 16 <= width - channels {
        let med = median_network(
            load(r0, x - channels),
            load(r0, x),
            load(r0, x + channels),
            load(r1, x - channels),
            load(r1, x),
            load(r1, x + channels),
            load(r2, x - channels),
            load(r2, x),
            load(r2, x + channels),
        );
        unsafe {
            vst1q_u8(dst.get_unchecked_mut(x..).as_mut_ptr().cast(), med.0);
        }
        x += 16;
    }

    while x + 8 <= width - channels {
        let med = median_network(
            load8(r0, x - channels),
            load8(r0, x),
            load8(r0, x + channels),
            load8(r1, x - channels),
            load8(r1, x),
            load8(r1, x + channels),
            load8(r2, x - channels),
            load8(r2, x),
            load8(r2, x + channels),
        );
        unsafe {
            vst1_u8(
                dst.get_unchecked_mut(x..).as_mut_ptr().cast(),
                vget_low_u8(med.0),
            );
        }
        x += 8;
    }

    while x < width - channels {
        match channels {
            1 => {
                let px = load_scalar_3x3::<1>(r0, r1, r2, x, width);
                dst[x] = px[0];
            }
            3 => {
                let px = load_scalar_3x3::<3>(r0, r1, r2, x, width);
                dst[x] = px[0];
                dst[x + 1] = px[1];
                dst[x + 2] = px[2];
            }
            4 => {
                let px = load_scalar_3x3::<4>(r0, r1, r2, x, width);
                dst[x] = px[0];
                dst[x + 1] = px[1];
                dst[x + 2] = px[2];
                dst[x + 3] = px[3];
            }
            _ => unreachable!(),
        }
        x += channels;
    }

    let x = width - channels;
    match channels {
        1 => {
            let px = load_scalar_3x3::<1>(r0, r1, r2, x, width);
            dst[x] = px[0];
        }
        3 => {
            let px = load_scalar_3x3::<3>(r0, r1, r2, x, width);
            dst[x] = px[0];
            dst[x + 1] = px[1];
            dst[x + 2] = px[2];
        }
        4 => {
            let px = load_scalar_3x3::<4>(r0, r1, r2, x, width);
            dst[x] = px[0];
            dst[x + 1] = px[1];
            dst[x + 2] = px[2];
            dst[x + 3] = px[3];
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Scalar wrapper so we can test the network with plain u8 arrays
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

    fn from_lane(vals: [u8; 16]) -> ScalarU8x16 {
        ScalarU8x16(vals)
    }

    // Brute-force median of 9 values
    fn reference_median(mut v: [u8; 9]) -> u8 {
        v.sort_unstable();
        v[4]
    }

    // Run median_network on 9 splat values and check against reference
    fn check_median(vals: [u8; 9]) {
        let expected = reference_median(vals);
        let result = median_network(
            splat(vals[0]),
            splat(vals[1]),
            splat(vals[2]),
            splat(vals[3]),
            splat(vals[4]),
            splat(vals[5]),
            splat(vals[6]),
            splat(vals[7]),
            splat(vals[8]),
        );
        assert_eq!(
            result.0[0], expected,
            "median_network({:?}) = {} but expected {}",
            vals, result.0[0], expected
        );
    }

    // -----------------------------------------------------------------------
    // Correctness: all lanes agree
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_lanes_agree() {
        // Each lane gets a different 9-element set; verify every lane independently
        let mut inputs: [[u8; 16]; 9] = [[0u8; 16]; 9];
        // lane k gets values (k, k+1, ..., k+8) mod 256
        for lane in 0..16usize {
            for slot in 0..9usize {
                inputs[slot][lane] = ((lane + slot * 17) % 256) as u8;
            }
        }
        let result = median_network(
            from_lane(inputs[0]),
            from_lane(inputs[1]),
            from_lane(inputs[2]),
            from_lane(inputs[3]),
            from_lane(inputs[4]),
            from_lane(inputs[5]),
            from_lane(inputs[6]),
            from_lane(inputs[7]),
            from_lane(inputs[8]),
        );
        for lane in 0..16usize {
            let vals = [
                inputs[0][lane],
                inputs[1][lane],
                inputs[2][lane],
                inputs[3][lane],
                inputs[4][lane],
                inputs[5][lane],
                inputs[6][lane],
                inputs[7][lane],
                inputs[8][lane],
            ];
            let expected = reference_median(vals);
            assert_eq!(
                result.0[lane], expected,
                "lane {lane}: median_network({vals:?}) = {} but expected {expected}",
                result.0[lane]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Basic cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_already_sorted() {
        check_median([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_reverse_sorted() {
        check_median([9, 8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_all_same() {
        check_median([7, 7, 7, 7, 7, 7, 7, 7, 7]);
    }

    #[test]
    fn test_all_zero() {
        check_median([0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_all_max() {
        check_median([255, 255, 255, 255, 255, 255, 255, 255, 255]);
    }

    // -----------------------------------------------------------------------
    // Edge values
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_max_boundaries() {
        check_median([0, 255, 0, 255, 128, 0, 255, 0, 255]);
    }

    #[test]
    fn test_single_outlier_high() {
        // One very high value should not affect median of otherwise uniform set
        check_median([5, 5, 5, 5, 5, 5, 5, 5, 255]);
    }

    #[test]
    fn test_single_outlier_low() {
        check_median([0, 100, 100, 100, 100, 100, 100, 100, 100]);
    }

    #[test]
    fn test_two_outliers_both_ends() {
        check_median([0, 0, 100, 100, 100, 100, 100, 255, 255]);
    }

    // -----------------------------------------------------------------------
    // Duplicates
    // -----------------------------------------------------------------------

    #[test]
    fn test_five_identical_median() {
        // Median must be the repeated value
        check_median([1, 2, 3, 5, 5, 5, 5, 5, 9]);
    }

    #[test]
    fn test_four_low_five_high() {
        check_median([1, 1, 1, 1, 200, 200, 200, 200, 200]);
    }

    #[test]
    fn test_four_high_five_low() {
        check_median([10, 10, 10, 10, 10, 200, 200, 200, 200]);
    }

    #[test]
    fn test_alternating() {
        check_median([0, 255, 0, 255, 0, 255, 0, 255, 128]);
    }

    // -----------------------------------------------------------------------
    // Positional: median must be the 5th-smallest regardless of input order
    // -----------------------------------------------------------------------

    #[test]
    fn test_median_is_position_4() {
        // Values where sorted order is unambiguous
        check_median([10, 20, 30, 40, 50, 60, 70, 80, 90]);
        check_median([90, 80, 70, 60, 50, 40, 30, 20, 10]);
        check_median([50, 10, 90, 30, 70, 20, 80, 40, 60]);
    }

    #[test]
    fn test_median_with_plateau_around_median() {
        // Three equal values straddle the median position
        check_median([10, 20, 50, 50, 50, 60, 70, 80, 90]);
    }

    // -----------------------------------------------------------------------
    // Exhaustive small-range sweep
    // -----------------------------------------------------------------------

    #[test]
    fn test_exhaustive_values_0_to_8() {
        // All permutations of [0..=8] — 9! = 362880, reasonable to run
        let base: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut perm = base;
        // Heap's algorithm
        let mut c = [0usize; 9];
        check_median(perm);
        let mut i = 0;
        while i < 9 {
            if c[i] < i {
                if i % 2 == 0 {
                    perm.swap(0, i);
                } else {
                    perm.swap(c[i], i);
                }
                check_median(perm);
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Property: output is always one of the 9 input values
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_is_one_of_inputs() {
        let cases: &[[u8; 9]] = &[
            [3, 1, 4, 1, 5, 9, 2, 6, 5],
            [255, 0, 128, 64, 192, 32, 96, 160, 224],
            [17, 17, 17, 42, 42, 42, 99, 99, 0],
        ];
        for &vals in cases {
            let result = median_network(
                splat(vals[0]),
                splat(vals[1]),
                splat(vals[2]),
                splat(vals[3]),
                splat(vals[4]),
                splat(vals[5]),
                splat(vals[6]),
                splat(vals[7]),
                splat(vals[8]),
            )
            .0[0];
            assert!(
                vals.contains(&result),
                "result {result} is not one of the inputs {vals:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property: output is >= the 4 smallest and <= the 4 largest
    // -----------------------------------------------------------------------

    #[test]
    fn test_median_bounds() {
        let cases: &[[u8; 9]] = &[
            [10, 20, 30, 40, 50, 60, 70, 80, 90],
            [0, 1, 2, 3, 127, 252, 253, 254, 255],
            [50, 50, 50, 50, 50, 50, 50, 50, 50],
        ];
        for &vals in cases {
            let mut sorted = vals;
            sorted.sort_unstable();
            let result = median_network(
                splat(vals[0]),
                splat(vals[1]),
                splat(vals[2]),
                splat(vals[3]),
                splat(vals[4]),
                splat(vals[5]),
                splat(vals[6]),
                splat(vals[7]),
                splat(vals[8]),
            )
            .0[0];
            assert!(
                result >= sorted[3] && result <= sorted[5],
                "median {result} out of bounds [{}, {}] for {vals:?}",
                sorted[3],
                sorted[5]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Regression: the specific bug (coex_max argument order)
    // -----------------------------------------------------------------------

    #[test]
    fn test_regression_coex_max_arg_order() {
        // This set was chosen to exercise the p4/p6 path specifically:
        // after the column sorts, p2=2, p4=5, p6=8 approximately.
        // The buggy version would leave p4 unmodified by coex_max!(p4,p6).
        check_median([8, 1, 6, 3, 5, 7, 4, 9, 2]);
        check_median([2, 9, 4, 7, 5, 3, 6, 1, 8]);
        // High value in position that flows through p6 into the buggy comparison
        check_median([0, 0, 200, 0, 100, 0, 255, 0, 50]);
        check_median([255, 0, 0, 0, 128, 255, 0, 255, 0]);
    }
}
