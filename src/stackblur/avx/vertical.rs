/*
 * // Copyright (c) Radzivon Bartoshyk 02/2025. All rights reserved.
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
use crate::sse::{load_u8_s32_fast, store_u8_s32};
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::{AsPrimitive, FromPrimitive};
use std::arch::x86_64::*;
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Shr, Sub, SubAssign};

pub(crate) struct VerticalAvxStackBlurPass<T, J, const COMPONENTS: usize> {
    _phantom_t: PhantomData<T>,
    _phantom_j: PhantomData<J>,
}

impl<T, J, const COMPONENTS: usize> Default for VerticalAvxStackBlurPass<T, J, COMPONENTS> {
    fn default() -> Self {
        VerticalAvxStackBlurPass::<T, J, COMPONENTS> {
            _phantom_t: Default::default(),
            _phantom_j: Default::default(),
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn stack_blur_avx_vertical_def<const CN: usize>(
    pixels: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    thread: usize,
    total_threads: usize,
) {
    stack_blur_avx_vertical::<CN, false>(
        pixels,
        stride,
        width,
        height,
        radius,
        thread,
        total_threads,
    );
}

//TODO: Re-enable when AVX512 is stabilized
// #[cfg(feature = "nightly_avx512")]
// #[target_feature(enable = "avx2", enable = "avxvnni")]
// unsafe fn stack_blur_avx_vertical_fma_vnni<const CN: usize>(
//     pixels: &UnsafeSlice<u8>,
//     stride: u32,
//     width: u32,
//     height: u32,
//     radius: u32,
//     thread: usize,
//     total_threads: usize,
// ) {
//     stack_blur_avx_vertical::<CN, true>(
//         pixels,
//         stride,
//         width,
//         height,
//         radius,
//         thread,
//         total_threads,
//     );
// }

#[inline(always)]
unsafe fn stack_blur_avx_vertical<const CN: usize, const VNNI: bool>(
    pixels: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    thread: usize,
    total_threads: usize,
) {
    let div = ((radius * 2) + 1) as usize;
    let mut yp;
    let mut sp;
    let mut stack_start;
    let mut stacks = vec![0i32; 4 * div * 8];

    let hm = height - 1;
    let div = (radius * 2) + 1;

    let mut src_ptr;
    let mut dst_ptr;

    let min_x = (thread * width as usize / total_threads) * CN;
    let max_x = ((thread + 1) * width as usize / total_threads) * CN;

    let mut cx = min_x;

    let v_mul_value = _mm256_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

    while cx + 32 < max_x {
        let mut sums0 = _mm256_setzero_si256();
        let mut sums1 = _mm256_setzero_si256();
        let mut sums2 = _mm256_setzero_si256();
        let mut sums3 = _mm256_setzero_si256();

        let mut sum_in0 = _mm256_setzero_si256();
        let mut sum_in1 = _mm256_setzero_si256();
        let mut sum_in2 = _mm256_setzero_si256();
        let mut sum_in3 = _mm256_setzero_si256();

        let mut sum_out0 = _mm256_setzero_si256();
        let mut sum_out1 = _mm256_setzero_si256();
        let mut sum_out2 = _mm256_setzero_si256();
        let mut sum_out3 = _mm256_setzero_si256();

        let mut src_ptr = cx; // x,0

        {
            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
            let src_pixel0 = _mm256_loadu_si256(src_ld as *const _);
            let lo0 = _mm256_unpacklo_epi8(src_pixel0, _mm256_setzero_si256());
            let hi1 = _mm256_unpackhi_epi8(src_pixel0, _mm256_setzero_si256());

            let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
            let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
            let i16_l1 = _mm256_unpacklo_epi16(hi1, _mm256_setzero_si256());
            let i16_h1 = _mm256_unpackhi_epi16(hi1, _mm256_setzero_si256());

            for i in 0..=radius {
                let stack_ptr = stacks.as_mut_ptr().add(i as usize * 4 * 8);

                _mm256_storeu_si256(stack_ptr as *mut _, i16_l0);
                _mm256_storeu_si256(stack_ptr.add(8) as *mut _, i16_h0);
                _mm256_storeu_si256(stack_ptr.add(16) as *mut _, i16_l1);
                _mm256_storeu_si256(stack_ptr.add(24) as *mut _, i16_h1);

                let w = _mm256_set1_epi32(i as i32 + 1);

                if VNNI {
                    #[cfg(feature = "nightly_avx512")]
                    {
                        sums0 = _mm256_dpwssd_avx_epi32(sums0, i16_l0, w);
                        sums1 = _mm256_dpwssd_avx_epi32(sums1, i16_h0, w);
                        sums2 = _mm256_dpwssd_avx_epi32(sums2, i16_l1, w);
                        sums3 = _mm256_dpwssd_avx_epi32(sums3, i16_h1, w);
                    }
                } else {
                    let j0 = _mm256_madd_epi16(i16_l0, w);
                    let j1 = _mm256_madd_epi16(i16_h0, w);
                    let j2 = _mm256_madd_epi16(i16_l1, w);
                    let j3 = _mm256_madd_epi16(i16_h1, w);

                    sums0 = _mm256_add_epi32(sums0, j0);
                    sums1 = _mm256_add_epi32(sums1, j1);
                    sums2 = _mm256_add_epi32(sums2, j2);
                    sums3 = _mm256_add_epi32(sums3, j3);
                }

                sum_out0 = _mm256_add_epi32(sum_out0, i16_l0);
                sum_out1 = _mm256_add_epi32(sum_out1, i16_h0);
                sum_out2 = _mm256_add_epi32(sum_out2, i16_l1);
                sum_out3 = _mm256_add_epi32(sum_out3, i16_h1);
            }
        }

        {
            for i in 1..=radius {
                if i <= hm {
                    src_ptr += stride as usize;
                }

                let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4 * 8);
                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                let src_pixel0 = _mm256_loadu_si256(src_ld as *const _);
                let lo0 = _mm256_unpacklo_epi8(src_pixel0, _mm256_setzero_si256());
                let hi1 = _mm256_unpackhi_epi8(src_pixel0, _mm256_setzero_si256());

                let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
                let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
                let i16_l1 = _mm256_unpacklo_epi16(hi1, _mm256_setzero_si256());
                let i16_h1 = _mm256_unpackhi_epi16(hi1, _mm256_setzero_si256());

                _mm256_storeu_si256(stack_ptr as *mut _, i16_l0);
                _mm256_storeu_si256(stack_ptr.add(8) as *mut _, i16_h0);
                _mm256_storeu_si256(stack_ptr.add(16) as *mut _, i16_l1);
                _mm256_storeu_si256(stack_ptr.add(24) as *mut _, i16_h1);

                let w = _mm256_set1_epi32(radius as i32 + 1 - i as i32);

                if VNNI {
                    #[cfg(feature = "nightly_avx512")]
                    {
                        sums0 = _mm256_dpwssd_avx_epi32(sums0, i16_l0, w);
                        sums1 = _mm256_dpwssd_avx_epi32(sums1, i16_h0, w);
                        sums2 = _mm256_dpwssd_avx_epi32(sums2, i16_l1, w);
                        sums3 = _mm256_dpwssd_avx_epi32(sums3, i16_h1, w);
                    }
                } else {
                    let j0 = _mm256_madd_epi16(i16_l0, w);
                    let j1 = _mm256_madd_epi16(i16_h0, w);
                    let j2 = _mm256_madd_epi16(i16_l1, w);
                    let j3 = _mm256_madd_epi16(i16_h1, w);

                    sums0 = _mm256_add_epi32(sums0, j0);
                    sums1 = _mm256_add_epi32(sums1, j1);
                    sums2 = _mm256_add_epi32(sums2, j2);
                    sums3 = _mm256_add_epi32(sums3, j3);
                }

                sum_in0 = _mm256_add_epi32(sum_in0, i16_l0);
                sum_in1 = _mm256_add_epi32(sum_in1, i16_h0);
                sum_in2 = _mm256_add_epi32(sum_in2, i16_l1);
                sum_in3 = _mm256_add_epi32(sum_in3, i16_h1);
            }
        }

        sp = radius;
        yp = radius;
        if yp > hm {
            yp = hm;
        }
        src_ptr = cx + yp as usize * stride as usize;
        let mut dst_ptr = cx;
        for _ in 0..height {
            let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut u8;

            let casted_sum0 = _mm256_cvtepi32_ps(sums0);
            let casted_sum1 = _mm256_cvtepi32_ps(sums1);
            let casted_sum2 = _mm256_cvtepi32_ps(sums2);
            let casted_sum3 = _mm256_cvtepi32_ps(sums3);

            let a0 = _mm256_mul_ps(casted_sum0, v_mul_value);
            let a1 = _mm256_mul_ps(casted_sum1, v_mul_value);
            let a2 = _mm256_mul_ps(casted_sum2, v_mul_value);
            let a3 = _mm256_mul_ps(casted_sum3, v_mul_value);

            let scaled_val0 = _mm256_cvtps_epi32(a0);
            let scaled_val1 = _mm256_cvtps_epi32(a1);
            let scaled_val2 = _mm256_cvtps_epi32(a2);
            let scaled_val3 = _mm256_cvtps_epi32(a3);

            let jv0 = _mm256_packus_epi32(scaled_val0, scaled_val1);
            let jv1 = _mm256_packus_epi32(scaled_val2, scaled_val3);

            _mm256_storeu_si256(store_ld as *mut _, _mm256_packus_epi16(jv0, jv1));

            dst_ptr += stride as usize;

            sums0 = _mm256_sub_epi32(sums0, sum_out0);
            sums1 = _mm256_sub_epi32(sums1, sum_out1);
            sums2 = _mm256_sub_epi32(sums2, sum_out2);
            sums3 = _mm256_sub_epi32(sums3, sum_out3);

            stack_start = sp + div - radius;
            if stack_start >= div {
                stack_start -= div;
            }

            let stack_ptr = stacks.as_mut_ptr().add(stack_start as usize * 4 * 8);

            let stack_val0 = _mm256_loadu_si256(stack_ptr as *const _);
            let stack_val1 = _mm256_loadu_si256(stack_ptr.add(8) as *const _);
            let stack_val2 = _mm256_loadu_si256(stack_ptr.add(16) as *const _);
            let stack_val3 = _mm256_loadu_si256(stack_ptr.add(24) as *const _);

            sum_out0 = _mm256_sub_epi32(sum_out0, stack_val0);
            sum_out1 = _mm256_sub_epi32(sum_out1, stack_val1);
            sum_out2 = _mm256_sub_epi32(sum_out2, stack_val2);
            sum_out3 = _mm256_sub_epi32(sum_out3, stack_val3);

            if yp < hm {
                src_ptr += stride as usize; // stride
                yp += 1;
            }

            let src_ld = pixels.slice.as_ptr().add(src_ptr);

            let src_pixel0 = _mm256_loadu_si256(src_ld as *const _);
            let lo0 = _mm256_unpacklo_epi8(src_pixel0, _mm256_setzero_si256());
            let hi1 = _mm256_unpackhi_epi8(src_pixel0, _mm256_setzero_si256());

            let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
            let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
            let i16_l1 = _mm256_unpacklo_epi16(hi1, _mm256_setzero_si256());
            let i16_h1 = _mm256_unpackhi_epi16(hi1, _mm256_setzero_si256());

            _mm256_storeu_si256(stack_ptr as *mut _, i16_l0);
            _mm256_storeu_si256(stack_ptr.add(8) as *mut _, i16_h0);
            _mm256_storeu_si256(stack_ptr.add(16) as *mut _, i16_l1);
            _mm256_storeu_si256(stack_ptr.add(24) as *mut _, i16_h1);

            sum_in0 = _mm256_add_epi32(sum_in0, i16_l0);
            sum_in1 = _mm256_add_epi32(sum_in1, i16_h0);
            sum_in2 = _mm256_add_epi32(sum_in2, i16_l1);
            sum_in3 = _mm256_add_epi32(sum_in3, i16_h1);

            sums0 = _mm256_add_epi32(sums0, sum_in0);
            sums1 = _mm256_add_epi32(sums1, sum_in1);
            sums2 = _mm256_add_epi32(sums2, sum_in2);
            sums3 = _mm256_add_epi32(sums3, sum_in3);

            sp += 1;

            if sp >= div {
                sp = 0;
            }
            let stack_ptr = stacks.as_mut_ptr().add(sp as usize * 4 * 8);

            let stack_val0 = _mm256_loadu_si256(stack_ptr as *const _);
            let stack_val1 = _mm256_loadu_si256(stack_ptr.add(8) as *const _);
            let stack_val2 = _mm256_loadu_si256(stack_ptr.add(16) as *const _);
            let stack_val3 = _mm256_loadu_si256(stack_ptr.add(24) as *const _);

            sum_out0 = _mm256_add_epi32(sum_out0, stack_val0);
            sum_out1 = _mm256_add_epi32(sum_out1, stack_val1);
            sum_out2 = _mm256_add_epi32(sum_out2, stack_val2);
            sum_out3 = _mm256_add_epi32(sum_out3, stack_val3);

            sum_in0 = _mm256_sub_epi32(sum_in0, stack_val0);
            sum_in1 = _mm256_sub_epi32(sum_in1, stack_val1);
            sum_in2 = _mm256_sub_epi32(sum_in2, stack_val2);
            sum_in3 = _mm256_sub_epi32(sum_in3, stack_val3);
        }

        cx += 32;
    }

    let v_mul_value = _mm_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

    while cx + 8 < max_x {
        let mut sums0 = _mm_setzero_si128();
        let mut sums1 = _mm_setzero_si128();

        let mut sum_in0 = _mm_setzero_si128();
        let mut sum_in1 = _mm_setzero_si128();

        let mut sum_out0 = _mm_setzero_si128();
        let mut sum_out1 = _mm_setzero_si128();

        let mut src_ptr = cx; // x,0

        let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

        {
            let src_pixel0 = _mm_loadu_si64(src_ld as *const _);
            let lo0 = _mm_unpacklo_epi8(src_pixel0, _mm_setzero_si128());

            let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
            let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

            for i in 0..=radius {
                let stack_ptr = stacks.as_mut_ptr().add(i as usize * 4 * 4);

                _mm_storeu_si128(stack_ptr as *mut _, i16_l0);
                _mm_storeu_si128(stack_ptr.add(4) as *mut _, i16_h0);

                let w = _mm_set1_epi32(i as i32 + 1);

                if VNNI {
                    #[cfg(feature = "nightly_avx512")]
                    {
                        sums0 = _mm_dpwssd_avx_epi32(sums0, i16_l0, w);
                        sums1 = _mm_dpwssd_avx_epi32(sums1, i16_h0, w);
                    }
                } else {
                    let j0 = _mm_madd_epi16(i16_l0, w);
                    let j1 = _mm_madd_epi16(i16_h0, w);

                    sums0 = _mm_add_epi32(sums0, j0);
                    sums1 = _mm_add_epi32(sums1, j1);
                }

                sum_out0 = _mm_add_epi32(sum_out0, i16_l0);
                sum_out1 = _mm_add_epi32(sum_out1, i16_h0);
            }
        }

        {
            for i in 1..=radius {
                if i <= hm {
                    src_ptr += stride as usize;
                }

                let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4 * 4);
                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                let src_pixel0 = _mm_loadu_si64(src_ld as *const u8);
                let lo0 = _mm_unpacklo_epi8(src_pixel0, _mm_setzero_si128());

                let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

                _mm_storeu_si128(stack_ptr as *mut _, i16_l0);
                _mm_storeu_si128(stack_ptr.add(4) as *mut _, i16_h0);

                let w = _mm_set1_epi32(radius as i32 + 1 - i as i32);

                if VNNI {
                    #[cfg(feature = "nightly_avx512")]
                    {
                        sums0 = _mm_dpwssd_avx_epi32(sums0, i16_l0, w);
                        sums1 = _mm_dpwssd_avx_epi32(sums1, i16_h0, w);
                    }
                } else {
                    let j0 = _mm_madd_epi16(i16_l0, w);
                    let j1 = _mm_madd_epi16(i16_h0, w);

                    sums0 = _mm_add_epi32(sums0, j0);
                    sums1 = _mm_add_epi32(sums1, j1);
                }

                sum_in0 = _mm_add_epi32(sum_in0, i16_l0);
                sum_in1 = _mm_add_epi32(sum_in1, i16_h0);
            }
        }

        sp = radius;
        yp = radius;
        if yp > hm {
            yp = hm;
        }
        src_ptr = cx + yp as usize * stride as usize;
        let mut dst_ptr = cx;
        for _ in 0..height {
            let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut u8;

            let casted_sum0 = _mm_cvtepi32_ps(sums0);
            let casted_sum1 = _mm_cvtepi32_ps(sums1);

            let a0 = _mm_mul_ps(casted_sum0, v_mul_value);
            let a1 = _mm_mul_ps(casted_sum1, v_mul_value);

            let scaled_val0 = _mm_cvtps_epi32(a0);
            let scaled_val1 = _mm_cvtps_epi32(a1);

            let jv0 = _mm_packus_epi32(scaled_val0, scaled_val1);

            _mm_storeu_si64(store_ld, _mm_packus_epi16(jv0, _mm_setzero_si128()));

            dst_ptr += stride as usize;

            sums0 = _mm_sub_epi32(sums0, sum_out0);
            sums1 = _mm_sub_epi32(sums1, sum_out1);

            stack_start = sp + div - radius;
            if stack_start >= div {
                stack_start -= div;
            }

            let stack_ptr = stacks.as_mut_ptr().add(stack_start as usize * 4 * 4);

            let stack_val0 = _mm_loadu_si128(stack_ptr as *const _);
            let stack_val1 = _mm_loadu_si128(stack_ptr.add(4) as *const _);

            sum_out0 = _mm_sub_epi32(sum_out0, stack_val0);
            sum_out1 = _mm_sub_epi32(sum_out1, stack_val1);

            if yp < hm {
                src_ptr += stride as usize; // stride
                yp += 1;
            }

            let src_ld = pixels.slice.as_ptr().add(src_ptr);

            let src_pixel0 = _mm_loadu_si64(src_ld as *const u8);
            let lo0 = _mm_unpacklo_epi8(src_pixel0, _mm_setzero_si128());

            let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
            let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

            _mm_storeu_si128(stack_ptr as *mut _, i16_l0);
            _mm_storeu_si128(stack_ptr.add(4) as *mut _, i16_h0);

            sum_in0 = _mm_add_epi32(sum_in0, i16_l0);
            sum_in1 = _mm_add_epi32(sum_in1, i16_h0);

            sums0 = _mm_add_epi32(sums0, sum_in0);
            sums1 = _mm_add_epi32(sums1, sum_in1);

            sp += 1;

            if sp >= div {
                sp = 0;
            }
            let stack_ptr = stacks.as_mut_ptr().add(sp as usize * 4 * 4);

            let stack_val0 = _mm_loadu_si128(stack_ptr as *const _);
            let stack_val1 = _mm_loadu_si128(stack_ptr.add(4) as *const _);

            sum_out0 = _mm_add_epi32(sum_out0, stack_val0);
            sum_out1 = _mm_add_epi32(sum_out1, stack_val1);

            sum_in0 = _mm_sub_epi32(sum_in0, stack_val0);
            sum_in1 = _mm_sub_epi32(sum_in1, stack_val1);
        }

        cx += 8;
    }

    while cx + CN < max_x {
        let mut sums = _mm_setzero_si128();
        let mut sum_in = _mm_setzero_si128();
        let mut sum_out = _mm_setzero_si128();

        src_ptr = cx; // x,0

        let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

        let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);

        for i in 0..=radius {
            let stack_ptr = stacks.as_mut_ptr().add(i as usize * 4);
            _mm_storeu_si128(stack_ptr as *mut __m128i, src_pixel);

            let w = _mm_set1_epi32(i as i32 + 1);

            if VNNI {
                #[cfg(feature = "nightly_avx512")]
                {
                    sums = _mm_dpwssd_avx_epi32(sums, src_pixel, w);
                }
            } else {
                sums = _mm_add_epi32(sums, _mm_madd_epi16(src_pixel, w));
            }
            sum_out = _mm_add_epi32(sum_out, src_pixel);
        }

        for i in 1..=radius {
            if i <= hm {
                src_ptr += stride as usize;
            }

            let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4);
            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
            let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);
            _mm_storeu_si128(stack_ptr as *mut __m128i, src_pixel);

            let w = _mm_set1_epi32(radius as i32 + 1 - i as i32);

            if VNNI {
                #[cfg(feature = "nightly_avx512")]
                {
                    sums = _mm_dpwssd_avx_epi32(sums, src_pixel, w);
                }
            } else {
                sums = _mm_add_epi32(sums, _mm_madd_epi16(src_pixel, w));
            }

            sum_in = _mm_add_epi32(sum_in, src_pixel);
        }

        sp = radius;
        yp = radius;
        if yp > hm {
            yp = hm;
        }
        src_ptr = cx + yp as usize * stride as usize;
        dst_ptr = cx;
        for _ in 0..height {
            let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut u8;
            let result = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(sums), v_mul_value));
            store_u8_s32::<CN>(store_ld, result);

            dst_ptr += stride as usize;

            sums = _mm_sub_epi32(sums, sum_out);

            stack_start = sp + div - radius;
            if stack_start >= div {
                stack_start -= div;
            }

            let stack_ptr = stacks.as_mut_ptr().add(stack_start as usize * 4);
            let stack_val = _mm_loadu_si128(stack_ptr as *const __m128i);
            sum_out = _mm_sub_epi32(sum_out, stack_val);

            if yp < hm {
                src_ptr += stride as usize; // stride
                yp += 1;
            }

            let src_ld = pixels.slice.as_ptr().add(src_ptr);
            let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);
            _mm_storeu_si128(stack_ptr as *mut __m128i, src_pixel);

            sum_in = _mm_add_epi32(sum_in, src_pixel);
            sums = _mm_add_epi32(sums, sum_in);

            sp += 1;

            if sp >= div {
                sp = 0;
            }
            let stack_ptr = stacks.as_mut_ptr().add(sp as usize * 4);
            let stack_val = _mm_loadu_si128(stack_ptr as *const __m128i);

            sum_out = _mm_add_epi32(sum_out, stack_val);
            sum_in = _mm_sub_epi32(sum_in, stack_val);
        }

        cx += CN;
    }

    const TAIL: usize = 1;

    while cx < max_x {
        let mut sums = _mm_setzero_si128();
        let mut sum_in = _mm_setzero_si128();
        let mut sum_out = _mm_setzero_si128();

        src_ptr = cx; // x,0

        let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

        let src_pixel = load_u8_s32_fast::<TAIL>(src_ld as *const u8);

        for i in 0..=radius {
            let stack_ptr = stacks.as_mut_ptr().add(i as usize * 4);
            _mm_storeu_si128(stack_ptr as *mut __m128i, src_pixel);

            let w = _mm_set1_epi32(i as i32 + 1);

            if VNNI {
                #[cfg(feature = "nightly_avx512")]
                {
                    sums = _mm_dpwssd_avx_epi32(sums, src_pixel, w);
                }
            } else {
                sums = _mm_add_epi32(sums, _mm_madd_epi16(src_pixel, w));
            }
            sum_out = _mm_add_epi32(sum_out, src_pixel);
        }

        for i in 1..=radius {
            if i <= hm {
                src_ptr += stride as usize;
            }

            let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4);
            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
            let src_pixel = load_u8_s32_fast::<TAIL>(src_ld as *const u8);
            _mm_storeu_si128(stack_ptr as *mut __m128i, src_pixel);

            let w = _mm_set1_epi32(radius as i32 + 1 - i as i32);

            if VNNI {
                #[cfg(feature = "nightly_avx512")]
                {
                    sums = _mm_dpwssd_avx_epi32(sums, src_pixel, w);
                }
            } else {
                sums = _mm_add_epi32(sums, _mm_madd_epi16(src_pixel, w));
            }

            sum_in = _mm_add_epi32(sum_in, src_pixel);
        }

        sp = radius;
        yp = radius;
        if yp > hm {
            yp = hm;
        }
        src_ptr = cx + yp as usize * stride as usize;
        dst_ptr = cx;
        for _ in 0..height {
            let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut u8;
            let result = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(sums), v_mul_value));
            store_u8_s32::<TAIL>(store_ld, result);

            dst_ptr += stride as usize;

            sums = _mm_sub_epi32(sums, sum_out);

            stack_start = sp + div - radius;
            if stack_start >= div {
                stack_start -= div;
            }

            let stack_ptr = stacks.as_mut_ptr().add(stack_start as usize * 4);
            let stack_val = _mm_loadu_si128(stack_ptr as *const __m128i);
            sum_out = _mm_sub_epi32(sum_out, stack_val);

            if yp < hm {
                src_ptr += stride as usize; // stride
                yp += 1;
            }

            let src_ld = pixels.slice.as_ptr().add(src_ptr);
            let src_pixel = load_u8_s32_fast::<TAIL>(src_ld as *const u8);
            _mm_storeu_si128(stack_ptr as *mut __m128i, src_pixel);

            sum_in = _mm_add_epi32(sum_in, src_pixel);
            sums = _mm_add_epi32(sums, sum_in);

            sp += 1;

            if sp >= div {
                sp = 0;
            }
            let stack_ptr = stacks.as_mut_ptr().add(sp as usize * 4);
            let stack_val = _mm_loadu_si128(stack_ptr as *const __m128i);

            sum_out = _mm_add_epi32(sum_out, stack_val);
            sum_in = _mm_sub_epi32(sum_in, stack_val);
        }

        cx += TAIL;
    }
}

impl<T, J, const COMPONENTS: usize> StackBlurWorkingPass<T, COMPONENTS>
    for VerticalAvxStackBlurPass<T, J, COMPONENTS>
where
    J: Copy
        + 'static
        + FromPrimitive
        + AddAssign<J>
        + Mul<Output = J>
        + Shr<Output = J>
        + Sub<Output = J>
        + AsPrimitive<f32>
        + SubAssign
        + AsPrimitive<T>
        + Default,
    T: Copy + AsPrimitive<J> + FromPrimitive,
    i32: AsPrimitive<J>,
    u32: AsPrimitive<J>,
    f32: AsPrimitive<T>,
    usize: AsPrimitive<J>,
{
    fn pass(
        &self,
        pixels: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        total_threads: usize,
    ) {
        unsafe {
            let pixels: &UnsafeSlice<u8> = std::mem::transmute(pixels);
            // #[cfg(feature = "nightly_avx512")]
            // {
            //     if std::arch::is_x86_feature_detected!("avxvnni") {
            //         return stack_blur_avx_vertical_fma_vnni::<COMPONENTS>(
            //             pixels,
            //             stride,
            //             width,
            //             height,
            //             radius,
            //             thread,
            //             total_threads,
            //         );
            //     }
            // }
            stack_blur_avx_vertical_def::<COMPONENTS>(
                pixels,
                stride,
                width,
                height,
                radius,
                thread,
                total_threads,
            );
        }
    }
}
