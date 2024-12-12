/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Shr, Sub, SubAssign};

pub struct VerticalSseStackBlurPass<T, J, const COMPONENTS: usize> {
    _phantom_t: PhantomData<T>,
    _phantom_j: PhantomData<J>,
}

impl<T, J, const COMPONENTS: usize> Default for VerticalSseStackBlurPass<T, J, COMPONENTS> {
    fn default() -> Self {
        VerticalSseStackBlurPass::<T, J, COMPONENTS> {
            _phantom_t: Default::default(),
            _phantom_j: Default::default(),
        }
    }
}

impl<T, J, const COMPONENTS: usize> VerticalSseStackBlurPass<T, J, COMPONENTS>
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
    #[target_feature(enable = "sse4.1")]
    unsafe fn pass_impl(
        &self,
        pixels: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        total_threads: usize,
    ) {
        let pixels: &UnsafeSlice<u8> = std::mem::transmute(pixels);
        let div = ((radius * 2) + 1) as usize;
        let mut yp;
        let mut sp;
        let mut stack_start;
        let mut stacks = vec![0i32; 4 * div * total_threads];

        let v_mul_value = _mm_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

        let hm = height - 1;
        let div = (radius * 2) + 1;

        let mut src_ptr;
        let mut dst_ptr;

        let min_x = thread * width as usize / total_threads;
        let max_x = (thread + 1) * width as usize / total_threads;

        for x in min_x..max_x {
            let mut sums = _mm_setzero_si128();
            let mut sum_in = _mm_setzero_si128();
            let mut sum_out = _mm_setzero_si128();

            src_ptr = COMPONENTS * x; // x,0

            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

            let src_pixel = load_u8_s32_fast::<COMPONENTS>(src_ld as *const u8);

            for i in 0..=radius {
                let stack_ptr = stacks.as_mut_ptr().add(i as usize * 4);
                _mm_storeu_si128(stack_ptr as *mut __m128i, src_pixel);
                sums = _mm_add_epi32(
                    sums,
                    _mm_madd_epi16(src_pixel, _mm_set1_epi32(i as i32 + 1)),
                );
                sum_out = _mm_add_epi32(sum_out, src_pixel);
            }

            for i in 1..=radius {
                if i <= hm {
                    src_ptr += stride as usize;
                }

                let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4);
                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                let src_pixel = load_u8_s32_fast::<COMPONENTS>(src_ld as *const u8);
                _mm_storeu_si128(stack_ptr as *mut __m128i, src_pixel);
                sums = _mm_add_epi32(
                    sums,
                    _mm_madd_epi16(src_pixel, _mm_set1_epi32(radius as i32 + 1 - i as i32)),
                );

                sum_in = _mm_add_epi32(sum_in, src_pixel);
            }

            sp = radius;
            yp = radius;
            if yp > hm {
                yp = hm;
            }
            src_ptr = COMPONENTS * x + yp as usize * stride as usize;
            dst_ptr = COMPONENTS * x;
            for _ in 0..height {
                let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut u8;
                const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                let result = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
                    _mm_cvtepi32_ps(sums),
                    v_mul_value,
                )));
                store_u8_s32::<COMPONENTS>(store_ld, result);

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
                let src_pixel = load_u8_s32_fast::<COMPONENTS>(src_ld as *const u8);
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
        }
    }
}

impl<T, J, const COMPONENTS: usize> StackBlurWorkingPass<T, COMPONENTS>
    for VerticalSseStackBlurPass<T, J, COMPONENTS>
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
            self.pass_impl(pixels, stride, width, height, radius, thread, total_threads);
        }
    }
}
