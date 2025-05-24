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
use crate::sse::{load_f32_f16, store_f32_f16};
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use half::f16;
use num_traits::{AsPrimitive, FromPrimitive};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Sub, SubAssign};

pub(crate) struct VerticalSseStackBlurPassFloat16<T, J, const CN: usize> {
    _phantom_t: PhantomData<T>,
    _phantom_j: PhantomData<J>,
}

impl<T, J, const CN: usize> Default for VerticalSseStackBlurPassFloat16<T, J, CN> {
    fn default() -> Self {
        VerticalSseStackBlurPassFloat16::<T, J, CN> {
            _phantom_t: Default::default(),
            _phantom_j: Default::default(),
        }
    }
}

impl<T, J, const CN: usize> VerticalSseStackBlurPassFloat16<T, J, CN>
where
    J: Copy
        + 'static
        + FromPrimitive
        + AddAssign<J>
        + Mul<Output = J>
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
        unsafe {
            let pixels: &UnsafeSlice<f16> = std::mem::transmute(pixels);
            let div = ((radius * 2) + 1) as usize;
            let mut yp;
            let mut sp;
            let mut stack_start;
            let mut stacks = vec![0.; 4 * div * total_threads];

            let v_mul_value = _mm_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

            let hm = height - 1;
            let div = (radius * 2) + 1;

            let mut src_ptr;
            let mut dst_ptr;

            let min_x = thread * width as usize / total_threads;
            let max_x = (thread + 1) * width as usize / total_threads;

            for x in min_x..max_x {
                let mut sums = _mm_setzero_ps();
                let mut sum_in = _mm_setzero_ps();
                let mut sum_out = _mm_setzero_ps();

                src_ptr = CN * x; // x,0

                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const f16;
                let src_pixel = load_f32_f16::<CN>(src_ld);

                for i in 0..=radius {
                    let stack_ptr = stacks.as_mut_ptr().add(i as usize * 4);
                    _mm_storeu_ps(stack_ptr, src_pixel);
                    sums = _mm_add_ps(sums, _mm_mul_ps(src_pixel, _mm_set1_ps(i as f32 + 1f32)));
                    sum_out = _mm_add_ps(sum_out, src_pixel);
                }

                for i in 1..=radius {
                    if i <= hm {
                        src_ptr += stride as usize;
                    }

                    let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4);
                    let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const f16;
                    let src_pixel = load_f32_f16::<CN>(src_ld);
                    _mm_storeu_ps(stack_ptr, src_pixel);
                    sums = _mm_add_ps(
                        sums,
                        _mm_mul_ps(src_pixel, _mm_set1_ps(radius as f32 + 1f32 - i as f32)),
                    );

                    sum_in = _mm_add_ps(sum_in, src_pixel);
                }

                sp = radius;
                yp = radius;
                if yp > hm {
                    yp = hm;
                }
                src_ptr = CN * x + yp as usize * stride as usize;
                dst_ptr = CN * x;
                for _ in 0..height {
                    let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut f16;
                    let blurred = _mm_mul_ps(sums, v_mul_value);
                    store_f32_f16::<CN>(store_ld, blurred);

                    dst_ptr += stride as usize;

                    sums = _mm_sub_ps(sums, sum_out);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }

                    let stack_ptr = stacks.as_mut_ptr().add(stack_start as usize * 4);
                    let stack_val = _mm_loadu_ps(stack_ptr as *const f32);
                    sum_out = _mm_sub_ps(sum_out, stack_val);

                    if yp < hm {
                        src_ptr += stride as usize; // stride
                        yp += 1;
                    }

                    let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const f16;
                    let src_pixel = load_f32_f16::<CN>(src_ld);
                    _mm_storeu_ps(stack_ptr, src_pixel);

                    sum_in = _mm_add_ps(sum_in, src_pixel);
                    sums = _mm_add_ps(sums, sum_in);

                    sp += 1;

                    if sp >= div {
                        sp = 0;
                    }
                    let stack_ptr = stacks.as_mut_ptr().add(sp as usize * 4);
                    let stack_val = _mm_loadu_ps(stack_ptr as *const f32);

                    sum_out = _mm_add_ps(sum_out, stack_val);
                    sum_in = _mm_sub_ps(sum_in, stack_val);
                }
            }
        }
    }
}

impl<T, J, const CN: usize> StackBlurWorkingPass<T, CN>
    for VerticalSseStackBlurPassFloat16<T, J, CN>
where
    J: Copy
        + 'static
        + FromPrimitive
        + AddAssign<J>
        + Mul<Output = J>
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
