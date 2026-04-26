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
use crate::filter1d::sse::utils::_mm_opt_fmlaf_ps;
use crate::sse::{load_f32, store_f32};
use crate::stackblur::sse::vertical_f32::SseF32x4;
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) struct HorizontalSseStackBlurPassFloat32<const CN: usize> {}

impl<const CN: usize> Default for HorizontalSseStackBlurPassFloat32<CN> {
    fn default() -> Self {
        HorizontalSseStackBlurPassFloat32::<CN> {}
    }
}

#[target_feature(enable = "sse4.1")]
fn horiz_f32_pass_stack_impl<const CN: usize>(
    pixels: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    thread: usize,
    total_threads: usize,
) {
    unsafe {
        let div = ((radius * 2) + 1) as usize;
        let v_mul_value = _mm_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));
        let mut xp;
        let mut sp;
        let mut stack_start;
        let mut stacks = vec![SseF32x4::default(); div];

        let wm = width - 1;
        let div = (radius * 2) + 1;

        let mut src_ptr;
        let mut dst_ptr;

        let min_y = thread * height as usize / total_threads;
        let max_y = (thread + 1) * height as usize / total_threads;

        for y in min_y..max_y {
            let mut sums = _mm_setzero_ps();
            let mut sum_in = _mm_setzero_ps();
            let mut sum_out = _mm_setzero_ps();

            src_ptr = stride as usize * y;

            let src_pixel = load_f32::<CN>(pixels.get_ptr(src_ptr));

            for i in 0..=radius {
                let stack_value = stacks.as_mut_ptr().add(i as usize);
                _mm_store_ps(stack_value.cast(), src_pixel);
                sums = _mm_opt_fmlaf_ps(sums, src_pixel, _mm_set1_ps((i + 1) as f32));
                sum_out = _mm_add_ps(sum_out, src_pixel);
            }

            for i in 1..=radius {
                if i <= wm {
                    src_ptr += CN;
                }
                let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize);
                let src_ld = pixels.get_ptr(src_ptr) as *const f32;
                let src_pixel = load_f32::<CN>(src_ld);
                _mm_store_ps(stack_ptr.cast(), src_pixel);
                sums = _mm_opt_fmlaf_ps(sums, src_pixel, _mm_set1_ps((radius + 1 - i) as f32));

                sum_in = _mm_add_ps(sum_in, src_pixel);
            }

            sp = radius;
            xp = radius;
            if xp > wm {
                xp = wm;
            }

            src_ptr = CN * xp as usize + y * stride as usize;
            dst_ptr = y * stride as usize;
            for _ in 0..width {
                let blurred = _mm_mul_ps(sums, v_mul_value);
                store_f32::<CN>(pixels.get_ptr(dst_ptr), blurred);
                dst_ptr += CN;

                sums = _mm_sub_ps(sums, sum_out);

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack = stacks.as_mut_ptr().add(stack_start as usize);

                let stack_val = _mm_load_ps(stack.cast());

                sum_out = _mm_sub_ps(sum_out, stack_val);

                if xp < wm {
                    src_ptr += CN;
                    xp += 1;
                }

                let src_ld = pixels.get_ptr(src_ptr) as *const f32;
                let src_pixel = load_f32::<CN>(src_ld);
                _mm_store_ps(stack.cast(), src_pixel);

                sum_in = _mm_add_ps(sum_in, src_pixel);
                sums = _mm_add_ps(sums, sum_in);

                sp += 1;
                if sp >= div {
                    sp = 0;
                }
                let stack = stacks.as_mut_ptr().add(sp as usize);
                let stack_val = _mm_load_ps(stack.cast());

                sum_out = _mm_add_ps(sum_out, stack_val);
                sum_in = _mm_sub_ps(sum_in, stack_val);
            }
        }
    }
}

impl<const CN: usize> StackBlurWorkingPass<f32, CN> for HorizontalSseStackBlurPassFloat32<CN> {
    fn pass(
        &self,
        pixels: &UnsafeSlice<f32>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        total_threads: usize,
    ) {
        unsafe {
            horiz_f32_pass_stack_impl::<CN>(
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
