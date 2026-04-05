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
use crate::neon::{load_f32_fast, store_f32};
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use crate::util::ScratchBuffer;
use std::arch::aarch64::*;

#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
pub(crate) struct NeonVectorF32x4([f32; 4]);

pub(crate) struct HorizontalNeonStackBlurPassFloat32<const CN: usize> {}

impl<const CN: usize> Default for HorizontalNeonStackBlurPassFloat32<CN> {
    fn default() -> Self {
        HorizontalNeonStackBlurPassFloat32::<CN> {}
    }
}

impl<const CN: usize> HorizontalNeonStackBlurPassFloat32<CN> {
    #[target_feature(enable = "neon")]
    fn pass_impl(
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
            let div = ((radius * 2) + 1) as usize;
            let v_mul_value = vdupq_n_f32(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));
            let mut xp;
            let mut sp;
            let mut stack_start;
            let mut scratch_buffer = ScratchBuffer::<NeonVectorF32x4, 2048>::new(div);
            let stacks = scratch_buffer.as_mut_slice();

            let wm = width - 1;
            let div = (radius * 2) + 1;

            let mut src_ptr;
            let mut dst_ptr;

            let min_y = thread * height as usize / total_threads;
            let max_y = (thread + 1) * height as usize / total_threads;

            for y in min_y..max_y {
                let mut sums = vdupq_n_f32(0.);
                let mut sum_in = vdupq_n_f32(0.);
                let mut sum_out = vdupq_n_f32(0.);

                src_ptr = stride as usize * y;

                let src_pixel = load_f32_fast::<CN>(pixels.get_ptr(src_ptr));

                for i in 0..=radius {
                    let stack_value = stacks.get_unchecked_mut(i as usize..);
                    vst1q_f32(stack_value.as_mut_ptr().cast(), src_pixel);
                    sums = vfmaq_n_f32(sums, src_pixel, (i + 1) as f32);
                    sum_out = vaddq_f32(sum_out, src_pixel);
                }

                for i in 1..=radius {
                    if i <= wm {
                        src_ptr += CN;
                    }
                    let stack_ptr = stacks.get_unchecked_mut((i + radius) as usize..);
                    let src_pixel = load_f32_fast::<CN>(pixels.get_ptr(src_ptr));
                    vst1q_f32(stack_ptr.as_mut_ptr().cast(), src_pixel);
                    sums = vfmaq_n_f32(sums, src_pixel, (radius + 1 - i) as f32);

                    sum_in = vaddq_f32(sum_in, src_pixel);
                }

                sp = radius;
                xp = radius;
                if xp > wm {
                    xp = wm;
                }

                src_ptr = CN * xp as usize + y * stride as usize;
                dst_ptr = y * stride as usize;
                for _ in 0..width {
                    let blurred = vmulq_f32(sums, v_mul_value);
                    store_f32::<CN>(pixels.get_ptr(dst_ptr), blurred);
                    dst_ptr += CN;

                    sums = vsubq_f32(sums, sum_out);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }
                    let stack = stacks.get_unchecked_mut(stack_start as usize..);

                    let stack_val = vld1q_f32(stack.as_ptr().cast());

                    sum_out = vsubq_f32(sum_out, stack_val);

                    if xp < wm {
                        src_ptr += CN;
                        xp += 1;
                    }

                    let src_pixel = load_f32_fast::<CN>(pixels.get_ptr(src_ptr));
                    vst1q_f32(stack.as_mut_ptr().cast(), src_pixel);

                    sum_in = vaddq_f32(sum_in, src_pixel);
                    sums = vaddq_f32(sums, sum_in);

                    sp += 1;
                    if sp >= div {
                        sp = 0;
                    }
                    let stack = stacks.get_unchecked(sp as usize..);
                    let stack_val = vld1q_f32(stack.as_ptr().cast());

                    sum_out = vaddq_f32(sum_out, stack_val);
                    sum_in = vsubq_f32(sum_in, stack_val);
                }
            }
        }
    }
}

impl<const CN: usize> StackBlurWorkingPass<f32, CN> for HorizontalNeonStackBlurPassFloat32<CN> {
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
            self.pass_impl(pixels, stride, width, height, radius, thread, total_threads);
        }
    }
}
