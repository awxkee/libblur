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
use crate::neon::{load_u8_s32_fast, store_u8_s32, store_u8_s32_x4};
use crate::stackblur::neon::NeonVectorI32x4;
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use crate::util::ScratchBuffer;
use std::arch::aarch64::*;

pub(crate) struct HorizontalNeonStackBlurPassQ0_31<const CN: usize> {}

impl<const CN: usize> Default for HorizontalNeonStackBlurPassQ0_31<CN> {
    fn default() -> Self {
        HorizontalNeonStackBlurPassQ0_31::<CN> {}
    }
}

impl<const CN: usize> StackBlurWorkingPass<u8, CN> for HorizontalNeonStackBlurPassQ0_31<CN> {
    fn pass(
        &self,
        pixels: &UnsafeSlice<u8>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        total_threads: usize,
    ) {
        unsafe { self.pass_impl(pixels, stride, width, height, radius, thread, total_threads) }
    }
}

impl<const CN: usize> HorizontalNeonStackBlurPassQ0_31<CN> {
    #[target_feature(enable = "rdm")]
    fn pass_impl(
        &self,
        pixels: &UnsafeSlice<u8>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        total_threads: usize,
    ) {
        unsafe {
            let min_y = thread * height as usize / total_threads;
            let max_y = (thread + 1) * height as usize / total_threads;

            let div = ((radius * 2) + 1) as usize;
            let mut _xp;
            let mut sp;
            let mut stack_start;
            let mut scratch_buffer = ScratchBuffer::<[NeonVectorI32x4; 4], 2048>::new(div);
            let stacks = scratch_buffer.as_mut_slice();

            const Q: f64 = ((1i64 << 31i64) - 1) as f64;
            let recip_scale = Q / ((radius as f64 + 1.) * (radius as f64 + 1.));
            let mul_value = vdupq_n_s32(recip_scale as i32);

            let wm = width - 1;
            let div = (radius * 2) + 1;

            let mut yy = min_y;

            while yy + 4 <= max_y {
                let mut sums0 = vdupq_n_s32(0i32);
                let mut sums1 = vdupq_n_s32(0i32);
                let mut sums2 = vdupq_n_s32(0i32);
                let mut sums3 = vdupq_n_s32(0i32);

                let mut sum_in0 = vdupq_n_s32(0i32);
                let mut sum_in1 = vdupq_n_s32(0i32);
                let mut sum_in2 = vdupq_n_s32(0i32);
                let mut sum_in3 = vdupq_n_s32(0i32);

                let mut sum_out0 = vdupq_n_s32(0i32);
                let mut sum_out1 = vdupq_n_s32(0i32);
                let mut sum_out2 = vdupq_n_s32(0i32);
                let mut sum_out3 = vdupq_n_s32(0i32);

                let mut src_ptr0 = stride as usize * yy;
                let mut src_ptr1 = stride as usize * (yy + 1);
                let mut src_ptr2 = stride as usize * (yy + 2);
                let mut src_ptr3 = stride as usize * (yy + 3);

                let src_pixel0 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr0));
                let src_pixel1 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr1));
                let src_pixel2 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr2));
                let src_pixel3 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr3));

                for i in 0..=radius {
                    let stack_value = stacks.get_unchecked_mut(i as usize);
                    vst1q_s32(stack_value.as_mut_ptr().cast(), src_pixel0);
                    vst1q_s32(stack_value[1..].as_mut_ptr().cast(), src_pixel1);
                    vst1q_s32(stack_value[2..].as_mut_ptr().cast(), src_pixel2);
                    vst1q_s32(stack_value[3..].as_mut_ptr().cast(), src_pixel3);

                    let w = i as i32 + 1;

                    sums0 = vmlaq_n_s32(sums0, src_pixel0, w);
                    sums1 = vmlaq_n_s32(sums1, src_pixel1, w);
                    sums2 = vmlaq_n_s32(sums2, src_pixel2, w);
                    sums3 = vmlaq_n_s32(sums3, src_pixel3, w);

                    sum_out0 = vaddq_s32(sum_out0, src_pixel0);
                    sum_out1 = vaddq_s32(sum_out1, src_pixel1);
                    sum_out2 = vaddq_s32(sum_out2, src_pixel2);
                    sum_out3 = vaddq_s32(sum_out3, src_pixel3);
                }

                for i in 1..=radius {
                    if i <= wm {
                        src_ptr0 += CN;
                        src_ptr1 += CN;
                        src_ptr2 += CN;
                        src_ptr3 += CN;
                    }
                    let stack_ptr = stacks.get_unchecked_mut((i + radius) as usize);

                    let src_pixel0 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr0));
                    let src_pixel1 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr1));
                    let src_pixel2 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr2));
                    let src_pixel3 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr3));

                    vst1q_s32(stack_ptr.as_mut_ptr().cast(), src_pixel0);
                    vst1q_s32(stack_ptr[1..].as_mut_ptr().cast(), src_pixel1);
                    vst1q_s32(stack_ptr[2..].as_mut_ptr().cast(), src_pixel2);
                    vst1q_s32(stack_ptr[3..].as_mut_ptr().cast(), src_pixel3);

                    let w = radius as i32 + 1 - i as i32;

                    sums0 = vmlaq_n_s32(sums0, src_pixel0, w);
                    sums1 = vmlaq_n_s32(sums1, src_pixel1, w);
                    sums2 = vmlaq_n_s32(sums2, src_pixel2, w);
                    sums3 = vmlaq_n_s32(sums3, src_pixel3, w);

                    sum_in0 = vaddq_s32(sum_in0, src_pixel0);
                    sum_in1 = vaddq_s32(sum_in1, src_pixel1);
                    sum_in2 = vaddq_s32(sum_in2, src_pixel2);
                    sum_in3 = vaddq_s32(sum_in3, src_pixel3);
                }

                sp = radius;
                _xp = radius;
                if _xp > wm {
                    _xp = wm;
                }

                src_ptr0 = CN * _xp as usize + yy * stride as usize;
                src_ptr1 = CN * _xp as usize + (yy + 1) * stride as usize;
                src_ptr2 = CN * _xp as usize + (yy + 2) * stride as usize;
                src_ptr3 = CN * _xp as usize + (yy + 3) * stride as usize;

                let mut dst_ptr0 = yy * stride as usize;
                let mut dst_ptr1 = (yy + 1) * stride as usize;
                let mut dst_ptr2 = (yy + 2) * stride as usize;
                let mut dst_ptr3 = (yy + 3) * stride as usize;

                for _ in 0..width {
                    let scaled_val0 = vqrdmulhq_s32(sums0, mul_value);
                    let scaled_val1 = vqrdmulhq_s32(sums1, mul_value);
                    let scaled_val2 = vqrdmulhq_s32(sums2, mul_value);
                    let scaled_val3 = vqrdmulhq_s32(sums3, mul_value);

                    store_u8_s32_x4::<CN>(
                        (
                            pixels.get_ptr(dst_ptr0),
                            pixels.get_ptr(dst_ptr1),
                            pixels.get_ptr(dst_ptr2),
                            pixels.get_ptr(dst_ptr3),
                        ),
                        int32x4x4_t(scaled_val0, scaled_val1, scaled_val2, scaled_val3),
                    );

                    dst_ptr0 += CN;
                    dst_ptr1 += CN;
                    dst_ptr2 += CN;
                    dst_ptr3 += CN;

                    sums0 = vsubq_s32(sums0, sum_out0);
                    sums1 = vsubq_s32(sums1, sum_out1);
                    sums2 = vsubq_s32(sums2, sum_out2);
                    sums3 = vsubq_s32(sums3, sum_out3);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }
                    let stack = stacks.get_unchecked_mut(stack_start as usize);

                    let stack_val0 = vld1q_s32(stack.as_ptr().cast());
                    let stack_val1 = vld1q_s32(stack[1..].as_ptr().cast());
                    let stack_val2 = vld1q_s32(stack[2..].as_ptr().cast());
                    let stack_val3 = vld1q_s32(stack[3..].as_ptr().cast());

                    sum_out0 = vsubq_s32(sum_out0, stack_val0);
                    sum_out1 = vsubq_s32(sum_out1, stack_val1);
                    sum_out2 = vsubq_s32(sum_out2, stack_val2);
                    sum_out3 = vsubq_s32(sum_out3, stack_val3);

                    if _xp < wm {
                        src_ptr0 += CN;
                        src_ptr1 += CN;
                        src_ptr2 += CN;
                        src_ptr3 += CN;

                        _xp += 1;
                    }

                    let src_pixel0 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr0));
                    let src_pixel1 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr1));
                    let src_pixel2 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr2));
                    let src_pixel3 = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr3));

                    vst1q_s32(stack.as_mut_ptr().cast(), src_pixel0);
                    vst1q_s32(stack[1..].as_mut_ptr().cast(), src_pixel1);
                    vst1q_s32(stack[2..].as_mut_ptr().cast(), src_pixel2);
                    vst1q_s32(stack[3..].as_mut_ptr().cast(), src_pixel3);

                    sum_in0 = vaddq_s32(sum_in0, src_pixel0);
                    sum_in1 = vaddq_s32(sum_in1, src_pixel1);
                    sum_in2 = vaddq_s32(sum_in2, src_pixel2);
                    sum_in3 = vaddq_s32(sum_in3, src_pixel3);

                    sums0 = vaddq_s32(sums0, sum_in0);
                    sums1 = vaddq_s32(sums1, sum_in1);
                    sums2 = vaddq_s32(sums2, sum_in2);
                    sums3 = vaddq_s32(sums3, sum_in3);

                    sp += 1;
                    if sp >= div {
                        sp = 0;
                    }
                    let stack = stacks.get_unchecked(sp as usize);
                    let stack_val0 = vld1q_s32(stack.as_ptr().cast());
                    let stack_val1 = vld1q_s32(stack[1..].as_ptr().cast());
                    let stack_val2 = vld1q_s32(stack[2..].as_ptr().cast());
                    let stack_val3 = vld1q_s32(stack[3..].as_ptr().cast());

                    sum_out0 = vaddq_s32(sum_out0, stack_val0);
                    sum_out1 = vaddq_s32(sum_out1, stack_val1);
                    sum_out2 = vaddq_s32(sum_out2, stack_val2);
                    sum_out3 = vaddq_s32(sum_out3, stack_val3);

                    sum_in0 = vsubq_s32(sum_in0, stack_val0);
                    sum_in1 = vsubq_s32(sum_in1, stack_val1);
                    sum_in2 = vsubq_s32(sum_in2, stack_val2);
                    sum_in3 = vsubq_s32(sum_in3, stack_val3);
                }

                yy += 4;
            }

            for y in yy..max_y {
                let mut sums = vdupq_n_s32(0i32);
                let mut sum_in = vdupq_n_s32(0i32);
                let mut sum_out = vdupq_n_s32(0i32);

                let mut src_ptr = stride as usize * y; // start of line (0,y)

                let src_pixel = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr));

                for i in 0..=radius {
                    let stack_value = stacks.get_unchecked_mut(i as usize);
                    vst1q_s32(stack_value.as_mut_ptr().cast(), src_pixel);
                    sums = vmlaq_n_s32(sums, src_pixel, i as i32 + 1);
                    sum_out = vaddq_s32(sum_out, src_pixel);
                }

                for i in 1..=radius {
                    if i <= wm {
                        src_ptr += CN;
                    }
                    let stack_ptr = stacks.get_unchecked_mut((i + radius) as usize);
                    let src_pixel = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr));
                    vst1q_s32(stack_ptr.as_mut_ptr().cast(), src_pixel);
                    sums = vmlaq_n_s32(sums, src_pixel, radius as i32 + 1 - i as i32);

                    sum_in = vaddq_s32(sum_in, src_pixel);
                }

                sp = radius;
                _xp = radius;
                if _xp > wm {
                    _xp = wm;
                }

                src_ptr = CN * _xp as usize + y * stride as usize;

                let mut dst_ptr = y * stride as usize;
                for _ in 0..width {
                    let store_ld = pixels.get_ptr(dst_ptr);

                    let scaled_val = vqrdmulhq_s32(sums, mul_value);
                    store_u8_s32::<CN>(store_ld, scaled_val);
                    dst_ptr += CN;

                    sums = vsubq_s32(sums, sum_out);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }
                    let stack = stacks.get_unchecked_mut(stack_start as usize);

                    let stack_val = vld1q_s32(stack.as_mut_ptr().cast());

                    sum_out = vsubq_s32(sum_out, stack_val);

                    if _xp < wm {
                        src_ptr += CN;
                        _xp += 1;
                    }

                    let src_pixel = load_u8_s32_fast::<CN>(pixels.get_ptr(src_ptr));
                    vst1q_s32(stack.as_mut_ptr().cast(), src_pixel);

                    sum_in = vaddq_s32(sum_in, src_pixel);
                    sums = vaddq_s32(sums, sum_in);

                    sp += 1;
                    if sp >= div {
                        sp = 0;
                    }
                    let stack = stacks.get_unchecked_mut(sp as usize);
                    let stack_val = vld1q_s32(stack.as_ptr().cast());

                    sum_out = vaddq_s32(sum_out, stack_val);
                    sum_in = vsubq_s32(sum_in, stack_val);
                }
            }
        }
    }
}
