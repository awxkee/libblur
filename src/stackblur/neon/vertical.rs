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
use crate::neon::{load_u8_s32_fast, store_u8_s32};
use crate::primitives::PrimitiveCast;
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Shr, Sub, SubAssign};

pub(crate) struct VerticalNeonStackBlurPass<T, J, const CN: usize> {
    _phantom_t: PhantomData<T>,
    _phantom_j: PhantomData<J>,
}

impl<T, J, const CN: usize> Default for VerticalNeonStackBlurPass<T, J, CN> {
    fn default() -> Self {
        VerticalNeonStackBlurPass {
            _phantom_t: Default::default(),
            _phantom_j: Default::default(),
        }
    }
}

impl<T, J, const CN: usize> StackBlurWorkingPass<T, CN> for VerticalNeonStackBlurPass<T, J, CN>
where
    J: Copy
        + 'static
        + AddAssign<J>
        + Mul<Output = J>
        + Shr<Output = J>
        + Sub<Output = J>
        + PrimitiveCast<f32>
        + SubAssign
        + PrimitiveCast<T>
        + Default,
    T: Copy + PrimitiveCast<J> + Default,
    i32: PrimitiveCast<J>,
    u32: PrimitiveCast<J>,
    f32: PrimitiveCast<T>,
    usize: PrimitiveCast<J>,
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
            let div = ((radius * 2) + 1) as usize;
            let mut yp;
            let mut sp;
            let mut stack_start;
            let mut stacks0 = vec![0i32; 4 * div * 4];

            let hm = height - 1;
            let div = (radius * 2) + 1;
            let mul_value = vdupq_n_f32(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

            let min_x = (thread * width as usize / total_threads) * CN;
            let max_x = ((thread + 1) * width as usize / total_threads) * CN;

            let mut cx = min_x;

            while cx + 16 < max_x {
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

                let mut src_ptr = cx; // x,0

                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

                {
                    let src_pixel0 = vld1q_u8(src_ld as *const u8);
                    let lo0 = vmovl_u8(vget_low_u8(src_pixel0));
                    let hi0 = vmovl_high_u8(src_pixel0);

                    let i16_l0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo0)));
                    let i16_h0 = vreinterpretq_s32_u32(vmovl_high_u16(lo0));
                    let i16_l1 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi0)));
                    let i16_h1 = vreinterpretq_s32_u32(vmovl_high_u16(hi0));

                    for i in 0..=radius {
                        let stack_ptr = stacks0.as_mut_ptr().add(i as usize * 4 * 4);

                        vst1q_s32(stack_ptr, i16_l0);
                        vst1q_s32(stack_ptr.add(4), i16_h0);
                        vst1q_s32(stack_ptr.add(8), i16_l1);
                        vst1q_s32(stack_ptr.add(12), i16_h1);

                        let w = vdupq_n_s32(i as i32 + 1);

                        sums0 = vmlaq_s32(sums0, i16_l0, w);
                        sums1 = vmlaq_s32(sums1, i16_h0, w);
                        sums2 = vmlaq_s32(sums2, i16_l1, w);
                        sums3 = vmlaq_s32(sums3, i16_h1, w);

                        sum_out0 = vaddq_s32(sum_out0, i16_l0);
                        sum_out1 = vaddq_s32(sum_out1, i16_h0);
                        sum_out2 = vaddq_s32(sum_out2, i16_l1);
                        sum_out3 = vaddq_s32(sum_out3, i16_h1);
                    }
                }

                {
                    for i in 1..=radius {
                        if i <= hm {
                            src_ptr += stride as usize;
                        }

                        let stack_ptr = stacks0.as_mut_ptr().add((i + radius) as usize * 4 * 4);
                        let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                        let src_pixel0 = vld1q_u8(src_ld as *const u8);
                        let lo0 = vmovl_u8(vget_low_u8(src_pixel0));
                        let hi0 = vmovl_high_u8(src_pixel0);

                        let i16_l0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo0)));
                        let i16_h0 = vreinterpretq_s32_u32(vmovl_high_u16(lo0));
                        let i16_l1 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi0)));
                        let i16_h1 = vreinterpretq_s32_u32(vmovl_high_u16(hi0));

                        vst1q_s32(stack_ptr, i16_l0);
                        vst1q_s32(stack_ptr.add(4), i16_h0);
                        vst1q_s32(stack_ptr.add(8), i16_l1);
                        vst1q_s32(stack_ptr.add(12), i16_h1);

                        let vj = vdupq_n_s32(radius as i32 + 1 - i as i32);

                        sums0 = vmlaq_s32(sums0, i16_l0, vj);
                        sums1 = vmlaq_s32(sums1, i16_h0, vj);
                        sums2 = vmlaq_s32(sums2, i16_l1, vj);
                        sums3 = vmlaq_s32(sums3, i16_h1, vj);

                        sum_in0 = vaddq_s32(sum_in0, i16_l0);
                        sum_in1 = vaddq_s32(sum_in1, i16_h0);
                        sum_in2 = vaddq_s32(sum_in2, i16_l1);
                        sum_in3 = vaddq_s32(sum_in3, i16_h1);
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

                    let casted_sum0 = vcvtq_f32_s32(sums0);
                    let casted_sum1 = vcvtq_f32_s32(sums1);
                    let casted_sum2 = vcvtq_f32_s32(sums2);
                    let casted_sum3 = vcvtq_f32_s32(sums3);

                    let a0 = vmulq_f32(casted_sum0, mul_value);
                    let a1 = vmulq_f32(casted_sum1, mul_value);
                    let a2 = vmulq_f32(casted_sum2, mul_value);
                    let a3 = vmulq_f32(casted_sum3, mul_value);

                    let scaled_val0 = vcvtaq_s32_f32(a0);
                    let scaled_val1 = vcvtaq_s32_f32(a1);
                    let scaled_val2 = vcvtaq_s32_f32(a2);
                    let scaled_val3 = vcvtaq_s32_f32(a3);

                    let jv0 = vcombine_u16(vqmovun_s32(scaled_val0), vqmovun_s32(scaled_val1));
                    let jv1 = vcombine_u16(vqmovun_s32(scaled_val2), vqmovun_s32(scaled_val3));

                    let values = vcombine_u8(vqmovn_u16(jv0), vqmovn_u16(jv1));
                    vst1q_u8(store_ld, values);

                    dst_ptr += stride as usize;

                    sums0 = vsubq_s32(sums0, sum_out0);
                    sums1 = vsubq_s32(sums1, sum_out1);
                    sums2 = vsubq_s32(sums2, sum_out2);
                    sums3 = vsubq_s32(sums3, sum_out3);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }

                    let stack_ptr = stacks0.as_mut_ptr().add(stack_start as usize * 4 * 4);

                    let stack_val0 = vld1q_s32(stack_ptr);
                    let stack_val1 = vld1q_s32(stack_ptr.add(4));
                    let stack_val2 = vld1q_s32(stack_ptr.add(8));
                    let stack_val3 = vld1q_s32(stack_ptr.add(12));

                    sum_out0 = vsubq_s32(sum_out0, stack_val0);
                    sum_out1 = vsubq_s32(sum_out1, stack_val1);
                    sum_out2 = vsubq_s32(sum_out2, stack_val2);
                    sum_out3 = vsubq_s32(sum_out3, stack_val3);

                    if yp < hm {
                        src_ptr += stride as usize; // stride
                        yp += 1;
                    }

                    let src_ld = pixels.slice.as_ptr().add(src_ptr);

                    let src_pixel0 = vld1q_u8(src_ld as *const u8);
                    let lo0 = vmovl_u8(vget_low_u8(src_pixel0));
                    let hi0 = vmovl_high_u8(src_pixel0);

                    let i16_l0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo0)));
                    let i16_h0 = vreinterpretq_s32_u32(vmovl_high_u16(lo0));
                    let i16_l1 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi0)));
                    let i16_h1 = vreinterpretq_s32_u32(vmovl_high_u16(hi0));

                    vst1q_s32(stack_ptr, i16_l0);
                    vst1q_s32(stack_ptr.add(4), i16_h0);
                    vst1q_s32(stack_ptr.add(8), i16_l1);
                    vst1q_s32(stack_ptr.add(12), i16_h1);

                    sum_in0 = vaddq_s32(sum_in0, i16_l0);
                    sum_in1 = vaddq_s32(sum_in1, i16_h0);
                    sum_in2 = vaddq_s32(sum_in2, i16_l1);
                    sum_in3 = vaddq_s32(sum_in3, i16_h1);

                    sums0 = vaddq_s32(sums0, sum_in0);
                    sums1 = vaddq_s32(sums1, sum_in1);
                    sums2 = vaddq_s32(sums2, sum_in2);
                    sums3 = vaddq_s32(sums3, sum_in3);

                    sp += 1;

                    if sp >= div {
                        sp = 0;
                    }
                    let stack_ptr = stacks0.as_mut_ptr().add(sp as usize * 4 * 4);

                    let stack_val0 = vld1q_s32(stack_ptr);
                    let stack_val1 = vld1q_s32(stack_ptr.add(4));
                    let stack_val2 = vld1q_s32(stack_ptr.add(8));
                    let stack_val3 = vld1q_s32(stack_ptr.add(12));

                    sum_out0 = vaddq_s32(sum_out0, stack_val0);
                    sum_out1 = vaddq_s32(sum_out1, stack_val1);
                    sum_out2 = vaddq_s32(sum_out2, stack_val2);
                    sum_out3 = vaddq_s32(sum_out3, stack_val3);

                    sum_in0 = vsubq_s32(sum_in0, stack_val0);
                    sum_in1 = vsubq_s32(sum_in1, stack_val1);
                    sum_in2 = vsubq_s32(sum_in2, stack_val2);
                    sum_in3 = vsubq_s32(sum_in3, stack_val3);
                }

                cx += 16;
            }

            while cx + 8 < max_x {
                let mut sums0 = vdupq_n_s32(0i32);
                let mut sums1 = vdupq_n_s32(0i32);

                let mut sum_in0 = vdupq_n_s32(0i32);
                let mut sum_in1 = vdupq_n_s32(0i32);

                let mut sum_out0 = vdupq_n_s32(0i32);
                let mut sum_out1 = vdupq_n_s32(0i32);

                let mut src_ptr = cx; // x,0

                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

                {
                    let src_pixel0 = vld1_u8(src_ld as *const u8);
                    let lo0 = vmovl_u8(src_pixel0);

                    let i16_l0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo0)));
                    let i16_h0 = vreinterpretq_s32_u32(vmovl_high_u16(lo0));

                    for i in 0..=radius {
                        let stack_ptr = stacks0.as_mut_ptr().add(i as usize * 4 * 4);

                        vst1q_s32(stack_ptr, i16_l0);
                        vst1q_s32(stack_ptr.add(4), i16_h0);

                        let w = vdupq_n_s32(i as i32 + 1);

                        sums0 = vmlaq_s32(sums0, i16_l0, w);
                        sums1 = vmlaq_s32(sums1, i16_h0, w);

                        sum_out0 = vaddq_s32(sum_out0, i16_l0);
                        sum_out1 = vaddq_s32(sum_out1, i16_h0);
                    }
                }

                {
                    for i in 1..=radius {
                        if i <= hm {
                            src_ptr += stride as usize;
                        }

                        let stack_ptr = stacks0.as_mut_ptr().add((i + radius) as usize * 4 * 4);
                        let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                        let src_pixel0 = vld1_u8(src_ld as *const u8);
                        let lo0 = vmovl_u8(src_pixel0);

                        let i16_l0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo0)));
                        let i16_h0 = vreinterpretq_s32_u32(vmovl_high_u16(lo0));

                        vst1q_s32(stack_ptr, i16_l0);
                        vst1q_s32(stack_ptr.add(4), i16_h0);

                        let vj = vdupq_n_s32(radius as i32 + 1 - i as i32);

                        sums0 = vmlaq_s32(sums0, i16_l0, vj);
                        sums1 = vmlaq_s32(sums1, i16_h0, vj);

                        sum_in0 = vaddq_s32(sum_in0, i16_l0);
                        sum_in1 = vaddq_s32(sum_in1, i16_h0);
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

                    let casted_sum0 = vcvtq_f32_s32(sums0);
                    let casted_sum1 = vcvtq_f32_s32(sums1);

                    let a0 = vmulq_f32(casted_sum0, mul_value);
                    let a1 = vmulq_f32(casted_sum1, mul_value);

                    let scaled_val0 = vcvtaq_s32_f32(a0);
                    let scaled_val1 = vcvtaq_s32_f32(a1);

                    let jv0 = vcombine_u16(vqmovun_s32(scaled_val0), vqmovun_s32(scaled_val1));

                    vst1_u8(store_ld, vqmovn_u16(jv0));

                    dst_ptr += stride as usize;

                    sums0 = vsubq_s32(sums0, sum_out0);
                    sums1 = vsubq_s32(sums1, sum_out1);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }

                    let stack_ptr = stacks0.as_mut_ptr().add(stack_start as usize * 4 * 4);

                    let stack_val0 = vld1q_s32(stack_ptr);
                    let stack_val1 = vld1q_s32(stack_ptr.add(4));

                    sum_out0 = vsubq_s32(sum_out0, stack_val0);
                    sum_out1 = vsubq_s32(sum_out1, stack_val1);

                    if yp < hm {
                        src_ptr += stride as usize; // stride
                        yp += 1;
                    }

                    let src_ld = pixels.slice.as_ptr().add(src_ptr);

                    let src_pixel0 = vld1_u8(src_ld as *const u8);
                    let lo0 = vmovl_u8(src_pixel0);

                    let i16_l0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo0)));
                    let i16_h0 = vreinterpretq_s32_u32(vmovl_high_u16(lo0));

                    vst1q_s32(stack_ptr, i16_l0);
                    vst1q_s32(stack_ptr.add(4), i16_h0);

                    sum_in0 = vaddq_s32(sum_in0, i16_l0);
                    sum_in1 = vaddq_s32(sum_in1, i16_h0);

                    sums0 = vaddq_s32(sums0, sum_in0);
                    sums1 = vaddq_s32(sums1, sum_in1);

                    sp += 1;

                    if sp >= div {
                        sp = 0;
                    }
                    let stack_ptr = stacks0.as_mut_ptr().add(sp as usize * 4 * 4);

                    let stack_val0 = vld1q_s32(stack_ptr);
                    let stack_val1 = vld1q_s32(stack_ptr.add(4));

                    sum_out0 = vaddq_s32(sum_out0, stack_val0);
                    sum_out1 = vaddq_s32(sum_out1, stack_val1);

                    sum_in0 = vsubq_s32(sum_in0, stack_val0);
                    sum_in1 = vsubq_s32(sum_in1, stack_val1);
                }

                cx += 8;
            }

            while cx + CN < max_x {
                let mut sums = vdupq_n_s32(0i32);
                let mut sum_in = vdupq_n_s32(0i32);
                let mut sum_out = vdupq_n_s32(0i32);

                let mut src_ptr = cx; // x,0

                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

                let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);

                for i in 0..=radius {
                    let stack_ptr = stacks0.as_mut_ptr().add(i as usize * 4);
                    vst1q_s32(stack_ptr, src_pixel);
                    sums = vmlaq_s32(sums, src_pixel, vdupq_n_s32(i as i32 + 1));
                    sum_out = vaddq_s32(sum_out, src_pixel);
                }

                for i in 1..=radius {
                    if i <= hm {
                        src_ptr += stride as usize;
                    }

                    let stack_ptr = stacks0.as_mut_ptr().add((i + radius) as usize * 4);
                    let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                    let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);
                    vst1q_s32(stack_ptr, src_pixel);
                    sums = vmlaq_s32(sums, src_pixel, vdupq_n_s32(radius as i32 + 1 - i as i32));

                    sum_in = vaddq_s32(sum_in, src_pixel);
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
                    let casted_sum = vcvtq_f32_s32(sums);
                    let scaled_val = vcvtaq_s32_f32(vmulq_f32(casted_sum, mul_value));
                    store_u8_s32::<CN>(store_ld, scaled_val);

                    dst_ptr += stride as usize;

                    sums = vsubq_s32(sums, sum_out);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }

                    let stack_ptr = stacks0.as_mut_ptr().add(stack_start as usize * 4);
                    let stack_val = vld1q_s32(stack_ptr);
                    sum_out = vsubq_s32(sum_out, stack_val);

                    if yp < hm {
                        src_ptr += stride as usize; // stride
                        yp += 1;
                    }

                    let src_ld = pixels.slice.as_ptr().add(src_ptr);
                    let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);
                    vst1q_s32(stack_ptr, src_pixel);

                    sum_in = vaddq_s32(sum_in, src_pixel);
                    sums = vaddq_s32(sums, sum_in);

                    sp += 1;

                    if sp >= div {
                        sp = 0;
                    }
                    let stack_ptr = stacks0.as_mut_ptr().add(sp as usize * 4);
                    let stack_val = vld1q_s32(stack_ptr);

                    sum_out = vaddq_s32(sum_out, stack_val);
                    sum_in = vsubq_s32(sum_in, stack_val);
                }

                cx += CN;
            }

            const TAIL: usize = 1;

            while cx < max_x {
                let mut sums = vdupq_n_s32(0i32);
                let mut sum_in = vdupq_n_s32(0i32);
                let mut sum_out = vdupq_n_s32(0i32);

                let mut src_ptr = cx; // x,0

                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

                let src_pixel = load_u8_s32_fast::<TAIL>(src_ld as *const u8);

                for i in 0..=radius {
                    let stack_ptr = stacks0.as_mut_ptr().add(i as usize * 4);
                    vst1q_s32(stack_ptr, src_pixel);
                    sums = vmlaq_s32(sums, src_pixel, vdupq_n_s32(i as i32 + 1));
                    sum_out = vaddq_s32(sum_out, src_pixel);
                }

                for i in 1..=radius {
                    if i <= hm {
                        src_ptr += stride as usize;
                    }

                    let stack_ptr = stacks0.as_mut_ptr().add((i + radius) as usize * 4);
                    let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                    let src_pixel = load_u8_s32_fast::<TAIL>(src_ld as *const u8);
                    vst1q_s32(stack_ptr, src_pixel);
                    sums = vmlaq_s32(sums, src_pixel, vdupq_n_s32(radius as i32 + 1 - i as i32));

                    sum_in = vaddq_s32(sum_in, src_pixel);
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
                    let casted_sum = vcvtq_f32_s32(sums);
                    let scaled_val = vcvtaq_s32_f32(vmulq_f32(casted_sum, mul_value));
                    store_u8_s32::<TAIL>(store_ld, scaled_val);

                    dst_ptr += stride as usize;

                    sums = vsubq_s32(sums, sum_out);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }

                    let stack_ptr = stacks0.as_mut_ptr().add(stack_start as usize * 4);
                    let stack_val = vld1q_s32(stack_ptr);
                    sum_out = vsubq_s32(sum_out, stack_val);

                    if yp < hm {
                        src_ptr += stride as usize; // stride
                        yp += 1;
                    }

                    let src_ld = pixels.slice.as_ptr().add(src_ptr);
                    let src_pixel = load_u8_s32_fast::<TAIL>(src_ld as *const u8);
                    vst1q_s32(stack_ptr, src_pixel);

                    sum_in = vaddq_s32(sum_in, src_pixel);
                    sums = vaddq_s32(sums, sum_in);

                    sp += 1;

                    if sp >= div {
                        sp = 0;
                    }
                    let stack_ptr = stacks0.as_mut_ptr().add(sp as usize * 4);
                    let stack_val = vld1q_s32(stack_ptr);

                    sum_out = vaddq_s32(sum_out, stack_val);
                    sum_in = vsubq_s32(sum_in, stack_val);
                }

                cx += TAIL;
            }
        }
    }
}
