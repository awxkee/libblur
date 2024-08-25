// Copyright (c) Radzivon Bartoshyk. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::mul_table::{MUL_TABLE_STACK_BLUR, SHR_TABLE_STACK_BLUR};
use crate::stack_blur::StackBlurPass;
use crate::unsafe_slice::UnsafeSlice;
use crate::wasm32::utils::{
    i32x4_pack_trunc_i64x2, load_u8_s32_fast, u16x8_pack_trunc_u8x16, u32x4_pack_trunc_u16x8,
    w_store_u8x8_m4,
};
use std::arch::wasm32::*;

pub fn stack_blur_pass_wasm_i32<const COMPONENTS: usize>(
    pixels: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    pass: StackBlurPass,
    thread: usize,
    total_threads: usize,
) {
    unsafe {
        stack_blur_pass_impl::<COMPONENTS>(
            pixels,
            stride,
            width,
            height,
            radius,
            pass,
            thread,
            total_threads,
        );
    }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn stack_blur_pass_impl<const COMPONENTS: usize>(
    pixels: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    pass: StackBlurPass,
    thread: usize,
    total_threads: usize,
) {
    let div = ((radius * 2) + 1) as usize;
    let (mut xp, mut yp);
    let mut sp;
    let mut stack_start;
    let mut stacks = vec![0i32; 4 * div];

    let mut sums: v128;
    let mut sum_in: v128;
    let mut sum_out: v128;

    let wm = width - 1;
    let hm = height - 1;
    let div = (radius * 2) + 1;
    let mul_sum = i64x2_splat(MUL_TABLE_STACK_BLUR[radius as usize] as i64);
    let shr_sum = SHR_TABLE_STACK_BLUR[radius as usize] as u32;

    let mut src_ptr;
    let mut dst_ptr;

    if pass == StackBlurPass::HORIZONTAL {
        let min_y = thread * height as usize / total_threads;
        let max_y = (thread + 1) * height as usize / total_threads;

        for y in min_y..max_y {
            sums = i32x4_splat(0i32);
            sum_in = i32x4_splat(0i32);
            sum_out = i32x4_splat(0i32);

            src_ptr = stride as usize * y; // start of line (0,y)

            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
            let src_pixel = load_u8_s32_fast::<COMPONENTS>(src_ld as *const u8);

            for i in 0..=radius {
                let stack_value = stacks.as_mut_ptr().add(i as usize * 4);
                v128_store(stack_value as *mut v128, src_pixel);
                sums = i32x4_add(sums, i32x4_mul(src_pixel, i32x4_splat(i as i32 + 1)));
                sum_out = i32x4_add(sum_out, src_pixel);
            }

            for i in 1..=radius {
                if i <= wm {
                    src_ptr += COMPONENTS;
                }
                let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4);
                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                let src_pixel = load_u8_s32_fast::<COMPONENTS>(src_ld as *const u8);
                v128_store(stack_ptr as *mut v128, src_pixel);
                sums = i32x4_add(
                    sums,
                    i32x4_mul(src_pixel, i32x4_splat(radius as i32 + 1 - i as i32)),
                );

                sum_in = i32x4_add(sum_in, src_pixel);
            }

            sp = radius;
            xp = radius;
            if xp > wm {
                xp = wm;
            }

            src_ptr = COMPONENTS * xp as usize + y * stride as usize;
            dst_ptr = y * stride as usize;
            for _ in 0..width {
                let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut u8;
                let blurred_lo =
                    i64x2_shr(i64x2_mul(i64x2_extend_low_i32x4(sums), mul_sum), shr_sum);
                let blurred_hi =
                    i64x2_shr(i64x2_mul(i64x2_extend_high_i32x4(sums), mul_sum), shr_sum);
                let blurred_i32 = i32x4_pack_trunc_i64x2(blurred_lo, blurred_hi);
                let prepared_u16 = u32x4_pack_trunc_u16x8(blurred_i32, blurred_i32);
                let blurred = u16x8_pack_trunc_u8x16(prepared_u16, prepared_u16);

                w_store_u8x8_m4::<COMPONENTS>(store_ld, blurred);
                dst_ptr += COMPONENTS;

                sums = i32x4_sub(sums, sum_out);

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack = stacks.as_mut_ptr().add(stack_start as usize * 4);

                let stack_val = v128_load(stack as *const v128);

                sum_out = i32x4_sub(sum_out, stack_val);

                if xp < wm {
                    src_ptr += COMPONENTS;
                    xp += 1;
                }

                let src_ld = pixels.slice.as_ptr().add(src_ptr);
                let src_pixel = load_u8_s32_fast::<COMPONENTS>(src_ld as *const u8);
                v128_store(stack as *mut v128, src_pixel);

                sum_in = i32x4_add(sum_in, src_pixel);
                sums = i32x4_add(sums, sum_in);

                sp += 1;
                if sp >= div {
                    sp = 0;
                }
                let stack = stacks.as_mut_ptr().add(sp as usize * 4);
                let stack_val = v128_load(stack as *const v128);

                sum_out = i32x4_add(sum_out, stack_val);
                sum_in = i32x4_add(sum_in, stack_val);
            }
        }
    } else if pass == StackBlurPass::VERTICAL {
        let min_x = thread * width as usize / total_threads;
        let max_x = (thread + 1) * width as usize / total_threads;

        for x in min_x..max_x {
            sums = i32x4_splat(0i32);
            sum_in = i32x4_splat(0i32);
            sum_out = i32x4_splat(0i32);

            src_ptr = COMPONENTS * x; // x,0

            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

            let src_pixel = load_u8_s32_fast::<COMPONENTS>(src_ld as *const u8);

            for i in 0..=radius {
                let stack_ptr = stacks.as_mut_ptr().add(i as usize * 4);
                v128_store(stack_ptr as *mut v128, src_pixel);
                sums = i32x4_add(sums, i32x4_mul(src_pixel, i32x4_splat(i as i32 + 1)));
                sum_out = i32x4_add(sum_out, src_pixel);
            }

            for i in 1..=radius {
                if i <= hm {
                    src_ptr += stride as usize;
                }

                let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4);
                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                let src_pixel = load_u8_s32_fast::<COMPONENTS>(src_ld as *const u8);
                v128_store(stack_ptr as *mut v128, src_pixel);
                sums = i32x4_add(
                    sums,
                    i32x4_mul(src_pixel, i32x4_splat(radius as i32 + 1 - i as i32)),
                );

                sum_in = i32x4_add(sum_in, src_pixel);
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
                let blurred_lo =
                    i64x2_shr(i64x2_mul(i64x2_extend_low_i32x4(sums), mul_sum), shr_sum);
                let blurred_hi =
                    i64x2_shr(i64x2_mul(i64x2_extend_high_i32x4(sums), mul_sum), shr_sum);
                let blurred_i32 = i32x4_pack_trunc_i64x2(blurred_lo, blurred_hi);
                let prepared_u16 = u32x4_pack_trunc_u16x8(blurred_i32, blurred_i32);
                let blurred = u16x8_pack_trunc_u8x16(prepared_u16, prepared_u16);
                w_store_u8x8_m4::<COMPONENTS>(store_ld, blurred);

                dst_ptr += stride as usize;

                sums = i32x4_sub(sums, sum_out);

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }

                let stack_ptr = stacks.as_mut_ptr().add(stack_start as usize * 4);
                let stack_val = v128_load(stack_ptr as *const v128);
                sum_out = i32x4_sub(sum_out, stack_val);

                if yp < hm {
                    src_ptr += stride as usize; // stride
                    yp += 1;
                }

                let src_ld = pixels.slice.as_ptr().add(src_ptr);
                let src_pixel = load_u8_s32_fast::<COMPONENTS>(src_ld as *const u8);
                v128_store(stack_ptr as *mut v128, src_pixel);

                sum_in = i32x4_add(sum_in, src_pixel);
                sums = i32x4_add(sums, sum_in);

                sp += 1;

                if sp >= div {
                    sp = 0;
                }
                let stack_ptr = stacks.as_mut_ptr().add(sp as usize * 4);
                let stack_val = v128_load(stack_ptr as *const v128);

                sum_out = i32x4_add(sum_out, stack_val);
                sum_in = i32x4_add(sum_in, stack_val);
            }
        }
    }
}
