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
use crate::sse::utils::{_mm_packus_epi64, store_u8_s32};
use crate::sse::{
    __mm128ix2, _mm_add_epi64x2, _mm_load_epi64x2, _mm_mul_n_epi64x2, _mm_set1_epi64x2,
    _mm_store_epi64x2, _mm_sub_epi64x2, load_u8_s64x2_fast,
};
use crate::stack_blur::StackBlurPass;
use crate::unsafe_slice::UnsafeSlice;
use erydanos::_mm_mul_epi64;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn stack_blur_pass_sse_i64<const COMPONENTS: usize>(
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
        let div = ((radius * 2) + 1) as usize;
        let (mut xp, mut yp);
        let mut sp;
        let mut stack_start;
        let mut stacks = vec![0i64; 4 * div * total_threads];

        let mut sums: __mm128ix2;
        let mut sum_in: __mm128ix2;
        let mut sum_out: __mm128ix2;

        let wm = width - 1;
        let hm = height - 1;
        let div = (radius * 2) + 1;
        let mul_sum = _mm_set1_epi64x(MUL_TABLE_STACK_BLUR[radius as usize] as i64);
        let shr_sum = _mm_setr_epi32(SHR_TABLE_STACK_BLUR[radius as usize], 0i32, 0i32, 0i32);

        let mut src_ptr;
        let mut dst_ptr;

        if pass == StackBlurPass::HORIZONTAL {
            let min_y = thread * height as usize / total_threads;
            let max_y = (thread + 1) * height as usize / total_threads;

            for y in min_y..max_y {
                sums = _mm_set1_epi64x2(0i64);
                sum_in = _mm_set1_epi64x2(0i64);
                sum_out = _mm_set1_epi64x2(0i64);

                src_ptr = stride as usize * y; // start of line (0,y)

                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                let src_pixel = load_u8_s64x2_fast::<COMPONENTS>(src_ld as *const u8);

                for i in 0..=radius {
                    let stack_value = stacks.as_mut_ptr().add(i as usize * 4);
                    _mm_store_epi64x2(stack_value, src_pixel);
                    sums = _mm_add_epi64x2(sums, _mm_mul_n_epi64x2(src_pixel, i as i64 + 1i64));
                    sum_out = _mm_add_epi64x2(sum_out, src_pixel);
                }

                for i in 1..=radius {
                    if i <= wm {
                        src_ptr += COMPONENTS;
                    }
                    let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4);
                    let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                    let src_pixel = load_u8_s64x2_fast::<COMPONENTS>(src_ld as *const u8);
                    _mm_store_epi64x2(stack_ptr, src_pixel);
                    sums = _mm_add_epi64x2(
                        sums,
                        _mm_mul_n_epi64x2(src_pixel, radius as i64 + 1i64 - i as i64),
                    );

                    sum_in = _mm_add_epi64x2(sum_in, src_pixel);
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
                    let blurred_hi = _mm_srl_epi64(_mm_mul_epi64(sums.1, mul_sum), shr_sum);
                    let blurred_lo = _mm_srl_epi64(_mm_mul_epi64(sums.0, mul_sum), shr_sum);
                    let blurred = _mm_packus_epi64(blurred_lo, blurred_hi);
                    store_u8_s32::<COMPONENTS>(store_ld, blurred);
                    dst_ptr += COMPONENTS;

                    sums = _mm_sub_epi64x2(sums, sum_out);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }
                    let stack = stacks.as_mut_ptr().add(stack_start as usize * 4);

                    let stack_val = _mm_load_epi64x2(stack as *const i64);

                    sum_out = _mm_sub_epi64x2(sum_out, stack_val);

                    if xp < wm {
                        src_ptr += COMPONENTS;
                        xp += 1;
                    }

                    let src_ld = pixels.slice.as_ptr().add(src_ptr);
                    let src_pixel = load_u8_s64x2_fast::<COMPONENTS>(src_ld as *const u8);
                    _mm_store_epi64x2(stack, src_pixel);

                    sum_in = _mm_add_epi64x2(sum_in, src_pixel);
                    sums = _mm_add_epi64x2(sums, sum_in);

                    sp += 1;
                    if sp >= div {
                        sp = 0;
                    }
                    let stack = stacks.as_mut_ptr().add(sp as usize * 4);
                    let stack_val = _mm_load_epi64x2(stack as *const i64);

                    sum_out = _mm_add_epi64x2(sum_out, stack_val);
                    sum_in = _mm_sub_epi64x2(sum_in, stack_val);
                }
            }
        } else if pass == StackBlurPass::VERTICAL {
            let min_x = thread * width as usize / total_threads;
            let max_x = (thread + 1) * width as usize / total_threads;

            for x in min_x..max_x {
                sums = _mm_set1_epi64x2(0i64);
                sum_in = _mm_set1_epi64x2(0i64);
                sum_out = _mm_set1_epi64x2(0i64);

                src_ptr = COMPONENTS * x; // x,0

                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

                let src_pixel = load_u8_s64x2_fast::<COMPONENTS>(src_ld as *const u8);

                for i in 0..=radius {
                    let stack_ptr = stacks.as_mut_ptr().add(i as usize * 4);
                    _mm_store_epi64x2(stack_ptr, src_pixel);
                    sums = _mm_add_epi64x2(sums, _mm_mul_n_epi64x2(src_pixel, i as i64 + 1i64));
                    sum_out = _mm_add_epi64x2(sum_out, src_pixel);
                }

                for i in 1..=radius {
                    if i <= hm {
                        src_ptr += stride as usize;
                    }

                    let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4);
                    let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                    let src_pixel = load_u8_s64x2_fast::<COMPONENTS>(src_ld as *const u8);
                    _mm_store_epi64x2(stack_ptr, src_pixel);
                    sums = _mm_add_epi64x2(
                        sums,
                        _mm_mul_n_epi64x2(src_pixel, radius as i64 + 1i64 - i as i64),
                    );

                    sum_in = _mm_add_epi64x2(sum_in, src_pixel);
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
                    let blurred_hi = _mm_srl_epi64(_mm_mul_epi64(sums.1, mul_sum), shr_sum);
                    let blurred_lo = _mm_srl_epi64(_mm_mul_epi64(sums.0, mul_sum), shr_sum);
                    let blurred = _mm_packus_epi64(blurred_lo, blurred_hi);
                    store_u8_s32::<COMPONENTS>(store_ld, blurred);

                    dst_ptr += stride as usize;

                    sums = _mm_sub_epi64x2(sums, sum_out);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }

                    let stack_ptr = stacks.as_mut_ptr().add(stack_start as usize * 4);
                    let stack_val = _mm_load_epi64x2(stack_ptr as *const i64);
                    sum_out = _mm_sub_epi64x2(sum_out, stack_val);

                    if yp < hm {
                        src_ptr += stride as usize; // stride
                        yp += 1;
                    }

                    let src_ld = pixels.slice.as_ptr().add(src_ptr);
                    let src_pixel = load_u8_s64x2_fast::<COMPONENTS>(src_ld as *const u8);
                    _mm_store_epi64x2(stack_ptr, src_pixel);

                    sum_in = _mm_add_epi64x2(sum_in, src_pixel);
                    sums = _mm_add_epi64x2(sums, sum_in);

                    sp += 1;

                    if sp >= div {
                        sp = 0;
                    }
                    let stack_ptr = stacks.as_mut_ptr().add(sp as usize * 4);
                    let stack_val = _mm_load_epi64x2(stack_ptr as *const i64);

                    sum_out = _mm_add_epi64x2(sum_out, stack_val);
                    sum_in = _mm_sub_epi64x2(sum_in, stack_val);
                }
            }
        }
    }
}
