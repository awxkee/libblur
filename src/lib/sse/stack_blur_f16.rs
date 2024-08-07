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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use crate::sse::{load_f32_f16, store_f32_f16};
use crate::stack_blur::StackBlurPass;
use crate::unsafe_slice::UnsafeSlice;

pub fn stack_blur_pass_sse_f16<const COMPONENTS: usize>(
    pixels: &UnsafeSlice<f16>,
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
        let radius_scale: f32 = (1f64 / (radius * (radius + 2) - 1) as f64) as f32;
        let (mut xp, mut yp);
        let mut sp;
        let mut stack_start;
        let mut stacks = vec![0f32; 4 * div * total_threads];

        let mut sums: __m128;
        let mut sum_in: __m128;
        let mut sum_out: __m128;

        let wm = width - 1;
        let hm = height - 1;
        let div = (radius * 2) + 1;
        let v_scale = _mm_set1_ps(radius_scale);

        let mut src_ptr;
        let mut dst_ptr;

        if pass == StackBlurPass::HORIZONTAL {
            let min_y = thread * height as usize / total_threads;
            let max_y = (thread + 1) * height as usize / total_threads;

            for y in min_y..max_y {
                sums = _mm_set1_ps(0.);
                sum_in = _mm_set1_ps(0.);
                sum_out = _mm_set1_ps(0.);

                src_ptr = stride as usize * y; // start of line (0,y)

                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const f16;
                let src_pixel = load_f32_f16::<COMPONENTS>(src_ld);

                for i in 0..=radius {
                    let stack_value = stacks.as_mut_ptr().add(i as usize * 4);
                    _mm_storeu_ps(stack_value, src_pixel);
                    sums = _mm_add_ps(sums, _mm_mul_ps(src_pixel, _mm_set1_ps(i as f32 + 1f32)));
                    sum_out = _mm_add_ps(sum_out, src_pixel);
                }

                for i in 1..=radius {
                    if i <= wm {
                        src_ptr += COMPONENTS;
                    }
                    let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 4);
                    let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const f16;
                    let src_pixel = load_f32_f16::<COMPONENTS>(src_ld);
                    _mm_storeu_ps(stack_ptr, src_pixel);
                    sums = _mm_add_ps(
                        sums,
                        _mm_mul_ps(src_pixel, _mm_set1_ps(radius as f32 + 1f32 - i as f32)),
                    );

                    sum_in = _mm_add_ps(sum_in, src_pixel);
                }

                sp = radius;
                xp = radius;
                if xp > wm {
                    xp = wm;
                }

                src_ptr = COMPONENTS * xp as usize + y * stride as usize;
                dst_ptr = y * stride as usize;
                for _ in 0..width {
                    let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut f16;
                    let blurred = _mm_mul_ps(sums, v_scale);
                    store_f32_f16::<COMPONENTS>(store_ld, blurred);
                    dst_ptr += COMPONENTS;

                    sums = _mm_sub_ps(sums, sum_out);

                    stack_start = sp + div - radius;
                    if stack_start >= div {
                        stack_start -= div;
                    }
                    let stack = stacks.as_mut_ptr().add(stack_start as usize * 4);

                    let stack_val = _mm_loadu_ps(stack);

                    sum_out = _mm_sub_ps(sum_out, stack_val);

                    if xp < wm {
                        src_ptr += COMPONENTS;
                        xp += 1;
                    }

                    let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const f16;
                    let src_pixel = load_f32_f16::<COMPONENTS>(src_ld);
                    _mm_storeu_ps(stack, src_pixel);

                    sum_in = _mm_add_ps(sum_in, src_pixel);
                    sums = _mm_add_ps(sums, sum_in);

                    sp += 1;
                    if sp >= div {
                        sp = 0;
                    }
                    let stack = stacks.as_mut_ptr().add(sp as usize * 4);
                    let stack_val = _mm_loadu_ps(stack);

                    sum_out = _mm_add_ps(sum_out, stack_val);
                    sum_in = _mm_sub_ps(sum_in, stack_val);
                }
            }
        } else if pass == StackBlurPass::VERTICAL {
            let min_x = thread * width as usize / total_threads;
            let max_x = (thread + 1) * width as usize / total_threads;

            for x in min_x..max_x {
                sums = _mm_set1_ps(0.);
                sum_in = _mm_set1_ps(0.);
                sum_out = _mm_set1_ps(0.);

                src_ptr = COMPONENTS * x; // x,0

                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const f16;
                let src_pixel = load_f32_f16::<COMPONENTS>(src_ld);

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
                    let src_pixel = load_f32_f16::<COMPONENTS>(src_ld);
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
                src_ptr = COMPONENTS * x + yp as usize * stride as usize;
                dst_ptr = COMPONENTS * x;
                for _ in 0..height {
                    let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut f16;
                    let blurred = _mm_mul_ps(sums, v_scale);
                    store_f32_f16::<COMPONENTS>(store_ld, blurred);

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
                    let src_pixel = load_f32_f16::<COMPONENTS>(src_ld);
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
