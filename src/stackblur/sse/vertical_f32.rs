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
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) struct VerticalSseStackBlurPassFloat32<const CN: usize> {}

impl<const CN: usize> Default for VerticalSseStackBlurPassFloat32<CN> {
    fn default() -> Self {
        VerticalSseStackBlurPassFloat32::<CN> {}
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Default)]
pub(crate) struct SseF32x4(pub(crate) [f32; 4]);

#[target_feature(enable = "sse4.1")]
unsafe fn stack_blur_pass_vert_sse<const CN: usize>(
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
        let mut yp;
        let mut sp;
        let mut stack_start;
        let mut stacks = vec![SseF32x4::default(); div];

        let v_mul_value = _mm_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

        let hm = height - 1;
        let div = (radius * 2) + 1;

        let mut src_ptr;

        let min_x = (thread * width as usize / total_threads) * CN;
        let max_x = ((thread + 1) * width as usize / total_threads) * CN;

        let mut cx = min_x;

        while cx + 4 <= max_x {
            let mut sums0 = _mm_setzero_ps();
            let mut sum_in0 = _mm_setzero_ps();
            let mut sum_out0 = _mm_setzero_ps();

            let mut src_ptr = cx;

            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

            {
                let src_pixel0 = _mm_loadu_ps(src_ld as *const _);

                for i in 0..=radius {
                    let stack_ptr = stacks.as_mut_ptr().add(i as usize);

                    _mm_store_ps(stack_ptr as *mut _, src_pixel0);

                    let w = _mm_set1_ps((i as i32 + 1) as f32);
                    sums0 = _mm_opt_fmlaf_ps(sums0, src_pixel0, w);
                    sum_out0 = _mm_add_ps(sum_out0, src_pixel0);
                }
            }

            {
                for i in 1..=radius {
                    if i <= hm {
                        src_ptr += stride as usize;
                    }

                    let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize);
                    let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
                    let src_pixel0 = _mm_loadu_ps(src_ld as *const _);

                    _mm_store_ps(stack_ptr as *mut _, src_pixel0);

                    let w = _mm_set1_ps((radius as i32 + 1 - i as i32) as f32);
                    sums0 = _mm_opt_fmlaf_ps(sums0, src_pixel0, w);
                    sum_in0 = _mm_add_ps(sum_in0, src_pixel0);
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
                let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut _;

                let a0 = _mm_mul_ps(sums0, v_mul_value);

                _mm_storeu_ps(store_ld, a0);

                dst_ptr += stride as usize;

                sums0 = _mm_sub_ps(sums0, sum_out0);

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }

                let stack_ptr = stacks.as_mut_ptr().add(stack_start as usize);

                let stack_val0 = _mm_load_ps(stack_ptr as *const _);

                sum_out0 = _mm_sub_ps(sum_out0, stack_val0);

                if yp < hm {
                    src_ptr += stride as usize; // stride
                    yp += 1;
                }

                let src_ld = pixels.slice.as_ptr().add(src_ptr);

                let src_pixel0 = _mm_loadu_ps(src_ld as *const _);

                _mm_store_ps(stack_ptr as *mut _, src_pixel0);

                sum_in0 = _mm_add_ps(sum_in0, src_pixel0);
                sums0 = _mm_add_ps(sums0, sum_in0);

                sp += 1;

                if sp >= div {
                    sp = 0;
                }
                let stack_ptr = stacks.as_mut_ptr().add(sp as usize);

                let stack_val0 = _mm_load_ps(stack_ptr as *const _);

                sum_out0 = _mm_add_ps(sum_out0, stack_val0);
                sum_in0 = _mm_sub_ps(sum_in0, stack_val0);
            }

            cx += 4;
        }

        for x in cx..max_x {
            let mut sums = _mm_setzero_ps();
            let mut sum_in = _mm_setzero_ps();
            let mut sum_out = _mm_setzero_ps();

            src_ptr = x;

            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const f32;

            let src_pixel = _mm_load_ss(src_ld);

            for i in 0..=radius {
                let stack_ptr = stacks.as_mut_ptr().add(i as usize);
                _mm_store_ps(stack_ptr.cast(), src_pixel);
                sums = _mm_opt_fmlaf_ps(sums, src_pixel, _mm_set1_ps((i + 1) as f32));
                sum_out = _mm_add_ps(sum_out, src_pixel);
            }

            for i in 1..=radius {
                if i <= hm {
                    src_ptr += stride as usize;
                }

                let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize);
                let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const f32;
                let src_pixel = _mm_load_ss(src_ld);
                _mm_store_ps(stack_ptr.cast(), src_pixel);
                sums = _mm_opt_fmlaf_ps(sums, src_pixel, _mm_set1_ps((radius + 1 - i) as f32));

                sum_in = _mm_add_ps(sum_in, src_pixel);
            }

            sp = radius;
            yp = radius;
            if yp > hm {
                yp = hm;
            }
            src_ptr = cx + yp as usize * stride as usize;
            let mut dst_ptr = cx;
            for _ in 0..height {
                let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut f32;
                let blurred = _mm_mul_ps(sums, v_mul_value);
                _mm_store_ss(store_ld, blurred);

                dst_ptr += stride as usize;

                sums = _mm_sub_ps(sums, sum_out);

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }

                let stack_ptr = stacks.as_mut_ptr().add(stack_start as usize);
                let stack_val = _mm_load_ps(stack_ptr as *const f32);
                sum_out = _mm_sub_ps(sum_out, stack_val);

                if yp < hm {
                    src_ptr += stride as usize; // stride
                    yp += 1;
                }

                let src_ld = pixels.slice.as_ptr().add(src_ptr);
                let src_pixel = _mm_load_ss(src_ld as *const f32);
                _mm_store_ps(stack_ptr.cast(), src_pixel);

                sum_in = _mm_add_ps(sum_in, src_pixel);
                sums = _mm_add_ps(sums, sum_in);

                sp += 1;

                if sp >= div {
                    sp = 0;
                }
                let stack_ptr = stacks.as_mut_ptr().add(sp as usize);
                let stack_val = _mm_load_ps(stack_ptr as *const f32);

                sum_out = _mm_add_ps(sum_out, stack_val);
                sum_in = _mm_sub_ps(sum_in, stack_val);
            }
        }
    }
}

impl<const CN: usize> StackBlurWorkingPass<f32, CN> for VerticalSseStackBlurPassFloat32<CN> {
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
            stack_blur_pass_vert_sse::<CN>(
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
