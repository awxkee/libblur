/*
 * // Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::sse::{load_f32, store_f32};
use crate::stackblur::avx::vertical_f32::Avx2F32x8;
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use crate::util::ScratchBuffer;
use std::arch::x86_64::*;

pub(crate) struct HorizontalAvxStackBlurPassFloat32<const CN: usize> {}

impl<const CN: usize> Default for HorizontalAvxStackBlurPassFloat32<CN> {
    fn default() -> Self {
        HorizontalAvxStackBlurPassFloat32::<CN> {}
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
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
        let v_mul_value = _mm256_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));
        let mut xp;
        let mut sp;
        let mut stack_start;
        let mut scratch_buffer = ScratchBuffer::<Avx2F32x8, 512>::new(div * 2);
        let stacks = scratch_buffer.as_mut_slice();

        let wm = width - 1;
        let div = (radius * 2) + 1;

        let min_y = thread * height as usize / total_threads;
        let max_y = (thread + 1) * height as usize / total_threads;

        let mut cy = min_y;

        while cy + 4 <= max_y {
            let mut sums0 = _mm256_setzero_ps();
            let mut sums1 = _mm256_setzero_ps();

            let mut sum_in0 = _mm256_setzero_ps();
            let mut sum_in1 = _mm256_setzero_ps();

            let mut sum_out0 = _mm256_setzero_ps();
            let mut sum_out1 = _mm256_setzero_ps();

            let mut src_ptr0 = stride as usize * cy;
            let mut src_ptr1 = stride as usize * (cy + 1);
            let mut src_ptr2 = stride as usize * (cy + 2);
            let mut src_ptr3 = stride as usize * (cy + 3);

            let src_pixel0 = load_f32::<CN>(pixels.get_ptr(src_ptr0));
            let src_pixel1 = load_f32::<CN>(pixels.get_ptr(src_ptr1));
            let src_pixel2 = load_f32::<CN>(pixels.get_ptr(src_ptr2));
            let src_pixel3 = load_f32::<CN>(pixels.get_ptr(src_ptr3));

            let x0 = _mm256_setr_m128(src_pixel0, src_pixel1);
            let x1 = _mm256_setr_m128(src_pixel2, src_pixel3);

            for i in 0..=radius {
                let stack_value = stacks.get_unchecked_mut(i as usize * 2..);
                _mm256_store_ps(stack_value.as_mut_ptr().cast(), x0);
                _mm256_store_ps(stack_value.get_unchecked_mut(1..).as_mut_ptr().cast(), x1);
                let w = _mm256_set1_ps((i + 1) as f32);
                sums0 = _mm256_fmadd_ps(x0, w, sums0);
                sums1 = _mm256_fmadd_ps(x1, w, sums1);
                sum_out0 = _mm256_add_ps(sum_out0, x0);
                sum_out1 = _mm256_add_ps(sum_out1, x1);
            }

            for i in 1..=radius {
                if i <= wm {
                    src_ptr0 += CN;
                    src_ptr1 += CN;
                    src_ptr2 += CN;
                    src_ptr3 += CN;
                }
                let stack_ptr = stacks.get_unchecked_mut((i + radius) as usize * 2..);
                let src_pixel0 = load_f32::<CN>(pixels.get_ptr(src_ptr0));
                let src_pixel1 = load_f32::<CN>(pixels.get_ptr(src_ptr1));
                let src_pixel2 = load_f32::<CN>(pixels.get_ptr(src_ptr2));
                let src_pixel3 = load_f32::<CN>(pixels.get_ptr(src_ptr3));

                let x0 = _mm256_setr_m128(src_pixel0, src_pixel1);
                let x1 = _mm256_setr_m128(src_pixel2, src_pixel3);

                _mm256_store_ps(stack_ptr.as_mut_ptr().cast(), x0);
                _mm256_store_ps(stack_ptr.get_unchecked_mut(1..).as_mut_ptr().cast(), x1);

                let w = _mm256_set1_ps((radius + 1 - i) as f32);
                sums0 = _mm256_fmadd_ps(x0, w, sums0);
                sums1 = _mm256_fmadd_ps(x1, w, sums1);

                sum_in0 = _mm256_add_ps(sum_in0, x0);
                sum_in1 = _mm256_add_ps(sum_in1, x1);
            }

            sp = radius;
            xp = radius;
            if xp > wm {
                xp = wm;
            }

            src_ptr0 = CN * xp as usize + cy * stride as usize;
            src_ptr1 = CN * xp as usize + (cy + 1) * stride as usize;
            src_ptr2 = CN * xp as usize + (cy + 2) * stride as usize;
            src_ptr3 = CN * xp as usize + (cy + 3) * stride as usize;
            let mut dst_ptr0 = cy * stride as usize;
            let mut dst_ptr1 = (cy + 1) * stride as usize;
            let mut dst_ptr2 = (cy + 2) * stride as usize;
            let mut dst_ptr3 = (cy + 3) * stride as usize;
            for _ in 0..width {
                let bx0 = _mm256_mul_ps(sums0, v_mul_value);
                let bx1 = _mm256_mul_ps(sums1, v_mul_value);

                store_f32::<CN>(pixels.get_ptr(dst_ptr0), _mm256_castps256_ps128(bx0));
                store_f32::<CN>(pixels.get_ptr(dst_ptr1), _mm256_extractf128_ps::<1>(bx0));
                store_f32::<CN>(pixels.get_ptr(dst_ptr2), _mm256_castps256_ps128(bx1));
                store_f32::<CN>(pixels.get_ptr(dst_ptr3), _mm256_extractf128_ps::<1>(bx1));

                dst_ptr0 += CN;
                dst_ptr1 += CN;
                dst_ptr2 += CN;
                dst_ptr3 += CN;

                sums0 = _mm256_sub_ps(sums0, sum_out0);
                sums1 = _mm256_sub_ps(sums1, sum_out1);

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack = stacks.get_unchecked_mut(stack_start as usize * 2..);

                let stack_val0 = _mm256_load_ps(stack.as_mut_ptr().cast());
                let stack_val1 = _mm256_load_ps(stack.get_unchecked_mut(1..).as_mut_ptr().cast());

                sum_out0 = _mm256_sub_ps(sum_out0, stack_val0);
                sum_out1 = _mm256_sub_ps(sum_out1, stack_val1);

                if xp < wm {
                    src_ptr0 += CN;
                    src_ptr1 += CN;
                    src_ptr2 += CN;
                    src_ptr3 += CN;
                    xp += 1;
                }

                let src_pixel0 = load_f32::<CN>(pixels.get_ptr(src_ptr0).cast());
                let src_pixel1 = load_f32::<CN>(pixels.get_ptr(src_ptr1).cast());
                let src_pixel2 = load_f32::<CN>(pixels.get_ptr(src_ptr2).cast());
                let src_pixel3 = load_f32::<CN>(pixels.get_ptr(src_ptr3).cast());

                let x0 = _mm256_setr_m128(src_pixel0, src_pixel1);
                let x1 = _mm256_setr_m128(src_pixel2, src_pixel3);

                _mm256_store_ps(stack.as_mut_ptr().cast(), x0);
                _mm256_store_ps(stack.get_unchecked_mut(1..).as_mut_ptr().cast(), x1);

                sum_in0 = _mm256_add_ps(sum_in0, x0);
                sum_in1 = _mm256_add_ps(sum_in1, x1);

                sums0 = _mm256_add_ps(sums0, sum_in0);
                sums1 = _mm256_add_ps(sums1, sum_in1);

                sp += 1;
                if sp >= div {
                    sp = 0;
                }
                let stack = stacks.get_unchecked(sp as usize * 2..);
                let stack_val0 = _mm256_load_ps(stack.as_ptr().cast());
                let stack_val1 = _mm256_load_ps(stack.get_unchecked(1..).as_ptr().cast());

                sum_out0 = _mm256_add_ps(sum_out0, stack_val0);
                sum_out1 = _mm256_add_ps(sum_out1, stack_val1);
                sum_in0 = _mm256_sub_ps(sum_in0, stack_val0);
                sum_in1 = _mm256_sub_ps(sum_in1, stack_val1);
            }

            cy += 4;
        }

        let v_mul_value = _mm_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

        let mut src_ptr;
        let mut dst_ptr;

        for y in cy..max_y {
            let mut sums = _mm_setzero_ps();
            let mut sum_in = _mm_setzero_ps();
            let mut sum_out = _mm_setzero_ps();

            src_ptr = stride as usize * y;

            let src_pixel = load_f32::<CN>(pixels.get_ptr(src_ptr));

            for i in 0..=radius {
                let stack_value = stacks.get_unchecked_mut(i as usize..);
                _mm_store_ps(stack_value.as_mut_ptr().cast(), src_pixel);
                sums = _mm_fmadd_ps(src_pixel, _mm_set1_ps((i + 1) as f32), sums);
                sum_out = _mm_add_ps(sum_out, src_pixel);
            }

            for i in 1..=radius {
                if i <= wm {
                    src_ptr += CN;
                }
                let stack_ptr = stacks.get_unchecked_mut((i + radius) as usize..);
                let src_ld = pixels.get_ptr(src_ptr) as *const f32;
                let src_pixel = load_f32::<CN>(src_ld);
                _mm_store_ps(stack_ptr.as_mut_ptr().cast(), src_pixel);
                sums = _mm_fmadd_ps(src_pixel, _mm_set1_ps((radius + 1 - i) as f32), sums);

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
                let stack = stacks.get_unchecked_mut(stack_start as usize..);

                let stack_val = _mm_load_ps(stack.as_mut_ptr().cast());

                sum_out = _mm_sub_ps(sum_out, stack_val);

                if xp < wm {
                    src_ptr += CN;
                    xp += 1;
                }

                let src_pixel = load_f32::<CN>(pixels.get_ptr(src_ptr));
                _mm_store_ps(stack.as_mut_ptr().cast(), src_pixel);

                sum_in = _mm_add_ps(sum_in, src_pixel);
                sums = _mm_add_ps(sums, sum_in);

                sp += 1;
                if sp >= div {
                    sp = 0;
                }
                let stack = stacks.get_unchecked(sp as usize..);
                let stack_val = _mm_load_ps(stack.as_ptr().cast());

                sum_out = _mm_add_ps(sum_out, stack_val);
                sum_in = _mm_sub_ps(sum_in, stack_val);
            }
        }
    }
}

impl<const CN: usize> StackBlurWorkingPass<f32, CN> for HorizontalAvxStackBlurPassFloat32<CN> {
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
