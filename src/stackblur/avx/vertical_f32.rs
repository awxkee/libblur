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
use crate::stackblur::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use crate::util::ScratchBuffer;
use std::arch::x86_64::*;

#[repr(C, align(32))]
#[derive(Copy, Clone, Default)]
pub(crate) struct Avx2F32x8(pub(crate) [f32; 8]);

pub(crate) struct VerticalAvxStackBlurPassFloat32<const CN: usize> {}

impl<const CN: usize> Default for VerticalAvxStackBlurPassFloat32<CN> {
    fn default() -> Self {
        VerticalAvxStackBlurPassFloat32::<CN> {}
    }
}
#[target_feature(enable = "avx2", enable = "fma")]
fn stack_blur_pass_vert_avx<const CN: usize>(
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
        let mut scratch_buffer = ScratchBuffer::<Avx2F32x8, 512>::new(div * 2);
        let stacks = scratch_buffer.as_mut_slice();

        let hm = height - 1;
        let div = (radius * 2) + 1;

        let min_x = (thread * width as usize / total_threads) * CN;
        let max_x = ((thread + 1) * width as usize / total_threads) * CN;

        let mut cx = min_x;

        let v_mul_value = _mm256_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

        while cx + 16 <= max_x {
            let mut sums0 = _mm256_setzero_ps();
            let mut sums1 = _mm256_setzero_ps();
            let mut sum_in0 = _mm256_setzero_ps();
            let mut sum_in1 = _mm256_setzero_ps();
            let mut sum_out0 = _mm256_setzero_ps();
            let mut sum_out1 = _mm256_setzero_ps();

            let mut src_ptr = cx;

            {
                let src_pixel0 = _mm256_loadu_ps(pixels.get_ptr(src_ptr).cast());
                let src_pixel1 = _mm256_loadu_ps(pixels.get_ptr(src_ptr + 8).cast());

                for i in 0..=radius {
                    let stack_ptr = stacks.get_unchecked_mut(i as usize * 2..);
                    _mm256_store_ps(stack_ptr.as_mut_ptr().cast(), src_pixel0);
                    _mm256_store_ps(
                        stack_ptr.get_unchecked_mut(1..).as_mut_ptr().cast(),
                        src_pixel1,
                    );

                    let w = _mm256_set1_ps((i as i32 + 1) as f32);
                    sums0 = _mm256_fmadd_ps(src_pixel0, w, sums0);
                    sums1 = _mm256_fmadd_ps(src_pixel1, w, sums1);
                    sum_out0 = _mm256_add_ps(sum_out0, src_pixel0);
                    sum_out1 = _mm256_add_ps(sum_out1, src_pixel1);
                }
            }

            {
                for i in 1..=radius {
                    if i <= hm {
                        src_ptr += stride as usize;
                    }

                    let stack_ptr = stacks.get_unchecked_mut((i + radius) as usize * 2..);
                    let src_pixel0 = _mm256_loadu_ps(pixels.get_ptr(src_ptr).cast());
                    let src_pixel1 = _mm256_loadu_ps(pixels.get_ptr(src_ptr + 8).cast());

                    _mm256_store_ps(stack_ptr.as_mut_ptr().cast(), src_pixel0);
                    _mm256_store_ps(
                        stack_ptr.get_unchecked_mut(1..).as_mut_ptr().cast(),
                        src_pixel1,
                    );

                    let w = _mm256_set1_ps((radius as i32 + 1 - i as i32) as f32);
                    sums0 = _mm256_fmadd_ps(src_pixel0, w, sums0);
                    sums1 = _mm256_fmadd_ps(src_pixel1, w, sums1);
                    sum_in0 = _mm256_add_ps(sum_in0, src_pixel0);
                    sum_in1 = _mm256_add_ps(sum_in1, src_pixel1);
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
                _mm256_storeu_ps(pixels.get_ptr(dst_ptr), _mm256_mul_ps(sums0, v_mul_value));
                _mm256_storeu_ps(
                    pixels.get_ptr(dst_ptr + 8),
                    _mm256_mul_ps(sums1, v_mul_value),
                );

                dst_ptr += stride as usize;

                sums0 = _mm256_sub_ps(sums0, sum_out0);
                sums1 = _mm256_sub_ps(sums1, sum_out1);

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }

                let stack_ptr = stacks.get_unchecked_mut(stack_start as usize * 2..);
                let stack_val0 = _mm256_load_ps(stack_ptr.as_mut_ptr().cast());
                let stack_val1 = _mm256_load_ps(stack_ptr.get_unchecked(1..).as_ptr().cast());

                sum_out0 = _mm256_sub_ps(sum_out0, stack_val0);
                sum_out1 = _mm256_sub_ps(sum_out1, stack_val1);

                if yp < hm {
                    src_ptr += stride as usize;
                    yp += 1;
                }

                let src_pixel0 = _mm256_loadu_ps(pixels.get_ptr(src_ptr).cast());
                let src_pixel1 = _mm256_loadu_ps(pixels.get_ptr(src_ptr + 8).cast());

                _mm256_store_ps(stack_ptr.as_mut_ptr().cast(), src_pixel0);
                _mm256_store_ps(
                    stack_ptr.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    src_pixel1,
                );

                sum_in0 = _mm256_add_ps(sum_in0, src_pixel0);
                sum_in1 = _mm256_add_ps(sum_in1, src_pixel1);
                sums0 = _mm256_add_ps(sums0, sum_in0);
                sums1 = _mm256_add_ps(sums1, sum_in1);

                sp += 1;
                if sp >= div {
                    sp = 0;
                }

                let stack_ptr = stacks.get_unchecked_mut(sp as usize * 2..);
                let stack_val0 = _mm256_load_ps(stack_ptr.as_ptr().cast());
                let stack_val1 = _mm256_load_ps(stack_ptr.get_unchecked(1..).as_ptr().cast());

                sum_out0 = _mm256_add_ps(sum_out0, stack_val0);
                sum_out1 = _mm256_add_ps(sum_out1, stack_val1);
                sum_in0 = _mm256_sub_ps(sum_in0, stack_val0);
                sum_in1 = _mm256_sub_ps(sum_in1, stack_val1);
            }

            cx += 16;
        }

        while cx + 8 <= max_x {
            let mut sums0 = _mm256_setzero_ps();
            let mut sum_in0 = _mm256_setzero_ps();
            let mut sum_out0 = _mm256_setzero_ps();

            let mut src_ptr = cx;

            {
                let src_pixel0 = _mm256_loadu_ps(pixels.get_ptr(src_ptr));

                for i in 0..=radius {
                    let stack_ptr = stacks.get_unchecked_mut(i as usize..);

                    _mm256_store_ps(stack_ptr.as_mut_ptr().cast(), src_pixel0);

                    let w = _mm256_set1_ps((i as i32 + 1) as f32);
                    sums0 = _mm256_fmadd_ps(src_pixel0, w, sums0);
                    sum_out0 = _mm256_add_ps(sum_out0, src_pixel0);
                }
            }

            {
                for i in 1..=radius {
                    if i <= hm {
                        src_ptr += stride as usize;
                    }

                    let stack_ptr = stacks.get_unchecked_mut((i + radius) as usize..);
                    let src_pixel0 = _mm256_loadu_ps(pixels.get_ptr(src_ptr));

                    _mm256_store_ps(stack_ptr.as_mut_ptr().cast(), src_pixel0);

                    let w = _mm256_set1_ps((radius as i32 + 1 - i as i32) as f32);
                    sums0 = _mm256_fmadd_ps(src_pixel0, w, sums0);
                    sum_in0 = _mm256_add_ps(sum_in0, src_pixel0);
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
                let a0 = _mm256_mul_ps(sums0, v_mul_value);

                _mm256_storeu_ps(pixels.get_ptr(dst_ptr), a0);

                dst_ptr += stride as usize;

                sums0 = _mm256_sub_ps(sums0, sum_out0);

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }

                let stack_ptr = stacks.get_unchecked_mut(stack_start as usize..);

                let stack_val0 = _mm256_load_ps(stack_ptr.as_ptr().cast());

                sum_out0 = _mm256_sub_ps(sum_out0, stack_val0);

                if yp < hm {
                    src_ptr += stride as usize; // stride
                    yp += 1;
                }

                let src_pixel0 = _mm256_loadu_ps(pixels.get_ptr(src_ptr));

                _mm256_store_ps(stack_ptr.as_mut_ptr().cast(), src_pixel0);

                sum_in0 = _mm256_add_ps(sum_in0, src_pixel0);
                sums0 = _mm256_add_ps(sums0, sum_in0);

                sp += 1;

                if sp >= div {
                    sp = 0;
                }
                let stack_ptr = stacks.get_unchecked(sp as usize..);

                let stack_val0 = _mm256_load_ps(stack_ptr.as_ptr().cast());

                sum_out0 = _mm256_add_ps(sum_out0, stack_val0);
                sum_in0 = _mm256_sub_ps(sum_in0, stack_val0);
            }

            cx += 8;
        }

        while cx < max_x {
            let mut sums0 = _mm_setzero_ps();
            let mut sum_in0 = _mm_setzero_ps();
            let mut sum_out0 = _mm_setzero_ps();

            let mut src_ptr = cx; // x,0

            {
                let src_pixel0 = _mm_load_ss(pixels.get_ptr(src_ptr));

                for i in 0..=radius {
                    let stack_ptr = stacks.get_unchecked_mut(i as usize..);

                    _mm_store_ps(stack_ptr.as_mut_ptr().cast(), src_pixel0);

                    let w = _mm_set1_ps((i as i32 + 1) as f32);
                    sums0 = _mm_fmadd_ps(src_pixel0, w, sums0);
                    sum_out0 = _mm_add_ps(sum_out0, src_pixel0);
                }
            }

            {
                for i in 1..=radius {
                    if i <= hm {
                        src_ptr += stride as usize;
                    }

                    let stack_ptr = stacks.get_unchecked_mut((i + radius) as usize..);
                    let src_pixel0 = _mm_load_ss(pixels.get_ptr(src_ptr));

                    _mm_storeu_ps(stack_ptr.as_mut_ptr().cast(), src_pixel0);

                    let w = _mm_set1_ps((radius as i32 + 1 - i as i32) as f32);
                    sums0 = _mm_fmadd_ps(src_pixel0, w, sums0);
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
                let a0 = _mm_mul_ps(sums0, _mm256_castps256_ps128(v_mul_value));

                _mm_store_ss(pixels.get_ptr(dst_ptr), a0);

                dst_ptr += stride as usize;

                sums0 = _mm_sub_ps(sums0, sum_out0);

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }

                let stack_ptr = stacks.get_unchecked_mut(stack_start as usize..);

                let stack_val0 = _mm_load_ps(stack_ptr.as_ptr().cast());

                sum_out0 = _mm_sub_ps(sum_out0, stack_val0);

                if yp < hm {
                    src_ptr += stride as usize; // stride
                    yp += 1;
                }

                let src_pixel0 = _mm_load_ss(pixels.get_ptr(src_ptr));

                _mm_store_ps(stack_ptr.as_mut_ptr().cast(), src_pixel0);

                sum_in0 = _mm_add_ps(sum_in0, src_pixel0);
                sums0 = _mm_add_ps(sums0, sum_in0);

                sp += 1;

                if sp >= div {
                    sp = 0;
                }
                let stack_ptr = stacks.get_unchecked(sp as usize..);

                let stack_val0 = _mm_load_ps(stack_ptr.as_ptr().cast());

                sum_out0 = _mm_add_ps(sum_out0, stack_val0);
                sum_in0 = _mm_sub_ps(sum_in0, stack_val0);
            }

            cx += 1;
        }
    }
}
impl<const CN: usize> StackBlurWorkingPass<f32, CN> for VerticalAvxStackBlurPassFloat32<CN> {
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
            stack_blur_pass_vert_avx::<CN>(
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
