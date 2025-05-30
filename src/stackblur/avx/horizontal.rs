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
use crate::sse::{load_u8_s32_fast, store_u8_s32};
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::{AsPrimitive, FromPrimitive};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Shr, Sub, SubAssign};

pub(crate) struct HorizontalAvxStackBlurPass<T, J, const CN: usize> {
    _phantom_t: PhantomData<T>,
    _phantom_j: PhantomData<J>,
}

impl<T, J, const CN: usize> Default for HorizontalAvxStackBlurPass<T, J, CN> {
    fn default() -> Self {
        HorizontalAvxStackBlurPass::<T, J, CN> {
            _phantom_t: Default::default(),
            _phantom_j: Default::default(),
        }
    }
}

#[repr(C, align(32))]
#[derive(Copy, Clone, Default, Debug)]
struct StackBlurCache([i32; 8]);

#[target_feature(enable = "avx2")]
unsafe fn avx_horiz_pass_impl<const CN: usize>(
    pixels: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    thread: usize,
    total_threads: usize,
) {
    let pixels: &UnsafeSlice<u8> = std::mem::transmute(pixels);
    let div = ((radius * 2) + 1) as usize;
    let mut xp;
    let mut sp;
    let mut stack_start;
    let mut stacks = vec![StackBlurCache::default(); div * 2];

    let v_mul_value = _mm256_set1_ps(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

    let wm = width - 1;
    let div = (radius * 2) + 1;

    let min_y = thread * height as usize / total_threads;
    let max_y = (thread + 1) * height as usize / total_threads;

    let mut yy = min_y;

    while yy + 4 < max_y {
        let mut sums0 = _mm256_setzero_si256();
        let mut sums1 = _mm256_setzero_si256();

        let mut sum_in0 = _mm256_setzero_si256();
        let mut sum_in1 = _mm256_setzero_si256();

        let mut sum_out0 = _mm256_setzero_si256();
        let mut sum_out1 = _mm256_setzero_si256();

        let mut src_ptr0 = stride as usize * yy;
        let mut src_ptr1 = stride as usize * (yy + 1);
        let mut src_ptr2 = stride as usize * (yy + 2);
        let mut src_ptr3 = stride as usize * (yy + 3);

        let src_ld0 = pixels.slice.as_ptr().add(src_ptr0) as *const i32;
        let src_ld1 = pixels.slice.as_ptr().add(src_ptr1) as *const i32;
        let src_ld2 = pixels.slice.as_ptr().add(src_ptr2) as *const i32;
        let src_ld3 = pixels.slice.as_ptr().add(src_ptr3) as *const i32;

        let src_pixel0 = load_u8_s32_fast::<CN>(src_ld0 as *const u8);
        let src_pixel1 = load_u8_s32_fast::<CN>(src_ld1 as *const u8);
        let src_pixel2 = load_u8_s32_fast::<CN>(src_ld2 as *const u8);
        let src_pixel3 = load_u8_s32_fast::<CN>(src_ld3 as *const u8);

        let px0 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(src_pixel0), src_pixel1);
        let px1 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(src_pixel2), src_pixel3);

        for i in 0..=radius {
            let stack_value = stacks.as_mut_ptr().add(i as usize * 2);
            _mm256_store_si256(stack_value as *mut _, px0);
            _mm256_store_si256(stack_value.add(1) as *mut _, px1);

            let w = _mm256_set1_epi32(i as i32 + 1);
            sums0 = _mm256_add_epi32(sums0, _mm256_madd_epi16(px0, w));
            sums1 = _mm256_add_epi32(sums1, _mm256_madd_epi16(px1, w));

            sum_out0 = _mm256_add_epi32(sum_out0, px0);
            sum_out1 = _mm256_add_epi32(sum_out1, px1);
        }

        for i in 1..=radius {
            if i <= wm {
                src_ptr0 += CN;
                src_ptr1 += CN;
            }
            let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 2);

            let src_pixel0 =
                load_u8_s32_fast::<CN>(pixels.slice.as_ptr().add(src_ptr0) as *const u8);
            let src_pixel1 =
                load_u8_s32_fast::<CN>(pixels.slice.as_ptr().add(src_ptr1) as *const u8);
            let src_pixel2 =
                load_u8_s32_fast::<CN>(pixels.slice.as_ptr().add(src_ptr2) as *const u8);
            let src_pixel3 =
                load_u8_s32_fast::<CN>(pixels.slice.as_ptr().add(src_ptr3) as *const u8);

            let px0 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(src_pixel0), src_pixel1);
            let px1 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(src_pixel2), src_pixel3);

            _mm256_store_si256(stack_ptr as *mut _, px0);
            _mm256_store_si256(stack_ptr.add(1) as *mut _, px1);

            let w = _mm256_set1_epi32(radius as i32 + 1 - i as i32);

            sums0 = _mm256_add_epi32(sums0, _mm256_madd_epi16(px0, w));
            sums1 = _mm256_add_epi32(sums1, _mm256_madd_epi16(px1, w));

            sum_in0 = _mm256_add_epi32(sum_in0, px0);
            sum_in1 = _mm256_add_epi32(sum_in1, px1);
        }

        sp = radius;
        xp = radius;
        if xp > wm {
            xp = wm;
        }

        src_ptr0 = CN * xp as usize + yy * stride as usize;
        src_ptr1 = CN * xp as usize + (yy + 1) * stride as usize;
        src_ptr2 = CN * xp as usize + (yy + 2) * stride as usize;
        src_ptr3 = CN * xp as usize + (yy + 3) * stride as usize;

        let mut dst_ptr0 = yy * stride as usize;
        let mut dst_ptr1 = (yy + 1) * stride as usize;
        let mut dst_ptr2 = (yy + 2) * stride as usize;
        let mut dst_ptr3 = (yy + 3) * stride as usize;

        for _ in 0..width {
            let store_ld0 = pixels.slice.as_ptr().add(dst_ptr0) as *mut u8;
            let store_ld1 = pixels.slice.as_ptr().add(dst_ptr1) as *mut u8;
            let store_ld2 = pixels.slice.as_ptr().add(dst_ptr2) as *mut u8;
            let store_ld3 = pixels.slice.as_ptr().add(dst_ptr3) as *mut u8;

            let o0 = _mm256_cvtepi32_ps(sums0);
            let o1 = _mm256_cvtepi32_ps(sums1);

            let r0 = _mm256_mul_ps(o0, v_mul_value);
            let r1 = _mm256_mul_ps(o1, v_mul_value);

            let cv0 = _mm256_cvtps_epi32(r0);
            let cv1 = _mm256_cvtps_epi32(r1);

            store_u8_s32::<CN>(store_ld0, _mm256_castsi256_si128(cv0));
            store_u8_s32::<CN>(store_ld1, _mm256_extracti128_si256::<1>(cv0));
            store_u8_s32::<CN>(store_ld2, _mm256_castsi256_si128(cv1));
            store_u8_s32::<CN>(store_ld3, _mm256_extracti128_si256::<1>(cv1));

            dst_ptr0 += CN;
            dst_ptr1 += CN;
            dst_ptr2 += CN;
            dst_ptr3 += CN;

            sums0 = _mm256_sub_epi32(sums0, sum_out0);
            sums1 = _mm256_sub_epi32(sums1, sum_out1);

            stack_start = sp + div - radius;
            if stack_start >= div {
                stack_start -= div;
            }
            let stack = stacks.as_mut_ptr().add(stack_start as usize * 2);

            let stack_val0 = _mm256_load_si256(stack as *const _);
            let stack_val1 = _mm256_load_si256(stack.add(1) as *const _);

            sum_out0 = _mm256_sub_epi32(sum_out0, stack_val0);
            sum_out1 = _mm256_sub_epi32(sum_out1, stack_val1);

            if xp < wm {
                src_ptr0 += CN;
                src_ptr1 += CN;
                src_ptr2 += CN;
                src_ptr3 += CN;
                xp += 1;
            }

            let src_ld0 = pixels.slice.as_ptr().add(src_ptr0);
            let src_ld1 = pixels.slice.as_ptr().add(src_ptr1);
            let src_ld2 = pixels.slice.as_ptr().add(src_ptr2);
            let src_ld3 = pixels.slice.as_ptr().add(src_ptr3);

            let src_pixel0 = load_u8_s32_fast::<CN>(src_ld0 as *const u8);
            let src_pixel1 = load_u8_s32_fast::<CN>(src_ld1 as *const u8);
            let src_pixel2 = load_u8_s32_fast::<CN>(src_ld2 as *const u8);
            let src_pixel3 = load_u8_s32_fast::<CN>(src_ld3 as *const u8);

            let px0 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(src_pixel0), src_pixel1);
            let px1 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(src_pixel2), src_pixel3);

            _mm256_store_si256(stack as *mut _, px0);
            _mm256_store_si256(stack.add(1) as *mut _, px1);

            sum_in0 = _mm256_add_epi32(sum_in0, px0);
            sum_in1 = _mm256_add_epi32(sum_in1, px1);

            sums0 = _mm256_add_epi32(sums0, sum_in0);
            sums1 = _mm256_add_epi32(sums1, sum_in1);

            sp += 1;
            if sp >= div {
                sp = 0;
            }
            let stack = stacks.as_mut_ptr().add(sp as usize * 2);
            let stack_val0 = _mm256_loadu_si256(stack as *const __m256i);
            let stack_val1 = _mm256_loadu_si256(stack.add(1) as *const __m256i);

            sum_out0 = _mm256_add_epi32(sum_out0, stack_val0);
            sum_out1 = _mm256_add_epi32(sum_out1, stack_val1);

            sum_in0 = _mm256_sub_epi32(sum_in0, stack_val0);
            sum_in1 = _mm256_sub_epi32(sum_in1, stack_val1);
        }

        yy += 4;
    }

    while yy + 2 < max_y {
        let mut sums0 = _mm256_setzero_si256();

        let mut sum_in0 = _mm256_setzero_si256();

        let mut sum_out0 = _mm256_setzero_si256();

        let mut src_ptr0 = stride as usize * yy;
        let mut src_ptr1 = stride as usize * (yy + 1);

        let src_ld0 = pixels.slice.as_ptr().add(src_ptr0) as *const i32;
        let src_ld1 = pixels.slice.as_ptr().add(src_ptr1) as *const i32;

        let src_pixel0 = load_u8_s32_fast::<CN>(src_ld0 as *const u8);
        let src_pixel1 = load_u8_s32_fast::<CN>(src_ld1 as *const u8);

        let px0 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(src_pixel0), src_pixel1);

        for i in 0..=radius {
            let stack_value = stacks.as_mut_ptr().add(i as usize * 2);
            _mm256_store_si256(stack_value as *mut _, px0);

            let w = _mm256_set1_epi32(i as i32 + 1);
            sums0 = _mm256_add_epi32(sums0, _mm256_madd_epi16(px0, w));

            sum_out0 = _mm256_add_epi32(sum_out0, px0);
        }

        for i in 1..=radius {
            if i <= wm {
                src_ptr0 += CN;
                src_ptr1 += CN;
            }
            let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 2);
            let src_pixel0 =
                load_u8_s32_fast::<CN>(pixels.slice.as_ptr().add(src_ptr0) as *const u8);
            let src_pixel1 =
                load_u8_s32_fast::<CN>(pixels.slice.as_ptr().add(src_ptr1) as *const u8);

            let px0 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(src_pixel0), src_pixel1);

            _mm256_store_si256(stack_ptr as *mut _, px0);

            let w = _mm256_set1_epi32(radius as i32 + 1 - i as i32);

            sums0 = _mm256_add_epi32(sums0, _mm256_madd_epi16(px0, w));

            sum_in0 = _mm256_add_epi32(sum_in0, px0);
        }

        sp = radius;
        xp = radius;
        if xp > wm {
            xp = wm;
        }

        src_ptr0 = CN * xp as usize + yy * stride as usize;
        src_ptr1 = CN * xp as usize + (yy + 1) * stride as usize;

        let mut dst_ptr0 = yy * stride as usize;
        let mut dst_ptr1 = (yy + 1) * stride as usize;

        for _ in 0..width {
            let store_ld0 = pixels.slice.as_ptr().add(dst_ptr0) as *mut u8;
            let store_ld1 = pixels.slice.as_ptr().add(dst_ptr1) as *mut u8;

            let o0 = _mm256_cvtepi32_ps(sums0);

            let r0 = _mm256_mul_ps(o0, v_mul_value);

            let cv0 = _mm256_cvtps_epi32(r0);

            store_u8_s32::<CN>(store_ld0, _mm256_castsi256_si128(cv0));
            store_u8_s32::<CN>(store_ld1, _mm256_extracti128_si256::<1>(cv0));

            dst_ptr0 += CN;
            dst_ptr1 += CN;

            sums0 = _mm256_sub_epi32(sums0, sum_out0);

            stack_start = sp + div - radius;
            if stack_start >= div {
                stack_start -= div;
            }
            let stack = stacks.as_mut_ptr().add(stack_start as usize * 2);

            let stack_val0 = _mm256_load_si256(stack as *const _);

            sum_out0 = _mm256_sub_epi32(sum_out0, stack_val0);

            if xp < wm {
                src_ptr0 += CN;
                src_ptr1 += CN;
                xp += 1;
            }

            let src_ld0 = pixels.slice.as_ptr().add(src_ptr0);
            let src_ld1 = pixels.slice.as_ptr().add(src_ptr1);

            let src_pixel0 = load_u8_s32_fast::<CN>(src_ld0 as *const u8);
            let src_pixel1 = load_u8_s32_fast::<CN>(src_ld1 as *const u8);

            let px0 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(src_pixel0), src_pixel1);

            _mm256_store_si256(stack as *mut _, px0);

            sum_in0 = _mm256_add_epi32(sum_in0, px0);

            sums0 = _mm256_add_epi32(sums0, sum_in0);

            sp += 1;
            if sp >= div {
                sp = 0;
            }
            let stack = stacks.as_mut_ptr().add(sp as usize * 2);
            let stack_val0 = _mm256_load_si256(stack as *const _);

            sum_out0 = _mm256_add_epi32(sum_out0, stack_val0);

            sum_in0 = _mm256_sub_epi32(sum_in0, stack_val0);
        }

        yy += 2;
    }

    for y in yy..max_y {
        let mut sums = _mm_setzero_si128();
        let mut sum_in = _mm_setzero_si128();
        let mut sum_out = _mm_setzero_si128();

        let mut src_ptr = stride as usize * y;

        let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
        let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);

        for i in 0..=radius {
            let stack_value = stacks.as_mut_ptr().add(i as usize * 2);
            _mm_storeu_si128(stack_value as *mut __m128i, src_pixel);
            sums = _mm_add_epi32(
                sums,
                _mm_madd_epi16(src_pixel, _mm_set1_epi32(i as i32 + 1)),
            );
            sum_out = _mm_add_epi32(sum_out, src_pixel);
        }

        for i in 1..=radius {
            if i <= wm {
                src_ptr += CN;
            }
            let stack_ptr = stacks.as_mut_ptr().add((i + radius) as usize * 2);
            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;
            let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);
            _mm_storeu_si128(stack_ptr as *mut __m128i, src_pixel);
            sums = _mm_add_epi32(
                sums,
                _mm_madd_epi16(src_pixel, _mm_set1_epi32(radius as i32 + 1 - i as i32)),
            );

            sum_in = _mm_add_epi32(sum_in, src_pixel);
        }

        sp = radius;
        xp = radius;
        if xp > wm {
            xp = wm;
        }

        src_ptr = CN * xp as usize + y * stride as usize;
        let mut dst_ptr = y * stride as usize;
        for _ in 0..width {
            let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut u8;
            let result = _mm_cvtps_epi32(_mm_mul_ps(
                _mm_cvtepi32_ps(sums),
                _mm256_castps256_ps128(v_mul_value),
            ));
            store_u8_s32::<CN>(store_ld, result);
            dst_ptr += CN;

            sums = _mm_sub_epi32(sums, sum_out);

            stack_start = sp + div - radius;
            if stack_start >= div {
                stack_start -= div;
            }
            let stack = stacks.as_mut_ptr().add(stack_start as usize * 2);

            let stack_val = _mm_loadu_si128(stack as *const __m128i);

            sum_out = _mm_sub_epi32(sum_out, stack_val);

            if xp < wm {
                src_ptr += CN;
                xp += 1;
            }

            let src_ld = pixels.slice.as_ptr().add(src_ptr);
            let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);
            _mm_storeu_si128(stack as *mut __m128i, src_pixel);

            sum_in = _mm_add_epi32(sum_in, src_pixel);
            sums = _mm_add_epi32(sums, sum_in);

            sp += 1;
            if sp >= div {
                sp = 0;
            }
            let stack = stacks.as_mut_ptr().add(sp as usize * 2);
            let stack_val = _mm_loadu_si128(stack as *const __m128i);

            sum_out = _mm_add_epi32(sum_out, stack_val);
            sum_in = _mm_sub_epi32(sum_in, stack_val);
        }
    }
}

impl<T, J, const CN: usize> StackBlurWorkingPass<T, CN> for HorizontalAvxStackBlurPass<T, J, CN>
where
    J: Copy
        + 'static
        + FromPrimitive
        + AddAssign<J>
        + Mul<Output = J>
        + Shr<Output = J>
        + Sub<Output = J>
        + AsPrimitive<f32>
        + SubAssign
        + AsPrimitive<T>
        + Default,
    T: Copy + AsPrimitive<J> + FromPrimitive,
    i32: AsPrimitive<J>,
    u32: AsPrimitive<J>,
    f32: AsPrimitive<T>,
    usize: AsPrimitive<J>,
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
            let pixels: &UnsafeSlice<u8> = std::mem::transmute(pixels);
            avx_horiz_pass_impl::<CN>(pixels, stride, width, height, radius, thread, total_threads);
        }
    }
}
