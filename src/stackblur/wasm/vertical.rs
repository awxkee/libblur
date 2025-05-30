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
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use crate::wasm32::{
    load_u8_s32_fast, u16x8_pack_trunc_u8x16, u32x4_pack_trunc_u16x8, w_store_u8x8_m4,
};
use num_traits::{AsPrimitive, FromPrimitive};
use std::arch::wasm32::*;
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Shr, Sub, SubAssign};

pub struct VerticalWasmStackBlurPass<T, J, const CN: usize> {
    _phantom_t: PhantomData<T>,
    _phantom_j: PhantomData<J>,
}

impl<T, J, const CN: usize> Default for VerticalWasmStackBlurPass<T, J, CN> {
    fn default() -> Self {
        VerticalWasmStackBlurPass::<T, J, CN> {
            _phantom_t: Default::default(),
            _phantom_j: Default::default(),
        }
    }
}

impl<T, J, const CN: usize> VerticalWasmStackBlurPass<T, J, CN>
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
    #[target_feature(enable = "simd128")]
    unsafe fn pass_impl(
        &self,
        pixels: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        total_threads: usize,
    ) {
        let pixels: &UnsafeSlice<u8> = std::mem::transmute(pixels);
        let div = ((radius * 2) + 1) as usize;
        let mut yp;
        let mut sp;
        let mut stack_start;
        let mut stacks = vec![0i32; 4 * div * total_threads];

        let v_mul_value = f32x4_splat(1. / ((radius as f32 + 1.) * (radius as f32 + 1.)));

        let hm = height - 1;
        let div = (radius * 2) + 1;

        let mut src_ptr;
        let mut dst_ptr;

        let min_x = thread * width as usize / total_threads;
        let max_x = (thread + 1) * width as usize / total_threads;

        for x in min_x..max_x {
            let mut sums = i32x4_splat(0i32);
            let mut sum_in = i32x4_splat(0i32);
            let mut sum_out = i32x4_splat(0i32);

            src_ptr = CN * x; // x,0

            let src_ld = pixels.slice.as_ptr().add(src_ptr) as *const i32;

            let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);

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
                let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);
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
            src_ptr = CN * x + yp as usize * stride as usize;
            dst_ptr = CN * x;
            for _ in 0..height {
                let store_ld = pixels.slice.as_ptr().add(dst_ptr) as *mut u8;
                let blurred = f32x4_ceil(f32x4_mul(f32x4_convert_i32x4(sums), v_mul_value));
                let prepared_u16 = u32x4_pack_trunc_u16x8(blurred, blurred);
                let blurred = u16x8_pack_trunc_u8x16(prepared_u16, prepared_u16);
                w_store_u8x8_m4::<CN>(store_ld, blurred);

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
                let src_pixel = load_u8_s32_fast::<CN>(src_ld as *const u8);
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

impl<T, J, const CN: usize> StackBlurWorkingPass<T, CN> for VerticalWasmStackBlurPass<T, J, CN>
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
            self.pass_impl(pixels, stride, width, height, radius, thread, total_threads);
        }
    }
}
