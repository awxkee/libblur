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
#![allow(dead_code)]
use crate::filter1d::arena::Arena;
use crate::filter1d::to_approx_storage_complex::ToApproxStorageComplex;
use crate::img_size::ImageSize;
use num_complex::Complex;
use num_traits::{AsPrimitive, MulAdd};

#[inline(always)]
pub(crate) fn wrap_complex<I: AsPrimitive<i32> + 'static + Copy>(c: &Complex<I>) -> Complex<i32> {
    Complex::new(c.re.as_(), c.im.as_())
}

pub(crate) fn filter_column_complex_q<T, I>(
    arena: Arena,
    arena_src: &[&[Complex<I>]],
    dst: &mut [T],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) where
    T: Copy + 'static,
    I: Copy + 'static + AsPrimitive<i32>,
    i32: ToApproxStorageComplex<T>,
{
    unsafe {
        let full_width = image_size.width * arena.components;

        let length = kernel.len();

        let mut cx = 0usize;

        let coeff = *kernel.get_unchecked(0);

        while cx + 4 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);

            let mut k0 = q_coeff * wrap_complex::<I>(v_src.get_unchecked(0));
            let mut k1 = q_coeff * wrap_complex::<I>(v_src.get_unchecked(1));
            let mut k2 = q_coeff * wrap_complex::<I>(v_src.get_unchecked(2));
            let mut k3 = q_coeff * wrap_complex::<I>(v_src.get_unchecked(3));

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);
                let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);
                k0 = wrap_complex::<I>(arena_src.get_unchecked(i).get_unchecked(cx))
                    .mul_add(q_coeff, k0);
                k1 = wrap_complex::<I>(arena_src.get_unchecked(i).get_unchecked(cx + 1))
                    .mul_add(q_coeff, k1);
                k2 = wrap_complex::<I>(arena_src.get_unchecked(i).get_unchecked(cx + 2))
                    .mul_add(q_coeff, k2);
                k3 = wrap_complex::<I>(arena_src.get_unchecked(i).get_unchecked(cx + 3))
                    .mul_add(q_coeff, k3);
            }

            *dst.get_unchecked_mut(cx) = k0.re.to_c_approx_();
            *dst.get_unchecked_mut(cx + 1) = k1.re.to_c_approx_();
            *dst.get_unchecked_mut(cx + 2) = k2.re.to_c_approx_();
            *dst.get_unchecked_mut(cx + 3) = k3.re.to_c_approx_();
            cx += 4;
        }

        for x in cx..full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);

            let mut k0 = q_coeff * wrap_complex::<I>(v_src.get_unchecked(0));

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);
                let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);
                k0 = wrap_complex::<I>(arena_src.get_unchecked(i).get_unchecked(x))
                    .mul_add(q_coeff, k0);
            }

            *dst.get_unchecked_mut(x) = k0.re.to_c_approx_();
        }
    }
}
