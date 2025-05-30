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
use crate::filter1d::to_approx_storage_complex::ApproxComplexLevel;
use crate::img_size::ImageSize;
use num_complex::Complex;
use num_traits::AsPrimitive;

#[inline(always)]
fn norm_q_complex<T: ApproxComplexLevel, I: Copy + 'static>(c: Complex<i32>) -> Complex<I>
where
    i32: AsPrimitive<I>,
{
    Complex::new(
        ((c.re + (1 << (T::Q - 1))) >> T::Q).as_(),
        ((c.im + (1 << (T::Q - 1))) >> T::Q).as_(),
    )
}

pub(crate) fn filter_row_complex_q<T, I>(
    arena: Arena,
    arena_src: &[T],
    dst: &mut [Complex<I>],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) where
    T: Copy + AsPrimitive<i32> + ApproxComplexLevel,
    I: Copy + 'static,
    i32: AsPrimitive<I>,
{
    unsafe {
        let width = image_size.width;

        let src = arena_src;

        let local_src = src;

        let length = kernel.len();

        let mut cx = 0usize;

        let max_width = width * arena.components;

        let coeff = *kernel.get_unchecked(0);

        while cx + 4 < max_width {
            let s_src = local_src.get_unchecked(cx..);

            let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);

            let mut k0 = q_coeff * ((*s_src.get_unchecked(0)).as_());
            let mut k1 = q_coeff * ((*s_src.get_unchecked(1)).as_());
            let mut k2 = q_coeff * ((*s_src.get_unchecked(2)).as_());
            let mut k3 = q_coeff * ((*s_src.get_unchecked(3)).as_());

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);
                let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);
                k0 += q_coeff * (*s_src.get_unchecked(i * arena.components)).as_();
                k1 += q_coeff * (*s_src.get_unchecked(i * arena.components + 1)).as_();
                k2 += q_coeff * (*s_src.get_unchecked(i * arena.components + 2)).as_();
                k3 += q_coeff * (*s_src.get_unchecked(i * arena.components + 3)).as_();
            }

            *dst.get_unchecked_mut(cx) = norm_q_complex::<T, I>(k0);
            *dst.get_unchecked_mut(cx + 1) = norm_q_complex::<T, I>(k1);
            *dst.get_unchecked_mut(cx + 2) = norm_q_complex::<T, I>(k2);
            *dst.get_unchecked_mut(cx + 3) = norm_q_complex::<T, I>(k3);
            cx += 4;
        }

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);

            let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);

            let mut k0 = q_coeff * (*shifted_src.get_unchecked(0)).as_();

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);
                let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);
                k0 += q_coeff * (*shifted_src.get_unchecked(i * arena.components)).as_();
            }
            *dst.get_unchecked_mut(x) = norm_q_complex::<T, I>(k0);
        }
    }
}
