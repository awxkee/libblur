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
use crate::filter1d::arena::Arena;
use crate::img_size::ImageSize;
use crate::to_storage::ToStorage;
use num_complex::Complex;
use num_traits::{AsPrimitive, MulAdd, Num};
use std::ops::{Add, Mul};

pub(crate) fn filter_column_complex<T, F>(
    arena: Arena,
    arena_src: &[&[Complex<F>]],
    dst: &mut [T],
    image_size: ImageSize,
    kernel: &[Complex<F>],
) where
    T: Copy + AsPrimitive<F>,
    F: ToStorage<T> + Mul<F, Output = F> + MulAdd<F, Output = F> + Add<F, Output = F> + Num,
{
    unsafe {
        let full_width = image_size.width * arena.components;

        let length = kernel.len();

        let mut cx = 0usize;

        let coeff = *kernel.get_unchecked(0);

        while cx + 4 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let mut k0 = coeff * (*v_src.get_unchecked(0));
            let mut k1 = coeff * (*v_src.get_unchecked(1));
            let mut k2 = coeff * (*v_src.get_unchecked(2));
            let mut k3 = coeff * (*v_src.get_unchecked(3));

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);
                k0 = arena_src
                    .get_unchecked(i)
                    .get_unchecked(cx)
                    .mul_add(&coeff, &k0);
                k1 = arena_src
                    .get_unchecked(i)
                    .get_unchecked(cx + 1)
                    .mul_add(&coeff, &k1);
                k2 = arena_src
                    .get_unchecked(i)
                    .get_unchecked(cx + 2)
                    .mul_add(&coeff, &k2);
                k3 = arena_src
                    .get_unchecked(i)
                    .get_unchecked(cx + 3)
                    .mul_add(&coeff, &k3);
            }

            *dst.get_unchecked_mut(cx) = k0.re.to_();
            *dst.get_unchecked_mut(cx + 1) = k1.re.to_();
            *dst.get_unchecked_mut(cx + 2) = k2.re.to_();
            *dst.get_unchecked_mut(cx + 3) = k3.re.to_();
            cx += 4;
        }

        for x in cx..full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let mut k0 = coeff * (*v_src.get_unchecked(0));

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);
                k0 = arena_src
                    .get_unchecked(i)
                    .get_unchecked(x)
                    .mul_add(&coeff, &k0);
            }

            *dst.get_unchecked_mut(x) = k0.re.to_();
        }
    }
}
