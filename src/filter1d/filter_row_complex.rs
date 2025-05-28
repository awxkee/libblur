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
use num_traits::{AsPrimitive, MulAdd};
use std::ops::Add;

pub(crate) fn filter_row_complex<T, F>(
    arena: Arena,
    arena_src: &[T],
    dst: &mut [Complex<F>],
    image_size: ImageSize,
    kernel: &[Complex<F>],
) where
    T: Copy + AsPrimitive<F>,
    F: ToStorage<T> + MulAdd<F, Output = F> + Add<F, Output = F> + num_traits::Num,
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

            let mut k0 = coeff * ((*s_src.get_unchecked(0)).as_());
            let mut k1 = coeff * ((*s_src.get_unchecked(1)).as_());
            let mut k2 = coeff * ((*s_src.get_unchecked(2)).as_());
            let mut k3 = coeff * ((*s_src.get_unchecked(3)).as_());

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);
                k0 = k0 + coeff * (*s_src.get_unchecked(i * arena.components)).as_();
                k1 = k1 + coeff * (*s_src.get_unchecked(i * arena.components + 1)).as_();
                k2 = k2 + coeff * (*s_src.get_unchecked(i * arena.components + 2)).as_();
                k3 = k3 + coeff * (*s_src.get_unchecked(i * arena.components + 3)).as_();
            }

            *dst.get_unchecked_mut(cx) = k0;
            *dst.get_unchecked_mut(cx + 1) = k1;
            *dst.get_unchecked_mut(cx + 2) = k2;
            *dst.get_unchecked_mut(cx + 3) = k3;
            cx += 4;
        }

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);
            let mut k0 = coeff * (*shifted_src.get_unchecked(0)).as_();

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);
                k0 = k0 + coeff * (*shifted_src.get_unchecked(i * arena.components)).as_();
            }
            *dst.get_unchecked_mut(x) = k0;
        }
    }
}
