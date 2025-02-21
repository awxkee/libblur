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
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use num_traits::AsPrimitive;
use std::ops::{Add, Mul, Shr};

pub fn filter_row_symmetric_approx<T, I, const N: usize>(
    _: Arena,
    arena_src: &[T],
    dst: &mut [T],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<I>],
) where
    T: Copy + AsPrimitive<I> + Default,
    I: Copy
        + Mul<Output = I>
        + Add<Output = I>
        + Shr<I, Output = I>
        + Default
        + 'static
        + ToApproxStorage<T>,
    i32: AsPrimitive<I>,
{
    unsafe {
        let width = image_size.width;

        let src = arena_src;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let max_width = width * N;

        let mut cx = 0usize;

        while cx + 4 < max_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);
            let shifted_src = src.get_unchecked(cx..);
            let mut k0 = shifted_src.get_unchecked(0).as_() * coeff.weight;
            let mut k1 = shifted_src.get_unchecked(1).as_() * coeff.weight;
            let mut k2 = shifted_src.get_unchecked(2).as_() * coeff.weight;
            let mut k3 = shifted_src.get_unchecked(3).as_() * coeff.weight;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 = k0
                    + (shifted_src.get_unchecked(i * N).as_()
                        + shifted_src.get_unchecked(rollback * N).as_())
                        * coeff.weight;

                k1 = k1
                    + (shifted_src.get_unchecked(i * N + 1).as_()
                        + shifted_src.get_unchecked(rollback * N + 1).as_())
                        * coeff.weight;

                k2 = k2
                    + (shifted_src.get_unchecked(i * N + 2).as_()
                        + shifted_src.get_unchecked(rollback * N + 2).as_())
                        * coeff.weight;

                k3 = k3
                    + (shifted_src.get_unchecked(i * N + 3).as_()
                        + shifted_src.get_unchecked(rollback * N + 3).as_())
                        * coeff.weight;
            }

            *dst.get_unchecked_mut(cx) = k0.to_approx_();
            *dst.get_unchecked_mut(cx + 1) = k1.to_approx_();
            *dst.get_unchecked_mut(cx + 2) = k2.to_approx_();
            *dst.get_unchecked_mut(cx + 3) = k3.to_approx_();
            cx += 4;
        }

        for x in cx..max_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);
            let shifted_src = src.get_unchecked(x..);
            let mut k0 = shifted_src.get_unchecked(0).as_() * coeff.weight;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 = k0
                    + (shifted_src.get_unchecked(i * N).as_()
                        + shifted_src.get_unchecked(rollback * N).as_())
                        * coeff.weight;
            }

            *dst.get_unchecked_mut(x) = k0.to_approx_();
        }
    }
}
