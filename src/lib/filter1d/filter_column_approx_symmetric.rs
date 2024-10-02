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
use crate::unsafe_slice::UnsafeSlice;
use num_traits::AsPrimitive;
use std::ops::{Add, Mul, Shr};

pub fn filter_column_symmetric_approx<T, I>(
    arena: Arena,
    arena_src: &[T],
    dst: &UnsafeSlice<T>,
    image_size: ImageSize,
    region: FilterRegion,
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
{
    unsafe {
        let src = arena_src;

        let arena_width = arena.width * arena.components;

        let _yy = region.start;

        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        for y in _yy..region.end {
            let mut _cx = 0usize;

            let local_src = src.get_unchecked((y * arena_width)..);

            while _cx + 4 < dst_stride {
                let coeff = scanned_kernel.get_unchecked(half_len).weight;

                let shifted_src = local_src.get_unchecked(_cx..);

                let mut k0 = (*shifted_src.get_unchecked(half_len * arena_width))
                    .as_()
                    .mul(coeff);
                let mut k1 = (*shifted_src.get_unchecked((half_len * arena_width) + 1))
                    .as_()
                    .mul(coeff);
                let mut k2 = (*shifted_src.get_unchecked((half_len * arena_width) + 2))
                    .as_()
                    .mul(coeff);
                let mut k3 = (*shifted_src.get_unchecked((half_len * arena_width) + 3))
                    .as_()
                    .mul(coeff);

                for i in 0..half_len {
                    let coeff = scanned_kernel.get_unchecked(i).weight;
                    let rollback = length - i - 1;
                    k0 = shifted_src
                        .get_unchecked(i * arena_width)
                        .as_()
                        .add(shifted_src.get_unchecked(rollback * arena_width).as_())
                        .mul(coeff)
                        .add(k0);
                    k1 = shifted_src
                        .get_unchecked(i * arena_width + 1)
                        .as_()
                        .add(shifted_src.get_unchecked(rollback * arena_width + 1).as_())
                        .mul(coeff)
                        .add(k1);
                    k2 = shifted_src
                        .get_unchecked(i * arena_width + 2)
                        .as_()
                        .add(shifted_src.get_unchecked(rollback * arena_width + 2).as_())
                        .mul(coeff)
                        .add(k2);
                    k3 = shifted_src
                        .get_unchecked(i * arena_width + 3)
                        .as_()
                        .add(shifted_src.get_unchecked(rollback * arena_width + 3).as_())
                        .mul(coeff)
                        .add(k3);
                }

                let dst_offset = y * dst_stride + _cx;

                dst.write(dst_offset, k0.to_approx_());
                dst.write(dst_offset + 1, k1.to_approx_());
                dst.write(dst_offset + 2, k2.to_approx_());
                dst.write(dst_offset + 3, k3.to_approx_());
                _cx += 4;
            }

            for x in _cx..dst_stride {
                let coeff = scanned_kernel.get_unchecked(half_len).weight;

                let shifted_src = local_src.get_unchecked(x..);

                let mut k0 = (*shifted_src.get_unchecked(half_len * arena_width))
                    .as_()
                    .mul(coeff);

                for i in 0..half_len {
                    let coeff = scanned_kernel.get_unchecked(i).weight;
                    let rollback = length - i - 1;
                    k0 = shifted_src
                        .get_unchecked(i * arena_width)
                        .as_()
                        .add(shifted_src.get_unchecked(rollback * arena_width).as_())
                        .mul(coeff)
                        .add(k0);
                }

                dst.write(y * dst_stride + x, k0.to_approx_());
            }
        }
    }
}
