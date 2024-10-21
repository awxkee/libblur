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
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::{AsPrimitive, MulAdd};
use std::ops::{Add, Mul};

pub fn filter_column<T, F>(
    arena: Arena,
    arena_src: &[&[T]],
    dst: &UnsafeSlice<T>,
    image_size: ImageSize,
    region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<F>],
) where
    T: Copy + AsPrimitive<F>,
    F: ToStorage<T> + Mul<F, Output = F> + MulAdd<F, Output = F> + Add<F, Output = F>,
{
    unsafe {
        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();

        let mut _cx = 0usize;

        let y = region.start;

        while _cx + 4 < dst_stride {
            let coeff = *scanned_kernel.get_unchecked(0);

            let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

            let mut k0 = v_src.get_unchecked(0).as_().mul(coeff.weight);
            let mut k1 = v_src.get_unchecked(1).as_().mul(coeff.weight);
            let mut k2 = v_src.get_unchecked(2).as_().mul(coeff.weight);
            let mut k3 = v_src.get_unchecked(3).as_().mul(coeff.weight);

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                k0 = mlaf(
                    k0,
                    arena_src.get_unchecked(i).get_unchecked(_cx).as_(),
                    coeff.weight,
                );
                k1 = mlaf(
                    k1,
                    arena_src.get_unchecked(i).get_unchecked(_cx + 1).as_(),
                    coeff.weight,
                );
                k2 = mlaf(
                    k2,
                    arena_src.get_unchecked(i).get_unchecked(_cx + 2).as_(),
                    coeff.weight,
                );
                k3 = mlaf(
                    k3,
                    arena_src.get_unchecked(i).get_unchecked(_cx + 3).as_(),
                    coeff.weight,
                );
            }

            let dst_offset = y * dst_stride + _cx;

            dst.write(dst_offset, k0.to_());
            dst.write(dst_offset + 1, k1.to_());
            dst.write(dst_offset + 2, k2.to_());
            dst.write(dst_offset + 3, k3.to_());
            _cx += 4;
        }

        for x in _cx..dst_stride {
            let coeff = *scanned_kernel.get_unchecked(0);

            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let mut k0 = v_src.get_unchecked(0).as_().mul(coeff.weight);

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                k0 = mlaf(
                    k0,
                    arena_src.get_unchecked(i).get_unchecked(x).as_(),
                    coeff.weight,
                );
            }

            dst.write(y * dst_stride + x, k0.to_());
        }
    }
}
