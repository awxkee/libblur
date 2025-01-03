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
use crate::filter1d::color_group::ColorGroup;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::{AsPrimitive, MulAdd};
use std::ops::{Add, Mul};

pub fn filter_color_group_row<T, F, const N: usize>(
    _: Arena,
    arena_src: &[T],
    dst: &UnsafeSlice<T>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<F>],
) where
    T: Copy + AsPrimitive<F> + Default,
    F: ToStorage<T> + Mul<Output = F> + MulAdd<F, Output = F> + Default + Add<F, Output = F>,
{
    unsafe {
        let width = image_size.width;

        let dst_stride = image_size.width * N;

        let y = filter_region.start;
        let local_src = arena_src;

        let length = scanned_kernel.iter().len();

        let mut _cx = 0usize;

        while _cx + 4 < width {
            let coeff = *scanned_kernel.get_unchecked(0);

            let shifted_src = local_src.get_unchecked((_cx * N)..);

            let mut k0 = ColorGroup::<N, F>::from_slice(shifted_src, 0).mul(coeff.weight);
            let mut k1 = ColorGroup::<N, F>::from_slice(shifted_src, N).mul(coeff.weight);
            let mut k2 = ColorGroup::<N, F>::from_slice(shifted_src, N * 2).mul(coeff.weight);
            let mut k3 = ColorGroup::<N, F>::from_slice(shifted_src, N * 3).mul(coeff.weight);

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                k0 = ColorGroup::<N, F>::from_slice(shifted_src, i * N).mul_add(k0, coeff.weight);
                k1 = ColorGroup::<N, F>::from_slice(shifted_src, (i + 1) * N)
                    .mul_add(k1, coeff.weight);
                k2 = ColorGroup::<N, F>::from_slice(shifted_src, (i + 2) * N)
                    .mul_add(k2, coeff.weight);
                k3 = ColorGroup::<N, F>::from_slice(shifted_src, (i + 3) * N)
                    .mul_add(k3, coeff.weight);
            }

            let dst_offset = y * dst_stride + _cx * N;

            k0.to_store(dst, dst_offset);
            k1.to_store(dst, dst_offset + N);
            k2.to_store(dst, dst_offset + N * 2);
            k3.to_store(dst, dst_offset + N * 3);
            _cx += 4;
        }

        for x in _cx..width {
            let coeff = *scanned_kernel.get_unchecked(0);
            let shifted_src = local_src.get_unchecked((x * N)..);
            let mut k0 = ColorGroup::<N, F>::from_slice(shifted_src, 0).mul(coeff.weight);

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                k0 = ColorGroup::<N, F>::from_slice(shifted_src, i * N).mul_add(k0, coeff.weight);
            }

            k0.to_store(dst, y * dst_stride + x * N);
        }
    }
}
