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
use crate::filter1d::color_group::{ld_group, ColorGroup};
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::AsPrimitive;
use std::ops::{Add, Mul, Shr};

pub fn filter_color_group_row_symmetric_approx<T, I, const N: usize>(
    _: Arena,
    arena_src: &[T],
    dst: &UnsafeSlice<T>,
    image_size: ImageSize,
    filter_region: FilterRegion,
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
    let width = image_size.width;

    let src = arena_src;

    let dst_stride = image_size.width * N;

    let length = scanned_kernel.len();
    let half_len = length / 2;

    let y = filter_region.start;

    let mut cx = 0usize;

    while cx + 8 < width {
        let v_cx = cx * N;
        let src = &src[cx * N..(v_cx + length * N + N * 8)];
        let coeff = scanned_kernel[half_len];

        let mut k0: ColorGroup<N, I> = ld_group!(src, N, half_len * N).mul(coeff.weight);
        let mut k1: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N).mul(coeff.weight);
        let mut k2: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N * 2).mul(coeff.weight);
        let mut k3: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N * 3).mul(coeff.weight);
        let mut k4: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N * 4).mul(coeff.weight);
        let mut k5: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N * 5).mul(coeff.weight);
        let mut k6: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N * 6).mul(coeff.weight);
        let mut k7: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N * 7).mul(coeff.weight);

        for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
            let rollback = length - i - 1;
            let fw = &src[(i * N)..((i + 8) * N)];
            let bw = &src[(rollback * N)..((rollback + 8) * N)];
            k0 = ld_group!(fw, N, 0)
                .add(ld_group!(bw, N, 0))
                .mul(coeff.weight)
                .add(k0);
            k1 = ld_group!(fw, N, N)
                .add(ld_group!(bw, N, N))
                .mul(coeff.weight)
                .add(k1);
            k2 = ld_group!(fw, N, 2 * N)
                .add(ld_group!(bw, N, 2 * N))
                .mul(coeff.weight)
                .add(k2);
            k3 = ld_group!(fw, N, 3 * N)
                .add(ld_group!(bw, N, 3 * N))
                .mul(coeff.weight)
                .add(k3);
            k4 = ld_group!(fw, N, 4 * N)
                .add(ld_group!(bw, N, 4 * N))
                .mul(coeff.weight)
                .add(k4);
            k5 = ld_group!(fw, N, 5 * N)
                .add(ld_group!(bw, N, 5 * N))
                .mul(coeff.weight)
                .add(k5);
            k6 = ld_group!(fw, N, 6 * N)
                .add(ld_group!(bw, N, 6 * N))
                .mul(coeff.weight)
                .add(k6);
            k7 = ld_group!(fw, N, 7 * N)
                .add(ld_group!(bw, N, 7 * N))
                .mul(coeff.weight)
                .add(k7);
        }

        let dst_offset = y * dst_stride + cx * N;

        k0.to_approx_store(dst, dst_offset);
        k1.to_approx_store(dst, dst_offset + N);
        k2.to_approx_store(dst, dst_offset + N * 2);
        k3.to_approx_store(dst, dst_offset + N * 3);
        k4.to_approx_store(dst, dst_offset + N * 4);
        k5.to_approx_store(dst, dst_offset + N * 5);
        k6.to_approx_store(dst, dst_offset + N * 6);
        k7.to_approx_store(dst, dst_offset + N * 7);
        cx += 8;
    }

    while cx + 4 < width {
        let v_cx = cx * N;
        let src = &src[cx * N..(v_cx + length * N + N * 4)];
        let coeff = scanned_kernel[half_len];

        let mut k0: ColorGroup<N, I> = ld_group!(src, N, half_len * N).mul(coeff.weight);
        let mut k1: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N).mul(coeff.weight);
        let mut k2: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N * 2).mul(coeff.weight);
        let mut k3: ColorGroup<N, I> = ld_group!(src, N, half_len * N + N * 3).mul(coeff.weight);

        for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
            let rollback = length - i - 1;
            let fw = &src[(i * N)..((i + 4) * N)];
            let bw = &src[(rollback * N)..((rollback + 4) * N)];
            k0 = ld_group!(fw, N, 0)
                .add(ld_group!(bw, N, 0))
                .mul(coeff.weight)
                .add(k0);
            k1 = ld_group!(fw, N, N)
                .add(ld_group!(bw, N, N))
                .mul(coeff.weight)
                .add(k1);
            k2 = ld_group!(fw, N, 2 * N)
                .add(ld_group!(bw, N, 2 * N))
                .mul(coeff.weight)
                .add(k2);
            k3 = ld_group!(fw, N, 3 * N)
                .add(ld_group!(bw, N, 3 * N))
                .mul(coeff.weight)
                .add(k3);
        }

        let dst_offset = y * dst_stride + cx * N;

        k0.to_approx_store(dst, dst_offset);
        k1.to_approx_store(dst, dst_offset + N);
        k2.to_approx_store(dst, dst_offset + N * 2);
        k3.to_approx_store(dst, dst_offset + N * 3);
        cx += 4;
    }

    for x in cx..width {
        let v_cx = x * N;
        let src = &src[v_cx..(v_cx + length * N)];
        let coeff = scanned_kernel[half_len];

        let mut k0: ColorGroup<N, I> = ld_group!(src, N, half_len * N).mul(coeff.weight);

        for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
            let rollback = length - i - 1;
            let fw = &src[(i * N)..((i + 1) * N)];
            let bw = &src[(rollback * N)..((rollback + 1) * N)];
            k0 = ld_group!(fw, N, 0)
                .add(ld_group!(bw, N, 0))
                .mul(coeff.weight)
                .add(k0);
        }

        k0.to_approx_store(dst, y * dst_stride + x * N);
    }
}
