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
use crate::filter1d::sse::utils::{
    _mm_mul_add_symm_epi8_by_epi16_x2, _mm_mul_add_symm_epi8_by_epi16_x4, _mm_mul_epi8_by_epi16_x2,
    _mm_mul_epi8_by_epi16_x4, _mm_pack_epi32_epi8, _mm_pack_epi32_x2_epi8,
};
use crate::img_size::ImageSize;
use crate::sse::{
    _mm_load_deinterleave_rgb, _mm_load_deinterleave_rgb_half, _mm_store_interleave_rgb,
    _mm_store_interleave_rgb_half,
};
use crate::unsafe_slice::UnsafeSlice;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

pub fn filter_rgb_row_sse_symm_u8_i32_app(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        filter_rgb_row_neon_u8_i32_impl(
            arena,
            arena_src,
            dst,
            image_size,
            filter_region,
            scanned_kernel,
        );
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn filter_rgb_row_neon_u8_i32_impl(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    let width = image_size.width;

    const N: usize = 3;

    let src = arena_src;

    let dst_stride = image_size.width * arena.components;

    let length = scanned_kernel.len();
    let half_len = length / 2;

    let y = filter_region.start;
    let local_src = src;

    let mut _cx = 0usize;

    while _cx + 16 < width {
        let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(half_len).weight as i16);

        let shifted_src = local_src.get_unchecked((_cx * N)..);

        let source =
            _mm_load_deinterleave_rgb(shifted_src.get_unchecked((half_len * N)..).as_ptr());
        let mut k0 = _mm_mul_epi8_by_epi16_x4(source.0, coeff);
        let mut k1 = _mm_mul_epi8_by_epi16_x4(source.1, coeff);
        let mut k2 = _mm_mul_epi8_by_epi16_x4(source.2, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(i).weight as i16);
            let v_source0 =
                _mm_load_deinterleave_rgb(shifted_src.get_unchecked((i * N)..).as_ptr());
            let v_source1 =
                _mm_load_deinterleave_rgb(shifted_src.get_unchecked((rollback * N)..).as_ptr());
            k0 = _mm_mul_add_symm_epi8_by_epi16_x4(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm_mul_add_symm_epi8_by_epi16_x4(k1, v_source0.1, v_source1.1, coeff);
            k2 = _mm_mul_add_symm_epi8_by_epi16_x4(k2, v_source0.2, v_source1.2, coeff);
        }

        let dst_offset = y * dst_stride + _cx * N;
        let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
        _mm_store_interleave_rgb(
            dst_ptr0,
            (
                _mm_pack_epi32_x2_epi8(k0),
                _mm_pack_epi32_x2_epi8(k1),
                _mm_pack_epi32_x2_epi8(k2),
            ),
        );
        _cx += 16;
    }

    while _cx + 8 < width {
        let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(half_len).weight as i16);

        let shifted_src = local_src.get_unchecked((_cx * N)..);

        let source =
            _mm_load_deinterleave_rgb_half(shifted_src.get_unchecked((half_len * N)..).as_ptr());
        let mut k0 = _mm_mul_epi8_by_epi16_x2(source.0, coeff);
        let mut k1 = _mm_mul_epi8_by_epi16_x2(source.1, coeff);
        let mut k2 = _mm_mul_epi8_by_epi16_x2(source.2, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(i).weight as i16);
            let v_source0 =
                _mm_load_deinterleave_rgb_half(shifted_src.get_unchecked((i * N)..).as_ptr());
            let v_source1 = _mm_load_deinterleave_rgb_half(
                shifted_src.get_unchecked((rollback * N)..).as_ptr(),
            );
            k0 = _mm_mul_add_symm_epi8_by_epi16_x2(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm_mul_add_symm_epi8_by_epi16_x2(k1, v_source0.1, v_source1.1, coeff);
            k2 = _mm_mul_add_symm_epi8_by_epi16_x2(k2, v_source0.2, v_source1.2, coeff);
        }

        let dst_offset = y * dst_stride + _cx * N;
        let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
        _mm_store_interleave_rgb_half(
            dst_ptr0,
            (
                _mm_pack_epi32_epi8(k0),
                _mm_pack_epi32_epi8(k1),
                _mm_pack_epi32_epi8(k2),
            ),
        );
        _cx += 8;
    }

    while _cx + 4 < width {
        let coeff = *scanned_kernel.get_unchecked(half_len);

        let shifted_src = local_src.get_unchecked((_cx * N)..);

        let mut k0 = ColorGroup::<N, i32>::from_slice(shifted_src, half_len * N).mul(coeff.weight);
        let mut k1 =
            ColorGroup::<N, i32>::from_slice(shifted_src, half_len * N + N).mul(coeff.weight);
        let mut k2 =
            ColorGroup::<N, i32>::from_slice(shifted_src, half_len * N + N * 2).mul(coeff.weight);
        let mut k3 =
            ColorGroup::<N, i32>::from_slice(shifted_src, half_len * N + N * 3).mul(coeff.weight);

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            k0 = ColorGroup::<N, i32>::from_slice(shifted_src, i * N)
                .add(ColorGroup::<N, i32>::from_slice(shifted_src, rollback * N))
                .mul(coeff.weight)
                .add(k0);
            k1 = ColorGroup::<N, i32>::from_slice(shifted_src, (i + 1) * N)
                .add(ColorGroup::<N, i32>::from_slice(
                    shifted_src,
                    (rollback + 1) * N,
                ))
                .mul(coeff.weight)
                .add(k1);
            k2 = ColorGroup::<N, i32>::from_slice(shifted_src, (i + 2) * N)
                .add(ColorGroup::<N, i32>::from_slice(
                    shifted_src,
                    (rollback + 2) * N,
                ))
                .mul(coeff.weight)
                .add(k2);
            k3 = ColorGroup::<N, i32>::from_slice(shifted_src, (i + 3) * N)
                .add(ColorGroup::<N, i32>::from_slice(
                    shifted_src,
                    (rollback + 3) * N,
                ))
                .mul(coeff.weight)
                .add(k3);
        }

        let dst_offset = y * dst_stride + _cx * N;

        k0.to_approx_store(dst, dst_offset);
        k1.to_approx_store(dst, dst_offset + N);
        k2.to_approx_store(dst, dst_offset + N * 2);
        k3.to_approx_store(dst, dst_offset + N * 3);
        _cx += 4;
    }

    for x in _cx..width {
        let coeff = *scanned_kernel.get_unchecked(half_len);
        let shifted_src = local_src.get_unchecked((x * N)..);
        let mut k0 = ColorGroup::<N, i32>::from_slice(shifted_src, half_len * N).mul(coeff.weight);

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            k0 = ColorGroup::<N, i32>::from_slice(shifted_src, i * N)
                .add(ColorGroup::<N, i32>::from_slice(shifted_src, rollback * N))
                .mul(coeff.weight)
                .add(k0);
        }

        k0.to_approx_store(dst, y * dst_stride + x * N);
    }
}
