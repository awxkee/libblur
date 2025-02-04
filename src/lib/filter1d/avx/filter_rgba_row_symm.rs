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
use crate::avx::{_mm256_load_deinterleave_rgba, _mm256_store_interleave_rgba};
use crate::filter1d::arena::Arena;
use crate::filter1d::avx::utils::{
    _mm256_mul_add_symm_epi8_by_ps_x4, _mm256_mul_epi8_by_ps_x4, _mm256_pack_ps_x4_epi8,
};
use crate::filter1d::color_group::ColorGroup;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::sse::utils::{
    _mm_mul_add_symm_epi8_by_ps_x2, _mm_mul_add_symm_epi8_by_ps_x4, _mm_mul_epi8_by_ps_x2,
    _mm_mul_epi8_by_ps_x4, _mm_pack_ps_x2_epi8, _mm_pack_ps_x4_epi8,
};
use crate::img_size::ImageSize;
use crate::sse::{
    _mm_load_deinterleave_rgba, _mm_load_deinterleave_rgba_half, _mm_store_interleave_rgba,
    _mm_store_interleave_rgba_half,
};
use crate::unsafe_slice::UnsafeSlice;
use num_traits::MulAdd;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_rgba_row_avx_symm_u8_f32(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let has_fma = std::arch::is_x86_feature_detected!("fma");
        if has_fma {
            filter_rgba_row_avx_symm_u8_f32_fma(
                arena,
                arena_src,
                dst,
                image_size,
                filter_region,
                scanned_kernel,
            );
        } else {
            filter_rgba_row_avx_u8_f32_def(
                arena,
                arena_src,
                dst,
                image_size,
                filter_region,
                scanned_kernel,
            );
        }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn filter_rgba_row_avx_symm_u8_f32_fma(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_rgba_row_avx_symm_u8_f32_impl::<true>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[target_feature(enable = "avx2")]
unsafe fn filter_rgba_row_avx_u8_f32_def(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_rgba_row_avx_symm_u8_f32_impl::<false>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[inline(always)]
unsafe fn filter_rgba_row_avx_symm_u8_f32_impl<const FMA: bool>(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let width = image_size.width;

    const N: usize = 4;

    let src = &arena_src;

    let dst_stride = image_size.width * arena.components;

    let length = scanned_kernel.len();
    let half_len = length / 2;

    let y = filter_region.start;
    let local_src = src;

    let mut _cx = 0usize;

    while _cx + 32 < width {
        let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

        let shifted_src = local_src.get_unchecked((_cx * N)..);

        let source =
            _mm256_load_deinterleave_rgba(shifted_src.get_unchecked((half_len * N)..).as_ptr());
        let mut k0 = _mm256_mul_epi8_by_ps_x4(source.0, coeff);
        let mut k1 = _mm256_mul_epi8_by_ps_x4(source.1, coeff);
        let mut k2 = _mm256_mul_epi8_by_ps_x4(source.2, coeff);
        let mut k3 = _mm256_mul_epi8_by_ps_x4(source.3, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
            let v_source0 =
                _mm256_load_deinterleave_rgba(shifted_src.get_unchecked((i * N)..).as_ptr());
            let v_source1 =
                _mm256_load_deinterleave_rgba(shifted_src.get_unchecked((rollback * N)..).as_ptr());
            k0 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k1, v_source0.1, v_source1.1, coeff);
            k2 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k2, v_source0.2, v_source1.2, coeff);
            k3 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k3, v_source0.3, v_source1.3, coeff);
        }

        let dst_offset = y * dst_stride + _cx * N;
        let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
        _mm256_store_interleave_rgba(
            dst_ptr0,
            (
                _mm256_pack_ps_x4_epi8(k0),
                _mm256_pack_ps_x4_epi8(k1),
                _mm256_pack_ps_x4_epi8(k2),
                _mm256_pack_ps_x4_epi8(k3),
            ),
        );
        _cx += 32;
    }

    while _cx + 16 < width {
        let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

        let shifted_src = local_src.get_unchecked((_cx * N)..);

        let source =
            _mm_load_deinterleave_rgba(shifted_src.get_unchecked((half_len * N)..).as_ptr());
        let mut k0 = _mm_mul_epi8_by_ps_x4(source.0, coeff);
        let mut k1 = _mm_mul_epi8_by_ps_x4(source.1, coeff);
        let mut k2 = _mm_mul_epi8_by_ps_x4(source.2, coeff);
        let mut k3 = _mm_mul_epi8_by_ps_x4(source.3, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(i).weight);
            let v_source0 =
                _mm_load_deinterleave_rgba(shifted_src.get_unchecked((i * N)..).as_ptr());
            let v_source1 =
                _mm_load_deinterleave_rgba(shifted_src.get_unchecked((rollback * N)..).as_ptr());
            k0 = _mm_mul_add_symm_epi8_by_ps_x4::<FMA>(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm_mul_add_symm_epi8_by_ps_x4::<FMA>(k1, v_source0.1, v_source1.1, coeff);
            k2 = _mm_mul_add_symm_epi8_by_ps_x4::<FMA>(k2, v_source0.2, v_source1.2, coeff);
            k3 = _mm_mul_add_symm_epi8_by_ps_x4::<FMA>(k3, v_source0.3, v_source1.3, coeff);
        }

        let dst_offset = y * dst_stride + _cx * N;
        let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
        _mm_store_interleave_rgba(
            dst_ptr0,
            (
                _mm_pack_ps_x4_epi8(k0),
                _mm_pack_ps_x4_epi8(k1),
                _mm_pack_ps_x4_epi8(k2),
                _mm_pack_ps_x4_epi8(k3),
            ),
        );
        _cx += 16;
    }

    while _cx + 8 < width {
        let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

        let shifted_src = local_src.get_unchecked((_cx * N)..);

        let source =
            _mm_load_deinterleave_rgba_half(shifted_src.get_unchecked((half_len * N)..).as_ptr());
        let mut k0 = _mm_mul_epi8_by_ps_x2(source.0, coeff);
        let mut k1 = _mm_mul_epi8_by_ps_x2(source.1, coeff);
        let mut k2 = _mm_mul_epi8_by_ps_x2(source.2, coeff);
        let mut k3 = _mm_mul_epi8_by_ps_x2(source.3, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(i).weight);
            let v_source0 =
                _mm_load_deinterleave_rgba_half(shifted_src.get_unchecked((i * N)..).as_ptr());
            let v_source1 = _mm_load_deinterleave_rgba_half(
                shifted_src.get_unchecked((rollback * N)..).as_ptr(),
            );
            k0 = _mm_mul_add_symm_epi8_by_ps_x2::<FMA>(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm_mul_add_symm_epi8_by_ps_x2::<FMA>(k1, v_source0.1, v_source1.1, coeff);
            k2 = _mm_mul_add_symm_epi8_by_ps_x2::<FMA>(k2, v_source0.2, v_source1.2, coeff);
            k3 = _mm_mul_add_symm_epi8_by_ps_x2::<FMA>(k3, v_source0.3, v_source1.3, coeff);
        }

        let dst_offset = y * dst_stride + _cx * N;
        let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
        _mm_store_interleave_rgba_half(
            dst_ptr0,
            (
                _mm_pack_ps_x2_epi8(k0),
                _mm_pack_ps_x2_epi8(k1),
                _mm_pack_ps_x2_epi8(k2),
                _mm_pack_ps_x2_epi8(k3),
            ),
        );
        _cx += 8;
    }

    while _cx + 4 < width {
        let coeff = *scanned_kernel.get_unchecked(half_len);

        let shifted_src = local_src.get_unchecked((_cx * N)..);

        let mut k0 = ColorGroup::<N, f32>::from_slice(shifted_src, half_len * N).mul(coeff.weight);
        let mut k1 =
            ColorGroup::<N, f32>::from_slice(shifted_src, (half_len * N) + N).mul(coeff.weight);
        let mut k2 =
            ColorGroup::<N, f32>::from_slice(shifted_src, (half_len * N) + N * 2).mul(coeff.weight);
        let mut k3 =
            ColorGroup::<N, f32>::from_slice(shifted_src, (half_len * N) + N * 3).mul(coeff.weight);

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            k0 = ColorGroup::<N, f32>::from_slice(shifted_src, i * N)
                .add(ColorGroup::<N, f32>::from_slice(shifted_src, rollback * N))
                .mul_add(k0, coeff.weight);
            k1 = ColorGroup::<N, f32>::from_slice(shifted_src, (i + 1) * N)
                .add(ColorGroup::<N, f32>::from_slice(
                    shifted_src,
                    (rollback + 1) * N,
                ))
                .mul_add(k1, coeff.weight);
            k2 = ColorGroup::<N, f32>::from_slice(shifted_src, (i + 2) * N)
                .add(ColorGroup::<N, f32>::from_slice(
                    shifted_src,
                    (rollback + 2) * N,
                ))
                .mul_add(k2, coeff.weight);
            k3 = ColorGroup::<N, f32>::from_slice(shifted_src, (i + 3) * N)
                .add(ColorGroup::<N, f32>::from_slice(
                    shifted_src,
                    (rollback + 3) * N,
                ))
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
        let coeff = *scanned_kernel.get_unchecked(half_len);
        let shifted_src = local_src.get_unchecked((x * N)..);
        let mut k0 = ColorGroup::<N, f32>::from_slice(shifted_src, half_len * N).mul(coeff.weight);

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            k0 = ColorGroup::<N, f32>::from_slice(shifted_src, i * N)
                .add(ColorGroup::<N, f32>::from_slice(shifted_src, rollback * N))
                .mul_add(k0, coeff.weight);
        }

        k0.to_store(dst, y * dst_stride + x * N);
    }
}
