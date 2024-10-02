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
use crate::avx::{
    _mm256_load_pack_x2, _mm256_load_pack_x3, _mm256_load_pack_x4, _mm256_store_pack_x2,
    _mm256_store_pack_x3, _mm256_store_pack_x4,
};
use crate::filter1d::arena::Arena;
use crate::filter1d::avx::utils::{
    _mm256_mul_add_epi8_by_ps_x4, _mm256_mul_add_symm_epi8_by_ps_x4, _mm256_mul_epi8_by_ps_x4,
    _mm256_pack_ps_x4_epi8,
};
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::sse::utils::{
    _mm_mul_add_epi8_by_ps_x4, _mm_mul_add_symm_epi8_by_ps_x4, _mm_mul_epi8_by_ps_x4,
    _mm_pack_ps_x4_epi8,
};
use crate::img_size::ImageSize;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::MulAdd;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

pub fn filter_column_avx_symm_u8_f32(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let has_fma = std::arch::is_x86_feature_detected!("fma");
    unsafe {
        if has_fma {
            filter_column_avx_symm_u8_f32_impl_fma(
                arena,
                arena_src,
                dst,
                image_size,
                filter_region,
                scanned_kernel,
            );
        } else {
            filter_column_avx_symm_u8_f32_impl_def(
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

#[target_feature(enable = "avx2")]
unsafe fn filter_column_avx_symm_u8_f32_impl_def(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_column_avx_symm_u8_f32_impl::<false>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn filter_column_avx_symm_u8_f32_impl_fma(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_column_avx_symm_u8_f32_impl::<true>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[inline(always)]
unsafe fn filter_column_avx_symm_u8_f32_impl<const FMA: bool>(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let src = arena_src;

    let arena_width = arena.width * arena.components;

    let image_width = image_size.width * arena.components;

    let dst_stride = image_size.width * arena.components;

    let length = scanned_kernel.len();
    let half_len = length / 2;

    for y in filter_region.start..filter_region.end {
        let mut _cx = 0usize;

        let local_src = src.get_unchecked((y * arena_width)..);

        while _cx + 128 < image_width {
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src = local_src.get_unchecked(_cx..);

            let source = _mm256_load_pack_x4(
                shifted_src
                    .get_unchecked((half_len * arena_width)..)
                    .as_ptr(),
            );
            let mut k0 = _mm256_mul_epi8_by_ps_x4(source.0, coeff);
            let mut k1 = _mm256_mul_epi8_by_ps_x4(source.1, coeff);
            let mut k2 = _mm256_mul_epi8_by_ps_x4(source.2, coeff);
            let mut k3 = _mm256_mul_epi8_by_ps_x4(source.3, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    _mm256_load_pack_x4(shifted_src.get_unchecked(i * arena_width..).as_ptr());
                let v_source1 = _mm256_load_pack_x4(
                    shifted_src.get_unchecked(rollback * arena_width..).as_ptr(),
                );
                k0 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k0, v_source0.0, v_source1.0, coeff);
                k1 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k1, v_source0.1, v_source1.1, coeff);
                k2 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k2, v_source0.2, v_source1.2, coeff);
                k3 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k3, v_source0.3, v_source1.3, coeff);
            }

            let dst_offset = y * dst_stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            _mm256_store_pack_x4(
                dst_ptr0,
                (
                    _mm256_pack_ps_x4_epi8(k0),
                    _mm256_pack_ps_x4_epi8(k1),
                    _mm256_pack_ps_x4_epi8(k2),
                    _mm256_pack_ps_x4_epi8(k3),
                ),
            );
            _cx += 128;
        }

        while _cx + 96 < image_width {
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src = local_src.get_unchecked(_cx..);

            let source = _mm256_load_pack_x3(
                shifted_src
                    .get_unchecked((half_len * arena_width)..)
                    .as_ptr(),
            );
            let mut k0 = _mm256_mul_epi8_by_ps_x4(source.0, coeff);
            let mut k1 = _mm256_mul_epi8_by_ps_x4(source.1, coeff);
            let mut k2 = _mm256_mul_epi8_by_ps_x4(source.2, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    _mm256_load_pack_x3(shifted_src.get_unchecked(i * arena_width..).as_ptr());
                let v_source1 = _mm256_load_pack_x3(
                    shifted_src.get_unchecked(rollback * arena_width..).as_ptr(),
                );
                k0 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k0, v_source0.0, v_source1.0, coeff);
                k1 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k1, v_source0.1, v_source1.1, coeff);
                k2 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k2, v_source0.2, v_source1.2, coeff);
            }

            let dst_offset = y * dst_stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            _mm256_store_pack_x3(
                dst_ptr0,
                (
                    _mm256_pack_ps_x4_epi8(k0),
                    _mm256_pack_ps_x4_epi8(k1),
                    _mm256_pack_ps_x4_epi8(k2),
                ),
            );
            _cx += 96;
        }

        while _cx + 64 < image_width {
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src = local_src.get_unchecked(_cx..);

            let source = _mm256_load_pack_x2(
                shifted_src
                    .get_unchecked((half_len * arena_width)..)
                    .as_ptr(),
            );
            let mut k0 = _mm256_mul_epi8_by_ps_x4(source.0, coeff);
            let mut k1 = _mm256_mul_epi8_by_ps_x4(source.1, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    _mm256_load_pack_x2(shifted_src.get_unchecked(i * arena_width..).as_ptr());
                let v_source1 = _mm256_load_pack_x2(
                    shifted_src.get_unchecked(rollback * arena_width..).as_ptr(),
                );
                k0 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k0, v_source0.0, v_source1.0, coeff);
                k1 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k1, v_source0.1, v_source1.1, coeff);
            }

            let dst_offset = y * dst_stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            _mm256_store_pack_x2(
                dst_ptr0,
                (_mm256_pack_ps_x4_epi8(k0), _mm256_pack_ps_x4_epi8(k1)),
            );
            _cx += 64;
        }

        while _cx + 32 < image_width {
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src = local_src.get_unchecked(_cx..);

            let source = _mm256_loadu_si256(
                shifted_src
                    .get_unchecked((half_len * arena_width)..)
                    .as_ptr() as *const __m256i,
            );
            let mut k0 = _mm256_mul_epi8_by_ps_x4(source, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0 = _mm256_loadu_si256(
                    shifted_src.get_unchecked(i * arena_width..).as_ptr() as *const __m256i,
                );
                let v_source1 = _mm256_loadu_si256(
                    shifted_src.get_unchecked(rollback * arena_width..).as_ptr() as *const __m256i,
                );
                k0 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k0, v_source0, v_source1, coeff);
            }

            let dst_offset = y * dst_stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            _mm256_storeu_si256(dst_ptr0 as *mut __m256i, _mm256_pack_ps_x4_epi8(k0));
            _cx += 32;
        }

        while _cx + 16 < image_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);

            let shifted_src = local_src.get_unchecked(_cx..);

            let source_0 = _mm_loadu_si128(
                shifted_src
                    .get_unchecked((half_len * arena_width)..)
                    .as_ptr() as *const __m128i,
            );
            let mut k0 = _mm_mul_epi8_by_ps_x4(source_0, _mm_set1_ps(coeff.weight));

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                let v_source_0 = _mm_loadu_si128(
                    shifted_src.get_unchecked((i * arena_width)..).as_ptr() as *const __m128i,
                );
                let v_source_1 = _mm_loadu_si128(
                    shifted_src
                        .get_unchecked((rollback * arena_width)..)
                        .as_ptr() as *const __m128i,
                );
                k0 = _mm_mul_add_symm_epi8_by_ps_x4::<FMA>(
                    k0,
                    v_source_0,
                    v_source_1,
                    _mm_set1_ps(coeff.weight),
                );
            }

            let dst_offset = y * dst_stride + _cx;
            let dst_ptr = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            _mm_storeu_si128(dst_ptr as *mut __m128i, _mm_pack_ps_x4_epi8(k0));
            _cx += 16;
        }

        while _cx + 4 < image_width {
            let coeff = scanned_kernel.get_unchecked(half_len).weight;

            let shifted_src = local_src.get_unchecked(_cx..);

            let mut k0 = (*shifted_src.get_unchecked(half_len * arena_width) as f32).mul(coeff);
            let mut k1 =
                (*shifted_src.get_unchecked((half_len * arena_width) + 1) as f32).mul(coeff);
            let mut k2 =
                (*shifted_src.get_unchecked((half_len * arena_width) + 2) as f32).mul(coeff);
            let mut k3 =
                (*shifted_src.get_unchecked((half_len * arena_width) + 3) as f32).mul(coeff);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = MulAdd::mul_add(
                    ((*shifted_src.get_unchecked(i * arena_width)) as f32)
                        .add(*shifted_src.get_unchecked((rollback) * arena_width) as f32),
                    coeff.weight,
                    k0,
                );
                k1 = MulAdd::mul_add(
                    ((*shifted_src.get_unchecked(i * arena_width + 1)) as f32)
                        .add((*shifted_src.get_unchecked(rollback * arena_width + 1)) as f32),
                    coeff.weight,
                    k1,
                );
                k2 = MulAdd::mul_add(
                    ((*shifted_src.get_unchecked(i * arena_width + 2)) as f32)
                        .add((*shifted_src.get_unchecked(rollback * arena_width + 2)) as f32),
                    coeff.weight,
                    k2,
                );
                k3 = MulAdd::mul_add(
                    ((*shifted_src.get_unchecked(i * arena_width + 3)) as f32)
                        .add((*shifted_src.get_unchecked(rollback * arena_width + 3)) as f32),
                    coeff.weight,
                    k3,
                );
            }

            let dst_offset = y * dst_stride + _cx;

            dst.write(dst_offset, k0.to_());
            dst.write(dst_offset + 1, k1.to_());
            dst.write(dst_offset + 2, k2.to_());
            dst.write(dst_offset + 3, k3.to_());
            _cx += 4;
        }

        for x in _cx..image_width {
            let coeff = scanned_kernel.get_unchecked(half_len).weight;

            let shifted_src = local_src.get_unchecked(x..);

            let mut k0 = (*shifted_src.get_unchecked(half_len * arena_width) as f32).mul(coeff);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = MulAdd::mul_add(
                    ((*shifted_src.get_unchecked(i * arena_width)) as f32)
                        .add(*shifted_src.get_unchecked((rollback) * arena_width) as f32),
                    coeff.weight,
                    k0,
                );
            }

            dst.write(y * dst_stride + x, k0.to_());
        }
    }
}
