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
    _mm256_load_pack_x2, _mm256_load_pack_x4, _mm256_store_pack_x2, _mm256_store_pack_x4,
};
use crate::filter1d::arena::Arena;
use crate::filter1d::avx::sse_utils::{
    _mm_mul_add_symm_epu16_by_epu16_x4, _mm_mul_epu16_widen, _mm_pack_epi32_x2_epu16,
};
use crate::filter1d::avx::utils::{
    _mm256_mul_add_symm_epu16_by_epu16_x4, _mm256_mul_epu16_widen, _mm256_pack_epi32_x4_epu16,
};
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_column_avx_symm_uq15_u16(
    arena: Arena,
    arena_src: &[&[u16]],
    dst: &mut [u16],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<u32>],
) {
    unsafe {
        filter_column_avx_symm_uq15_u16_impl(
            arena,
            arena_src,
            dst,
            image_size,
            filter_region,
            scanned_kernel,
        );
    }
}

#[target_feature(enable = "avx2")]
unsafe fn filter_column_avx_symm_uq15_u16_impl(
    arena: Arena,
    arena_src: &[&[u16]],
    dst: &mut [u16],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<u32>],
) {
    unsafe {
        let image_width = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let mut cx = 0usize;

        while cx + 64 < image_width {
            let coeff = _mm256_set1_epi32(scanned_kernel.get_unchecked(half_len).weight as i32);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = _mm256_load_pack_x4(v_src.as_ptr() as *const _);
            let mut k0 = _mm256_mul_epu16_widen(source.0, coeff);
            let mut k1 = _mm256_mul_epu16_widen(source.1, coeff);
            let mut k2 = _mm256_mul_epu16_widen(source.2, coeff);
            let mut k3 = _mm256_mul_epu16_widen(source.3, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
                let v_source0 = _mm256_load_pack_x4(
                    arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
                );
                let v_source1 = _mm256_load_pack_x4(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr() as *const _,
                );
                k0 = _mm256_mul_add_symm_epu16_by_epu16_x4(k0, v_source0.0, v_source1.0, coeff);
                k1 = _mm256_mul_add_symm_epu16_by_epu16_x4(k1, v_source0.1, v_source1.1, coeff);
                k2 = _mm256_mul_add_symm_epu16_by_epu16_x4(k2, v_source0.2, v_source1.2, coeff);
                k3 = _mm256_mul_add_symm_epu16_by_epu16_x4(k3, v_source0.3, v_source1.3, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_store_pack_x4(
                dst_ptr0 as *mut _,
                (
                    _mm256_pack_epi32_x4_epu16(k0),
                    _mm256_pack_epi32_x4_epu16(k1),
                    _mm256_pack_epi32_x4_epu16(k2),
                    _mm256_pack_epi32_x4_epu16(k3),
                ),
            );
            cx += 64;
        }

        while cx + 32 < image_width {
            let coeff = _mm256_set1_epi32(scanned_kernel.get_unchecked(half_len).weight as i32);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = _mm256_load_pack_x2(v_src.as_ptr() as *const _);
            let mut k0 = _mm256_mul_epu16_widen(source.0, coeff);
            let mut k1 = _mm256_mul_epu16_widen(source.1, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
                let v_source0 = _mm256_load_pack_x2(
                    arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
                );
                let v_source1 = _mm256_load_pack_x2(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr() as *const _,
                );
                k0 = _mm256_mul_add_symm_epu16_by_epu16_x4(k0, v_source0.0, v_source1.0, coeff);
                k1 = _mm256_mul_add_symm_epu16_by_epu16_x4(k1, v_source0.1, v_source1.1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_store_pack_x2(
                dst_ptr0 as *mut _,
                (
                    _mm256_pack_epi32_x4_epu16(k0),
                    _mm256_pack_epi32_x4_epu16(k1),
                ),
            );
            cx += 32;
        }

        while cx + 16 < image_width {
            let coeff = _mm256_set1_epi32(scanned_kernel.get_unchecked(half_len).weight as i32);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = _mm256_loadu_si256(v_src.as_ptr() as *const __m256i);
            let mut k0 = _mm256_mul_epu16_widen(source, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
                let v_source0 = _mm256_loadu_si256(
                    arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m256i,
                );
                let v_source1 = _mm256_loadu_si256(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr() as *const __m256i,
                );
                k0 = _mm256_mul_add_symm_epu16_by_epu16_x4(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_storeu_si256(dst_ptr0 as *mut __m256i, _mm256_pack_epi32_x4_epu16(k0));
            cx += 16;
        }

        while cx + 8 < image_width {
            let coeff = _mm_set1_epi32(scanned_kernel.get_unchecked(half_len).weight as i32);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = _mm_loadu_si128(v_src.as_ptr() as *const __m128i);
            let mut k0 = _mm_mul_epu16_widen(source, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
                let v_source0 = _mm_loadu_si128(
                    arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m128i,
                );
                let v_source1 = _mm_loadu_si128(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr() as *const __m128i,
                );
                k0 = _mm_mul_add_symm_epu16_by_epu16_x4(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm_storeu_si128(dst_ptr0 as *mut __m128i, _mm_pack_epi32_x2_epu16(k0));
            cx += 8;
        }

        while cx + 4 < image_width {
            let coeff = _mm_set1_epi32(scanned_kernel.get_unchecked(half_len).weight as i32);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = _mm_loadu_si64(v_src.as_ptr() as *const _);
            let mut k0 = _mm_mullo_epi32(_mm_unpacklo_epi16(source, _mm_setzero_si128()), coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
                let v_source0 = _mm_loadu_si64(
                    arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
                );
                let v_source1 = _mm_loadu_si64(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr() as *const _,
                );
                k0 = _mm_add_epi32(
                    k0,
                    _mm_mullo_epi32(
                        _mm_add_epi32(
                            _mm_unpacklo_epi16(v_source0, _mm_setzero_si128()),
                            _mm_unpacklo_epi16(v_source1, _mm_setzero_si128()),
                        ),
                        coeff,
                    ),
                );
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            let rnd = _mm_set1_epi32((1 << 14) - 1);
            _mm_storeu_si64(
                dst_ptr0 as *mut _,
                _mm_packus_epi32(
                    _mm_srli_epi32::<15>(_mm_add_epi32(k0, rnd)),
                    _mm_setzero_si128(),
                ),
            );
            cx += 4;
        }

        for x in cx..image_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(x..);

            let mut k0 = ((*v_src.get_unchecked(0)) as u32).mul(coeff.weight);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = ((*arena_src.get_unchecked(i).get_unchecked(x)) as u32)
                    .add((*arena_src.get_unchecked(rollback).get_unchecked(x)) as u32)
                    .mul(coeff.weight)
                    .add(k0);
            }

            *dst.get_unchecked_mut(x) = k0.to_approx_();
        }
    }
}
