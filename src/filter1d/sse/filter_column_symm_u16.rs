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
use crate::filter1d::sse::utils::{
    _mm_mul_add_symm_epi16_by_ps, _mm_mul_add_symm_epi16_by_ps_x2, _mm_mul_epi16_by_ps,
    _mm_mul_epi16_by_ps_x2, _mm_pack_ps_epi16, _mm_pack_ps_x2_epi16,
};
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::sse::{
    _mm_load_pack_x2, _mm_load_pack_x3, _mm_load_pack_x4, _mm_store_pack_x2, _mm_store_pack_x3,
    _mm_store_pack_x4,
};
use crate::to_storage::ToStorage;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_column_symm_sse_u16_f32(
    arena: Arena,
    arena_src: &[&[u16]],
    dst: &mut [u16],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let has_fma = std::arch::is_x86_feature_detected!("fma");
    unsafe {
        if has_fma {
            filter_column_symm_sse_u16_f32_fma(
                arena,
                arena_src,
                dst,
                image_size,
                filter_region,
                scanned_kernel,
            );
        } else {
            filter_column_symm_sse_u16_f32_def(
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

#[target_feature(enable = "sse4.1", enable = "fma")]
unsafe fn filter_column_symm_sse_u16_f32_fma(
    arena: Arena,
    arena_src: &[&[u16]],
    dst: &mut [u16],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_column_symm_sse_u16_f32_impl::<true>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[target_feature(enable = "sse4.1")]
unsafe fn filter_column_symm_sse_u16_f32_def(
    arena: Arena,
    arena_src: &[&[u16]],
    dst: &mut [u16],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_column_symm_sse_u16_f32_impl::<false>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[inline(always)]
unsafe fn filter_column_symm_sse_u16_f32_impl<const FMA: bool>(
    arena: Arena,
    arena_src: &[&[u16]],
    dst: &mut [u16],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let image_width = image_size.width * arena.components;

    let length = scanned_kernel.len();
    let half_len = length / 2;

    let mut cx = 0usize;

    let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

    while cx + 32 < image_width {
        let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

        let source = _mm_load_pack_x4(v_src.as_ptr() as *const _);
        let mut k0 = _mm_mul_epi16_by_ps_x2::<FMA>(source.0, coeff);
        let mut k1 = _mm_mul_epi16_by_ps_x2::<FMA>(source.1, coeff);
        let mut k2 = _mm_mul_epi16_by_ps_x2::<FMA>(source.2, coeff);
        let mut k3 = _mm_mul_epi16_by_ps_x2::<FMA>(source.3, coeff);

        for i in 0..half_len {
            let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(i).weight);
            let rollback = length - i - 1;
            let v_source0 = _mm_load_pack_x4(
                arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
            );
            let v_source1 = _mm_load_pack_x4(
                arena_src
                    .get_unchecked(rollback)
                    .get_unchecked(cx..)
                    .as_ptr() as *const _,
            );
            k0 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k1, v_source0.1, v_source1.1, coeff);
            k2 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k2, v_source0.2, v_source1.2, coeff);
            k3 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k3, v_source0.3, v_source1.3, coeff);
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_store_pack_x4(
            dst_ptr0 as *mut _,
            (
                _mm_pack_ps_x2_epi16(k0),
                _mm_pack_ps_x2_epi16(k1),
                _mm_pack_ps_x2_epi16(k2),
                _mm_pack_ps_x2_epi16(k3),
            ),
        );
        cx += 32;
    }

    while cx + 24 < image_width {
        let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

        let source = _mm_load_pack_x3(v_src.as_ptr() as *const _);
        let mut k0 = _mm_mul_epi16_by_ps_x2::<FMA>(source.0, coeff);
        let mut k1 = _mm_mul_epi16_by_ps_x2::<FMA>(source.1, coeff);
        let mut k2 = _mm_mul_epi16_by_ps_x2::<FMA>(source.2, coeff);

        for i in 0..half_len {
            let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(i).weight);
            let rollback = length - i - 1;
            let v_source0 = _mm_load_pack_x3(
                arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
            );
            let v_source1 = _mm_load_pack_x3(
                arena_src
                    .get_unchecked(rollback)
                    .get_unchecked(cx..)
                    .as_ptr() as *const _,
            );
            k0 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k1, v_source0.1, v_source1.1, coeff);
            k2 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k2, v_source0.2, v_source1.2, coeff);
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_store_pack_x3(
            dst_ptr0 as *mut _,
            (
                _mm_pack_ps_x2_epi16(k0),
                _mm_pack_ps_x2_epi16(k1),
                _mm_pack_ps_x2_epi16(k2),
            ),
        );
        cx += 24;
    }

    while cx + 16 < image_width {
        let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

        let source = _mm_load_pack_x2(v_src.as_ptr() as *const _);
        let mut k0 = _mm_mul_epi16_by_ps_x2::<FMA>(source.0, coeff);
        let mut k1 = _mm_mul_epi16_by_ps_x2::<FMA>(source.1, coeff);

        for i in 0..half_len {
            let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(i).weight);
            let rollback = length - i - 1;
            let v_source0 = _mm_load_pack_x2(
                arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
            );
            let v_source1 = _mm_load_pack_x2(
                arena_src
                    .get_unchecked(rollback)
                    .get_unchecked(cx..)
                    .as_ptr() as *const _,
            );
            k0 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k1, v_source0.1, v_source1.1, coeff);
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_store_pack_x2(
            dst_ptr0 as *mut _,
            (_mm_pack_ps_x2_epi16(k0), _mm_pack_ps_x2_epi16(k1)),
        );

        cx += 16;
    }

    while cx + 8 < image_width {
        let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

        let source_0 = _mm_loadu_si128(v_src.as_ptr() as *const __m128i);
        let mut k0 = _mm_mul_epi16_by_ps_x2::<FMA>(source_0, coeff);

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            let v_source_0 = _mm_loadu_si128(
                arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m128i,
            );
            let v_source_1 = _mm_loadu_si128(
                arena_src
                    .get_unchecked(rollback)
                    .get_unchecked(cx..)
                    .as_ptr() as *const __m128i,
            );
            k0 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(
                k0,
                v_source_0,
                v_source_1,
                _mm_set1_ps(coeff.weight),
            );
        }

        let dst_ptr = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_storeu_si128(dst_ptr as *mut __m128i, _mm_pack_ps_x2_epi16(k0));
        cx += 8;
    }

    let coeff = *scanned_kernel.get_unchecked(half_len);

    while cx + 4 < image_width {
        let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

        let source_0 = _mm_loadu_si64(v_src.as_ptr() as *const _);
        let mut k0 = _mm_mul_epi16_by_ps::<FMA>(source_0, _mm_set1_ps(coeff.weight));

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            let v_source_0 =
                _mm_loadu_si64(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _);
            let v_source_1 = _mm_loadu_si64(
                arena_src
                    .get_unchecked(rollback)
                    .get_unchecked(cx..)
                    .as_ptr() as *const _,
            );
            k0 = _mm_mul_add_symm_epi16_by_ps::<FMA>(
                k0,
                v_source_0,
                v_source_1,
                _mm_set1_ps(coeff.weight),
            );
        }

        let dst_ptr = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_storeu_si64(dst_ptr as *mut _, _mm_pack_ps_epi16(k0));
        cx += 4;
    }

    for x in cx..image_width {
        let v_src = arena_src.get_unchecked(half_len).get_unchecked(x..);

        let mut k0 = (*v_src.get_unchecked(0) as f32).mul(coeff.weight);

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            k0 = mlaf(
                k0,
                ((*arena_src.get_unchecked(i).get_unchecked(x)) as f32)
                    .add(*arena_src.get_unchecked(rollback).get_unchecked(x) as f32),
                coeff.weight,
            );
        }

        *dst.get_unchecked_mut(x) = k0.to_();
    }
}
