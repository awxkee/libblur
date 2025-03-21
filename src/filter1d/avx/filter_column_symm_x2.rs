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
use crate::filter1d::avx::utils::{
    _mm256_mul_add_symm_epi8_by_ps_x4, _mm256_mul_epi8_by_ps_x4, _mm256_pack_ps_x4_epi8,
};
use crate::filter1d::filter_1d_column_handler::FilterBrows;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::sse::utils::{
    _mm_mul_add_symm_epi8_by_ps_x4, _mm_mul_epi8_by_ps_x4, _mm_pack_ps_x4_epi8,
};
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_column_avx_symm_u8_f32_x2(
    arena: Arena,
    brows: FilterBrows<u8>,
    dst: &mut [u8],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let has_fma = std::arch::is_x86_feature_detected!("fma");
    unsafe {
        if has_fma {
            filter_column_avx_symm_u8_f32_impl_fma_x2(
                arena,
                brows,
                dst,
                image_size,
                dst_stride,
                scanned_kernel,
            );
        } else {
            filter_column_avx_symm_u8_f32_impl_def_x2(
                arena,
                brows,
                dst,
                image_size,
                dst_stride,
                scanned_kernel,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn filter_column_avx_symm_u8_f32_impl_def_x2(
    arena: Arena,
    brows: FilterBrows<u8>,
    dst: &mut [u8],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_column_avx_symm_u8_f32_impl::<false>(
        arena,
        brows,
        dst,
        image_size,
        dst_stride,
        scanned_kernel,
    );
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn filter_column_avx_symm_u8_f32_impl_fma_x2(
    arena: Arena,
    brows: FilterBrows<u8>,
    dst: &mut [u8],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_column_avx_symm_u8_f32_impl::<true>(
        arena,
        brows,
        dst,
        image_size,
        dst_stride,
        scanned_kernel,
    );
}

#[inline(always)]
unsafe fn filter_column_avx_symm_u8_f32_impl<const FMA: bool>(
    arena: Arena,
    brows: FilterBrows<u8>,
    dst: &mut [u8],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let image_width = image_size.width * arena.components;

    let length = scanned_kernel.len();
    let half_len = length / 2;

    let mut cx = 0usize;

    let (dst0, dst1) = dst.split_at_mut(dst_stride);

    let brows0 = brows.brows[0];
    let brows1 = brows.brows[1];

    while cx + 32 < image_width {
        let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

        let shifted_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
        let shifted_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);

        let source0 = _mm256_loadu_si256(shifted_src0.as_ptr() as *const __m256i);
        let source1 = _mm256_loadu_si256(shifted_src1.as_ptr() as *const __m256i);

        let mut k0 = _mm256_mul_epi8_by_ps_x4::<FMA>(source0, coeff);
        let mut k1 = _mm256_mul_epi8_by_ps_x4::<FMA>(source1, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
            let v_source0_0 = _mm256_loadu_si256(
                brows0.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m256i,
            );
            let v_source1_0 = _mm256_loadu_si256(
                brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr() as *const __m256i,
            );
            let v_source0_1 = _mm256_loadu_si256(
                brows1.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m256i,
            );
            let v_source1_1 = _mm256_loadu_si256(
                brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr() as *const __m256i,
            );
            k0 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k0, v_source0_0, v_source1_0, coeff);
            k1 = _mm256_mul_add_symm_epi8_by_ps_x4::<FMA>(k1, v_source0_1, v_source1_1, coeff);
        }

        let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
        let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();

        _mm256_storeu_si256(dst_ptr0 as *mut __m256i, _mm256_pack_ps_x4_epi8(k0));
        _mm256_storeu_si256(dst_ptr1 as *mut __m256i, _mm256_pack_ps_x4_epi8(k1));

        cx += 32;
    }

    while cx + 16 < image_width {
        let coeff = *scanned_kernel.get_unchecked(half_len);

        let shifted_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
        let shifted_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);

        let source_0 = _mm_loadu_si128(shifted_src0.as_ptr() as *const __m128i);
        let source_1 = _mm_loadu_si128(shifted_src1.as_ptr() as *const __m128i);

        let mut k0 = _mm_mul_epi8_by_ps_x4::<FMA>(source_0, _mm_set1_ps(coeff.weight));
        let mut k1 = _mm_mul_epi8_by_ps_x4::<FMA>(source_1, _mm_set1_ps(coeff.weight));

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            let v_source_0_0 = _mm_loadu_si128(
                brows0.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m128i,
            );
            let v_source_1_0 = _mm_loadu_si128(
                brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr() as *const __m128i,
            );
            let v_source_0_1 = _mm_loadu_si128(
                brows1.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m128i,
            );
            let v_source_1_1 = _mm_loadu_si128(
                brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr() as *const __m128i,
            );
            k0 = _mm_mul_add_symm_epi8_by_ps_x4::<FMA>(
                k0,
                v_source_0_0,
                v_source_1_0,
                _mm_set1_ps(coeff.weight),
            );
            k1 = _mm_mul_add_symm_epi8_by_ps_x4::<FMA>(
                k1,
                v_source_0_1,
                v_source_1_1,
                _mm_set1_ps(coeff.weight),
            );
        }

        let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
        let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();

        _mm_storeu_si128(dst_ptr0 as *mut __m128i, _mm_pack_ps_x4_epi8(k0));
        _mm_storeu_si128(dst_ptr1 as *mut __m128i, _mm_pack_ps_x4_epi8(k1));

        cx += 16;
    }

    while cx + 4 < image_width {
        let coeff = scanned_kernel.get_unchecked(half_len).weight;

        let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
        let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);

        let mut k0_0 = (*v_src0.get_unchecked(0) as f32).mul(coeff);
        let mut k1_0 = (*v_src0.get_unchecked(1) as f32).mul(coeff);
        let mut k2_0 = (*v_src0.get_unchecked(2) as f32).mul(coeff);
        let mut k3_0 = (*v_src0.get_unchecked(3) as f32).mul(coeff);

        let mut k0_1 = (*v_src1.get_unchecked(0) as f32).mul(coeff);
        let mut k1_1 = (*v_src1.get_unchecked(1) as f32).mul(coeff);
        let mut k2_1 = (*v_src1.get_unchecked(2) as f32).mul(coeff);
        let mut k3_1 = (*v_src1.get_unchecked(3) as f32).mul(coeff);

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            k0_0 = mlaf(
                k0_0,
                ((*brows0.get_unchecked(i).get_unchecked(cx)) as f32)
                    .add(*brows0.get_unchecked(rollback).get_unchecked(cx) as f32),
                coeff.weight,
            );
            k1_0 = mlaf(
                k1_0,
                ((*brows0.get_unchecked(i).get_unchecked(cx + 1)) as f32)
                    .add(*brows0.get_unchecked(rollback).get_unchecked(cx + 1) as f32),
                coeff.weight,
            );
            k2_0 = mlaf(
                k2_0,
                ((*brows0.get_unchecked(i).get_unchecked(cx + 2)) as f32)
                    .add(*brows0.get_unchecked(rollback).get_unchecked(cx + 2) as f32),
                coeff.weight,
            );
            k3_0 = mlaf(
                k3_0,
                ((*brows0.get_unchecked(i).get_unchecked(cx + 3)) as f32)
                    .add(*brows0.get_unchecked(rollback).get_unchecked(cx + 3) as f32),
                coeff.weight,
            );

            k0_1 = mlaf(
                k0_1,
                ((*brows1.get_unchecked(i).get_unchecked(cx)) as f32)
                    .add(*brows1.get_unchecked(rollback).get_unchecked(cx) as f32),
                coeff.weight,
            );
            k1_1 = mlaf(
                k1_1,
                ((*brows1.get_unchecked(i).get_unchecked(cx + 1)) as f32)
                    .add(*brows1.get_unchecked(rollback).get_unchecked(cx + 1) as f32),
                coeff.weight,
            );
            k2_1 = mlaf(
                k2_1,
                ((*brows1.get_unchecked(i).get_unchecked(cx + 2)) as f32)
                    .add(*brows1.get_unchecked(rollback).get_unchecked(cx + 2) as f32),
                coeff.weight,
            );
            k3_1 = mlaf(
                k3_1,
                ((*brows1.get_unchecked(i).get_unchecked(cx + 3)) as f32)
                    .add(*brows1.get_unchecked(rollback).get_unchecked(cx + 3) as f32),
                coeff.weight,
            );
        }

        *dst0.get_unchecked_mut(cx) = k0_0.to_();
        *dst0.get_unchecked_mut(cx + 1) = k1_0.to_();
        *dst0.get_unchecked_mut(cx + 2) = k2_0.to_();
        *dst0.get_unchecked_mut(cx + 3) = k3_0.to_();

        *dst1.get_unchecked_mut(cx) = k0_1.to_();
        *dst1.get_unchecked_mut(cx + 1) = k1_1.to_();
        *dst1.get_unchecked_mut(cx + 2) = k2_1.to_();
        *dst1.get_unchecked_mut(cx + 3) = k3_1.to_();
        cx += 4;
    }

    for x in cx..image_width {
        let coeff = scanned_kernel.get_unchecked(half_len).weight;

        let v_src0 = brows0.get_unchecked(half_len).get_unchecked(x..);
        let v_src1 = brows1.get_unchecked(half_len).get_unchecked(x..);

        let mut k0 = (*v_src0.get_unchecked(0) as f32).mul(coeff);
        let mut k1 = (*v_src1.get_unchecked(0) as f32).mul(coeff);

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;
            k0 = mlaf(
                k0,
                ((*brows0.get_unchecked(i).get_unchecked(x)) as f32)
                    .add(*brows0.get_unchecked(rollback).get_unchecked(x) as f32),
                coeff.weight,
            );

            k1 = mlaf(
                k1,
                ((*brows1.get_unchecked(i).get_unchecked(x)) as f32)
                    .add(*brows1.get_unchecked(rollback).get_unchecked(x) as f32),
                coeff.weight,
            );
        }

        *dst0.get_unchecked_mut(x) = k0.to_();
        *dst1.get_unchecked_mut(x) = k0.to_();
    }
}
