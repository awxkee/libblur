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
use crate::avx::{_mm256_load_pack_x2, _mm256_store_pack_x2};
use crate::filter1d::arena::Arena;
use crate::filter1d::avx::utils::{
    _mm256_mul_add_symm_epi16_by_ps_x2, _mm256_mul_epi16_by_ps_x2, _mm256_pack_ps_x2_epi16,
};
use crate::filter1d::filter_1d_column_handler::FilterBrows;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::sse::utils::{
    _mm_mul_add_symm_epi16_by_ps, _mm_mul_add_symm_epi16_by_ps_x2, _mm_mul_epi16_by_ps,
    _mm_mul_epi16_by_ps_x2, _mm_pack_ps_epi16, _mm_pack_ps_x2_epi16,
};
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_column_avx_symm_u16_f32_x2(
    arena: Arena,
    brows: FilterBrows<u16>,
    dst: &mut [u16],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let has_fma = std::arch::is_x86_feature_detected!("fma");
    unsafe {
        if has_fma {
            filter_column_avx_symm_u16_f32_impl_fma_x2(
                arena,
                brows,
                dst,
                image_size,
                dst_stride,
                scanned_kernel,
            );
        } else {
            filter_column_avx_symm_u16_f32_impl_def_x2(
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
unsafe fn filter_column_avx_symm_u16_f32_impl_def_x2(
    arena: Arena,
    brows: FilterBrows<u16>,
    dst: &mut [u16],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let unit = ExecutionUnit::<false>::default();
    unit.pass(arena, brows, dst, image_size, dst_stride, scanned_kernel);
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn filter_column_avx_symm_u16_f32_impl_fma_x2(
    arena: Arena,
    brows: FilterBrows<u16>,
    dst: &mut [u16],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let unit = ExecutionUnit::<true>::default();
    unit.pass(arena, brows, dst, image_size, dst_stride, scanned_kernel);
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        arena: Arena,
        brows: FilterBrows<u16>,
        dst: &mut [u16],
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

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = _mm256_load_pack_x2(v_src0.as_ptr() as *const _);
            let source1 = _mm256_load_pack_x2(v_src1.as_ptr() as *const _);

            let mut k0_0 = _mm256_mul_epi16_by_ps_x2::<FMA>(source0.0, coeff);
            let mut k1_0 = _mm256_mul_epi16_by_ps_x2::<FMA>(source0.1, coeff);

            let mut k0_1 = _mm256_mul_epi16_by_ps_x2::<FMA>(source1.0, coeff);
            let mut k1_1 = _mm256_mul_epi16_by_ps_x2::<FMA>(source1.1, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0_0 = _mm256_load_pack_x2(
                    brows0.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
                );
                let v_source1_0 = _mm256_load_pack_x2(
                    brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr() as *const _,
                );

                let v_source0_1 = _mm256_load_pack_x2(
                    brows1.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
                );
                let v_source1_1 = _mm256_load_pack_x2(
                    brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr() as *const _,
                );

                k0_0 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(
                    k0_0,
                    v_source0_0.0,
                    v_source1_0.0,
                    coeff,
                );
                k1_0 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(
                    k1_0,
                    v_source0_0.1,
                    v_source1_0.1,
                    coeff,
                );

                k0_1 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(
                    k0_0,
                    v_source0_1.0,
                    v_source1_1.0,
                    coeff,
                );
                k1_1 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(
                    k1_0,
                    v_source0_1.1,
                    v_source1_1.1,
                    coeff,
                );
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();

            _mm256_store_pack_x2(
                dst_ptr0 as *mut _,
                (_mm256_pack_ps_x2_epi16(k0_0), _mm256_pack_ps_x2_epi16(k1_0)),
            );

            _mm256_store_pack_x2(
                dst_ptr1 as *mut _,
                (_mm256_pack_ps_x2_epi16(k0_1), _mm256_pack_ps_x2_epi16(k1_1)),
            );
            cx += 32;
        }

        while cx + 16 < image_width {
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = _mm256_loadu_si256(v_src0.as_ptr() as *const __m256i);
            let source1 = _mm256_loadu_si256(v_src1.as_ptr() as *const __m256i);

            let mut k0 = _mm256_mul_epi16_by_ps_x2::<FMA>(source0, coeff);
            let mut k1 = _mm256_mul_epi16_by_ps_x2::<FMA>(source1, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0_0 = _mm256_loadu_si256(
                    brows0.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m256i,
                );
                let v_source1_0 =
                    _mm256_loadu_si256(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr()
                        as *const __m256i);
                let v_source0_1 = _mm256_loadu_si256(
                    brows1.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m256i,
                );
                let v_source1_1 =
                    _mm256_loadu_si256(brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr()
                        as *const __m256i);
                k0 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(k0, v_source0_0, v_source1_0, coeff);
                k1 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(k1, v_source0_1, v_source1_1, coeff);
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();

            _mm256_storeu_si256(dst_ptr0 as *mut __m256i, _mm256_pack_ps_x2_epi16(k0));
            _mm256_storeu_si256(dst_ptr1 as *mut __m256i, _mm256_pack_ps_x2_epi16(k1));

            cx += 16;
        }

        while cx + 8 < image_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);

            let source_0 = _mm_loadu_si128(v_src0.as_ptr() as *const __m128i);
            let source_1 = _mm_loadu_si128(v_src1.as_ptr() as *const __m128i);

            let mut k0 = _mm_mul_epi16_by_ps_x2::<FMA>(source_0, _mm_set1_ps(coeff.weight));
            let mut k1 = _mm_mul_epi16_by_ps_x2::<FMA>(source_1, _mm_set1_ps(coeff.weight));

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                let v_source0_0 = _mm_loadu_si128(
                    brows0.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m128i,
                );
                let v_source1_0 =
                    _mm_loadu_si128(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr()
                        as *const __m128i);
                let v_source0_1 = _mm_loadu_si128(
                    brows0.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m128i,
                );
                let v_source1_1 =
                    _mm_loadu_si128(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr()
                        as *const __m128i);
                k0 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(
                    k0,
                    v_source0_0,
                    v_source1_0,
                    _mm_set1_ps(coeff.weight),
                );
                k1 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(
                    k1,
                    v_source0_1,
                    v_source1_1,
                    _mm_set1_ps(coeff.weight),
                );
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();

            _mm_storeu_si128(dst_ptr0 as *mut __m128i, _mm_pack_ps_x2_epi16(k0));
            _mm_storeu_si128(dst_ptr1 as *mut __m128i, _mm_pack_ps_x2_epi16(k1));

            cx += 8;
        }

        while cx + 4 < image_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);

            let source_0 = _mm_loadu_si64(v_src0.as_ptr() as *const _);
            let source_1 = _mm_loadu_si64(v_src1.as_ptr() as *const _);

            let mut k0 = _mm_mul_epi16_by_ps::<FMA>(source_0, _mm_set1_ps(coeff.weight));
            let mut k1 = _mm_mul_epi16_by_ps::<FMA>(source_1, _mm_set1_ps(coeff.weight));

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                let v_source0_0 = _mm_loadu_si64(
                    brows0.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
                );
                let v_source1_0 = _mm_loadu_si64(
                    brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr() as *const _,
                );
                let v_source0_1 = _mm_loadu_si64(
                    brows1.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
                );
                let v_source1_1 = _mm_loadu_si64(
                    brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr() as *const _,
                );
                k0 = _mm_mul_add_symm_epi16_by_ps::<FMA>(
                    k0,
                    v_source0_0,
                    v_source1_0,
                    _mm_set1_ps(coeff.weight),
                );
                k1 = _mm_mul_add_symm_epi16_by_ps::<FMA>(
                    k1,
                    v_source0_1,
                    v_source1_1,
                    _mm_set1_ps(coeff.weight),
                );
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();

            _mm_storeu_si64(dst_ptr0 as *mut _, _mm_pack_ps_epi16(k0));
            _mm_storeu_si64(dst_ptr1 as *mut _, _mm_pack_ps_epi16(k1));

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
            *dst1.get_unchecked_mut(x) = k1.to_();
        }
    }
}
