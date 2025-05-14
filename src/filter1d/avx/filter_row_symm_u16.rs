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
    _mm_mul_add_symm_epi16_by_ps, _mm_mul_add_symm_epi16_by_ps_x2, _mm_mul_epi16_by_ps,
    _mm_mul_epi16_by_ps_x2, _mm_pack_ps_epi16, _mm_pack_ps_x2_epi16,
};
use crate::filter1d::avx::utils::{
    _mm256_mul_add_symm_epi16_by_ps_x2, _mm256_mul_epi16_by_ps_x2, _mm256_pack_ps_x2_epi16,
};
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn filter_row_avx_symm_u16_f32<const N: usize>(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [u16],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let has_fma = std::arch::is_x86_feature_detected!("fma");
        if has_fma {
            filter_row_avx_symm_u16_f32_fma::<N>(
                arena,
                arena_src,
                dst,
                image_size,
                filter_region,
                scanned_kernel,
            );
        } else {
            filter_row_avx_u16_f32_def::<N>(
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
unsafe fn filter_row_avx_symm_u16_f32_fma<const N: usize>(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [u16],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let execution = RowSymmExecutionUnit::<true, N>::default();
    execution.pass(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[target_feature(enable = "avx2")]
unsafe fn filter_row_avx_u16_f32_def<const N: usize>(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [u16],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    let execution = RowSymmExecutionUnit::<false, N>::default();
    execution.pass(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[derive(Debug, Default, Copy, Clone)]
struct RowSymmExecutionUnit<const FMA: bool, const N: usize> {}

impl<const FMA: bool, const N: usize> RowSymmExecutionUnit<FMA, N> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        _: Arena,
        arena_src: &[u16],
        dst: &mut [u16],
        image_size: ImageSize,
        _: FilterRegion,
        scanned_kernel: &[ScanPoint1d<f32>],
    ) {
        let width = image_size.width;

        let src = &arena_src;

        let max_width = width * N;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let local_src = src;

        let mut cx = 0usize;

        let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(half_len).weight);

        while cx + 64 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let source =
                _mm256_load_pack_x4(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
            let mut k0 = _mm256_mul_epi16_by_ps_x2::<FMA>(source.0, coeff);
            let mut k1 = _mm256_mul_epi16_by_ps_x2::<FMA>(source.1, coeff);
            let mut k2 = _mm256_mul_epi16_by_ps_x2::<FMA>(source.2, coeff);
            let mut k3 = _mm256_mul_epi16_by_ps_x2::<FMA>(source.3, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    _mm256_load_pack_x4(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                let v_source1 = _mm256_load_pack_x4(
                    shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                );
                k0 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(k0, v_source0.0, v_source1.0, coeff);
                k1 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(k1, v_source0.1, v_source1.1, coeff);
                k2 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(k2, v_source0.2, v_source1.2, coeff);
                k3 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(k3, v_source0.3, v_source1.3, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_store_pack_x4(
                dst_ptr0 as *mut _,
                (
                    _mm256_pack_ps_x2_epi16(k0),
                    _mm256_pack_ps_x2_epi16(k1),
                    _mm256_pack_ps_x2_epi16(k2),
                    _mm256_pack_ps_x2_epi16(k3),
                ),
            );
            cx += 64;
        }

        while cx + 32 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let source =
                _mm256_load_pack_x2(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
            let mut k0 = _mm256_mul_epi16_by_ps_x2::<FMA>(source.0, coeff);
            let mut k1 = _mm256_mul_epi16_by_ps_x2::<FMA>(source.1, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    _mm256_load_pack_x2(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                let v_source1 = _mm256_load_pack_x2(
                    shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                );
                k0 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(k0, v_source0.0, v_source1.0, coeff);
                k1 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(k1, v_source0.1, v_source1.1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_store_pack_x2(
                dst_ptr0 as *mut _,
                (_mm256_pack_ps_x2_epi16(k0), _mm256_pack_ps_x2_epi16(k1)),
            );
            cx += 32;
        }

        while cx + 16 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let source =
                _mm256_loadu_si256(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
            let mut k0 = _mm256_mul_epi16_by_ps_x2::<FMA>(source, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    _mm256_loadu_si256(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                let v_source1 = _mm256_loadu_si256(
                    shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                );
                k0 = _mm256_mul_add_symm_epi16_by_ps_x2::<FMA>(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_storeu_si256(dst_ptr0 as *mut _, _mm256_pack_ps_x2_epi16(k0));
            cx += 16;
        }

        while cx + 8 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let source =
                _mm_loadu_si128(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
            let mut k0 = _mm_mul_epi16_by_ps_x2::<FMA>(source, _mm256_castps256_ps128(coeff));

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    _mm_loadu_si128(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                let v_source1 = _mm_loadu_si128(
                    shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                );
                k0 = _mm_mul_add_symm_epi16_by_ps_x2::<FMA>(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm_storeu_si128(dst_ptr0 as *mut _, _mm_pack_ps_x2_epi16(k0));
            cx += 8;
        }

        while cx + 4 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let source =
                _mm_loadu_si64(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
            let mut k0 = _mm_mul_epi16_by_ps::<FMA>(source, _mm256_castps256_ps128(coeff));

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    _mm_loadu_si64(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                let v_source1 = _mm_loadu_si64(
                    shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                );
                k0 = _mm_mul_add_symm_epi16_by_ps::<FMA>(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm_storeu_si64(dst_ptr0 as *mut _, _mm_pack_ps_epi16(k0));
            cx += 4;
        }

        let coeff = *scanned_kernel.get_unchecked(half_len);

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);
            let mut k0 = *shifted_src.get_unchecked(half_len * N) as f32 * coeff.weight;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 = mlaf(
                    k0,
                    *shifted_src.get_unchecked(i * N) as f32
                        + *shifted_src.get_unchecked(rollback * N) as f32,
                    coeff.weight,
                );
            }

            use crate::to_storage::ToStorage;
            *dst.get_unchecked_mut(x) = k0.to_();
        }
    }
}
