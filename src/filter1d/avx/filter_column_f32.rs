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
    _mm256_load_pack_ps_x2, _mm256_load_pack_ps_x4, _mm256_store_pack_ps_x2,
    _mm256_store_pack_ps_x4,
};
use crate::filter1d::arena::Arena;
use crate::filter1d::avx::utils::_mm256_opt_fmlaf_ps;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::sse::utils::_mm_opt_fmlaf_ps;
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Mul;

pub(crate) fn filter_column_avx_f32_f32(
    arena: Arena,
    arena_src: &[&[f32]],
    dst: &mut [f32],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let has_fma = std::arch::is_x86_feature_detected!("fma");
        if has_fma {
            filter_column_avx_f32_f32_impl_fma(
                arena,
                arena_src,
                dst,
                image_size,
                filter_region,
                scanned_kernel,
            );
        } else {
            filter_column_avx_f32_f32_impl_def(
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
unsafe fn filter_column_avx_f32_f32_impl_fma(
    arena: Arena,
    arena_src: &[&[f32]],
    dst: &mut [f32],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_column_avx_f32_f32_impl::<true>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[target_feature(enable = "avx2")]
unsafe fn filter_column_avx_f32_f32_impl_def(
    arena: Arena,
    arena_src: &[&[f32]],
    dst: &mut [f32],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_column_avx_f32_f32_impl::<false>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[inline(always)]
unsafe fn filter_column_avx_f32_f32_impl<const FMA: bool>(
    arena: Arena,
    arena_src: &[&[f32]],
    dst: &mut [f32],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();

        let mut cx = 0usize;

        while cx + 32 < dst_stride {
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(0).weight);

            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let source = _mm256_load_pack_ps_x4(v_src.as_ptr());
            let mut k0 = _mm256_mul_ps(source.0, coeff);
            let mut k1 = _mm256_mul_ps(source.1, coeff);
            let mut k2 = _mm256_mul_ps(source.2, coeff);
            let mut k3 = _mm256_mul_ps(source.3, coeff);

            for i in 1..length {
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source =
                    _mm256_load_pack_ps_x4(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                k0 = _mm256_opt_fmlaf_ps::<FMA>(k0, v_source.0, coeff);
                k1 = _mm256_opt_fmlaf_ps::<FMA>(k1, v_source.1, coeff);
                k2 = _mm256_opt_fmlaf_ps::<FMA>(k2, v_source.2, coeff);
                k3 = _mm256_opt_fmlaf_ps::<FMA>(k3, v_source.3, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_store_pack_ps_x4(dst_ptr0, (k0, k1, k2, k3));
            cx += 32;
        }

        while cx + 16 < dst_stride {
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(0).weight);

            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let source = _mm256_load_pack_ps_x2(v_src.as_ptr());
            let mut k0 = _mm256_mul_ps(source.0, coeff);
            let mut k1 = _mm256_mul_ps(source.1, coeff);

            for i in 1..length {
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source =
                    _mm256_load_pack_ps_x2(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                k0 = _mm256_opt_fmlaf_ps::<FMA>(k0, v_source.0, coeff);
                k1 = _mm256_opt_fmlaf_ps::<FMA>(k1, v_source.1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_store_pack_ps_x2(dst_ptr0, (k0, k1));
            cx += 16;
        }

        while cx + 8 < dst_stride {
            let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(0).weight);

            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let source = _mm256_loadu_ps(v_src.as_ptr());
            let mut k0 = _mm256_mul_ps(source, coeff);

            for i in 1..length {
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                let v_source =
                    _mm256_loadu_ps(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                k0 = _mm256_opt_fmlaf_ps::<FMA>(k0, v_source, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_storeu_ps(dst_ptr0, k0);

            cx += 8;
        }

        while cx + 4 < dst_stride {
            let coeff = *scanned_kernel.get_unchecked(0);

            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let source_0 = _mm_loadu_ps(v_src.as_ptr());
            let mut k0 = _mm_mul_ps(source_0, _mm_set1_ps(coeff.weight));

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                let v_source_0 =
                    _mm_loadu_ps(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                k0 = _mm_opt_fmlaf_ps::<FMA>(k0, v_source_0, _mm_set1_ps(coeff.weight));
            }

            let dst_ptr = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm_storeu_ps(dst_ptr, k0);
            cx += 4;
        }

        for x in cx..dst_stride {
            let coeff = *scanned_kernel.get_unchecked(0);

            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let mut k0 = (*v_src.get_unchecked(0)).mul(coeff.weight);

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                k0 = mlaf(
                    k0,
                    *arena_src.get_unchecked(i).get_unchecked(x),
                    coeff.weight,
                );
            }

            *dst.get_unchecked_mut(x) = k0.to_();
        }
    }
}
