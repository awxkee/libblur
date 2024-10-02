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
use crate::avx::{_mm256_load_deinterleave_rgb_ps, _mm256_store_interleave_rgb_ps};
use crate::filter1d::arena::Arena;
use crate::filter1d::avx::utils::_mm256_opt_fmlaf_ps;
use crate::filter1d::color_group::ColorGroup;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::sse::utils::_mm_opt_fmlaf_ps;
use crate::img_size::ImageSize;
use crate::sse::{_mm_load_deinterleave_rgb_ps, _mm_store_interleave_rgb_ps};
use crate::unsafe_slice::UnsafeSlice;
use num_traits::MulAdd;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Mul;

pub fn filter_rgb_row_avx_f32_f32(
    arena: Arena,
    arena_src: &[f32],
    dst: &UnsafeSlice<f32>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let has_fma = std::arch::is_x86_feature_detected!("fma");
        if has_fma {
            filter_rgb_row_avx_f32_f32_fma(
                arena,
                arena_src,
                dst,
                image_size,
                filter_region,
                scanned_kernel,
            );
        } else {
            filter_rgb_row_avx_f32_f32_def(
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
unsafe fn filter_rgb_row_avx_f32_f32_fma(
    arena: Arena,
    arena_src: &[f32],
    dst: &UnsafeSlice<f32>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_rgb_row_avx_f32_f32_impl::<true>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[target_feature(enable = "avx2")]
unsafe fn filter_rgb_row_avx_f32_f32_def(
    arena: Arena,
    arena_src: &[f32],
    dst: &UnsafeSlice<f32>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    filter_rgb_row_avx_f32_f32_impl::<false>(
        arena,
        arena_src,
        dst,
        image_size,
        filter_region,
        scanned_kernel,
    );
}

#[inline(always)]
unsafe fn filter_rgb_row_avx_f32_f32_impl<const FMA: bool>(
    arena: Arena,
    arena_src: &[f32],
    dst: &UnsafeSlice<f32>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let width = image_size.width;

        const N: usize = 3;

        let src = &arena_src;

        let dst_stride = image_size.width * arena.components;

        let arena_width = arena.width * N;

        for y in filter_region.start..filter_region.end {
            let local_src = src.get_unchecked((y * arena_width)..);

            let length = scanned_kernel.iter().len();

            let mut _cx = 0usize;

            while _cx + 8 < width {
                let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(0).weight);

                let shifted_src = local_src.get_unchecked((_cx * N)..);

                let source = _mm256_load_deinterleave_rgb_ps(shifted_src.as_ptr());
                let mut k0 = _mm256_mul_ps(source.0, coeff);
                let mut k1 = _mm256_mul_ps(source.1, coeff);
                let mut k2 = _mm256_mul_ps(source.2, coeff);

                for i in 1..length {
                    let coeff = _mm256_set1_ps(scanned_kernel.get_unchecked(i).weight);
                    let v_source = _mm256_load_deinterleave_rgb_ps(
                        shifted_src.get_unchecked((i * N)..).as_ptr(),
                    );
                    k0 = _mm256_opt_fmlaf_ps::<FMA>(k0, v_source.0, coeff);
                    k1 = _mm256_opt_fmlaf_ps::<FMA>(k1, v_source.1, coeff);
                    k2 = _mm256_opt_fmlaf_ps::<FMA>(k2, v_source.2, coeff);
                }

                let dst_offset = y * dst_stride + _cx * N;
                let dst_ptr0 = (dst.slice.as_ptr() as *mut f32).add(dst_offset);
                _mm256_store_interleave_rgb_ps(dst_ptr0, (k0, k1, k2));
                _cx += 8;
            }

            while _cx + 4 < width {
                let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(0).weight);

                let shifted_src = local_src.get_unchecked((_cx * N)..);

                let source = _mm_load_deinterleave_rgb_ps(shifted_src.as_ptr());
                let mut k0 = _mm_mul_ps(source.0, coeff);
                let mut k1 = _mm_mul_ps(source.1, coeff);
                let mut k2 = _mm_mul_ps(source.2, coeff);

                for i in 1..length {
                    let coeff = _mm_set1_ps(scanned_kernel.get_unchecked(i).weight);
                    let v_source =
                        _mm_load_deinterleave_rgb_ps(shifted_src.get_unchecked((i * N)..).as_ptr());
                    k0 = _mm_opt_fmlaf_ps::<FMA>(k0, v_source.0, coeff);
                    k1 = _mm_opt_fmlaf_ps::<FMA>(k1, v_source.1, coeff);
                    k2 = _mm_opt_fmlaf_ps::<FMA>(k2, v_source.2, coeff);
                }

                let dst_offset = y * dst_stride + _cx * N;
                let dst_ptr0 = (dst.slice.as_ptr() as *mut f32).add(dst_offset);
                _mm_store_interleave_rgb_ps(dst_ptr0, (k0, k1, k2));
                _cx += 4;
            }

            for x in _cx..width {
                let coeff = *scanned_kernel.get_unchecked(0);
                let shifted_src = local_src.get_unchecked((x * N)..);
                let mut k0 = ColorGroup::<N, f32>::from_slice(shifted_src, 0).mul(coeff.weight);

                for i in 1..length {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    k0 = ColorGroup::<N, f32>::from_slice(shifted_src, i * N)
                        .mul_add(k0, coeff.weight);
                }

                k0.to_store(dst, y * dst_stride + x * N);
            }
        }
    }
}
