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
use crate::filter1d::sse::utils::{_mm_madd_epi8_by_epi16_x4, _mm_madd_symm_epi8_by_epi16_x4};
use crate::img_size::ImageSize;
use crate::sse::{_mm_load_pack_x3, _mm_store_pack_x3};
use crate::to_storage::ToStorage;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn filter_row_sse_symm_u8_i16<const N: usize>(
    arena: Arena,
    arena_src: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i16>],
) {
    unsafe {
        filter_row_sse_symm_u8_i16_impl::<N>(
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
unsafe fn filter_row_sse_symm_u8_i16_impl<const N: usize>(
    _: Arena,
    arena_src: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i16>],
) {
    let width = image_size.width;

    let src = arena_src;

    let length = scanned_kernel.len();
    let half_len = length / 2;

    let local_src = src;

    let mut cx = 0usize;

    let max_width = width * N;

    let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(half_len).weight);

    while cx + 48 < max_width {
        let shifted_src = local_src.get_unchecked(cx..);

        let source = _mm_load_pack_x3(shifted_src.get_unchecked(half_len * N..).as_ptr());
        let mut k0 = _mm_madd_epi8_by_epi16_x4(source.0, coeff);
        let mut k1 = _mm_madd_epi8_by_epi16_x4(source.1, coeff);
        let mut k2 = _mm_madd_epi8_by_epi16_x4(source.2, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(i).weight);
            let v_source0 = _mm_load_pack_x3(shifted_src.get_unchecked((i * N)..).as_ptr());
            let v_source1 = _mm_load_pack_x3(shifted_src.get_unchecked((rollback * N)..).as_ptr());
            k0 = _mm_madd_symm_epi8_by_epi16_x4(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm_madd_symm_epi8_by_epi16_x4(k1, v_source0.1, v_source1.1, coeff);
            k2 = _mm_madd_symm_epi8_by_epi16_x4(k2, v_source0.2, v_source1.2, coeff);
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_store_pack_x3(
            dst_ptr0,
            (
                _mm_packus_epi16(k0.0, k0.1),
                _mm_packus_epi16(k1.0, k1.1),
                _mm_packus_epi16(k2.0, k2.1),
            ),
        );
        cx += 48;
    }

    while cx + 16 < max_width {
        let shifted_src = local_src.get_unchecked(cx..);

        let source =
            _mm_loadu_si128(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
        let mut k0 = _mm_madd_epi8_by_epi16_x4(source, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(i).weight);
            let v_source0 =
                _mm_loadu_si128(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
            let v_source1 =
                _mm_loadu_si128(shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _);
            k0 = _mm_madd_symm_epi8_by_epi16_x4(k0, v_source0, v_source1, coeff);
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_storeu_si128(dst_ptr0 as *mut _, _mm_packus_epi16(k0.0, k0.1));
        cx += 16;
    }

    let coeff = *scanned_kernel.get_unchecked(half_len);

    while cx + 4 < max_width {
        let shifted_src = local_src.get_unchecked(cx..);
        let mut k0 = *shifted_src.get_unchecked(half_len * N) as i16 * coeff.weight;
        let mut k1 = *shifted_src.get_unchecked(half_len * N + 1) as i16 * coeff.weight;
        let mut k2 = *shifted_src.get_unchecked(half_len * N + 2) as i16 * coeff.weight;
        let mut k3 = *shifted_src.get_unchecked(half_len * N + 3) as i16 * coeff.weight;

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;

            k0 += (*shifted_src.get_unchecked(i * N) as i16
                + *shifted_src.get_unchecked(rollback * N) as i16)
                * coeff.weight;

            k1 += (*shifted_src.get_unchecked(i * N + 1) as i16
                + *shifted_src.get_unchecked(rollback * N + 1) as i16)
                * coeff.weight;

            k2 += (*shifted_src.get_unchecked(i * N + 2) as i16
                + *shifted_src.get_unchecked(rollback * N + 2) as i16)
                * coeff.weight;

            k3 += (*shifted_src.get_unchecked(i * N + 3) as i16
                + *shifted_src.get_unchecked(rollback * N + 3) as i16)
                * coeff.weight;
        }

        *dst.get_unchecked_mut(cx) = k0.to_();
        *dst.get_unchecked_mut(cx + 1) = k1.to_();
        *dst.get_unchecked_mut(cx + 2) = k2.to_();
        *dst.get_unchecked_mut(cx + 3) = k3.to_();
        cx += 4;
    }

    for x in cx..max_width {
        let shifted_src = local_src.get_unchecked(x..);
        let mut k0 = *shifted_src.get_unchecked(half_len * N) as i16 * coeff.weight;

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;

            k0 += (*shifted_src.get_unchecked(i * N) as i16
                + *shifted_src.get_unchecked(rollback * N) as i16)
                * coeff.weight;
        }

        *dst.get_unchecked_mut(x) = k0.to_();
    }
}
