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
use crate::filter1d::sse::utils::{_mm_mull_add_epi8_by_epi16_x4, _mm_mull_epi8_by_epi16_x4};
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::sse::{
    _mm_load_pack_x2, _mm_load_pack_x3, _mm_load_pack_x4, _mm_store_pack_x2, _mm_store_pack_x3,
    _mm_store_pack_x4,
};
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Mul;

pub(crate) fn filter_column_sse_u8_i16(
    arena: Arena,
    arena_src: &[&[u8]],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i16>],
) {
    unsafe {
        filter_column_sse_u8_i16_impl(
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
unsafe fn filter_column_sse_u8_i16_impl(
    arena: Arena,
    arena_src: &[&[u8]],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i16>],
) {
    let image_width = image_size.width * arena.components;

    let dst_stride = image_size.width * arena.components;

    let y = filter_region.start;

    let length = scanned_kernel.len();

    let mut _cx = 0usize;

    let zeros = _mm_setzero_si128();

    while _cx + 64 < image_width {
        let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(0).weight);

        let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

        let source = _mm_load_pack_x4(v_src.as_ptr());
        let mut k0 = _mm_mull_epi8_by_epi16_x4(source.0, coeff);
        let mut k1 = _mm_mull_epi8_by_epi16_x4(source.1, coeff);
        let mut k2 = _mm_mull_epi8_by_epi16_x4(source.2, coeff);
        let mut k3 = _mm_mull_epi8_by_epi16_x4(source.3, coeff);

        for i in 1..length {
            let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(i).weight);
            let v_source =
                _mm_load_pack_x4(arena_src.get_unchecked(i).get_unchecked(_cx..).as_ptr());
            k0 = _mm_mull_add_epi8_by_epi16_x4(k0, v_source.0, coeff);
            k1 = _mm_mull_add_epi8_by_epi16_x4(k1, v_source.1, coeff);
            k2 = _mm_mull_add_epi8_by_epi16_x4(k2, v_source.2, coeff);
            k3 = _mm_mull_add_epi8_by_epi16_x4(k3, v_source.3, coeff);
        }

        let dst_offset = y * dst_stride + _cx;
        let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
        _mm_store_pack_x4(
            dst_ptr0,
            (
                _mm_packus_epi16(_mm_max_epi16(k0.0, zeros), _mm_max_epi16(k0.1, zeros)),
                _mm_packus_epi16(_mm_max_epi16(k1.0, zeros), _mm_max_epi16(k1.1, zeros)),
                _mm_packus_epi16(_mm_max_epi16(k2.0, zeros), _mm_max_epi16(k2.1, zeros)),
                _mm_packus_epi16(_mm_max_epi16(k3.0, zeros), _mm_max_epi16(k3.1, zeros)),
            ),
        );
        _cx += 64;
    }

    while _cx + 48 < image_width {
        let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(0).weight);

        let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

        let source = _mm_load_pack_x3(v_src.as_ptr());
        let mut k0 = _mm_mull_epi8_by_epi16_x4(source.0, coeff);
        let mut k1 = _mm_mull_epi8_by_epi16_x4(source.1, coeff);
        let mut k2 = _mm_mull_epi8_by_epi16_x4(source.2, coeff);

        for i in 1..length {
            let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(i).weight);
            let v_source =
                _mm_load_pack_x3(arena_src.get_unchecked(i).get_unchecked(_cx..).as_ptr());
            k0 = _mm_mull_add_epi8_by_epi16_x4(k0, v_source.0, coeff);
            k1 = _mm_mull_add_epi8_by_epi16_x4(k1, v_source.1, coeff);
            k2 = _mm_mull_add_epi8_by_epi16_x4(k2, v_source.2, coeff);
        }

        let dst_offset = y * dst_stride + _cx;
        let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
        _mm_store_pack_x3(
            dst_ptr0,
            (
                _mm_packus_epi16(_mm_max_epi16(k0.0, zeros), _mm_max_epi16(k0.1, zeros)),
                _mm_packus_epi16(_mm_max_epi16(k1.0, zeros), _mm_max_epi16(k1.1, zeros)),
                _mm_packus_epi16(_mm_max_epi16(k2.0, zeros), _mm_max_epi16(k2.1, zeros)),
            ),
        );
        _cx += 48;
    }

    while _cx + 32 < image_width {
        let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(0).weight);

        let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

        let source = _mm_load_pack_x2(v_src.as_ptr());
        let mut k0 = _mm_mull_epi8_by_epi16_x4(source.0, coeff);
        let mut k1 = _mm_mull_epi8_by_epi16_x4(source.1, coeff);

        for i in 1..length {
            let coeff = _mm_set1_epi16(scanned_kernel.get_unchecked(i).weight);
            let v_source =
                _mm_load_pack_x2(arena_src.get_unchecked(i).get_unchecked(_cx..).as_ptr());
            k0 = _mm_mull_add_epi8_by_epi16_x4(k0, v_source.0, coeff);
            k1 = _mm_mull_add_epi8_by_epi16_x4(k1, v_source.1, coeff);
        }

        let dst_offset = y * dst_stride + _cx;
        let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
        _mm_store_pack_x2(
            dst_ptr0,
            (
                _mm_packus_epi16(_mm_max_epi16(k0.0, zeros), _mm_max_epi16(k0.1, zeros)),
                _mm_packus_epi16(_mm_max_epi16(k1.0, zeros), _mm_max_epi16(k1.1, zeros)),
            ),
        );

        _cx += 32;
    }

    while _cx + 16 < image_width {
        let coeff = *scanned_kernel.get_unchecked(0);

        let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

        let source_0 = _mm_loadu_si128(v_src.as_ptr() as *const __m128i);
        let mut k0 = _mm_mull_epi8_by_epi16_x4(source_0, _mm_set1_epi16(coeff.weight));

        for i in 1..length {
            let coeff = *scanned_kernel.get_unchecked(i);
            let v_source_0 = _mm_loadu_si128(
                arena_src.get_unchecked(i).get_unchecked(_cx..).as_ptr() as *const __m128i,
            );
            k0 = _mm_mull_add_epi8_by_epi16_x4(k0, v_source_0, _mm_set1_epi16(coeff.weight));
        }

        let dst_offset = y * dst_stride + _cx;
        let dst_ptr = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
        _mm_storeu_si128(
            dst_ptr as *mut __m128i,
            _mm_packus_epi16(_mm_max_epi16(k0.0, zeros), _mm_max_epi16(k0.1, zeros)),
        );
        _cx += 16;
    }

    while _cx + 4 < image_width {
        let coeff = *scanned_kernel.get_unchecked(0);

        let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

        let mut k0 = (*v_src.get_unchecked(0) as i16).mul(coeff.weight);
        let mut k1 = (*v_src.get_unchecked(1) as i16).mul(coeff.weight);
        let mut k2 = (*v_src.get_unchecked(2) as i16).mul(coeff.weight);
        let mut k3 = (*v_src.get_unchecked(3) as i16).mul(coeff.weight);

        for i in 1..length {
            let coeff = *scanned_kernel.get_unchecked(i);
            k0 = mlaf(
                k0,
                (*arena_src.get_unchecked(i).get_unchecked(_cx)) as i16,
                coeff.weight,
            );
            k1 = mlaf(
                k1,
                (*arena_src.get_unchecked(i).get_unchecked(_cx + 1)) as i16,
                coeff.weight,
            );
            k2 = mlaf(
                k2,
                (*arena_src.get_unchecked(i).get_unchecked(_cx + 2)) as i16,
                coeff.weight,
            );
            k3 = mlaf(
                k3,
                (*arena_src.get_unchecked(i).get_unchecked(_cx + 3)) as i16,
                coeff.weight,
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
        let coeff = *scanned_kernel.get_unchecked(0);

        let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

        let mut k0 = ((*v_src.get_unchecked(0)) as i16).mul(coeff.weight);

        for i in 1..length {
            let coeff = *scanned_kernel.get_unchecked(i);
            k0 = mlaf(
                k0,
                (*arena_src.get_unchecked(i).get_unchecked(_cx)) as i16,
                coeff.weight,
            );
        }

        dst.write(y * dst_stride + x, k0.to_());
    }
}
