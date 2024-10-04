/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::filter1d::sse::utils::{_mm_mull_add_epi8_by_epi16_x4, _mm_mull_epi8_by_epi16_x4};
use crate::filter1d::Arena;
use crate::filter2d::scan_point_2d::ScanPoint2d;
use crate::sse::{_mm_load_pack_x2, _mm_load_pack_x4, _mm_store_pack_x2, _mm_store_pack_x4};
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageSize;
use num_traits::MulAdd;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Mul;

pub fn convolve_segment_sse_2d_u8_i16(
    arena: Arena,
    arena_source: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<i16>],
    y: usize,
) {
    unsafe {
        convolve_segment_sse_2d_u8_i16_impl(
            arena,
            arena_source,
            dst,
            image_size,
            prepared_kernel,
            y,
        );
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_segment_sse_2d_u8_i16_impl(
    arena: Arena,
    arena_source: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<i16>],
    y: usize,
) {
    unsafe {
        let width = image_size.width;
        let stride = image_size.width;

        let dx = arena.pad_w as i64;
        let dy = arena.pad_h as i64;

        let arena_width = arena.width;

        let offsets = prepared_kernel
            .iter()
            .map(|&x| {
                arena_source.get_unchecked(
                    ((x.y + dy + y as i64) as usize * arena_width + (x.x + dx) as usize)..,
                )
            })
            .collect::<Vec<_>>();

        let length = prepared_kernel.len();

        let zeros = _mm_setzero_si128();

        let mut _cx = 0usize;

        while _cx + 64 < length {
            let k_weight = _mm_set1_epi16(prepared_kernel.get_unchecked(0).weight);
            let items0 = _mm_load_pack_x4(offsets.get_unchecked(0).get_unchecked(_cx..).as_ptr());
            let mut k0 = _mm_mull_epi8_by_epi16_x4(items0.0, k_weight);
            let mut k1 = _mm_mull_epi8_by_epi16_x4(items0.1, k_weight);
            let mut k2 = _mm_mull_epi8_by_epi16_x4(items0.2, k_weight);
            let mut k3 = _mm_mull_epi8_by_epi16_x4(items0.3, k_weight);
            for i in 1..length {
                let weight = _mm_set1_epi16(prepared_kernel.get_unchecked(i).weight);
                let s_ptr = offsets.get_unchecked(i);
                let items0 = _mm_load_pack_x4(s_ptr.get_unchecked(_cx..).as_ptr());
                k0 = _mm_mull_add_epi8_by_epi16_x4(k0, items0.0, weight);
                k1 = _mm_mull_add_epi8_by_epi16_x4(k1, items0.1, weight);
                k2 = _mm_mull_add_epi8_by_epi16_x4(k2, items0.2, weight);
                k3 = _mm_mull_add_epi8_by_epi16_x4(k3, items0.3, weight);
            }
            let dst_offset = y * stride + _cx;
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

        while _cx + 32 < length {
            let k_weight = _mm_set1_epi16(prepared_kernel.get_unchecked(0).weight);
            let items0 = _mm_load_pack_x2(offsets.get_unchecked(0).get_unchecked(_cx..).as_ptr());
            let mut k0 = _mm_mull_epi8_by_epi16_x4(items0.0, k_weight);
            let mut k1 = _mm_mull_epi8_by_epi16_x4(items0.1, k_weight);
            for i in 1..length {
                let weight = _mm_set1_epi16(prepared_kernel.get_unchecked(i).weight);
                let s_ptr = offsets.get_unchecked(i);
                let items0 = _mm_load_pack_x2(s_ptr.get_unchecked(_cx..).as_ptr());
                k0 = _mm_mull_add_epi8_by_epi16_x4(k0, items0.0, weight);
                k1 = _mm_mull_add_epi8_by_epi16_x4(k1, items0.1, weight);
            }
            let dst_offset = y * stride + _cx;
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

        while _cx + 16 < length {
            let k_weight = _mm_set1_epi16(prepared_kernel.get_unchecked(0).weight);
            let items0 = _mm_loadu_si128(
                offsets.get_unchecked(0).get_unchecked(_cx..).as_ptr() as *const __m128i
            );
            let mut k0 = _mm_mull_epi8_by_epi16_x4(items0, k_weight);
            for i in 1..length {
                let weight = _mm_set1_epi16(prepared_kernel.get_unchecked(i).weight);
                let items0 = _mm_loadu_si128(
                    offsets.get_unchecked(i).get_unchecked(_cx..).as_ptr() as *const __m128i,
                );
                k0 = _mm_mull_add_epi8_by_epi16_x4(k0, items0, weight);
            }
            let dst_offset = y * stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            _mm_storeu_si128(
                dst_ptr0 as *mut __m128i,
                _mm_packus_epi16(_mm_max_epi16(k0.0, zeros), _mm_max_epi16(k0.1, zeros)),
            );
            _cx += 16;
        }

        while _cx + 4 < width {
            let k_weight = prepared_kernel.get_unchecked(0).weight;

            let mut k0 = ((*offsets.get_unchecked(0).get_unchecked(_cx)) as i16).mul(k_weight);
            let mut k1 = ((*offsets.get_unchecked(0).get_unchecked(_cx + 1)) as i16).mul(k_weight);
            let mut k2 = ((*offsets.get_unchecked(0).get_unchecked(_cx + 2)) as i16).mul(k_weight);
            let mut k3 = ((*offsets.get_unchecked(0).get_unchecked(_cx + 3)) as i16).mul(k_weight);

            for i in 1..length {
                let weight = prepared_kernel.get_unchecked(i).weight;
                k0 = ((*offsets.get_unchecked(i).get_unchecked(_cx)) as i16).mul_add(weight, k0);
                k1 =
                    ((*offsets.get_unchecked(i).get_unchecked(_cx + 1)) as i16).mul_add(weight, k1);
                k2 =
                    ((*offsets.get_unchecked(i).get_unchecked(_cx + 2)) as i16).mul_add(weight, k2);
                k3 =
                    ((*offsets.get_unchecked(i).get_unchecked(_cx + 3)) as i16).mul_add(weight, k3);
            }

            let dst_offset = y * stride + _cx;

            dst.write(dst_offset, k0.to_());
            dst.write(dst_offset + 1, k1.to_());
            dst.write(dst_offset + 2, k2.to_());
            dst.write(dst_offset + 3, k3.to_());
            _cx += 4;
        }

        for x in _cx..width {
            let k_weight = prepared_kernel.get_unchecked(0).weight;

            let mut k0 = ((*(*offsets.get_unchecked(0)).get_unchecked(x)) as i16).mul(k_weight);

            for i in 1..length {
                k0 = ((*offsets.get_unchecked(i).get_unchecked(x)) as i16).mul_add(k_weight, k0);
            }
            dst.write(y * stride + x, k0.to_());
        }
    }
}
