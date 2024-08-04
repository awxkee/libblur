// Copyright (c) Radzivon Bartoshyk. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use erydanos::_mm256_prefer_fma_ps;

use crate::gaussian::avx::utils::{avx2_pack_u16, avx2_pack_u32, load_u8_f32_fast, load_u8_u32_one};
use crate::unsafe_slice::UnsafeSlice;

pub fn gaussian_blur_vertical_pass_impl_avx<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undef_unsafe_dst) };
    let half_kernel = (kernel_size / 2) as i32;

    let zeros = unsafe { _mm256_setzero_ps() };
    let total_length = width as usize * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut cx = 0usize;

        unsafe {
            while cx + 64 < total_length {
                let mut store0 = zeros;
                let mut store1 = zeros;
                let mut store2 = zeros;
                let mut store3 = zeros;
                let mut store4 = zeros;
                let mut store5 = zeros;
                let mut store6 = zeros;
                let mut store7 = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm256_set1_ps(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let row_0 = _mm256_loadu_si256(s_ptr as *const __m256i);
                    let row_1 = _mm256_loadu_si256(s_ptr.add(32) as *const __m256i);

                    let low_part = _mm256_castsi256_si128(row_0);

                    let hi_16 = _mm_unpackhi_epi8(low_part, _mm_setzero_si128());
                    let lo_16 = _mm_unpacklo_epi8(low_part, _mm_setzero_si128());

                    let lo_32 = _mm256_cvtepu16_epi32(lo_16);
                    let hi_32 = _mm256_cvtepu16_epi32(hi_16);

                    let lo_f = _mm256_cvtepi32_ps(lo_32);
                    let hi_f = _mm256_cvtepi32_ps(hi_32);

                    store0 = _mm256_prefer_fma_ps(store0, lo_f, f_weight);
                    store1 = _mm256_prefer_fma_ps(store1, hi_f, f_weight);

                    let high_part = _mm256_extracti128_si256::<1>(row_0);

                    let hi_16 = _mm_unpackhi_epi8(high_part, _mm_setzero_si128());
                    let lo_16 = _mm_unpacklo_epi8(high_part, _mm_setzero_si128());

                    let lo_32 = _mm256_cvtepu16_epi32(lo_16);
                    let hi_32 = _mm256_cvtepu16_epi32(hi_16);

                    let lo_f = _mm256_cvtepi32_ps(lo_32);
                    let hi_f = _mm256_cvtepi32_ps(hi_32);

                    store2 = _mm256_prefer_fma_ps(store2, lo_f, f_weight);
                    store3 = _mm256_prefer_fma_ps(store3, hi_f, f_weight);

                    let low_part = _mm256_castsi256_si128(row_1);

                    let hi_16 = _mm_unpackhi_epi8(low_part, _mm_setzero_si128());
                    let lo_16 = _mm_unpacklo_epi8(low_part, _mm_setzero_si128());

                    let lo_32 = _mm256_cvtepu16_epi32(lo_16);
                    let hi_32 = _mm256_cvtepu16_epi32(hi_16);

                    let lo_f = _mm256_cvtepi32_ps(lo_32);
                    let hi_f = _mm256_cvtepi32_ps(hi_32);

                    store4 = _mm256_prefer_fma_ps(store4, lo_f, f_weight);
                    store5 = _mm256_prefer_fma_ps(store5, hi_f, f_weight);

                    let high_part = _mm256_extracti128_si256::<1>(row_1);

                    let hi_16 = _mm_unpackhi_epi8(high_part, _mm_setzero_si128());
                    let lo_16 = _mm_unpacklo_epi8(high_part, _mm_setzero_si128());

                    let lo_32 = _mm256_cvtepu16_epi32(lo_16);
                    let hi_32 = _mm256_cvtepu16_epi32(hi_16);

                    let lo_f = _mm256_cvtepi32_ps(lo_32);
                    let hi_f = _mm256_cvtepi32_ps(hi_32);

                    store6 = _mm256_prefer_fma_ps(store6, lo_f, f_weight);
                    store7 = _mm256_prefer_fma_ps(store7, hi_f, f_weight);

                    r += 1;
                }

                store0 = _mm256_round_ps::<0x00>(store0);
                store1 = _mm256_round_ps::<0x00>(store1);
                store2 = _mm256_round_ps::<0x00>(store2);
                store3 = _mm256_round_ps::<0x00>(store3);
                store4 = _mm256_round_ps::<0x00>(store4);
                store5 = _mm256_round_ps::<0x00>(store5);
                store6 = _mm256_round_ps::<0x00>(store6);
                store7 = _mm256_round_ps::<0x00>(store7);

                let store_0_i32 = _mm256_cvtps_epi32(store0);
                let store_1_i32 = _mm256_cvtps_epi32(store1);
                let store_2_i32 = _mm256_cvtps_epi32(store2);
                let store_3_i32 = _mm256_cvtps_epi32(store3);
                let store_4_i32 = _mm256_cvtps_epi32(store4);
                let store_5_i32 = _mm256_cvtps_epi32(store5);
                let store_6_i32 = _mm256_cvtps_epi32(store6);
                let store_7_i32 = _mm256_cvtps_epi32(store7);

                let packed_row_16_lo_0 = avx2_pack_u32(store_0_i32, store_1_i32);
                let packed_row_16_hi_0 = avx2_pack_u32(store_2_i32, store_3_i32);
                let packed_row_16_lo_1 = avx2_pack_u32(store_4_i32, store_5_i32);
                let packed_row_16_hi_1 = avx2_pack_u32(store_6_i32, store_7_i32);

                let packed_row_0 = avx2_pack_u16(packed_row_16_lo_0, packed_row_16_hi_0);
                let packed_row_1 = avx2_pack_u16(packed_row_16_lo_1, packed_row_16_hi_1);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                _mm256_storeu_si256(dst_ptr as *mut __m256i, packed_row_0);
                _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, packed_row_1);

                cx += 64;
            }

            while cx + 32 < total_length {
                let mut store0 = zeros;
                let mut store1 = zeros;
                let mut store2 = zeros;
                let mut store3 = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm256_set1_ps(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let row = _mm256_loadu_si256(s_ptr as *const __m256i);

                    let low_part = _mm256_castsi256_si128(row);

                    let hi_16 = _mm_unpackhi_epi8(low_part, _mm_setzero_si128());
                    let lo_16 = _mm_unpacklo_epi8(low_part, _mm_setzero_si128());

                    let lo_32 = _mm256_cvtepu16_epi32(lo_16);
                    let hi_32 = _mm256_cvtepu16_epi32(hi_16);

                    let lo_f = _mm256_cvtepi32_ps(lo_32);
                    let hi_f = _mm256_cvtepi32_ps(hi_32);

                    store0 = _mm256_prefer_fma_ps(store0, lo_f, f_weight);
                    store1 = _mm256_prefer_fma_ps(store1, hi_f, f_weight);

                    let high_part = _mm256_extracti128_si256::<1>(row);

                    let hi_16 = _mm_unpackhi_epi8(high_part, _mm_setzero_si128());
                    let lo_16 = _mm_unpacklo_epi8(high_part, _mm_setzero_si128());

                    let lo_32 = _mm256_cvtepu16_epi32(lo_16);
                    let hi_32 = _mm256_cvtepu16_epi32(hi_16);

                    let lo_f = _mm256_cvtepi32_ps(lo_32);
                    let hi_f = _mm256_cvtepi32_ps(hi_32);

                    store2 = _mm256_prefer_fma_ps(store2, lo_f, f_weight);
                    store3 = _mm256_prefer_fma_ps(store3, hi_f, f_weight);

                    r += 1;
                }

                store0 = _mm256_round_ps::<0x00>(store0);
                store1 = _mm256_round_ps::<0x00>(store1);
                store2 = _mm256_round_ps::<0x00>(store2);
                store3 = _mm256_round_ps::<0x00>(store3);

                let store_0_i32 = _mm256_cvtps_epi32(store0);
                let store_1_i32 = _mm256_cvtps_epi32(store1);
                let store_2_i32 = _mm256_cvtps_epi32(store2);
                let store_3_i32 = _mm256_cvtps_epi32(store3);

                let packed_row_16_lo = avx2_pack_u32(store_0_i32, store_1_i32);
                let packed_row_16_hi = avx2_pack_u32(store_2_i32, store_3_i32);

                let packed_row = avx2_pack_u16(packed_row_16_lo, packed_row_16_hi);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                _mm256_storeu_si256(dst_ptr as *mut __m256i, packed_row);

                cx += 32;
            }

            while cx + 16 < total_length {
                let mut store0 = zeros;
                let mut store1 = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm256_set1_ps(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = _mm_loadu_si128(s_ptr as *const __m128i);
                    let hi_16 = _mm_unpackhi_epi8(pixels_u8, _mm_setzero_si128());
                    let lo_16 = _mm_unpacklo_epi8(pixels_u8, _mm_setzero_si128());

                    let lo_32 = _mm256_cvtepu16_epi32(lo_16);
                    let hi_32 = _mm256_cvtepu16_epi32(hi_16);

                    let lo_f = _mm256_cvtepi32_ps(lo_32);
                    let hi_f = _mm256_cvtepi32_ps(hi_32);

                    store0 = _mm256_prefer_fma_ps(store0, lo_f, f_weight);
                    store1 = _mm256_prefer_fma_ps(store1, hi_f, f_weight);

                    r += 1;
                }

                store0 = _mm256_round_ps::<0x00>(store0);
                store1 = _mm256_round_ps::<0x00>(store1);

                let store_0_i32 = _mm256_cvtps_epi32(store0);
                let store_1_i32 = _mm256_cvtps_epi32(store1);

                let store_0 = _mm256_castsi256_si128(store_0_i32);
                let store_1 = _mm256_extracti128_si256::<1>(store_0_i32);
                let store_2 = _mm256_castsi256_si128(store_1_i32);
                let store_3 = _mm256_extracti128_si256::<1>(store_1_i32);

                let store_lo = _mm_packus_epi32(store_0, store_1);
                let store_hi = _mm_packus_epi32(store_2, store_3);
                let store = _mm_packus_epi16(store_lo, store_hi);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                _mm_storeu_si128(dst_ptr as *mut __m128i, store);

                cx += 16;
            }

            while cx + 8 < total_length {
                let mut store = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm256_set1_ps(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = _mm_loadu_si64(s_ptr);
                    let pixels_u16 = _mm_unpacklo_epi8(pixels_u8, _mm_setzero_si128());
                    let epi32 = _mm256_cvtepu16_epi32(pixels_u16);
                    let pixel_pair = _mm256_cvtepi32_ps(epi32);
                    store = _mm256_prefer_fma_ps(store, pixel_pair, f_weight);

                    r += 1;
                }

                store = _mm256_round_ps::<0x00>(store);

                let store_i32 = _mm256_cvtps_epi32(store);

                let store_0 = _mm256_castsi256_si128(store_i32);
                let store_1 = _mm256_extracti128_si256::<1>(store_i32);

                let store_lo = _mm_packus_epi32(store_0, store_1);
                let store = _mm_packus_epi16(store_lo, store_lo);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                std::ptr::copy_nonoverlapping(&store as *const _ as *const u8, dst_ptr, 8);

                cx += 8;
            }

            while cx + 4 < total_length {
                let mut store0 = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm256_set1_ps(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let lo_lo = load_u8_f32_fast::<4>(s_ptr);
                    store0 = _mm256_prefer_fma_ps(store0, lo_lo, f_weight);

                    r += 1;
                }

                store0 = _mm256_round_ps::<0x00>(store0);
                let hi = _mm256_extractf128_ps::<1>(store0);
                let lo = _mm256_castps256_ps128(store0);
                let store = _mm_add_ps(lo, hi);
                let store_0 = _mm_cvtps_epi32(store);
                let store_c = _mm_packus_epi32(store_0, store_0);
                let store_lo = _mm_packus_epi16(store_c, store_c);
                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut i32;

                let pixel = _mm_extract_epi32::<0>(store_lo);
                dst_ptr.write_unaligned(pixel);

                cx += 4;
            }

            while cx < total_length {
                let mut store0 = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm256_setr_ps(weight, 0., 0., 0., 0., 0., 0., 0.);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u32 = load_u8_u32_one(s_ptr);
                    let lo_lo = _mm256_cvtepi32_ps(pixels_u32);
                    store0 = _mm256_prefer_fma_ps(store0, lo_lo, f_weight);

                    r += 1;
                }

                store0 = _mm256_round_ps::<0x00>(store0);

                let hi = _mm256_extractf128_ps::<1>(store0);
                let lo = _mm256_castps256_ps128(store0);

                let store = _mm_add_ps(lo, hi);

                let store_0 = _mm_cvtps_epi32(store);

                let store_c = _mm_packus_epi32(store_0, store_0);
                let store_lo = _mm_packus_epi16(store_c, store_c);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                let pixel = _mm_extract_epi32::<0>(store_lo);
                let bytes = pixel.to_le_bytes();
                dst_ptr.write_unaligned(bytes[0]);

                cx += 1;
            }
        }
    }
}
