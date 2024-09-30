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
use crate::gaussian::gaussian_approx::{PRECISION, ROUNDING_APPROX};
use crate::sse::load_u8_u16_one;
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, reflect_101, reflect_index, EdgeMode};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn gaussian_blur_vertical_pass_approx_sse<const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[u8],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[i16],
    start_y: u32,
    end_y: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        gaussian_blur_vertical_pass_impl_sse_impl::<CHANNEL_CONFIGURATION>(
            undef_src,
            src_stride,
            undef_unsafe_dst,
            dst_stride,
            width,
            height,
            kernel_size,
            kernel,
            start_y,
            end_y,
            edge_mode,
        );
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn gaussian_blur_vertical_pass_impl_sse_impl<const CHANNEL_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[i16],
    start_y: u32,
    end_y: u32,
    edge_mode: EdgeMode,
) {
    let half_kernel = (kernel_size / 2) as i32;

    let zeros_si = _mm_setzero_si128();
    let total_length = width as usize * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut cx = 0usize;

        unsafe {
            while cx + 32 < total_length {
                let mut store0 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store1 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store2 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store3 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store4 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store5 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store6 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store7 = _mm_set1_epi32(ROUNDING_APPROX);

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_set1_epi16(weight);

                    let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8_lo = _mm_loadu_si128(s_ptr as *const __m128i);
                    let pixels_u8_hi = _mm_loadu_si128(s_ptr.add(16) as *const __m128i);
                    let hi_16 = _mm_unpackhi_epi8(pixels_u8_lo, zeros_si);
                    let lo_16 = _mm_unpacklo_epi8(pixels_u8_lo, zeros_si);
                    let lo_lo = _mm_unpacklo_epi16(lo_16, zeros_si);
                    store0 = _mm_add_epi32(store0, _mm_madd_epi16(lo_lo, f_weight));
                    let lo_hi = _mm_unpackhi_epi16(lo_16, zeros_si);
                    store1 = _mm_add_epi32(store1, _mm_madd_epi16(lo_hi, f_weight));
                    let hi_lo = _mm_unpacklo_epi16(hi_16, zeros_si);
                    store2 = _mm_add_epi32(store2, _mm_madd_epi16(hi_lo, f_weight));
                    let hi_hi = _mm_unpackhi_epi16(hi_16, zeros_si);
                    store3 = _mm_add_epi32(store3, _mm_madd_epi16(hi_hi, f_weight));

                    let hi_16 = _mm_unpackhi_epi8(pixels_u8_hi, zeros_si);
                    let lo_16 = _mm_unpacklo_epi8(pixels_u8_hi, zeros_si);
                    let lo_lo = _mm_unpacklo_epi16(lo_16, zeros_si);
                    store4 = _mm_add_epi32(store4, _mm_madd_epi16(lo_lo, f_weight));
                    let lo_hi = _mm_unpackhi_epi16(lo_16, zeros_si);
                    store5 = _mm_add_epi32(store5, _mm_madd_epi16(lo_hi, f_weight));
                    let hi_lo = _mm_unpacklo_epi16(hi_16, zeros_si);
                    store6 = _mm_add_epi32(store6, _mm_madd_epi16(hi_lo, f_weight));
                    let hi_hi = _mm_unpackhi_epi16(hi_16, zeros_si);
                    store7 = _mm_add_epi32(store7, _mm_madd_epi16(hi_hi, f_weight));

                    r += 1;
                }

                let store_0 = _mm_srai_epi32::<PRECISION>(store0);
                let store_1 = _mm_srai_epi32::<PRECISION>(store1);
                let store_2 = _mm_srai_epi32::<PRECISION>(store2);
                let store_3 = _mm_srai_epi32::<PRECISION>(store3);
                let store_4 = _mm_srai_epi32::<PRECISION>(store4);
                let store_5 = _mm_srai_epi32::<PRECISION>(store5);
                let store_6 = _mm_srai_epi32::<PRECISION>(store6);
                let store_7 = _mm_srai_epi32::<PRECISION>(store7);

                let store_lo = _mm_max_epi16(_mm_packs_epi32(store_0, store_1), zeros_si);
                let store_hi = _mm_max_epi16(_mm_packs_epi32(store_2, store_3), zeros_si);
                let store_x = _mm_packus_epi16(store_lo, store_hi);

                let store_lo = _mm_max_epi16(_mm_packs_epi32(store_4, store_5), zeros_si);
                let store_hi = _mm_max_epi16(_mm_packs_epi32(store_6, store_7), zeros_si);
                let store_k = _mm_packus_epi16(store_lo, store_hi);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                _mm_storeu_si128(dst_ptr as *mut __m128i, store_x);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, store_k);

                cx += 32;
            }

            while cx + 16 < total_length {
                let mut store0 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store1 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store2 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store3 = _mm_set1_epi32(ROUNDING_APPROX);

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_set1_epi16(weight);

                    let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = _mm_loadu_si128(s_ptr as *const __m128i);
                    let hi_16 = _mm_unpackhi_epi8(pixels_u8, zeros_si);
                    let lo_16 = _mm_unpacklo_epi8(pixels_u8, zeros_si);
                    let lo_lo = _mm_unpacklo_epi16(lo_16, zeros_si);
                    store0 = _mm_add_epi32(store0, _mm_madd_epi16(lo_lo, f_weight));
                    let lo_hi = _mm_unpackhi_epi16(lo_16, zeros_si);
                    store1 = _mm_add_epi32(store1, _mm_madd_epi16(lo_hi, f_weight));
                    let hi_lo = _mm_unpacklo_epi16(hi_16, zeros_si);
                    store2 = _mm_add_epi32(store2, _mm_madd_epi16(hi_lo, f_weight));
                    let hi_hi = _mm_unpackhi_epi16(hi_16, zeros_si);
                    store3 = _mm_add_epi32(store3, _mm_madd_epi16(hi_hi, f_weight));

                    r += 1;
                }

                let store_0 = _mm_srai_epi32::<PRECISION>(store0);
                let store_1 = _mm_srai_epi32::<PRECISION>(store1);
                let store_2 = _mm_srai_epi32::<PRECISION>(store2);
                let store_3 = _mm_srai_epi32::<PRECISION>(store3);

                let store_lo = _mm_max_epi16(_mm_packs_epi32(store_0, store_1), zeros_si);
                let store_hi = _mm_max_epi16(_mm_packs_epi32(store_2, store_3), zeros_si);
                let store = _mm_packus_epi16(store_lo, store_hi);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                _mm_storeu_si128(dst_ptr as *mut __m128i, store);

                cx += 16;
            }

            while cx + 8 < total_length {
                let mut store0 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store1 = _mm_set1_epi32(ROUNDING_APPROX);

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_set1_epi16(weight);

                    let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = _mm_unpacklo_epi8(_mm_loadu_si64(s_ptr), zeros_si);
                    let low_pixels = _mm_unpacklo_epi16(pixels_u8, zeros_si);
                    store0 = _mm_add_epi32(store0, _mm_madd_epi16(low_pixels, f_weight));
                    let hi_pixels = _mm_unpackhi_epi16(pixels_u8, zeros_si);
                    store1 = _mm_add_epi32(store1, _mm_madd_epi16(hi_pixels, f_weight));

                    r += 1;
                }

                let store_0 = _mm_srai_epi32::<PRECISION>(store0);
                let store_1 = _mm_srai_epi32::<PRECISION>(store1);

                let store_lo = _mm_max_epi16(_mm_packs_epi32(store_0, store_1), zeros_si);
                let store = _mm_packus_epi16(store_lo, store_lo);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                std::ptr::copy_nonoverlapping(&store as *const _ as *const u8, dst_ptr, 8);

                cx += 8;
            }

            while cx + 4 < total_length {
                let mut store0 = _mm_set1_epi32(ROUNDING_APPROX);

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_set1_epi16(weight);

                    let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let read_element = (s_ptr as *const i32).read_unaligned();
                    let vec_size = _mm_unpacklo_epi16(
                        _mm_unpacklo_epi8(_mm_setr_epi32(read_element, 0, 0, 0), zeros_si),
                        zeros_si,
                    );
                    store0 = _mm_add_epi32(store0, _mm_madd_epi16(vec_size, f_weight));

                    r += 1;
                }

                let store_0 = _mm_max_epi32(_mm_srai_epi32::<PRECISION>(store0), zeros_si);

                let store_c = _mm_packus_epi32(store_0, store_0);
                let store_lo = _mm_packus_epi16(store_c, store_c);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut i32;

                let pixel = _mm_extract_epi32::<0>(store_lo);
                dst_ptr.write_unaligned(pixel);

                cx += 4;
            }

            while cx < total_length {
                let mut store0 = _mm_set1_epi32(ROUNDING_APPROX);

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_set1_epi16(weight);

                    let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u16 = load_u8_u16_one(s_ptr);
                    store0 = _mm_add_epi32(store0, _mm_madd_epi16(pixels_u16, f_weight));

                    r += 1;
                }

                let store_0 = _mm_max_epi32(_mm_srai_epi32::<PRECISION>(store0), zeros_si);

                let store_c = _mm_packus_epi32(store_0, store_0);
                let store_lo = _mm_packus_epi16(store_c, store_c);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                let pixel = _mm_extract_epi8::<0>(store_lo) as u8;
                dst_ptr.write_unaligned(pixel);

                cx += 1;
            }
        }
    }
}
