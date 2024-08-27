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
use crate::sse::load_u8_s16_fast;
use crate::unsafe_slice::UnsafeSlice;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[macro_export]
macro_rules! write_u8_by_channels_sse_approx {
    ($store:expr, $channels:expr, $unsafe_dst:expr, $dst_shift: expr, $x: expr) => {{
        let px = $x as usize * $channels;

        let px_32 = _mm_max_epi32(_mm_srai_epi32::<PRECISION>($store), _mm_setzero_si128());
        let px_16 = _mm_packus_epi32(px_32, px_32);
        let px_8 = _mm_packus_epi16(px_16, px_16);
        let pixel = _mm_extract_epi32::<0>(px_8);
        let unsafe_offset = $dst_shift + px;
        if CHANNEL_CONFIGURATION == 4 {
            let dst_ptr = $unsafe_dst.slice.as_ptr().add(unsafe_offset) as *mut i32;
            dst_ptr.write_unaligned(pixel);
        } else {
            let pixel_bytes = pixel.to_le_bytes();
            let dst_ptr = $unsafe_dst.slice.as_ptr().add(unsafe_offset) as *mut u16;
            let lowest = u16::from_le_bytes([pixel_bytes[0], pixel_bytes[1]]);
            dst_ptr.write_unaligned(lowest);
            $unsafe_dst.write(unsafe_offset + 2, pixel_bytes[2]);
        }
    }};
}

pub fn gaussian_blur_horizontal_pass_approx_sse<const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[u8],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[i16],
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        gaussian_blur_horizontal_pass_approx_impl::<CHANNEL_CONFIGURATION>(
            undef_src,
            src_stride,
            undef_unsafe_dst,
            dst_stride,
            width,
            kernel_size,
            kernel,
            start_y,
            end_y,
        );
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn gaussian_blur_horizontal_pass_approx_impl<const CHANNEL_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[i16],
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let zeros_si = _mm_setzero_si128();
        let half_kernel = (kernel_size / 2) as i32;

        let shuffle_lo = if CHANNEL_CONFIGURATION == 4 {
            _mm_setr_epi8(0, -1, 4, -1, 1, -1, 5, -1, 2, -1, 6, -1, 3, -1, 7, -1)
        } else {
            _mm_setr_epi8(0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1)
        };

        let shuffle_hi = if CHANNEL_CONFIGURATION == 4 {
            _mm_setr_epi8(8, -1, 12, -1, 9, -1, 13, -1, 10, -1, 14, -1, 11, -1, 15, -1)
        } else {
            _mm_setr_epi8(6, -1, 9, -1, 7, -1, 10, -1, 8, -1, 11, -1, -1, -1, -1, -1)
        };

        let mut cy = start_y;

        for y in (cy..end_y.saturating_sub(4)).step_by(4) {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store_0 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store_1 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store_2 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store_3 = _mm_set1_epi32(ROUNDING_APPROX);

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let mut pixel_colors_0 = load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let s_ptr_next_1 = src.as_ptr().add(y_src_shift + src_stride as usize); // Here we're always at zero
                    let mut pixel_colors_1 =
                        load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_1);
                    let s_ptr_next_2 = s_ptr_next_1.add(src_stride as usize); // Here we're always at zero
                    let mut pixel_colors_2 =
                        load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_2);
                    let s_ptr_next_3 = s_ptr_next_2.add(src_stride as usize); // Here we're always at zero
                    let mut pixel_colors_3 =
                        load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_3);

                    pixel_colors_0 = _mm_unpacklo_epi16(pixel_colors_0, zeros_si);
                    pixel_colors_1 = _mm_unpacklo_epi16(pixel_colors_1, zeros_si);
                    pixel_colors_2 = _mm_unpacklo_epi16(pixel_colors_2, zeros_si);
                    pixel_colors_3 = _mm_unpacklo_epi16(pixel_colors_3, zeros_si);

                    for i in 0..diff as usize {
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = _mm_set1_epi16(weights.read_unaligned());
                        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(pixel_colors_0, f_weight));
                        store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(pixel_colors_1, f_weight));
                        store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(pixel_colors_2, f_weight));
                        store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(pixel_colors_3, f_weight));
                    }
                    r += diff as i32;
                }

                while r + 4 <= half_kernel
                    && x as i64 + r as i64 + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                        < width as i64
                {
                    let px =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize
                            * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_1 = s_ptr.add(src_stride as usize);
                    let pixel_colors_0 = _mm_loadu_si128(s_ptr as *const __m128i);
                    let pixel_colors_1 = _mm_loadu_si128(s_ptr_1 as *const __m128i);
                    let pixel_colors_2 =
                        _mm_loadu_si128(s_ptr_1.add(src_stride as usize) as *const __m128i);
                    let pixel_colors_3 =
                        _mm_loadu_si128(s_ptr_1.add(src_stride as usize * 2) as *const __m128i);
                    let weights_ptr = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights0 = _mm_set1_epi32((weights_ptr as *const i32).read_unaligned());
                    let weights1 = _mm_set1_epi32((weights_ptr.add(2) as *const i32).read_unaligned());

                    let lo_0 = _mm_shuffle_epi8(pixel_colors_0, shuffle_lo);
                    let hi_0 = _mm_shuffle_epi8(pixel_colors_0, shuffle_hi);
                    let lo_1 = _mm_shuffle_epi8(pixel_colors_1, shuffle_lo);
                    let hi_1 = _mm_shuffle_epi8(pixel_colors_1, shuffle_hi);
                    let lo_2 = _mm_shuffle_epi8(pixel_colors_2, shuffle_lo);
                    let hi_2 = _mm_shuffle_epi8(pixel_colors_2, shuffle_hi);
                    let lo_3 = _mm_shuffle_epi8(pixel_colors_3, shuffle_lo);
                    let hi_3 = _mm_shuffle_epi8(pixel_colors_3, shuffle_hi);

                    store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo_0, weights0));
                    store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(hi_0, weights1));

                    store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo_1, weights0));
                    store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(hi_1, weights1));

                    store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo_2, weights0));
                    store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(hi_2, weights1));

                    store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo_3, weights0));
                    store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(hi_3, weights1));

                    r += 4;
                }

                while r + 2 <= half_kernel
                    && x as i64 + r as i64 + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                        < width as i64
                {
                    let px =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize
                            * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_1 = s_ptr.add(src_stride as usize);
                    let pixel_colors_0 = _mm_loadu_si64(s_ptr);
                    let pixel_colors_1 = _mm_loadu_si64(s_ptr_1);
                    let pixel_colors_2 = _mm_loadu_si64(s_ptr_1.add(src_stride as usize));
                    let pixel_colors_3 = _mm_loadu_si64(s_ptr_1.add(src_stride as usize * 2));
                    let weights_ptr = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = _mm_set1_epi32((weights_ptr as *const i32).read_unaligned());

                    let lo_0 = _mm_shuffle_epi8(pixel_colors_0, shuffle_lo);
                    let lo_1 = _mm_shuffle_epi8(pixel_colors_1, shuffle_lo);
                    let lo_2 = _mm_shuffle_epi8(pixel_colors_2, shuffle_lo);
                    let lo_3 = _mm_shuffle_epi8(pixel_colors_3, shuffle_lo);

                    store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo_0, weights));
                    store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo_1, weights));
                    store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo_2, weights));
                    store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo_3, weights));

                    r += 2;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_set1_epi16(weight);

                    let mut pixel_colors_0 = load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let mut pixel_colors_1 =
                        load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr.add(src_stride as usize));
                    let mut pixel_colors_2 = load_u8_s16_fast::<CHANNEL_CONFIGURATION>(
                        s_ptr.add(src_stride as usize * 2),
                    );
                    let mut pixel_colors_3 = load_u8_s16_fast::<CHANNEL_CONFIGURATION>(
                        s_ptr.add(src_stride as usize * 3),
                    );

                    pixel_colors_0 = _mm_unpacklo_epi16(pixel_colors_0, zeros_si);
                    pixel_colors_1 = _mm_unpacklo_epi16(pixel_colors_1, zeros_si);
                    pixel_colors_2 = _mm_unpacklo_epi16(pixel_colors_2, zeros_si);
                    pixel_colors_3 = _mm_unpacklo_epi16(pixel_colors_3, zeros_si);

                    store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(pixel_colors_0, f_weight));
                    store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(pixel_colors_1, f_weight));
                    store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(pixel_colors_2, f_weight));
                    store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(pixel_colors_3, f_weight));

                    r += 1;
                }

                write_u8_by_channels_sse_approx!(
                    store_0,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    y_dst_shift,
                    x
                );
                let off1 = y_dst_shift + dst_stride as usize;
                write_u8_by_channels_sse_approx!(
                    store_1,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    off1,
                    x
                );
                let off2 = off1 + dst_stride as usize;
                write_u8_by_channels_sse_approx!(
                    store_2,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    off2,
                    x
                );
                let off3 = off2 + dst_stride as usize;
                write_u8_by_channels_sse_approx!(
                    store_3,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    off3,
                    x
                );
            }

            cy = y;
        }

        for y in (cy..end_y.saturating_sub(2)).step_by(2) {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store_0 = _mm_set1_epi32(ROUNDING_APPROX);
                let mut store_1 = _mm_set1_epi32(ROUNDING_APPROX);

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let mut pixel_colors_0 = load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift + src_stride as usize); // Here we're always at zero
                    let mut pixel_colors_1 = load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr_next);
                    pixel_colors_0 = _mm_unpacklo_epi16(pixel_colors_0, zeros_si);
                    pixel_colors_1 = _mm_unpacklo_epi16(pixel_colors_1, zeros_si);
                    for i in 0..diff as usize {
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = _mm_set1_epi16(weights.read_unaligned());
                        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(pixel_colors_0, f_weight));
                        store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(pixel_colors_1, f_weight));
                    }
                    r += diff as i32;
                }

                while r + 4 <= half_kernel
                    && x as i64 + r as i64 + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                        < width as i64
                {
                    let px =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize
                            * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_1 = s_ptr.add(src_stride as usize);
                    let pixel_colors_0 = _mm_loadu_si128(s_ptr as *const __m128i);
                    let pixel_colors_1 = _mm_loadu_si128(s_ptr_1 as *const __m128i);
                    let weights_ptr = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights0 = _mm_set1_epi32((weights_ptr as *const i32).read_unaligned());
                    let weights1 = _mm_set1_epi32((weights_ptr.add(2) as *const i32).read_unaligned());

                    let lo_0 = _mm_shuffle_epi8(pixel_colors_0, shuffle_lo);
                    let hi_0 = _mm_shuffle_epi8(pixel_colors_0, shuffle_hi);
                    let lo_1 = _mm_shuffle_epi8(pixel_colors_1, shuffle_lo);
                    let hi_1 = _mm_shuffle_epi8(pixel_colors_1, shuffle_hi);

                    store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo_0, weights0));
                    store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(hi_0, weights1));

                    store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo_1, weights0));
                    store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(hi_1, weights1));

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors_0 = load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let mut pixel_colors_1 =
                        load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr.add(src_stride as usize));

                    pixel_colors_0 = _mm_unpacklo_epi16(pixel_colors_0, zeros_si);
                    pixel_colors_1 = _mm_unpacklo_epi16(pixel_colors_1, zeros_si);

                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_set1_epi16(weight);
                    store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(pixel_colors_0, f_weight));
                    store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(pixel_colors_1, f_weight));

                    r += 1;
                }

                write_u8_by_channels_sse_approx!(
                    store_0,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    y_dst_shift,
                    x
                );
                let off1 = y_dst_shift + dst_stride as usize;
                write_u8_by_channels_sse_approx!(
                    store_1,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    off1,
                    x
                );
            }

            cy = y;
        }

        for y in cy..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store = _mm_set1_epi32(ROUNDING_APPROX);

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let pixel_colors_0 = load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    for i in 0..diff as usize {
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = _mm_set1_epi16(weights.read_unaligned());
                        store = _mm_add_epi32(store, _mm_madd_epi16(pixel_colors_0, f_weight));
                    }
                    r += diff as i32;
                }

                while r + 4 <= half_kernel && x as i64 + r as i64 + 6 < width as i64 {
                    let px =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize
                            * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors = _mm_loadu_si128(s_ptr as *const __m128i);
                    let weights_ptr = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights0 = _mm_set1_epi32((weights_ptr as *const i32).read_unaligned());
                    let weights1 = _mm_set1_epi32((weights_ptr.add(2) as *const i32).read_unaligned());

                    let lo = _mm_shuffle_epi8(pixel_colors, shuffle_lo);
                    let hi = _mm_shuffle_epi8(pixel_colors, shuffle_hi);

                    store = _mm_add_epi32(store, _mm_madd_epi16(lo, weights0));
                    store = _mm_add_epi32(store, _mm_madd_epi16(hi, weights1));

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors = load_u8_s16_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    pixel_colors = _mm_unpacklo_epi16(pixel_colors, zeros_si);
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_set1_epi16(weight);
                    store = _mm_add_epi32(store, _mm_madd_epi16(pixel_colors, f_weight));

                    r += 1;
                }

                write_u8_by_channels_sse_approx!(
                    store,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    y_dst_shift,
                    x
                );
            }
        }
    }
}
