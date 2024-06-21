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

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
pub mod sse_filter {
    use crate::gaussian::gaussian_filter::GaussianFilter;
    use crate::sse_utils::sse_utils::{_mm_prefer_fma_ps, load_u8_f32_fast, load_u8_u32_one};
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use crate::unsafe_slice::UnsafeSlice;

    pub fn gaussian_blur_horizontal_pass_filter_sse<const CHANNEL_CONFIGURATION: usize>(
        src: &[u8],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        filter: &Vec<GaussianFilter>,
        start_y: u32,
        end_y: u32,
    ) {
        #[rustfmt::skip]
        let shuffle_rgb =
            unsafe { _mm_setr_epi8(0, 1, 2, -1, 3, 4,
                                   5, -1, 6, 7, 8, -1,
                                   9, 10, 11, -1) };

        let zeros_si = unsafe { _mm_setzero_si128() };

        let mut cy = start_y;

        for y in (cy..end_y.saturating_sub(2)).step_by(2) {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store_0 = unsafe { _mm_setzero_ps() };
                let mut store_1 = unsafe { _mm_setzero_ps() };

                let current_filter = unsafe { filter.get_unchecked(x as usize) };
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut j = 0usize;

                unsafe {
                    while j + 4 < current_filter.size
                        && filter_start as i64
                            + j as i64
                            + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                            < width as i64
                    {
                        let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let s_ptr_1 = s_ptr.add(src_stride as usize);
                        let mut pixel_colors_0 = _mm_loadu_si128(s_ptr as *const __m128i);
                        let mut pixel_colors_1 = _mm_loadu_si128(s_ptr_1 as *const __m128i);
                        if CHANNEL_CONFIGURATION == 3 {
                            pixel_colors_0 = _mm_shuffle_epi8(pixel_colors_0, shuffle_rgb);
                            pixel_colors_1 = _mm_shuffle_epi8(pixel_colors_1, shuffle_rgb);
                        }
                        let mut pixel_colors_u16_0 = _mm_unpacklo_epi8(pixel_colors_0, zeros_si);
                        let mut pixel_colors_u32_0 =
                            _mm_unpacklo_epi16(pixel_colors_u16_0, zeros_si);
                        let mut pixel_colors_f32_0 = _mm_cvtepi32_ps(pixel_colors_u32_0);

                        let mut pixel_colors_u16_1 = _mm_unpacklo_epi8(pixel_colors_1, zeros_si);
                        let mut pixel_colors_u32_1 =
                            _mm_unpacklo_epi16(pixel_colors_u16_1, zeros_si);
                        let mut pixel_colors_f32_1 = _mm_cvtepi32_ps(pixel_colors_u32_1);

                        let mut weight = *filter_weights.get_unchecked(j);
                        let mut f_weight = _mm_set1_ps(weight);
                        store_0 = _mm_prefer_fma_ps(store_0, pixel_colors_f32_0, f_weight);
                        store_1 = _mm_prefer_fma_ps(store_1, pixel_colors_f32_1, f_weight);

                        pixel_colors_u32_0 = _mm_unpackhi_epi16(pixel_colors_u16_0, zeros_si);
                        pixel_colors_f32_0 = _mm_cvtepi32_ps(pixel_colors_u32_0);

                        pixel_colors_u32_1 = _mm_unpackhi_epi16(pixel_colors_u16_1, zeros_si);
                        pixel_colors_f32_1 = _mm_cvtepi32_ps(pixel_colors_u32_1);

                        weight = *filter_weights.get_unchecked(j + 1);
                        f_weight = _mm_set1_ps(weight);
                        store_0 = _mm_prefer_fma_ps(store_0, pixel_colors_f32_0, f_weight);
                        store_1 = _mm_prefer_fma_ps(store_1, pixel_colors_f32_1, f_weight);

                        pixel_colors_u16_0 = _mm_unpackhi_epi8(pixel_colors_0, zeros_si);
                        pixel_colors_u32_0 = _mm_unpacklo_epi16(pixel_colors_u16_0, zeros_si);

                        pixel_colors_u16_1 = _mm_unpackhi_epi8(pixel_colors_1, zeros_si);
                        pixel_colors_u32_1 = _mm_unpacklo_epi16(pixel_colors_u16_1, zeros_si);

                        pixel_colors_f32_0 = _mm_cvtepi32_ps(pixel_colors_u32_0);
                        pixel_colors_f32_1 = _mm_cvtepi32_ps(pixel_colors_u32_1);

                        let mut weight = *filter_weights.get_unchecked(j + 2);
                        let mut f_weight = _mm_set1_ps(weight);
                        store_0 = _mm_prefer_fma_ps(store_0, pixel_colors_f32_0, f_weight);
                        store_1 = _mm_prefer_fma_ps(store_1, pixel_colors_f32_1, f_weight);

                        pixel_colors_u32_0 = _mm_unpackhi_epi16(pixel_colors_u16_0, zeros_si);
                        pixel_colors_f32_0 = _mm_cvtepi32_ps(pixel_colors_u32_0);

                        pixel_colors_u32_1 = _mm_unpackhi_epi16(pixel_colors_u16_1, zeros_si);
                        pixel_colors_f32_1 = _mm_cvtepi32_ps(pixel_colors_u32_1);

                        weight = *filter_weights.get_unchecked(j + 3);
                        f_weight = _mm_set1_ps(weight);
                        store_0 = _mm_prefer_fma_ps(store_0, pixel_colors_f32_0, f_weight);
                        store_1 = _mm_prefer_fma_ps(store_1, pixel_colors_f32_1, f_weight);

                        j += 4;
                    }
                }

                unsafe {
                    while j + 2 < current_filter.size
                        && filter_start as i64
                            + j as i64
                            + (if CHANNEL_CONFIGURATION == 4 { 3 } else { 3 })
                            < width as i64
                    {
                        let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let s_ptr_1 = s_ptr.add(src_stride as usize);
                        let mut pixel_colors_0 = _mm_loadu_si128(s_ptr as *const __m128i);
                        let mut pixel_colors_1 = _mm_loadu_si128(s_ptr_1 as *const __m128i);
                        if CHANNEL_CONFIGURATION == 3 {
                            pixel_colors_0 = _mm_shuffle_epi8(pixel_colors_0, shuffle_rgb);
                            pixel_colors_1 = _mm_shuffle_epi8(pixel_colors_1, shuffle_rgb);
                        }
                        let pixel_colors_u16_0 = _mm_unpacklo_epi8(pixel_colors_0, zeros_si);
                        let mut pixel_colors_u32_0 =
                            _mm_unpacklo_epi16(pixel_colors_u16_0, zeros_si);
                        let mut pixel_colors_f32_0 = _mm_cvtepi32_ps(pixel_colors_u32_0);

                        let pixel_colors_u16_1 = _mm_unpacklo_epi8(pixel_colors_1, zeros_si);
                        let mut pixel_colors_u32_1 =
                            _mm_unpacklo_epi16(pixel_colors_u16_1, zeros_si);
                        let mut pixel_colors_f32_1 = _mm_cvtepi32_ps(pixel_colors_u32_1);

                        let mut weight = *filter_weights.get_unchecked(j);
                        let mut f_weight = _mm_set1_ps(weight);
                        store_0 = _mm_prefer_fma_ps(store_0, pixel_colors_f32_0, f_weight);
                        store_1 = _mm_prefer_fma_ps(store_1, pixel_colors_f32_1, f_weight);

                        pixel_colors_u32_0 = _mm_unpackhi_epi16(pixel_colors_u16_0, zeros_si);
                        pixel_colors_f32_0 = _mm_cvtepi32_ps(pixel_colors_u32_0);

                        pixel_colors_u32_1 = _mm_unpackhi_epi16(pixel_colors_u16_1, zeros_si);
                        pixel_colors_f32_1 = _mm_cvtepi32_ps(pixel_colors_u32_1);

                        weight = *filter_weights.get_unchecked(j + 1);
                        f_weight = _mm_set1_ps(weight);
                        store_0 = _mm_prefer_fma_ps(store_0, pixel_colors_f32_0, f_weight);
                        store_1 = _mm_prefer_fma_ps(store_1, pixel_colors_f32_1, f_weight);

                        j += 2;
                    }
                }

                unsafe {
                    while j < current_filter.size {
                        let current_x = filter_start + j;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_f32_0 = load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                        let pixel_colors_f32_1 = load_u8_f32_fast::<CHANNEL_CONFIGURATION>(
                            s_ptr.add(src_stride as usize),
                        );
                        let weight = *filter_weights.get_unchecked(j);
                        let f_weight = _mm_set1_ps(weight);
                        store_0 = _mm_prefer_fma_ps(store_0, pixel_colors_f32_0, f_weight);
                        store_1 = _mm_prefer_fma_ps(store_0, pixel_colors_f32_1, f_weight);

                        j += 1;
                    }
                }

                let px = x as usize * CHANNEL_CONFIGURATION;

                const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                let px_32 = unsafe { _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store_0)) };
                let px_16 = unsafe { _mm_packus_epi32(px_32, px_32) };
                let px_8 = unsafe { _mm_packus_epi16(px_16, px_16) };
                let pixel = unsafe { _mm_extract_epi32::<0>(px_8) };

                if CHANNEL_CONFIGURATION == 4 {
                    unsafe {
                        let unsafe_offset = y_dst_shift + px;
                        let dst_ptr = unsafe_dst.slice.as_ptr().add(unsafe_offset) as *mut i32;
                        dst_ptr.write_unaligned(pixel);
                    }
                } else {
                    let pixel_bytes = pixel.to_le_bytes();
                    unsafe {
                        let unsafe_offset = y_dst_shift + px;
                        unsafe_dst.write(unsafe_offset, pixel_bytes[0]);
                        unsafe_dst.write(unsafe_offset + 1, pixel_bytes[1]);
                        unsafe_dst.write(unsafe_offset + 2, pixel_bytes[2]);
                    }
                }

                let px_32 = unsafe { _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store_1)) };
                let px_16 = unsafe { _mm_packus_epi32(px_32, px_32) };
                let px_8 = unsafe { _mm_packus_epi16(px_16, px_16) };
                let pixel = unsafe { _mm_extract_epi32::<0>(px_8) };

                if CHANNEL_CONFIGURATION == 4 {
                    unsafe {
                        let unsafe_offset = y_dst_shift + src_stride as usize + px;
                        let dst_ptr = unsafe_dst.slice.as_ptr().add(unsafe_offset) as *mut i32;
                        dst_ptr.write_unaligned(pixel);
                    }
                } else {
                    let pixel_bytes = pixel.to_le_bytes();

                    unsafe {
                        let unsafe_offset = y_dst_shift + src_stride as usize + px;
                        unsafe_dst.write(unsafe_offset, pixel_bytes[0]);
                        unsafe_dst.write(unsafe_offset + 1, pixel_bytes[1]);
                        unsafe_dst.write(unsafe_offset + 2, pixel_bytes[2]);
                    }
                }
            }

            cy = y;
        }

        for y in cy..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store = unsafe { _mm_setzero_ps() };

                let current_filter = unsafe { filter.get_unchecked(x as usize) };
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut j = 0usize;

                unsafe {
                    while j + 4 < current_filter.size
                        && filter_start as i64
                            + j as i64
                            + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                            < width as i64
                    {
                        let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let mut pixel_colors = _mm_loadu_si128(s_ptr as *const __m128i);
                        if CHANNEL_CONFIGURATION == 3 {
                            pixel_colors = _mm_shuffle_epi8(pixel_colors, shuffle_rgb);
                        }
                        let mut pixel_colors_u16 = _mm_unpacklo_epi8(pixel_colors, zeros_si);
                        let mut pixel_colors_u32 = _mm_unpacklo_epi16(pixel_colors_u16, zeros_si);
                        let mut pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);
                        let mut weight = *filter_weights.get_unchecked(j);
                        let mut f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        pixel_colors_u32 = _mm_unpackhi_epi16(pixel_colors_u16, zeros_si);
                        pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);

                        weight = *filter_weights.get_unchecked(j + 1);
                        f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        pixel_colors_u16 = _mm_unpackhi_epi8(pixel_colors, zeros_si);
                        pixel_colors_u32 = _mm_unpacklo_epi16(pixel_colors_u16, zeros_si);
                        let mut pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);
                        let mut weight = *filter_weights.get_unchecked(j + 2);
                        let mut f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        pixel_colors_u32 = _mm_unpackhi_epi16(pixel_colors_u16, zeros_si);
                        pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);

                        weight = *filter_weights.get_unchecked(j + 3);
                        f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        j += 4;
                    }
                }

                unsafe {
                    while j + 2 < current_filter.size
                        && filter_start as i64
                            + j as i64
                            + (if CHANNEL_CONFIGURATION == 4 { 2 } else { 3 })
                            < width as i64
                    {
                        let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let mut pixel_colors = _mm_loadu_si128(s_ptr as *const __m128i);
                        if CHANNEL_CONFIGURATION == 3 {
                            pixel_colors = _mm_shuffle_epi8(pixel_colors, shuffle_rgb);
                        }
                        let pixel_colors_u16 = _mm_unpacklo_epi8(pixel_colors, zeros_si);
                        let mut pixel_colors_u32 = _mm_unpacklo_epi16(pixel_colors_u16, zeros_si);
                        let mut pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);
                        let mut weight = *filter_weights.get_unchecked(j);
                        let mut f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        pixel_colors_u32 = _mm_unpackhi_epi16(pixel_colors_u16, zeros_si);
                        pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);

                        weight = *filter_weights.get_unchecked(j + 1);
                        f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);
                        j += 2;
                    }
                }

                unsafe {
                    while j < current_filter.size {
                        let current_x = filter_start + j;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_f32 = load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                        let weight = *filter_weights.get_unchecked(j);
                        let f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        j += 1;
                    }
                }

                let px = x as usize * CHANNEL_CONFIGURATION;

                const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                let px_32 = unsafe { _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store)) };
                let px_16 = unsafe { _mm_packus_epi32(px_32, px_32) };
                let px_8 = unsafe { _mm_packus_epi16(px_16, px_16) };
                let pixel = unsafe { _mm_extract_epi32::<0>(px_8) };

                if CHANNEL_CONFIGURATION == 4 {
                    unsafe {
                        let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut i32;
                        dst_ptr.write_unaligned(pixel);
                    }
                } else {
                    let pixel_bytes = pixel.to_le_bytes();
                    unsafe {
                        let unsafe_offset = y_dst_shift + px;
                        unsafe_dst.write(unsafe_offset, pixel_bytes[0]);
                        unsafe_dst.write(unsafe_offset + 1, pixel_bytes[1]);
                        unsafe_dst.write(unsafe_offset + 2, pixel_bytes[2]);
                    }
                }
            }
        }
    }

    pub fn gaussian_blur_vertical_pass_filter_sse<const CHANNEL_CONFIGURATION: usize>(
        src: &[u8],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        _: u32,
        filter: &Vec<GaussianFilter>,
        start_y: u32,
        end_y: u32,
    ) {
        const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

        let zeros = unsafe { _mm_setzero_ps() };
        let zeros_si = unsafe { _mm_setzero_si128() };
        let total_length = width as usize * CHANNEL_CONFIGURATION;
        for y in start_y..end_y {
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut cx = 0usize;

            let current_filter = unsafe { filter.get_unchecked(y as usize) };
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            unsafe {
                while cx + 32 < total_length {
                    let mut store0 = zeros;
                    let mut store1 = zeros;
                    let mut store2 = zeros;
                    let mut store3 = zeros;
                    let mut store4 = zeros;
                    let mut store5 = zeros;
                    let mut store6 = zeros;
                    let mut store7 = zeros;

                    let mut j = 0usize;
                    while j < current_filter.size {
                        let weight = *filter_weights.get_unchecked(j);
                        let f_weight = _mm_set1_ps(weight);

                        let py = filter_start + j;
                        let y_src_shift = py * src_stride as usize;
                        let s_ptr = src.as_ptr().add(y_src_shift + cx);
                        let pixels_u8_lo = _mm_loadu_si128(s_ptr as *const __m128i);
                        let pixels_u8_hi = _mm_loadu_si128(s_ptr.add(16) as *const __m128i);
                        let hi_16 = _mm_unpackhi_epi8(pixels_u8_lo, zeros_si);
                        let lo_16 = _mm_unpacklo_epi8(pixels_u8_lo, zeros_si);
                        let lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo_16, zeros_si));
                        store0 = _mm_prefer_fma_ps(store0, lo_lo, f_weight);
                        let lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros_si));
                        store1 = _mm_prefer_fma_ps(store1, lo_hi, f_weight);
                        let hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi_16, zeros_si));
                        store2 = _mm_prefer_fma_ps(store2, hi_lo, f_weight);
                        let hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi_16, zeros_si));
                        store3 = _mm_prefer_fma_ps(store3, hi_hi, f_weight);

                        let hi_16 = _mm_unpackhi_epi8(pixels_u8_hi, zeros_si);
                        let lo_16 = _mm_unpacklo_epi8(pixels_u8_hi, zeros_si);
                        let lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo_16, zeros_si));
                        store4 = _mm_prefer_fma_ps(store4, lo_lo, f_weight);
                        let lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros_si));
                        store5 = _mm_prefer_fma_ps(store5, lo_hi, f_weight);
                        let hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi_16, zeros_si));
                        store6 = _mm_prefer_fma_ps(store6, hi_lo, f_weight);
                        let hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi_16, zeros_si));
                        store7 = _mm_prefer_fma_ps(store7, hi_hi, f_weight);

                        j += 1;
                    }

                    let store_0 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store0));
                    let store_1 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store1));
                    let store_2 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store2));
                    let store_3 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store3));
                    let store_4 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store4));
                    let store_5 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store5));
                    let store_6 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store6));
                    let store_7 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store7));

                    let store_lo = _mm_packus_epi32(store_0, store_1);
                    let store_hi = _mm_packus_epi32(store_2, store_3);
                    let store_x = _mm_packus_epi16(store_lo, store_hi);

                    let store_lo = _mm_packus_epi32(store_4, store_5);
                    let store_hi = _mm_packus_epi32(store_6, store_7);
                    let store_k = _mm_packus_epi16(store_lo, store_hi);

                    let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                    _mm_storeu_si128(dst_ptr as *mut __m128i, store_x);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, store_k);

                    cx += 32;
                }

                while cx + 16 < total_length {
                    let mut store0 = zeros;
                    let mut store1 = zeros;
                    let mut store2 = zeros;
                    let mut store3 = zeros;

                    let mut j = 0usize;
                    while j < current_filter.size {
                        let weight = *filter_weights.get_unchecked(j);
                        let f_weight = _mm_set1_ps(weight);

                        let py = filter_start + j;
                        let y_src_shift = py * src_stride as usize;
                        let s_ptr = src.as_ptr().add(y_src_shift + cx);
                        let pixels_u8 = _mm_loadu_si128(s_ptr as *const __m128i);
                        let hi_16 = _mm_unpackhi_epi8(pixels_u8, zeros_si);
                        let lo_16 = _mm_unpacklo_epi8(pixels_u8, zeros_si);
                        let lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo_16, zeros_si));
                        store0 = _mm_prefer_fma_ps(store0, lo_lo, f_weight);
                        let lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros_si));
                        store1 = _mm_prefer_fma_ps(store1, lo_hi, f_weight);
                        let hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi_16, zeros_si));
                        store2 = _mm_prefer_fma_ps(store2, hi_lo, f_weight);
                        let hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi_16, zeros_si));
                        store3 = _mm_prefer_fma_ps(store3, hi_hi, f_weight);

                        j += 1;
                    }

                    let store_0 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store0));
                    let store_1 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store1));
                    let store_2 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store2));
                    let store_3 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store3));

                    let store_lo = _mm_packus_epi32(store_0, store_1);
                    let store_hi = _mm_packus_epi32(store_2, store_3);
                    let store = _mm_packus_epi16(store_lo, store_hi);

                    let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                    _mm_storeu_si128(dst_ptr as *mut __m128i, store);

                    cx += 16;
                }

                while cx + 8 < total_length {
                    let mut store0 = zeros;
                    let mut store1 = zeros;

                    let mut j = 0usize;
                    while j < current_filter.size {
                        let weight = *filter_weights.get_unchecked(j);
                        let f_weight = _mm_set1_ps(weight);

                        let py = filter_start + j;
                        let y_src_shift = py * src_stride as usize;
                        let s_ptr = src.as_ptr().add(y_src_shift + cx);
                        let pixels_u8 = _mm_loadu_si64(s_ptr);
                        let pixels_u16 = _mm_unpacklo_epi8(pixels_u8, zeros_si);
                        let lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(pixels_u16, zeros_si));
                        store0 = _mm_prefer_fma_ps(store0, lo_lo, f_weight);
                        let lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(pixels_u16, zeros_si));
                        store1 = _mm_prefer_fma_ps(store1, lo_hi, f_weight);

                        j += 1;
                    }

                    let store_0 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store0));
                    let store_1 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store1));

                    let store_lo = _mm_packus_epi32(store_0, store_1);
                    let store = _mm_packus_epi16(store_lo, store_lo);

                    let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                    std::ptr::copy_nonoverlapping(&store as *const _ as *const u8, dst_ptr, 8);

                    cx += 8;
                }

                while cx + 4 < total_length {
                    let mut store0 = zeros;

                    let mut j = 0usize;
                    while j < current_filter.size {
                        let weight = *filter_weights.get_unchecked(j);
                        let f_weight = _mm_set1_ps(weight);

                        let py = filter_start + j;
                        let y_src_shift = py * src_stride as usize;
                        let s_ptr = src.as_ptr().add(y_src_shift + cx);
                        let lo_lo = load_u8_f32_fast::<4>(s_ptr);
                        store0 = _mm_prefer_fma_ps(store0, lo_lo, f_weight);

                        j += 1;
                    }

                    let store_0 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store0));

                    let store_c = _mm_packus_epi32(store_0, store_0);
                    let store_lo = _mm_packus_epi16(store_c, store_c);

                    let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut i32;

                    let pixel = _mm_extract_epi32::<0>(store_lo);
                    dst_ptr.write_unaligned(pixel);

                    cx += 4;
                }

                while cx < total_length {
                    let mut store0 = zeros;

                    let mut j = 0usize;
                    while j < current_filter.size {
                        let weight = *filter_weights.get_unchecked(j);
                        let f_weight = _mm_set1_ps(weight);

                        let py = filter_start + j;
                        let y_src_shift = py * src_stride as usize;
                        let s_ptr = src.as_ptr().add(y_src_shift + cx);
                        let pixels_u32 = load_u8_u32_one(s_ptr);
                        let lo_lo = _mm_cvtepi32_ps(pixels_u32);
                        store0 = _mm_prefer_fma_ps(store0, lo_lo, f_weight);

                        j += 1;
                    }

                    let store_0 = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store0));

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
}
