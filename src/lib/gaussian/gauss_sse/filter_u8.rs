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

use crate::gaussian::gauss_sse::gauss_utils::_mm_opt_fma_ps;
use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::sse::{
    _mm_broadcast_first, _mm_broadcast_fourth, _mm_broadcast_second, _mm_broadcast_third,
    load_u8_f32_fast, load_u8_u32_one,
};
use crate::unsafe_slice::UnsafeSlice;
use crate::write_u8_by_channels_sse;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[macro_export]
macro_rules! accumulate_4_forward_sse_u8 {
    ($store:expr, $pixel_colors:expr, $weights:expr, $fma: expr) => {{
        let zeros_si = _mm_setzero_si128();
        let mut pixel_colors_u16 = _mm_unpacklo_epi8($pixel_colors, zeros_si);
        let mut pixel_colors_u32 = _mm_unpacklo_epi16(pixel_colors_u16, zeros_si);
        let mut pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);

        let first_weight = _mm_broadcast_first($weights);
        $store = _mm_opt_fma_ps::<$fma>($store, pixel_colors_f32, first_weight);

        pixel_colors_u32 = _mm_unpackhi_epi16(pixel_colors_u16, zeros_si);
        pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);

        $store = _mm_opt_fma_ps::<$fma>($store, pixel_colors_f32, _mm_broadcast_second($weights));

        pixel_colors_u16 = _mm_unpackhi_epi8($pixel_colors, zeros_si);
        pixel_colors_u32 = _mm_unpacklo_epi16(pixel_colors_u16, zeros_si);
        let mut pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);
        $store = _mm_opt_fma_ps::<$fma>($store, pixel_colors_f32, _mm_broadcast_third($weights));

        pixel_colors_u32 = _mm_unpackhi_epi16(pixel_colors_u16, zeros_si);
        pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);

        $store = _mm_opt_fma_ps::<$fma>($store, pixel_colors_f32, _mm_broadcast_fourth($weights));
    }};
}

#[macro_export]
macro_rules! accumulate_2_forward_sse_u8 {
    ($store:expr, $pixel_colors:expr, $weights:expr, $fma: expr) => {{
        let zeros_si = _mm_setzero_si128();
        let pixel_colors_u16 = _mm_unpacklo_epi8($pixel_colors, zeros_si);
        let mut pixel_colors_u32 = _mm_unpacklo_epi16(pixel_colors_u16, zeros_si);
        let mut pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);
        let first_weight = _mm_broadcast_first($weights);
        $store = _mm_opt_fma_ps::<$fma>($store, pixel_colors_f32, first_weight);

        pixel_colors_u32 = _mm_unpackhi_epi16(pixel_colors_u16, zeros_si);
        pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);

        $store = _mm_opt_fma_ps::<$fma>($store, pixel_colors_f32, _mm_broadcast_second($weights));
    }};
}

pub fn gaussian_blur_horizontal_pass_filter_sse<
    T,
    const CHANNEL_CONFIGURATION: usize,
    const FMA: bool,
>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        if FMA {
            gaussian_blur_horizontal_pass_filter_sse_gen_fma::<T, CHANNEL_CONFIGURATION>(
                undef_src,
                src_stride,
                undef_unsafe_dst,
                dst_stride,
                width,
                filter,
                start_y,
                end_y,
            );
        } else {
            gaussian_blur_horizontal_pass_filter_sse_gen::<T, CHANNEL_CONFIGURATION>(
                undef_src,
                src_stride,
                undef_unsafe_dst,
                dst_stride,
                width,
                filter,
                start_y,
                end_y,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn gaussian_blur_horizontal_pass_filter_sse_gen<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    gaussian_blur_horizontal_pass_filter_sse_impl::<T, CHANNEL_CONFIGURATION, false>(
        undef_src,
        src_stride,
        undef_unsafe_dst,
        dst_stride,
        width,
        filter,
        start_y,
        end_y,
    );
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn gaussian_blur_horizontal_pass_filter_sse_gen_fma<
    T,
    const CHANNEL_CONFIGURATION: usize,
>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    gaussian_blur_horizontal_pass_filter_sse_impl::<T, CHANNEL_CONFIGURATION, true>(
        undef_src,
        src_stride,
        undef_unsafe_dst,
        dst_stride,
        width,
        filter,
        start_y,
        end_y,
    );
}

unsafe fn gaussian_blur_horizontal_pass_filter_sse_impl<
    T,
    const CHANNEL_CONFIGURATION: usize,
    const FMA: bool,
>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    let src: &[u8] = std::mem::transmute(undef_src);
    let unsafe_dst: &UnsafeSlice<'_, u8> = std::mem::transmute(undef_unsafe_dst);
    #[rustfmt::skip]
        let shuffle_rgb = _mm_setr_epi8(0, 1, 2, -1, 3, 4,
                                                5, -1, 6, 7, 8, -1,
                                                9, 10, 11, -1);

    let mut cy = start_y;

    for y in (cy..end_y.saturating_sub(2)).step_by(2) {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let mut store_0 = _mm_setzero_ps();
            let mut store_1 = _mm_setzero_ps();

            let current_filter = filter.get_unchecked(x as usize);
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            let mut j = 0usize;

            while j + 4 < current_filter.size
                && filter_start as i64 + j as i64 + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                    < width as i64
            {
                let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let s_ptr_1 = s_ptr.add(src_stride as usize);
                let mut pixel_colors_0 = _mm_loadu_si128(s_ptr as *const __m128i);
                let mut pixel_colors_1 = _mm_loadu_si128(s_ptr_1 as *const __m128i);

                let weights_ptr = filter_weights.as_ptr().add(j);
                let weights = _mm_loadu_ps(weights_ptr);

                if CHANNEL_CONFIGURATION == 3 {
                    pixel_colors_0 = _mm_shuffle_epi8(pixel_colors_0, shuffle_rgb);
                    pixel_colors_1 = _mm_shuffle_epi8(pixel_colors_1, shuffle_rgb);
                }

                accumulate_4_forward_sse_u8!(store_0, pixel_colors_0, weights, FMA);
                accumulate_4_forward_sse_u8!(store_1, pixel_colors_1, weights, FMA);

                j += 4;
            }

            while j + 2 < current_filter.size
                && filter_start as i64 + j as i64 + (if CHANNEL_CONFIGURATION == 4 { 3 } else { 3 })
                    < width as i64
            {
                let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let s_ptr_1 = s_ptr.add(src_stride as usize);
                let mut pixel_colors_0 = _mm_loadu_si128(s_ptr as *const __m128i);
                let mut pixel_colors_1 = _mm_loadu_si128(s_ptr_1 as *const __m128i);

                let weights_ptr = filter_weights.as_ptr().add(j);
                let weights = _mm_castsi128_ps(_mm_loadu_si64(weights_ptr as *const u8));

                if CHANNEL_CONFIGURATION == 3 {
                    pixel_colors_0 = _mm_shuffle_epi8(pixel_colors_0, shuffle_rgb);
                    pixel_colors_1 = _mm_shuffle_epi8(pixel_colors_1, shuffle_rgb);
                }

                accumulate_2_forward_sse_u8!(store_0, pixel_colors_0, weights, FMA);
                accumulate_2_forward_sse_u8!(store_1, pixel_colors_1, weights, FMA);

                j += 2;
            }

            while j < current_filter.size {
                let current_x = filter_start + j;
                let px = current_x * CHANNEL_CONFIGURATION;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_f32_0 = load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                let pixel_colors_f32_1 =
                    load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr.add(src_stride as usize));
                let weight = *filter_weights.get_unchecked(j);
                let f_weight = _mm_set1_ps(weight);
                store_0 = _mm_opt_fma_ps::<FMA>(store_0, pixel_colors_f32_0, f_weight);
                store_1 = _mm_opt_fma_ps::<FMA>(store_0, pixel_colors_f32_1, f_weight);

                j += 1;
            }

            write_u8_by_channels_sse!(store_0, CHANNEL_CONFIGURATION, unsafe_dst, y_dst_shift, x);
            let off1 = y_dst_shift + dst_stride as usize;
            write_u8_by_channels_sse!(store_1, CHANNEL_CONFIGURATION, unsafe_dst, off1, x);
        }

        cy = y;
    }

    for y in cy..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let mut store = _mm_setzero_ps();

            let current_filter = filter.get_unchecked(x as usize);
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            let mut j = 0usize;

            while j + 4 < current_filter.size
                && filter_start as i64 + j as i64 + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                    < width as i64
            {
                let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let mut pixel_colors = _mm_loadu_si128(s_ptr as *const __m128i);

                let weights_ptr = filter_weights.as_ptr().add(j);
                let weights = _mm_loadu_ps(weights_ptr);

                if CHANNEL_CONFIGURATION == 3 {
                    pixel_colors = _mm_shuffle_epi8(pixel_colors, shuffle_rgb);
                }

                accumulate_4_forward_sse_u8!(store, pixel_colors, weights, FMA);

                j += 4;
            }

            while j + 2 < current_filter.size
                && filter_start as i64 + j as i64 + (if CHANNEL_CONFIGURATION == 4 { 2 } else { 3 })
                    < width as i64
            {
                let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let mut pixel_colors = _mm_loadu_si128(s_ptr as *const __m128i);
                let weights_ptr = filter_weights.as_ptr().add(j);
                let weights = _mm_castsi128_ps(_mm_loadu_si64(weights_ptr as *const u8));
                if CHANNEL_CONFIGURATION == 3 {
                    pixel_colors = _mm_shuffle_epi8(pixel_colors, shuffle_rgb);
                }

                accumulate_2_forward_sse_u8!(store, pixel_colors, weights, FMA);

                j += 2;
            }

            while j < current_filter.size {
                let current_x = filter_start + j;
                let px = current_x * CHANNEL_CONFIGURATION;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_f32 = load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                let weight = filter_weights.as_ptr().add(j);
                let f_weight = _mm_load1_ps(weight);
                store = _mm_opt_fma_ps::<FMA>(store, pixel_colors_f32, f_weight);

                j += 1;
            }

            write_u8_by_channels_sse!(store, CHANNEL_CONFIGURATION, unsafe_dst, y_dst_shift, x);
        }
    }
}

pub fn gaussian_blur_vertical_pass_filter_sse<
    T,
    const CHANNEL_CONFIGURATION: usize,
    const FMA: bool,
>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        if FMA {
            gaussian_blur_vertical_pass_filter_sse_fma::<T, CHANNEL_CONFIGURATION>(
                undef_src,
                src_stride,
                undef_unsafe_dst,
                dst_stride,
                width,
                height,
                filter,
                start_y,
                end_y,
            );
        } else {
            gaussian_blur_vertical_pass_filter_sse_gen::<T, CHANNEL_CONFIGURATION>(
                undef_src,
                src_stride,
                undef_unsafe_dst,
                dst_stride,
                width,
                height,
                filter,
                start_y,
                end_y,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1,fma")]
unsafe fn gaussian_blur_vertical_pass_filter_sse_fma<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    gaussian_blur_vertical_pass_filter_sse_impl::<T, CHANNEL_CONFIGURATION, true>(
        undef_src,
        src_stride,
        undef_unsafe_dst,
        dst_stride,
        width,
        height,
        filter,
        start_y,
        end_y,
    );
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn gaussian_blur_vertical_pass_filter_sse_gen<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    gaussian_blur_vertical_pass_filter_sse_impl::<T, CHANNEL_CONFIGURATION, false>(
        undef_src,
        src_stride,
        undef_unsafe_dst,
        dst_stride,
        width,
        height,
        filter,
        start_y,
        end_y,
    );
}

unsafe fn gaussian_blur_vertical_pass_filter_sse_impl<
    T,
    const CHANNEL_CONFIGURATION: usize,
    const FMA: bool,
>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    _: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undef_unsafe_dst) };
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
                    store0 = _mm_opt_fma_ps::<FMA>(store0, lo_lo, f_weight);
                    let lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros_si));
                    store1 = _mm_opt_fma_ps::<FMA>(store1, lo_hi, f_weight);
                    let hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi_16, zeros_si));
                    store2 = _mm_opt_fma_ps::<FMA>(store2, hi_lo, f_weight);
                    let hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi_16, zeros_si));
                    store3 = _mm_opt_fma_ps::<FMA>(store3, hi_hi, f_weight);

                    let hi_16 = _mm_unpackhi_epi8(pixels_u8_hi, zeros_si);
                    let lo_16 = _mm_unpacklo_epi8(pixels_u8_hi, zeros_si);
                    let lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo_16, zeros_si));
                    store4 = _mm_opt_fma_ps::<FMA>(store4, lo_lo, f_weight);
                    let lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros_si));
                    store5 = _mm_opt_fma_ps::<FMA>(store5, lo_hi, f_weight);
                    let hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi_16, zeros_si));
                    store6 = _mm_opt_fma_ps::<FMA>(store6, hi_lo, f_weight);
                    let hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi_16, zeros_si));
                    store7 = _mm_opt_fma_ps::<FMA>(store7, hi_hi, f_weight);

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
                    store0 = _mm_opt_fma_ps::<FMA>(store0, lo_lo, f_weight);
                    let lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros_si));
                    store1 = _mm_opt_fma_ps::<FMA>(store1, lo_hi, f_weight);
                    let hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi_16, zeros_si));
                    store2 = _mm_opt_fma_ps::<FMA>(store2, hi_lo, f_weight);
                    let hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi_16, zeros_si));
                    store3 = _mm_opt_fma_ps::<FMA>(store3, hi_hi, f_weight);

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
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = _mm_load1_ps(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = _mm_loadu_si64(s_ptr);
                    let pixels_u16 = _mm_unpacklo_epi8(pixels_u8, zeros_si);
                    let lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(pixels_u16, zeros_si));
                    store0 = _mm_opt_fma_ps::<FMA>(store0, lo_lo, f_weight);
                    let lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(pixels_u16, zeros_si));
                    store1 = _mm_opt_fma_ps::<FMA>(store1, lo_hi, f_weight);

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
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = _mm_load1_ps(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let lo_lo = load_u8_f32_fast::<4>(s_ptr);
                    store0 = _mm_opt_fma_ps::<FMA>(store0, lo_lo, f_weight);

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
                    store0 = _mm_opt_fma_ps::<FMA>(store0, lo_lo, f_weight);

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
