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

use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::neon::{load_u8_u32_fast, load_u8_u32_one, prefer_vfma_f32, prefer_vfmaq_f32};
use crate::unsafe_slice::UnsafeSlice;
use crate::{accumulate_2_forward_u8, accumulate_4_by_4_forward_u8, accumulate_4_forward_u8};
use std::arch::aarch64::*;

#[macro_export]
macro_rules! write_u8_by_channels {
    ($store:expr, $channels:expr, $unsafe_dst:expr, $dst_shift: expr, $x: expr) => {{
        let px = $x as usize * $channels;
        let offset = $dst_shift + px;

        let px_16 = vqmovn_u32(vcvtaq_u32_f32($store));
        let px_8 = vqmovn_u16(vcombine_u16(px_16, px_16));
        let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(px_8));
        if CHANNEL_CONFIGURATION == 4 {
            let dst_ptr = $unsafe_dst.slice.as_ptr().add(offset) as *mut u32;
            dst_ptr.write_unaligned(pixel);
        } else {
            let bits = pixel.to_le_bytes();
            $unsafe_dst.write(offset, bits[0]);
            $unsafe_dst.write(offset + 1, bits[1]);
            $unsafe_dst.write(offset + 2, bits[2]);
        }
    }};
}

pub fn gaussian_blur_horizontal_pass_filter_neon<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &Vec<GaussianFilter>,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let src: &[u8] = std::mem::transmute(undef_src);
        let unsafe_dst: &UnsafeSlice<'_, u8> = std::mem::transmute(undef_unsafe_dst);
        let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
        let shuffle_1 = vld1_u8(shuf_table_1.as_ptr());
        let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
        let shuffle_2 = vld1_u8(shuf_table_2.as_ptr());
        let shuffle = vcombine_u8(shuffle_1, shuffle_2);

        let mut cy = start_y;

        for y in (cy..end_y.saturating_sub(2)).step_by(2) {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store_0: float32x4_t = vdupq_n_f32(0.);
                let mut store_1: float32x4_t = vdupq_n_f32(0.);

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut j = 0usize;

                if CHANNEL_CONFIGURATION == 4 {
                    while j + 16 < current_filter.size && x as i64 + j as i64 + 16i64 < width as i64
                    {
                        let px = std::cmp::min(
                            std::cmp::max(filter_start as i64 + j as i64, 0),
                            (width - 1) as i64,
                        ) as usize
                            * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_u8_x4(s_ptr);
                        let pixel_colors_1 = vld1q_u8_x4(s_ptr.add(src_stride as usize));

                        let weight_ptr = filter_weights.as_ptr().add(j);
                        let weights = vld1q_f32_x4(weight_ptr);

                        accumulate_4_by_4_forward_u8!(store_0, pixel_colors_0, weights);
                        accumulate_4_by_4_forward_u8!(store_1, pixel_colors_1, weights);

                        j += 16;
                    }
                }

                while j + 4 < current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                        < width as i64
                {
                    let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors_0: uint8x16_t = vld1q_u8(s_ptr);
                    let mut pixel_colors_1: uint8x16_t = vld1q_u8(s_ptr.add(src_stride as usize));
                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors_0 = vqtbl1q_u8(pixel_colors_0, shuffle);
                        pixel_colors_1 = vqtbl1q_u8(pixel_colors_1, shuffle);
                    }
                    let weight_ptr = filter_weights.as_ptr().add(j);

                    let weights: float32x4_t = vld1q_f32(weight_ptr);

                    accumulate_4_forward_u8!(store_0, pixel_colors_0, weights);
                    accumulate_4_forward_u8!(store_1, pixel_colors_1, weights);

                    j += 4;
                }

                while j + 2 <= current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 2 } else { 3 })
                        < width as i64
                {
                    let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors_0: uint8x8_t = vld1_u8(s_ptr);
                    let mut pixel_colors_1: uint8x8_t = vld1_u8(s_ptr.add(src_stride as usize));

                    let weights_ptr = filter_weights.as_ptr().add(j);
                    let weights = vld1_f32(weights_ptr);

                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors_0 = vtbl1_u8(pixel_colors_0, shuffle_1);
                        pixel_colors_1 = vtbl1_u8(pixel_colors_1, shuffle_1);
                    }

                    accumulate_2_forward_u8!(store_0, pixel_colors_0, weights);
                    accumulate_2_forward_u8!(store_1, pixel_colors_1, weights);

                    j += 2;
                }

                while j < current_filter.size {
                    let current_x = filter_start + j;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u32_0 = load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let pixel_colors_u32_1 =
                        load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr.add(src_stride as usize));
                    let pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);
                    let pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    j += 1;
                }

                write_u8_by_channels!(store_0, CHANNEL_CONFIGURATION, unsafe_dst, y_dst_shift, x);
                write_u8_by_channels!(
                    store_1,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    y_dst_shift + src_stride as usize,
                    x
                );
            }
            cy = y;
        }

        for y in cy..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store: float32x4_t = vdupq_n_f32(0f32);

                let mut j = 0;

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                while j + 4 < current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                        < width as i64
                {
                    let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors: uint8x16_t = vld1q_u8(s_ptr);
                    let weights_ptr = filter_weights.as_ptr().add(j);

                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors = vqtbl1q_u8(pixel_colors, shuffle);
                    }

                    let weights: float32x4_t = vld1q_f32(weights_ptr);

                    accumulate_4_forward_u8!(store, pixel_colors, weights);

                    j += 4;
                }

                while j + 2 < current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 2 } else { 3 })
                        < width as i64
                {
                    let current_x = filter_start + j;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let weights_ptr = filter_weights.as_ptr().add(j);
                    let weights = vld1_f32(weights_ptr);
                    let mut pixel_color = vld1_u8(s_ptr);
                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_color = vtbl1_u8(pixel_color, shuffle_1);
                    }

                    accumulate_2_forward_u8!(store, pixel_color, weights);

                    j += 2;
                }

                while j < current_filter.size {
                    let current_x = filter_start + j;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u32 = load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    j += 1;
                }

                write_u8_by_channels!(store, CHANNEL_CONFIGURATION, unsafe_dst, y_dst_shift, x);
            }
        }
    }
}

pub fn gaussian_blur_vertical_pass_filter_neon<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    _: u32,
    filter: &Vec<GaussianFilter>,
    start_y: u32,
    end_y: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undef_unsafe_dst) };
    let zeros = unsafe { vdupq_n_f32(0f32) };

    let total_size = CHANNEL_CONFIGURATION * width as usize;

    for y in start_y..end_y {
        let y_dst_shift = y as usize * dst_stride as usize;

        let current_filter = unsafe { filter.get_unchecked(y as usize) };
        let filter_start = current_filter.start;
        let filter_weights = &current_filter.filter;

        let mut cx = 0usize;

        unsafe {
            while cx + 32 < total_size {
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
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8x2 = vld1q_u8_x2(s_ptr);
                    let hi_16 = vmovl_high_u8(pixels_u8x2.0);
                    let lo_16 = vmovl_u8(vget_low_u8(pixels_u8x2.0));
                    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16)));
                    store0 = prefer_vfmaq_f32(store0, lo_lo, f_weight);
                    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo_16));
                    store1 = prefer_vfmaq_f32(store1, lo_hi, f_weight);
                    let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16)));
                    store2 = prefer_vfmaq_f32(store2, hi_lo, f_weight);
                    let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi_16));
                    store3 = prefer_vfmaq_f32(store3, hi_hi, f_weight);

                    let hi_16 = vmovl_high_u8(pixels_u8x2.1);
                    let lo_16 = vmovl_u8(vget_low_u8(pixels_u8x2.1));
                    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16)));
                    store4 = prefer_vfmaq_f32(store4, lo_lo, f_weight);
                    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo_16));
                    store5 = prefer_vfmaq_f32(store5, lo_hi, f_weight);
                    let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16)));
                    store6 = prefer_vfmaq_f32(store6, hi_lo, f_weight);
                    let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi_16));
                    store7 = prefer_vfmaq_f32(store7, hi_hi, f_weight);

                    j += 1;
                }

                let store_0 = vcvtaq_u32_f32(store0);
                let store_1 = vcvtaq_u32_f32(store1);
                let store_2 = vcvtaq_u32_f32(store2);
                let store_3 = vcvtaq_u32_f32(store3);
                let store_4 = vcvtaq_u32_f32(store4);
                let store_5 = vcvtaq_u32_f32(store5);
                let store_6 = vcvtaq_u32_f32(store6);
                let store_7 = vcvtaq_u32_f32(store7);

                let store_lo = vcombine_u16(vmovn_u32(store_0), vmovn_u32(store_1));
                let store_hi = vcombine_u16(vmovn_u32(store_2), vmovn_u32(store_3));
                let store_x = vcombine_u8(vqmovn_u16(store_lo), vqmovn_u16(store_hi));

                let store_lo = vcombine_u16(vmovn_u32(store_4), vmovn_u32(store_5));
                let store_hi = vcombine_u16(vmovn_u32(store_6), vmovn_u32(store_7));
                let store_k = vcombine_u8(vqmovn_u16(store_lo), vqmovn_u16(store_hi));

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                let store = uint8x16x2_t(store_x, store_k);
                vst1q_u8_x2(dst_ptr, store);

                cx += 32;
            }

            while cx + 16 < total_size {
                let mut store0: float32x4_t = zeros;
                let mut store1: float32x4_t = zeros;
                let mut store2: float32x4_t = zeros;
                let mut store3: float32x4_t = zeros;

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = vld1q_u8(s_ptr);
                    let hi_16 = vmovl_high_u8(pixels_u8);
                    let lo_16 = vmovl_u8(vget_low_u8(pixels_u8));
                    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16)));
                    store0 = prefer_vfmaq_f32(store0, lo_lo, f_weight);
                    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo_16));
                    store1 = prefer_vfmaq_f32(store1, lo_hi, f_weight);
                    let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16)));
                    store2 = prefer_vfmaq_f32(store2, hi_lo, f_weight);
                    let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi_16));
                    store3 = prefer_vfmaq_f32(store3, hi_hi, f_weight);

                    j += 1;
                }

                let store_0 = vcvtaq_u32_f32(store0);
                let store_1 = vcvtaq_u32_f32(store1);
                let store_2 = vcvtaq_u32_f32(store2);
                let store_3 = vcvtaq_u32_f32(store3);

                let store_lo = vcombine_u16(vmovn_u32(store_0), vmovn_u32(store_1));
                let store_hi = vcombine_u16(vmovn_u32(store_2), vmovn_u32(store_3));
                let store = vcombine_u8(vqmovn_u16(store_lo), vqmovn_u16(store_hi));

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                vst1q_u8(dst_ptr, store);

                cx += 16;
            }

            while cx + 8 < total_size {
                let mut store0: float32x4_t = zeros;
                let mut store1: float32x4_t = zeros;

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = vld1_u8(s_ptr);
                    let pixels_u16 = vmovl_u8(pixels_u8);
                    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(pixels_u16)));
                    store0 = prefer_vfmaq_f32(store0, lo_lo, f_weight);
                    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(pixels_u16));
                    store1 = prefer_vfmaq_f32(store1, lo_hi, f_weight);

                    j += 1;
                }

                let store_0 = vcvtaq_u32_f32(store0);
                let store_1 = vcvtaq_u32_f32(store1);

                let store_lo = vcombine_u16(vmovn_u32(store_0), vmovn_u32(store_1));
                let store = vqmovn_u16(store_lo);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                vst1_u8(dst_ptr, store);

                cx += 8;
            }

            while cx + 4 < total_size {
                let mut store0: float32x4_t = zeros;

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u32 = load_u8_u32_fast::<4>(s_ptr);
                    let lo_lo = vcvtq_f32_u32(pixels_u32);
                    store0 = prefer_vfmaq_f32(store0, lo_lo, f_weight);

                    j += 1;
                }

                let store_0 = vcvtaq_u32_f32(store0);

                let store_c = vmovn_u32(store_0);
                let store_lo = vcombine_u16(store_c, store_c);
                let store = vqmovn_u16(store_lo);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u32;

                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store));
                dst_ptr.write_unaligned(pixel);

                cx += 4;
            }

            while cx < total_size {
                let mut store0 = vdup_n_f32(0f32);

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight = vdup_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u32 = load_u8_u32_one(s_ptr);
                    let lo_lo = vcvt_f32_u32(pixels_u32);
                    store0 = prefer_vfma_f32(store0, lo_lo, f_weight);

                    j += 1;
                }

                let store_0 = vcvta_u32_f32(store0);

                let store_c = vmovn_u32(vcombine_u32(store_0, store_0));
                let store_lo = vcombine_u16(store_c, store_c);
                let store = vqmovn_u16(store_lo);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store));
                let bytes = pixel.to_le_bytes();
                dst_ptr.write_unaligned(bytes[0]);

                cx += 1;
            }
        }
    }
}
