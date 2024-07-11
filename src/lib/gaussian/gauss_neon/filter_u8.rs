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
use crate::neon::{
    load_u8_u16_x2_fast, load_u8_u32_fast, load_u8_u32_one, prefer_vfma_f32, prefer_vfmaq_f32,
};
use std::arch::aarch64::*;

use crate::unsafe_slice::UnsafeSlice;

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
    let src: &[u8] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undef_unsafe_dst) };
    let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
    let shuffle_1 = unsafe { vld1_u8(shuf_table_1.as_ptr()) };
    let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
    let shuffle_2 = unsafe { vld1_u8(shuf_table_2.as_ptr()) };
    let shuffle = unsafe { vcombine_u8(shuffle_1, shuffle_2) };

    let mut cy = start_y;

    for y in (cy..end_y.saturating_sub(2)).step_by(2) {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let mut store_0: float32x4_t = unsafe { vdupq_n_f32(0f32) };
            let mut store_1: float32x4_t = unsafe { vdupq_n_f32(0f32) };

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
                    let mut pixel_colors_0: uint8x16_t = vld1q_u8(s_ptr);
                    let mut pixel_colors_1: uint8x16_t = vld1q_u8(s_ptr.add(src_stride as usize));
                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors_0 = vqtbl1q_u8(pixel_colors_0, shuffle);
                        pixel_colors_1 = vqtbl1q_u8(pixel_colors_1, shuffle);
                    }
                    let mut pixel_colors_u16_0 = vmovl_u8(vget_low_u8(pixel_colors_0));
                    let mut pixel_colors_u32_0 = vmovl_u16(vget_low_u16(pixel_colors_u16_0));
                    let mut pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);

                    let mut pixel_colors_u16_1 = vmovl_u8(vget_low_u8(pixel_colors_1));
                    let mut pixel_colors_u32_1 = vmovl_u16(vget_low_u16(pixel_colors_u16_1));
                    let mut pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    let mut weight = *filter_weights.get_unchecked(j);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    pixel_colors_u32_0 = vmovl_high_u16(pixel_colors_u16_0);
                    pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);

                    pixel_colors_u32_1 = vmovl_high_u16(pixel_colors_u16_1);
                    pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    weight = *filter_weights.get_unchecked(j + 1);
                    f_weight = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    pixel_colors_u16_0 = vmovl_u8(vget_high_u8(pixel_colors_0));
                    pixel_colors_u16_1 = vmovl_u8(vget_high_u8(pixel_colors_1));

                    pixel_colors_u32_0 = vmovl_u16(vget_low_u16(pixel_colors_u16_0));
                    pixel_colors_u32_1 = vmovl_u16(vget_low_u16(pixel_colors_u16_1));

                    pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);
                    pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    weight = *filter_weights.get_unchecked(j + 2);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    pixel_colors_u32_0 = vmovl_high_u16(pixel_colors_u16_0);
                    pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);

                    pixel_colors_u32_1 = vmovl_high_u16(pixel_colors_u16_1);
                    pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    weight = *filter_weights.get_unchecked(j + 3);
                    f_weight = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    j += 4;
                }
            }

            unsafe {
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
                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors_0 = vtbl1_u8(pixel_colors_0, shuffle_1);
                        pixel_colors_1 = vtbl1_u8(pixel_colors_1, shuffle_1);
                    }
                    let pixel_colors_u16_0 = vmovl_u8(pixel_colors_0);
                    let mut pixel_colors_u32_0 = vmovl_u16(vget_low_u16(pixel_colors_u16_0));
                    let mut pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);

                    let pixel_colors_u16_1 = vmovl_u8(pixel_colors_1);
                    let mut pixel_colors_u32_1 = vmovl_u16(vget_low_u16(pixel_colors_u16_1));
                    let mut pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    let mut weight = *filter_weights.get_unchecked(j);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    pixel_colors_u32_0 = vmovl_high_u16(pixel_colors_u16_0);
                    pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);

                    pixel_colors_u32_1 = vmovl_high_u16(pixel_colors_u16_1);
                    pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    weight = *filter_weights.get_unchecked(j + 1);
                    f_weight = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    j += 2;
                }
            }

            unsafe {
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
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

            let px_16 = unsafe { vqmovn_u32(vcvtaq_u32_f32(store_0)) };
            let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
            let pixel_0 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

            let px_16 = unsafe { vqmovn_u32(vcvtaq_u32_f32(store_1)) };
            let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
            let pixel_1 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

            if CHANNEL_CONFIGURATION == 4 {
                unsafe {
                    let unsafe_offset = y_dst_shift + px;
                    let dst_ptr = unsafe_dst.slice.as_ptr().add(unsafe_offset) as *mut u32;
                    dst_ptr.write_unaligned(pixel_0);
                }
            } else {
                let pixel_bytes_0 = pixel_0.to_le_bytes();
                unsafe {
                    let offset = y_dst_shift + px;
                    unsafe_dst.write(offset, pixel_bytes_0[0]);
                    unsafe_dst.write(offset + 1, pixel_bytes_0[1]);
                    unsafe_dst.write(offset + 2, pixel_bytes_0[2]);
                }
            }

            let offset = y_dst_shift + px + src_stride as usize;
            if CHANNEL_CONFIGURATION == 4 {
                unsafe {
                    let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut u32;
                    dst_ptr.write_unaligned(pixel_1);
                }
            } else {
                let pixel_bytes_1 = pixel_1.to_le_bytes();
                unsafe {
                    unsafe_dst.write(offset, pixel_bytes_1[0]);
                    unsafe_dst.write(offset + 1, pixel_bytes_1[1]);
                    unsafe_dst.write(offset + 2, pixel_bytes_1[2]);
                }
            }
        }
        cy = y;
    }

    for y in cy..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let mut store: float32x4_t = unsafe { vdupq_n_f32(0f32) };

            let mut j = 0;

            let current_filter = unsafe { filter.get_unchecked(x as usize) };
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            unsafe {
                while j + 4 < current_filter.size
                    && filter_start as i64 + j as i64 + 6 < width as i64
                {
                    let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors: uint8x16_t = vld1q_u8(s_ptr);
                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors = vqtbl1q_u8(pixel_colors, shuffle);
                    }
                    let mut pixel_colors_u16 = vmovl_u8(vget_low_u8(pixel_colors));
                    let mut pixel_colors_u32 = vmovl_u16(vget_low_u16(pixel_colors_u16));
                    let mut pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);
                    let mut weight = *filter_weights.get_unchecked(j);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    pixel_colors_u32 = vmovl_high_u16(pixel_colors_u16);
                    pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);

                    weight = *filter_weights.get_unchecked(j + 1);
                    f_weight = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    pixel_colors_u16 = vmovl_u8(vget_high_u8(pixel_colors));

                    pixel_colors_u32 = vmovl_u16(vget_low_u16(pixel_colors_u16));
                    let mut pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);
                    let mut weight = *filter_weights.get_unchecked(j + 2);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    pixel_colors_u32 = vmovl_high_u16(pixel_colors_u16);
                    pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);

                    weight = *filter_weights.get_unchecked(j + 3);
                    f_weight = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    j += 4;
                }
            }

            unsafe {
                while j + 2 < current_filter.size
                    && filter_start as i64 + j as i64 + 2 < width as i64
                {
                    let current_x = filter_start + j;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let weight_0 = *filter_weights.get_unchecked(j);
                    let weight_1 = *filter_weights.get_unchecked(j + 1);
                    let pixel_colors_u32 = load_u8_u16_x2_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let pixel_colors_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(pixel_colors_u32)));
                    let f_weight = vdupq_n_f32(weight_0);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    let pixel_colors_f32 = vcvtq_f32_u32(vmovl_high_u16(pixel_colors_u32));
                    let f_weight = vdupq_n_f32(weight_1);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    j += 2;
                }
            }

            unsafe {
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
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

            let px_16 = unsafe { vqmovn_u32(vcvtaq_u32_f32(store)) };
            let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
            let pixel = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };
            let offset = y_dst_shift + px;
            if CHANNEL_CONFIGURATION == 4 {
                unsafe {
                    let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut u32;
                    dst_ptr.write_unaligned(pixel);
                }
            } else {
                let bits = pixel.to_le_bytes();
                unsafe {
                    unsafe_dst.write(offset, bits[0]);
                    unsafe_dst.write(offset + 1, bits[1]);
                    unsafe_dst.write(offset + 2, bits[2]);
                }
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
