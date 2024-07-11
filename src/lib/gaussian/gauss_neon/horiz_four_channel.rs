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

use std::arch::aarch64::*;

use crate::neon::{load_u8_f32_fast, load_u8_u16_x2_fast, load_u8_u32_fast, prefer_vfmaq_f32};
use crate::unsafe_slice::UnsafeSlice;

pub fn gaussian_blur_horizontal_pass_neon<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let src: &[u8] = std::mem::transmute(undef_src);
        let unsafe_dst: &UnsafeSlice<'_, u8> = std::mem::transmute(undef_unsafe_dst);
        let half_kernel = (kernel_size / 2) as i32;

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
                let mut store_0: float32x4_t = vdupq_n_f32(0f32);
                let mut store_1: float32x4_t = vdupq_n_f32(0f32);

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let pixel_colors_f32_0 = load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift + src_stride as usize); // Here we're always at zero
                    let pixel_colors_f32_1 = load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next);
                    for i in 0..diff as usize {
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = vdupq_n_f32(weights.read_unaligned());
                        store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                        store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);
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

                    let mut weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    pixel_colors_u32_0 = vmovl_high_u16(pixel_colors_u16_0);
                    pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);

                    pixel_colors_u32_1 = vmovl_high_u16(pixel_colors_u16_1);
                    pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    weight = *kernel.get_unchecked((r + half_kernel + 1) as usize);
                    f_weight = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    pixel_colors_u16_0 = vmovl_u8(vget_high_u8(pixel_colors_0));
                    pixel_colors_u16_1 = vmovl_u8(vget_high_u8(pixel_colors_1));

                    pixel_colors_u32_0 = vmovl_u16(vget_low_u16(pixel_colors_u16_0));
                    pixel_colors_u32_1 = vmovl_u16(vget_low_u16(pixel_colors_u16_1));

                    pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);
                    pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    weight = *kernel.get_unchecked((r + half_kernel + 2) as usize);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    pixel_colors_u32_0 = vmovl_high_u16(pixel_colors_u16_0);
                    pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);

                    pixel_colors_u32_1 = vmovl_high_u16(pixel_colors_u16_1);
                    pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    weight = *kernel.get_unchecked((r + half_kernel + 3) as usize);
                    f_weight = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    r += 4;
                }

                while r + 2 <= half_kernel
                    && x as i64 + r as i64 + (if CHANNEL_CONFIGURATION == 4 { 2 } else { 3 })
                        < width as i64
                {
                    let px =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize
                            * CHANNEL_CONFIGURATION;
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

                    let mut weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    pixel_colors_u32_0 = vmovl_high_u16(pixel_colors_u16_0);
                    pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);

                    pixel_colors_u32_1 = vmovl_high_u16(pixel_colors_u16_1);
                    pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);

                    weight = *kernel.get_unchecked((r + half_kernel + 1) as usize);
                    f_weight = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    r += 2;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u32_0 = load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let pixel_colors_u32_1 =
                        load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr.add(src_stride as usize));
                    let pixel_colors_f32_0 = vcvtq_f32_u32(pixel_colors_u32_0);
                    let pixel_colors_f32_1 = vcvtq_f32_u32(pixel_colors_u32_1);
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);
                    store_0 = prefer_vfmaq_f32(store_0, pixel_colors_f32_0, f_weight);
                    store_1 = prefer_vfmaq_f32(store_1, pixel_colors_f32_1, f_weight);

                    r += 1;
                }

                let px = x as usize * CHANNEL_CONFIGURATION;

                let px_16 = vqmovn_u32(vcvtaq_u32_f32(store_0));
                let px_8 = vqmovn_u16(vcombine_u16(px_16, px_16));
                let pixel_0 = vget_lane_u32::<0>(vreinterpret_u32_u8(px_8));

                let px_16 = vqmovn_u32(vcvtaq_u32_f32(store_1));
                let px_8 = vqmovn_u16(vcombine_u16(px_16, px_16));
                let pixel_1 = vget_lane_u32::<0>(vreinterpret_u32_u8(px_8));

                if CHANNEL_CONFIGURATION == 4 {
                    let unsafe_offset = y_dst_shift + px;
                    let dst_ptr = unsafe_dst.slice.as_ptr().add(unsafe_offset) as *mut u32;
                    dst_ptr.write_unaligned(pixel_0);
                } else {
                    let pixel_bytes_0 = pixel_0.to_le_bytes();
                    let offset = y_dst_shift + px;
                    unsafe_dst.write(offset, pixel_bytes_0[0]);
                    unsafe_dst.write(offset + 1, pixel_bytes_0[1]);
                    unsafe_dst.write(offset + 2, pixel_bytes_0[2]);
                }

                let offset = y_dst_shift + px + src_stride as usize;
                if CHANNEL_CONFIGURATION == 4 {
                    let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut u32;
                    dst_ptr.write_unaligned(pixel_1);
                } else {
                    let pixel_bytes_1 = pixel_1.to_le_bytes();
                    unsafe_dst.write(offset, pixel_bytes_1[0]);
                    unsafe_dst.write(offset + 1, pixel_bytes_1[1]);
                    unsafe_dst.write(offset + 2, pixel_bytes_1[2]);
                }
            }
            cy = y;
        }

        for y in cy..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store: float32x4_t = vdupq_n_f32(0f32);

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let pixel_colors_f32_0 = load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    for i in 0..diff as usize {
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = vdupq_n_f32(weights.read_unaligned());
                        store = prefer_vfmaq_f32(store, pixel_colors_f32_0, f_weight);
                    }
                    r += diff as i32;
                }

                while r + 4 <= half_kernel && x as i64 + r as i64 + 6 < width as i64 {
                    let px =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize
                            * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors: uint8x16_t = vld1q_u8(s_ptr);
                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors = vqtbl1q_u8(pixel_colors, shuffle);
                    }
                    let mut pixel_colors_u16 = vmovl_u8(vget_low_u8(pixel_colors));
                    let mut pixel_colors_u32 = vmovl_u16(vget_low_u16(pixel_colors_u16));
                    let mut pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);
                    let mut weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    pixel_colors_u32 = vmovl_high_u16(pixel_colors_u16);
                    pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);

                    weight = *kernel.get_unchecked((r + half_kernel + 1) as usize);
                    f_weight = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    pixel_colors_u16 = vmovl_u8(vget_high_u8(pixel_colors));

                    pixel_colors_u32 = vmovl_u16(vget_low_u16(pixel_colors_u16));
                    let mut pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);
                    let mut weight = *kernel.get_unchecked((r + half_kernel + 2) as usize);
                    let mut f_weight: float32x4_t = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    pixel_colors_u32 = vmovl_high_u16(pixel_colors_u16);
                    pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);

                    weight = *kernel.get_unchecked((r + half_kernel + 3) as usize);
                    f_weight = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 4;
                }

                while r + 2 <= half_kernel && x as i64 + r as i64 + 2 < width as i64 {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let weight_0 = *kernel.get_unchecked((r + half_kernel) as usize);
                    let weight_1 = *kernel.get_unchecked((r + half_kernel + 1) as usize);
                    let pixel_colors_u32 = load_u8_u16_x2_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let pixel_colors_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(pixel_colors_u32)));
                    let f_weight = vdupq_n_f32(weight_0);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    let pixel_colors_f32 = vcvtq_f32_u32(vmovl_high_u16(pixel_colors_u32));
                    let f_weight = vdupq_n_f32(weight_1);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 2;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u32 = load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let pixel_colors_f32 = vcvtq_f32_u32(pixel_colors_u32);
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 1;
                }

                let px = x as usize * CHANNEL_CONFIGURATION;

                let px_16 = vqmovn_u32(vcvtaq_u32_f32(store));
                let px_8 = vqmovn_u16(vcombine_u16(px_16, px_16));
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(px_8));

                let offset = y_dst_shift + px;
                if CHANNEL_CONFIGURATION == 4 {
                    let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut u32;
                    dst_ptr.write_unaligned(pixel);
                } else {
                    let bits = pixel.to_le_bytes();
                    unsafe_dst.write(offset, bits[0]);
                    unsafe_dst.write(offset + 1, bits[1]);
                    unsafe_dst.write(offset + 2, bits[2]);
                }
            }
        }
    }
}
