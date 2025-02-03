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

use crate::neon::{load_u8_u16, load_u8_u32_fast, store_u8_u32, vmulq_u32_f32};
use crate::unsafe_slice::UnsafeSlice;

pub(crate) fn box_blur_horizontal_pass_neon<T, const CHANNEL_CONFIGURATION: usize>(
    undefined_src: &[T],
    src_stride: u32,
    undefined_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undefined_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undefined_dst) };

    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = unsafe { vdupq_n_u32(edge_count) };

    let v_weight = unsafe { vdupq_n_f32(1f32 / (radius * 2) as f32) };

    let half_kernel = kernel_size / 2;

    let mut yy = start_y;

    for y in (yy..end_y.saturating_sub(4)).step_by(4) {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store_0: uint32x4_t;
        let mut store_1: uint32x4_t;
        let mut store_2: uint32x4_t;
        let mut store_3: uint32x4_t;
        {
            let s_ptr_0 = unsafe { src.as_ptr().add(y_src_shift) };
            let edge_colors_0 = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr_0) };
            store_0 = unsafe { vmulq_u32(edge_colors_0, v_edge_count) };

            let s_ptr_1 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize) };
            let edge_colors_1 = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr_1) };
            store_1 = unsafe { vmulq_u32(edge_colors_1, v_edge_count) };

            let s_ptr_2 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize * 2) };
            let edge_colors_2 = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr_2) };
            store_2 = unsafe { vmulq_u32(edge_colors_2, v_edge_count) };

            let s_ptr_3 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize * 3) };
            let edge_colors_3 = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr_3) };
            store_3 = unsafe { vmulq_u32(edge_colors_3, v_edge_count) };
        }

        for x in 1..std::cmp::min(half_kernel, width) {
            let px = x as usize * CHANNEL_CONFIGURATION;

            let s_ptr_0 = unsafe { src.as_ptr().add(y_src_shift + px) };
            let edge_colors_0 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_0) };
            store_0 = unsafe { vaddw_u16(store_0, edge_colors_0) };

            let s_ptr_1 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize + px) };
            let edge_colors_1 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_1) };
            store_1 = unsafe { vaddw_u16(store_1, edge_colors_1) };

            let s_ptr_2 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px) };
            let edge_colors_2 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_2) };
            store_2 = unsafe { vaddw_u16(store_2, edge_colors_2) };

            let s_ptr_3 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px) };
            let edge_colors_3 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_3) };
            store_3 = unsafe { vaddw_u16(store_3, edge_colors_3) };
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            {
                let previous_x = std::cmp::max(x as i64 - half_kernel as i64, 0) as usize;
                let previous = previous_x * CHANNEL_CONFIGURATION;

                let s_ptr_0 = unsafe { src.as_ptr().add(y_src_shift + previous) };
                let edge_colors_0 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_0) };
                store_0 = unsafe { vsubw_u16(store_0, edge_colors_0) };

                let s_ptr_1 = unsafe {
                    src.as_ptr()
                        .add(y_src_shift + src_stride as usize + previous)
                };
                let edge_colors_1 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_1) };
                store_1 = unsafe { vsubw_u16(store_1, edge_colors_1) };

                let s_ptr_2 = unsafe {
                    src.as_ptr()
                        .add(y_src_shift + src_stride as usize * 2 + previous)
                };
                let edge_colors_2 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_2) };
                store_2 = unsafe { vsubw_u16(store_2, edge_colors_2) };

                let s_ptr_3 = unsafe {
                    src.as_ptr()
                        .add(y_src_shift + src_stride as usize * 3 + previous)
                };
                let edge_colors_3 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_3) };
                store_3 = unsafe { vsubw_u16(store_3, edge_colors_3) };
            }

            // add next
            {
                let next_x = std::cmp::min(x + half_kernel, width - 1) as usize;

                let next = next_x * CHANNEL_CONFIGURATION;

                let s_ptr_0 = unsafe { src.as_ptr().add(y_src_shift + next) };
                let edge_colors_0 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_0) };
                store_0 = unsafe { vaddw_u16(store_0, edge_colors_0) };

                let s_ptr_1 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize + next) };
                let edge_colors_1 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_1) };
                store_1 = unsafe { vaddw_u16(store_1, edge_colors_1) };

                let s_ptr_2 = unsafe {
                    src.as_ptr()
                        .add(y_src_shift + src_stride as usize * 2 + next)
                };
                let edge_colors_2 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_2) };
                store_2 = unsafe { vaddw_u16(store_2, edge_colors_2) };

                let s_ptr_3 = unsafe {
                    src.as_ptr()
                        .add(y_src_shift + src_stride as usize * 3 + next)
                };
                let edge_colors_3 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_3) };
                store_3 = unsafe { vaddw_u16(store_3, edge_colors_3) };
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

            let scale_store = unsafe { vmulq_u32_f32(store_0, v_weight) };
            let px_16 = unsafe { vqmovn_u32(scale_store) };
            let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
            let pixel_0 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

            let scale_store = unsafe { vmulq_u32_f32(store_1, v_weight) };
            let px_16 = unsafe { vqmovn_u32(scale_store) };
            let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
            let pixel_1 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

            let scale_store = unsafe { vmulq_u32_f32(store_2, v_weight) };
            let px_16 = unsafe { vqmovn_u32(scale_store) };
            let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
            let pixel_2 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

            let scale_store = unsafe { vmulq_u32_f32(store_3, v_weight) };
            let px_16 = unsafe { vqmovn_u32(scale_store) };
            let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
            let pixel_3 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

            let bytes_offset_0 = y_dst_shift + px;
            let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
            let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
            let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;
            if CHANNEL_CONFIGURATION == 4 {
                unsafe {
                    let dst_ptr_0 = unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut u32;
                    dst_ptr_0.write_unaligned(pixel_0);

                    let dst_ptr_1 = unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut u32;
                    dst_ptr_1.write_unaligned(pixel_1);

                    let dst_ptr_2 = unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut u32;
                    dst_ptr_2.write_unaligned(pixel_2);

                    let dst_ptr_3 = unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut u32;
                    dst_ptr_3.write_unaligned(pixel_3);
                }
            } else {
                let bits_0 = pixel_0.to_le_bytes();
                unsafe {
                    unsafe_dst.write(bytes_offset_0, bits_0[0]);
                    unsafe_dst.write(bytes_offset_0 + 1, bits_0[1]);
                    unsafe_dst.write(bytes_offset_0 + 2, bits_0[2]);
                }

                let bits_1 = pixel_1.to_le_bytes();
                unsafe {
                    unsafe_dst.write(bytes_offset_1, bits_1[0]);
                    unsafe_dst.write(bytes_offset_1 + 1, bits_1[1]);
                    unsafe_dst.write(bytes_offset_1 + 2, bits_1[2]);
                }

                let bits_2 = pixel_2.to_le_bytes();
                unsafe {
                    unsafe_dst.write(bytes_offset_2, bits_2[0]);
                    unsafe_dst.write(bytes_offset_2 + 1, bits_2[1]);
                    unsafe_dst.write(bytes_offset_2 + 2, bits_2[2]);
                }

                let bits_3 = pixel_3.to_le_bytes();
                unsafe {
                    unsafe_dst.write(bytes_offset_3, bits_3[0]);
                    unsafe_dst.write(bytes_offset_3 + 1, bits_3[1]);
                    unsafe_dst.write(bytes_offset_3 + 2, bits_3[2]);
                }
            }
        }

        yy = y;
    }

    for y in yy..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store: uint32x4_t;
        {
            let s_ptr = unsafe { src.as_ptr().add(y_src_shift) };
            let edge_colors = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr) };
            store = unsafe { vmulq_u32(edge_colors, v_edge_count) };
        }

        for x in 1..std::cmp::min(half_kernel, width) {
            let px = x as usize * CHANNEL_CONFIGURATION;
            let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
            let edge_colors = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
            store = unsafe { vaddw_u16(store, edge_colors) };
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            {
                let previous_x = std::cmp::max(x as i64 - half_kernel as i64, 0) as usize;
                let previous = previous_x * CHANNEL_CONFIGURATION;
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + previous) };
                let edge_colors = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                store = unsafe { vsubw_u16(store, edge_colors) };
            }

            // add next
            {
                let next_x = std::cmp::min(x + half_kernel, width - 1) as usize;

                let next = next_x * CHANNEL_CONFIGURATION;

                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + next) };
                let edge_colors = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                store = unsafe { vaddw_u16(store, edge_colors) };
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

            let scale_store = unsafe { vmulq_u32_f32(store, v_weight) };

            let bytes_offset = y_dst_shift + px;
            unsafe {
                let dst_ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u8;
                store_u8_u32::<CHANNEL_CONFIGURATION>(dst_ptr, scale_store);
            }
        }
    }
}

pub(crate) fn box_blur_vertical_pass_neon<T, const CHANNEL_CONFIGURATION: usize>(
    undefined_src: &[T],
    src_stride: u32,
    undefined_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    _: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undefined_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undefined_unsafe_dst) };

    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = unsafe { vdupq_n_u32(edge_count) };

    let half_kernel = kernel_size / 2;

    let start_x = start_x as usize;
    let end_x = end_x as usize;

    let v_weight = unsafe { vdupq_n_f32(1f32 / (radius * 2) as f32) };

    let mut cx = start_x;

    while cx + 16 < end_x {
        let px = cx;

        let mut store_0: uint32x4_t;
        let mut store_1: uint32x4_t;
        let mut store_2: uint32x4_t;
        let mut store_3: uint32x4_t;

        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge = vld1q_u8(s_ptr);
            let lo0 = vmovl_u8(vget_low_u8(edge));
            let hi0 = vmovl_high_u8(edge);

            let i16_l0 = vmovl_u16(vget_low_u16(lo0));
            let i16_h0 = vmovl_high_u16(lo0);
            let i16_l1 = vmovl_u16(vget_low_u16(hi0));
            let i16_h1 = vmovl_high_u16(hi0);

            store_0 = vmulq_u32(i16_l0, v_edge_count);
            store_1 = vmulq_u32(i16_h0, v_edge_count);

            store_2 = vmulq_u32(i16_l1, v_edge_count);
            store_3 = vmulq_u32(i16_h1, v_edge_count);
        }

        unsafe {
            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge = vld1q_u8(s_ptr);
                let lo0 = vmovl_u8(vget_low_u8(edge));
                let hi0 = vmovl_high_u8(edge);

                let i16_l0 = vget_low_u16(lo0);
                let i16_h0 = lo0;
                let i16_l1 = vget_low_u16(hi0);
                let i16_h1 = hi0;

                store_0 = vaddw_u16(store_0, i16_l0);
                store_1 = vaddw_high_u16(store_1, i16_h0);
                store_2 = vaddw_u16(store_2, i16_l1);
                store_3 = vaddw_high_u16(store_3, i16_h1);
            }
        }

        unsafe {
            for y in 0..height {
                // preload edge pixels
                let next =
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge = vld1q_u8(s_ptr);
                    let lo0 = vmovl_u8(vget_low_u8(edge));
                    let hi0 = vmovl_high_u8(edge);

                    let i16_l0 = vget_low_u16(lo0);
                    let i16_h0 = lo0;
                    let i16_l1 = vget_low_u16(hi0);
                    let i16_h1 = hi0;

                    store_0 = vsubw_u16(store_0, i16_l0);
                    store_1 = vsubw_high_u16(store_1, i16_h0);
                    store_2 = vsubw_u16(store_2, i16_l1);
                    store_3 = vsubw_high_u16(store_3, i16_h1);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge = vld1q_u8(s_ptr);
                    let lo0 = vmovl_u8(vget_low_u8(edge));
                    let hi0 = vmovl_high_u8(edge);

                    let i16_l0 = vget_low_u16(lo0);
                    let i16_h0 = lo0;
                    let i16_l1 = vget_low_u16(hi0);
                    let i16_h1 = hi0;

                    store_0 = vaddw_u16(store_0, i16_l0);
                    store_1 = vaddw_high_u16(store_1, i16_h0);
                    store_2 = vaddw_u16(store_2, i16_l1);
                    store_3 = vaddw_high_u16(store_3, i16_h1);
                }

                let px = cx;

                let scale_store_0 = vmulq_u32_f32(store_0, v_weight);
                let scale_store_1 = vmulq_u32_f32(store_1, v_weight);
                let scale_store_2 = vmulq_u32_f32(store_2, v_weight);
                let scale_store_3 = vmulq_u32_f32(store_3, v_weight);

                let offset = y_dst_shift + px;
                let ptr = unsafe_dst.slice.get_unchecked(offset).get();
                let px_16_lo0 = vqmovn_u32(scale_store_0);
                let px_16_hi0 = vqmovn_u32(scale_store_1);
                let px_16_lo1 = vqmovn_u32(scale_store_2);
                let px_16_hi2 = vqmovn_u32(scale_store_3);
                let px0 = vqmovn_u16(vcombine_u16(px_16_lo0, px_16_hi0));
                let px1 = vqmovn_u16(vcombine_u16(px_16_lo1, px_16_hi2));
                vst1q_u8(ptr, vcombine_u8(px0, px1));
            }
        }

        cx += 16;
    }

    while cx + 8 < end_x {
        let px = cx;

        let mut store_0: uint32x4_t;
        let mut store_1: uint32x4_t;

        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge = vld1_u8(s_ptr);
            let lo0 = vmovl_u8(edge);

            let i16_l0 = vmovl_u16(vget_low_u16(lo0));
            let i16_h0 = vmovl_high_u16(lo0);

            store_0 = vmulq_u32(i16_l0, v_edge_count);
            store_1 = vmulq_u32(i16_h0, v_edge_count);
        }

        unsafe {
            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge = vld1_u8(s_ptr);
                let lo0 = vmovl_u8(edge);

                let i16_l0 = vget_low_u16(lo0);
                let i16_h0 = lo0;

                store_0 = vaddw_u16(store_0, i16_l0);
                store_1 = vaddw_high_u16(store_1, i16_h0);
            }
        }

        unsafe {
            for y in 0..height {
                // preload edge pixels
                let next =
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge = vld1_u8(s_ptr);
                    let lo0 = vmovl_u8(edge);

                    let i16_l0 = vget_low_u16(lo0);
                    let i16_h0 = lo0;

                    store_0 = vsubw_u16(store_0, i16_l0);
                    store_1 = vsubw_high_u16(store_1, i16_h0);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge = vld1_u8(s_ptr);
                    let lo0 = vmovl_u8(edge);

                    let i16_l0 = vget_low_u16(lo0);
                    let i16_h0 = lo0;

                    store_0 = vaddw_u16(store_0, i16_l0);
                    store_1 = vaddw_high_u16(store_1, i16_h0);
                }

                let px = cx;

                let scale_store_0 = vmulq_u32_f32(store_0, v_weight);
                let scale_store_1 = vmulq_u32_f32(store_1, v_weight);

                let offset = y_dst_shift + px;
                let ptr = unsafe_dst.slice.get_unchecked(offset).get();
                let px_16_lo0 = vqmovn_u32(scale_store_0);
                let px_16_hi0 = vqmovn_u32(scale_store_1);
                let px0 = vqmovn_u16(vcombine_u16(px_16_lo0, px_16_hi0));
                vst1_u8(ptr, px0);
            }
        }

        cx += 8;
    }

    while cx + CHANNEL_CONFIGURATION * 2 < end_x {
        let px = cx;

        let mut store_0: uint32x4_t;
        let mut store_1: uint32x4_t;

        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge_colors_0 = load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
            let edge_colors_1 =
                load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr.add(CHANNEL_CONFIGURATION));
            store_0 = vmulq_u32(edge_colors_0, v_edge_count);
            store_1 = vmulq_u32(edge_colors_1, v_edge_count);
        }

        unsafe {
            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let edge_colors_0 = load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr);
                let edge_colors_1 =
                    load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr.add(CHANNEL_CONFIGURATION));
                store_0 = vaddw_u16(store_0, edge_colors_0);
                store_1 = vaddw_u16(store_1, edge_colors_1);
            }
        }

        unsafe {
            for y in 0..height {
                // preload edge pixels
                let next =
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge_colors_0 = load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr);
                    let edge_colors_1 =
                        load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr.add(CHANNEL_CONFIGURATION));
                    store_0 = vsubw_u16(store_0, edge_colors_0);
                    store_1 = vsubw_u16(store_1, edge_colors_1);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge_colors_0 = load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr);
                    let edge_colors_1 =
                        load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr.add(CHANNEL_CONFIGURATION));
                    store_0 = vaddw_u16(store_0, edge_colors_0);
                    store_1 = vaddw_u16(store_1, edge_colors_1);
                }

                let px = cx;

                let scale_store_0 = vmulq_u32_f32(store_0, v_weight);
                let scale_store_1 = vmulq_u32_f32(store_1, v_weight);
                if CHANNEL_CONFIGURATION == 3 {
                    let px_16 = vqmovn_u32(scale_store_0);
                    let px_8 = vqmovn_u16(vcombine_u16(px_16, px_16));
                    let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(px_8));
                    let pixel_bits_0 = pixel.to_le_bytes();

                    let px_16 = vqmovn_u32(scale_store_1);
                    let px_8 = vqmovn_u16(vcombine_u16(px_16, px_16));
                    let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(px_8));
                    let pixel_bits_1 = pixel.to_le_bytes();

                    {
                        let offset = y_dst_shift + px;
                        unsafe_dst.write(offset, pixel_bits_0[0]);
                        unsafe_dst.write(offset + 1, pixel_bits_0[1]);
                        unsafe_dst.write(offset + 2, pixel_bits_0[2]);
                        if CHANNEL_CONFIGURATION == 4 {
                            unsafe_dst.write(offset + 3, pixel_bits_0[3]);
                        }
                    }

                    {
                        let offset = y_dst_shift + px + CHANNEL_CONFIGURATION;
                        unsafe_dst.write(offset, pixel_bits_1[0]);
                        unsafe_dst.write(offset + 1, pixel_bits_1[1]);
                        unsafe_dst.write(offset + 2, pixel_bits_1[2]);
                        if CHANNEL_CONFIGURATION == 4 {
                            unsafe_dst.write(offset + 3, pixel_bits_1[3]);
                        }
                    }
                } else {
                    let offset = y_dst_shift + px;
                    let ptr = unsafe_dst.slice.get_unchecked(offset).get();
                    let px_16_lo = vqmovn_u32(scale_store_0);
                    let px_16_hi = vqmovn_u32(scale_store_1);
                    let px = vqmovn_u16(vcombine_u16(px_16_lo, px_16_hi));
                    vst1_u8(ptr, px);
                }
            }
        }

        cx += CHANNEL_CONFIGURATION * 2;
    }

    const TAIL_CN: usize = 1;

    unsafe {
        for x in cx..end_x {
            let px = x;

            let mut store: uint32x4_t;
            {
                let s_ptr = src.as_ptr().add(px);
                let edge_colors = load_u8_u32_fast::<TAIL_CN>(s_ptr);
                store = vmulq_u32(edge_colors, v_edge_count);
            }

            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let edge_colors = load_u8_u16::<TAIL_CN>(s_ptr);
                store = vaddw_u16(store, edge_colors);
            }

            for y in 0..height {
                // preload edge pixels
                let next =
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge_colors = load_u8_u16::<TAIL_CN>(s_ptr);
                    store = vsubw_u16(store, edge_colors);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge_colors = load_u8_u16::<TAIL_CN>(s_ptr);
                    store = vaddw_u16(store, edge_colors);
                }

                let px = x;

                let scale_store = vmulq_u32_f32(store, v_weight);
                let bytes_offset = y_dst_shift + px;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u8;
                store_u8_u32::<TAIL_CN>(dst_ptr, scale_store);
            }
        }
    }
}
