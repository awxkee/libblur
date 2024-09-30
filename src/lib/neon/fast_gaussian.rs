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

use crate::neon::{load_u8_s32_fast, store_u8x8_m4, vmulq_s32_f32};
use crate::{clamp_edge, reflect_101, reflect_index, EdgeMode};
use std::arch::aarch64::*;

use crate::unsafe_slice::UnsafeSlice;

pub fn fast_gaussian_horizontal_pass_neon_u8<T, const CHANNELS_COUNT: usize>(
    undefined_slice: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let bytes: &UnsafeSlice<'_, u8> = std::mem::transmute(undefined_slice);
        let mut buffer: [[i32; 4]; 1024] = [[0; 4]; 1024];
        let initial_sum = ((radius * radius) >> 1) as i32;

        let radius_64 = radius as i64;
        let width_wide = width as i64;
        let v_weight = vdupq_n_f32((1f64 / (radius as f64 * radius as f64)) as f32);
        for y in start..std::cmp::min(height, end) {
            let mut diffs: int32x4_t = vdupq_n_s32(0);
            let mut summs: int32x4_t = vdupq_n_s32(initial_sum);

            let current_y = ((y as i64) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px =
                        ((std::cmp::max(x, 0) as u32) * CHANNELS_COUNT as u32) as usize;

                    let prepared_px_s32 = vreinterpretq_u32_s32(vmulq_s32_f32(summs, v_weight));
                    let prepared_u16 = vqmovn_u32(prepared_px_s32);
                    let prepared_u8 = vqmovn_u16(vcombine_u16(prepared_u16, prepared_u16));

                    let bytes_offset = current_y + current_px;
                    let dst_ptr = (bytes.slice.as_ptr() as *mut u8).add(bytes_offset);
                    store_u8x8_m4::<CHANNELS_COUNT>(dst_ptr, prepared_u8);

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let d_buf_ptr = buffer.get_unchecked_mut(d_arr_index).as_mut_ptr();
                    let mut d_stored = vld1q_s32(d_buf_ptr);
                    d_stored = vshlq_n_s32::<1>(d_stored);

                    let buf_ptr = buffer.get_unchecked_mut(arr_index).as_mut_ptr();
                    let a_stored = vld1q_s32(buf_ptr);

                    diffs = vaddq_s32(diffs, vsubq_s32(a_stored, d_stored));
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let buf_ptr = buffer.get_unchecked_mut(arr_index).as_mut_ptr();
                    let mut stored = vld1q_s32(buf_ptr);
                    stored = vshlq_n_s32::<1>(stored);
                    diffs = vsubq_s32(diffs, stored);
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide - 1);
                let next_row_px = next_row_x * CHANNELS_COUNT;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut u8;
                let pixel_color = load_u8_s32_fast::<CHANNELS_COUNT>(s_ptr);

                let arr_index = ((x + radius_64) & 1023) as usize;
                let buf_ptr = buffer.get_unchecked_mut(arr_index).as_mut_ptr();

                diffs = vaddq_s32(diffs, pixel_color);
                summs = vaddq_s32(summs, diffs);
                vst1q_s32(buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fast_gaussian_vertical_pass_neon_u8<T, const CHANNELS_COUNT: usize>(
    undefined_slice: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let bytes: &UnsafeSlice<'_, u8> = std::mem::transmute(undefined_slice);
        let mut buffer: [[i32; 4]; 1024] = [[0; 4]; 1024];
        let initial_sum = ((radius * radius) >> 1) as i32;

        let height_wide = height as i64;

        let radius_64 = radius as i64;
        let v_weight = vdupq_n_f32((1f64 / (radius as f64 * radius as f64)) as f32);
        for x in start..std::cmp::min(width, end) {
            let mut diffs: int32x4_t = vdupq_n_s32(0);
            let mut summs: int32x4_t = vdupq_n_s32(initial_sum);

            let start_y = 0 - 2 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;

                if y >= 0 {
                    let current_px = ((std::cmp::max(x, 0)) * CHANNELS_COUNT as u32) as usize;

                    let prepared_px_s32 = vreinterpretq_u32_s32(vmulq_s32_f32(summs, v_weight));
                    let prepared_u16 = vqmovn_u32(prepared_px_s32);
                    let prepared_u8 = vqmovn_u16(vcombine_u16(prepared_u16, prepared_u16));

                    let bytes_offset = current_y + current_px;

                    let dst_ptr = (bytes.slice.as_ptr() as *mut u8).add(bytes_offset);
                    store_u8x8_m4::<CHANNELS_COUNT>(dst_ptr, prepared_u8);

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let d_buf_ptr = buffer.get_unchecked_mut(d_arr_index).as_mut_ptr();
                    let mut d_stored = vld1q_s32(d_buf_ptr);
                    d_stored = vshlq_n_s32::<1>(d_stored);

                    let buf_ptr = buffer.get_unchecked_mut(arr_index).as_mut_ptr();
                    let a_stored = vld1q_s32(buf_ptr);

                    diffs = vaddq_s32(diffs, vsubq_s32(a_stored, d_stored));
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let buf_ptr = buffer.get_unchecked_mut(arr_index).as_mut_ptr();
                    let mut stored = vld1q_s32(buf_ptr);
                    stored = vshlq_n_s32::<1>(stored);
                    diffs = vsubq_s32(diffs, stored);
                }

                let next_row_y =
                    clamp_edge!(edge_mode, y + radius_64, 0, height_wide - 1) * (stride as usize);
                let next_row_x = (x * CHANNELS_COUNT as u32) as usize;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut u8;
                let pixel_color = load_u8_s32_fast::<CHANNELS_COUNT>(s_ptr);

                let arr_index = ((y + radius_64) & 1023) as usize;
                let buf_ptr = buffer.get_unchecked_mut(arr_index).as_mut_ptr();

                diffs = vaddq_s32(diffs, pixel_color);
                summs = vaddq_s32(summs, diffs);

                vst1q_s32(buf_ptr, pixel_color);
            }
        }
    }
}
