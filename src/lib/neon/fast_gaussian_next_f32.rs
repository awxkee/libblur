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

use crate::neon::utils::{load_f32_fast, store_f32};
use crate::reflect_index;
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, reflect_101, EdgeMode};

pub fn fast_gaussian_next_vertical_pass_neon_f32<T, const CHANNELS_COUNT: usize>(
    undef_bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let bytes: &UnsafeSlice<'_, f32> = std::mem::transmute(undef_bytes);
        let mut buffer: [[f32; 4]; 1024] = [[0f32; 4]; 1024];

        let height_wide = height as i64;

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let f_weight = vdupq_n_f32(weight);
        for x in start..std::cmp::min(width, end) {
            let mut diffs: float32x4_t = vdupq_n_f32(0f32);
            let mut ders: float32x4_t = vdupq_n_f32(0f32);
            let mut summs: float32x4_t = vdupq_n_f32(0f32);

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;

                if y >= 0 {
                    let current_px = (std::cmp::max(x, 0)) as usize * CHANNELS_COUNT;

                    let prepared_px = vmulq_f32(summs, f_weight);
                    let dst_ptr = bytes.slice.as_ptr().add(current_y + current_px) as *mut f32;
                    store_f32::<CHANNELS_COUNT>(dst_ptr, prepared_px);

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let buf_ptr = buffer.as_mut_ptr().add(d_arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);

                    let buf_ptr_1 = buffer.as_mut_ptr().add(d_arr_index_1) as *mut f32;
                    let stored_1 = vld1q_f32(buf_ptr_1);

                    let buf_ptr_2 = buffer.as_mut_ptr().add(d_arr_index_2) as *mut f32;
                    let stored_2 = vld1q_f32(buf_ptr_2);

                    let new_diff =
                        vsubq_f32(vmulq_n_f32(vsubq_f32(stored, stored_1), 3f32), stored_2);
                    diffs = vaddq_f32(diffs, new_diff);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);

                    let buf_ptr_1 = buffer.as_mut_ptr().add(arr_index_1) as *mut f32;
                    let stored_1 = vld1q_f32(buf_ptr_1);

                    let new_diff = vmulq_n_f32(vsubq_f32(stored, stored_1), 3f32);

                    diffs = vaddq_f32(diffs, new_diff);
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);
                    diffs = vsubq_f32(diffs, vmulq_n_f32(stored, 3f32));
                }

                let next_row_y =
                    clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide - 1)
                        * (stride as usize);
                let next_row_x = x as usize * CHANNELS_COUNT;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut f32;

                let pixel_color = load_f32_fast::<CHANNELS_COUNT>(s_ptr);

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer.as_mut_ptr().add(arr_index) as *mut f32;

                diffs = vaddq_f32(diffs, pixel_color);
                ders = vaddq_f32(ders, diffs);
                summs = vaddq_f32(summs, ders);
                vst1q_f32(buf_ptr, pixel_color);
            }
        }
    }
}

pub fn fast_gaussian_next_horizontal_pass_neon_f32<T, const CHANNELS_COUNT: usize>(
    undef_bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let mut buffer: [[f32; 4]; 1024] = [[0f32; 4]; 1024];
        let bytes: &UnsafeSlice<'_, f32> = std::mem::transmute(undef_bytes);

        let width_wide = width as i64;

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let f_weight = vdupq_n_f32(weight);

        for y in start..std::cmp::min(height, end) {
            let mut diffs: float32x4_t = vdupq_n_f32(0f32);
            let mut ders: float32x4_t = vdupq_n_f32(0f32);
            let mut summs: float32x4_t = vdupq_n_f32(0f32);

            let current_y = ((y as i64) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CHANNELS_COUNT;

                    let prepared_px = vmulq_f32(summs, f_weight);

                    let dst_ptr = bytes.slice.as_ptr().add(current_y + current_px) as *mut f32;
                    store_f32::<CHANNELS_COUNT>(dst_ptr, prepared_px);

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let buf_ptr = buffer.as_mut_ptr().add(d_arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);

                    let buf_ptr_1 = buffer.as_mut_ptr().add(d_arr_index_1) as *mut f32;
                    let stored_1 = vld1q_f32(buf_ptr_1);

                    let buf_ptr_2 = buffer.as_mut_ptr().add(d_arr_index_2) as *mut f32;
                    let stored_2 = vld1q_f32(buf_ptr_2);

                    let new_diff =
                        vsubq_f32(vmulq_n_f32(vsubq_f32(stored, stored_1), 3f32), stored_2);
                    diffs = vaddq_f32(diffs, new_diff);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);

                    let buf_ptr_1 = buffer.as_mut_ptr().add(arr_index_1) as *mut f32;
                    let stored_1 = vld1q_f32(buf_ptr_1);

                    let new_diff = vmulq_n_f32(vsubq_f32(stored, stored_1), 3f32);

                    diffs = vaddq_f32(diffs, new_diff);
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);
                    diffs = vsubq_f32(diffs, vmulq_n_f32(stored, 3f32));
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide - 1);
                let next_row_px = next_row_x * CHANNELS_COUNT;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut f32;
                let pixel_color = load_f32_fast::<CHANNELS_COUNT>(s_ptr);

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer.as_mut_ptr().add(arr_index) as *mut f32;

                diffs = vaddq_f32(diffs, pixel_color);
                ders = vaddq_f32(ders, diffs);
                summs = vaddq_f32(summs, ders);
                vst1q_f32(buf_ptr, pixel_color);
            }
        }
    }
}
