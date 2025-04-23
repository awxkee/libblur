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
use crate::{clamp_edge, EdgeMode};

#[repr(C, align(16))]
#[derive(Copy, Clone, Default)]
pub(crate) struct NeonF32x4(pub(crate) [f32; 4]);

pub(crate) fn fgn_vertical_pass_neon_f32<T, const CHANNELS_COUNT: usize>(
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

        let mut full_buffer = Box::new([NeonF32x4::default(); 1024 * 4]);

        let (bf0, rem) = full_buffer.split_at_mut(1024);
        let (bf1, rem) = rem.split_at_mut(1024);
        let (bf2, bf3) = rem.split_at_mut(1024);

        let height_wide = height as i64;

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let f_weight = vdupq_n_f32(weight);

        let mut xx = start as usize;

        while xx + 4 < width.min(end) as usize {
            let mut diffs0 = vdupq_n_f32(0f32);
            let mut diffs1 = vdupq_n_f32(0f32);
            let mut diffs2 = vdupq_n_f32(0f32);
            let mut diffs3 = vdupq_n_f32(0f32);

            let mut ders0 = vdupq_n_f32(0f32);
            let mut ders1 = vdupq_n_f32(0f32);
            let mut ders2 = vdupq_n_f32(0f32);
            let mut ders3 = vdupq_n_f32(0f32);

            let mut summs0 = vdupq_n_f32(0f32);
            let mut summs1 = vdupq_n_f32(0f32);
            let mut summs2 = vdupq_n_f32(0f32);
            let mut summs3 = vdupq_n_f32(0f32);

            let current_px0 = xx * CHANNELS_COUNT;
            let current_px1 = (xx + 1) * CHANNELS_COUNT;
            let current_px2 = (xx + 2) * CHANNELS_COUNT;
            let current_px3 = (xx + 3) * CHANNELS_COUNT;

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * (stride as i64)) as usize;

                    let prepared_px0 = vmulq_f32(summs0, f_weight);
                    let prepared_px1 = vmulq_f32(summs1, f_weight);
                    let prepared_px2 = vmulq_f32(summs2, f_weight);
                    let prepared_px3 = vmulq_f32(summs3, f_weight);

                    let dst_ptr0 = bytes.slice.as_ptr().add(current_y + current_px0) as *mut f32;
                    let dst_ptr1 = bytes.slice.as_ptr().add(current_y + current_px1) as *mut f32;
                    let dst_ptr2 = bytes.slice.as_ptr().add(current_y + current_px2) as *mut f32;
                    let dst_ptr3 = bytes.slice.as_ptr().add(current_y + current_px3) as *mut f32;

                    store_f32::<CHANNELS_COUNT>(dst_ptr0, prepared_px0);
                    store_f32::<CHANNELS_COUNT>(dst_ptr1, prepared_px1);
                    store_f32::<CHANNELS_COUNT>(dst_ptr2, prepared_px2);
                    store_f32::<CHANNELS_COUNT>(dst_ptr3, prepared_px3);

                    let d_a_1 = ((y + radius_64) & 1023) as usize;
                    let d_a_2 = ((y - radius_64) & 1023) as usize;
                    let d_i = (y & 1023) as usize;

                    let sd0 = vld1q_f32(bf0.as_mut_ptr().add(d_i) as *mut f32);
                    let sd1 = vld1q_f32(bf1.as_mut_ptr().add(d_i) as *mut f32);
                    let sd2 = vld1q_f32(bf2.as_mut_ptr().add(d_i) as *mut f32);
                    let sd3 = vld1q_f32(bf3.as_mut_ptr().add(d_i) as *mut f32);

                    let sd_1_0 = vld1q_f32(bf0.as_mut_ptr().add(d_a_1) as *mut f32);
                    let sd_1_1 = vld1q_f32(bf1.as_mut_ptr().add(d_a_1) as *mut f32);
                    let sd_1_2 = vld1q_f32(bf2.as_mut_ptr().add(d_a_1) as *mut f32);
                    let sd_1_3 = vld1q_f32(bf3.as_mut_ptr().add(d_a_1) as *mut f32);

                    let j0 = vsubq_f32(sd0, sd_1_0);
                    let j1 = vsubq_f32(sd1, sd_1_1);
                    let j2 = vsubq_f32(sd2, sd_1_2);
                    let j3 = vsubq_f32(sd3, sd_1_3);

                    let sd_2_0 = vld1q_f32(bf0.as_mut_ptr().add(d_a_2) as *mut f32);
                    let sd_2_1 = vld1q_f32(bf1.as_mut_ptr().add(d_a_2) as *mut f32);
                    let sd_2_2 = vld1q_f32(bf2.as_mut_ptr().add(d_a_2) as *mut f32);
                    let sd_2_3 = vld1q_f32(bf3.as_mut_ptr().add(d_a_2) as *mut f32);

                    let new_diff0 = vfmaq_n_f32(vnegq_f32(sd_2_0), j0, 3f32);
                    let new_diff1 = vfmaq_n_f32(vnegq_f32(sd_2_1), j1, 3f32);
                    let new_diff2 = vfmaq_n_f32(vnegq_f32(sd_2_2), j2, 3f32);
                    let new_diff3 = vfmaq_n_f32(vnegq_f32(sd_2_3), j3, 3f32);

                    diffs0 = vaddq_f32(diffs0, new_diff0);
                    diffs1 = vaddq_f32(diffs1, new_diff1);
                    diffs2 = vaddq_f32(diffs2, new_diff2);
                    diffs3 = vaddq_f32(diffs3, new_diff3);
                } else if y + radius_64 >= 0 {
                    let a_i = (y & 1023) as usize;
                    let a_i_1 = ((y + radius_64) & 1023) as usize;
                    let sd0 = vld1q_f32(bf0.as_mut_ptr().add(a_i) as *mut f32);
                    let sd1 = vld1q_f32(bf1.as_mut_ptr().add(a_i) as *mut f32);
                    let sd2 = vld1q_f32(bf2.as_mut_ptr().add(a_i) as *mut f32);
                    let sd3 = vld1q_f32(bf3.as_mut_ptr().add(a_i) as *mut f32);

                    let sd_1_0 = vld1q_f32(bf0.as_mut_ptr().add(a_i_1) as *mut f32);
                    let sd_1_1 = vld1q_f32(bf1.as_mut_ptr().add(a_i_1) as *mut f32);
                    let sd_1_2 = vld1q_f32(bf2.as_mut_ptr().add(a_i_1) as *mut f32);
                    let sd_1_3 = vld1q_f32(bf3.as_mut_ptr().add(a_i_1) as *mut f32);

                    diffs0 = vfmaq_n_f32(diffs0, vsubq_f32(sd0, sd_1_0), 3f32);
                    diffs1 = vfmaq_n_f32(diffs1, vsubq_f32(sd1, sd_1_1), 3f32);
                    diffs2 = vfmaq_n_f32(diffs2, vsubq_f32(sd2, sd_1_2), 3f32);
                    diffs3 = vfmaq_n_f32(diffs3, vsubq_f32(sd3, sd_1_3), 3f32);
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let sd0 = vld1q_f32(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                    let sd1 = vld1q_f32(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                    let sd2 = vld1q_f32(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                    let sd3 = vld1q_f32(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                    diffs0 = vfmaq_n_f32(diffs0, sd0, -3f32);
                    diffs1 = vfmaq_n_f32(diffs1, sd1, -3f32);
                    diffs2 = vfmaq_n_f32(diffs2, sd2, -3f32);
                    diffs3 = vfmaq_n_f32(diffs3, sd3, -3f32);
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * (stride as usize);

                let s_ptr0 = bytes.slice.as_ptr().add(next_row_y + current_px0) as *mut f32;
                let s_ptr1 = bytes.slice.as_ptr().add(next_row_y + current_px1) as *mut f32;
                let s_ptr2 = bytes.slice.as_ptr().add(next_row_y + current_px2) as *mut f32;
                let s_ptr3 = bytes.slice.as_ptr().add(next_row_y + current_px3) as *mut f32;

                let pixel_color0 = load_f32_fast::<CHANNELS_COUNT>(s_ptr0);
                let pixel_color1 = load_f32_fast::<CHANNELS_COUNT>(s_ptr1);
                let pixel_color2 = load_f32_fast::<CHANNELS_COUNT>(s_ptr2);
                let pixel_color3 = load_f32_fast::<CHANNELS_COUNT>(s_ptr3);

                let a_i = ((y + 2 * radius_64) & 1023) as usize;

                vst1q_f32(bf0.as_mut_ptr().add(a_i) as *mut f32, pixel_color0);
                vst1q_f32(bf1.as_mut_ptr().add(a_i) as *mut f32, pixel_color1);
                vst1q_f32(bf2.as_mut_ptr().add(a_i) as *mut f32, pixel_color2);
                vst1q_f32(bf3.as_mut_ptr().add(a_i) as *mut f32, pixel_color3);

                diffs0 = vaddq_f32(diffs0, pixel_color0);
                diffs1 = vaddq_f32(diffs1, pixel_color1);
                diffs2 = vaddq_f32(diffs2, pixel_color2);
                diffs3 = vaddq_f32(diffs3, pixel_color3);

                ders0 = vaddq_f32(ders0, diffs0);
                ders1 = vaddq_f32(ders1, diffs1);
                ders2 = vaddq_f32(ders2, diffs2);
                ders3 = vaddq_f32(ders3, diffs3);

                summs0 = vaddq_f32(summs0, ders0);
                summs1 = vaddq_f32(summs1, ders1);
                summs2 = vaddq_f32(summs2, ders2);
                summs3 = vaddq_f32(summs3, ders3);
            }

            xx += 4;
        }

        for x in xx..width.min(end) as usize {
            let mut diffs = vdupq_n_f32(0f32);
            let mut ders = vdupq_n_f32(0f32);
            let mut summs = vdupq_n_f32(0f32);

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * (stride as i64)) as usize;
                    let current_px = x * CHANNELS_COUNT;

                    let prepared_px = vmulq_f32(summs, f_weight);
                    let dst_ptr = bytes.slice.as_ptr().add(current_y + current_px) as *mut f32;
                    store_f32::<CHANNELS_COUNT>(dst_ptr, prepared_px);

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let buf_ptr = bf0.as_mut_ptr().add(d_arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);

                    let buf_ptr_1 = bf0.as_mut_ptr().add(d_arr_index_1) as *mut f32;
                    let stored_1 = vld1q_f32(buf_ptr_1);

                    let buf_ptr_2 = bf0.as_mut_ptr().add(d_arr_index_2) as *mut f32;
                    let stored_2 = vld1q_f32(buf_ptr_2);

                    let new_diff =
                        vfmaq_n_f32(vnegq_f32(stored_2), vsubq_f32(stored, stored_1), 3f32);
                    diffs = vaddq_f32(diffs, new_diff);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);

                    let buf_ptr_1 = bf0.as_mut_ptr().add(arr_index_1) as *mut f32;
                    let stored_1 = vld1q_f32(buf_ptr_1);

                    diffs = vfmaq_f32(diffs, vsubq_f32(stored, stored_1), vdupq_n_f32(3f32));
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);
                    diffs = vfmaq_f32(diffs, stored, vdupq_n_f32(-3f32));
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * (stride as usize);
                let next_row_x = x * CHANNELS_COUNT;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut f32;

                let pixel_color = load_f32_fast::<CHANNELS_COUNT>(s_ptr);

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;

                diffs = vaddq_f32(diffs, pixel_color);
                ders = vaddq_f32(ders, diffs);
                summs = vaddq_f32(summs, ders);
                vst1q_f32(buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fgn_horizontal_pass_neon_f32<T, const CHANNELS_COUNT: usize>(
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
        let mut full_buffer = Box::new([NeonF32x4::default(); 1024 * 4]);

        let (bf0, rem) = full_buffer.split_at_mut(1024);
        let (bf1, rem) = rem.split_at_mut(1024);
        let (bf2, bf3) = rem.split_at_mut(1024);

        let bytes: &UnsafeSlice<'_, f32> = std::mem::transmute(undef_bytes);

        let width_wide = width as i64;

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let f_weight = vdupq_n_f32(weight);

        let mut yy = start as usize;

        while yy + 4 < height.min(end) as usize {
            let mut diffs0 = vdupq_n_f32(0f32);
            let mut diffs1 = vdupq_n_f32(0f32);
            let mut diffs2 = vdupq_n_f32(0f32);
            let mut diffs3 = vdupq_n_f32(0f32);

            let mut ders0 = vdupq_n_f32(0f32);
            let mut ders1 = vdupq_n_f32(0f32);
            let mut ders2 = vdupq_n_f32(0f32);
            let mut ders3 = vdupq_n_f32(0f32);

            let mut summs0 = vdupq_n_f32(0f32);
            let mut summs1 = vdupq_n_f32(0f32);
            let mut summs2 = vdupq_n_f32(0f32);
            let mut summs3 = vdupq_n_f32(0f32);

            let start_x = 0 - 3 * radius_64;

            let current_y0 = ((yy as i64) * (stride as i64)) as usize;
            let current_y1 = (((yy + 1) as i64) * (stride as i64)) as usize;
            let current_y2 = (((yy + 2) as i64) * (stride as i64)) as usize;
            let current_y3 = (((yy + 3) as i64) * (stride as i64)) as usize;

            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CHANNELS_COUNT;

                    let prepared_px0 = vmulq_f32(summs0, f_weight);
                    let prepared_px1 = vmulq_f32(summs1, f_weight);
                    let prepared_px2 = vmulq_f32(summs2, f_weight);
                    let prepared_px3 = vmulq_f32(summs3, f_weight);

                    let dst_ptr0 = bytes.slice.as_ptr().add(current_y0 + current_px) as *mut f32;
                    let dst_ptr1 = bytes.slice.as_ptr().add(current_y1 + current_px) as *mut f32;
                    let dst_ptr2 = bytes.slice.as_ptr().add(current_y2 + current_px) as *mut f32;
                    let dst_ptr3 = bytes.slice.as_ptr().add(current_y3 + current_px) as *mut f32;

                    store_f32::<CHANNELS_COUNT>(dst_ptr0, prepared_px0);
                    store_f32::<CHANNELS_COUNT>(dst_ptr1, prepared_px1);
                    store_f32::<CHANNELS_COUNT>(dst_ptr2, prepared_px2);
                    store_f32::<CHANNELS_COUNT>(dst_ptr3, prepared_px3);

                    let d_a_1 = ((x + radius_64) & 1023) as usize;
                    let d_a_2 = ((x - radius_64) & 1023) as usize;
                    let d_i = (x & 1023) as usize;

                    let sd0 = vld1q_f32(bf0.as_mut_ptr().add(d_i) as *mut f32);
                    let sd1 = vld1q_f32(bf1.as_mut_ptr().add(d_i) as *mut f32);
                    let sd2 = vld1q_f32(bf2.as_mut_ptr().add(d_i) as *mut f32);
                    let sd3 = vld1q_f32(bf3.as_mut_ptr().add(d_i) as *mut f32);

                    let sd_1_0 = vld1q_f32(bf0.as_mut_ptr().add(d_a_1) as *mut f32);
                    let sd_1_1 = vld1q_f32(bf1.as_mut_ptr().add(d_a_1) as *mut f32);
                    let sd_1_2 = vld1q_f32(bf2.as_mut_ptr().add(d_a_1) as *mut f32);
                    let sd_1_3 = vld1q_f32(bf3.as_mut_ptr().add(d_a_1) as *mut f32);

                    let j0 = vsubq_f32(sd0, sd_1_0);
                    let j1 = vsubq_f32(sd1, sd_1_1);
                    let j2 = vsubq_f32(sd2, sd_1_2);
                    let j3 = vsubq_f32(sd3, sd_1_3);

                    let sd_2_0 = vld1q_f32(bf0.as_mut_ptr().add(d_a_2) as *mut f32);
                    let sd_2_1 = vld1q_f32(bf1.as_mut_ptr().add(d_a_2) as *mut f32);
                    let sd_2_2 = vld1q_f32(bf2.as_mut_ptr().add(d_a_2) as *mut f32);
                    let sd_2_3 = vld1q_f32(bf3.as_mut_ptr().add(d_a_2) as *mut f32);

                    let new_diff0 = vfmaq_n_f32(vnegq_f32(sd_2_0), j0, 3f32);
                    let new_diff1 = vfmaq_n_f32(vnegq_f32(sd_2_1), j1, 3f32);
                    let new_diff2 = vfmaq_n_f32(vnegq_f32(sd_2_2), j2, 3f32);
                    let new_diff3 = vfmaq_n_f32(vnegq_f32(sd_2_3), j3, 3f32);

                    diffs0 = vaddq_f32(diffs0, new_diff0);
                    diffs1 = vaddq_f32(diffs1, new_diff1);
                    diffs2 = vaddq_f32(diffs2, new_diff2);
                    diffs3 = vaddq_f32(diffs3, new_diff3);
                } else if x + radius_64 >= 0 {
                    let a_i = (x & 1023) as usize;
                    let a_i_1 = ((x + radius_64) & 1023) as usize;
                    let sd0 = vld1q_f32(bf0.as_mut_ptr().add(a_i) as *mut f32);
                    let sd1 = vld1q_f32(bf1.as_mut_ptr().add(a_i) as *mut f32);
                    let sd2 = vld1q_f32(bf2.as_mut_ptr().add(a_i) as *mut f32);
                    let sd3 = vld1q_f32(bf3.as_mut_ptr().add(a_i) as *mut f32);

                    let sd_1_0 = vld1q_f32(bf0.as_mut_ptr().add(a_i_1) as *mut f32);
                    let sd_1_1 = vld1q_f32(bf1.as_mut_ptr().add(a_i_1) as *mut f32);
                    let sd_1_2 = vld1q_f32(bf2.as_mut_ptr().add(a_i_1) as *mut f32);
                    let sd_1_3 = vld1q_f32(bf3.as_mut_ptr().add(a_i_1) as *mut f32);

                    diffs0 = vfmaq_n_f32(diffs0, vsubq_f32(sd0, sd_1_0), 3f32);
                    diffs1 = vfmaq_n_f32(diffs1, vsubq_f32(sd1, sd_1_1), 3f32);
                    diffs2 = vfmaq_n_f32(diffs2, vsubq_f32(sd2, sd_1_2), 3f32);
                    diffs3 = vfmaq_n_f32(diffs3, vsubq_f32(sd3, sd_1_3), 3f32);
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let sd0 = vld1q_f32(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                    let sd1 = vld1q_f32(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                    let sd2 = vld1q_f32(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                    let sd3 = vld1q_f32(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                    diffs0 = vfmaq_n_f32(diffs0, sd0, -3f32);
                    diffs1 = vfmaq_n_f32(diffs1, sd1, -3f32);
                    diffs2 = vfmaq_n_f32(diffs2, sd2, -3f32);
                    diffs3 = vfmaq_n_f32(diffs3, sd3, -3f32);
                }

                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let next_row_px = next_row_x * CHANNELS_COUNT;

                let s_ptr0 = bytes.slice.as_ptr().add(current_y0 + next_row_px) as *mut f32;
                let s_ptr1 = bytes.slice.as_ptr().add(current_y1 + next_row_px) as *mut f32;
                let s_ptr2 = bytes.slice.as_ptr().add(current_y2 + next_row_px) as *mut f32;
                let s_ptr3 = bytes.slice.as_ptr().add(current_y3 + next_row_px) as *mut f32;

                let pixel_color0 = load_f32_fast::<CHANNELS_COUNT>(s_ptr0);
                let pixel_color1 = load_f32_fast::<CHANNELS_COUNT>(s_ptr1);
                let pixel_color2 = load_f32_fast::<CHANNELS_COUNT>(s_ptr2);
                let pixel_color3 = load_f32_fast::<CHANNELS_COUNT>(s_ptr3);

                let a_i = ((x + 2 * radius_64) & 1023) as usize;

                vst1q_f32(bf0.as_mut_ptr().add(a_i) as *mut f32, pixel_color0);
                vst1q_f32(bf1.as_mut_ptr().add(a_i) as *mut f32, pixel_color1);
                vst1q_f32(bf2.as_mut_ptr().add(a_i) as *mut f32, pixel_color2);
                vst1q_f32(bf3.as_mut_ptr().add(a_i) as *mut f32, pixel_color3);

                diffs0 = vaddq_f32(diffs0, pixel_color0);
                diffs1 = vaddq_f32(diffs1, pixel_color1);
                diffs2 = vaddq_f32(diffs2, pixel_color2);
                diffs3 = vaddq_f32(diffs3, pixel_color3);

                ders0 = vaddq_f32(ders0, diffs0);
                ders1 = vaddq_f32(ders1, diffs1);
                ders2 = vaddq_f32(ders2, diffs2);
                ders3 = vaddq_f32(ders3, diffs3);

                summs0 = vaddq_f32(summs0, ders0);
                summs1 = vaddq_f32(summs1, ders1);
                summs2 = vaddq_f32(summs2, ders2);
                summs3 = vaddq_f32(summs3, ders3);
            }

            yy += 4;
        }

        for y in yy..height.min(end) as usize {
            let mut diffs: float32x4_t = vdupq_n_f32(0f32);
            let mut ders: float32x4_t = vdupq_n_f32(0f32);
            let mut summs: float32x4_t = vdupq_n_f32(0f32);

            let start_x = 0 - 3 * radius_64;

            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_y = ((y as i64) * (stride as i64)) as usize;
                    let current_px = x as usize * CHANNELS_COUNT;

                    let prepared_px = vmulq_f32(summs, f_weight);

                    let dst_ptr = bytes.slice.as_ptr().add(current_y + current_px) as *mut f32;
                    store_f32::<CHANNELS_COUNT>(dst_ptr, prepared_px);

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let buf_ptr = bf0.as_mut_ptr().add(d_arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);

                    let buf_ptr_1 = bf0.as_mut_ptr().add(d_arr_index_1) as *mut f32;
                    let stored_1 = vld1q_f32(buf_ptr_1);

                    let buf_ptr_2 = bf0.as_mut_ptr().add(d_arr_index_2) as *mut f32;
                    let stored_2 = vld1q_f32(buf_ptr_2);

                    let new_diff =
                        vfmaq_n_f32(vnegq_f32(stored_2), vsubq_f32(stored, stored_1), 3f32);
                    diffs = vaddq_f32(diffs, new_diff);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);

                    let buf_ptr_1 = bf0.as_mut_ptr().add(arr_index_1) as *mut f32;
                    let stored_1 = vld1q_f32(buf_ptr_1);

                    diffs = vfmaq_f32(diffs, vsubq_f32(stored, stored_1), vdupq_n_f32(3f32));
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);
                    diffs = vfmaq_f32(diffs, stored, vdupq_n_f32(-3f32));
                }

                let next_row_y = y * (stride as usize);
                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let next_row_px = next_row_x * CHANNELS_COUNT;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut f32;
                let pixel_color = load_f32_fast::<CHANNELS_COUNT>(s_ptr);

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;

                diffs = vaddq_f32(diffs, pixel_color);
                ders = vaddq_f32(ders, diffs);
                summs = vaddq_f32(summs, ders);
                vst1q_f32(buf_ptr, pixel_color);
            }
        }
    }
}
