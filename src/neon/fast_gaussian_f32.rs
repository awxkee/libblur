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

use crate::neon::fast_gaussian_next_f32::NeonF32x4;
use crate::neon::{load_f32_fast, store_f32};
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, reflect_index, EdgeMode};

pub(crate) fn fg_vertical_pass_neon_f32<T, const CN: usize>(
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

        let mut bf0 = Box::new([NeonF32x4::default(); 1024]);
        let mut bf1 = Box::new([NeonF32x4::default(); 1024]);
        let mut bf2 = Box::new([NeonF32x4::default(); 1024]);
        let mut bf3 = Box::new([NeonF32x4::default(); 1024]);

        let height_wide = height as i64;

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32));
        let f_weight = vdupq_n_f32(weight);

        let mut xx = start as usize;

        while xx + 4 < width.min(end) as usize {
            let mut diffs0 = vdupq_n_f32(0f32);
            let mut diffs1 = vdupq_n_f32(0f32);
            let mut diffs2 = vdupq_n_f32(0f32);
            let mut diffs3 = vdupq_n_f32(0f32);

            let mut sums0 = vdupq_n_f32(0f32);
            let mut sums1 = vdupq_n_f32(0f32);
            let mut sums2 = vdupq_n_f32(0f32);
            let mut sums3 = vdupq_n_f32(0f32);

            let current_px0 = xx * CN;
            let current_px1 = (xx + 1) * CN;
            let current_px2 = (xx + 2) * CN;
            let current_px3 = (xx + 3) * CN;

            let start_y = 0 - 2 * radius as i64;
            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * (stride as i64)) as usize;

                    let prepared_px0 = vmulq_f32(sums0, f_weight);
                    let prepared_px1 = vmulq_f32(sums1, f_weight);
                    let prepared_px2 = vmulq_f32(sums2, f_weight);
                    let prepared_px3 = vmulq_f32(sums3, f_weight);

                    let dst_ptr0 = bytes.slice.as_ptr().add(current_y + current_px0) as *mut f32;
                    let dst_ptr1 = bytes.slice.as_ptr().add(current_y + current_px1) as *mut f32;
                    let dst_ptr2 = bytes.slice.as_ptr().add(current_y + current_px2) as *mut f32;
                    let dst_ptr3 = bytes.slice.as_ptr().add(current_y + current_px3) as *mut f32;

                    store_f32::<CN>(dst_ptr0, prepared_px0);
                    store_f32::<CN>(dst_ptr1, prepared_px1);
                    store_f32::<CN>(dst_ptr2, prepared_px2);
                    store_f32::<CN>(dst_ptr3, prepared_px3);

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let d_s0 = vld1q_f32(bf0.as_mut_ptr().add(d_arr_index) as *mut f32);
                    let d_s1 = vld1q_f32(bf1.as_mut_ptr().add(d_arr_index) as *mut f32);
                    let d_s2 = vld1q_f32(bf2.as_mut_ptr().add(d_arr_index) as *mut f32);
                    let d_s3 = vld1q_f32(bf3.as_mut_ptr().add(d_arr_index) as *mut f32);

                    let a_s0 = vld1q_f32(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                    let a_s1 = vld1q_f32(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                    let a_s2 = vld1q_f32(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                    let a_s3 = vld1q_f32(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                    diffs0 = vaddq_f32(diffs0, vfmaq_n_f32(a_s0, d_s0, -2f32));
                    diffs1 = vaddq_f32(diffs1, vfmaq_n_f32(a_s1, d_s1, -2f32));
                    diffs2 = vaddq_f32(diffs2, vfmaq_n_f32(a_s2, d_s2, -2f32));
                    diffs3 = vaddq_f32(diffs3, vfmaq_n_f32(a_s3, d_s3, -2f32));
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let s0 = vld1q_f32(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                    let s1 = vld1q_f32(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                    let s2 = vld1q_f32(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                    let s3 = vld1q_f32(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                    diffs0 = vfmaq_n_f32(diffs0, s0, -2f32);
                    diffs1 = vfmaq_n_f32(diffs1, s1, -2f32);
                    diffs2 = vfmaq_n_f32(diffs2, s2, -2f32);
                    diffs3 = vfmaq_n_f32(diffs3, s3, -2f32);
                }

                let next_row_y =
                    clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * (stride as usize);

                let s_ptr0 = bytes.slice.as_ptr().add(next_row_y + current_px0) as *mut f32;
                let s_ptr1 = bytes.slice.as_ptr().add(next_row_y + current_px1) as *mut f32;
                let s_ptr2 = bytes.slice.as_ptr().add(next_row_y + current_px2) as *mut f32;
                let s_ptr3 = bytes.slice.as_ptr().add(next_row_y + current_px3) as *mut f32;

                let px0 = load_f32_fast::<CN>(s_ptr0);
                let px1 = load_f32_fast::<CN>(s_ptr1);
                let px2 = load_f32_fast::<CN>(s_ptr2);
                let px3 = load_f32_fast::<CN>(s_ptr3);

                let arr_index = ((y + radius_64) & 1023) as usize;

                vst1q_f32(bf0.as_mut_ptr().add(arr_index) as *mut f32, px0);
                vst1q_f32(bf1.as_mut_ptr().add(arr_index) as *mut f32, px1);
                vst1q_f32(bf2.as_mut_ptr().add(arr_index) as *mut f32, px2);
                vst1q_f32(bf3.as_mut_ptr().add(arr_index) as *mut f32, px3);

                diffs0 = vaddq_f32(diffs0, px0);
                diffs1 = vaddq_f32(diffs1, px1);
                diffs2 = vaddq_f32(diffs2, px2);
                diffs3 = vaddq_f32(diffs3, px3);

                sums0 = vaddq_f32(sums0, diffs0);
                sums1 = vaddq_f32(sums1, diffs1);
                sums2 = vaddq_f32(sums2, diffs2);
                sums3 = vaddq_f32(sums3, diffs3);
            }

            xx += 4;
        }

        for x in xx..width.min(end) as usize {
            let mut diffs = vdupq_n_f32(0f32);
            let mut summs = vdupq_n_f32(0f32);

            let start_y = 0 - 2 * radius as i64;
            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * (stride as i64)) as usize;
                    let current_px = x * CN;

                    let prepared_px = vmulq_f32(summs, f_weight);

                    let dst_ptr = bytes.slice.as_ptr().add(current_y + current_px) as *mut f32;
                    store_f32::<CN>(dst_ptr, prepared_px);

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let d_buf_ptr = bf0.as_mut_ptr().add(d_arr_index) as *mut f32;
                    let d_stored = vld1q_f32(d_buf_ptr);

                    let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                    let a_stored = vld1q_f32(buf_ptr);

                    diffs = vaddq_f32(diffs, vfmaq_n_f32(a_stored, d_stored, -2f32));
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);
                    diffs = vfmaq_f32(diffs, stored, vdupq_n_f32(-2f32));
                }

                let next_row_y =
                    clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * (stride as usize);
                let next_row_x = x * CN;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut f32;
                let pixel_color = load_f32_fast::<CN>(s_ptr);

                let arr_index = ((y + radius_64) & 1023) as usize;
                let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;

                diffs = vaddq_f32(diffs, pixel_color);
                summs = vaddq_f32(summs, diffs);
                vst1q_f32(buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fg_horizontal_pass_neon_f32<T, const CN: usize>(
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

        let mut bf0 = Box::new([NeonF32x4::default(); 1024]);
        let mut bf1 = Box::new([NeonF32x4::default(); 1024]);
        let mut bf2 = Box::new([NeonF32x4::default(); 1024]);
        let mut bf3 = Box::new([NeonF32x4::default(); 1024]);

        let radius_64 = radius as i64;
        let width_wide = width as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32));
        let f_weight = vdupq_n_f32(weight);

        let mut yy = start as usize;

        while yy + 4 < height.min(end) as usize {
            let mut diffs0 = vdupq_n_f32(0f32);
            let mut diffs1 = vdupq_n_f32(0f32);
            let mut diffs2 = vdupq_n_f32(0f32);
            let mut diffs3 = vdupq_n_f32(0f32);

            let mut sums0 = vdupq_n_f32(0f32);
            let mut sums1 = vdupq_n_f32(0f32);
            let mut sums2 = vdupq_n_f32(0f32);
            let mut sums3 = vdupq_n_f32(0f32);

            let current_y0 = ((yy as i64) * (stride as i64)) as usize;
            let current_y1 = ((yy as i64 + 1) * (stride as i64)) as usize;
            let current_y2 = ((yy as i64 + 2) * (stride as i64)) as usize;
            let current_y3 = ((yy as i64 + 3) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let prepared_px0 = vmulq_f32(sums0, f_weight);
                    let prepared_px1 = vmulq_f32(sums1, f_weight);
                    let prepared_px2 = vmulq_f32(sums2, f_weight);
                    let prepared_px3 = vmulq_f32(sums3, f_weight);

                    let dst_ptr0 = bytes.slice.as_ptr().add(current_y0 + current_px) as *mut f32;
                    let dst_ptr1 = bytes.slice.as_ptr().add(current_y1 + current_px) as *mut f32;
                    let dst_ptr2 = bytes.slice.as_ptr().add(current_y2 + current_px) as *mut f32;
                    let dst_ptr3 = bytes.slice.as_ptr().add(current_y3 + current_px) as *mut f32;

                    store_f32::<CN>(dst_ptr0, prepared_px0);
                    store_f32::<CN>(dst_ptr1, prepared_px1);
                    store_f32::<CN>(dst_ptr2, prepared_px2);
                    store_f32::<CN>(dst_ptr3, prepared_px3);

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let d_s0 = vld1q_f32(bf0.as_mut_ptr().add(d_arr_index) as *mut f32);
                    let d_s1 = vld1q_f32(bf1.as_mut_ptr().add(d_arr_index) as *mut f32);
                    let d_s2 = vld1q_f32(bf2.as_mut_ptr().add(d_arr_index) as *mut f32);
                    let d_s3 = vld1q_f32(bf3.as_mut_ptr().add(d_arr_index) as *mut f32);

                    let a_s0 = vld1q_f32(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                    let a_s1 = vld1q_f32(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                    let a_s2 = vld1q_f32(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                    let a_s3 = vld1q_f32(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                    diffs0 = vaddq_f32(diffs0, vfmaq_n_f32(a_s0, d_s0, -2f32));
                    diffs1 = vaddq_f32(diffs1, vfmaq_n_f32(a_s1, d_s1, -2f32));
                    diffs2 = vaddq_f32(diffs2, vfmaq_n_f32(a_s2, d_s2, -2f32));
                    diffs3 = vaddq_f32(diffs3, vfmaq_n_f32(a_s3, d_s3, -2f32));
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let s0 = vld1q_f32(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                    let s1 = vld1q_f32(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                    let s2 = vld1q_f32(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                    let s3 = vld1q_f32(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                    diffs0 = vfmaq_n_f32(diffs0, s0, -2f32);
                    diffs1 = vfmaq_n_f32(diffs1, s1, -2f32);
                    diffs2 = vfmaq_n_f32(diffs2, s2, -2f32);
                    diffs3 = vfmaq_n_f32(diffs3, s3, -2f32);
                }

                let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr0 = bytes.slice.as_ptr().add(current_y0 + next_row_px) as *mut f32;
                let s_ptr1 = bytes.slice.as_ptr().add(current_y1 + next_row_px) as *mut f32;
                let s_ptr2 = bytes.slice.as_ptr().add(current_y2 + next_row_px) as *mut f32;
                let s_ptr3 = bytes.slice.as_ptr().add(current_y3 + next_row_px) as *mut f32;

                let px0 = load_f32_fast::<CN>(s_ptr0);
                let px1 = load_f32_fast::<CN>(s_ptr1);
                let px2 = load_f32_fast::<CN>(s_ptr2);
                let px3 = load_f32_fast::<CN>(s_ptr3);

                let arr_index = ((x + radius_64) & 1023) as usize;

                vst1q_f32(bf0.as_mut_ptr().add(arr_index) as *mut f32, px0);
                vst1q_f32(bf1.as_mut_ptr().add(arr_index) as *mut f32, px1);
                vst1q_f32(bf2.as_mut_ptr().add(arr_index) as *mut f32, px2);
                vst1q_f32(bf3.as_mut_ptr().add(arr_index) as *mut f32, px3);

                diffs0 = vaddq_f32(diffs0, px0);
                diffs1 = vaddq_f32(diffs1, px1);
                diffs2 = vaddq_f32(diffs2, px2);
                diffs3 = vaddq_f32(diffs3, px3);

                sums0 = vaddq_f32(sums0, diffs0);
                sums1 = vaddq_f32(sums1, diffs1);
                sums2 = vaddq_f32(sums2, diffs2);
                sums3 = vaddq_f32(sums3, diffs3);
            }

            yy += 4;
        }

        for y in yy..height.min(end) as usize {
            let mut diffs = vdupq_n_f32(0f32);
            let mut summs = vdupq_n_f32(0f32);

            let current_y = ((y as i64) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let prepared_px = vmulq_f32(summs, f_weight);

                    let dst_ptr = bytes.slice.as_ptr().add(current_y + current_px) as *mut f32;
                    store_f32::<CN>(dst_ptr, prepared_px);

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let d_buf_ptr = bf0.as_mut_ptr().add(d_arr_index) as *mut f32;
                    let d_stored = vld1q_f32(d_buf_ptr);

                    let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                    let a_stored = vld1q_f32(buf_ptr);

                    diffs = vaddq_f32(diffs, vfmaq_n_f32(a_stored, d_stored, -2f32));
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = vld1q_f32(buf_ptr);
                    diffs = vfmaq_f32(diffs, stored, vdupq_n_f32(-2f32));
                }

                let next_row_y = y * (stride as usize);
                let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut f32;
                let pixel_color = load_f32_fast::<CN>(s_ptr);

                let arr_index = ((x + radius_64) & 1023) as usize;
                let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;

                diffs = vaddq_f32(diffs, pixel_color);
                summs = vaddq_f32(summs, diffs);
                vst1q_f32(buf_ptr, pixel_color);
            }
        }
    }
}
