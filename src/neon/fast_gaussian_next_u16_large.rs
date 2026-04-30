// Copyright (c) Radzivon Bartoshyk 04/2026. All rights reserved.
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

use crate::EdgeMode;
use crate::edge_mode::clamp_edge;
use crate::neon::fast_gaussian::NeonI32x4;
use crate::neon::{load_u16_s32_fast, store_u16x4, vmulq_by_3_s32};
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn vadds_i64x2(a_lo: int64x2_t, a_hi: int64x2_t, b: int32x4_t) -> (int64x2_t, int64x2_t) {
    let b_lo = vmovl_s32(vget_low_s32(b));
    let b_hi = vmovl_s32(vget_high_s32(b));
    (vaddq_s64(a_lo, b_lo), vaddq_s64(a_hi, b_hi))
}

pub(crate) fn fgn_vertical_pass_neon_u16_large_r<const CN: usize>(
    bytes: &UnsafeSlice<u16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        fgn_vertical_pass_neon_u16_large_r_impl::<CN>(
            bytes, stride, width, height, radius, start, end, edge_mode,
        );
    }
}

#[target_feature(enable = "neon")]
fn fgn_vertical_pass_neon_u16_large_r_impl<const CN: usize>(
    bytes: &UnsafeSlice<u16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let mut buffer = [NeonI32x4::default(); 1024];

        let height_wide = height as i64;
        let radius_64 = radius as i64;

        let weight = 1.0f64 / ((radius as f64) * (radius as f64) * (radius as f64));

        for x in start..width.min(end) {
            let mut diffs = vdupq_n_s32(0);

            let mut ders_lo = vdupq_n_s64(0);
            let mut ders_hi = vdupq_n_s64(0);

            let mut summs_lo = vdupq_n_s64(0);
            let mut summs_hi = vdupq_n_s64(0);

            let current_px = (x * CN as u32) as usize;

            let start_y = -3 * radius_64;
            for y in start_y..height_wide {
                let current_y = (y * stride as i64) as usize;

                if y >= 0 {
                    let mut s_lo = vcvtq_f64_s64(summs_lo);
                    let mut s_hi = vcvtq_f64_s64(summs_hi);

                    s_lo = vmulq_n_f64(s_lo, weight);
                    s_hi = vmulq_n_f64(s_hi, weight);

                    let q_lo = vcvtaq_s64_f64(s_lo);
                    let q_hi = vcvtaq_s64_f64(s_hi);

                    let prepared_s32 = vcombine_u32(vqmovun_s64(q_lo), vqmovun_s64(q_hi));
                    let prepared_u16 = vqmovn_u32(prepared_s32);

                    let dst_ptr = bytes.get_ptr(current_y + current_px);
                    store_u16x4::<CN>(dst_ptr, prepared_u16);

                    let d_arr_index = (y & 1023) as usize;
                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;

                    let stored = vld1q_s32(buffer.get_unchecked(d_arr_index..).as_ptr().cast());
                    let stored_1 = vld1q_s32(buffer.get_unchecked(d_arr_index_1..).as_ptr().cast());
                    let stored_2 = vld1q_s32(buffer.get_unchecked(d_arr_index_2..).as_ptr().cast());

                    let new_diff = vsubq_s32(vmulq_by_3_s32(vsubq_s32(stored, stored_1)), stored_2);
                    diffs = vaddq_s32(diffs, new_diff);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;

                    let stored = vld1q_s32(buffer.get_unchecked(arr_index..).as_ptr().cast());
                    let stored_1 = vld1q_s32(buffer.get_unchecked(arr_index_1..).as_ptr().cast());

                    let q = vsubq_s32(stored, stored_1);
                    diffs = vmlaq_n_s32(diffs, q, 3);
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let stored = vld1q_s32(buffer.get_unchecked(arr_index).0.as_ptr().cast());
                    diffs = vmlaq_n_s32(diffs, stored, -3);
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * stride as usize;

                let s_ptr = bytes.get_ptr(next_row_y + current_px);
                let pixel_color = load_u16_s32_fast::<CN>(s_ptr);

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer.get_unchecked_mut(arr_index..).as_mut_ptr().cast();

                diffs = vaddq_s32(diffs, pixel_color);

                (ders_lo, ders_hi) = vadds_i64x2(ders_lo, ders_hi, diffs);

                summs_lo = vaddq_s64(summs_lo, ders_lo);
                summs_hi = vaddq_s64(summs_hi, ders_hi);

                vst1q_s32(buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fgn_horizontal_pass_neon_u16_large_r<const CN: usize>(
    bytes: &UnsafeSlice<u16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        fgn_horizontal_pass_neon_u16_large_r_impl::<CN>(
            bytes, stride, width, height, radius, start, end, edge_mode,
        );
    }
}

#[target_feature(enable = "neon")]
fn fgn_horizontal_pass_neon_u16_large_r_impl<const CN: usize>(
    bytes: &UnsafeSlice<u16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let mut buffer = [NeonI32x4::default(); 1024];

        let width_wide = width as i64;
        let radius_64 = radius as i64;

        let weight = 1.0f64 / ((radius as f64) * (radius as f64) * (radius as f64));

        for y in start..height.min(end) {
            let mut diffs = vdupq_n_s32(0);

            let mut ders_lo = vdupq_n_s64(0);
            let mut ders_hi = vdupq_n_s64(0);

            let mut summs_lo = vdupq_n_s64(0);
            let mut summs_hi = vdupq_n_s64(0);

            let current_y = (y as i64 * stride as i64) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let mut s_lo = vcvtq_f64_s64(summs_lo);
                    let mut s_hi = vcvtq_f64_s64(summs_hi);

                    s_lo = vmulq_n_f64(s_lo, weight);
                    s_hi = vmulq_n_f64(s_hi, weight);

                    let q_lo = vcvtaq_s64_f64(s_lo);
                    let q_hi = vcvtaq_s64_f64(s_hi);

                    let prepared_s32 = vcombine_u32(vqmovun_s64(q_lo), vqmovun_s64(q_hi));
                    let prepared_u16 = vqmovn_u32(prepared_s32);

                    let dst_ptr = bytes.get_ptr(current_y + current_px);
                    store_u16x4::<CN>(dst_ptr, prepared_u16);

                    let d_arr_index = (x & 1023) as usize;
                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;

                    let stored = vld1q_s32(buffer.get_unchecked(d_arr_index..).as_ptr().cast());
                    let stored_1 = vld1q_s32(buffer.get_unchecked(d_arr_index_1..).as_ptr().cast());
                    let stored_2 = vld1q_s32(buffer.get_unchecked(d_arr_index_2..).as_ptr().cast());

                    let new_diff = vsubq_s32(vmulq_by_3_s32(vsubq_s32(stored, stored_1)), stored_2);
                    diffs = vaddq_s32(diffs, new_diff);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;

                    let stored = vld1q_s32(buffer.get_unchecked(arr_index..).as_ptr().cast());
                    let stored_1 = vld1q_s32(buffer.get_unchecked(arr_index_1..).as_ptr().cast());

                    diffs = vmlaq_n_s32(diffs, vsubq_s32(stored, stored_1), 3);
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let stored = vld1q_s32(buffer.get_unchecked(arr_index..).as_ptr().cast());
                    diffs = vmlaq_n_s32(diffs, stored, -3);
                }

                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let s_ptr = bytes.get_ptr(current_y + next_row_x * CN);
                let pixel_color = load_u16_s32_fast::<CN>(s_ptr);

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer.get_unchecked_mut(arr_index..).as_mut_ptr().cast();

                diffs = vaddq_s32(diffs, pixel_color);

                (ders_lo, ders_hi) = vadds_i64x2(ders_lo, ders_hi, diffs);
                summs_lo = vaddq_s64(summs_lo, ders_lo);
                summs_hi = vaddq_s64(summs_hi, ders_hi);

                vst1q_s32(buf_ptr, pixel_color);
            }
        }
    }
}
