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

use crate::neon::fast_gaussian::NeonI32x4;
use crate::neon::{load_u16_s32_fast, store_u16_s32_x5, store_u16x4, vmulq_s32_f32};
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, reflect_index, EdgeMode};
use std::arch::aarch64::*;

pub(crate) fn fg_horizontal_pass_neon_u16<const CN: usize>(
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
        let mut buffer0 = Box::new([NeonI32x4::default(); 1024]);
        let mut buffer1 = Box::new([NeonI32x4::default(); 1024]);
        let mut buffer2 = Box::new([NeonI32x4::default(); 1024]);
        let mut buffer3 = Box::new([NeonI32x4::default(); 1024]);
        let mut buffer4 = Box::new([NeonI32x4::default(); 1024]);

        let initial_sum = ((radius * radius) >> 1) as i32;

        let radius_64 = radius as i64;
        let width_wide = width as i64;
        let v_weight = vdupq_n_f32((1f64 / (radius as f64 * radius as f64)) as f32);

        let mut yy = start;

        while yy + 5 < height.min(end) {
            let mut diffs0 = vdupq_n_s32(0);
            let mut diffs1 = vdupq_n_s32(0);
            let mut diffs2 = vdupq_n_s32(0);
            let mut diffs3 = vdupq_n_s32(0);
            let mut diffs4 = vdupq_n_s32(0);

            let mut summs0 = vdupq_n_s32(initial_sum);
            let mut summs1 = vdupq_n_s32(initial_sum);
            let mut summs2 = vdupq_n_s32(initial_sum);
            let mut summs3 = vdupq_n_s32(initial_sum);
            let mut summs4 = vdupq_n_s32(initial_sum);

            let current_y0 = ((yy as i64) * (stride as i64)) as usize;
            let current_y1 = ((yy as i64 + 1) * (stride as i64)) as usize;
            let current_y2 = ((yy as i64 + 2) * (stride as i64)) as usize;
            let current_y3 = ((yy as i64 + 3) * (stride as i64)) as usize;
            let current_y4 = ((yy as i64 + 4) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let c0 = vcvtq_f32_s32(summs0);
                    let c1 = vcvtq_f32_s32(summs1);
                    let c2 = vcvtq_f32_s32(summs2);
                    let c3 = vcvtq_f32_s32(summs3);
                    let c4 = vcvtq_f32_s32(summs4);

                    let p0 = vmulq_f32(c0, v_weight);
                    let p1 = vmulq_f32(c1, v_weight);
                    let p2 = vmulq_f32(c2, v_weight);
                    let p3 = vmulq_f32(c3, v_weight);
                    let p4 = vmulq_f32(c4, v_weight);

                    let prepared_px0 = vcvtaq_s32_f32(p0);
                    let prepared_px1 = vcvtaq_s32_f32(p1);
                    let prepared_px2 = vcvtaq_s32_f32(p2);
                    let prepared_px3 = vcvtaq_s32_f32(p3);
                    let prepared_px4 = vcvtaq_s32_f32(p4);

                    let dst_ptr0 = (bytes.slice.as_ptr() as *mut u16).add(current_y0 + current_px);
                    let dst_ptr1 = (bytes.slice.as_ptr() as *mut u16).add(current_y1 + current_px);
                    let dst_ptr2 = (bytes.slice.as_ptr() as *mut u16).add(current_y2 + current_px);
                    let dst_ptr3 = (bytes.slice.as_ptr() as *mut u16).add(current_y3 + current_px);
                    let dst_ptr4 = (bytes.slice.as_ptr() as *mut u16).add(current_y4 + current_px);

                    store_u16_s32_x5::<CN>(
                        (dst_ptr0, dst_ptr1, dst_ptr2, dst_ptr3, dst_ptr4),
                        int32x4x4_t(prepared_px0, prepared_px1, prepared_px2, prepared_px3),
                        prepared_px4,
                    );

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let d_stored0 =
                        vld1q_s32(buffer0.get_unchecked_mut(d_arr_index).0.as_mut_ptr());
                    let d_stored1 =
                        vld1q_s32(buffer1.get_unchecked_mut(d_arr_index).0.as_mut_ptr());
                    let d_stored2 =
                        vld1q_s32(buffer2.get_unchecked_mut(d_arr_index).0.as_mut_ptr());
                    let d_stored3 =
                        vld1q_s32(buffer3.get_unchecked_mut(d_arr_index).0.as_mut_ptr());
                    let d_stored4 =
                        vld1q_s32(buffer4.get_unchecked_mut(d_arr_index).0.as_mut_ptr());

                    let a_stored0 = vld1q_s32(buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let a_stored1 = vld1q_s32(buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let a_stored2 = vld1q_s32(buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let a_stored3 = vld1q_s32(buffer3.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let a_stored4 = vld1q_s32(buffer4.get_unchecked_mut(arr_index).0.as_mut_ptr());

                    diffs0 = vaddq_s32(
                        diffs0,
                        vsubq_s32(a_stored0, vaddq_s32(d_stored0, d_stored0)),
                    );
                    diffs1 = vaddq_s32(
                        diffs1,
                        vsubq_s32(a_stored1, vaddq_s32(d_stored1, d_stored1)),
                    );
                    diffs2 = vaddq_s32(
                        diffs2,
                        vsubq_s32(a_stored2, vaddq_s32(d_stored2, d_stored2)),
                    );
                    diffs3 = vaddq_s32(
                        diffs3,
                        vsubq_s32(a_stored3, vaddq_s32(d_stored3, d_stored3)),
                    );
                    diffs4 = vaddq_s32(
                        diffs4,
                        vsubq_s32(a_stored4, vaddq_s32(d_stored4, d_stored4)),
                    );
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let mut stored0 =
                        vld1q_s32(buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let mut stored1 =
                        vld1q_s32(buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let mut stored2 =
                        vld1q_s32(buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let mut stored3 =
                        vld1q_s32(buffer3.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let mut stored4 =
                        vld1q_s32(buffer4.get_unchecked_mut(arr_index).0.as_mut_ptr());

                    stored0 = vshlq_n_s32::<1>(stored0);
                    stored1 = vshlq_n_s32::<1>(stored1);
                    stored2 = vshlq_n_s32::<1>(stored2);
                    stored3 = vshlq_n_s32::<1>(stored3);
                    stored4 = vshlq_n_s32::<1>(stored4);

                    diffs0 = vsubq_s32(diffs0, stored0);
                    diffs1 = vsubq_s32(diffs1, stored1);
                    diffs2 = vsubq_s32(diffs2, stored2);
                    diffs3 = vsubq_s32(diffs3, stored3);
                    diffs4 = vsubq_s32(diffs4, stored4);
                }

                let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr0 = bytes.slice.as_ptr().add(current_y0 + next_row_px) as *mut u16;
                let s_ptr1 = bytes.slice.as_ptr().add(current_y1 + next_row_px) as *mut u16;
                let s_ptr2 = bytes.slice.as_ptr().add(current_y2 + next_row_px) as *mut u16;
                let s_ptr3 = bytes.slice.as_ptr().add(current_y3 + next_row_px) as *mut u16;
                let s_ptr4 = bytes.slice.as_ptr().add(current_y4 + next_row_px) as *mut u16;

                let pixel_color0 = load_u16_s32_fast::<CN>(s_ptr0);
                let pixel_color1 = load_u16_s32_fast::<CN>(s_ptr1);
                let pixel_color2 = load_u16_s32_fast::<CN>(s_ptr2);
                let pixel_color3 = load_u16_s32_fast::<CN>(s_ptr3);
                let pixel_color4 = load_u16_s32_fast::<CN>(s_ptr4);

                let arr_index = ((x + radius_64) & 1023) as usize;

                vst1q_s32(
                    buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color0,
                );
                vst1q_s32(
                    buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color1,
                );
                vst1q_s32(
                    buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color2,
                );
                vst1q_s32(
                    buffer3.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color3,
                );
                vst1q_s32(
                    buffer4.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color4,
                );

                diffs0 = vaddq_s32(diffs0, pixel_color0);
                diffs1 = vaddq_s32(diffs1, pixel_color1);
                diffs2 = vaddq_s32(diffs2, pixel_color2);
                diffs3 = vaddq_s32(diffs3, pixel_color3);
                diffs4 = vaddq_s32(diffs4, pixel_color4);

                summs0 = vaddq_s32(summs0, diffs0);
                summs1 = vaddq_s32(summs1, diffs1);
                summs2 = vaddq_s32(summs2, diffs2);
                summs3 = vaddq_s32(summs3, diffs3);
                summs4 = vaddq_s32(summs4, diffs4);
            }

            yy += 5;
        }

        for y in yy..height.min(end) {
            let mut diffs: int32x4_t = vdupq_n_s32(0);
            let mut summs: int32x4_t = vdupq_n_s32(initial_sum);

            let current_y = ((y as i64) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = (x as u32 * CN as u32) as usize;

                    let prepared_px_s32 = vreinterpretq_u32_s32(vmulq_s32_f32(summs, v_weight));
                    let prepared_u16 = vqmovn_u32(prepared_px_s32);

                    let bytes_offset = current_y + current_px;
                    let dst_ptr = (bytes.slice.as_ptr() as *mut u16).add(bytes_offset);
                    store_u16x4::<CN>(dst_ptr, prepared_u16);

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let d_buf_ptr = buffer0.get_unchecked_mut(d_arr_index).0.as_mut_ptr();
                    let d_stored = vld1q_s32(d_buf_ptr);

                    let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                    let a_stored = vld1q_s32(buf_ptr);

                    diffs = vaddq_s32(diffs, vmlaq_s32(a_stored, d_stored, vdupq_n_s32(-2)));
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                    let mut stored = vld1q_s32(buf_ptr);
                    stored = vshlq_n_s32::<1>(stored);
                    diffs = vsubq_s32(diffs, stored);
                }

                let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr = bytes.slice.as_ptr().add(current_y + next_row_px) as *mut u16;
                let pixel_color = load_u16_s32_fast::<CN>(s_ptr);

                let arr_index = ((x + radius_64) & 1023) as usize;
                let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();

                diffs = vaddq_s32(diffs, pixel_color);
                summs = vaddq_s32(summs, diffs);
                vst1q_s32(buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fg_vertical_pass_neon_u16<const CN: usize>(
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
        let mut buffer0 = Box::new([NeonI32x4::default(); 1024]);
        let mut buffer1 = Box::new([NeonI32x4::default(); 1024]);
        let mut buffer2 = Box::new([NeonI32x4::default(); 1024]);
        let mut buffer3 = Box::new([NeonI32x4::default(); 1024]);
        let mut buffer4 = Box::new([NeonI32x4::default(); 1024]);

        let initial_sum = ((radius * radius) >> 1) as i32;

        let height_wide = height as i64;

        let radius_64 = radius as i64;
        let v_weight = vdupq_n_f32((1f64 / (radius as f64 * radius as f64)) as f32);

        let mut xx = start;

        while xx + 5 < width.min(end) {
            let mut diffs0 = vdupq_n_s32(0);
            let mut diffs1 = vdupq_n_s32(0);
            let mut diffs2 = vdupq_n_s32(0);
            let mut diffs3 = vdupq_n_s32(0);
            let mut diffs4 = vdupq_n_s32(0);

            let mut summs0 = vdupq_n_s32(initial_sum);
            let mut summs1 = vdupq_n_s32(initial_sum);
            let mut summs2 = vdupq_n_s32(initial_sum);
            let mut summs3 = vdupq_n_s32(initial_sum);
            let mut summs4 = vdupq_n_s32(initial_sum);

            let start_y = 0 - 2 * radius as i64;

            let current_px0 = (xx * CN as u32) as usize;
            let current_px1 = ((xx + 1) * CN as u32) as usize;
            let current_px2 = ((xx + 2) * CN as u32) as usize;
            let current_px3 = ((xx + 3) * CN as u32) as usize;
            let current_px4 = ((xx + 4) * CN as u32) as usize;

            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * (stride as i64)) as usize;

                    let c0 = vcvtq_f32_s32(summs0);
                    let c1 = vcvtq_f32_s32(summs1);
                    let c2 = vcvtq_f32_s32(summs2);
                    let c3 = vcvtq_f32_s32(summs3);
                    let c4 = vcvtq_f32_s32(summs4);

                    let p0 = vmulq_f32(c0, v_weight);
                    let p1 = vmulq_f32(c1, v_weight);
                    let p2 = vmulq_f32(c2, v_weight);
                    let p3 = vmulq_f32(c3, v_weight);
                    let p4 = vmulq_f32(c4, v_weight);

                    let prepared_px0 = vcvtaq_s32_f32(p0);
                    let prepared_px1 = vcvtaq_s32_f32(p1);
                    let prepared_px2 = vcvtaq_s32_f32(p2);
                    let prepared_px3 = vcvtaq_s32_f32(p3);
                    let prepared_px4 = vcvtaq_s32_f32(p4);

                    let dst_ptr0 = (bytes.slice.as_ptr() as *mut u16).add(current_y + current_px0);
                    let dst_ptr1 = (bytes.slice.as_ptr() as *mut u16).add(current_y + current_px1);
                    let dst_ptr2 = (bytes.slice.as_ptr() as *mut u16).add(current_y + current_px2);
                    let dst_ptr3 = (bytes.slice.as_ptr() as *mut u16).add(current_y + current_px3);
                    let dst_ptr4 = (bytes.slice.as_ptr() as *mut u16).add(current_y + current_px4);

                    store_u16_s32_x5::<CN>(
                        (dst_ptr0, dst_ptr1, dst_ptr2, dst_ptr3, dst_ptr4),
                        int32x4x4_t(prepared_px0, prepared_px1, prepared_px2, prepared_px3),
                        prepared_px4,
                    );

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let d_stored0 =
                        vld1q_s32(buffer0.get_unchecked_mut(d_arr_index).0.as_mut_ptr());
                    let d_stored1 =
                        vld1q_s32(buffer1.get_unchecked_mut(d_arr_index).0.as_mut_ptr());
                    let d_stored2 =
                        vld1q_s32(buffer2.get_unchecked_mut(d_arr_index).0.as_mut_ptr());
                    let d_stored3 =
                        vld1q_s32(buffer3.get_unchecked_mut(d_arr_index).0.as_mut_ptr());
                    let d_stored4 =
                        vld1q_s32(buffer4.get_unchecked_mut(d_arr_index).0.as_mut_ptr());

                    let a_stored0 = vld1q_s32(buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let a_stored1 = vld1q_s32(buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let a_stored2 = vld1q_s32(buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let a_stored3 = vld1q_s32(buffer3.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let a_stored4 = vld1q_s32(buffer4.get_unchecked_mut(arr_index).0.as_mut_ptr());

                    diffs0 = vaddq_s32(
                        diffs0,
                        vsubq_s32(a_stored0, vaddq_s32(d_stored0, d_stored0)),
                    );
                    diffs1 = vaddq_s32(
                        diffs1,
                        vsubq_s32(a_stored1, vaddq_s32(d_stored1, d_stored1)),
                    );
                    diffs2 = vaddq_s32(
                        diffs2,
                        vsubq_s32(a_stored2, vaddq_s32(d_stored2, d_stored2)),
                    );
                    diffs3 = vaddq_s32(
                        diffs3,
                        vsubq_s32(a_stored3, vaddq_s32(d_stored3, d_stored3)),
                    );
                    diffs4 = vaddq_s32(
                        diffs4,
                        vsubq_s32(a_stored4, vaddq_s32(d_stored4, d_stored4)),
                    );
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;

                    let mut stored0 =
                        vld1q_s32(buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let mut stored1 =
                        vld1q_s32(buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let mut stored2 =
                        vld1q_s32(buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let mut stored3 =
                        vld1q_s32(buffer3.get_unchecked_mut(arr_index).0.as_mut_ptr());
                    let mut stored4 =
                        vld1q_s32(buffer4.get_unchecked_mut(arr_index).0.as_mut_ptr());

                    stored0 = vshlq_n_s32::<1>(stored0);
                    stored1 = vshlq_n_s32::<1>(stored1);
                    stored2 = vshlq_n_s32::<1>(stored2);
                    stored3 = vshlq_n_s32::<1>(stored3);
                    stored4 = vshlq_n_s32::<1>(stored4);

                    diffs0 = vsubq_s32(diffs0, stored0);
                    diffs1 = vsubq_s32(diffs1, stored1);
                    diffs2 = vsubq_s32(diffs2, stored2);
                    diffs3 = vsubq_s32(diffs3, stored3);
                    diffs4 = vsubq_s32(diffs4, stored4);
                }

                let next_row_y =
                    clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * (stride as usize);

                let s_ptr0 = bytes.slice.as_ptr().add(next_row_y + current_px0) as *mut u16;
                let s_ptr1 = bytes.slice.as_ptr().add(next_row_y + current_px1) as *mut u16;
                let s_ptr2 = bytes.slice.as_ptr().add(next_row_y + current_px2) as *mut u16;
                let s_ptr3 = bytes.slice.as_ptr().add(next_row_y + current_px3) as *mut u16;
                let s_ptr4 = bytes.slice.as_ptr().add(next_row_y + current_px4) as *mut u16;

                let pixel_color0 = load_u16_s32_fast::<CN>(s_ptr0);
                let pixel_color1 = load_u16_s32_fast::<CN>(s_ptr1);
                let pixel_color2 = load_u16_s32_fast::<CN>(s_ptr2);
                let pixel_color3 = load_u16_s32_fast::<CN>(s_ptr3);
                let pixel_color4 = load_u16_s32_fast::<CN>(s_ptr4);

                let arr_index = ((y + radius_64) & 1023) as usize;

                diffs0 = vaddq_s32(diffs0, pixel_color0);
                diffs1 = vaddq_s32(diffs1, pixel_color1);
                diffs2 = vaddq_s32(diffs2, pixel_color2);
                diffs3 = vaddq_s32(diffs3, pixel_color3);
                diffs4 = vaddq_s32(diffs4, pixel_color4);

                vst1q_s32(
                    buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color0,
                );
                vst1q_s32(
                    buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color1,
                );
                vst1q_s32(
                    buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color2,
                );
                vst1q_s32(
                    buffer3.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color3,
                );
                vst1q_s32(
                    buffer4.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color4,
                );

                summs0 = vaddq_s32(summs0, diffs0);
                summs1 = vaddq_s32(summs1, diffs1);
                summs2 = vaddq_s32(summs2, diffs2);
                summs3 = vaddq_s32(summs3, diffs3);
                summs4 = vaddq_s32(summs4, diffs4);
            }

            xx += 5;
        }

        for x in xx..width.min(end) {
            let mut diffs: int32x4_t = vdupq_n_s32(0);
            let mut summs: int32x4_t = vdupq_n_s32(initial_sum);

            let start_y = 0 - 2 * radius as i64;
            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * (stride as i64)) as usize;

                    let current_px = (x * CN as u32) as usize;

                    let prepared_px_s32 = vreinterpretq_u32_s32(vmulq_s32_f32(summs, v_weight));
                    let prepared_u16 = vqmovn_u32(prepared_px_s32);

                    let bytes_offset = current_y + current_px;

                    let dst_ptr = (bytes.slice.as_ptr() as *mut u16).add(bytes_offset);
                    store_u16x4::<CN>(dst_ptr, prepared_u16);

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let d_buf_ptr = buffer0.get_unchecked_mut(d_arr_index).0.as_mut_ptr();
                    let d_stored = vld1q_s32(d_buf_ptr);

                    let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                    let a_stored = vld1q_s32(buf_ptr);

                    diffs = vaddq_s32(diffs, vmlaq_s32(a_stored, d_stored, vdupq_n_s32(-2)));
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                    let stored = vld1q_s32(buf_ptr);
                    diffs = vsubq_s32(diffs, vaddq_s32(stored, stored));
                }

                let next_row_y =
                    clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * (stride as usize);
                let next_row_x = (x * CN as u32) as usize;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut u16;
                let pixel_color = load_u16_s32_fast::<CN>(s_ptr);

                let arr_index = ((y + radius_64) & 1023) as usize;
                let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();

                diffs = vaddq_s32(diffs, pixel_color);
                summs = vaddq_s32(summs, diffs);

                vst1q_s32(buf_ptr, pixel_color);
            }
        }
    }
}
