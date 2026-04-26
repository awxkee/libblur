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
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;

#[repr(C, align(16))]
#[derive(Copy, Clone, Default)]
pub(crate) struct SveI32x4(pub(crate) [i32; 4]);

pub(crate) fn fg_horizontal_pass_neon_u8_sve<const CN: usize>(
    slice: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        fg_horizontal_pass_neon_sve::<CN>(
            slice, stride, width, height, radius, start, end, edge_mode,
        );
    }
}

#[target_feature(enable = "sve2", enable = "sve")]
fn fg_horizontal_pass_neon_sve<const CN: usize>(
    bytes: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let mut full_buffer = [SveI32x4::default(); 1024 * 5];

        let (buffer0, rem) = full_buffer.split_at_mut(1024);
        let (buffer1, rem) = rem.split_at_mut(1024);
        let (buffer2, rem) = rem.split_at_mut(1024);
        let (buffer3, buffer4) = rem.split_at_mut(1024);

        let initial_sum = ((radius * radius) >> 1) as i32;

        let radius_64 = radius as i64;
        let width_wide = width as i64;

        let pv_cn = svwhilelt_b32_u32(0u32, CN as u32);

        const Q: f64 = ((1i64 << 31i64) - 1) as f64;
        let weight = Q / (radius as f64 * radius as f64);
        let v_weight = weight as i32;

        let mut yy = start;

        while yy + 5 <= height.min(end) {
            let mut diffs0 = svdup_n_s32(0);
            let mut diffs1 = svdup_n_s32(0);
            let mut diffs2 = svdup_n_s32(0);
            let mut diffs3 = svdup_n_s32(0);
            let mut diffs4 = svdup_n_s32(0);

            let mut summs0 = svdup_n_s32(initial_sum);
            let mut summs1 = svdup_n_s32(initial_sum);
            let mut summs2 = svdup_n_s32(initial_sum);
            let mut summs3 = svdup_n_s32(initial_sum);
            let mut summs4 = svdup_n_s32(initial_sum);

            let current_y0 = ((yy as i64) * (stride as i64)) as usize;
            let current_y1 = ((yy as i64 + 1) * (stride as i64)) as usize;
            let current_y2 = ((yy as i64 + 2) * (stride as i64)) as usize;
            let current_y3 = ((yy as i64 + 3) * (stride as i64)) as usize;
            let current_y4 = ((yy as i64 + 4) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let prepared_px0 = svqrdmulh_n_s32(summs0, v_weight);
                    let prepared_px1 = svqrdmulh_n_s32(summs1, v_weight);
                    let prepared_px2 = svqrdmulh_n_s32(summs2, v_weight);
                    let prepared_px3 = svqrdmulh_n_s32(summs3, v_weight);
                    let prepared_px4 = svqrdmulh_n_s32(summs4, v_weight);

                    let dst_ptr0 = bytes.get_ptr(current_y0 + current_px);
                    let dst_ptr1 = bytes.get_ptr(current_y1 + current_px);
                    let dst_ptr2 = bytes.get_ptr(current_y2 + current_px);
                    let dst_ptr3 = bytes.get_ptr(current_y3 + current_px);
                    let dst_ptr4 = bytes.get_ptr(current_y4 + current_px);

                    svst1b_u32(pv_cn, dst_ptr0, svreinterpret_u32_s32(prepared_px0));
                    svst1b_u32(pv_cn, dst_ptr1, svreinterpret_u32_s32(prepared_px1));
                    svst1b_u32(pv_cn, dst_ptr2, svreinterpret_u32_s32(prepared_px2));
                    svst1b_u32(pv_cn, dst_ptr3, svreinterpret_u32_s32(prepared_px3));
                    svst1b_u32(pv_cn, dst_ptr4, svreinterpret_u32_s32(prepared_px4));

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let d_stored0 = svld1_s32(pv_cn, buffer0.get_unchecked(d_arr_index).0.as_ptr());
                    let d_stored1 = svld1_s32(pv_cn, buffer1.get_unchecked(d_arr_index).0.as_ptr());
                    let d_stored2 = svld1_s32(pv_cn, buffer2.get_unchecked(d_arr_index).0.as_ptr());
                    let d_stored3 = svld1_s32(pv_cn, buffer3.get_unchecked(d_arr_index).0.as_ptr());
                    let d_stored4 = svld1_s32(pv_cn, buffer4.get_unchecked(d_arr_index).0.as_ptr());

                    let a_stored0 = svld1_s32(pv_cn, buffer0.get_unchecked(arr_index).0.as_ptr());
                    let a_stored1 = svld1_s32(pv_cn, buffer1.get_unchecked(arr_index).0.as_ptr());
                    let a_stored2 = svld1_s32(pv_cn, buffer2.get_unchecked(arr_index).0.as_ptr());
                    let a_stored3 = svld1_s32(pv_cn, buffer3.get_unchecked(arr_index).0.as_ptr());
                    let a_stored4 = svld1_s32(pv_cn, buffer4.get_unchecked(arr_index).0.as_ptr());

                    let d2_0 = svadd_s32_x(pv_cn, d_stored0, d_stored0);
                    let d2_1 = svadd_s32_x(pv_cn, d_stored1, d_stored1);
                    let d2_2 = svadd_s32_x(pv_cn, d_stored2, d_stored2);
                    let d2_3 = svadd_s32_x(pv_cn, d_stored3, d_stored3);
                    let d2_4 = svadd_s32_x(pv_cn, d_stored4, d_stored4);

                    let sub0 = svsub_s32_x(pv_cn, a_stored0, d2_0);
                    let sub1 = svsub_s32_x(pv_cn, a_stored1, d2_1);
                    let sub2 = svsub_s32_x(pv_cn, a_stored2, d2_2);
                    let sub3 = svsub_s32_x(pv_cn, a_stored3, d2_3);
                    let sub4 = svsub_s32_x(pv_cn, a_stored4, d2_4);

                    diffs0 = svadd_s32_x(pv_cn, diffs0, sub0);
                    diffs1 = svadd_s32_x(pv_cn, diffs1, sub1);
                    diffs2 = svadd_s32_x(pv_cn, diffs2, sub2);
                    diffs3 = svadd_s32_x(pv_cn, diffs3, sub3);
                    diffs4 = svadd_s32_x(pv_cn, diffs4, sub4);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let mut stored0 = svld1_s32(pv_cn, buffer0.get_unchecked(arr_index).0.as_ptr());
                    let mut stored1 = svld1_s32(pv_cn, buffer1.get_unchecked(arr_index).0.as_ptr());
                    let mut stored2 = svld1_s32(pv_cn, buffer2.get_unchecked(arr_index).0.as_ptr());
                    let mut stored3 = svld1_s32(pv_cn, buffer3.get_unchecked(arr_index).0.as_ptr());
                    let mut stored4 = svld1_s32(pv_cn, buffer4.get_unchecked(arr_index).0.as_ptr());

                    stored0 = svlsl_n_s32_x(pv_cn, stored0, 1);
                    stored1 = svlsl_n_s32_x(pv_cn, stored1, 1);
                    stored2 = svlsl_n_s32_x(pv_cn, stored2, 1);
                    stored3 = svlsl_n_s32_x(pv_cn, stored3, 1);
                    stored4 = svlsl_n_s32_x(pv_cn, stored4, 1);

                    diffs0 = svsub_s32_x(pv_cn, diffs0, stored0);
                    diffs1 = svsub_s32_x(pv_cn, diffs1, stored1);
                    diffs2 = svsub_s32_x(pv_cn, diffs2, stored2);
                    diffs3 = svsub_s32_x(pv_cn, diffs3, stored3);
                    diffs4 = svsub_s32_x(pv_cn, diffs4, stored4);
                }

                let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr0 = bytes.get_ptr(current_y0 + next_row_px);
                let s_ptr1 = bytes.get_ptr(current_y1 + next_row_px);
                let s_ptr2 = bytes.get_ptr(current_y2 + next_row_px);
                let s_ptr3 = bytes.get_ptr(current_y3 + next_row_px);
                let s_ptr4 = bytes.get_ptr(current_y4 + next_row_px);

                let pixel_color0 = svld1ub_s32(pv_cn, s_ptr0);
                let pixel_color1 = svld1ub_s32(pv_cn, s_ptr1);
                let pixel_color2 = svld1ub_s32(pv_cn, s_ptr2);
                let pixel_color3 = svld1ub_s32(pv_cn, s_ptr3);
                let pixel_color4 = svld1ub_s32(pv_cn, s_ptr4);

                let arr_index = ((x + radius_64) & 1023) as usize;

                svst1_s32(
                    pv_cn,
                    buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color0,
                );
                svst1_s32(
                    pv_cn,
                    buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color1,
                );
                svst1_s32(
                    pv_cn,
                    buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color2,
                );
                svst1_s32(
                    pv_cn,
                    buffer3.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color3,
                );
                svst1_s32(
                    pv_cn,
                    buffer4.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color4,
                );

                diffs0 = svadd_s32_x(pv_cn, diffs0, pixel_color0);
                diffs1 = svadd_s32_x(pv_cn, diffs1, pixel_color1);
                diffs2 = svadd_s32_x(pv_cn, diffs2, pixel_color2);
                diffs3 = svadd_s32_x(pv_cn, diffs3, pixel_color3);
                diffs4 = svadd_s32_x(pv_cn, diffs4, pixel_color4);

                summs0 = svadd_s32_x(pv_cn, summs0, diffs0);
                summs1 = svadd_s32_x(pv_cn, summs1, diffs1);
                summs2 = svadd_s32_x(pv_cn, summs2, diffs2);
                summs3 = svadd_s32_x(pv_cn, summs3, diffs3);
                summs4 = svadd_s32_x(pv_cn, summs4, diffs4);
            }

            yy += 5;
        }

        for y in yy..height.min(end) {
            let mut diffs = svdup_n_s32(0);
            let mut summs = svdup_n_s32(initial_sum);

            let current_y = ((y as i64) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = (x as u32 * CN as u32) as usize;

                    let prepared_px_s32 = svqrdmulh_n_s32(summs, v_weight);

                    let bytes_offset = current_y + current_px;
                    svst1b_u32(
                        pv_cn,
                        bytes.get_ptr(bytes_offset).cast(),
                        svreinterpret_u32_s32(prepared_px_s32),
                    );

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let d_buf_ptr = buffer0.get_unchecked(d_arr_index).0.as_ptr();
                    let d_stored = svld1_s32(pv_cn, d_buf_ptr);

                    let buf_ptr = buffer0.get_unchecked(arr_index).0.as_ptr();
                    let a_stored = svld1_s32(pv_cn, buf_ptr);

                    diffs = svadd_s32_x(
                        pv_cn,
                        diffs,
                        svsub_s32_x(pv_cn, a_stored, svadd_s32_x(pv_cn, d_stored, d_stored)),
                    );
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked(arr_index).0.as_ptr();
                    let mut stored = svld1_s32(pv_cn, buf_ptr);
                    stored = svlsl_n_s32_x(pv_cn, stored, 1);
                    diffs = svsub_s32_x(pv_cn, diffs, stored);
                }

                let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let pixel_color = svld1ub_s32(pv_cn, bytes.get_ptr(current_y + next_row_px));

                let arr_index = ((x + radius_64) & 1023) as usize;
                let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();

                diffs = svadd_s32_x(pv_cn, diffs, pixel_color);
                summs = svadd_s32_x(pv_cn, summs, diffs);
                svst1_s32(pv_cn, buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fg_vertical_pass_neon_u8_sve<const CN: usize>(
    slice: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        fg_vertical_pass_neon_sve::<CN>(
            slice, stride, width, height, radius, start, end, edge_mode,
        );
    }
}
#[target_feature(enable = "sve2", enable = "sve")]
fn fg_vertical_pass_neon_sve<const CN: usize>(
    bytes: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let mut full_buffer = [SveI32x4::default(); 1024 * 5];

        let (buffer0, rem) = full_buffer.split_at_mut(1024);
        let (buffer1, rem) = rem.split_at_mut(1024);
        let (buffer2, rem) = rem.split_at_mut(1024);
        let (buffer3, buffer4) = rem.split_at_mut(1024);

        let pv_cn = svwhilelt_b32_u32(0u32, CN as u32);

        let initial_sum = ((radius * radius) >> 1) as i32;

        let height_wide = height as i64;
        let radius_64 = radius as i64;

        const Q: f64 = ((1i64 << 31i64) - 1) as f64;
        let weight = Q / (radius as f64 * radius as f64);
        let v_weight = weight as i32;

        let mut xx = start;

        while xx + 5 <= width.min(end) {
            let mut diffs0 = svdup_n_s32(0);
            let mut diffs1 = svdup_n_s32(0);
            let mut diffs2 = svdup_n_s32(0);
            let mut diffs3 = svdup_n_s32(0);
            let mut diffs4 = svdup_n_s32(0);

            let mut summs0 = svdup_n_s32(initial_sum);
            let mut summs1 = svdup_n_s32(initial_sum);
            let mut summs2 = svdup_n_s32(initial_sum);
            let mut summs3 = svdup_n_s32(initial_sum);
            let mut summs4 = svdup_n_s32(initial_sum);

            let current_px0 = (xx * CN as u32) as usize;
            let current_px1 = ((xx + 1) * CN as u32) as usize;
            let current_px2 = ((xx + 2) * CN as u32) as usize;
            let current_px3 = ((xx + 3) * CN as u32) as usize;
            let current_px4 = ((xx + 4) * CN as u32) as usize;

            let start_y = 0 - 2 * radius_64;

            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * stride as i64) as usize;

                    let prepared_px0 = svqrdmulh_n_s32(summs0, v_weight);
                    let prepared_px1 = svqrdmulh_n_s32(summs1, v_weight);
                    let prepared_px2 = svqrdmulh_n_s32(summs2, v_weight);
                    let prepared_px3 = svqrdmulh_n_s32(summs3, v_weight);
                    let prepared_px4 = svqrdmulh_n_s32(summs4, v_weight);

                    svst1b_u32(
                        pv_cn,
                        bytes.get_ptr(current_y + current_px0),
                        svreinterpret_u32_s32(prepared_px0),
                    );
                    svst1b_u32(
                        pv_cn,
                        bytes.get_ptr(current_y + current_px1),
                        svreinterpret_u32_s32(prepared_px1),
                    );
                    svst1b_u32(
                        pv_cn,
                        bytes.get_ptr(current_y + current_px2),
                        svreinterpret_u32_s32(prepared_px2),
                    );
                    svst1b_u32(
                        pv_cn,
                        bytes.get_ptr(current_y + current_px3),
                        svreinterpret_u32_s32(prepared_px3),
                    );
                    svst1b_u32(
                        pv_cn,
                        bytes.get_ptr(current_y + current_px4),
                        svreinterpret_u32_s32(prepared_px4),
                    );

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let d_stored0 = svld1_s32(pv_cn, buffer0.get_unchecked(d_arr_index).0.as_ptr());
                    let d_stored1 = svld1_s32(pv_cn, buffer1.get_unchecked(d_arr_index).0.as_ptr());
                    let d_stored2 = svld1_s32(pv_cn, buffer2.get_unchecked(d_arr_index).0.as_ptr());
                    let d_stored3 = svld1_s32(pv_cn, buffer3.get_unchecked(d_arr_index).0.as_ptr());
                    let d_stored4 = svld1_s32(pv_cn, buffer4.get_unchecked(d_arr_index).0.as_ptr());

                    let a_stored0 = svld1_s32(pv_cn, buffer0.get_unchecked(arr_index).0.as_ptr());
                    let a_stored1 = svld1_s32(pv_cn, buffer1.get_unchecked(arr_index).0.as_ptr());
                    let a_stored2 = svld1_s32(pv_cn, buffer2.get_unchecked(arr_index).0.as_ptr());
                    let a_stored3 = svld1_s32(pv_cn, buffer3.get_unchecked(arr_index).0.as_ptr());
                    let a_stored4 = svld1_s32(pv_cn, buffer4.get_unchecked(arr_index).0.as_ptr());

                    let d2_0 = svadd_s32_x(pv_cn, d_stored0, d_stored0);
                    let d2_1 = svadd_s32_x(pv_cn, d_stored1, d_stored1);
                    let d2_2 = svadd_s32_x(pv_cn, d_stored2, d_stored2);
                    let d2_3 = svadd_s32_x(pv_cn, d_stored3, d_stored3);
                    let d2_4 = svadd_s32_x(pv_cn, d_stored4, d_stored4);

                    let sub0 = svsub_s32_x(pv_cn, a_stored0, d2_0);
                    let sub1 = svsub_s32_x(pv_cn, a_stored1, d2_1);
                    let sub2 = svsub_s32_x(pv_cn, a_stored2, d2_2);
                    let sub3 = svsub_s32_x(pv_cn, a_stored3, d2_3);
                    let sub4 = svsub_s32_x(pv_cn, a_stored4, d2_4);

                    diffs0 = svadd_s32_x(pv_cn, diffs0, sub0);
                    diffs1 = svadd_s32_x(pv_cn, diffs1, sub1);
                    diffs2 = svadd_s32_x(pv_cn, diffs2, sub2);
                    diffs3 = svadd_s32_x(pv_cn, diffs3, sub3);
                    diffs4 = svadd_s32_x(pv_cn, diffs4, sub4);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;

                    let mut stored0 = svld1_s32(pv_cn, buffer0.get_unchecked(arr_index).0.as_ptr());
                    let mut stored1 = svld1_s32(pv_cn, buffer1.get_unchecked(arr_index).0.as_ptr());
                    let mut stored2 = svld1_s32(pv_cn, buffer2.get_unchecked(arr_index).0.as_ptr());
                    let mut stored3 = svld1_s32(pv_cn, buffer3.get_unchecked(arr_index).0.as_ptr());
                    let mut stored4 = svld1_s32(pv_cn, buffer4.get_unchecked(arr_index).0.as_ptr());

                    stored0 = svlsl_n_s32_x(pv_cn, stored0, 1);
                    stored1 = svlsl_n_s32_x(pv_cn, stored1, 1);
                    stored2 = svlsl_n_s32_x(pv_cn, stored2, 1);
                    stored3 = svlsl_n_s32_x(pv_cn, stored3, 1);
                    stored4 = svlsl_n_s32_x(pv_cn, stored4, 1);

                    diffs0 = svsub_s32_x(pv_cn, diffs0, stored0);
                    diffs1 = svsub_s32_x(pv_cn, diffs1, stored1);
                    diffs2 = svsub_s32_x(pv_cn, diffs2, stored2);
                    diffs3 = svsub_s32_x(pv_cn, diffs3, stored3);
                    diffs4 = svsub_s32_x(pv_cn, diffs4, stored4);
                }

                let next_row_y =
                    clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * stride as usize;

                let pixel_color0 = svld1ub_s32(pv_cn, bytes.get_ptr(next_row_y + current_px0));
                let pixel_color1 = svld1ub_s32(pv_cn, bytes.get_ptr(next_row_y + current_px1));
                let pixel_color2 = svld1ub_s32(pv_cn, bytes.get_ptr(next_row_y + current_px2));
                let pixel_color3 = svld1ub_s32(pv_cn, bytes.get_ptr(next_row_y + current_px3));
                let pixel_color4 = svld1ub_s32(pv_cn, bytes.get_ptr(next_row_y + current_px4));

                let arr_index = ((y + radius_64) & 1023) as usize;

                svst1_s32(
                    pv_cn,
                    buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color0,
                );
                svst1_s32(
                    pv_cn,
                    buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color1,
                );
                svst1_s32(
                    pv_cn,
                    buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color2,
                );
                svst1_s32(
                    pv_cn,
                    buffer3.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color3,
                );
                svst1_s32(
                    pv_cn,
                    buffer4.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color4,
                );

                diffs0 = svadd_s32_x(pv_cn, diffs0, pixel_color0);
                diffs1 = svadd_s32_x(pv_cn, diffs1, pixel_color1);
                diffs2 = svadd_s32_x(pv_cn, diffs2, pixel_color2);
                diffs3 = svadd_s32_x(pv_cn, diffs3, pixel_color3);
                diffs4 = svadd_s32_x(pv_cn, diffs4, pixel_color4);

                summs0 = svadd_s32_x(pv_cn, summs0, diffs0);
                summs1 = svadd_s32_x(pv_cn, summs1, diffs1);
                summs2 = svadd_s32_x(pv_cn, summs2, diffs2);
                summs3 = svadd_s32_x(pv_cn, summs3, diffs3);
                summs4 = svadd_s32_x(pv_cn, summs4, diffs4);
            }

            xx += 5;
        }

        for x in xx..width.min(end) {
            let mut diffs = svdup_n_s32(0);
            let mut summs = svdup_n_s32(initial_sum);

            let current_px = (x * CN as u32) as usize;
            let start_y = 0 - 2 * radius_64;

            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * stride as i64) as usize;

                    let prepared_px = svqrdmulh_n_s32(summs, v_weight);
                    svst1b_u32(
                        pv_cn,
                        bytes.get_ptr(current_y + current_px),
                        svreinterpret_u32_s32(prepared_px),
                    );

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let d_stored = svld1_s32(pv_cn, buffer0.get_unchecked(d_arr_index).0.as_ptr());
                    let a_stored = svld1_s32(pv_cn, buffer0.get_unchecked(arr_index).0.as_ptr());

                    diffs = svadd_s32_x(
                        pv_cn,
                        diffs,
                        svsub_s32_x(pv_cn, a_stored, svadd_s32_x(pv_cn, d_stored, d_stored)),
                    );
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let stored = svld1_s32(pv_cn, buffer0.get_unchecked(arr_index).0.as_ptr());
                    diffs = svsub_s32_x(pv_cn, diffs, svlsl_n_s32_x(pv_cn, stored, 1));
                }

                let next_row_y =
                    clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * stride as usize;

                let pixel_color = svld1ub_s32(pv_cn, bytes.get_ptr(next_row_y + current_px));

                let arr_index = ((y + radius_64) & 1023) as usize;
                svst1_s32(
                    pv_cn,
                    buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr(),
                    pixel_color,
                );

                diffs = svadd_s32_x(pv_cn, diffs, pixel_color);
                summs = svadd_s32_x(pv_cn, summs, diffs);
            }
        }
    }
}
