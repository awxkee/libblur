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
use crate::sve::fast_gaussian_q0_31::SveI32x4;
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;

pub(crate) fn fgn_vertical_pass_neon_u8_sve<const CN: usize>(
    undefined_slice: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        fgn_vertical_pass_neon_impl_sve::<CN>(
            undefined_slice,
            stride,
            width,
            height,
            radius,
            start,
            end,
            edge_mode,
        );
    }
}

#[target_feature(enable = "sve2", enable = "sve")]
fn fgn_vertical_pass_neon_impl_sve<const CN: usize>(
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
        let mut full_buffer = [SveI32x4::default(); 1024 * 4];

        let (buffer0, rem) = full_buffer.split_at_mut(1024);
        let (buffer1, rem) = rem.split_at_mut(1024);
        let (buffer2, buffer3) = rem.split_at_mut(1024);

        let height_wide = height as i64;

        let radius_64 = radius as i64;
        const Q: f64 = ((1i64 << 31i64) - 1) as f64;
        let weight = Q / ((radius as f64) * (radius as f64) * (radius as f64));
        let v_weight = weight as i32;

        let pv_cn = svwhilelt_b32_u32(0u32, CN as u32);

        let mut xx = start;

        while xx + 4 <= width.min(end) {
            let mut diffs0 = svdup_n_s32(0);
            let mut diffs1 = svdup_n_s32(0);
            let mut diffs2 = svdup_n_s32(0);
            let mut diffs3 = svdup_n_s32(0);

            let mut ders0 = svdup_n_s32(0);
            let mut ders1 = svdup_n_s32(0);
            let mut ders2 = svdup_n_s32(0);
            let mut ders3 = svdup_n_s32(0);

            let mut summs0 = svdup_n_s32(0);
            let mut summs1 = svdup_n_s32(0);
            let mut summs2 = svdup_n_s32(0);
            let mut summs3 = svdup_n_s32(0);

            let start_y = 0 - 3 * radius as i64;

            let current_px0 = (xx * CN as u32) as usize;
            let current_px1 = ((xx + 1) * CN as u32) as usize;
            let current_px2 = ((xx + 2) * CN as u32) as usize;
            let current_px3 = ((xx + 3) * CN as u32) as usize;

            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;

                if y >= 0 {
                    let prepared_px0 = svqrdmulh_n_s32(summs0, v_weight);
                    let prepared_px1 = svqrdmulh_n_s32(summs1, v_weight);
                    let prepared_px2 = svqrdmulh_n_s32(summs2, v_weight);
                    let prepared_px3 = svqrdmulh_n_s32(summs3, v_weight);

                    let dst_ptr0 = bytes.get_ptr(current_y + current_px0);
                    let dst_ptr1 = bytes.get_ptr(current_y + current_px1);
                    let dst_ptr2 = bytes.get_ptr(current_y + current_px2);
                    let dst_ptr3 = bytes.get_ptr(current_y + current_px3);

                    svst1b_u32(pv_cn, dst_ptr0, svreinterpret_u32_s32(prepared_px0));
                    svst1b_u32(pv_cn, dst_ptr1, svreinterpret_u32_s32(prepared_px1));
                    svst1b_u32(pv_cn, dst_ptr2, svreinterpret_u32_s32(prepared_px2));
                    svst1b_u32(pv_cn, dst_ptr3, svreinterpret_u32_s32(prepared_px3));

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let stored0 =
                        svld1_s32(pv_cn, buffer0.get_unchecked(d_arr_index..).as_ptr().cast());
                    let stored1 =
                        svld1_s32(pv_cn, buffer1.get_unchecked(d_arr_index..).as_ptr().cast());
                    let stored2 =
                        svld1_s32(pv_cn, buffer2.get_unchecked(d_arr_index..).as_ptr().cast());
                    let stored3 =
                        svld1_s32(pv_cn, buffer3.get_unchecked(d_arr_index..).as_ptr().cast());

                    let stored_10 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                    );
                    let stored_11 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                    );
                    let stored_12 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                    );
                    let stored_13 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                    );

                    let j0 = svsub_s32_x(pv_cn, stored0, stored_10);
                    let j1 = svsub_s32_x(pv_cn, stored1, stored_11);
                    let j2 = svsub_s32_x(pv_cn, stored2, stored_12);
                    let j3 = svsub_s32_x(pv_cn, stored3, stored_13);

                    let stored_20 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                    );
                    let stored_21 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                    );
                    let stored_22 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                    );
                    let stored_23 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                    );

                    let k0 = svmul_n_s32_x(pv_cn, j0, 3);
                    let k1 = svmul_n_s32_x(pv_cn, j1, 3);
                    let k2 = svmul_n_s32_x(pv_cn, j2, 3);
                    let k3 = svmul_n_s32_x(pv_cn, j3, 3);

                    let new_diff0 = svsub_s32_x(pv_cn, k0, stored_20);
                    let new_diff1 = svsub_s32_x(pv_cn, k1, stored_21);
                    let new_diff2 = svsub_s32_x(pv_cn, k2, stored_22);
                    let new_diff3 = svsub_s32_x(pv_cn, k3, stored_23);

                    diffs0 = svadd_s32_x(pv_cn, diffs0, new_diff0);
                    diffs1 = svadd_s32_x(pv_cn, diffs1, new_diff1);
                    diffs2 = svadd_s32_x(pv_cn, diffs2, new_diff2);
                    diffs3 = svadd_s32_x(pv_cn, diffs3, new_diff3);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;

                    let stored0 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored1 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored2 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored3 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );

                    let stored_10 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked_mut(arr_index_1..).as_mut_ptr().cast(),
                    );
                    let stored_11 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked_mut(arr_index_1..).as_mut_ptr().cast(),
                    );
                    let stored_12 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked_mut(arr_index_1..).as_mut_ptr().cast(),
                    );
                    let stored_13 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked_mut(arr_index_1..).as_mut_ptr().cast(),
                    );

                    let q0 = svsub_s32_x(pv_cn, stored0, stored_10);
                    let q1 = svsub_s32_x(pv_cn, stored1, stored_11);
                    let q2 = svsub_s32_x(pv_cn, stored2, stored_12);
                    let q3 = svsub_s32_x(pv_cn, stored3, stored_13);

                    diffs0 = svmla_n_s32_x(pv_cn, diffs0, q0, 3);
                    diffs1 = svmla_n_s32_x(pv_cn, diffs1, q1, 3);
                    diffs2 = svmla_n_s32_x(pv_cn, diffs2, q2, 3);
                    diffs3 = svmla_n_s32_x(pv_cn, diffs3, q3, 3);
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;

                    let stored0 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored1 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored2 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored3 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );

                    diffs0 = svmla_n_s32_x(pv_cn, diffs0, stored0, -3);
                    diffs1 = svmla_n_s32_x(pv_cn, diffs1, stored1, -3);
                    diffs2 = svmla_n_s32_x(pv_cn, diffs2, stored2, -3);
                    diffs3 = svmla_n_s32_x(pv_cn, diffs3, stored3, -3);
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * (stride as usize);

                let s_ptr0 = bytes.get_ptr(next_row_y + current_px0);
                let s_ptr1 = bytes.get_ptr(next_row_y + current_px1);
                let s_ptr2 = bytes.get_ptr(next_row_y + current_px2);
                let s_ptr3 = bytes.get_ptr(next_row_y + current_px3);

                let pixel_color0 = svld1ub_s32(pv_cn, s_ptr0);
                let pixel_color1 = svld1ub_s32(pv_cn, s_ptr1);
                let pixel_color2 = svld1ub_s32(pv_cn, s_ptr2);
                let pixel_color3 = svld1ub_s32(pv_cn, s_ptr3);

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;

                let buf_ptr0 = buffer0.get_unchecked_mut(arr_index..).as_mut_ptr().cast();
                let buf_ptr1 = buffer1.get_unchecked_mut(arr_index..).as_mut_ptr().cast();
                let buf_ptr2 = buffer2.get_unchecked_mut(arr_index..).as_mut_ptr().cast();
                let buf_ptr3 = buffer3.get_unchecked_mut(arr_index..).as_mut_ptr().cast();

                diffs0 = svadd_s32_x(pv_cn, diffs0, pixel_color0);
                diffs1 = svadd_s32_x(pv_cn, diffs1, pixel_color1);
                diffs2 = svadd_s32_x(pv_cn, diffs2, pixel_color2);
                diffs3 = svadd_s32_x(pv_cn, diffs3, pixel_color3);

                svst1_s32(pv_cn, buf_ptr0, pixel_color0);
                svst1_s32(pv_cn, buf_ptr1, pixel_color1);
                svst1_s32(pv_cn, buf_ptr2, pixel_color2);
                svst1_s32(pv_cn, buf_ptr3, pixel_color3);

                ders0 = svadd_s32_x(pv_cn, ders0, diffs0);
                ders1 = svadd_s32_x(pv_cn, ders1, diffs1);
                ders2 = svadd_s32_x(pv_cn, ders2, diffs2);
                ders3 = svadd_s32_x(pv_cn, ders3, diffs3);

                summs0 = svadd_s32_x(pv_cn, summs0, ders0);
                summs1 = svadd_s32_x(pv_cn, summs1, ders1);
                summs2 = svadd_s32_x(pv_cn, summs2, ders2);
                summs3 = svadd_s32_x(pv_cn, summs3, ders3);
            }
            xx += 4;
        }

        for x in xx..width.min(end) {
            let mut diffs = svdup_n_s32(0);
            let mut ders = svdup_n_s32(0);
            let mut summs = svdup_n_s32(0);

            let current_px = (x * CN as u32) as usize;

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;

                if y >= 0 {
                    let new_px = svqrdmulh_n_s32(summs, v_weight);

                    let bytes_offset = current_y + current_px;
                    svst1b_u32(
                        pv_cn,
                        bytes.get_ptr(bytes_offset),
                        svreinterpret_u32_s32(new_px),
                    );

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let stored =
                        svld1_s32(pv_cn, buffer0.get_unchecked(d_arr_index..).as_ptr().cast());
                    let stored_1 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                    );
                    let stored_2 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                    );

                    let new_diff = svsub_s32_x(
                        pv_cn,
                        svmul_n_s32_x(pv_cn, svsub_s32_x(pv_cn, stored, stored_1), 3),
                        stored_2,
                    );
                    diffs = svadd_s32_x(pv_cn, diffs, new_diff);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index..).as_mut_ptr().cast();
                    let stored = svld1_s32(pv_cn, buf_ptr);

                    let buf_ptr_1 = buffer0.get_unchecked_mut(arr_index_1..).as_mut_ptr().cast();
                    let stored_1 = svld1_s32(pv_cn, buf_ptr_1);

                    diffs = svmla_n_s32_x(pv_cn, diffs, svsub_s32_x(pv_cn, stored, stored_1), 3);
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_ptr().cast();
                    let stored = svld1_s32(pv_cn, buf_ptr);
                    diffs = svmla_n_s32_x(pv_cn, diffs, stored, -3);
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * (stride as usize);
                let next_row_x = (x * CN as u32) as usize;

                let s_ptr = bytes.get_ptr(next_row_y + next_row_x);

                let pixel_color = svld1ub_s32(pv_cn, s_ptr);

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer0.get_unchecked_mut(arr_index..).as_mut_ptr().cast();

                diffs = svadd_s32_x(pv_cn, diffs, pixel_color);
                ders = svadd_s32_x(pv_cn, ders, diffs);
                summs = svadd_s32_x(pv_cn, summs, ders);
                svst1_s32(pv_cn, buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fgn_horizontal_pass_neon_u8_sve<const CN: usize>(
    undefined_slice: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        fgn_horizontal_pass_neon_impl_sve::<CN>(
            undefined_slice,
            stride,
            width,
            height,
            radius,
            start,
            end,
            edge_mode,
        );
    }
}

#[target_feature(enable = "sve2", enable = "sve")]
fn fgn_horizontal_pass_neon_impl_sve<const CN: usize>(
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
        let mut full_buffer = [SveI32x4::default(); 1024 * 4];

        let (buffer0, rem) = full_buffer.split_at_mut(1024);
        let (buffer1, rem) = rem.split_at_mut(1024);
        let (buffer2, buffer3) = rem.split_at_mut(1024);

        let width_wide = width as i64;

        let radius_64 = radius as i64;
        const Q: f64 = ((1i64 << 31i64) - 1) as f64;
        let weight = Q / ((radius as f64) * (radius as f64) * (radius as f64));
        let v_weight = weight as i32;

        let pv_cn = svwhilelt_b32_u32(0u32, CN as u32);

        let mut yy = start;

        while yy + 4 <= height.min(end) {
            let mut diffs0 = svdup_n_s32(0);
            let mut diffs1 = svdup_n_s32(0);
            let mut diffs2 = svdup_n_s32(0);
            let mut diffs3 = svdup_n_s32(0);

            let mut ders0 = svdup_n_s32(0);
            let mut ders1 = svdup_n_s32(0);
            let mut ders2 = svdup_n_s32(0);
            let mut ders3 = svdup_n_s32(0);

            let mut summs0 = svdup_n_s32(0);
            let mut summs1 = svdup_n_s32(0);
            let mut summs2 = svdup_n_s32(0);
            let mut summs3 = svdup_n_s32(0);

            let current_y0 = ((yy as i64) * (stride as i64)) as usize;
            let current_y1 = ((yy as i64 + 1) * (stride as i64)) as usize;
            let current_y2 = ((yy as i64 + 2) * (stride as i64)) as usize;
            let current_y3 = ((yy as i64 + 3) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let prepared_px0 = svqrdmulh_n_s32(summs0, v_weight);
                    let prepared_px1 = svqrdmulh_n_s32(summs1, v_weight);
                    let prepared_px2 = svqrdmulh_n_s32(summs2, v_weight);
                    let prepared_px3 = svqrdmulh_n_s32(summs3, v_weight);

                    let dst_ptr0 = bytes.get_ptr(current_y0 + current_px);
                    let dst_ptr1 = bytes.get_ptr(current_y1 + current_px);
                    let dst_ptr2 = bytes.get_ptr(current_y2 + current_px);
                    let dst_ptr3 = bytes.get_ptr(current_y3 + current_px);

                    svst1b_u32(pv_cn, dst_ptr0, svreinterpret_u32_s32(prepared_px0));
                    svst1b_u32(pv_cn, dst_ptr1, svreinterpret_u32_s32(prepared_px1));
                    svst1b_u32(pv_cn, dst_ptr2, svreinterpret_u32_s32(prepared_px2));
                    svst1b_u32(pv_cn, dst_ptr3, svreinterpret_u32_s32(prepared_px3));

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let stored0 =
                        svld1_s32(pv_cn, buffer0.get_unchecked(d_arr_index..).as_ptr().cast());
                    let stored1 =
                        svld1_s32(pv_cn, buffer1.get_unchecked(d_arr_index..).as_ptr().cast());
                    let stored2 =
                        svld1_s32(pv_cn, buffer2.get_unchecked(d_arr_index..).as_ptr().cast());
                    let stored3 =
                        svld1_s32(pv_cn, buffer3.get_unchecked(d_arr_index..).as_ptr().cast());

                    let stored_10 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                    );
                    let stored_11 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                    );
                    let stored_12 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                    );
                    let stored_13 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                    );

                    let j0 = svsub_s32_x(pv_cn, stored0, stored_10);
                    let j1 = svsub_s32_x(pv_cn, stored1, stored_11);
                    let j2 = svsub_s32_x(pv_cn, stored2, stored_12);
                    let j3 = svsub_s32_x(pv_cn, stored3, stored_13);

                    let stored_20 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                    );
                    let stored_21 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                    );
                    let stored_22 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                    );
                    let stored_23 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                    );

                    let k0 = svmul_n_s32_x(pv_cn, j0, 3);
                    let k1 = svmul_n_s32_x(pv_cn, j1, 3);
                    let k2 = svmul_n_s32_x(pv_cn, j2, 3);
                    let k3 = svmul_n_s32_x(pv_cn, j3, 3);

                    let new_diff0 = svsub_s32_x(pv_cn, k0, stored_20);
                    let new_diff1 = svsub_s32_x(pv_cn, k1, stored_21);
                    let new_diff2 = svsub_s32_x(pv_cn, k2, stored_22);
                    let new_diff3 = svsub_s32_x(pv_cn, k3, stored_23);

                    diffs0 = svadd_s32_x(pv_cn, diffs0, new_diff0);
                    diffs1 = svadd_s32_x(pv_cn, diffs1, new_diff1);
                    diffs2 = svadd_s32_x(pv_cn, diffs2, new_diff2);
                    diffs3 = svadd_s32_x(pv_cn, diffs3, new_diff3);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;

                    let stored0 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored1 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored2 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored3 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );

                    let stored_10 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked_mut(arr_index_1..).as_mut_ptr().cast(),
                    );
                    let stored_11 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked_mut(arr_index_1..).as_mut_ptr().cast(),
                    );
                    let stored_12 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked_mut(arr_index_1..).as_mut_ptr().cast(),
                    );
                    let stored_13 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked_mut(arr_index_1..).as_mut_ptr().cast(),
                    );

                    let q0 = svsub_s32_x(pv_cn, stored0, stored_10);
                    let q1 = svsub_s32_x(pv_cn, stored1, stored_11);
                    let q2 = svsub_s32_x(pv_cn, stored2, stored_12);
                    let q3 = svsub_s32_x(pv_cn, stored3, stored_13);

                    diffs0 = svmla_n_s32_x(pv_cn, diffs0, q0, 3);
                    diffs1 = svmla_n_s32_x(pv_cn, diffs1, q1, 3);
                    diffs2 = svmla_n_s32_x(pv_cn, diffs2, q2, 3);
                    diffs3 = svmla_n_s32_x(pv_cn, diffs3, q3, 3);
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let stored0 = svld1_s32(
                        pv_cn,
                        buffer0.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored1 = svld1_s32(
                        pv_cn,
                        buffer1.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored2 = svld1_s32(
                        pv_cn,
                        buffer2.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );
                    let stored3 = svld1_s32(
                        pv_cn,
                        buffer3.get_unchecked_mut(arr_index..).as_mut_ptr().cast(),
                    );

                    diffs0 = svmla_n_s32_x(pv_cn, diffs0, stored0, -3);
                    diffs1 = svmla_n_s32_x(pv_cn, diffs1, stored1, -3);
                    diffs2 = svmla_n_s32_x(pv_cn, diffs2, stored2, -3);
                    diffs3 = svmla_n_s32_x(pv_cn, diffs3, stored3, -3);
                }

                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr0 = bytes.get_ptr(current_y0 + next_row_px);
                let s_ptr1 = bytes.get_ptr(current_y1 + next_row_px);
                let s_ptr2 = bytes.get_ptr(current_y2 + next_row_px);
                let s_ptr3 = bytes.get_ptr(current_y3 + next_row_px);

                let pixel_color0 = svld1ub_s32(pv_cn, s_ptr0);
                let pixel_color1 = svld1ub_s32(pv_cn, s_ptr1);
                let pixel_color2 = svld1ub_s32(pv_cn, s_ptr2);
                let pixel_color3 = svld1ub_s32(pv_cn, s_ptr3);

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr0 = buffer0.get_unchecked_mut(arr_index..).as_mut_ptr().cast();
                let buf_ptr1 = buffer1.get_unchecked_mut(arr_index..).as_mut_ptr().cast();
                let buf_ptr2 = buffer2.get_unchecked_mut(arr_index..).as_mut_ptr().cast();
                let buf_ptr3 = buffer3.get_unchecked_mut(arr_index..).as_mut_ptr().cast();

                diffs0 = svadd_s32_x(pv_cn, diffs0, pixel_color0);
                diffs1 = svadd_s32_x(pv_cn, diffs1, pixel_color1);
                diffs2 = svadd_s32_x(pv_cn, diffs2, pixel_color2);
                diffs3 = svadd_s32_x(pv_cn, diffs3, pixel_color3);

                svst1_s32(pv_cn, buf_ptr0, pixel_color0);
                svst1_s32(pv_cn, buf_ptr1, pixel_color1);
                svst1_s32(pv_cn, buf_ptr2, pixel_color2);
                svst1_s32(pv_cn, buf_ptr3, pixel_color3);

                ders0 = svadd_s32_x(pv_cn, ders0, diffs0);
                ders1 = svadd_s32_x(pv_cn, ders1, diffs1);
                ders2 = svadd_s32_x(pv_cn, ders2, diffs2);
                ders3 = svadd_s32_x(pv_cn, ders3, diffs3);

                summs0 = svadd_s32_x(pv_cn, summs0, ders0);
                summs1 = svadd_s32_x(pv_cn, summs1, ders1);
                summs2 = svadd_s32_x(pv_cn, summs2, ders2);
                summs3 = svadd_s32_x(pv_cn, summs3, ders3);
            }

            yy += 4;
        }

        for y in yy..height.min(end) {
            let mut diffs = svdup_n_s32(0);
            let mut ders = svdup_n_s32(0);
            let mut summs = svdup_n_s32(0);

            let current_y = ((y as i64) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let next_px = svqrdmulh_n_s32(summs, v_weight);

                    let bytes_offset = current_y + current_px;

                    let dst_ptr = bytes.get_ptr(bytes_offset);
                    svst1b_u32(pv_cn, dst_ptr, svreinterpret_u32_s32(next_px));

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let buf_ptr = buffer0.get_unchecked_mut(d_arr_index..).as_ptr().cast();
                    let stored = svld1_s32(pv_cn, buf_ptr);

                    let buf_ptr_1 = buffer0.get_unchecked_mut(d_arr_index_1..).as_ptr().cast();
                    let stored_1 = svld1_s32(pv_cn, buf_ptr_1);

                    let buf_ptr_2 = buffer0.get_unchecked_mut(d_arr_index_2..).as_ptr().cast();
                    let stored_2 = svld1_s32(pv_cn, buf_ptr_2);

                    let new_diff = svsub_s32_x(
                        pv_cn,
                        svmul_n_s32_x(pv_cn, svsub_s32_x(pv_cn, stored, stored_1), 3),
                        stored_2,
                    );
                    diffs = svadd_s32_x(pv_cn, diffs, new_diff);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index..).as_ptr().cast();
                    let stored = svld1_s32(pv_cn, buf_ptr);

                    let buf_ptr_1 = buffer0.get_unchecked_mut(arr_index_1..).as_ptr().cast();
                    let stored_1 = svld1_s32(pv_cn, buf_ptr_1);

                    diffs = svmla_n_s32_x(pv_cn, diffs, svsub_s32_x(pv_cn, stored, stored_1), 3);
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index..).as_ptr().cast();
                    let stored = svld1_s32(pv_cn, buf_ptr);
                    diffs = svmla_n_s32_x(pv_cn, diffs, stored, -3);
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr = bytes.get_ptr(next_row_y + next_row_px);

                let pixel_color = svld1ub_s32(pv_cn, s_ptr);

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer0.get_unchecked_mut(arr_index..).as_mut_ptr().cast();

                diffs = svadd_s32_x(pv_cn, diffs, pixel_color);
                ders = svadd_s32_x(pv_cn, ders, diffs);
                summs = svadd_s32_x(pv_cn, summs, ders);
                svst1_s32(pv_cn, buf_ptr, pixel_color);
            }
        }
    }
}
