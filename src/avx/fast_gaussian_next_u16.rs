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

use crate::EdgeMode;
use crate::avx::fast_gaussian_next::AvxSseI32x8;
use crate::avx::utils::_mm256_mul_by_3_epi32;
use crate::edge_mode::clamp_edge;
use crate::sse::utils::load_u16_s32_fast;
use crate::sse::{_mm_mul_by_3_epi32, store_u16_u32};
use crate::unsafe_slice::UnsafeSlice;
use std::arch::x86_64::*;

pub(crate) fn fgn_vertical_pass_avx_u16<const CN: usize>(
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
        let horizontal_unit = VerticalExecutionUnit::<CN>::default();
        horizontal_unit.pass(bytes, stride, width, height, radius, start, end, edge_mode);
    }
}

#[derive(Copy, Clone, Default)]
struct VerticalExecutionUnit<const CN: usize> {}

impl<const CN: usize> VerticalExecutionUnit<CN> {
    #[target_feature(enable = "avx2")]
    fn pass(
        &self,
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
            let mut full_buffer = [AvxSseI32x8::default(); 1024 * 3];

            let (buffer0, rem) = full_buffer.split_at_mut(1024);
            let (buffer1, buffer2) = rem.split_at_mut(1024);

            let height_wide = height as i64;

            let radius_64 = radius as i64;

            let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
            let v_weight = _mm256_set1_ps(weight);

            let mut xx = start;

            while xx + 6 <= width.min(end) {
                let mut diffs0 = _mm256_setzero_si256();
                let mut diffs1 = _mm256_setzero_si256();
                let mut diffs2 = _mm256_setzero_si256();

                let mut ders0 = _mm256_setzero_si256();
                let mut ders1 = _mm256_setzero_si256();
                let mut ders2 = _mm256_setzero_si256();

                let mut summs0 = _mm256_setzero_si256();
                let mut summs1 = _mm256_setzero_si256();
                let mut summs2 = _mm256_setzero_si256();

                let start_y = 0 - 3 * radius as i64;

                let current_px0 = (xx * CN as u32) as usize;
                let current_px1 = ((xx + 1) * CN as u32) as usize;
                let current_px2 = ((xx + 2) * CN as u32) as usize;
                let current_px3 = ((xx + 3) * CN as u32) as usize;
                let current_px4 = ((xx + 4) * CN as u32) as usize;
                let current_px5 = ((xx + 5) * CN as u32) as usize;

                for y in start_y..height_wide {
                    if y >= 0 {
                        let ps01 = _mm256_cvtepi32_ps(summs0);
                        let ps23 = _mm256_cvtepi32_ps(summs1);
                        let ps45 = _mm256_cvtepi32_ps(summs2);

                        let r01 = _mm256_mul_ps(ps01, v_weight);
                        let r23 = _mm256_mul_ps(ps23, v_weight);
                        let r45 = _mm256_mul_ps(ps45, v_weight);

                        let e01 = _mm256_cvtps_epi32(r01);
                        let e23 = _mm256_cvtps_epi32(r23);
                        let e45 = _mm256_cvtps_epi32(r45);

                        let prepared_px0 = _mm256_castsi256_si128(e01);
                        let prepared_px1 = _mm256_extracti128_si256::<1>(e01);
                        let prepared_px2 = _mm256_castsi256_si128(e23);
                        let prepared_px3 = _mm256_extracti128_si256::<1>(e23);
                        let prepared_px4 = _mm256_castsi256_si128(e45);
                        let prepared_px5 = _mm256_extracti128_si256::<1>(e45);

                        let current_y = (y * (stride as i64)) as usize;

                        store_u16_u32::<CN>(bytes.get_ptr(current_y + current_px0), prepared_px0);
                        store_u16_u32::<CN>(bytes.get_ptr(current_y + current_px1), prepared_px1);
                        store_u16_u32::<CN>(bytes.get_ptr(current_y + current_px2), prepared_px2);
                        store_u16_u32::<CN>(bytes.get_ptr(current_y + current_px3), prepared_px3);
                        store_u16_u32::<CN>(bytes.get_ptr(current_y + current_px4), prepared_px4);
                        store_u16_u32::<CN>(bytes.get_ptr(current_y + current_px5), prepared_px5);

                        let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                        let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                        let d_arr_index = (y & 1023) as usize;

                        let stored0 =
                            _mm256_load_si256(buffer0.get_unchecked(d_arr_index..).as_ptr().cast());
                        let stored1 =
                            _mm256_load_si256(buffer1.get_unchecked(d_arr_index..).as_ptr().cast());
                        let stored2 =
                            _mm256_load_si256(buffer2.get_unchecked(d_arr_index..).as_ptr().cast());

                        let stored_10 = _mm256_load_si256(
                            buffer0.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                        );
                        let stored_11 = _mm256_load_si256(
                            buffer1.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                        );
                        let stored_21 = _mm256_load_si256(
                            buffer2.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                        );

                        let j0 = _mm256_sub_epi32(stored0, stored_10);
                        let j1 = _mm256_sub_epi32(stored1, stored_11);
                        let j2 = _mm256_sub_epi32(stored2, stored_21);

                        let stored_20 = _mm256_load_si256(
                            buffer0.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                        );
                        let stored_21 = _mm256_load_si256(
                            buffer1.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                        );
                        let stored_22 = _mm256_load_si256(
                            buffer2.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                        );

                        let k0 = _mm256_mul_by_3_epi32(j0);
                        let k1 = _mm256_mul_by_3_epi32(j1);
                        let k2 = _mm256_mul_by_3_epi32(j2);

                        let new_diff0 = _mm256_sub_epi32(k0, stored_20);
                        let new_diff1 = _mm256_sub_epi32(k1, stored_21);
                        let new_diff2 = _mm256_sub_epi32(k2, stored_22);

                        diffs0 = _mm256_add_epi32(diffs0, new_diff0);
                        diffs1 = _mm256_add_epi32(diffs1, new_diff1);
                        diffs2 = _mm256_add_epi32(diffs2, new_diff2);
                    } else if y + radius_64 >= 0 {
                        let arr_index = (y & 1023) as usize;
                        let arr_index_1 = ((y + radius_64) & 1023) as usize;

                        let stored0 =
                            _mm256_load_si256(buffer0.get_unchecked(arr_index).0.as_ptr().cast());
                        let stored1 =
                            _mm256_load_si256(buffer1.get_unchecked(arr_index).0.as_ptr().cast());
                        let stored2 =
                            _mm256_load_si256(buffer2.get_unchecked(arr_index).0.as_ptr().cast());

                        let stored_10 =
                            _mm256_load_si256(buffer0.get_unchecked(arr_index_1).0.as_ptr().cast());
                        let stored_11 =
                            _mm256_load_si256(buffer1.get_unchecked(arr_index_1).0.as_ptr().cast());
                        let stored_21 =
                            _mm256_load_si256(buffer2.get_unchecked(arr_index_1).0.as_ptr().cast());

                        let new_diff0 = _mm256_mul_by_3_epi32(_mm256_sub_epi32(stored0, stored_10));
                        let new_diff1 = _mm256_mul_by_3_epi32(_mm256_sub_epi32(stored1, stored_11));
                        let new_diff2 = _mm256_mul_by_3_epi32(_mm256_sub_epi32(stored2, stored_21));

                        diffs0 = _mm256_add_epi32(diffs0, new_diff0);
                        diffs1 = _mm256_add_epi32(diffs1, new_diff1);
                        diffs2 = _mm256_add_epi32(diffs2, new_diff2);
                    } else if y + 2 * radius_64 >= 0 {
                        let arr_index = ((y + radius_64) & 1023) as usize;

                        let stored0 =
                            _mm256_load_si256(buffer0.get_unchecked(arr_index).0.as_ptr().cast());
                        let stored1 =
                            _mm256_load_si256(buffer1.get_unchecked(arr_index).0.as_ptr().cast());
                        let stored2 =
                            _mm256_load_si256(buffer2.get_unchecked(arr_index).0.as_ptr().cast());

                        diffs0 = _mm256_sub_epi32(diffs0, _mm256_mul_by_3_epi32(stored0));
                        diffs1 = _mm256_sub_epi32(diffs1, _mm256_mul_by_3_epi32(stored1));
                        diffs2 = _mm256_sub_epi32(diffs2, _mm256_mul_by_3_epi32(stored2));
                    }

                    let next_row_y =
                        clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                            * (stride as usize);

                    let pixel_color0 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(next_row_y + current_px0));
                    let pixel_color1 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(next_row_y + current_px1));
                    let pixel_color2 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(next_row_y + current_px2));
                    let pixel_color3 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(next_row_y + current_px3));
                    let pixel_color4 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(next_row_y + current_px4));
                    let pixel_color5 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(next_row_y + current_px5));

                    let arr_index = ((y + 2 * radius_64) & 1023) as usize;

                    let buf_ptr0 = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                    let buf_ptr1 = buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr();
                    let buf_ptr2 = buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr();

                    let px01 = _mm256_inserti128_si256::<1>(
                        _mm256_castsi128_si256(pixel_color0),
                        pixel_color1,
                    );
                    let px23 = _mm256_inserti128_si256::<1>(
                        _mm256_castsi128_si256(pixel_color2),
                        pixel_color3,
                    );
                    let px45 = _mm256_inserti128_si256::<1>(
                        _mm256_castsi128_si256(pixel_color4),
                        pixel_color5,
                    );

                    diffs0 = _mm256_add_epi32(diffs0, px01);
                    diffs1 = _mm256_add_epi32(diffs1, px23);
                    diffs2 = _mm256_add_epi32(diffs2, px45);

                    _mm256_store_si256(buf_ptr0.cast(), px01);
                    _mm256_store_si256(buf_ptr1.cast(), px23);
                    _mm256_store_si256(buf_ptr2.cast(), px45);

                    ders0 = _mm256_add_epi32(ders0, diffs0);
                    ders1 = _mm256_add_epi32(ders1, diffs1);
                    ders2 = _mm256_add_epi32(ders2, diffs2);

                    summs0 = _mm256_add_epi32(summs0, ders0);
                    summs1 = _mm256_add_epi32(summs1, ders1);
                    summs2 = _mm256_add_epi32(summs2, ders2);
                }
                xx += 6;
            }

            for x in xx..width.min(end) {
                let mut diffs = _mm_setzero_si128();
                let mut ders = _mm_setzero_si128();
                let mut summs = _mm_setzero_si128();

                let start_y = 0 - 3 * radius as i64;
                for y in start_y..height_wide {
                    if y >= 0 {
                        let current_px = (x * CN as u32) as usize;

                        let prepared_px_s32 = _mm_cvtps_epi32(_mm_mul_ps(
                            _mm_cvtepi32_ps(summs),
                            _mm256_castps256_ps128(v_weight),
                        ));

                        let current_y = (y * (stride as i64)) as usize;

                        let bytes_offset = current_y + current_px;

                        let dst_ptr = bytes.get_ptr(bytes_offset);
                        store_u16_u32::<CN>(dst_ptr, prepared_px_s32);

                        let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                        let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                        let d_arr_index = (y & 1023) as usize;

                        let buf_ptr = buffer0.get_unchecked(d_arr_index).0.as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());

                        let buf_ptr_1 = buffer0.get_unchecked(d_arr_index_1..).as_ptr();
                        let stored_1 = _mm_load_si128(buf_ptr_1.cast());

                        let buf_ptr_2 = buffer0.get_unchecked(d_arr_index_2..).as_ptr();
                        let stored_2 = _mm_load_si128(buf_ptr_2.cast());

                        let new_diff = _mm_sub_epi32(
                            _mm_mul_by_3_epi32(_mm_sub_epi32(stored, stored_1)),
                            stored_2,
                        );
                        diffs = _mm_add_epi32(diffs, new_diff);
                    } else if y + radius_64 >= 0 {
                        let arr_index = (y & 1023) as usize;
                        let arr_index_1 = ((y + radius_64) & 1023) as usize;
                        let buf_ptr = buffer0.get_unchecked(arr_index).0.as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());

                        let buf_ptr_1 = buffer0.get_unchecked(arr_index_1).0.as_ptr();
                        let stored_1 = _mm_load_si128(buf_ptr_1.cast());

                        let new_diff = _mm_mul_by_3_epi32(_mm_sub_epi32(stored, stored_1));

                        diffs = _mm_add_epi32(diffs, new_diff);
                    } else if y + 2 * radius_64 >= 0 {
                        let arr_index = ((y + radius_64) & 1023) as usize;
                        let buf_ptr = buffer0.get_unchecked(arr_index).0.as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());
                        diffs = _mm_sub_epi32(diffs, _mm_mul_by_3_epi32(stored));
                    }

                    let next_row_y =
                        clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                            * (stride as usize);
                    let next_row_x = (x * CN as u32) as usize;

                    let pixel_color =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(next_row_y + next_row_x));

                    let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();

                    diffs = _mm_add_epi32(diffs, pixel_color);
                    ders = _mm_add_epi32(ders, diffs);
                    summs = _mm_add_epi32(summs, ders);
                    _mm_store_si128(buf_ptr.cast(), pixel_color);
                }
            }
        }
    }
}

pub(crate) fn fgn_horizontal_pass_avx_u16<const CN: usize>(
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
        let unit = HorizontalExecutionUnit::<CN>::default();
        unit.pass(bytes, stride, width, height, radius, start, end, edge_mode);
    }
}

#[derive(Copy, Clone, Default)]
struct HorizontalExecutionUnit<const CN: usize> {}

impl<const CN: usize> HorizontalExecutionUnit<CN> {
    #[target_feature(enable = "avx2")]
    fn pass(
        &self,
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
            let mut full_buffer = [AvxSseI32x8::default(); 1024 * 2];

            let (buffer0, buffer1) = full_buffer.split_at_mut(1024);

            let width_wide = width as i64;

            let radius_64 = radius as i64;

            let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
            let v_weight = _mm256_set1_ps(weight);

            let mut yy = start;

            while yy + 6 <= height.min(end) {
                let mut diffs0 = _mm256_setzero_si256();
                let mut diffs1 = _mm256_setzero_si256();

                let mut ders0 = _mm256_setzero_si256();
                let mut ders1 = _mm256_setzero_si256();

                let mut summs0 = _mm256_setzero_si256();
                let mut summs1 = _mm256_setzero_si256();

                let current_y0 = ((yy as i64) * (stride as i64)) as usize;
                let current_y1 = ((yy as i64 + 1) * (stride as i64)) as usize;
                let current_y2 = ((yy as i64 + 2) * (stride as i64)) as usize;
                let current_y3 = ((yy as i64 + 3) * (stride as i64)) as usize;

                for x in (0 - 3 * radius_64)..(width as i64) {
                    if x >= 0 {
                        let current_px = x as usize * CN;

                        let ps01 = _mm256_cvtepi32_ps(summs0);
                        let ps23 = _mm256_cvtepi32_ps(summs1);

                        let r01 = _mm256_mul_ps(ps01, v_weight);
                        let r23 = _mm256_mul_ps(ps23, v_weight);

                        let e01 = _mm256_cvtps_epi32(r01);
                        let e23 = _mm256_cvtps_epi32(r23);

                        let prepared_px0 = _mm256_castsi256_si128(e01);
                        let prepared_px1 = _mm256_extracti128_si256::<1>(e01);
                        let prepared_px2 = _mm256_castsi256_si128(e23);
                        let prepared_px3 = _mm256_extracti128_si256::<1>(e23);

                        let dst_ptr0 = bytes.get_ptr(current_y0 + current_px);
                        let dst_ptr1 = bytes.get_ptr(current_y1 + current_px);
                        let dst_ptr2 = bytes.get_ptr(current_y2 + current_px);
                        let dst_ptr3 = bytes.get_ptr(current_y3 + current_px);

                        store_u16_u32::<CN>(dst_ptr0, prepared_px0);
                        store_u16_u32::<CN>(dst_ptr1, prepared_px1);
                        store_u16_u32::<CN>(dst_ptr2, prepared_px2);
                        store_u16_u32::<CN>(dst_ptr3, prepared_px3);

                        let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                        let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                        let d_arr_index = (x & 1023) as usize;

                        let stored0 =
                            _mm256_load_si256(buffer0.get_unchecked(d_arr_index..).as_ptr().cast());
                        let stored1 =
                            _mm256_load_si256(buffer1.get_unchecked(d_arr_index..).as_ptr().cast());

                        let stored_10 = _mm256_load_si256(
                            buffer0.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                        );
                        let stored_11 = _mm256_load_si256(
                            buffer1.get_unchecked(d_arr_index_1..).as_ptr().cast(),
                        );

                        let j0 = _mm256_sub_epi32(stored0, stored_10);
                        let j1 = _mm256_sub_epi32(stored1, stored_11);

                        let stored_20 = _mm256_load_si256(
                            buffer0.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                        );
                        let stored_21 = _mm256_load_si256(
                            buffer1.get_unchecked(d_arr_index_2..).as_ptr().cast(),
                        );

                        let k0 = _mm256_mul_by_3_epi32(j0);
                        let k1 = _mm256_mul_by_3_epi32(j1);

                        let new_diff0 = _mm256_sub_epi32(k0, stored_20);
                        let new_diff1 = _mm256_sub_epi32(k1, stored_21);

                        diffs0 = _mm256_add_epi32(diffs0, new_diff0);
                        diffs1 = _mm256_add_epi32(diffs1, new_diff1);
                    } else if x + radius_64 >= 0 {
                        let arr_index = (x & 1023) as usize;
                        let arr_index_1 = ((x + radius_64) & 1023) as usize;

                        let stored0 =
                            _mm256_load_si256(buffer0.get_unchecked(arr_index).0.as_ptr().cast());
                        let stored1 =
                            _mm256_load_si256(buffer1.get_unchecked(arr_index).0.as_ptr().cast());

                        let stored_10 =
                            _mm256_load_si256(buffer0.get_unchecked(arr_index_1).0.as_ptr().cast());
                        let stored_11 =
                            _mm256_load_si256(buffer1.get_unchecked(arr_index_1).0.as_ptr().cast());

                        let new_diff0 = _mm256_mul_by_3_epi32(_mm256_sub_epi32(stored0, stored_10));
                        let new_diff1 = _mm256_mul_by_3_epi32(_mm256_sub_epi32(stored1, stored_11));

                        diffs0 = _mm256_add_epi32(diffs0, new_diff0);
                        diffs1 = _mm256_add_epi32(diffs1, new_diff1);
                    } else if x + 2 * radius_64 >= 0 {
                        let arr_index = ((x + radius_64) & 1023) as usize;
                        let stored0 =
                            _mm256_load_si256(buffer0.get_unchecked(arr_index).0.as_ptr().cast());
                        let stored1 =
                            _mm256_load_si256(buffer1.get_unchecked(arr_index).0.as_ptr().cast());

                        diffs0 = _mm256_sub_epi32(diffs0, _mm256_mul_by_3_epi32(stored0));
                        diffs1 = _mm256_sub_epi32(diffs1, _mm256_mul_by_3_epi32(stored1));
                    }

                    let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                    let next_row_px = next_row_x * CN;

                    let pixel_color0 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(current_y0 + next_row_px));
                    let pixel_color1 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(current_y1 + next_row_px));
                    let pixel_color2 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(current_y2 + next_row_px));
                    let pixel_color3 =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(current_y3 + next_row_px));

                    let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                    let buf_ptr0 = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                    let buf_ptr1 = buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr();

                    let px01 = _mm256_inserti128_si256::<1>(
                        _mm256_castsi128_si256(pixel_color0),
                        pixel_color1,
                    );
                    let px23 = _mm256_inserti128_si256::<1>(
                        _mm256_castsi128_si256(pixel_color2),
                        pixel_color3,
                    );

                    diffs0 = _mm256_add_epi32(diffs0, px01);
                    diffs1 = _mm256_add_epi32(diffs1, px23);

                    _mm256_store_si256(buf_ptr0.cast(), px01);
                    _mm256_store_si256(buf_ptr1.cast(), px23);

                    ders0 = _mm256_add_epi32(ders0, diffs0);
                    ders1 = _mm256_add_epi32(ders1, diffs1);

                    summs0 = _mm256_add_epi32(summs0, ders0);
                    summs1 = _mm256_add_epi32(summs1, ders1);
                }

                yy += 4;
            }

            for y in yy..height.min(end) {
                let mut diffs = _mm_setzero_si128();
                let mut ders = _mm_setzero_si128();
                let mut summs = _mm_setzero_si128();

                let current_y = ((y as i64) * (stride as i64)) as usize;

                for x in (0 - 3 * radius_64)..(width as i64) {
                    if x >= 0 {
                        let current_px = x as usize * CN;

                        let prepared_px_s32 = _mm_cvtps_epi32(_mm_mul_ps(
                            _mm_cvtepi32_ps(summs),
                            _mm256_castps256_ps128(v_weight),
                        ));

                        let bytes_offset = current_y + current_px;

                        let dst_ptr = bytes.get_ptr(bytes_offset);
                        store_u16_u32::<CN>(dst_ptr, prepared_px_s32);

                        let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                        let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                        let d_arr_index = (x & 1023) as usize;

                        let buf_ptr = buffer0.get_unchecked(d_arr_index).0.as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());

                        let buf_ptr_1 = buffer0.get_unchecked(d_arr_index_1).0.as_ptr();
                        let stored_1 = _mm_load_si128(buf_ptr_1.cast());

                        let buf_ptr_2 = buffer0.get_unchecked(d_arr_index_2).0.as_ptr();
                        let stored_2 = _mm_load_si128(buf_ptr_2.cast());

                        let new_diff = _mm_sub_epi32(
                            _mm_mul_by_3_epi32(_mm_sub_epi32(stored, stored_1)),
                            stored_2,
                        );
                        diffs = _mm_add_epi32(diffs, new_diff);
                    } else if x + radius_64 >= 0 {
                        let arr_index = (x & 1023) as usize;
                        let arr_index_1 = ((x + radius_64) & 1023) as usize;
                        let buf_ptr = buffer0.get_unchecked(arr_index..).as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());

                        let buf_ptr_1 = buffer0.get_unchecked(arr_index_1..).as_ptr();
                        let stored_1 = _mm_load_si128(buf_ptr_1.cast());

                        let new_diff = _mm_mul_by_3_epi32(_mm_sub_epi32(stored, stored_1));

                        diffs = _mm_add_epi32(diffs, new_diff);
                    } else if x + 2 * radius_64 >= 0 {
                        let arr_index = ((x + radius_64) & 1023) as usize;
                        let buf_ptr = buffer0.get_unchecked(arr_index..).as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());
                        diffs = _mm_sub_epi32(diffs, _mm_mul_by_3_epi32(stored));
                    }

                    let next_row_y = (y as usize) * (stride as usize);
                    let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                    let next_row_px = next_row_x * CN;

                    let pixel_color =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(next_row_y + next_row_px));

                    let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();

                    diffs = _mm_add_epi32(diffs, pixel_color);
                    ders = _mm_add_epi32(ders, diffs);
                    summs = _mm_add_epi32(summs, ders);
                    _mm_store_si128(buf_ptr.cast(), pixel_color);
                }
            }
        }
    }
}
