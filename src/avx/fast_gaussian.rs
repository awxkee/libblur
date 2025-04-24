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

use crate::avx::fast_gaussian_next::AvxSseI32x8;
use crate::reflect_index;
use crate::sse::store_u8_u32;
use crate::sse::utils::load_u8_s32_fast;
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, EdgeMode};
use std::arch::x86_64::*;

pub(crate) fn fg_horizontal_pass_sse_u8<T, const CN: usize>(
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
        fg_horizontal_pass_avx_def::<CN>(
            bytes, stride, width, height, radius, start, end, edge_mode,
        );
    }
}

#[target_feature(enable = "avx2")]
unsafe fn fg_horizontal_pass_avx_def<const CN: usize>(
    bytes: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    fg_horizontal_pass_avx_u8_impl::<CN>(
        bytes, stride, width, height, radius, start, end, edge_mode,
    );
}

#[inline(always)]
unsafe fn fg_horizontal_pass_avx_u8_impl<const CN: usize>(
    bytes: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    let mut buffer0 = Box::new([AvxSseI32x8::default(); 1024]);
    let mut buffer1 = Box::new([AvxSseI32x8::default(); 1024]);
    let mut buffer2 = Box::new([AvxSseI32x8::default(); 1024]);

    let initial_sum = ((radius * radius) >> 1) as i32;

    let radius_64 = radius as i64;
    let width_wide = width as i64;

    let v_weight = _mm256_set1_ps(1f32 / (radius as f32 * radius as f32));

    let mut yy = start;

    while yy + 6 < height.min(end) {
        let mut diffs0 = _mm256_setzero_si256();
        let mut diffs1 = _mm256_setzero_si256();
        let mut diffs2 = _mm256_setzero_si256();

        let mut summs0 = _mm256_set1_epi32(initial_sum);
        let mut summs1 = _mm256_set1_epi32(initial_sum);
        let mut summs2 = _mm256_set1_epi32(initial_sum);

        let current_y0 = ((yy as i64) * (stride as i64)) as usize;
        let current_y1 = ((yy as i64 + 1) * (stride as i64)) as usize;
        let current_y2 = ((yy as i64 + 2) * (stride as i64)) as usize;
        let current_y3 = ((yy as i64 + 3) * (stride as i64)) as usize;
        let current_y4 = ((yy as i64 + 4) * (stride as i64)) as usize;
        let current_y5 = ((yy as i64 + 5) * (stride as i64)) as usize;

        let start_x = 0 - 2 * radius_64;
        for x in start_x..(width as i64) {
            if x >= 0 {
                let current_px = (x as u32 * CN as u32) as usize;

                let ps01 = _mm256_cvtepi32_ps(summs0);
                let ps23 = _mm256_cvtepi32_ps(summs1);
                let ps45 = _mm256_cvtepi32_ps(summs1);

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

                let dst_ptr0 = (bytes.slice.as_ptr() as *mut u8).add(current_y0 + current_px);
                let dst_ptr1 = (bytes.slice.as_ptr() as *mut u8).add(current_y1 + current_px);
                let dst_ptr2 = (bytes.slice.as_ptr() as *mut u8).add(current_y2 + current_px);
                let dst_ptr3 = (bytes.slice.as_ptr() as *mut u8).add(current_y3 + current_px);
                let dst_ptr4 = (bytes.slice.as_ptr() as *mut u8).add(current_y4 + current_px);
                let dst_ptr5 = (bytes.slice.as_ptr() as *mut u8).add(current_y5 + current_px);

                store_u8_u32::<CN>(dst_ptr0, prepared_px0);
                store_u8_u32::<CN>(dst_ptr1, prepared_px1);
                store_u8_u32::<CN>(dst_ptr2, prepared_px2);
                store_u8_u32::<CN>(dst_ptr3, prepared_px3);
                store_u8_u32::<CN>(dst_ptr4, prepared_px4);
                store_u8_u32::<CN>(dst_ptr5, prepared_px5);

                let arr_index = ((x - radius_64) & 1023) as usize;
                let d_arr_index = (x & 1023) as usize;

                let mut d_stored0 = _mm256_load_si256(
                    buffer0.get_unchecked_mut(d_arr_index).0.as_mut_ptr() as *const _,
                );
                let mut d_stored1 = _mm256_load_si256(
                    buffer1.get_unchecked_mut(d_arr_index).0.as_mut_ptr() as *const _,
                );
                let mut d_stored2 = _mm256_load_si256(
                    buffer2.get_unchecked_mut(d_arr_index).0.as_mut_ptr() as *const _,
                );

                d_stored0 = _mm256_slli_epi32::<1>(d_stored0);
                d_stored1 = _mm256_slli_epi32::<1>(d_stored1);
                d_stored2 = _mm256_slli_epi32::<1>(d_stored2);

                let a_stored0 = _mm256_load_si256(
                    buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr() as *const _,
                );
                let a_stored1 = _mm256_load_si256(
                    buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr() as *const _,
                );
                let a_stored2 = _mm256_load_si256(
                    buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr() as *const _,
                );

                diffs0 = _mm256_add_epi32(diffs0, _mm256_sub_epi32(a_stored0, d_stored0));
                diffs1 = _mm256_add_epi32(diffs1, _mm256_sub_epi32(a_stored1, d_stored1));
                diffs2 = _mm256_add_epi32(diffs2, _mm256_sub_epi32(a_stored2, d_stored2));
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                let mut stored0 = _mm256_load_si256(
                    buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr() as *const _,
                );
                let mut stored1 = _mm256_load_si256(
                    buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr() as *const _,
                );
                let mut stored2 = _mm256_load_si256(
                    buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr() as *const _,
                );

                stored0 = _mm256_slli_epi32::<1>(stored0);
                stored1 = _mm256_slli_epi32::<1>(stored1);
                stored2 = _mm256_slli_epi32::<1>(stored2);

                diffs0 = _mm256_sub_epi32(diffs0, stored0);
                diffs1 = _mm256_sub_epi32(diffs1, stored1);
                diffs2 = _mm256_sub_epi32(diffs2, stored2);
            }

            let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
            let next_row_px = next_row_x * CN;

            let s_ptr0 = bytes.slice.as_ptr().add(current_y0 + next_row_px) as *mut u8;
            let s_ptr1 = bytes.slice.as_ptr().add(current_y1 + next_row_px) as *mut u8;
            let s_ptr2 = bytes.slice.as_ptr().add(current_y2 + next_row_px) as *mut u8;
            let s_ptr3 = bytes.slice.as_ptr().add(current_y3 + next_row_px) as *mut u8;
            let s_ptr4 = bytes.slice.as_ptr().add(current_y4 + next_row_px) as *mut u8;
            let s_ptr5 = bytes.slice.as_ptr().add(current_y5 + next_row_px) as *mut u8;

            let pixel_color0 = load_u8_s32_fast::<CN>(s_ptr0);
            let pixel_color1 = load_u8_s32_fast::<CN>(s_ptr1);
            let pixel_color2 = load_u8_s32_fast::<CN>(s_ptr2);
            let pixel_color3 = load_u8_s32_fast::<CN>(s_ptr3);
            let pixel_color4 = load_u8_s32_fast::<CN>(s_ptr4);
            let pixel_color5 = load_u8_s32_fast::<CN>(s_ptr5);

            let arr_index = ((x + radius_64) & 1023) as usize;

            let px01 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(pixel_color0), pixel_color1);
            let px23 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(pixel_color2), pixel_color3);
            let px45 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(pixel_color4), pixel_color5);

            _mm256_store_si256(
                buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _,
                px01,
            );
            _mm256_store_si256(
                buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _,
                px23,
            );
            _mm256_store_si256(
                buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _,
                px45,
            );

            diffs0 = _mm256_add_epi32(diffs0, px01);
            diffs1 = _mm256_add_epi32(diffs1, px23);
            diffs2 = _mm256_add_epi32(diffs2, px45);

            summs0 = _mm256_add_epi32(summs0, diffs0);
            summs1 = _mm256_add_epi32(summs1, diffs1);
            summs2 = _mm256_add_epi32(summs2, diffs2);
        }

        yy += 6;
    }

    for y in yy..height.min(end) {
        let mut diffs = _mm_setzero_si128();
        let mut summs = _mm_set1_epi32(initial_sum);

        let current_y = ((y as i64) * (stride as i64)) as usize;

        let start_x = 0 - 2 * radius_64;
        for x in start_x..(width as i64) {
            if x >= 0 {
                let current_px = (x as u32 * CN as u32) as usize;

                let pixel_f32 =
                    _mm_mul_ps(_mm_cvtepi32_ps(summs), _mm256_castps256_ps128(v_weight));
                let pixel_u32 = _mm_cvtps_epi32(pixel_f32);

                let bytes_offset = current_y + current_px;

                let dst_ptr = (bytes.slice.as_ptr() as *mut u8).add(bytes_offset);
                store_u8_u32::<CN>(dst_ptr, pixel_u32);

                let arr_index = ((x - radius_64) & 1023) as usize;
                let d_arr_index = (x & 1023) as usize;

                let d_buf_ptr = buffer0.as_mut_ptr().add(d_arr_index) as *const i32;
                let mut d_stored = _mm_load_si128(d_buf_ptr as *const __m128i);
                d_stored = _mm_slli_epi32::<1>(d_stored);

                let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                let a_stored = _mm_load_si128(buf_ptr as *const __m128i);

                diffs = _mm_add_epi32(diffs, _mm_sub_epi32(a_stored, d_stored));
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                let mut stored = _mm_load_si128(buf_ptr as *const __m128i);
                stored = _mm_slli_epi32::<1>(stored);
                diffs = _mm_sub_epi32(diffs, stored);
            }

            let next_row_y = (y as usize) * (stride as usize);
            let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
            let next_row_px = next_row_x * CN;

            let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut u8;
            let pixel_color = load_u8_s32_fast::<CN>(s_ptr);

            let arr_index = ((x + radius_64) & 1023) as usize;
            let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();

            diffs = _mm_add_epi32(diffs, pixel_color);
            summs = _mm_add_epi32(summs, diffs);

            _mm_store_si128(buf_ptr as *mut __m128i, pixel_color);
        }
    }
}

pub(crate) fn fg_vertical_pass_avx_u8<T, const CN: usize>(
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
        fg_vertical_pass_avx_u8_def::<CN>(
            bytes, stride, width, height, radius, start, end, edge_mode,
        );
    }
}

#[target_feature(enable = "avx2")]
unsafe fn fg_vertical_pass_avx_u8_def<const CN: usize>(
    bytes: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    let mut buffer0 = Box::new([AvxSseI32x8::default(); 1024]);
    let mut buffer1 = Box::new([AvxSseI32x8::default(); 1024]);
    let mut buffer2 = Box::new([AvxSseI32x8::default(); 1024]);

    let initial_sum = ((radius * radius) >> 1) as i32;

    let height_wide = height as i64;

    let radius_64 = radius as i64;

    let v_weight = _mm256_set1_ps(1f32 / (radius as f32 * radius as f32));

    let mut xx = start;

    while xx + 6 < width.min(end) {
        let mut diffs0 = _mm256_setzero_si256();
        let mut diffs1 = _mm256_setzero_si256();
        let mut diffs2 = _mm256_setzero_si256();

        let mut summs0 = _mm256_set1_epi32(initial_sum);
        let mut summs1 = _mm256_set1_epi32(initial_sum);
        let mut summs2 = _mm256_set1_epi32(initial_sum);

        let start_y = 0 - 2 * radius as i64;

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

                let dst_ptr0 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px0);
                let dst_ptr1 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px1);
                let dst_ptr2 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px2);
                let dst_ptr3 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px3);
                let dst_ptr4 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px4);
                let dst_ptr5 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px5);

                store_u8_u32::<CN>(dst_ptr0, prepared_px0);
                store_u8_u32::<CN>(dst_ptr1, prepared_px1);
                store_u8_u32::<CN>(dst_ptr2, prepared_px2);
                store_u8_u32::<CN>(dst_ptr3, prepared_px3);
                store_u8_u32::<CN>(dst_ptr4, prepared_px4);
                store_u8_u32::<CN>(dst_ptr5, prepared_px5);

                let arr_index = ((y - radius_64) & 1023) as usize;
                let d_arr_index = (y & 1023) as usize;

                let mut d_stored0 = _mm256_load_si256(
                    buffer0.get_unchecked_mut(d_arr_index).0.as_mut_ptr() as *const _,
                );
                let mut d_stored1 = _mm256_load_si256(
                    buffer1.get_unchecked_mut(d_arr_index).0.as_mut_ptr() as *const _,
                );
                let mut d_stored2 = _mm256_load_si256(
                    buffer2.get_unchecked_mut(d_arr_index).0.as_mut_ptr() as *const _,
                );

                d_stored0 = _mm256_slli_epi32::<1>(d_stored0);
                d_stored1 = _mm256_slli_epi32::<1>(d_stored1);
                d_stored2 = _mm256_slli_epi32::<1>(d_stored2);

                let a_stored0 = _mm256_load_si256(
                    buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr() as *const _,
                );
                let a_stored1 = _mm256_load_si256(
                    buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr() as *const _,
                );
                let a_stored2 = _mm256_load_si256(
                    buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr() as *const _,
                );

                diffs0 = _mm256_add_epi32(diffs0, _mm256_sub_epi32(a_stored0, d_stored0));
                diffs1 = _mm256_add_epi32(diffs1, _mm256_sub_epi32(a_stored1, d_stored1));
                diffs2 = _mm256_add_epi32(diffs2, _mm256_sub_epi32(a_stored2, d_stored2));
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;

                let mut stored0 = _mm256_load_si256(
                    buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _,
                );
                let mut stored1 = _mm256_load_si256(
                    buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _,
                );
                let mut stored2 = _mm256_load_si256(
                    buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _,
                );

                stored0 = _mm256_slli_epi32::<1>(stored0);
                stored1 = _mm256_slli_epi32::<1>(stored1);
                stored2 = _mm256_slli_epi32::<1>(stored2);

                diffs0 = _mm256_sub_epi32(diffs0, stored0);
                diffs1 = _mm256_sub_epi32(diffs1, stored1);
                diffs2 = _mm256_sub_epi32(diffs2, stored2);
            }

            let next_row_y =
                clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * (stride as usize);

            let s_ptr0 = bytes.slice.as_ptr().add(next_row_y + current_px0) as *mut u8;
            let s_ptr1 = bytes.slice.as_ptr().add(next_row_y + current_px1) as *mut u8;
            let s_ptr2 = bytes.slice.as_ptr().add(next_row_y + current_px2) as *mut u8;
            let s_ptr3 = bytes.slice.as_ptr().add(next_row_y + current_px3) as *mut u8;
            let s_ptr4 = bytes.slice.as_ptr().add(next_row_y + current_px4) as *mut u8;
            let s_ptr5 = bytes.slice.as_ptr().add(next_row_y + current_px5) as *mut u8;

            let pixel_color0 = load_u8_s32_fast::<CN>(s_ptr0);
            let pixel_color1 = load_u8_s32_fast::<CN>(s_ptr1);
            let pixel_color2 = load_u8_s32_fast::<CN>(s_ptr2);
            let pixel_color3 = load_u8_s32_fast::<CN>(s_ptr3);
            let pixel_color4 = load_u8_s32_fast::<CN>(s_ptr4);
            let pixel_color5 = load_u8_s32_fast::<CN>(s_ptr5);

            let arr_index = ((y + radius_64) & 1023) as usize;

            let px01 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(pixel_color0), pixel_color1);
            let px23 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(pixel_color2), pixel_color3);
            let px45 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(pixel_color4), pixel_color5);

            diffs0 = _mm256_add_epi32(diffs0, px01);
            diffs1 = _mm256_add_epi32(diffs1, px23);
            diffs2 = _mm256_add_epi32(diffs2, px45);

            _mm256_store_si256(
                buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _,
                px01,
            );
            _mm256_store_si256(
                buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _,
                px23,
            );
            _mm256_store_si256(
                buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _,
                px45,
            );

            summs0 = _mm256_add_epi32(summs0, diffs0);
            summs1 = _mm256_add_epi32(summs1, diffs1);
            summs2 = _mm256_add_epi32(summs2, diffs2);
        }

        xx += 6;
    }

    for x in xx..width.min(end) {
        let mut diffs = _mm_setzero_si128();
        let mut summs = _mm_set1_epi32(initial_sum);

        let current_px = (x * CN as u32) as usize;

        let start_y = 0 - 2 * radius as i64;
        for y in start_y..height_wide {
            if y >= 0 {
                let pixel_f32 =
                    _mm_mul_ps(_mm_cvtepi32_ps(summs), _mm256_castps256_ps128(v_weight));

                let current_y = (y * (stride as i64)) as usize;

                let pixel_u32 = _mm_cvtps_epi32(pixel_f32);

                let bytes_offset = current_y + current_px;

                let dst_ptr = (bytes.slice.as_ptr() as *mut u8).add(bytes_offset);
                store_u8_u32::<CN>(dst_ptr, pixel_u32);

                let arr_index = ((y - radius_64) & 1023) as usize;
                let d_arr_index = (y & 1023) as usize;

                let d_buf_ptr = buffer0.get_unchecked_mut(d_arr_index).0.as_mut_ptr();
                let mut d_stored = _mm_load_si128(d_buf_ptr as *const __m128i);
                d_stored = _mm_slli_epi32::<1>(d_stored);

                let buf_ptr = buffer0.as_mut_ptr().add(arr_index) as *mut i32;
                let a_stored = _mm_load_si128(buf_ptr as *const __m128i);

                diffs = _mm_add_epi32(diffs, _mm_sub_epi32(a_stored, d_stored));
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;
                let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                let mut stored = _mm_load_si128(buf_ptr as *const __m128i);
                stored = _mm_slli_epi32::<1>(stored);
                diffs = _mm_sub_epi32(diffs, stored);
            }

            let next_row_y =
                clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * (stride as usize);
            let next_row_x = (x * CN as u32) as usize;

            let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut u8;
            let pixel_color = load_u8_s32_fast::<CN>(s_ptr);

            let arr_index = ((y + radius_64) & 1023) as usize;
            let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr();

            diffs = _mm_add_epi32(diffs, pixel_color);

            _mm_store_si128(buf_ptr as *mut __m128i, pixel_color);

            summs = _mm_add_epi32(summs, diffs);
        }
    }
}
