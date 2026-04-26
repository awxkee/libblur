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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::EdgeMode;
use crate::edge_mode::clamp_edge;
use crate::sse::{load_f32_f16, store_f32_f16};
use crate::unsafe_slice::UnsafeSlice;
use core::f16;

pub(crate) fn fast_gaussian_next_vertical_pass_sse_f16<const CN: usize>(
    slice: &UnsafeSlice<f16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        fast_gaussian_next_vertical_pass_sse_f16_impl::<CN>(
            slice, stride, width, height, radius, start, end, edge_mode,
        );
    }
}

#[target_feature(enable = "sse4.1", enable = "f16c")]
fn fast_gaussian_next_vertical_pass_sse_f16_impl<const CN: usize>(
    bytes: &UnsafeSlice<f16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let mut buffer: [[f32; 4]; 1024] = [[0.; 4]; 1024];

        let height_wide = height as i64;

        let threes = _mm_set1_ps(3.);

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let v_weight = _mm_set1_ps(weight);
        for x in start..std::cmp::min(width, end) {
            let mut diffs = _mm_setzero_ps();
            let mut ders = _mm_setzero_ps();
            let mut summs = _mm_setzero_ps();

            let current_px = (x * CN as u32) as usize;

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * (stride as i64)) as usize;
                    let bytes_offset = current_y + current_px;

                    let pixel = _mm_mul_ps(summs, v_weight);
                    store_f32_f16::<CN>(bytes.get_ptr(bytes_offset), pixel);

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let buf_ptr = buffer.get_unchecked(d_arr_index).as_ptr();
                    let stored = _mm_loadu_ps(buf_ptr);

                    let buf_ptr_1 = buffer.get_unchecked(d_arr_index_1).as_ptr();
                    let stored_1 = _mm_loadu_ps(buf_ptr_1);

                    let buf_ptr_2 = buffer.get_unchecked(d_arr_index_2).as_ptr();
                    let stored_2 = _mm_loadu_ps(buf_ptr_2);

                    let new_diff =
                        _mm_sub_ps(_mm_mul_ps(_mm_sub_ps(stored, stored_1), threes), stored_2);
                    diffs = _mm_add_ps(diffs, new_diff);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer.get_unchecked(arr_index).as_ptr();
                    let stored = _mm_loadu_ps(buf_ptr);

                    let buf_ptr_1 = buffer.get_unchecked(arr_index_1).as_ptr();
                    let stored_1 = _mm_loadu_ps(buf_ptr_1);

                    let new_diff = _mm_mul_ps(_mm_sub_ps(stored, stored_1), threes);

                    diffs = _mm_add_ps(diffs, new_diff);
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer.get_unchecked(arr_index).as_ptr();
                    let stored = _mm_loadu_ps(buf_ptr);
                    diffs = _mm_sub_ps(diffs, _mm_mul_ps(stored, threes));
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * (stride as usize);
                let next_row_x = (x * CN as u32) as usize;

                let s_ptr = bytes.get_ptr(next_row_y + next_row_x);

                let pixel_color = load_f32_f16::<CN>(s_ptr);

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer.get_unchecked_mut(arr_index).as_mut_ptr();

                diffs = _mm_add_ps(diffs, pixel_color);
                ders = _mm_add_ps(ders, diffs);
                summs = _mm_add_ps(summs, ders);
                _mm_storeu_ps(buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fast_gaussian_next_horizontal_pass_sse_f16<const CN: usize>(
    slice: &UnsafeSlice<f16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        fast_gaussian_next_horizontal_pass_sse_f16_impl::<CN>(
            slice, stride, width, height, radius, start, end, edge_mode,
        );
    }
}

#[target_feature(enable = "sse4.1", enable = "f16c")]
fn fast_gaussian_next_horizontal_pass_sse_f16_impl<const CN: usize>(
    undefined_slice: &UnsafeSlice<f16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        let bytes: &UnsafeSlice<'_, f16> = std::mem::transmute(undefined_slice);
        let mut buffer: [[f32; 4]; 1024] = [[0.; 4]; 1024];

        let width_wide = width as i64;

        let threes = _mm_set1_ps(3.);

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let v_weight = _mm_set1_ps(weight);
        for y in start..std::cmp::min(height, end) {
            let mut diffs = _mm_setzero_ps();
            let mut ders = _mm_setzero_ps();
            let mut summs = _mm_setzero_ps();

            let current_y = ((y as i64) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let bytes_offset = current_y + current_px;

                    let pixel = _mm_mul_ps(summs, v_weight);
                    store_f32_f16::<CN>(bytes.get_ptr(bytes_offset), pixel);

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let buf_ptr = buffer.get_unchecked(d_arr_index).as_ptr();
                    let stored = _mm_loadu_ps(buf_ptr);

                    let buf_ptr_1 = buffer.get_unchecked(d_arr_index_1).as_ptr();
                    let stored_1 = _mm_loadu_ps(buf_ptr_1);

                    let buf_ptr_2 = buffer.get_unchecked(d_arr_index_2).as_ptr();
                    let stored_2 = _mm_loadu_ps(buf_ptr_2);

                    let new_diff =
                        _mm_sub_ps(_mm_mul_ps(_mm_sub_ps(stored, stored_1), threes), stored_2);
                    diffs = _mm_add_ps(diffs, new_diff);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer.get_unchecked(arr_index).as_ptr();
                    let stored = _mm_loadu_ps(buf_ptr);

                    let buf_ptr_1 = buffer.get_unchecked(arr_index_1).as_ptr();
                    let stored_1 = _mm_loadu_ps(buf_ptr_1);

                    let new_diff = _mm_mul_ps(_mm_sub_ps(stored, stored_1), threes);

                    diffs = _mm_add_ps(diffs, new_diff);
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer.get_unchecked(arr_index).as_ptr();
                    let stored = _mm_loadu_ps(buf_ptr);
                    diffs = _mm_sub_ps(diffs, _mm_mul_ps(stored, threes));
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr = bytes.get_ptr(next_row_y + next_row_px);

                let pixel_color = load_f32_f16::<CN>(s_ptr);

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer.get_unchecked_mut(arr_index).as_mut_ptr();

                diffs = _mm_add_ps(diffs, pixel_color);
                ders = _mm_add_ps(ders, diffs);
                summs = _mm_add_ps(summs, ders);
                _mm_storeu_ps(buf_ptr, pixel_color);
            }
        }
    }
}
