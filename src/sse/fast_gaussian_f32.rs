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

use crate::reflect_index;
use crate::sse::{_mm_opt_fnmlaf_ps, load_f32, store_f32};
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, EdgeMode};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn fg_horizontal_pass_sse_f32<T, const CN: usize>(
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
        let bytes: &UnsafeSlice<'_, f32> = std::mem::transmute(undefined_slice);
        if std::arch::is_x86_feature_detected!("fma") {
            fg_horizontal_pass_sse_f32_fma::<CN>(
                bytes, stride, width, height, radius, start, end, edge_mode,
            );
        } else {
            fg_horizontal_pass_sse_f32_def::<CN>(
                bytes, stride, width, height, radius, start, end, edge_mode,
            );
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn fg_horizontal_pass_sse_f32_def<const CN: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    fg_horizontal_pass_sse_f32_impl::<CN, false>(
        bytes, stride, width, height, radius, start, end, edge_mode,
    );
}

#[target_feature(enable = "sse4.1", enable = "fma")]
unsafe fn fg_horizontal_pass_sse_f32_fma<const CN: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    fg_horizontal_pass_sse_f32_impl::<CN, true>(
        bytes, stride, width, height, radius, start, end, edge_mode,
    );
}

#[inline(always)]
unsafe fn fg_horizontal_pass_sse_f32_impl<const CN: usize, const FMA: bool>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    let mut bf0 = Box::new([[0f32; 4]; 1024]);
    let mut bf1 = Box::new([[0f32; 4]; 1024]);
    let mut bf2 = Box::new([[0f32; 4]; 1024]);
    let mut bf3 = Box::new([[0f32; 4]; 1024]);

    let radius_64 = radius as i64;
    let width_wide = width as i64;

    let v_double = _mm_set1_ps(2.);
    let v_weight = _mm_set1_ps(1f32 / (radius as f32 * radius as f32));

    let mut yy = start as usize;

    while yy + 4 < height.min(end) as usize {
        let mut diffs0 = _mm_setzero_ps();
        let mut diffs1 = _mm_setzero_ps();
        let mut diffs2 = _mm_setzero_ps();
        let mut diffs3 = _mm_setzero_ps();

        let mut sums0 = _mm_setzero_ps();
        let mut sums1 = _mm_setzero_ps();
        let mut sums2 = _mm_setzero_ps();
        let mut sums3 = _mm_setzero_ps();

        let current_y0 = ((yy as i64) * (stride as i64)) as usize;
        let current_y1 = ((yy as i64 + 1) * (stride as i64)) as usize;
        let current_y2 = ((yy as i64 + 2) * (stride as i64)) as usize;
        let current_y3 = ((yy as i64 + 3) * (stride as i64)) as usize;

        let start_x = 0 - 2 * radius_64;
        for x in start_x..(width as i64) {
            if x >= 0 {
                let current_px = x as usize * CN;

                let prepared_px0 = _mm_mul_ps(sums0, v_weight);
                let prepared_px1 = _mm_mul_ps(sums1, v_weight);
                let prepared_px2 = _mm_mul_ps(sums2, v_weight);
                let prepared_px3 = _mm_mul_ps(sums3, v_weight);

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

                let d_s0 = _mm_loadu_ps(bf0.as_mut_ptr().add(d_arr_index) as *mut f32);
                let d_s1 = _mm_loadu_ps(bf1.as_mut_ptr().add(d_arr_index) as *mut f32);
                let d_s2 = _mm_loadu_ps(bf2.as_mut_ptr().add(d_arr_index) as *mut f32);
                let d_s3 = _mm_loadu_ps(bf3.as_mut_ptr().add(d_arr_index) as *mut f32);

                let a_s0 = _mm_loadu_ps(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                let a_s1 = _mm_loadu_ps(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                let a_s2 = _mm_loadu_ps(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                let a_s3 = _mm_loadu_ps(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                diffs0 = _mm_add_ps(diffs0, _mm_opt_fnmlaf_ps::<FMA>(a_s0, d_s0, v_double));
                diffs1 = _mm_add_ps(diffs1, _mm_opt_fnmlaf_ps::<FMA>(a_s1, d_s1, v_double));
                diffs2 = _mm_add_ps(diffs2, _mm_opt_fnmlaf_ps::<FMA>(a_s2, d_s2, v_double));
                diffs3 = _mm_add_ps(diffs3, _mm_opt_fnmlaf_ps::<FMA>(a_s3, d_s3, v_double));
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                let s0 = _mm_loadu_ps(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                let s1 = _mm_loadu_ps(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                let s2 = _mm_loadu_ps(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                let s3 = _mm_loadu_ps(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                diffs0 = _mm_opt_fnmlaf_ps::<FMA>(diffs0, s0, v_double);
                diffs1 = _mm_opt_fnmlaf_ps::<FMA>(diffs1, s1, v_double);
                diffs2 = _mm_opt_fnmlaf_ps::<FMA>(diffs2, s2, v_double);
                diffs3 = _mm_opt_fnmlaf_ps::<FMA>(diffs3, s3, v_double);
            }

            let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
            let next_row_px = next_row_x * CN;

            let s_ptr0 = bytes.slice.as_ptr().add(current_y0 + next_row_px) as *mut f32;
            let s_ptr1 = bytes.slice.as_ptr().add(current_y1 + next_row_px) as *mut f32;
            let s_ptr2 = bytes.slice.as_ptr().add(current_y2 + next_row_px) as *mut f32;
            let s_ptr3 = bytes.slice.as_ptr().add(current_y3 + next_row_px) as *mut f32;

            let px0 = load_f32::<CN>(s_ptr0);
            let px1 = load_f32::<CN>(s_ptr1);
            let px2 = load_f32::<CN>(s_ptr2);
            let px3 = load_f32::<CN>(s_ptr3);

            let arr_index = ((x + radius_64) & 1023) as usize;

            _mm_storeu_ps(bf0.as_mut_ptr().add(arr_index) as *mut f32, px0);
            _mm_storeu_ps(bf1.as_mut_ptr().add(arr_index) as *mut f32, px1);
            _mm_storeu_ps(bf2.as_mut_ptr().add(arr_index) as *mut f32, px2);
            _mm_storeu_ps(bf3.as_mut_ptr().add(arr_index) as *mut f32, px3);

            diffs0 = _mm_add_ps(diffs0, px0);
            diffs1 = _mm_add_ps(diffs1, px1);
            diffs2 = _mm_add_ps(diffs2, px2);
            diffs3 = _mm_add_ps(diffs3, px3);

            sums0 = _mm_add_ps(sums0, diffs0);
            sums1 = _mm_add_ps(sums1, diffs1);
            sums2 = _mm_add_ps(sums2, diffs2);
            sums3 = _mm_add_ps(sums3, diffs3);
        }

        yy += 4;
    }

    for y in yy..height.min(end) as usize {
        let mut diffs = _mm_setzero_ps();
        let mut summs = _mm_setzero_ps();

        let current_y = ((y as i64) * (stride as i64)) as usize;

        let start_x = 0 - 2 * radius_64;
        for x in start_x..(width as i64) {
            if x >= 0 {
                let current_px = x as usize * CN;

                let pixel = _mm_mul_ps(summs, v_weight);

                let bytes_offset = current_y + current_px;

                let dst_ptr = bytes.slice.as_ptr().add(bytes_offset) as *mut f32;
                store_f32::<CN>(dst_ptr, pixel);

                let arr_index = ((x - radius_64) & 1023) as usize;
                let d_arr_index = (x & 1023) as usize;

                let d_buf_ptr = bf0.get_unchecked_mut(d_arr_index).as_mut_ptr();
                let d_stored = _mm_loadu_ps(d_buf_ptr);

                let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                let a_stored = _mm_loadu_ps(buf_ptr);

                diffs = _mm_add_ps(
                    diffs,
                    _mm_opt_fnmlaf_ps::<FMA>(a_stored, d_stored, v_double),
                );
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                let buf_ptr = bf0.get_unchecked_mut(arr_index).as_mut_ptr();
                let stored = _mm_loadu_ps(buf_ptr);
                diffs = _mm_opt_fnmlaf_ps::<FMA>(diffs, stored, v_double);
            }

            let next_row_y = y * (stride as usize);
            let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide);
            let next_row_px = next_row_x * CN;

            let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut f32;
            let pixel_color = load_f32::<CN>(s_ptr);

            let arr_index = ((x + radius_64) & 1023) as usize;
            let buf_ptr = bf0.get_unchecked_mut(arr_index).as_mut_ptr();

            diffs = _mm_add_ps(diffs, pixel_color);
            summs = _mm_add_ps(summs, diffs);
            _mm_storeu_ps(buf_ptr, pixel_color);
        }
    }
}

pub(crate) fn fg_vertical_pass_sse_f32<T, const CN: usize>(
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
        let bytes: &UnsafeSlice<'_, f32> = std::mem::transmute(undefined_slice);

        if std::arch::is_x86_feature_detected!("fma") {
            fg_vertical_pass_sse_f32_fma::<CN>(
                bytes, stride, width, height, radius, start, end, edge_mode,
            );
        } else {
            fg_vertical_pass_sse_f32_def::<CN>(
                bytes, stride, width, height, radius, start, end, edge_mode,
            );
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn fg_vertical_pass_sse_f32_def<const CN: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    fg_vertical_pass_sse_f32_impl::<CN, false>(
        bytes, stride, width, height, radius, start, end, edge_mode,
    );
}

#[target_feature(enable = "sse4.1", enable = "fma")]
unsafe fn fg_vertical_pass_sse_f32_fma<const CN: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    fg_vertical_pass_sse_f32_impl::<CN, true>(
        bytes, stride, width, height, radius, start, end, edge_mode,
    );
}

#[inline(always)]
unsafe fn fg_vertical_pass_sse_f32_impl<const CN: usize, const FMA: bool>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    let mut bf0 = Box::new([[0f32; 4]; 1024]);
    let mut bf1 = Box::new([[0f32; 4]; 1024]);
    let mut bf2 = Box::new([[0f32; 4]; 1024]);
    let mut bf3 = Box::new([[0f32; 4]; 1024]);

    let v_double = _mm_set1_ps(2.);
    let v_weight = _mm_set1_ps(1f32 / (radius as f32 * radius as f32));

    let height_wide = height as i64;

    let radius_64 = radius as i64;

    let mut xx = start as usize;

    while xx + 4 < width.min(end) as usize {
        let mut diffs0 = _mm_setzero_ps();
        let mut diffs1 = _mm_setzero_ps();
        let mut diffs2 = _mm_setzero_ps();
        let mut diffs3 = _mm_setzero_ps();

        let mut sums0 = _mm_setzero_ps();
        let mut sums1 = _mm_setzero_ps();
        let mut sums2 = _mm_setzero_ps();
        let mut sums3 = _mm_setzero_ps();

        let current_px0 = xx * CN;
        let current_px1 = (xx + 1) * CN;
        let current_px2 = (xx + 2) * CN;
        let current_px3 = (xx + 3) * CN;

        let start_y = 0 - 2 * radius as i64;
        for y in start_y..height_wide {
            if y >= 0 {
                let current_y = (y * (stride as i64)) as usize;

                let prepared_px0 = _mm_mul_ps(sums0, v_weight);
                let prepared_px1 = _mm_mul_ps(sums1, v_weight);
                let prepared_px2 = _mm_mul_ps(sums2, v_weight);
                let prepared_px3 = _mm_mul_ps(sums3, v_weight);

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

                let d_s0 = _mm_loadu_ps(bf0.as_mut_ptr().add(d_arr_index) as *mut f32);
                let d_s1 = _mm_loadu_ps(bf1.as_mut_ptr().add(d_arr_index) as *mut f32);
                let d_s2 = _mm_loadu_ps(bf2.as_mut_ptr().add(d_arr_index) as *mut f32);
                let d_s3 = _mm_loadu_ps(bf3.as_mut_ptr().add(d_arr_index) as *mut f32);

                let a_s0 = _mm_loadu_ps(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                let a_s1 = _mm_loadu_ps(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                let a_s2 = _mm_loadu_ps(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                let a_s3 = _mm_loadu_ps(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                diffs0 = _mm_add_ps(diffs0, _mm_opt_fnmlaf_ps::<FMA>(a_s0, d_s0, v_double));
                diffs1 = _mm_add_ps(diffs1, _mm_opt_fnmlaf_ps::<FMA>(a_s1, d_s1, v_double));
                diffs2 = _mm_add_ps(diffs2, _mm_opt_fnmlaf_ps::<FMA>(a_s2, d_s2, v_double));
                diffs3 = _mm_add_ps(diffs3, _mm_opt_fnmlaf_ps::<FMA>(a_s3, d_s3, v_double));
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;
                let s0 = _mm_loadu_ps(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                let s1 = _mm_loadu_ps(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                let s2 = _mm_loadu_ps(bf2.as_mut_ptr().add(arr_index) as *mut f32);
                let s3 = _mm_loadu_ps(bf3.as_mut_ptr().add(arr_index) as *mut f32);

                diffs0 = _mm_opt_fnmlaf_ps::<FMA>(diffs0, s0, v_double);
                diffs1 = _mm_opt_fnmlaf_ps::<FMA>(diffs1, s1, v_double);
                diffs2 = _mm_opt_fnmlaf_ps::<FMA>(diffs2, s2, v_double);
                diffs3 = _mm_opt_fnmlaf_ps::<FMA>(diffs3, s3, v_double);
            }

            let next_row_y =
                clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * (stride as usize);

            let s_ptr0 = bytes.slice.as_ptr().add(next_row_y + current_px0) as *mut f32;
            let s_ptr1 = bytes.slice.as_ptr().add(next_row_y + current_px1) as *mut f32;
            let s_ptr2 = bytes.slice.as_ptr().add(next_row_y + current_px2) as *mut f32;
            let s_ptr3 = bytes.slice.as_ptr().add(next_row_y + current_px3) as *mut f32;

            let px0 = load_f32::<CN>(s_ptr0);
            let px1 = load_f32::<CN>(s_ptr1);
            let px2 = load_f32::<CN>(s_ptr2);
            let px3 = load_f32::<CN>(s_ptr3);

            let arr_index = ((y + radius_64) & 1023) as usize;

            _mm_storeu_ps(bf0.as_mut_ptr().add(arr_index) as *mut f32, px0);
            _mm_storeu_ps(bf1.as_mut_ptr().add(arr_index) as *mut f32, px1);
            _mm_storeu_ps(bf2.as_mut_ptr().add(arr_index) as *mut f32, px2);
            _mm_storeu_ps(bf3.as_mut_ptr().add(arr_index) as *mut f32, px3);

            diffs0 = _mm_add_ps(diffs0, px0);
            diffs1 = _mm_add_ps(diffs1, px1);
            diffs2 = _mm_add_ps(diffs2, px2);
            diffs3 = _mm_add_ps(diffs3, px3);

            sums0 = _mm_add_ps(sums0, diffs0);
            sums1 = _mm_add_ps(sums1, diffs1);
            sums2 = _mm_add_ps(sums2, diffs2);
            sums3 = _mm_add_ps(sums3, diffs3);
        }

        xx += 4;
    }

    for x in xx..width.min(end) as usize {
        let mut diffs = _mm_setzero_ps();
        let mut summs = _mm_setzero_ps();

        let start_y = 0 - 2 * radius as i64;

        let current_px = x * CN;

        for y in start_y..height_wide {
            let current_y = (y * (stride as i64)) as usize;

            if y >= 0 {
                let pixel = _mm_mul_ps(summs, v_weight);

                let bytes_offset = current_y + current_px;

                let dst_ptr = bytes.slice.as_ptr().add(bytes_offset) as *mut f32;
                store_f32::<CN>(dst_ptr, pixel);

                let arr_index = ((y - radius_64) & 1023) as usize;
                let d_arr_index = (y & 1023) as usize;

                let d_buf_ptr = bf0.get_unchecked_mut(d_arr_index).as_mut_ptr();
                let d_stored = _mm_loadu_ps(d_buf_ptr);

                let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                let a_stored = _mm_loadu_ps(buf_ptr);

                diffs = _mm_add_ps(
                    diffs,
                    _mm_opt_fnmlaf_ps::<FMA>(a_stored, d_stored, v_double),
                );
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;
                let buf_ptr = bf0.get_unchecked_mut(arr_index).as_mut_ptr();
                let stored = _mm_loadu_ps(buf_ptr);
                diffs = _mm_opt_fnmlaf_ps::<FMA>(diffs, stored, v_double);
            }

            let next_row_y =
                clamp_edge!(edge_mode, y + radius_64, 0, height_wide) * (stride as usize);
            let next_row_x = x * CN;

            let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut f32;
            let pixel_color = load_f32::<CN>(s_ptr);

            let arr_index = ((y + radius_64) & 1023) as usize;
            let buf_ptr = bf0.get_unchecked_mut(arr_index).as_mut_ptr();

            diffs = _mm_add_ps(diffs, pixel_color);
            summs = _mm_add_ps(summs, diffs);
            _mm_storeu_ps(buf_ptr, pixel_color);
        }
    }
}
