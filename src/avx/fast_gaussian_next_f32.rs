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

use crate::avx::utils::{_mm256_opt_fmlaf_ps, _mm256_opt_fnmlaf_ps, _mm256_opt_fnmlsf_ps};
use crate::sse::{_mm_opt_fmlaf_ps, _mm_opt_fnmlaf_ps, _mm_opt_fnmlsf_ps, load_f32, store_f32};
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, EdgeMode};
use std::arch::x86_64::*;

pub(crate) fn fgn_vertical_pass_avx_f32<T, const CN: usize>(
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
            fgn_vertical_pass_avx_f32_fma::<CN>(
                bytes, stride, width, height, radius, start, end, edge_mode,
            );
        } else {
            fgn_vertical_pass_avx_f32_def::<CN>(
                bytes, stride, width, height, radius, start, end, edge_mode,
            );
        }
    }
}

#[repr(C, align(32))]
#[derive(Copy, Clone, Default)]
pub(crate) struct AvxSseF32x8(pub(crate) [f32; 8]);

#[target_feature(enable = "avx2")]
unsafe fn fgn_vertical_pass_avx_f32_def<const CN: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    let executor = VerticalGaussianExecutorF32::<CN, false>::default();
    executor.vertical_pass(bytes, stride, width, height, radius, start, end, edge_mode);
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fgn_vertical_pass_avx_f32_fma<const CN: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    let executor = VerticalGaussianExecutorF32::<CN, true>::default();
    executor.vertical_pass(bytes, stride, width, height, radius, start, end, edge_mode);
}

/// LLVM has issues sometimes with inline large methods, this need to avoid this.
#[derive(Default)]
struct VerticalGaussianExecutorF32<const CN: usize, const FMA: bool> {}

impl<const CN: usize, const FMA: bool> VerticalGaussianExecutorF32<CN, FMA> {
    #[inline(always)]
    unsafe fn vertical_pass(
        &self,
        bytes: &UnsafeSlice<f32>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        edge_mode: EdgeMode,
    ) {
        let mut full_buffer = Box::new([AvxSseF32x8::default(); 1024 * 3]);

        let (bf0, rem) = full_buffer.split_at_mut(1024);
        let (bf1, bf2) = rem.split_at_mut(1024);

        let height_wide = height as i64;

        let threes = _mm256_set1_ps(3.);

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let v_weight = _mm256_set1_ps(weight);

        let mut xx = start as usize;

        while xx + 6 < width.min(end) as usize {
            let mut diffs0 = _mm256_setzero_ps();
            let mut diffs1 = _mm256_setzero_ps();
            let mut diffs2 = _mm256_setzero_ps();

            let mut ders0 = _mm256_setzero_ps();
            let mut ders1 = _mm256_setzero_ps();
            let mut ders2 = _mm256_setzero_ps();

            let mut summs0 = _mm256_setzero_ps();
            let mut summs1 = _mm256_setzero_ps();
            let mut summs2 = _mm256_setzero_ps();

            let current_px0 = xx * CN;
            let current_px1 = (xx + 1) * CN;
            let current_px2 = (xx + 2) * CN;
            let current_px3 = (xx + 3) * CN;
            let current_px4 = (xx + 4) * CN;
            let current_px5 = (xx + 5) * CN;

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * (stride as i64)) as usize;

                    let r01 = _mm256_mul_ps(summs0, v_weight);
                    let r23 = _mm256_mul_ps(summs1, v_weight);
                    let r45 = _mm256_mul_ps(summs2, v_weight);

                    let dst_ptr0 = bytes.slice.as_ptr().add(current_y + current_px0) as *mut f32;
                    let dst_ptr1 = bytes.slice.as_ptr().add(current_y + current_px1) as *mut f32;
                    let dst_ptr2 = bytes.slice.as_ptr().add(current_y + current_px2) as *mut f32;
                    let dst_ptr3 = bytes.slice.as_ptr().add(current_y + current_px3) as *mut f32;
                    let dst_ptr4 = bytes.slice.as_ptr().add(current_y + current_px4) as *mut f32;
                    let dst_ptr5 = bytes.slice.as_ptr().add(current_y + current_px5) as *mut f32;

                    store_f32::<CN>(dst_ptr0, _mm256_castps256_ps128(r01));
                    store_f32::<CN>(dst_ptr1, _mm256_extractf128_ps::<1>(r01));
                    store_f32::<CN>(dst_ptr2, _mm256_castps256_ps128(r23));
                    store_f32::<CN>(dst_ptr3, _mm256_extractf128_ps::<1>(r23));
                    store_f32::<CN>(dst_ptr4, _mm256_castps256_ps128(r45));
                    store_f32::<CN>(dst_ptr5, _mm256_extractf128_ps::<1>(r45));

                    let d_a_1 = ((y + radius_64) & 1023) as usize;
                    let d_a_2 = ((y - radius_64) & 1023) as usize;
                    let d_i = (y & 1023) as usize;

                    let sd0 = _mm256_load_ps(bf0.as_mut_ptr().add(d_i) as *mut f32);
                    let sd1 = _mm256_load_ps(bf1.as_mut_ptr().add(d_i) as *mut f32);
                    let sd2 = _mm256_load_ps(bf2.as_mut_ptr().add(d_i) as *mut f32);

                    let sd_1_0 = _mm256_load_ps(bf0.as_mut_ptr().add(d_a_1) as *mut f32);
                    let sd_1_1 = _mm256_load_ps(bf1.as_mut_ptr().add(d_a_1) as *mut f32);
                    let sd_1_2 = _mm256_load_ps(bf2.as_mut_ptr().add(d_a_1) as *mut f32);

                    let j0 = _mm256_sub_ps(sd0, sd_1_0);
                    let j1 = _mm256_sub_ps(sd1, sd_1_1);
                    let j2 = _mm256_sub_ps(sd2, sd_1_2);

                    let sd_2_0 = _mm256_load_ps(bf0.as_mut_ptr().add(d_a_2) as *mut f32);
                    let sd_2_1 = _mm256_load_ps(bf1.as_mut_ptr().add(d_a_2) as *mut f32);
                    let sd_2_2 = _mm256_load_ps(bf2.as_mut_ptr().add(d_a_2) as *mut f32);

                    let new_diff0 = _mm256_opt_fnmlsf_ps::<FMA>(sd_2_0, j0, threes);
                    let new_diff1 = _mm256_opt_fnmlsf_ps::<FMA>(sd_2_1, j1, threes);
                    let new_diff2 = _mm256_opt_fnmlsf_ps::<FMA>(sd_2_2, j2, threes);

                    diffs0 = _mm256_add_ps(diffs0, new_diff0);
                    diffs1 = _mm256_add_ps(diffs1, new_diff1);
                    diffs2 = _mm256_add_ps(diffs2, new_diff2);
                } else if y + radius_64 >= 0 {
                    let a_i = (y & 1023) as usize;
                    let a_i_1 = ((y + radius_64) & 1023) as usize;
                    let sd0 = _mm256_load_ps(bf0.as_mut_ptr().add(a_i) as *mut f32);
                    let sd1 = _mm256_load_ps(bf1.as_mut_ptr().add(a_i) as *mut f32);
                    let sd2 = _mm256_load_ps(bf2.as_mut_ptr().add(a_i) as *mut f32);

                    let sd_1_0 = _mm256_load_ps(bf0.as_mut_ptr().add(a_i_1) as *mut f32);
                    let sd_1_1 = _mm256_load_ps(bf1.as_mut_ptr().add(a_i_1) as *mut f32);
                    let sd_1_2 = _mm256_load_ps(bf2.as_mut_ptr().add(a_i_1) as *mut f32);

                    diffs0 = _mm256_opt_fmlaf_ps::<FMA>(diffs0, _mm256_sub_ps(sd0, sd_1_0), threes);
                    diffs1 = _mm256_opt_fmlaf_ps::<FMA>(diffs1, _mm256_sub_ps(sd1, sd_1_1), threes);
                    diffs2 = _mm256_opt_fmlaf_ps::<FMA>(diffs2, _mm256_sub_ps(sd2, sd_1_2), threes);
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let sd0 = _mm256_load_ps(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                    let sd1 = _mm256_load_ps(bf1.as_mut_ptr().add(arr_index) as *mut f32);
                    let sd2 = _mm256_load_ps(bf2.as_mut_ptr().add(arr_index) as *mut f32);

                    diffs0 = _mm256_opt_fnmlaf_ps::<FMA>(diffs0, sd0, threes);
                    diffs1 = _mm256_opt_fnmlaf_ps::<FMA>(diffs1, sd1, threes);
                    diffs2 = _mm256_opt_fnmlaf_ps::<FMA>(diffs2, sd2, threes);
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * (stride as usize);

                let s_ptr0 = bytes.slice.as_ptr().add(next_row_y + current_px0) as *mut f32;
                let s_ptr1 = bytes.slice.as_ptr().add(next_row_y + current_px1) as *mut f32;
                let s_ptr2 = bytes.slice.as_ptr().add(next_row_y + current_px2) as *mut f32;
                let s_ptr3 = bytes.slice.as_ptr().add(next_row_y + current_px3) as *mut f32;
                let s_ptr4 = bytes.slice.as_ptr().add(next_row_y + current_px4) as *mut f32;
                let s_ptr5 = bytes.slice.as_ptr().add(next_row_y + current_px5) as *mut f32;

                let pixel_color0 = load_f32::<CN>(s_ptr0);
                let pixel_color1 = load_f32::<CN>(s_ptr1);
                let pixel_color2 = load_f32::<CN>(s_ptr2);
                let pixel_color3 = load_f32::<CN>(s_ptr3);
                let pixel_color4 = load_f32::<CN>(s_ptr4);
                let pixel_color5 = load_f32::<CN>(s_ptr5);

                let a_i = ((y + 2 * radius_64) & 1023) as usize;

                let px01 =
                    _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(pixel_color0), pixel_color1);
                let px23 =
                    _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(pixel_color2), pixel_color3);
                let px45 =
                    _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(pixel_color4), pixel_color5);

                _mm256_store_ps(bf0.as_mut_ptr().add(a_i) as *mut f32, px01);
                _mm256_store_ps(bf1.as_mut_ptr().add(a_i) as *mut f32, px23);
                _mm256_store_ps(bf2.as_mut_ptr().add(a_i) as *mut f32, px45);

                diffs0 = _mm256_add_ps(diffs0, px01);
                diffs1 = _mm256_add_ps(diffs1, px23);
                diffs2 = _mm256_add_ps(diffs2, px45);

                ders0 = _mm256_add_ps(ders0, diffs0);
                ders1 = _mm256_add_ps(ders1, diffs1);
                ders2 = _mm256_add_ps(ders2, diffs2);

                summs0 = _mm256_add_ps(summs0, ders0);
                summs1 = _mm256_add_ps(summs1, ders1);
                summs2 = _mm256_add_ps(summs2, ders2);
            }

            xx += 6;
        }

        for x in xx..width.min(end) as usize {
            let mut diffs = _mm_setzero_ps();
            let mut ders = _mm_setzero_ps();
            let mut summs = _mm_setzero_ps();

            let current_px = x * CN;

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                if y >= 0 {
                    let current_y = (y * (stride as i64)) as usize;
                    let bytes_offset = current_y + current_px;

                    let pixel = _mm_mul_ps(summs, _mm256_castps256_ps128(v_weight));
                    let dst_ptr = bytes.slice.as_ptr().add(bytes_offset) as *mut f32;
                    store_f32::<CN>(dst_ptr, pixel);

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let buf_ptr = bf0.get_unchecked_mut(d_arr_index).0.as_mut_ptr();
                    let stored = _mm_load_ps(buf_ptr);

                    let buf_ptr_1 = bf0.as_mut_ptr().add(d_arr_index_1) as *mut f32;
                    let stored_1 = _mm_load_ps(buf_ptr_1);

                    let buf_ptr_2 = bf0.as_mut_ptr().add(d_arr_index_2) as *mut f32;
                    let stored_2 = _mm_load_ps(buf_ptr_2);

                    let new_diff = _mm_opt_fnmlsf_ps::<FMA>(
                        stored_2,
                        _mm_sub_ps(stored, stored_1),
                        _mm256_castps256_ps128(threes),
                    );
                    diffs = _mm_add_ps(diffs, new_diff);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = bf0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                    let stored = _mm_load_ps(buf_ptr);

                    let buf_ptr_1 = bf0.get_unchecked_mut(arr_index_1).0.as_mut_ptr();
                    let stored_1 = _mm_load_ps(buf_ptr_1);

                    diffs = _mm_opt_fmlaf_ps::<FMA>(
                        diffs,
                        _mm_sub_ps(stored, stored_1),
                        _mm256_castps256_ps128(threes),
                    );
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = bf0.get_unchecked_mut(arr_index).0.as_mut_ptr();
                    let stored = _mm_load_ps(buf_ptr);
                    diffs = _mm_opt_fnmlaf_ps::<FMA>(diffs, stored, _mm256_castps256_ps128(threes));
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * (stride as usize);
                let next_row_x = x * CN;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut f32;

                let pixel_color = load_f32::<CN>(s_ptr);

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                let buf_ptr = bf0.get_unchecked_mut(arr_index).0.as_mut_ptr();

                diffs = _mm_add_ps(diffs, pixel_color);
                ders = _mm_add_ps(ders, diffs);
                summs = _mm_add_ps(summs, ders);
                _mm_store_ps(buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fgn_horizontal_pass_avx_f32<T, const CN: usize>(
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
            fgn_horizontal_pass_avx_f32_fma::<CN>(
                bytes, stride, width, height, radius, start, end, edge_mode,
            );
        } else {
            fgn_horizontal_pass_avx_f32_def::<CN>(
                bytes, stride, width, height, radius, start, end, edge_mode,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn fgn_horizontal_pass_avx_f32_def<const CN: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    let executor = HorizontalAvxF32Executor::<CN, false>::default();
    executor.horizontal_pass(bytes, stride, width, height, radius, start, end, edge_mode);
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fgn_horizontal_pass_avx_f32_fma<const CN: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) {
    let executor = HorizontalAvxF32Executor::<CN, true>::default();
    executor.horizontal_pass(bytes, stride, width, height, radius, start, end, edge_mode);
}

#[derive(Copy, Clone, Default)]
struct HorizontalAvxF32Executor<const CN: usize, const FMA: bool> {}

impl<const CN: usize, const FMA: bool> HorizontalAvxF32Executor<CN, FMA> {
    #[inline(always)]
    unsafe fn horizontal_pass(
        &self,
        bytes: &UnsafeSlice<f32>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        edge_mode: EdgeMode,
    ) {
        let mut bf0 = Box::new([AvxSseF32x8::default(); 1024]);
        let mut bf1 = Box::new([AvxSseF32x8::default(); 1024]);

        let width_wide = width as i64;

        let threes = _mm256_set1_ps(3.);

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let v_weight = _mm256_set1_ps(weight);

        let mut yy = start as usize;

        while yy + 4 < height.min(end) as usize {
            let mut diffs0 = _mm256_setzero_ps();
            let mut diffs1 = _mm256_setzero_ps();

            let mut ders0 = _mm256_setzero_ps();
            let mut ders1 = _mm256_setzero_ps();

            let mut summs0 = _mm256_setzero_ps();
            let mut summs1 = _mm256_setzero_ps();

            let start_x = 0 - 3 * radius_64;

            let current_y0 = ((yy as i64) * (stride as i64)) as usize;
            let current_y1 = (((yy + 1) as i64) * (stride as i64)) as usize;
            let current_y2 = (((yy + 2) as i64) * (stride as i64)) as usize;
            let current_y3 = (((yy + 3) as i64) * (stride as i64)) as usize;

            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let r01 = _mm256_mul_ps(summs0, v_weight);
                    let r23 = _mm256_mul_ps(summs1, v_weight);

                    let dst_ptr0 = bytes.slice.as_ptr().add(current_y0 + current_px) as *mut f32;
                    let dst_ptr1 = bytes.slice.as_ptr().add(current_y1 + current_px) as *mut f32;
                    let dst_ptr2 = bytes.slice.as_ptr().add(current_y2 + current_px) as *mut f32;
                    let dst_ptr3 = bytes.slice.as_ptr().add(current_y3 + current_px) as *mut f32;

                    store_f32::<CN>(dst_ptr0, _mm256_castps256_ps128(r01));
                    store_f32::<CN>(dst_ptr1, _mm256_extractf128_ps::<1>(r01));
                    store_f32::<CN>(dst_ptr2, _mm256_castps256_ps128(r23));
                    store_f32::<CN>(dst_ptr3, _mm256_extractf128_ps::<1>(r23));

                    let d_a_1 = ((x + radius_64) & 1023) as usize;
                    let d_a_2 = ((x - radius_64) & 1023) as usize;
                    let d_i = (x & 1023) as usize;

                    let sd0 = _mm256_load_ps(bf0.as_mut_ptr().add(d_i) as *mut f32);
                    let sd1 = _mm256_load_ps(bf1.as_mut_ptr().add(d_i) as *mut f32);

                    let sd_1_0 = _mm256_load_ps(bf0.as_mut_ptr().add(d_a_1) as *mut f32);
                    let sd_1_1 = _mm256_load_ps(bf1.as_mut_ptr().add(d_a_1) as *mut f32);

                    let j0 = _mm256_sub_ps(sd0, sd_1_0);
                    let j1 = _mm256_sub_ps(sd1, sd_1_1);

                    let sd_2_0 = _mm256_load_ps(bf0.as_mut_ptr().add(d_a_2) as *mut f32);
                    let sd_2_1 = _mm256_load_ps(bf1.as_mut_ptr().add(d_a_2) as *mut f32);

                    let new_diff0 = _mm256_opt_fnmlsf_ps::<FMA>(sd_2_0, j0, threes);
                    let new_diff1 = _mm256_opt_fnmlsf_ps::<FMA>(sd_2_1, j1, threes);

                    diffs0 = _mm256_add_ps(diffs0, new_diff0);
                    diffs1 = _mm256_add_ps(diffs1, new_diff1);
                } else if x + radius_64 >= 0 {
                    let a_i = (x & 1023) as usize;
                    let a_i_1 = ((x + radius_64) & 1023) as usize;
                    let sd0 = _mm256_load_ps(bf0.as_mut_ptr().add(a_i) as *mut f32);
                    let sd1 = _mm256_load_ps(bf1.as_mut_ptr().add(a_i) as *mut f32);

                    let sd_1_0 = _mm256_load_ps(bf0.as_mut_ptr().add(a_i_1) as *mut f32);
                    let sd_1_1 = _mm256_load_ps(bf1.as_mut_ptr().add(a_i_1) as *mut f32);

                    diffs0 = _mm256_opt_fmlaf_ps::<FMA>(diffs0, _mm256_sub_ps(sd0, sd_1_0), threes);
                    diffs1 = _mm256_opt_fmlaf_ps::<FMA>(diffs1, _mm256_sub_ps(sd1, sd_1_1), threes);
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let sd0 = _mm256_load_ps(bf0.as_mut_ptr().add(arr_index) as *mut f32);
                    let sd1 = _mm256_load_ps(bf1.as_mut_ptr().add(arr_index) as *mut f32);

                    diffs0 = _mm256_opt_fnmlaf_ps::<FMA>(diffs0, sd0, threes);
                    diffs1 = _mm256_opt_fnmlaf_ps::<FMA>(diffs1, sd1, threes);
                }

                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr0 = bytes.slice.as_ptr().add(current_y0 + next_row_px) as *mut f32;
                let s_ptr1 = bytes.slice.as_ptr().add(current_y1 + next_row_px) as *mut f32;
                let s_ptr2 = bytes.slice.as_ptr().add(current_y2 + next_row_px) as *mut f32;
                let s_ptr3 = bytes.slice.as_ptr().add(current_y3 + next_row_px) as *mut f32;

                let pixel_color0 = load_f32::<CN>(s_ptr0);
                let pixel_color1 = load_f32::<CN>(s_ptr1);
                let pixel_color2 = load_f32::<CN>(s_ptr2);
                let pixel_color3 = load_f32::<CN>(s_ptr3);

                let a_i = ((x + 2 * radius_64) & 1023) as usize;

                let px01 =
                    _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(pixel_color0), pixel_color1);
                let px23 =
                    _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(pixel_color2), pixel_color3);

                _mm256_store_ps(bf0.as_mut_ptr().add(a_i) as *mut f32, px01);
                _mm256_store_ps(bf1.as_mut_ptr().add(a_i) as *mut f32, px23);

                diffs0 = _mm256_add_ps(diffs0, px01);
                diffs1 = _mm256_add_ps(diffs1, px23);

                ders0 = _mm256_add_ps(ders0, diffs0);
                ders1 = _mm256_add_ps(ders1, diffs1);

                summs0 = _mm256_add_ps(summs0, ders0);
                summs1 = _mm256_add_ps(summs1, ders1);
            }

            yy += 4;
        }

        for y in yy..height.min(end) as usize {
            let mut diffs = _mm_setzero_ps();
            let mut ders = _mm_setzero_ps();
            let mut summs = _mm_setzero_ps();

            let current_y = ((y as i64) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let bytes_offset = current_y + current_px;

                    let pixel = _mm_mul_ps(summs, _mm256_castps256_ps128(v_weight));
                    let dst_ptr = bytes.slice.as_ptr().add(bytes_offset) as *mut f32;
                    store_f32::<CN>(dst_ptr, pixel);

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let buf_ptr = bf0.get_unchecked_mut(d_arr_index).0.as_mut_ptr();
                    let stored = _mm_load_ps(buf_ptr);

                    let buf_ptr_1 = bf0.get_unchecked_mut(d_arr_index_1).0.as_mut_ptr();
                    let stored_1 = _mm_load_ps(buf_ptr_1);

                    let buf_ptr_2 = bf0.get_unchecked_mut(d_arr_index_2).0.as_mut_ptr();
                    let stored_2 = _mm_load_ps(buf_ptr_2);

                    let new_diff = _mm_opt_fnmlsf_ps::<FMA>(
                        stored_2,
                        _mm_sub_ps(stored, stored_1),
                        _mm256_castps256_ps128(threes),
                    );
                    diffs = _mm_add_ps(diffs, new_diff);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = bf0.as_mut_ptr().add(arr_index) as *mut f32;
                    let stored = _mm_load_ps(buf_ptr);

                    let buf_ptr_1 = bf0.as_mut_ptr().add(arr_index_1);
                    let stored_1 = _mm_load_ps(buf_ptr_1 as *const f32);

                    diffs = _mm_opt_fmlaf_ps::<FMA>(
                        diffs,
                        _mm_sub_ps(stored, stored_1),
                        _mm256_castps256_ps128(threes),
                    );
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = bf0.as_mut_ptr().add(arr_index);
                    let stored = _mm_load_ps(buf_ptr as *const f32);
                    diffs = _mm_opt_fnmlaf_ps::<FMA>(diffs, stored, _mm256_castps256_ps128(threes));
                }

                let next_row_y = y * (stride as usize);
                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut f32;

                let pixel_color = load_f32::<CN>(s_ptr);

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr = bf0.get_unchecked_mut(arr_index).0.as_mut_ptr();

                diffs = _mm_add_ps(diffs, pixel_color);
                ders = _mm_add_ps(ders, diffs);
                summs = _mm_add_ps(summs, ders);
                _mm_store_ps(buf_ptr, pixel_color);
            }
        }
    }
}
