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
use crate::avx::fast_gaussian_next::AvxSseI32x8;
use crate::edge_mode::clamp_edge;
use crate::sse::utils::load_u16_s32_fast;
use crate::sse::{_mm_mul_by_3_epi32, store_u16_u32};
use crate::unsafe_slice::UnsafeSlice;
use std::arch::x86_64::*;

pub(crate) fn fgn_vertical_pass_avx_u16_large<const CN: usize>(
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

#[inline]
#[target_feature(enable = "avx2")]
fn _xx256_cvtepi64_pd(x: __m256i) -> __m256d {
    let magic = _mm256_castpd_si256(_mm256_set1_pd(f64::from_bits(0x0018000000000000)));
    let x = _mm256_add_epi64(x, magic);
    _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_castsi256_pd(magic))
}

#[inline]
#[target_feature(enable = "avx2")]
fn _xx256_cvtpd_epi64(x: __m256d) -> __m256i {
    let magic = _mm256_set1_pd(f64::from_bits(0x0018000000000000));
    let x = _mm256_add_pd(x, magic);
    _mm256_sub_epi64(_mm256_castpd_si256(x), _mm256_castpd_si256(magic))
}

#[inline]
#[target_feature(enable = "avx2")]
fn _xx256_cvtepi64_epi32(v: __m256i) -> __m128i {
    let shuffled = _mm256_shuffle_epi32::<0x88>(v);
    let packed = _mm256_permute4x64_epi64::<0x08>(shuffled);
    _mm256_castsi256_si128(packed)
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
            let mut buffer = [AvxSseI32x8::default(); 1024];

            let height_wide = height as i64;

            let radius_64 = radius as i64;

            let weight = 1.0 / ((radius as f64) * (radius as f64) * (radius as f64));
            let v_weight = _mm256_set1_pd(weight);

            for x in start..width.min(end) {
                let mut diffs = _mm_setzero_si128();
                let mut ders = _mm256_setzero_si256();
                let mut summs = _mm256_setzero_si256();

                let start_y = -3 * radius as i64;
                for y in start_y..height_wide {
                    if y >= 0 {
                        let current_px = (x * CN as u32) as usize;

                        let prepared_px_s64 =
                            _xx256_cvtpd_epi64(_mm256_mul_pd(_xx256_cvtepi64_pd(summs), v_weight));
                        let prepared_px_s32 = _xx256_cvtepi64_epi32(prepared_px_s64);

                        let current_y = (y * (stride as i64)) as usize;

                        let bytes_offset = current_y + current_px;

                        let dst_ptr = bytes.get_ptr(bytes_offset);
                        store_u16_u32::<CN>(dst_ptr, prepared_px_s32);

                        let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                        let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                        let d_arr_index = (y & 1023) as usize;

                        let buf_ptr = buffer.get_unchecked(d_arr_index).0.as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());

                        let buf_ptr_1 = buffer.get_unchecked(d_arr_index_1..).as_ptr();
                        let stored_1 = _mm_load_si128(buf_ptr_1.cast());

                        let buf_ptr_2 = buffer.get_unchecked(d_arr_index_2..).as_ptr();
                        let stored_2 = _mm_load_si128(buf_ptr_2.cast());

                        let new_diff = _mm_sub_epi32(
                            _mm_mul_by_3_epi32(_mm_sub_epi32(stored, stored_1)),
                            stored_2,
                        );
                        diffs = _mm_add_epi32(diffs, new_diff);
                    } else if y + radius_64 >= 0 {
                        let arr_index = (y & 1023) as usize;
                        let arr_index_1 = ((y + radius_64) & 1023) as usize;
                        let buf_ptr = buffer.get_unchecked(arr_index).0.as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());

                        let buf_ptr_1 = buffer.get_unchecked(arr_index_1).0.as_ptr();
                        let stored_1 = _mm_load_si128(buf_ptr_1.cast());

                        let new_diff = _mm_mul_by_3_epi32(_mm_sub_epi32(stored, stored_1));

                        diffs = _mm_add_epi32(diffs, new_diff);
                    } else if y + 2 * radius_64 >= 0 {
                        let arr_index = ((y + radius_64) & 1023) as usize;
                        let buf_ptr = buffer.get_unchecked_mut(arr_index).0.as_mut_ptr();
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
                    let buf_ptr = buffer.get_unchecked_mut(arr_index).0.as_mut_ptr();

                    diffs = _mm_add_epi32(diffs, pixel_color);
                    ders = _mm256_add_epi64(ders, _mm256_cvtepi32_epi64(diffs));
                    summs = _mm256_add_epi64(summs, ders);
                    _mm_store_si128(buf_ptr.cast(), pixel_color);
                }
            }
        }
    }
}

pub(crate) fn fgn_horizontal_pass_avx_u16_large<const CN: usize>(
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
            let mut buffer = [AvxSseI32x8::default(); 1024];

            let width_wide = width as i64;

            let radius_64 = radius as i64;

            let weight = 1.0 / ((radius as f64) * (radius as f64) * (radius as f64));
            let v_weight = _mm256_set1_pd(weight);

            for y in start..height.min(end) {
                let mut diffs = _mm_setzero_si128();
                let mut ders = _mm256_setzero_si256();
                let mut summs = _mm256_setzero_si256();

                let current_y = ((y as i64) * (stride as i64)) as usize;

                for x in (0 - 3 * radius_64)..(width as i64) {
                    if x >= 0 {
                        let current_px = x as usize * CN;

                        let prepared_px_s64 =
                            _xx256_cvtpd_epi64(_mm256_mul_pd(_xx256_cvtepi64_pd(summs), v_weight));
                        let prepared_px_s32 = _xx256_cvtepi64_epi32(prepared_px_s64);

                        let bytes_offset = current_y + current_px;

                        let dst_ptr = bytes.get_ptr(bytes_offset);
                        store_u16_u32::<CN>(dst_ptr, prepared_px_s32);

                        let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                        let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                        let d_arr_index = (x & 1023) as usize;

                        let buf_ptr = buffer.get_unchecked(d_arr_index).0.as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());

                        let buf_ptr_1 = buffer.get_unchecked(d_arr_index_1).0.as_ptr();
                        let stored_1 = _mm_load_si128(buf_ptr_1.cast());

                        let buf_ptr_2 = buffer.get_unchecked(d_arr_index_2).0.as_ptr();
                        let stored_2 = _mm_load_si128(buf_ptr_2.cast());

                        let new_diff = _mm_sub_epi32(
                            _mm_mul_by_3_epi32(_mm_sub_epi32(stored, stored_1)),
                            stored_2,
                        );
                        diffs = _mm_add_epi32(diffs, new_diff);
                    } else if x + radius_64 >= 0 {
                        let arr_index = (x & 1023) as usize;
                        let arr_index_1 = ((x + radius_64) & 1023) as usize;
                        let buf_ptr = buffer.get_unchecked(arr_index..).as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());

                        let buf_ptr_1 = buffer.get_unchecked(arr_index_1..).as_ptr();
                        let stored_1 = _mm_load_si128(buf_ptr_1.cast());

                        let new_diff = _mm_mul_by_3_epi32(_mm_sub_epi32(stored, stored_1));

                        diffs = _mm_add_epi32(diffs, new_diff);
                    } else if x + 2 * radius_64 >= 0 {
                        let arr_index = ((x + radius_64) & 1023) as usize;
                        let buf_ptr = buffer.get_unchecked(arr_index..).as_ptr();
                        let stored = _mm_load_si128(buf_ptr.cast());
                        diffs = _mm_sub_epi32(diffs, _mm_mul_by_3_epi32(stored));
                    }

                    let next_row_y = (y as usize) * (stride as usize);
                    let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                    let next_row_px = next_row_x * CN;

                    let pixel_color =
                        load_u16_s32_fast::<CN>(bytes.get_ptr(next_row_y + next_row_px));

                    let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                    let buf_ptr = buffer.get_unchecked_mut(arr_index).0.as_mut_ptr();

                    diffs = _mm_add_epi32(diffs, pixel_color);
                    ders = _mm256_add_epi64(ders, _mm256_cvtepi32_epi64(diffs));
                    summs = _mm256_add_epi64(summs, ders);
                    _mm_store_si128(buf_ptr.cast(), pixel_color);
                }
            }
        }
    }
}
