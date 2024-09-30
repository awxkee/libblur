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

use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::neon::{prefer_vfmaq_f32, vhsumq_f32};
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, reflect_101, reflect_index, EdgeMode};
use std::arch::aarch64::*;

macro_rules! accumulate_4_items {
    ($store0:expr, $pixel_colors:expr, $weights:expr) => {{
        $store0 = prefer_vfmaq_f32($store0, $pixel_colors.0, $weights.0);
        $store0 = prefer_vfmaq_f32($store0, $pixel_colors.1, $weights.1);
        $store0 = prefer_vfmaq_f32($store0, $pixel_colors.2, $weights.2);
        $store0 = prefer_vfmaq_f32($store0, $pixel_colors.3, $weights.3);
    }};
}

pub fn gaussian_horiz_one_chan_f32<T>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
    edge_mode: EdgeMode,
) {
    let src: &[f32] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, f32> = unsafe { std::mem::transmute(undef_unsafe_dst) };
    let half_kernel = (kernel_size / 2) as i32;

    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let y_src_shift_next = y_src_shift + src_stride as usize;
                let y_dst_shift_next = y_dst_shift + dst_stride as usize;

                let mut store0: float32x4_t = vdupq_n_f32(0f32);
                let mut store1: float32x4_t = vdupq_n_f32(0f32);

                let mut r = -half_kernel;

                let zeros = vdupq_n_f32(0f32);

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();

                    let start_x = edge_value_check;

                    for i in 0..diff as usize {
                        let s_ptr = src.as_ptr().add(
                            y_src_shift
                                + clamp_edge!(
                                    edge_mode,
                                    start_x + i as i64,
                                    0i64,
                                    width as i64 - 1
                                ),
                        );
                        let pixel_colors_f32_0 = vld1q_lane_f32::<0>(s_ptr, zeros);
                        let s_ptr_next = src.as_ptr().add(y_src_shift_next); // Here we're always at zero
                        let pixel_colors_f32_1 = vld1q_lane_f32::<0>(s_ptr_next, zeros);

                        let weight = kernel.as_ptr().add(i);
                        let f_weight = vld1q_lane_f32::<0>(weight, zeros);
                        store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_0, f_weight);
                        store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_1, f_weight);
                    }
                    r += diff as i32;
                }

                while r + 32 <= half_kernel && ((x as i64 + r as i64 + 32i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set0_0 = vld1q_f32_x4(s_ptr);
                    let pixel_colors_f32_set0_1 = vld1q_f32_x4(s_ptr.add(16));

                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);

                    let pixel_colors_f32_set_next_0 = vld1q_f32_x4(s_ptr_next);
                    let pixel_colors_f32_set_next_1 = vld1q_f32_x4(s_ptr_next.add(16));

                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set_0 = vld1q_f32_x4(weight);
                    let weights_set_1 = vld1q_f32_x4(weight.add(16));

                    accumulate_4_items!(store0, pixel_colors_f32_set0_0, weights_set_0);
                    accumulate_4_items!(store0, pixel_colors_f32_set0_1, weights_set_1);

                    accumulate_4_items!(store1, pixel_colors_f32_set_next_0, weights_set_0);
                    accumulate_4_items!(store1, pixel_colors_f32_set_next_1, weights_set_1);

                    r += 32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = vld1q_f32_x4(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_set_1 = vld1q_f32_x4(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set = vld1q_f32_x4(weight);

                    accumulate_4_items!(store0, pixel_colors_f32_set_0, weights_set);
                    accumulate_4_items!(store1, pixel_colors_f32_set_1, weights_set);

                    r += 16;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_0 = vld1q_f32(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = vld1q_f32(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vld1q_f32(weight);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_1, f_weight);

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        clamp_edge!(edge_mode, x as i64 + r as i64, 0i64, width as i64 - 1);
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_0 = vld1q_lane_f32::<0>(s_ptr, zeros);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = vld1q_lane_f32::<0>(s_ptr_next, zeros);
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_1, f_weight);

                    r += 1;
                }

                let agg0 = vhsumq_f32(store0);
                let offset = y_dst_shift + x as usize;
                let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr0.write_unaligned(agg0);

                let agg1 = vhsumq_f32(store1);
                let offset = y_dst_shift_next + x as usize;
                let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr1.write_unaligned(agg1);
            }
        }
        _cy = y;
    }

    for y in _cy..end_y {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store: float32x4_t = vdupq_n_f32(0f32);

                let mut r = -half_kernel;

                let zeros = vdupq_n_f32(0f32);

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();

                    let start_x = edge_value_check;

                    for i in 0..diff as usize {
                        let s_ptr = src.as_ptr().add(
                            y_src_shift
                                + clamp_edge!(
                                    edge_mode,
                                    start_x + i as i64,
                                    0i64,
                                    width as i64 - 1
                                ),
                        );
                        let pixel_colors_f32 = vld1q_lane_f32::<0>(s_ptr, zeros);
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = vld1q_lane_f32::<0>(weights, zeros);
                        store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);
                    }
                    r += diff as i32;
                }

                while r + 32 <= half_kernel && ((x as i64 + r as i64 + 32i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = vld1q_f32_x4(s_ptr);
                    let pixel_colors_f32_set_1 = vld1q_f32_x4(s_ptr.add(16));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set_0 = vld1q_f32_x4(weight);
                    let weights_set_1 = vld1q_f32_x4(weight.add(16));

                    accumulate_4_items!(store, pixel_colors_f32_set_0, weights_set_0);
                    accumulate_4_items!(store, pixel_colors_f32_set_1, weights_set_1);

                    r += 32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set = vld1q_f32_x4(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set = vld1q_f32_x4(weight);

                    accumulate_4_items!(store, pixel_colors_f32_set, weights_set);

                    r += 16;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = vld1q_f32(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vld1q_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        clamp_edge!(edge_mode, x as i64 + r as i64, 0i64, width as i64 - 1);
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = vld1q_lane_f32::<0>(s_ptr, zeros);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vld1q_lane_f32::<0>(weight, zeros);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 1;
                }

                let agg = vhsumq_f32(store);
                let offset = y_dst_shift + x as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr.write_unaligned(agg);
            }
        }
    }
}

pub fn gaussian_horiz_one_chan_filter_f32<T>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &[GaussianFilter<f32>],
    start_y: u32,
    end_y: u32,
) {
    let src: &[f32] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, f32> = unsafe { std::mem::transmute(undef_unsafe_dst) };

    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let y_src_shift_next = y_src_shift + src_stride as usize;
                let y_dst_shift_next = y_dst_shift + dst_stride as usize;

                let mut store0: float32x4_t = vdupq_n_f32(0f32);
                let mut store1: float32x4_t = vdupq_n_f32(0f32);

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut r = 0;

                let zeros = vdupq_n_f32(0f32);

                while r + 32 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 32i64) < width as i64)
                {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set0_0 = vld1q_f32_x4(s_ptr);
                    let pixel_colors_f32_set0_1 = vld1q_f32_x4(s_ptr.add(16));

                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);

                    let pixel_colors_f32_set_next_0 = vld1q_f32_x4(s_ptr_next);
                    let pixel_colors_f32_set_next_1 = vld1q_f32_x4(s_ptr_next.add(16));

                    let weight = filter_weights.as_ptr().add(r);
                    let weights_set_0 = vld1q_f32_x4(weight);
                    let weights_set_1 = vld1q_f32_x4(weight.add(16));

                    accumulate_4_items!(store0, pixel_colors_f32_set0_0, weights_set_0);
                    accumulate_4_items!(store0, pixel_colors_f32_set0_1, weights_set_1);

                    accumulate_4_items!(store1, pixel_colors_f32_set_next_0, weights_set_0);
                    accumulate_4_items!(store1, pixel_colors_f32_set_next_1, weights_set_1);

                    r += 32;
                }

                while r + 16 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 16i64) < width as i64)
                {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = vld1q_f32_x4(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_set_1 = vld1q_f32_x4(s_ptr_next);
                    let weight = filter_weights.as_ptr().add(r);
                    let weights_set = vld1q_f32_x4(weight);

                    accumulate_4_items!(store0, pixel_colors_f32_set_0, weights_set);
                    accumulate_4_items!(store1, pixel_colors_f32_set_1, weights_set);

                    r += 16;
                }

                while r + 4 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 4i64) < width as i64)
                {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_0 = vld1q_f32(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = vld1q_f32(s_ptr_next);
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight: float32x4_t = vld1q_f32(weight);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_1, f_weight);

                    r += 4;
                }

                while r < current_filter.size {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_0 = vld1q_lane_f32::<0>(s_ptr, zeros);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = vld1q_lane_f32::<0>(s_ptr_next, zeros);
                    let weight = filter_weights.as_ptr().add(r).read_unaligned();
                    let f_weight: float32x4_t = vdupq_n_f32(weight);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_1, f_weight);

                    r += 1;
                }

                let agg0 = vhsumq_f32(store0);
                let offset = y_dst_shift + x as usize;
                let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr0.write_unaligned(agg0);

                let agg1 = vhsumq_f32(store1);
                let offset = y_dst_shift_next + x as usize;
                let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr1.write_unaligned(agg1);
            }
        }
        _cy = y;
    }

    for y in _cy..end_y {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store: float32x4_t = vdupq_n_f32(0f32);

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut r = 0usize;

                let zeros = vdupq_n_f32(0f32);

                while r + 32 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 32i64) < width as i64)
                {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = vld1q_f32_x4(s_ptr);
                    let pixel_colors_f32_set_1 = vld1q_f32_x4(s_ptr.add(16));
                    let weight = filter_weights.as_ptr().add(r);
                    let weights_set_0 = vld1q_f32_x4(weight);
                    let weights_set_1 = vld1q_f32_x4(weight.add(16));

                    accumulate_4_items!(store, pixel_colors_f32_set_0, weights_set_0);
                    accumulate_4_items!(store, pixel_colors_f32_set_1, weights_set_1);

                    r += 32;
                }

                while r + 16 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 16i64) < width as i64)
                {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set = vld1q_f32_x4(s_ptr);
                    let weight = filter_weights.as_ptr().add(r);
                    let weights_set = vld1q_f32_x4(weight);

                    accumulate_4_items!(store, pixel_colors_f32_set, weights_set);

                    r += 16;
                }

                while r + 4 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 4i64) < width as i64)
                {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = vld1q_f32(s_ptr);
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight: float32x4_t = vld1q_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 4;
                }

                while r < current_filter.size {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = vld1q_lane_f32::<0>(s_ptr, zeros);
                    let weight = filter_weights.as_ptr().add(r).read_unaligned();
                    let f_weight: float32x4_t = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 1;
                }

                let agg = vhsumq_f32(store);
                let offset = y_dst_shift + x as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr.write_unaligned(agg);
            }
        }
    }
}
