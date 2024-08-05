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
use crate::neon::{load_f32_fast, prefer_vfmaq_f32, store_f32, vsplit_rgb_5};
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;

macro_rules! accumulate_5_items {
    ($store0:expr, $set:expr, $f_weights:expr, $last_weight:expr) => {{
        $store0 = prefer_vfmaq_f32($store0, $set.0, vdupq_laneq_f32::<0>($f_weights));
        $store0 = prefer_vfmaq_f32($store0, $set.1, vdupq_laneq_f32::<1>($f_weights));
        $store0 = prefer_vfmaq_f32($store0, $set.2, vdupq_laneq_f32::<2>($f_weights));
        $store0 = prefer_vfmaq_f32($store0, $set.3, vdupq_laneq_f32::<3>($f_weights));
        $store0 = prefer_vfmaq_f32($store0, $set.4, vdupq_n_f32($last_weight));
    }};
}

macro_rules! accumulate_4_items {
    ($store0:expr, $set:expr, $f_weights:expr) => {{
        $store0 = prefer_vfmaq_f32($store0, $set.0, vdupq_laneq_f32::<0>($f_weights));
        $store0 = prefer_vfmaq_f32($store0, $set.1, vdupq_laneq_f32::<1>($f_weights));
        $store0 = prefer_vfmaq_f32($store0, $set.2, vdupq_laneq_f32::<2>($f_weights));
        $store0 = prefer_vfmaq_f32($store0, $set.3, vdupq_laneq_f32::<3>($f_weights));
    }};
}

pub fn gaussian_horiz_t_f_chan_f32<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let src: &[f32] = std::mem::transmute(undef_src);
        let unsafe_dst: &UnsafeSlice<'_, f32> = std::mem::transmute(undef_unsafe_dst);
        let half_kernel = (kernel_size / 2) as i32;

        let mut _cy = start_y;

        for y in (_cy..end_y.saturating_sub(4)).step_by(4) {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store0: float32x4_t = vdupq_n_f32(0.);
                let mut store1: float32x4_t = vdupq_n_f32(0.);
                let mut store2: float32x4_t = vdupq_n_f32(0.);
                let mut store3: float32x4_t = vdupq_n_f32(0.);

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let pixel_colors_0 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let s_ptr_next_1 = src.as_ptr().add(y_src_shift + src_stride as usize); // Here we're always at zero
                    let pixel_colors_1 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_1);
                    let s_ptr_next_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2); // Here we're always at zero
                    let pixel_colors_2 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_2);
                    let s_ptr_next_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3); // Here we're always at zero
                    let pixel_colors_3 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_3);
                    for i in 0..diff as usize {
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = vdupq_n_f32(weights.read_unaligned());
                        store0 = prefer_vfmaq_f32(store0, pixel_colors_0, f_weight);
                        store1 = prefer_vfmaq_f32(store1, pixel_colors_1, f_weight);
                        store2 = prefer_vfmaq_f32(store2, pixel_colors_2, f_weight);
                        store3 = prefer_vfmaq_f32(store3, pixel_colors_3, f_weight);
                    }
                    r += diff as i32;
                }

                if CHANNEL_CONFIGURATION == 4 {
                    while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let pixel_colors_2 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 2));
                        let pixel_colors_3 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 3));
                        let pixel_colors_0_n = vld1q_f32_x4(s_ptr.add(16));
                        let pixel_colors_1_n = vld1q_f32_x4(s_ptr.add(src_stride as usize).add(16));
                        let pixel_colors_2_n =
                            vld1q_f32_x4(s_ptr.add(src_stride as usize * 2).add(16));
                        let pixel_colors_3_n =
                            vld1q_f32_x4(s_ptr.add(src_stride as usize * 3).add(16));
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);

                        let f_weights0 = vld1q_f32(weight);
                        let f_weights1 = vld1q_f32(weight.add(4));

                        accumulate_4_items!(store0, pixel_colors_0, f_weights0);
                        accumulate_4_items!(store1, pixel_colors_1, f_weights0);
                        accumulate_4_items!(store2, pixel_colors_2, f_weights0);
                        accumulate_4_items!(store3, pixel_colors_3, f_weights0);

                        accumulate_4_items!(store0, pixel_colors_0_n, f_weights1);
                        accumulate_4_items!(store1, pixel_colors_1_n, f_weights1);
                        accumulate_4_items!(store2, pixel_colors_2_n, f_weights1);
                        accumulate_4_items!(store3, pixel_colors_3_n, f_weights1);

                        r += 8;
                    }

                    while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let pixel_colors_2 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 2));
                        let pixel_colors_3 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 3));
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                        let f_weights = vld1q_f32(weight);

                        accumulate_4_items!(store0, pixel_colors_0, f_weights);
                        accumulate_4_items!(store1, pixel_colors_1, f_weights);
                        accumulate_4_items!(store2, pixel_colors_2, f_weights);
                        accumulate_4_items!(store3, pixel_colors_3, f_weights);

                        r += 4;
                    }
                } else if CHANNEL_CONFIGURATION == 3 {
                    while r + 5 <= half_kernel && ((x as i64 + r as i64 + 6i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        // (r0 g0 b0 r1) (g1 b1 r2 g2) (b2 r3 g3 b3) (r4 g4 b5 undef)
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let pixel_colors_2 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 2));
                        let pixel_colors_3 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 3));
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                        let f_weights = vld1q_f32(weight);
                        let last_weight = weight.add(4).read_unaligned();
                        // r0 g0 b0 0
                        let set0 = vsplit_rgb_5(pixel_colors_0);
                        let set1 = vsplit_rgb_5(pixel_colors_1);
                        let set2 = vsplit_rgb_5(pixel_colors_2);
                        let set3 = vsplit_rgb_5(pixel_colors_3);

                        accumulate_5_items!(store0, set0, f_weights, last_weight);
                        accumulate_5_items!(store1, set1, f_weights, last_weight);
                        accumulate_5_items!(store2, set2, f_weights, last_weight);
                        accumulate_5_items!(store3, set3, f_weights, last_weight);

                        r += 5;
                    }
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let s_ptr_next = s_ptr.add(src_stride as usize);
                    let pixel_colors_1 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next);
                    let s_ptr_next_2 = s_ptr.add(src_stride as usize * 2);
                    let pixel_colors_2 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_2);
                    let s_ptr_next_3 = s_ptr.add(src_stride as usize * 3);
                    let pixel_colors_3 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_3);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight.read_unaligned());
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_1, f_weight);
                    store2 = prefer_vfmaq_f32(store2, pixel_colors_2, f_weight);
                    store3 = prefer_vfmaq_f32(store3, pixel_colors_3, f_weight);

                    r += 1;
                }

                let offset = y_dst_shift + x as usize * CHANNEL_CONFIGURATION;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store0);

                let offset = offset + dst_stride as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store1);

                let offset = offset + dst_stride as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store2);

                let offset = offset + dst_stride as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store3);
            }
            _cy = y;
        }

        for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store0: float32x4_t = vdupq_n_f32(0.);
                let mut store1: float32x4_t = vdupq_n_f32(0.);

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let pixel_colors_0 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift + src_stride as usize); // Here we're always at zero
                    let pixel_colors_1 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next);
                    for i in 0..diff as usize {
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = vdupq_n_f32(weights.read_unaligned());
                        store0 = prefer_vfmaq_f32(store0, pixel_colors_0, f_weight);
                        store1 = prefer_vfmaq_f32(store1, pixel_colors_1, f_weight);
                    }
                    r += diff as i32;
                }

                if CHANNEL_CONFIGURATION == 4 {
                    while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let pixel_colors_n_0 = vld1q_f32_x4(s_ptr.add(16));
                        let pixel_colors_n_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize).add(16));
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);

                        let f_weights0 = vld1q_f32(weight);
                        let f_weights1 = vld1q_f32(weight.add(4));

                        accumulate_4_items!(store0, pixel_colors_0, f_weights0);
                        accumulate_4_items!(store1, pixel_colors_1, f_weights0);

                        accumulate_4_items!(store0, pixel_colors_n_0, f_weights1);
                        accumulate_4_items!(store1, pixel_colors_n_1, f_weights1);

                        r += 8;
                    }

                    while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);

                        let f_weights = vld1q_f32(weight);

                        accumulate_4_items!(store0, pixel_colors_0, f_weights);
                        accumulate_4_items!(store1, pixel_colors_1, f_weights);

                        r += 4;
                    }
                } else if CHANNEL_CONFIGURATION == 3 {
                    while r + 5 <= half_kernel && ((x as i64 + r as i64 + 6i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        // (r0 g0 b0 r1) (g1 b1 r2 g2) (b2 r3 g3 b3) (r4 g4 b5 undef)
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        // r0 g0 b0 0
                        let set0 = vsplit_rgb_5(pixel_colors_0);
                        let set1 = vsplit_rgb_5(pixel_colors_1);
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                        let f_weights = vld1q_f32(weight);
                        let last_weight = weight.add(4).read_unaligned();

                        accumulate_5_items!(store0, set0, f_weights, last_weight);
                        accumulate_5_items!(store1, set1, f_weights, last_weight);

                        r += 5;
                    }
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let s_ptr_next = s_ptr.add(src_stride as usize);
                    let pixel_colors_1 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight.read_unaligned());
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_1, f_weight);

                    r += 1;
                }

                let offset = y_dst_shift + x as usize * CHANNEL_CONFIGURATION;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store0);

                let offset = offset + dst_stride as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store1);
            }
            _cy = y;
        }

        for y in _cy..end_y {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store: float32x4_t = vdupq_n_f32(0f32);

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let pixel_colors_f32 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    for i in 0..diff as usize {
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = vdupq_n_f32(weights.read_unaligned());
                        store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);
                    }
                    r += diff as i32;
                }

                if CHANNEL_CONFIGURATION == 4 {
                    while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_n_0 = vld1q_f32_x4(s_ptr.add(16));
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);

                        let f_weights0 = vld1q_f32(weight);
                        let f_weights1 = vld1q_f32(weight.add(4));

                        accumulate_4_items!(store, pixel_colors_0, f_weights0);
                        accumulate_4_items!(store, pixel_colors_n_0, f_weights1);

                        r += 8;
                    }

                    while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let px = vld1q_f32_x4(s_ptr);
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);

                        let f_weights = vld1q_f32(weight);

                        accumulate_4_items!(store, px, f_weights);

                        r += 4;
                    }
                } else if CHANNEL_CONFIGURATION == 3 {
                    while r + 5 <= half_kernel && ((x as i64 + r as i64 + 6i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        // (r0 g0 b0 r1) (g1 b1 r2 g2) (b2 r3 g3 b3) (r4 g4 b5 undef)
                        let pixel_colors_o = vld1q_f32_x4(s_ptr);
                        // r0 g0 b0 0
                        let set = vsplit_rgb_5(pixel_colors_o);
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                        let f_weights = vld1q_f32(weight);
                        let last_weight = weight.add(4).read_unaligned();

                        accumulate_5_items!(store, set, f_weights, last_weight);

                        r += 5;
                    }
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight.read_unaligned());
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 1;
                }

                let offset = y_dst_shift + x as usize * CHANNEL_CONFIGURATION;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store);
            }
        }
    }
}

pub fn gaussian_horiz_t_f_chan_filter_f32<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &Vec<GaussianFilter>,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let src: &[f32] = std::mem::transmute(undef_src);
        let unsafe_dst: &UnsafeSlice<'_, f32> = std::mem::transmute(undef_unsafe_dst);

        let mut _cy = start_y;

        for y in (_cy..end_y.saturating_sub(4)).step_by(4) {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store0: float32x4_t = vdupq_n_f32(0.);
                let mut store1: float32x4_t = vdupq_n_f32(0.);
                let mut store2: float32x4_t = vdupq_n_f32(0.);
                let mut store3: float32x4_t = vdupq_n_f32(0.);

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut r = 0;

                if CHANNEL_CONFIGURATION == 4 {
                    while r + 8 < current_filter.size
                        && ((filter_start as i64 + r as i64 + 8i64) < width as i64)
                    {
                        let current_x = std::cmp::min(
                            std::cmp::max(filter_start as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_0_n = vld1q_f32_x4(s_ptr.add(16));
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let pixel_colors_1_n = vld1q_f32_x4(s_ptr.add(src_stride as usize).add(16));
                        let pixel_colors_2 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 2));
                        let pixel_colors_2_n =
                            vld1q_f32_x4(s_ptr.add(src_stride as usize * 2).add(16));
                        let pixel_colors_3 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 3));
                        let pixel_colors_3_n =
                            vld1q_f32_x4(s_ptr.add(src_stride as usize * 3).add(16));
                        let weight = filter_weights.as_ptr().add(r);

                        let f_weights0 = vld1q_f32(weight);
                        let f_weights1 = vld1q_f32(weight.add(4));

                        accumulate_4_items!(store0, pixel_colors_0, f_weights0);
                        accumulate_4_items!(store1, pixel_colors_1, f_weights0);
                        accumulate_4_items!(store2, pixel_colors_2, f_weights0);
                        accumulate_4_items!(store3, pixel_colors_3, f_weights0);

                        accumulate_4_items!(store0, pixel_colors_0_n, f_weights1);
                        accumulate_4_items!(store1, pixel_colors_1_n, f_weights1);
                        accumulate_4_items!(store2, pixel_colors_2_n, f_weights1);
                        accumulate_4_items!(store3, pixel_colors_3_n, f_weights1);

                        r += 8;
                    }

                    while r + 4 < current_filter.size
                        && ((filter_start as i64 + r as i64 + 48i64) < width as i64)
                    {
                        let current_x = std::cmp::min(
                            std::cmp::max(filter_start as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let pixel_colors_2 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 2));
                        let pixel_colors_3 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 3));
                        let weight = filter_weights.as_ptr().add(r);
                        let f_weights = vld1q_f32(weight);

                        accumulate_4_items!(store0, pixel_colors_0, f_weights);
                        accumulate_4_items!(store1, pixel_colors_1, f_weights);
                        accumulate_4_items!(store2, pixel_colors_2, f_weights);
                        accumulate_4_items!(store3, pixel_colors_3, f_weights);

                        r += 4;
                    }
                } else if CHANNEL_CONFIGURATION == 3 {
                    while r + 5 < current_filter.size
                        && ((filter_start as i64 + r as i64 + 6i64) < width as i64)
                    {
                        let current_x = std::cmp::min(
                            std::cmp::max(filter_start as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        // (r0 g0 b0 r1) (g1 b1 r2 g2) (b2 r3 g3 b3) (r4 g4 b5 undef)
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let pixel_colors_2 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 2));
                        let pixel_colors_3 = vld1q_f32_x4(s_ptr.add(src_stride as usize * 3));
                        let weight = filter_weights.as_ptr().add(r);
                        let f_weights = vld1q_f32(weight);
                        let last_weight = weight.add(4).read_unaligned();
                        // r0 g0 b0 0
                        let set0 = vsplit_rgb_5(pixel_colors_0);
                        let set1 = vsplit_rgb_5(pixel_colors_1);
                        let set2 = vsplit_rgb_5(pixel_colors_2);
                        let set3 = vsplit_rgb_5(pixel_colors_3);

                        accumulate_5_items!(store0, set0, f_weights, last_weight);
                        accumulate_5_items!(store1, set1, f_weights, last_weight);
                        accumulate_5_items!(store2, set2, f_weights, last_weight);
                        accumulate_5_items!(store3, set3, f_weights, last_weight);

                        r += 5;
                    }
                }

                while r < current_filter.size {
                    let current_x = std::cmp::min(
                        std::cmp::max(filter_start as i64 + r as i64, 0),
                        (width - 1) as i64,
                    ) as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let s_ptr_next = s_ptr.add(src_stride as usize);
                    let pixel_colors_1 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next);
                    let s_ptr_next_2 = s_ptr.add(src_stride as usize * 2);
                    let pixel_colors_2 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_2);
                    let s_ptr_next_3 = s_ptr.add(src_stride as usize * 3);
                    let pixel_colors_3 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next_3);
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight: float32x4_t = vdupq_n_f32(weight.read_unaligned());
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_1, f_weight);
                    store2 = prefer_vfmaq_f32(store2, pixel_colors_2, f_weight);
                    store3 = prefer_vfmaq_f32(store3, pixel_colors_3, f_weight);

                    r += 1;
                }

                let offset = y_dst_shift + x as usize * CHANNEL_CONFIGURATION;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store0);

                let offset = offset + dst_stride as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store1);

                let offset = offset + dst_stride as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store2);

                let offset = offset + dst_stride as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store3);
            }
            _cy = y;
        }

        for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store0: float32x4_t = vdupq_n_f32(0.);
                let mut store1: float32x4_t = vdupq_n_f32(0.);

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut r = 0;

                if CHANNEL_CONFIGURATION == 4 {
                    while r + 8 < current_filter.size
                        && ((filter_start as i64 + r as i64 + 8i64) < width as i64)
                    {
                        let current_x = std::cmp::min(
                            std::cmp::max(filter_start as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let pixel_colors_n_0 = vld1q_f32_x4(s_ptr.add(16));
                        let pixel_colors_n_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize).add(16));
                        let weight = filter_weights.as_ptr().add(r);

                        let f_weights0 = vld1q_f32(weight);
                        let f_weights1 = vld1q_f32(weight.add(4));

                        accumulate_4_items!(store0, pixel_colors_0, f_weights0);
                        accumulate_4_items!(store1, pixel_colors_1, f_weights0);

                        accumulate_4_items!(store0, pixel_colors_n_0, f_weights1);
                        accumulate_4_items!(store1, pixel_colors_n_1, f_weights1);

                        r += 8;
                    }

                    while r + 4 < current_filter.size
                        && ((filter_start as i64 + r as i64 + 4i64) < width as i64)
                    {
                        let current_x = std::cmp::min(
                            std::cmp::max(filter_start as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        let weight = filter_weights.as_ptr().add(r);

                        let f_weights = vld1q_f32(weight);

                        accumulate_4_items!(store0, pixel_colors_0, f_weights);
                        accumulate_4_items!(store1, pixel_colors_1, f_weights);

                        r += 4;
                    }
                } else if CHANNEL_CONFIGURATION == 3 {
                    while r + 5 < current_filter.size
                        && ((filter_start as i64 + r as i64 + 6i64) < width as i64)
                    {
                        let current_x = std::cmp::min(
                            std::cmp::max(filter_start as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        // (r0 g0 b0 r1) (g1 b1 r2 g2) (b2 r3 g3 b3) (r4 g4 b5 undef)
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_1 = vld1q_f32_x4(s_ptr.add(src_stride as usize));
                        // r0 g0 b0 0
                        let set0 = vsplit_rgb_5(pixel_colors_0);
                        let set1 = vsplit_rgb_5(pixel_colors_1);
                        let weight = filter_weights.as_ptr().add(r);
                        let f_weights = vld1q_f32(weight);
                        let last_weight = weight.add(4).read_unaligned();

                        accumulate_5_items!(store0, set0, f_weights, last_weight);
                        accumulate_5_items!(store1, set1, f_weights, last_weight);

                        r += 5;
                    }
                }

                while r < current_filter.size {
                    let current_x = std::cmp::min(
                        std::cmp::max(filter_start as i64 + r as i64, 0),
                        (width - 1) as i64,
                    ) as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let s_ptr_next = s_ptr.add(src_stride as usize);
                    let pixel_colors_1 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr_next);
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight: float32x4_t = vdupq_n_f32(weight.read_unaligned());
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_1, f_weight);

                    r += 1;
                }

                let offset = y_dst_shift + x as usize * CHANNEL_CONFIGURATION;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store0);

                let offset = offset + dst_stride as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store1);
            }
            _cy = y;
        }

        for y in _cy..end_y {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store: float32x4_t = vdupq_n_f32(0f32);

                let mut r = 0;

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                if CHANNEL_CONFIGURATION == 4 {
                    while r + 8 < current_filter.size
                        && ((filter_start as i64 + r as i64 + 8i64) < width as i64)
                    {
                        let current_x = std::cmp::min(
                            std::cmp::max(filter_start as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_f32_x4(s_ptr);
                        let pixel_colors_n_0 = vld1q_f32_x4(s_ptr.add(16));
                        let weight = filter_weights.as_ptr().add(r);

                        let f_weights0 = vld1q_f32(weight);
                        let f_weights1 = vld1q_f32(weight.add(4));

                        accumulate_4_items!(store, pixel_colors_0, f_weights0);

                        accumulate_4_items!(store, pixel_colors_n_0, f_weights1);

                        r += 8;
                    }

                    while r + 4 < current_filter.size
                        && ((filter_start as i64 + r as i64 + 4i64) < width as i64)
                    {
                        let current_x = std::cmp::min(
                            std::cmp::max(filter_start as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let px = vld1q_f32_x4(s_ptr);
                        let weight = filter_weights.as_ptr().add(r);

                        let f_weights = vld1q_f32(weight);

                        accumulate_4_items!(store, px, f_weights);

                        r += 4;
                    }
                } else if CHANNEL_CONFIGURATION == 3 {
                    while r + 5 < current_filter.size
                        && ((filter_start as i64 + r as i64 + 6i64) < width as i64)
                    {
                        let current_x = std::cmp::min(
                            std::cmp::max(filter_start as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        // (r0 g0 b0 r1) (g1 b1 r2 g2) (b2 r3 g3 b3) (r4 g4 b5 undef)
                        let pixel_colors_o = vld1q_f32_x4(s_ptr);
                        // r0 g0 b0 0
                        let set = vsplit_rgb_5(pixel_colors_o);
                        let weight = filter_weights.as_ptr().add(r);
                        let f_weights = vld1q_f32(weight);
                        let last_weight = weight.add(4).read_unaligned();

                        accumulate_5_items!(store, set, f_weights, last_weight);

                        r += 5;
                    }
                }

                while r < current_filter.size {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = load_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight: float32x4_t = vdupq_n_f32(weight.read_unaligned());
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 1;
                }

                let offset = y_dst_shift + x as usize * CHANNEL_CONFIGURATION;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store);
            }
        }
    }
}
