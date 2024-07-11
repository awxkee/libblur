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

use crate::neon::{load_f32_fast, prefer_vfmaq_f32, store_f32, vsplit_rgb};
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;

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

                        store0 = prefer_vfmaq_f32(
                            store0,
                            pixel_colors_0.0,
                            vdupq_n_f32(vgetq_lane_f32::<0>(f_weights)),
                        );
                        store0 = prefer_vfmaq_f32(
                            store0,
                            pixel_colors_0.1,
                            vdupq_n_f32(vgetq_lane_f32::<1>(f_weights)),
                        );
                        store0 = prefer_vfmaq_f32(
                            store0,
                            pixel_colors_0.2,
                            vdupq_n_f32(vgetq_lane_f32::<2>(f_weights)),
                        );
                        store0 = prefer_vfmaq_f32(
                            store0,
                            pixel_colors_0.3,
                            vdupq_n_f32(vgetq_lane_f32::<3>(f_weights)),
                        );

                        store1 = prefer_vfmaq_f32(
                            store1,
                            pixel_colors_1.0,
                            vdupq_n_f32(vgetq_lane_f32::<0>(f_weights)),
                        );
                        store1 = prefer_vfmaq_f32(
                            store1,
                            pixel_colors_1.1,
                            vdupq_n_f32(vgetq_lane_f32::<1>(f_weights)),
                        );
                        store1 = prefer_vfmaq_f32(
                            store1,
                            pixel_colors_1.2,
                            vdupq_n_f32(vgetq_lane_f32::<2>(f_weights)),
                        );
                        store1 = prefer_vfmaq_f32(
                            store1,
                            pixel_colors_1.3,
                            vdupq_n_f32(vgetq_lane_f32::<3>(f_weights)),
                        );

                        r += 4;
                    }
                } else if CHANNEL_CONFIGURATION == 3 {
                    while r + 4 <= half_kernel && ((x as i64 + r as i64 + 6i64) < width as i64) {
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
                        let (p0, p1, p2, p3) = vsplit_rgb(pixel_colors_0);
                        let (p4, p5, p6, p7) = vsplit_rgb(pixel_colors_1);
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                        let f_weights = vld1q_f32(weight);

                        store0 = prefer_vfmaq_f32(
                            store0,
                            p0,
                            vdupq_n_f32(vgetq_lane_f32::<0>(f_weights)),
                        );
                        store0 = prefer_vfmaq_f32(
                            store0,
                            p1,
                            vdupq_n_f32(vgetq_lane_f32::<1>(f_weights)),
                        );
                        store0 = prefer_vfmaq_f32(
                            store0,
                            p2,
                            vdupq_n_f32(vgetq_lane_f32::<2>(f_weights)),
                        );
                        store0 = prefer_vfmaq_f32(
                            store0,
                            p3,
                            vdupq_n_f32(vgetq_lane_f32::<3>(f_weights)),
                        );

                        store1 = prefer_vfmaq_f32(
                            store1,
                            p4,
                            vdupq_n_f32(vgetq_lane_f32::<0>(f_weights)),
                        );
                        store1 = prefer_vfmaq_f32(
                            store1,
                            p5,
                            vdupq_n_f32(vgetq_lane_f32::<1>(f_weights)),
                        );
                        store1 = prefer_vfmaq_f32(
                            store1,
                            p6,
                            vdupq_n_f32(vgetq_lane_f32::<2>(f_weights)),
                        );
                        store1 = prefer_vfmaq_f32(
                            store1,
                            p7,
                            vdupq_n_f32(vgetq_lane_f32::<3>(f_weights)),
                        );

                        r += 4;
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

                        store = prefer_vfmaq_f32(
                            store,
                            px.0,
                            vdupq_n_f32(vgetq_lane_f32::<0>(f_weights)),
                        );
                        store = prefer_vfmaq_f32(
                            store,
                            px.1,
                            vdupq_n_f32(vgetq_lane_f32::<1>(f_weights)),
                        );
                        store = prefer_vfmaq_f32(
                            store,
                            px.2,
                            vdupq_n_f32(vgetq_lane_f32::<2>(f_weights)),
                        );
                        store = prefer_vfmaq_f32(
                            store,
                            px.3,
                            vdupq_n_f32(vgetq_lane_f32::<3>(f_weights)),
                        );

                        r += 4;
                    }
                } else if CHANNEL_CONFIGURATION == 3 {
                    while r + 4 <= half_kernel && ((x as i64 + r as i64 + 6i64) < width as i64) {
                        let current_x = std::cmp::min(
                            std::cmp::max(x as i64 + r as i64, 0),
                            (width - 1) as i64,
                        ) as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        // (r0 g0 b0 r1) (g1 b1 r2 g2) (b2 r3 g3 b3) (r4 g4 b5 undef)
                        let pixel_colors_o = vld1q_f32_x4(s_ptr);
                        // r0 g0 b0 0
                        let (p0, p1, p2, p3) = vsplit_rgb(pixel_colors_o);
                        let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                        let f_weights = vld1q_f32(weight);

                        store = prefer_vfmaq_f32(
                            store,
                            p0,
                            vdupq_n_f32(vgetq_lane_f32::<0>(f_weights)),
                        );
                        store = prefer_vfmaq_f32(
                            store,
                            p1,
                            vdupq_n_f32(vgetq_lane_f32::<1>(f_weights)),
                        );
                        store = prefer_vfmaq_f32(
                            store,
                            p2,
                            vdupq_n_f32(vgetq_lane_f32::<2>(f_weights)),
                        );
                        store = prefer_vfmaq_f32(
                            store,
                            p3,
                            vdupq_n_f32(vgetq_lane_f32::<3>(f_weights)),
                        );

                        r += 4;
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
