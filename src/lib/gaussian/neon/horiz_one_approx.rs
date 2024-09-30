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

use crate::gaussian::gaussian_approx::{PRECISION, ROUNDING_APPROX};
use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::neon::load_u8_u16_fast;
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, reflect_101, reflect_index, EdgeMode};
use std::arch::aarch64::*;

#[inline]
pub(crate) unsafe fn vhsumq_s32(a: int32x4_t) -> u8 {
    let va = vadd_s32(vget_low_s32(a), vget_high_s32(a));
    let accum = vshr_n_s32::<PRECISION>(vpadd_s32(va, va));
    let compressed = vmin_s32(vmax_s32(accum, vdup_n_s32(0)), vdup_n_s32(255));
    vget_lane_s32::<0>(compressed) as u8
}

macro_rules! accumulate_4_forward {
    ($store:expr, $pixel_colors:expr, $weights:expr) => {{
        let pixel_colors_low_u16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8($pixel_colors)));
        $store = vmlal_s16(
            $store,
            vget_low_s16(pixel_colors_low_u16),
            vget_low_s16($weights.0),
        );
        $store = vmlal_high_s16($store, pixel_colors_low_u16, $weights.0);

        let pixel_colors_high_u16 = vreinterpretq_s16_u16(vmovl_high_u8($pixel_colors));
        $store = vmlal_s16(
            $store,
            vget_low_s16(pixel_colors_high_u16),
            vget_low_s16($weights.1),
        );
        $store = vmlal_high_s16($store, pixel_colors_high_u16, $weights.1);
    }};
}

macro_rules! accumulate_2_forward {
    ($store:expr, $pixel_colors:expr, $weights:expr) => {{
        let pixel_u16 = vreinterpretq_s16_u16(vmovl_u8($pixel_colors));
        $store = vmlal_s16($store, vget_low_s16(pixel_u16), vget_low_s16($weights));
        $store = vmlal_high_s16($store, pixel_u16, $weights);
    }};
}

pub fn gaussian_horiz_one_approx_u8(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[i16],
    start_y: u32,
    end_y: u32,
    edge_mode: EdgeMode,
) {
    let half_kernel = (kernel_size / 2) as i32;

    let mut _cy = start_y;

    let initial_value = unsafe { vsetq_lane_s32::<0>(ROUNDING_APPROX, vdupq_n_s32(0)) };

    for y in (_cy..end_y.saturating_sub(4)).step_by(4) {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let y_src_shift_next = y_src_shift + src_stride as usize;

                let mut store0 = initial_value;
                let mut store1 = initial_value;
                let mut store2 = initial_value;
                let mut store3 = initial_value;

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();

                    let start_x = edge_value_check;

                    for i in 0..diff as usize {
                        let s_ptr = src.as_ptr().add(
                            y_src_shift
                                + clamp_edge!(edge_mode, start_x + i as i64, 0, width as i64 - 1),
                        );
                        let pixel_colors_0 =
                            vset_lane_s16::<0>(s_ptr.read_unaligned() as i16, vdup_n_s16(0));
                        let s_ptr_next = src.as_ptr().add(y_src_shift_next); // Here we're always at zero
                        let pixel_colors_1 =
                            vset_lane_s16::<0>(s_ptr_next.read_unaligned() as i16, vdup_n_s16(0));

                        let s_ptr_next_2 = src.as_ptr().add(y_src_shift_next + src_stride as usize); // Here we're always at zero
                        let pixel_colors_2 =
                            vset_lane_s16::<0>(s_ptr_next_2.read_unaligned() as i16, vdup_n_s16(0));

                        let s_ptr_next_3 =
                            src.as_ptr().add(y_src_shift_next + src_stride as usize * 2); // Here we're always at zero
                        let pixel_colors_3 =
                            vset_lane_s16::<0>(s_ptr_next_3.read_unaligned() as i16, vdup_n_s16(0));

                        let weights = kernel.as_ptr().add(i);
                        let f_weight = vld1_lane_s16::<0>(weights, vdup_n_s16(0));
                        store0 = vmlal_s16(store0, pixel_colors_0, f_weight);
                        store1 = vmlal_s16(store1, pixel_colors_1, f_weight);
                        store2 = vmlal_s16(store2, pixel_colors_2, f_weight);
                        store3 = vmlal_s16(store3, pixel_colors_3, f_weight);
                    }
                    r += diff as i32;
                }

                while r + 32 <= half_kernel && ((x as i64 + r as i64 + 32i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8x2_0 = vld1q_u8_x2(s_ptr);
                    let pixel_colors_u8x2_1 = vld1q_u8_x2(s_ptr_next);
                    let pixel_colors_u8x2_2 = vld1q_u8_x2(s_ptr_next.add(src_stride as usize));
                    let pixel_colors_u8x2_3 = vld1q_u8_x2(s_ptr_next.add(src_stride as usize * 2));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = vld1q_s16_x4(weight);

                    accumulate_4_forward!(store0, pixel_colors_u8x2_0.0, (weights.0, weights.1));
                    accumulate_4_forward!(store0, pixel_colors_u8x2_0.1, (weights.2, weights.3));

                    accumulate_4_forward!(store1, pixel_colors_u8x2_1.0, (weights.0, weights.1));
                    accumulate_4_forward!(store1, pixel_colors_u8x2_1.1, (weights.2, weights.3));

                    accumulate_4_forward!(store2, pixel_colors_u8x2_2.0, (weights.0, weights.1));
                    accumulate_4_forward!(store2, pixel_colors_u8x2_2.1, (weights.2, weights.3));

                    accumulate_4_forward!(store3, pixel_colors_u8x2_3.0, (weights.0, weights.1));
                    accumulate_4_forward!(store3, pixel_colors_u8x2_3.1, (weights.2, weights.3));

                    r += 32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8_0 = vld1q_u8(s_ptr);
                    let pixel_colors_u8_1 = vld1q_u8(s_ptr_next);
                    let pixel_colors_u8_2 = vld1q_u8(s_ptr_next.add(src_stride as usize));
                    let pixel_colors_u8_3 = vld1q_u8(s_ptr_next.add(src_stride as usize * 2));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = vld1q_s16_x2(weight);

                    accumulate_4_forward!(store0, pixel_colors_u8_0, weights);
                    accumulate_4_forward!(store1, pixel_colors_u8_1, weights);
                    accumulate_4_forward!(store2, pixel_colors_u8_2, weights);
                    accumulate_4_forward!(store3, pixel_colors_u8_3, weights);

                    r += 16;
                }

                while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8_0 = vld1_u8(s_ptr);
                    let pixel_colors_u8_1 = vld1_u8(s_ptr_next);
                    let pixel_colors_u8_2 = vld1_u8(s_ptr_next.add(src_stride as usize));
                    let pixel_colors_u8_3 = vld1_u8(s_ptr_next.add(src_stride as usize * 2));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = vld1q_s16(weight);

                    accumulate_2_forward!(store0, pixel_colors_u8_0, weights);
                    accumulate_2_forward!(store1, pixel_colors_u8_1, weights);
                    accumulate_2_forward!(store2, pixel_colors_u8_2, weights);
                    accumulate_2_forward!(store3, pixel_colors_u8_3, weights);

                    r += 8;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr));

                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_1 = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr_next));

                    let s_ptr_next_2 = s_ptr_next.add(src_stride as usize);
                    let pixel_colors_2 = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr_next_2));

                    let s_ptr_next_3 = s_ptr_next.add(src_stride as usize * 2);
                    let pixel_colors_3 = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr_next_3));

                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = vld1_s16(weight);
                    store0 = vmlal_s16(store0, pixel_colors_0, f_weight);
                    store1 = vmlal_s16(store1, pixel_colors_1, f_weight);
                    store2 = vmlal_s16(store2, pixel_colors_2, f_weight);
                    store3 = vmlal_s16(store3, pixel_colors_3, f_weight);

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        clamp_edge!(edge_mode, x as i64 + r as i64, 0, width as i64 - 1);
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_0 =
                        vset_lane_s16::<0>(s_ptr.read_unaligned() as i16, vdup_n_s16(0));
                    let pixel_colors_f32_1 =
                        vset_lane_s16::<0>(s_ptr_next.read_unaligned() as i16, vdup_n_s16(0));
                    let pixel_colors_f32_2 = vset_lane_s16::<0>(
                        s_ptr_next.add(src_stride as usize).read_unaligned() as i16,
                        vdup_n_s16(0),
                    );
                    let pixel_colors_f32_3 = vset_lane_s16::<0>(
                        s_ptr_next.add(src_stride as usize * 2).read_unaligned() as i16,
                        vdup_n_s16(0),
                    );
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = vld1_dup_s16(weight);
                    store0 = vmlal_s16(store0, pixel_colors_f32_0, f_weight);
                    store1 = vmlal_s16(store1, pixel_colors_f32_1, f_weight);
                    store2 = vmlal_s16(store2, pixel_colors_f32_2, f_weight);
                    store3 = vmlal_s16(store3, pixel_colors_f32_3, f_weight);

                    r += 1;
                }

                let agg0 = vhsumq_s32(store0);
                let offset0 = y_dst_shift + x as usize;
                let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset0) as *mut u8;
                dst_ptr0.write_unaligned(agg0);

                let agg1 = vhsumq_s32(store1);
                let offset1 = offset0 + dst_stride as usize;
                let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset1) as *mut u8;
                dst_ptr1.write_unaligned(agg1);

                let agg2 = vhsumq_s32(store2);
                let offset2 = offset0 + dst_stride as usize * 2;
                let dst_ptr2 = unsafe_dst.slice.as_ptr().add(offset2) as *mut u8;
                dst_ptr2.write_unaligned(agg2);

                let agg3 = vhsumq_s32(store3);
                let offset3 = offset0 + dst_stride as usize * 3;
                let dst_ptr3 = unsafe_dst.slice.as_ptr().add(offset3) as *mut u8;
                dst_ptr3.write_unaligned(agg3);
            }
        }
        _cy = y;
    }

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let y_src_shift_next = y_src_shift + src_stride as usize;

                let mut store0 = initial_value;
                let mut store1 = initial_value;

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();

                    let start_x = edge_value_check;

                    for i in 0..diff as usize {
                        let s_ptr = src.as_ptr().add(
                            y_src_shift
                                + clamp_edge!(edge_mode, start_x + i as i64, 0, width as i64 - 1),
                        );
                        let pixel_colors_f32_0 =
                            vset_lane_s16::<0>(s_ptr.read_unaligned() as i16, vdup_n_s16(0));
                        let s_ptr_next = src.as_ptr().add(y_src_shift_next); // Here we're always at zero
                        let pixel_colors_f32_1 =
                            vset_lane_s16::<0>(s_ptr_next.read_unaligned() as i16, vdup_n_s16(0));

                        let weights = kernel.as_ptr().add(i);
                        let f_weight = vld1_dup_s16(weights);
                        store0 = vmlal_s16(store0, pixel_colors_f32_0, f_weight);
                        store1 = vmlal_s16(store1, pixel_colors_f32_1, f_weight);
                    }
                    r += diff as i32;
                }

                while r + 32 <= half_kernel && ((x as i64 + r as i64 + 32i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8x2 = vld1q_u8_x2(s_ptr);
                    let pixel_colors_u8x2_next = vld1q_u8_x2(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = vld1q_s16_x4(weight);

                    accumulate_4_forward!(store0, pixel_colors_u8x2.0, (weights.0, weights.1));
                    accumulate_4_forward!(store0, pixel_colors_u8x2.1, (weights.2, weights.3));

                    // Next row

                    accumulate_4_forward!(store1, pixel_colors_u8x2_next.0, (weights.0, weights.1));
                    accumulate_4_forward!(store1, pixel_colors_u8x2_next.1, (weights.2, weights.3));

                    r += 32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8_0 = vld1q_u8(s_ptr);
                    let pixel_colors_u8_1 = vld1q_u8(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = vld1q_s16_x2(weight);

                    accumulate_4_forward!(store0, pixel_colors_u8_0, weights);
                    accumulate_4_forward!(store1, pixel_colors_u8_1, weights);

                    r += 16;
                }

                while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8_0 = vld1_u8(s_ptr);
                    let pixel_colors_u8_1 = vld1_u8(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = vld1q_s16(weight);

                    accumulate_2_forward!(store0, pixel_colors_u8_0, weights);
                    accumulate_2_forward!(store1, pixel_colors_u8_1, weights);

                    r += 8;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr));

                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_1 = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr_next));

                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = vld1_s16(weight);
                    store0 = vmlal_s16(store0, pixel_colors_0, f_weight);
                    store1 = vmlal_s16(store1, pixel_colors_1, f_weight);

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        clamp_edge!(edge_mode, x as i64 + r as i64, 0, width as i64 - 1);
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_0 =
                        vset_lane_s16::<0>(s_ptr.read_unaligned() as i16, vdup_n_s16(0));
                    let pixel_colors_1 =
                        vset_lane_s16::<0>(s_ptr_next.read_unaligned() as i16, vdup_n_s16(0));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = vld1_dup_s16(weight);
                    store0 = vmlal_s16(store0, pixel_colors_0, f_weight);
                    store1 = vmlal_s16(store1, pixel_colors_1, f_weight);

                    r += 1;
                }

                let agg0 = vhsumq_s32(store0);
                let offset0 = y_dst_shift + x as usize;
                let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset0) as *mut u8;
                dst_ptr0.write_unaligned(agg0);

                let agg1 = vhsumq_s32(store1);
                let offset1 = offset0 + dst_stride as usize;
                let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset1) as *mut u8;
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

                let mut store = initial_value;

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();

                    let start_x = edge_value_check;

                    for i in 0..diff as usize {
                        let s_ptr = src.as_ptr().add(
                            y_src_shift
                                + clamp_edge!(edge_mode, start_x + i as i64, 0, width as i64 - 1),
                        );
                        let pixel_colors =
                            vset_lane_s16::<0>(s_ptr.read_unaligned() as i16, vdup_n_s16(0));

                        let weights = kernel.as_ptr().add(i);
                        let weight = vld1_dup_s16(weights);
                        store = vmlal_s16(store, pixel_colors, weight);
                    }
                    r += diff as i32;
                }

                while r + 32 <= half_kernel && ((x as i64 + r as i64 + 32i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u8x2 = vld1q_u8_x2(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = vld1q_s16_x4(weight);

                    accumulate_4_forward!(store, pixel_colors_u8x2.0, (weights.0, weights.1));
                    accumulate_4_forward!(store, pixel_colors_u8x2.1, (weights.2, weights.3));

                    r += 32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u8 = vld1q_u8(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = vld1q_s16_x2(weight);

                    accumulate_4_forward!(store, pixel_colors_u8, weights);

                    r += 16;
                }

                while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u8 = vld1_u8(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = vld1q_s16(weight);

                    accumulate_2_forward!(store, pixel_colors_u8, weights);

                    r += 8;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x = (x as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = vld1_s16(weight);
                    store = vmlal_s16(store, pixel_colors, f_weight);

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        clamp_edge!(edge_mode, x as i64 + r as i64, 0, width as i64 - 1);
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors =
                        vset_lane_s16::<0>(s_ptr.read_unaligned() as i16, vdup_n_s16(0));
                    let weights = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = vld1_dup_s16(weights);
                    store = vmlal_s16(store, pixel_colors, f_weight);

                    r += 1;
                }

                let agg = vhsumq_s32(store);
                let offset = y_dst_shift + x as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut u8;
                dst_ptr.write_unaligned(agg);
            }
        }
    }
}

pub fn gaussian_horiz_one_chan_filter_approx(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    filter: &[GaussianFilter<i16>],
    start_y: u32,
    end_y: u32,
) {
    let mut _cy = start_y;

    let initial_value = unsafe { vsetq_lane_s32::<0>(ROUNDING_APPROX, vdupq_n_s32(0)) };

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        unsafe {
            for x in 0..width {
                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let y_src_shift_next = y_src_shift + src_stride as usize;

                let mut store0 = initial_value;
                let mut store1 = initial_value;

                let mut r = 0;

                while r + 32 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 32i64) < width as i64)
                {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8x2 = vld1q_u8_x2(s_ptr);
                    let pixel_colors_u8x2_next = vld1q_u8_x2(s_ptr_next);
                    let weight = filter_weights.as_ptr().add(r);
                    let full_weights = vld1q_s16_x4(weight);

                    accumulate_4_forward!(
                        store0,
                        pixel_colors_u8x2.0,
                        (full_weights.0, full_weights.1)
                    );
                    accumulate_4_forward!(
                        store0,
                        pixel_colors_u8x2.1,
                        (full_weights.2, full_weights.3)
                    );

                    // Next row

                    accumulate_4_forward!(
                        store1,
                        pixel_colors_u8x2_next.0,
                        (full_weights.0, full_weights.1)
                    );
                    accumulate_4_forward!(
                        store1,
                        pixel_colors_u8x2_next.1,
                        (full_weights.2, full_weights.3)
                    );

                    r += 32;
                }

                while r + 16 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 16i64) < width as i64)
                {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8_0 = vld1q_u8(s_ptr);
                    let pixel_colors_u8_1 = vld1q_u8(s_ptr_next);
                    let weight = filter_weights.as_ptr().add(r);
                    let weights = vld1q_s16_x2(weight);

                    accumulate_4_forward!(store0, pixel_colors_u8_0, weights);
                    accumulate_4_forward!(store1, pixel_colors_u8_1, weights);

                    r += 16;
                }

                while r + 8 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 8i64) < width as i64)
                {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8_0 = vld1_u8(s_ptr);
                    let pixel_colors_u8_1 = vld1_u8(s_ptr_next);
                    let weight = filter_weights.as_ptr().add(r);
                    let weights = vld1q_s16(weight);

                    accumulate_2_forward!(store0, pixel_colors_u8_0, weights);
                    accumulate_2_forward!(store1, pixel_colors_u8_1, weights);

                    r += 8;
                }

                while r + 4 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 4i64) < width as i64)
                {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr));

                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_1 = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr_next));

                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight = vld1_s16(weight);
                    store0 = vmlal_s16(store0, pixel_colors_0, f_weight);
                    store1 = vmlal_s16(store1, pixel_colors_1, f_weight);

                    r += 4;
                }

                while r < current_filter.size {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_0 =
                        vset_lane_s16::<0>(s_ptr.read_unaligned() as i16, vdup_n_s16(0));
                    let pixel_colors_1 =
                        vset_lane_s16::<0>(s_ptr_next.read_unaligned() as i16, vdup_n_s16(0));
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight = vld1_dup_s16(weight);
                    store0 = vmlal_s16(store0, pixel_colors_0, f_weight);
                    store1 = vmlal_s16(store1, pixel_colors_1, f_weight);

                    r += 1;
                }

                let agg0 = vhsumq_s32(store0);
                let offset0 = y_dst_shift + x as usize;
                let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset0) as *mut u8;
                dst_ptr0.write_unaligned(agg0);

                let agg1 = vhsumq_s32(store1);
                let offset1 = offset0 + dst_stride as usize;
                let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset1) as *mut u8;
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

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut store = initial_value;

                let mut r = 0;

                while r + 32 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 32i64) < width as i64)
                {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u8x2 = vld1q_u8_x2(s_ptr);
                    let weight = filter_weights.as_ptr().add(r);
                    let weights = vld1q_s16_x4(weight);

                    accumulate_4_forward!(store, pixel_colors_u8x2.0, (weights.0, weights.1));
                    accumulate_4_forward!(store, pixel_colors_u8x2.1, (weights.2, weights.3));

                    r += 32;
                }

                while r + 16 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 16i64) < width as i64)
                {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u8 = vld1q_u8(s_ptr);
                    let weight = filter_weights.as_ptr().add(r);
                    let weights = vld1q_s16_x2(weight);

                    accumulate_4_forward!(store, pixel_colors_u8, weights);

                    r += 16;
                }

                while r + 8 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 8i64) < width as i64)
                {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_u8 = vld1_u8(s_ptr);
                    let weight = filter_weights.as_ptr().add(r);
                    let weights = vld1q_s16(weight);

                    accumulate_2_forward!(store, pixel_colors_u8, weights);

                    r += 8;
                }

                while r + 4 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 4i64) < width as i64)
                {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr));
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight = vld1_dup_s16(weight);
                    store = vmlal_s16(store, pixel_colors, f_weight);

                    r += 4;
                }

                while r < current_filter.size {
                    let current_x = (filter_start as i64 + r as i64) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let value = vset_lane_s16::<0>(s_ptr.read_unaligned() as i16, vdup_n_s16(0));
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight = vld1_dup_s16(weight);
                    store = vmlal_s16(store, value, f_weight);

                    r += 1;
                }

                let agg = vhsumq_s32(store);
                let offset = y_dst_shift + x as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut u8;
                dst_ptr.write_unaligned(agg);
            }
        }
    }
}
