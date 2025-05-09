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

use crate::neon::fast_gaussian::NeonI32x4;
use crate::neon::{load_u8_s32_fast, store_u8_s32_x4, store_u8x8_m4, vmulq_by_3_s32};
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, EdgeMode};
use std::arch::aarch64::*;

pub(crate) fn fgn_vertical_pass_neon_u8_rdm<T, const CN: usize>(
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
        fgn_vertical_pass_neon_impl_rdm::<T, CN>(
            undefined_slice,
            stride,
            width,
            height,
            radius,
            start,
            end,
            edge_mode,
        );
    }
}

#[target_feature(enable = "rdm")]
unsafe fn fgn_vertical_pass_neon_impl_rdm<T, const CN: usize>(
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

        let mut full_buffer = Box::new([NeonI32x4::default(); 1024 * 4]);

        let (buffer0, rem) = full_buffer.split_at_mut(1024);
        let (buffer1, rem) = rem.split_at_mut(1024);
        let (buffer2, buffer3) = rem.split_at_mut(1024);

        let height_wide = height as i64;

        let radius_64 = radius as i64;
        const Q: f64 = ((1i64 << 31i64) - 1) as f64;
        let weight = Q / ((radius as f64) * (radius as f64) * (radius as f64));
        let v_weight = vdupq_n_s32(weight as i32);

        let mut xx = start;

        while xx + 4 < width.min(end) {
            let mut diffs0 = vdupq_n_s32(0);
            let mut diffs1 = vdupq_n_s32(0);
            let mut diffs2 = vdupq_n_s32(0);
            let mut diffs3 = vdupq_n_s32(0);

            let mut ders0 = vdupq_n_s32(0);
            let mut ders1 = vdupq_n_s32(0);
            let mut ders2 = vdupq_n_s32(0);
            let mut ders3 = vdupq_n_s32(0);

            let mut summs0 = vdupq_n_s32(0);
            let mut summs1 = vdupq_n_s32(0);
            let mut summs2 = vdupq_n_s32(0);
            let mut summs3 = vdupq_n_s32(0);

            let start_y = 0 - 3 * radius as i64;

            let current_px0 = (xx * CN as u32) as usize;
            let current_px1 = ((xx + 1) * CN as u32) as usize;
            let current_px2 = ((xx + 2) * CN as u32) as usize;
            let current_px3 = ((xx + 3) * CN as u32) as usize;

            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;

                if y >= 0 {
                    let prepared_px0 = vqrdmulhq_s32(summs0, v_weight);
                    let prepared_px1 = vqrdmulhq_s32(summs1, v_weight);
                    let prepared_px2 = vqrdmulhq_s32(summs2, v_weight);
                    let prepared_px3 = vqrdmulhq_s32(summs3, v_weight);

                    let dst_ptr0 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px0);
                    let dst_ptr1 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px1);
                    let dst_ptr2 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px2);
                    let dst_ptr3 = (bytes.slice.as_ptr() as *mut u8).add(current_y + current_px3);

                    store_u8_s32_x4::<CN>(
                        (dst_ptr0, dst_ptr1, dst_ptr2, dst_ptr3),
                        int32x4x4_t(prepared_px0, prepared_px1, prepared_px2, prepared_px3),
                    );

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let stored0 = vld1q_s32(buffer0.as_mut_ptr().add(d_arr_index) as *const i32);
                    let stored1 = vld1q_s32(buffer1.as_mut_ptr().add(d_arr_index) as *const i32);
                    let stored2 = vld1q_s32(buffer2.as_mut_ptr().add(d_arr_index) as *const i32);
                    let stored3 = vld1q_s32(buffer3.as_mut_ptr().add(d_arr_index) as *const i32);

                    let stored_10 =
                        vld1q_s32(buffer0.as_mut_ptr().add(d_arr_index_1) as *const i32);
                    let stored_11 =
                        vld1q_s32(buffer1.as_mut_ptr().add(d_arr_index_1) as *const i32);
                    let stored_12 =
                        vld1q_s32(buffer2.as_mut_ptr().add(d_arr_index_1) as *const i32);
                    let stored_13 =
                        vld1q_s32(buffer3.as_mut_ptr().add(d_arr_index_1) as *const i32);

                    let j0 = vsubq_s32(stored0, stored_10);
                    let j1 = vsubq_s32(stored1, stored_11);
                    let j2 = vsubq_s32(stored2, stored_12);
                    let j3 = vsubq_s32(stored3, stored_13);

                    let stored_20 =
                        vld1q_s32(buffer0.as_mut_ptr().add(d_arr_index_2) as *const i32);
                    let stored_21 =
                        vld1q_s32(buffer1.as_mut_ptr().add(d_arr_index_2) as *const i32);
                    let stored_22 =
                        vld1q_s32(buffer2.as_mut_ptr().add(d_arr_index_2) as *const i32);
                    let stored_23 =
                        vld1q_s32(buffer3.as_mut_ptr().add(d_arr_index_2) as *const i32);

                    let k0 = vmulq_by_3_s32(j0);
                    let k1 = vmulq_by_3_s32(j1);
                    let k2 = vmulq_by_3_s32(j2);
                    let k3 = vmulq_by_3_s32(j3);

                    let new_diff0 = vsubq_s32(k0, stored_20);
                    let new_diff1 = vsubq_s32(k1, stored_21);
                    let new_diff2 = vsubq_s32(k2, stored_22);
                    let new_diff3 = vsubq_s32(k3, stored_23);

                    diffs0 = vaddq_s32(diffs0, new_diff0);
                    diffs1 = vaddq_s32(diffs1, new_diff1);
                    diffs2 = vaddq_s32(diffs2, new_diff2);
                    diffs3 = vaddq_s32(diffs3, new_diff3);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;

                    let stored0 =
                        vld1q_s32(buffer0.get_unchecked_mut(arr_index..).as_mut_ptr() as *const _);
                    let stored1 =
                        vld1q_s32(buffer1.get_unchecked_mut(arr_index..).as_mut_ptr() as *const _);
                    let stored2 =
                        vld1q_s32(buffer2.get_unchecked_mut(arr_index..).as_mut_ptr() as *const _);
                    let stored3 =
                        vld1q_s32(buffer3.get_unchecked_mut(arr_index..).as_mut_ptr() as *const _);

                    let stored_10 = vld1q_s32(
                        buffer0.get_unchecked_mut(arr_index_1..).as_mut_ptr() as *const _,
                    );
                    let stored_11 = vld1q_s32(
                        buffer1.get_unchecked_mut(arr_index_1..).as_mut_ptr() as *const _,
                    );
                    let stored_12 = vld1q_s32(
                        buffer2.get_unchecked_mut(arr_index_1..).as_mut_ptr() as *const _,
                    );
                    let stored_13 = vld1q_s32(
                        buffer3.get_unchecked_mut(arr_index_1..).as_mut_ptr() as *const _,
                    );

                    diffs0 = vmlaq_s32(diffs0, vsubq_s32(stored0, stored_10), vdupq_n_s32(3));
                    diffs1 = vmlaq_s32(diffs1, vsubq_s32(stored1, stored_11), vdupq_n_s32(3));
                    diffs2 = vmlaq_s32(diffs2, vsubq_s32(stored2, stored_12), vdupq_n_s32(3));
                    diffs3 = vmlaq_s32(diffs3, vsubq_s32(stored3, stored_13), vdupq_n_s32(3));
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;

                    let stored0 =
                        vld1q_s32(buffer0.get_unchecked_mut(arr_index..).as_mut_ptr() as *const _);
                    let stored1 =
                        vld1q_s32(buffer1.get_unchecked_mut(arr_index..).as_mut_ptr() as *const _);
                    let stored2 =
                        vld1q_s32(buffer2.get_unchecked_mut(arr_index..).as_mut_ptr() as *const _);
                    let stored3 =
                        vld1q_s32(buffer3.get_unchecked_mut(arr_index..).as_mut_ptr() as *const _);

                    diffs0 = vmlaq_s32(diffs0, stored0, vdupq_n_s32(-3));
                    diffs1 = vmlaq_s32(diffs1, stored1, vdupq_n_s32(-3));
                    diffs2 = vmlaq_s32(diffs2, stored2, vdupq_n_s32(-3));
                    diffs3 = vmlaq_s32(diffs3, stored3, vdupq_n_s32(-3));
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * (stride as usize);

                let s_ptr0 = bytes.slice.as_ptr().add(next_row_y + current_px0) as *mut u8;
                let s_ptr1 = bytes.slice.as_ptr().add(next_row_y + current_px1) as *mut u8;
                let s_ptr2 = bytes.slice.as_ptr().add(next_row_y + current_px2) as *mut u8;
                let s_ptr3 = bytes.slice.as_ptr().add(next_row_y + current_px3) as *mut u8;

                let pixel_color0 = load_u8_s32_fast::<CN>(s_ptr0);
                let pixel_color1 = load_u8_s32_fast::<CN>(s_ptr1);
                let pixel_color2 = load_u8_s32_fast::<CN>(s_ptr2);
                let pixel_color3 = load_u8_s32_fast::<CN>(s_ptr3);

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;

                let buf_ptr0 = buffer0.get_unchecked_mut(arr_index..).as_mut_ptr() as *mut _;
                let buf_ptr1 = buffer1.get_unchecked_mut(arr_index..).as_mut_ptr() as *mut _;
                let buf_ptr2 = buffer2.get_unchecked_mut(arr_index..).as_mut_ptr() as *mut _;
                let buf_ptr3 = buffer3.get_unchecked_mut(arr_index..).as_mut_ptr() as *mut _;

                diffs0 = vaddq_s32(diffs0, pixel_color0);
                diffs1 = vaddq_s32(diffs1, pixel_color1);
                diffs2 = vaddq_s32(diffs2, pixel_color2);
                diffs3 = vaddq_s32(diffs3, pixel_color3);

                vst1q_s32(buf_ptr0, pixel_color0);
                vst1q_s32(buf_ptr1, pixel_color1);
                vst1q_s32(buf_ptr2, pixel_color2);
                vst1q_s32(buf_ptr3, pixel_color3);

                ders0 = vaddq_s32(ders0, diffs0);
                ders1 = vaddq_s32(ders1, diffs1);
                ders2 = vaddq_s32(ders2, diffs2);
                ders3 = vaddq_s32(ders3, diffs3);

                summs0 = vaddq_s32(summs0, ders0);
                summs1 = vaddq_s32(summs1, ders1);
                summs2 = vaddq_s32(summs2, ders2);
                summs3 = vaddq_s32(summs3, ders3);
            }
            xx += 4;
        }

        for x in xx..width.min(end) {
            let mut diffs = vdupq_n_s32(0);
            let mut ders = vdupq_n_s32(0);
            let mut summs = vdupq_n_s32(0);

            let current_px = (x * CN as u32) as usize;

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;

                if y >= 0 {
                    let prepared_px_s32 = vqrdmulhq_s32(summs, v_weight);
                    let prepared_u16 = vqmovun_s32(prepared_px_s32);
                    let prepared_u8 = vqmovn_u16(vcombine_u16(prepared_u16, prepared_u16));

                    let bytes_offset = current_y + current_px;

                    let dst_ptr = (bytes.slice.as_ptr() as *mut u8).add(bytes_offset);
                    store_u8x8_m4::<CN>(dst_ptr, prepared_u8);

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let buf_ptr = buffer0.as_mut_ptr().add(d_arr_index) as *const i32;
                    let stored = vld1q_s32(buf_ptr);

                    let buf_ptr_1 = buffer0.as_mut_ptr().add(d_arr_index_1) as *const i32;
                    let stored_1 = vld1q_s32(buf_ptr_1);

                    let buf_ptr_2 = buffer0.as_mut_ptr().add(d_arr_index_2) as *const i32;
                    let stored_2 = vld1q_s32(buf_ptr_2);

                    let new_diff = vsubq_s32(vmulq_by_3_s32(vsubq_s32(stored, stored_1)), stored_2);
                    diffs = vaddq_s32(diffs, new_diff);
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index..).as_mut_ptr() as *const _;
                    let stored = vld1q_s32(buf_ptr);

                    let buf_ptr_1 =
                        buffer0.get_unchecked_mut(arr_index_1..).as_mut_ptr() as *const _;
                    let stored_1 = vld1q_s32(buf_ptr_1);

                    diffs = vmlaq_s32(diffs, vsubq_s32(stored, stored_1), vdupq_n_s32(3));
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_ptr() as *const _;
                    let stored = vld1q_s32(buf_ptr);
                    diffs = vmlaq_s32(diffs, stored, vdupq_n_s32(-3));
                }

                let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
                    * (stride as usize);
                let next_row_x = (x * CN as u32) as usize;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut u8;

                let pixel_color = load_u8_s32_fast::<CN>(s_ptr);

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer0.get_unchecked_mut(arr_index..).as_mut_ptr() as *mut _;

                diffs = vaddq_s32(diffs, pixel_color);
                ders = vaddq_s32(ders, diffs);
                summs = vaddq_s32(summs, ders);
                vst1q_s32(buf_ptr, pixel_color);
            }
        }
    }
}

pub(crate) fn fgn_horizontal_pass_neon_u8_rdm<T, const CN: usize>(
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
        fgn_horizontal_pass_neon_impl::<T, CN>(
            undefined_slice,
            stride,
            width,
            height,
            radius,
            start,
            end,
            edge_mode,
        );
    }
}

#[target_feature(enable = "rdm")]
unsafe fn fgn_horizontal_pass_neon_impl<T, const CN: usize>(
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

        let mut full_buffer = Box::new([NeonI32x4::default(); 1024 * 4]);

        let (buffer0, rem) = full_buffer.split_at_mut(1024);
        let (buffer1, rem) = rem.split_at_mut(1024);
        let (buffer2, buffer3) = rem.split_at_mut(1024);

        let width_wide = width as i64;

        let radius_64 = radius as i64;
        const Q: f64 = ((1i64 << 31i64) - 1) as f64;
        let weight = Q / ((radius as f64) * (radius as f64) * (radius as f64));
        let v_weight = vdupq_n_s32(weight as i32);

        let mut yy = start;

        while yy + 4 < height.min(end) {
            let mut diffs0 = vdupq_n_s32(0);
            let mut diffs1 = vdupq_n_s32(0);
            let mut diffs2 = vdupq_n_s32(0);
            let mut diffs3 = vdupq_n_s32(0);

            let mut ders0 = vdupq_n_s32(0);
            let mut ders1 = vdupq_n_s32(0);
            let mut ders2 = vdupq_n_s32(0);
            let mut ders3 = vdupq_n_s32(0);

            let mut summs0 = vdupq_n_s32(0);
            let mut summs1 = vdupq_n_s32(0);
            let mut summs2 = vdupq_n_s32(0);
            let mut summs3 = vdupq_n_s32(0);

            let current_y0 = ((yy as i64) * (stride as i64)) as usize;
            let current_y1 = ((yy as i64 + 1) * (stride as i64)) as usize;
            let current_y2 = ((yy as i64 + 2) * (stride as i64)) as usize;
            let current_y3 = ((yy as i64 + 3) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let prepared_px0 = vqrdmulhq_s32(summs0, v_weight);
                    let prepared_px1 = vqrdmulhq_s32(summs1, v_weight);
                    let prepared_px2 = vqrdmulhq_s32(summs2, v_weight);
                    let prepared_px3 = vqrdmulhq_s32(summs3, v_weight);

                    let dst_ptr0 = (bytes.slice.as_ptr() as *mut u8).add(current_y0 + current_px);
                    let dst_ptr1 = (bytes.slice.as_ptr() as *mut u8).add(current_y1 + current_px);
                    let dst_ptr2 = (bytes.slice.as_ptr() as *mut u8).add(current_y2 + current_px);
                    let dst_ptr3 = (bytes.slice.as_ptr() as *mut u8).add(current_y3 + current_px);

                    store_u8_s32_x4::<CN>(
                        (dst_ptr0, dst_ptr1, dst_ptr2, dst_ptr3),
                        int32x4x4_t(prepared_px0, prepared_px1, prepared_px2, prepared_px3),
                    );

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let stored0 = vld1q_s32(buffer0.as_mut_ptr().add(d_arr_index) as *const i32);
                    let stored1 = vld1q_s32(buffer1.as_mut_ptr().add(d_arr_index) as *const i32);
                    let stored2 = vld1q_s32(buffer2.as_mut_ptr().add(d_arr_index) as *const i32);
                    let stored3 = vld1q_s32(buffer3.as_mut_ptr().add(d_arr_index) as *const i32);

                    let stored_10 =
                        vld1q_s32(buffer0.as_mut_ptr().add(d_arr_index_1) as *const i32);
                    let stored_11 =
                        vld1q_s32(buffer1.as_mut_ptr().add(d_arr_index_1) as *const i32);
                    let stored_12 =
                        vld1q_s32(buffer2.as_mut_ptr().add(d_arr_index_1) as *const i32);
                    let stored_13 =
                        vld1q_s32(buffer3.as_mut_ptr().add(d_arr_index_1) as *const i32);

                    let j0 = vsubq_s32(stored0, stored_10);
                    let j1 = vsubq_s32(stored1, stored_11);
                    let j2 = vsubq_s32(stored2, stored_12);
                    let j3 = vsubq_s32(stored3, stored_13);

                    let stored_20 =
                        vld1q_s32(buffer0.as_mut_ptr().add(d_arr_index_2) as *const i32);
                    let stored_21 =
                        vld1q_s32(buffer1.as_mut_ptr().add(d_arr_index_2) as *const i32);
                    let stored_22 =
                        vld1q_s32(buffer2.as_mut_ptr().add(d_arr_index_2) as *const i32);
                    let stored_23 =
                        vld1q_s32(buffer3.as_mut_ptr().add(d_arr_index_2) as *const i32);

                    let k0 = vmulq_by_3_s32(j0);
                    let k1 = vmulq_by_3_s32(j1);
                    let k2 = vmulq_by_3_s32(j2);
                    let k3 = vmulq_by_3_s32(j3);

                    let new_diff0 = vsubq_s32(k0, stored_20);
                    let new_diff1 = vsubq_s32(k1, stored_21);
                    let new_diff2 = vsubq_s32(k2, stored_22);
                    let new_diff3 = vsubq_s32(k3, stored_23);

                    diffs0 = vaddq_s32(diffs0, new_diff0);
                    diffs1 = vaddq_s32(diffs1, new_diff1);
                    diffs2 = vaddq_s32(diffs2, new_diff2);
                    diffs3 = vaddq_s32(diffs3, new_diff3);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;

                    let stored0 =
                        vld1q_s32(buffer0.get_unchecked_mut(arr_index..).as_ptr() as *const _);
                    let stored1 =
                        vld1q_s32(buffer1.get_unchecked_mut(arr_index..).as_ptr() as *const _);
                    let stored2 =
                        vld1q_s32(buffer2.get_unchecked_mut(arr_index..).as_ptr() as *const _);
                    let stored3 =
                        vld1q_s32(buffer3.get_unchecked_mut(arr_index..).as_ptr() as *const _);

                    let stored_10 =
                        vld1q_s32(buffer0.get_unchecked_mut(arr_index_1..).as_ptr() as *const _);
                    let stored_11 =
                        vld1q_s32(buffer1.get_unchecked_mut(arr_index_1..).as_ptr() as *const _);
                    let stored_12 =
                        vld1q_s32(buffer2.get_unchecked_mut(arr_index_1..).as_ptr() as *const _);
                    let stored_13 =
                        vld1q_s32(buffer3.get_unchecked_mut(arr_index_1..).as_ptr() as *const _);

                    diffs0 = vmlaq_s32(diffs0, vsubq_s32(stored0, stored_10), vdupq_n_s32(3));
                    diffs1 = vmlaq_s32(diffs1, vsubq_s32(stored1, stored_11), vdupq_n_s32(3));
                    diffs2 = vmlaq_s32(diffs2, vsubq_s32(stored2, stored_12), vdupq_n_s32(3));
                    diffs3 = vmlaq_s32(diffs3, vsubq_s32(stored3, stored_13), vdupq_n_s32(3));
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let stored0 =
                        vld1q_s32(buffer0.get_unchecked_mut(arr_index..).as_ptr() as *const _);
                    let stored1 =
                        vld1q_s32(buffer1.get_unchecked_mut(arr_index..).as_ptr() as *const _);
                    let stored2 =
                        vld1q_s32(buffer2.get_unchecked_mut(arr_index..).as_ptr() as *const _);
                    let stored3 =
                        vld1q_s32(buffer3.get_unchecked_mut(arr_index..).as_ptr() as *const _);

                    diffs0 = vmlaq_s32(diffs0, stored0, vdupq_n_s32(-3));
                    diffs1 = vmlaq_s32(diffs1, stored1, vdupq_n_s32(-3));
                    diffs2 = vmlaq_s32(diffs2, stored2, vdupq_n_s32(-3));
                    diffs3 = vmlaq_s32(diffs3, stored3, vdupq_n_s32(-3));
                }

                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr0 = bytes.slice.as_ptr().add(current_y0 + next_row_px) as *mut u8;
                let s_ptr1 = bytes.slice.as_ptr().add(current_y1 + next_row_px) as *mut u8;
                let s_ptr2 = bytes.slice.as_ptr().add(current_y2 + next_row_px) as *mut u8;
                let s_ptr3 = bytes.slice.as_ptr().add(current_y3 + next_row_px) as *mut u8;

                let pixel_color0 = load_u8_s32_fast::<CN>(s_ptr0);
                let pixel_color1 = load_u8_s32_fast::<CN>(s_ptr1);
                let pixel_color2 = load_u8_s32_fast::<CN>(s_ptr2);
                let pixel_color3 = load_u8_s32_fast::<CN>(s_ptr3);

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr0 = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _;
                let buf_ptr1 = buffer1.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _;
                let buf_ptr2 = buffer2.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _;
                let buf_ptr3 = buffer3.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _;

                diffs0 = vaddq_s32(diffs0, pixel_color0);
                diffs1 = vaddq_s32(diffs1, pixel_color1);
                diffs2 = vaddq_s32(diffs2, pixel_color2);
                diffs3 = vaddq_s32(diffs3, pixel_color3);

                vst1q_s32(buf_ptr0, pixel_color0);
                vst1q_s32(buf_ptr1, pixel_color1);
                vst1q_s32(buf_ptr2, pixel_color2);
                vst1q_s32(buf_ptr3, pixel_color3);

                ders0 = vaddq_s32(ders0, diffs0);
                ders1 = vaddq_s32(ders1, diffs1);
                ders2 = vaddq_s32(ders2, diffs2);
                ders3 = vaddq_s32(ders3, diffs3);

                summs0 = vaddq_s32(summs0, ders0);
                summs1 = vaddq_s32(summs1, ders1);
                summs2 = vaddq_s32(summs2, ders2);
                summs3 = vaddq_s32(summs3, ders3);
            }

            yy += 4;
        }

        for y in yy..height.min(end) {
            let mut diffs: int32x4_t = vdupq_n_s32(0);
            let mut ders: int32x4_t = vdupq_n_s32(0);
            let mut summs: int32x4_t = vdupq_n_s32(0);

            let current_y = ((y as i64) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CN;

                    let prepared_px_s32 = vqrdmulhq_s32(summs, v_weight);
                    let prepared_u16 = vqmovun_s32(prepared_px_s32);
                    let prepared_u8 = vqmovn_u16(vcombine_u16(prepared_u16, prepared_u16));

                    let bytes_offset = current_y + current_px;

                    let dst_ptr = (bytes.slice.as_ptr() as *mut u8).add(bytes_offset);
                    store_u8x8_m4::<CN>(dst_ptr, prepared_u8);

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let buf_ptr = buffer0.get_unchecked_mut(d_arr_index..).as_ptr() as *const _;
                    let stored = vld1q_s32(buf_ptr);

                    let buf_ptr_1 = buffer0.get_unchecked_mut(d_arr_index_1..).as_ptr() as *const _;
                    let stored_1 = vld1q_s32(buf_ptr_1);

                    let buf_ptr_2 = buffer0.get_unchecked_mut(d_arr_index_2..).as_ptr() as *const _;
                    let stored_2 = vld1q_s32(buf_ptr_2);

                    let new_diff = vsubq_s32(vmulq_by_3_s32(vsubq_s32(stored, stored_1)), stored_2);
                    diffs = vaddq_s32(diffs, new_diff);
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index..).as_ptr() as *const _;
                    let stored = vld1q_s32(buf_ptr);

                    let buf_ptr_1 = buffer0.get_unchecked_mut(arr_index_1..).as_ptr() as *const _;
                    let stored_1 = vld1q_s32(buf_ptr_1);

                    diffs = vmlaq_s32(diffs, vsubq_s32(stored, stored_1), vdupq_n_s32(3));
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer0.get_unchecked_mut(arr_index..).as_ptr() as *const _;
                    let stored = vld1q_s32(buf_ptr);
                    diffs = vmlaq_s32(diffs, stored, vdupq_n_s32(-3));
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide);
                let next_row_px = next_row_x * CN;

                let s_ptr = bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut u8;

                let pixel_color = load_u8_s32_fast::<CN>(s_ptr);

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer0.get_unchecked_mut(arr_index).0.as_mut_ptr() as *mut _;

                diffs = vaddq_s32(diffs, pixel_color);
                ders = vaddq_s32(ders, diffs);
                summs = vaddq_s32(summs, ders);
                vst1q_s32(buf_ptr, pixel_color);
            }
        }
    }
}
