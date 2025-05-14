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

use std::arch::aarch64::*;

use crate::neon::{
    load_u8, load_u8_u16, store_u8_s32_x4, store_u8_u32, vmulq_u16_low_f32, vmulq_u32_f32,
};
use crate::unsafe_slice::UnsafeSlice;

#[inline(always)]
unsafe fn mul_set_v4(
    s1: uint32x4_t,
    s2: uint32x4_t,
    s3: uint32x4_t,
    s4: uint32x4_t,
    w: float32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
    let cv1 = vcvtq_f32_u32(s1);
    let cv2 = vcvtq_f32_u32(s2);
    let cv3 = vcvtq_f32_u32(s3);
    let cv4 = vcvtq_f32_u32(s4);

    let m1 = vmulq_f32(cv1, w);
    let m2 = vmulq_f32(cv2, w);
    let m3 = vmulq_f32(cv3, w);
    let m4 = vmulq_f32(cv4, w);

    let cv1 = vcvtaq_u32_f32(m1);
    let cv2 = vcvtaq_u32_f32(m2);
    let cv3 = vcvtaq_u32_f32(m3);
    let cv4 = vcvtaq_u32_f32(m4);
    (cv1, cv2, cv3, cv4)
}

#[inline(always)]
unsafe fn mul_set_v2(s1: uint32x4_t, s2: uint32x4_t, w: float32x4_t) -> (uint32x4_t, uint32x4_t) {
    let cv1 = vcvtq_f32_u32(s1);
    let cv2 = vcvtq_f32_u32(s2);

    let m1 = vmulq_f32(cv1, w);
    let m2 = vmulq_f32(cv2, w);

    let cv1 = vcvtaq_u32_f32(m1);
    let cv2 = vcvtaq_u32_f32(m2);
    (cv1, cv2)
}

#[inline(always)]
unsafe fn mul_set(s1: uint32x4_t, w: float32x4_t) -> uint32x4_t {
    let cv1 = vcvtq_f32_u32(s1);
    let m1 = vmulq_f32(cv1, w);
    vcvtaq_u32_f32(m1)
}

#[inline(always)]
unsafe fn mul_set_low_u16_v4(
    s1: uint16x8_t,
    s2: uint16x8_t,
    s3: uint16x8_t,
    s4: uint16x8_t,
    w: float32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
    let s1 = vmovl_u16(vget_low_u16(s1));
    let s2 = vmovl_u16(vget_low_u16(s2));
    let s3 = vmovl_u16(vget_low_u16(s3));
    let s4 = vmovl_u16(vget_low_u16(s4));

    let cv1 = vcvtq_f32_u32(s1);
    let cv2 = vcvtq_f32_u32(s2);
    let cv3 = vcvtq_f32_u32(s3);
    let cv4 = vcvtq_f32_u32(s4);

    let m1 = vmulq_f32(cv1, w);
    let m2 = vmulq_f32(cv2, w);
    let m3 = vmulq_f32(cv3, w);
    let m4 = vmulq_f32(cv4, w);

    let cv1 = vcvtaq_u32_f32(m1);
    let cv2 = vcvtaq_u32_f32(m2);
    let cv3 = vcvtaq_u32_f32(m3);
    let cv4 = vcvtaq_u32_f32(m4);
    (cv1, cv2, cv3, cv4)
}

pub(crate) fn box_blur_horizontal_pass_neon<T, const CN: usize>(
    undefined_src: &[T],
    src_stride: u32,
    undefined_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undefined_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undefined_dst) };
    if radius < 240 {
        box_blur_horizontal_pass_neon_impl_low_rad::<CN>(
            src, src_stride, unsafe_dst, dst_stride, width, radius, start_y, end_y,
        );
    } else {
        box_blur_horizontal_pass_neon_impl::<CN>(
            src, src_stride, unsafe_dst, dst_stride, width, radius, start_y, end_y,
        );
    }
}

fn box_blur_horizontal_pass_neon_impl_low_rad<const CN: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = unsafe { vdup_n_u8(edge_count as u8) };

    let v_weight = unsafe { vdupq_n_f32(1f32 / (radius * 2) as f32) };

    let half_kernel = kernel_size / 2;

    let mut yy = start_y;

    while yy + 4 < end_y {
        let y = yy;
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store_0: uint16x8_t;
        let mut store_1: uint16x8_t;
        let mut store_2: uint16x8_t;
        let mut store_3: uint16x8_t;

        unsafe {
            let s_ptr_0 = src.as_ptr().add(y_src_shift);
            let edge_colors_0 = load_u8::<CN>(s_ptr_0);

            let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize);
            let edge_colors_1 = load_u8::<CN>(s_ptr_1);

            let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2);
            let edge_colors_2 = load_u8::<CN>(s_ptr_2);

            let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3);
            let edge_colors_3 = load_u8::<CN>(s_ptr_3);

            store_0 = vmull_u8(edge_colors_0, v_edge_count);
            store_1 = vmull_u8(edge_colors_1, v_edge_count);
            store_2 = vmull_u8(edge_colors_2, v_edge_count);
            store_3 = vmull_u8(edge_colors_3, v_edge_count);
        }

        unsafe {
            let mut xx = 1;

            if CN == 4 {
                let shuf1 = vcombine_u8(
                    vcreate_u8(u64::from_ne_bytes([0, 4, 8, 12, 1, 5, 9, 13])),
                    vcreate_u8(u64::from_ne_bytes([2, 6, 10, 14, 3, 7, 11, 15])),
                );

                while xx + 4 < half_kernel as usize && xx + 4 < width as usize {
                    let px = xx.min(width as usize - 1) * CN;
                    let s_ptr_0 = src.as_ptr().add(y_src_shift + px);
                    let mut edge_colors_0 = vld1q_u8(s_ptr_0);

                    let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + px);
                    let mut edge_colors_1 = vld1q_u8(s_ptr_1);

                    let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px);
                    let mut edge_colors_2 = vld1q_u8(s_ptr_2);

                    let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px);
                    let mut edge_colors_3 = vld1q_u8(s_ptr_3);

                    edge_colors_0 = vqtbl1q_u8(edge_colors_0, shuf1);
                    edge_colors_1 = vqtbl1q_u8(edge_colors_1, shuf1);
                    edge_colors_2 = vqtbl1q_u8(edge_colors_2, shuf1);
                    edge_colors_3 = vqtbl1q_u8(edge_colors_3, shuf1);

                    let v0 = vpaddlq_u8(edge_colors_0);
                    let v1 = vpaddlq_u8(edge_colors_1);
                    let v2 = vpaddlq_u8(edge_colors_2);
                    let v3 = vpaddlq_u8(edge_colors_3);

                    store_0 = vaddq_u16(store_0, vpaddq_u16(v0, v0));
                    store_1 = vaddq_u16(store_1, vpaddq_u16(v1, v1));
                    store_2 = vaddq_u16(store_2, vpaddq_u16(v2, v2));
                    store_3 = vaddq_u16(store_3, vpaddq_u16(v3, v3));

                    xx += 4;
                }
            }

            for x in xx..half_kernel as usize {
                let px = x.min(width as usize - 1) * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + px);
                let edge_colors_0 = load_u8::<CN>(s_ptr_0);

                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + px);
                let edge_colors_1 = load_u8::<CN>(s_ptr_1);

                let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px);
                let edge_colors_2 = load_u8::<CN>(s_ptr_2);

                let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px);
                let edge_colors_3 = load_u8::<CN>(s_ptr_3);

                store_0 = vaddw_u8(store_0, edge_colors_0);
                store_1 = vaddw_u8(store_1, edge_colors_1);
                store_2 = vaddw_u8(store_2, edge_colors_2);
                store_3 = vaddw_u8(store_3, edge_colors_3);
            }
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            unsafe {
                let previous_x = (x as i64 - half_kernel as i64).max(0) as usize;
                let previous = previous_x * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + previous);
                let edge_colors_0 = load_u8::<CN>(s_ptr_0);

                let s_ptr_1 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize + previous);
                let edge_colors_1 = load_u8::<CN>(s_ptr_1);

                let s_ptr_2 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 2 + previous);
                let edge_colors_2 = load_u8::<CN>(s_ptr_2);

                let s_ptr_3 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 3 + previous);
                let edge_colors_3 = load_u8::<CN>(s_ptr_3);

                store_0 = vsubw_u8(store_0, edge_colors_0);
                store_1 = vsubw_u8(store_1, edge_colors_1);
                store_2 = vsubw_u8(store_2, edge_colors_2);
                store_3 = vsubw_u8(store_3, edge_colors_3);
            }

            // add next
            unsafe {
                let next_x = (x + half_kernel).min(width - 1) as usize;

                let next = next_x * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + next);
                let edge_colors_0 = load_u8::<CN>(s_ptr_0);

                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + next);
                let edge_colors_1 = load_u8::<CN>(s_ptr_1);

                let s_ptr_2 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 2 + next);
                let edge_colors_2 = load_u8::<CN>(s_ptr_2);

                let s_ptr_3 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 3 + next);
                let edge_colors_3 = load_u8::<CN>(s_ptr_3);

                store_0 = vaddw_u8(store_0, edge_colors_0);
                store_1 = vaddw_u8(store_1, edge_colors_1);
                store_2 = vaddw_u8(store_2, edge_colors_2);
                store_3 = vaddw_u8(store_3, edge_colors_3);
            }

            let px = x as usize * CN;

            unsafe {
                let (scale_store0, scale_store1, scale_store2, scale_store3) =
                    mul_set_low_u16_v4(store_0, store_1, store_2, store_3, v_weight);

                let px_160 = vqmovn_u32(scale_store0);
                let px_161 = vqmovn_u32(scale_store1);
                let px_162 = vqmovn_u32(scale_store2);
                let px_163 = vqmovn_u32(scale_store3);

                let px_80 = vqmovn_u16(vcombine_u16(px_160, px_160));
                let px_81 = vqmovn_u16(vcombine_u16(px_161, px_161));
                let px_82 = vqmovn_u16(vcombine_u16(px_162, px_162));
                let px_83 = vqmovn_u16(vcombine_u16(px_163, px_163));

                let bytes_offset_0 = y_dst_shift + px;
                let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
                let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
                let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;
                if CN == 4 {
                    let dst_ptr_0 = unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut u32;
                    vst1_lane_u32::<0>(dst_ptr_0, vreinterpret_u32_u8(px_80));

                    let dst_ptr_1 = unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut u32;
                    vst1_lane_u32::<0>(dst_ptr_1, vreinterpret_u32_u8(px_81));

                    let dst_ptr_2 = unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut u32;
                    vst1_lane_u32::<0>(dst_ptr_2, vreinterpret_u32_u8(px_82));

                    let dst_ptr_3 = unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut u32;
                    vst1_lane_u32::<0>(dst_ptr_3, vreinterpret_u32_u8(px_83));
                } else {
                    let dst_ptr_0 = unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut u8;
                    vst1_lane_u16::<0>(dst_ptr_0 as *mut u16, vreinterpret_u16_u8(px_80));
                    vst1_lane_u8::<2>(dst_ptr_0.add(2), px_80);

                    let dst_ptr_1 = unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut u8;
                    vst1_lane_u16::<0>(dst_ptr_1 as *mut u16, vreinterpret_u16_u8(px_81));
                    vst1_lane_u8::<2>(dst_ptr_1.add(2), px_81);

                    let dst_ptr_2 = unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut u8;
                    vst1_lane_u16::<0>(dst_ptr_2 as *mut u16, vreinterpret_u16_u8(px_82));
                    vst1_lane_u8::<2>(dst_ptr_2.add(2), px_82);

                    let dst_ptr_3 = unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut u8;
                    vst1_lane_u16::<0>(dst_ptr_3 as *mut u16, vreinterpret_u16_u8(px_83));
                    vst1_lane_u8::<2>(dst_ptr_3.add(2), px_83);
                }
            }
        }

        yy += 4;
    }

    for y in yy..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store: uint16x8_t;
        unsafe {
            let s_ptr = src.as_ptr().add(y_src_shift);
            let edge_colors = load_u8::<CN>(s_ptr);
            store = vmull_u8(edge_colors, v_edge_count);
        }

        unsafe {
            for x in 1..half_kernel as usize {
                let px = x.min(width as usize - 1) * CN;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let edge_colors = load_u8::<CN>(s_ptr);
                store = vaddw_u8(store, edge_colors);
            }
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            unsafe {
                let previous_x = (x as isize - half_kernel as isize).max(0) as usize;
                let previous = previous_x * CN;
                let s_ptr = src.as_ptr().add(y_src_shift + previous);
                let edge_colors = load_u8::<CN>(s_ptr);
                store = vsubw_u8(store, edge_colors);
            }

            // add next
            unsafe {
                let next_x = (x + half_kernel).min(width - 1) as usize;

                let next = next_x * CN;

                let s_ptr = src.as_ptr().add(y_src_shift + next);
                let edge_colors = load_u8::<CN>(s_ptr);
                store = vaddw_u8(store, edge_colors);
            }

            let px = x as usize * CN;

            let scale_store = unsafe { vmulq_u16_low_f32(store, v_weight) };

            let bytes_offset = y_dst_shift + px;

            unsafe {
                let dst_ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u8;
                store_u8_u32::<CN>(dst_ptr, scale_store);
            }
        }
    }
}

fn box_blur_horizontal_pass_neon_impl<const CN: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = unsafe { vdupq_n_u16(edge_count as u16) };

    let v_weight = unsafe { vdupq_n_f32(1f32 / (radius * 2) as f32) };

    let half_kernel = kernel_size / 2;

    let mut yy = start_y;

    while yy + 4 < end_y {
        let y = yy;
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store_0: uint32x4_t;
        let mut store_1: uint32x4_t;
        let mut store_2: uint32x4_t;
        let mut store_3: uint32x4_t;

        unsafe {
            let s_ptr_0 = src.as_ptr().add(y_src_shift);
            let edge_colors_0 = load_u8_u16::<CN>(s_ptr_0);

            let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize);
            let edge_colors_1 = load_u8_u16::<CN>(s_ptr_1);

            let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2);
            let edge_colors_2 = load_u8_u16::<CN>(s_ptr_2);

            let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3);
            let edge_colors_3 = load_u8_u16::<CN>(s_ptr_3);

            store_0 = vmull_u16(edge_colors_0, vget_low_u16(v_edge_count));
            store_1 = vmull_u16(edge_colors_1, vget_low_u16(v_edge_count));
            store_2 = vmull_u16(edge_colors_2, vget_low_u16(v_edge_count));
            store_3 = vmull_u16(edge_colors_3, vget_low_u16(v_edge_count));
        }

        unsafe {
            for x in 1..half_kernel as usize {
                let px = x.min(width as usize - 1) * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + px);
                let edge_colors_0 = load_u8_u16::<CN>(s_ptr_0);

                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + px);
                let edge_colors_1 = load_u8_u16::<CN>(s_ptr_1);

                let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px);
                let edge_colors_2 = load_u8_u16::<CN>(s_ptr_2);

                let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px);
                let edge_colors_3 = load_u8_u16::<CN>(s_ptr_3);

                store_0 = vaddw_u16(store_0, edge_colors_0);
                store_1 = vaddw_u16(store_1, edge_colors_1);
                store_2 = vaddw_u16(store_2, edge_colors_2);
                store_3 = vaddw_u16(store_3, edge_colors_3);
            }
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            unsafe {
                let previous_x = (x as i64 - half_kernel as i64).max(0) as usize;
                let previous = previous_x * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + previous);
                let edge_colors_0 = load_u8_u16::<CN>(s_ptr_0);

                let s_ptr_1 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize + previous);
                let edge_colors_1 = load_u8_u16::<CN>(s_ptr_1);

                let s_ptr_2 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 2 + previous);
                let edge_colors_2 = load_u8_u16::<CN>(s_ptr_2);

                let s_ptr_3 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 3 + previous);
                let edge_colors_3 = load_u8_u16::<CN>(s_ptr_3);

                store_0 = vsubw_u16(store_0, edge_colors_0);
                store_1 = vsubw_u16(store_1, edge_colors_1);
                store_2 = vsubw_u16(store_2, edge_colors_2);
                store_3 = vsubw_u16(store_3, edge_colors_3);
            }

            // add next
            unsafe {
                let next_x = (x + half_kernel).min(width - 1) as usize;

                let next = next_x * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + next);
                let edge_colors_0 = load_u8_u16::<CN>(s_ptr_0);

                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + next);
                let edge_colors_1 = load_u8_u16::<CN>(s_ptr_1);

                let s_ptr_2 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 2 + next);
                let edge_colors_2 = load_u8_u16::<CN>(s_ptr_2);

                let s_ptr_3 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 3 + next);
                let edge_colors_3 = load_u8_u16::<CN>(s_ptr_3);

                store_0 = vaddw_u16(store_0, edge_colors_0);
                store_1 = vaddw_u16(store_1, edge_colors_1);
                store_2 = vaddw_u16(store_2, edge_colors_2);
                store_3 = vaddw_u16(store_3, edge_colors_3);
            }

            let px = x as usize * CN;

            unsafe {
                let (ss0, ss1, ss2, ss3) = mul_set_v4(store_0, store_1, store_2, store_3, v_weight);

                let bytes_offset_0 = y_dst_shift + px;
                let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
                let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
                let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;

                store_u8_s32_x4::<CN>(
                    (
                        unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut _,
                        unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut _,
                        unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut _,
                        unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut _,
                    ),
                    int32x4x4_t(
                        vreinterpretq_s32_u32(ss0),
                        vreinterpretq_s32_u32(ss1),
                        vreinterpretq_s32_u32(ss2),
                        vreinterpretq_s32_u32(ss3),
                    ),
                );
            }
        }

        yy += 4;
    }

    for y in yy..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store: uint32x4_t;
        unsafe {
            let s_ptr = src.as_ptr().add(y_src_shift);
            let edge_colors = load_u8_u16::<CN>(s_ptr);
            store = vmull_u16(edge_colors, vget_low_u16(v_edge_count));
        }

        unsafe {
            for x in 1..half_kernel as usize {
                let px = x.min(width as usize - 1) * CN;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let edge_colors = load_u8_u16::<CN>(s_ptr);
                store = vaddw_u16(store, edge_colors);
            }
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            unsafe {
                let previous_x = (x as isize - half_kernel as isize).max(0) as usize;
                let previous = previous_x * CN;
                let s_ptr = src.as_ptr().add(y_src_shift + previous);
                let edge_colors = load_u8_u16::<CN>(s_ptr);
                store = vsubw_u16(store, edge_colors);
            }

            // add next
            unsafe {
                let next_x = (x + half_kernel).min(width - 1) as usize;

                let next = next_x * CN;

                let s_ptr = src.as_ptr().add(y_src_shift + next);
                let edge_colors = load_u8_u16::<CN>(s_ptr);
                store = vaddw_u16(store, edge_colors);
            }

            let px = x as usize * CN;

            let scale_store = unsafe { vmulq_u32_f32(store, v_weight) };

            let bytes_offset = y_dst_shift + px;
            unsafe {
                let dst_ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u8;
                store_u8_u32::<CN>(dst_ptr, scale_store);
            }
        }
    }
}

pub(crate) fn box_blur_vertical_pass_neon<T>(
    undefined_src: &[T],
    src_stride: u32,
    undefined_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    w: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    unsafe {
        let src: &[u8] = std::mem::transmute(undefined_src);
        let unsafe_dst: &UnsafeSlice<'_, u8> = std::mem::transmute(undefined_unsafe_dst);
        if radius < 240 {
            box_blur_vertical_pass_neon_low_rad(
                src, src_stride, unsafe_dst, dst_stride, w, height, radius, start_x, end_x,
            );
        } else {
            box_blur_vertical_pass_neon_any(
                src, src_stride, unsafe_dst, dst_stride, w, height, radius, start_x, end_x,
            );
        }
    }
}

unsafe fn box_blur_vertical_pass_neon_low_rad(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    _: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = vdupq_n_u8(edge_count as u8);

    let half_kernel = kernel_size / 2;

    let start_x = start_x as usize;
    let end_x = end_x as usize;

    let v_weight = vdupq_n_f32(1f32 / (radius * 2) as f32);

    assert!(end_x >= start_x);

    let buf_size = end_x - start_x;

    let buf_cap = buf_size.div_ceil(8) * 8 + 8;

    let mut buffer = vec![0u16; buf_cap];

    let mut cx = start_x;

    // Configure initial accumulator

    let mut buf_cx = 0usize;

    while cx + 32 < end_x {
        let px = cx;

        let mut store_0: uint16x8_t;
        let mut store_1: uint16x8_t;
        let mut store_2: uint16x8_t;
        let mut store_3: uint16x8_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1q_u8(s_ptr);
        let edge1 = vld1q_u8(s_ptr.add(16));

        store_0 = vmull_u8(vget_low_u8(edge0), vget_low_u8(v_edge_count));
        store_1 = vmull_high_u8(edge0, v_edge_count);

        store_2 = vmull_u8(vget_low_u8(edge1), vget_low_u8(v_edge_count));
        store_3 = vmull_high_u8(edge1, v_edge_count);

        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1q_u8(s_ptr);
            let edge1 = vld1q_u8(s_ptr.add(16));

            store_0 = vaddw_u8(store_0, vget_low_u8(edge0));
            store_1 = vaddw_high_u8(store_1, edge0);
            store_2 = vaddw_u8(store_2, vget_low_u8(edge1));
            store_3 = vaddw_high_u8(store_3, edge1);
        }

        vst1q_u16(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
        vst1q_u16(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_1);
        vst1q_u16(
            buffer.get_unchecked_mut(buf_cx + 16..).as_mut_ptr(),
            store_2,
        );
        vst1q_u16(
            buffer.get_unchecked_mut(buf_cx + 24..).as_mut_ptr(),
            store_3,
        );

        cx += 32;
        buf_cx += 32;
    }

    while cx + 16 < end_x {
        let px = cx;

        let mut store_0: uint16x8_t;
        let mut store_1: uint16x8_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1q_u8(s_ptr);

        store_0 = vmull_u8(vget_low_u8(edge0), vget_low_u8(v_edge_count));
        store_1 = vmull_high_u8(edge0, v_edge_count);

        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1q_u8(s_ptr);

            store_0 = vaddw_u8(store_0, vget_low_u8(edge0));
            store_1 = vaddw_high_u8(store_1, edge0);
        }

        vst1q_u16(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
        vst1q_u16(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_1);

        cx += 16;
        buf_cx += 16;
    }

    while cx + 8 < end_x {
        let px = cx;

        let mut store_0: uint16x8_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1_u8(s_ptr);

        store_0 = vmull_u8(edge0, vget_low_u8(v_edge_count));

        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1_u8(s_ptr);

            store_0 = vaddw_u8(store_0, edge0);
        }

        vst1q_u16(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);

        cx += 8;
        buf_cx += 8;
    }

    while cx < end_x {
        let px = cx;

        let mut store_0: uint16x8_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1q_lane_u8::<0>(s_ptr, vdupq_n_u8(0));

        store_0 = vmull_u8(vget_low_u8(edge0), vget_low_u8(v_edge_count));

        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1q_lane_u8::<0>(s_ptr, vdupq_n_u8(0));

            store_0 = vaddw_u8(store_0, vget_low_u8(edge0));
        }

        vst1q_lane_u16::<0>(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);

        cx += 1;
        buf_cx += 1;
    }

    for y in 0..height as usize {
        let mut cx = start_x;

        let mut buf_cx = 0usize;

        while cx + 32 < end_x {
            let px = cx;

            let mut store_0 = vld1q_u16(buffer.get_unchecked(buf_cx..).as_ptr());
            let mut store_1 = vld1q_u16(buffer.get_unchecked(buf_cx + 8..).as_ptr());
            let mut store_2 = vld1q_u16(buffer.get_unchecked(buf_cx + 16..).as_ptr());
            let mut store_3 = vld1q_u16(buffer.get_unchecked(buf_cx + 24..).as_ptr());

            // preload edge pixels
            let next = (y + half_kernel as usize).min(height as usize - 1) * src_stride as usize;
            let previous =
                (y as isize - half_kernel as isize).max(0) as usize * src_stride as usize;
            let y_dst_shift = dst_stride as usize * y;

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge0 = vld1q_u8(s_ptr);
                let edge1 = vld1q_u8(s_ptr.add(16));

                store_0 = vsubw_u8(store_0, vget_low_u8(edge0));
                store_1 = vsubw_high_u8(store_1, edge0);

                store_2 = vsubw_u8(store_2, vget_low_u8(edge1));
                store_3 = vsubw_high_u8(store_3, edge1);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge0 = vld1q_u8(s_ptr);
                let edge1 = vld1q_u8(s_ptr.add(16));

                store_0 = vaddw_u8(store_0, vget_low_u8(edge0));
                store_1 = vaddw_high_u8(store_1, edge0);

                store_2 = vaddw_u8(store_2, vget_low_u8(edge1));
                store_3 = vaddw_high_u8(store_3, edge1);
            }

            let px = cx;

            let lw0 = vmovl_u16(vget_low_u16(store_0));
            let lw1 = vmovl_high_u16(store_0);
            let lw2 = vmovl_u16(vget_low_u16(store_1));
            let lw3 = vmovl_high_u16(store_1);

            let lw4 = vmovl_u16(vget_low_u16(store_2));
            let lw5 = vmovl_high_u16(store_2);
            let lw6 = vmovl_u16(vget_low_u16(store_3));
            let lw7 = vmovl_high_u16(store_3);

            let (scale_store_0, scale_store_1, scale_store_2, scale_store_3) =
                mul_set_v4(lw0, lw1, lw2, lw3, v_weight);

            vst1q_u16(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
            vst1q_u16(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_1);
            vst1q_u16(
                buffer.get_unchecked_mut(buf_cx + 16..).as_mut_ptr(),
                store_2,
            );
            vst1q_u16(
                buffer.get_unchecked_mut(buf_cx + 24..).as_mut_ptr(),
                store_3,
            );

            let (scale_store_4, scale_store_5, scale_store_6, scale_store_7) =
                mul_set_v4(lw4, lw5, lw6, lw7, v_weight);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vqmovn_u32(scale_store_0);
            let px_16_hi0 = vqmovn_u32(scale_store_1);
            let px_16_lo1 = vqmovn_u32(scale_store_2);
            let px_16_hi2 = vqmovn_u32(scale_store_3);
            let px_16_lo3 = vqmovn_u32(scale_store_4);
            let px_16_hi4 = vqmovn_u32(scale_store_5);
            let px_16_lo5 = vqmovn_u32(scale_store_6);
            let px_16_hi6 = vqmovn_u32(scale_store_7);

            let px0 = vqmovn_u16(vcombine_u16(px_16_lo0, px_16_hi0));
            let px1 = vqmovn_u16(vcombine_u16(px_16_lo1, px_16_hi2));
            let px2 = vqmovn_u16(vcombine_u16(px_16_lo3, px_16_hi4));
            let px3 = vqmovn_u16(vcombine_u16(px_16_lo5, px_16_hi6));
            vst1q_u8(ptr, vcombine_u8(px0, px1));
            vst1q_u8(ptr.add(16), vcombine_u8(px2, px3));

            cx += 32;
            buf_cx += 32;
        }

        while cx + 16 < end_x {
            let px = cx;

            let mut store_0 = vld1q_u16(buffer.get_unchecked(buf_cx..).as_ptr());
            let mut store_1 = vld1q_u16(buffer.get_unchecked(buf_cx + 8..).as_ptr());

            // preload edge pixels
            let next = (y + half_kernel as usize).min(height as usize - 1) * src_stride as usize;
            let previous =
                (y as isize - half_kernel as isize).max(0) as usize * src_stride as usize;
            let y_dst_shift = dst_stride as usize * y;

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge = vld1q_u8(s_ptr);

                store_0 = vsubw_u8(store_0, vget_low_u8(edge));
                store_1 = vsubw_high_u8(store_1, edge);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge = vld1q_u8(s_ptr);

                store_0 = vaddw_u8(store_0, vget_low_u8(edge));
                store_1 = vaddw_high_u8(store_1, edge);
            }

            let px = cx;

            let lw0 = vmovl_u16(vget_low_u16(store_0));
            let lw1 = vmovl_high_u16(store_0);
            let lw2 = vmovl_u16(vget_low_u16(store_1));
            let lw3 = vmovl_high_u16(store_1);

            let (scale_store_0, scale_store_1, scale_store_2, scale_store_3) =
                mul_set_v4(lw0, lw1, lw2, lw3, v_weight);

            vst1q_u16(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
            vst1q_u16(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_1);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vqmovn_u32(scale_store_0);
            let px_16_hi0 = vqmovn_u32(scale_store_1);
            let px_16_lo1 = vqmovn_u32(scale_store_2);
            let px_16_hi2 = vqmovn_u32(scale_store_3);

            let px0 = vqmovn_u16(vcombine_u16(px_16_lo0, px_16_hi0));
            let px1 = vqmovn_u16(vcombine_u16(px_16_lo1, px_16_hi2));
            vst1q_u8(ptr, vcombine_u8(px0, px1));

            cx += 16;
            buf_cx += 16;
        }

        while cx + 8 < end_x {
            let px = cx;

            let mut store_0 = vld1q_u16(buffer.get_unchecked(buf_cx..).as_ptr());

            // preload edge pixels
            let next = (y + half_kernel as usize).min(height as usize - 1) * src_stride as usize;
            let previous =
                (y as isize - half_kernel as isize).max(0) as usize * src_stride as usize;
            let y_dst_shift = dst_stride as usize * y;

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge = vld1_u8(s_ptr);

                store_0 = vsubw_u8(store_0, edge);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge = vld1_u8(s_ptr);

                store_0 = vaddw_u8(store_0, edge);
            }

            let px = cx;

            let lw0 = vmovl_u16(vget_low_u16(store_0));
            let lw1 = vmovl_high_u16(store_0);

            let scale_store_0 = vmulq_u32_f32(lw0, v_weight);
            let scale_store_1 = vmulq_u32_f32(lw1, v_weight);

            vst1q_u16(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vqmovn_u32(scale_store_0);
            let px_16_hi0 = vqmovn_u32(scale_store_1);
            let px0 = vqmovn_u16(vcombine_u16(px_16_lo0, px_16_hi0));
            vst1_u8(ptr, px0);

            cx += 8;
            buf_cx += 8;
        }

        const TAIL_CN: usize = 1;

        for (x, buf_cx) in (cx..end_x).zip(buf_cx..buf_cap) {
            let px = x;

            let mut store =
                vld1q_lane_u16::<0>(buffer.get_unchecked(buf_cx..).as_ptr(), vdupq_n_u16(0));

            // preload edge pixels
            let next = (y + half_kernel as usize).min(height as usize - 1) * src_stride as usize;
            let previous = (y as i64 - half_kernel as i64).max(0) as usize * src_stride as usize;
            let y_dst_shift = dst_stride as usize * y;

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge_colors = load_u8::<TAIL_CN>(s_ptr);
                store = vsubw_u8(store, edge_colors);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge_colors = load_u8::<TAIL_CN>(s_ptr);
                store = vaddw_u8(store, edge_colors);
            }

            let px = x;

            let scale_store = vmulq_u16_low_f32(store, v_weight);

            vst1q_lane_u16::<0>(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store);

            let bytes_offset = y_dst_shift + px;
            let dst_ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u8;
            store_u8_u32::<TAIL_CN>(dst_ptr, scale_store);
        }
    }
}

unsafe fn box_blur_vertical_pass_neon_any(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    _: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = vdupq_n_u16(edge_count as u16);

    let half_kernel = kernel_size / 2;

    let start_x = start_x as usize;
    let end_x = end_x as usize;

    let v_weight = vdupq_n_f32(1f32 / (radius * 2) as f32);

    assert!(end_x >= start_x);

    let buf_size = end_x - start_x;

    let buf_cap = buf_size.div_ceil(4) * 4 + 4;

    let mut buffer = vec![0u32; buf_cap];

    let mut cx = start_x;
    let mut buf_cx = 0usize;

    while cx + 32 < end_x {
        let px = cx;

        let mut store_0: uint32x4_t;
        let mut store_1: uint32x4_t;
        let mut store_2: uint32x4_t;
        let mut store_3: uint32x4_t;
        let mut store_4: uint32x4_t;
        let mut store_5: uint32x4_t;
        let mut store_6: uint32x4_t;
        let mut store_7: uint32x4_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1q_u8(s_ptr);
        let edge1 = vld1q_u8(s_ptr.add(16));

        let lo0 = vmovl_u8(vget_low_u8(edge0));
        let hi0 = vmovl_high_u8(edge0);
        let lo1 = vmovl_u8(vget_low_u8(edge1));
        let hi1 = vmovl_high_u8(edge1);

        let i16_l0 = vget_low_u16(lo0);
        let i16_h0 = lo0;
        let i16_l1 = vget_low_u16(hi0);
        let i16_h1 = hi0;

        let i16_l01 = vget_low_u16(lo1);
        let i16_h01 = lo1;
        let i16_l11 = vget_low_u16(hi1);
        let i16_h11 = hi1;

        store_0 = vmull_u16(i16_l0, vget_low_u16(v_edge_count));
        store_1 = vmull_high_u16(i16_h0, v_edge_count);

        store_2 = vmull_u16(i16_l1, vget_low_u16(v_edge_count));
        store_3 = vmull_high_u16(i16_h1, v_edge_count);

        store_4 = vmull_u16(i16_l01, vget_low_u16(v_edge_count));
        store_5 = vmull_high_u16(i16_h01, v_edge_count);

        store_6 = vmull_u16(i16_l11, vget_low_u16(v_edge_count));
        store_7 = vmull_high_u16(i16_h11, v_edge_count);

        for y in 1..half_kernel.min(height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1q_u8(s_ptr);
            let edge1 = vld1q_u8(s_ptr.add(16));

            let lo0 = vmovl_u8(vget_low_u8(edge0));
            let hi0 = vmovl_high_u8(edge0);
            let lo1 = vmovl_u8(vget_low_u8(edge1));
            let hi1 = vmovl_high_u8(edge1);

            let i16_l0 = vget_low_u16(lo0);
            let i16_h0 = lo0;
            let i16_l1 = vget_low_u16(hi0);
            let i16_h1 = hi0;

            let i16_l01 = vget_low_u16(lo1);
            let i16_h01 = lo1;
            let i16_l11 = vget_low_u16(hi1);
            let i16_h11 = hi1;

            store_0 = vaddw_u16(store_0, i16_l0);
            store_1 = vaddw_high_u16(store_1, i16_h0);
            store_2 = vaddw_u16(store_2, i16_l1);
            store_3 = vaddw_high_u16(store_3, i16_h1);

            store_4 = vaddw_u16(store_4, i16_l01);
            store_5 = vaddw_high_u16(store_5, i16_h01);
            store_6 = vaddw_u16(store_6, i16_l11);
            store_7 = vaddw_high_u16(store_7, i16_h11);
        }

        vst1q_u32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
        vst1q_u32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);
        vst1q_u32(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_2);
        vst1q_u32(
            buffer.get_unchecked_mut(buf_cx + 12..).as_mut_ptr(),
            store_3,
        );
        vst1q_u32(
            buffer.get_unchecked_mut(buf_cx + 16..).as_mut_ptr(),
            store_4,
        );
        vst1q_u32(
            buffer.get_unchecked_mut(buf_cx + 20..).as_mut_ptr(),
            store_5,
        );
        vst1q_u32(
            buffer.get_unchecked_mut(buf_cx + 24..).as_mut_ptr(),
            store_6,
        );
        vst1q_u32(
            buffer.get_unchecked_mut(buf_cx + 28..).as_mut_ptr(),
            store_7,
        );

        cx += 32;
        buf_cx += 32;
    }

    while cx + 16 < end_x {
        let px = cx;

        let mut store_0: uint32x4_t;
        let mut store_1: uint32x4_t;
        let mut store_2: uint32x4_t;
        let mut store_3: uint32x4_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1q_u8(s_ptr);

        let lo0 = vmovl_u8(vget_low_u8(edge0));
        let hi0 = vmovl_high_u8(edge0);

        let i16_l0 = vget_low_u16(lo0);
        let i16_h0 = lo0;
        let i16_l1 = vget_low_u16(hi0);
        let i16_h1 = hi0;

        store_0 = vmull_u16(i16_l0, vget_low_u16(v_edge_count));
        store_1 = vmull_high_u16(i16_h0, v_edge_count);

        store_2 = vmull_u16(i16_l1, vget_low_u16(v_edge_count));
        store_3 = vmull_high_u16(i16_h1, v_edge_count);

        for y in 1..half_kernel.min(height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1q_u8(s_ptr);

            let lo0 = vmovl_u8(vget_low_u8(edge0));
            let hi0 = vmovl_high_u8(edge0);

            let i16_l0 = vget_low_u16(lo0);
            let i16_h0 = lo0;
            let i16_l1 = vget_low_u16(hi0);
            let i16_h1 = hi0;

            store_0 = vaddw_u16(store_0, i16_l0);
            store_1 = vaddw_high_u16(store_1, i16_h0);
            store_2 = vaddw_u16(store_2, i16_l1);
            store_3 = vaddw_high_u16(store_3, i16_h1);
        }

        vst1q_u32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
        vst1q_u32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);
        vst1q_u32(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_2);
        vst1q_u32(
            buffer.get_unchecked_mut(buf_cx + 12..).as_mut_ptr(),
            store_3,
        );

        cx += 16;
        buf_cx += 16;
    }

    while cx + 8 < end_x {
        let px = cx;

        let mut store_0: uint32x4_t;
        let mut store_1: uint32x4_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1_u8(s_ptr);

        let lo0 = vmovl_u8(edge0);

        let i16_l0 = vget_low_u16(lo0);
        let i16_h0 = lo0;

        store_0 = vmull_u16(i16_l0, vget_low_u16(v_edge_count));
        store_1 = vmull_high_u16(i16_h0, v_edge_count);

        for y in 1..half_kernel.min(height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1_u8(s_ptr);

            let lo0 = vmovl_u8(edge0);

            let i16_l0 = vget_low_u16(lo0);
            let i16_h0 = lo0;

            store_0 = vaddw_u16(store_0, i16_l0);
            store_1 = vaddw_high_u16(store_1, i16_h0);
        }

        vst1q_u32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
        vst1q_u32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);

        cx += 8;
        buf_cx += 8;
    }

    while cx < end_x {
        let px = cx;

        let mut store_0: uint32x4_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1_lane_u8::<0>(s_ptr, vdup_n_u8(0));

        let lo0 = vmovl_u8(edge0);

        let i16_l0 = vget_low_u16(lo0);

        store_0 = vmull_u16(i16_l0, vget_low_u16(v_edge_count));

        for y in 1..half_kernel.min(height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1_lane_u8::<0>(s_ptr, vdup_n_u8(0));
            let lo0 = vmovl_u8(edge0);
            let i16_l0 = vget_low_u16(lo0);
            store_0 = vaddw_u16(store_0, i16_l0);
        }

        vst1q_lane_u32::<0>(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);

        cx += 1;
        buf_cx += 1;
    }

    for y in 0..height {
        let mut buf_cx = 0usize;
        let mut cx = start_x;

        // preload edge pixels
        let next = (y + half_kernel).min(height - 1) as usize * src_stride as usize;
        let previous = (y as isize - half_kernel as isize).max(0) as usize * src_stride as usize;
        let y_dst_shift = dst_stride as usize * y as usize;

        while cx + 32 < end_x {
            let px = cx;

            let mut store_0 = vld1q_u32(buffer.get_unchecked(buf_cx..).as_ptr());
            let mut store_1 = vld1q_u32(buffer.get_unchecked(buf_cx + 4..).as_ptr());
            let mut store_2 = vld1q_u32(buffer.get_unchecked(buf_cx + 8..).as_ptr());
            let mut store_3 = vld1q_u32(buffer.get_unchecked(buf_cx + 12..).as_ptr());
            let mut store_4 = vld1q_u32(buffer.get_unchecked(buf_cx + 16..).as_ptr());
            let mut store_5 = vld1q_u32(buffer.get_unchecked(buf_cx + 20..).as_ptr());
            let mut store_6 = vld1q_u32(buffer.get_unchecked(buf_cx + 24..).as_ptr());
            let mut store_7 = vld1q_u32(buffer.get_unchecked(buf_cx + 28..).as_ptr());

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge0 = vld1q_u8(s_ptr);
                let edge1 = vld1q_u8(s_ptr.add(16));

                let lo0 = vmovl_u8(vget_low_u8(edge0));
                let hi0 = vmovl_high_u8(edge0);
                let lo1 = vmovl_u8(vget_low_u8(edge1));
                let hi1 = vmovl_high_u8(edge1);

                let i16_l0 = vget_low_u16(lo0);
                let i16_h0 = lo0;
                let i16_l1 = vget_low_u16(hi0);
                let i16_h1 = hi0;

                let i16_l01 = vget_low_u16(lo1);
                let i16_h01 = lo1;
                let i16_l11 = vget_low_u16(hi1);
                let i16_h11 = hi1;

                store_0 = vsubw_u16(store_0, i16_l0);
                store_1 = vsubw_high_u16(store_1, i16_h0);
                store_2 = vsubw_u16(store_2, i16_l1);
                store_3 = vsubw_high_u16(store_3, i16_h1);

                store_4 = vsubw_u16(store_4, i16_l01);
                store_5 = vsubw_high_u16(store_5, i16_h01);
                store_6 = vsubw_u16(store_6, i16_l11);
                store_7 = vsubw_high_u16(store_7, i16_h11);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge0 = vld1q_u8(s_ptr);
                let edge1 = vld1q_u8(s_ptr.add(16));

                let lo0 = vmovl_u8(vget_low_u8(edge0));
                let hi0 = vmovl_high_u8(edge0);
                let lo1 = vmovl_u8(vget_low_u8(edge1));
                let hi1 = vmovl_high_u8(edge1);

                let i16_l0 = vget_low_u16(lo0);
                let i16_h0 = lo0;
                let i16_l1 = vget_low_u16(hi0);
                let i16_h1 = hi0;

                let i16_l01 = vget_low_u16(lo1);
                let i16_h01 = lo1;
                let i16_l11 = vget_low_u16(hi1);
                let i16_h11 = hi1;

                store_0 = vaddw_u16(store_0, i16_l0);
                store_1 = vaddw_high_u16(store_1, i16_h0);
                store_2 = vaddw_u16(store_2, i16_l1);
                store_3 = vaddw_high_u16(store_3, i16_h1);

                store_4 = vaddw_u16(store_4, i16_l01);
                store_5 = vaddw_high_u16(store_5, i16_h01);
                store_6 = vaddw_u16(store_6, i16_l11);
                store_7 = vaddw_high_u16(store_7, i16_h11);
            }

            let px = cx;

            vst1q_u32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
            vst1q_u32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);
            vst1q_u32(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_2);
            vst1q_u32(
                buffer.get_unchecked_mut(buf_cx + 12..).as_mut_ptr(),
                store_3,
            );
            vst1q_u32(
                buffer.get_unchecked_mut(buf_cx + 16..).as_mut_ptr(),
                store_4,
            );
            vst1q_u32(
                buffer.get_unchecked_mut(buf_cx + 20..).as_mut_ptr(),
                store_5,
            );
            vst1q_u32(
                buffer.get_unchecked_mut(buf_cx + 24..).as_mut_ptr(),
                store_6,
            );
            vst1q_u32(
                buffer.get_unchecked_mut(buf_cx + 28..).as_mut_ptr(),
                store_7,
            );

            let (scale_store_0, scale_store_1, scale_store_2, scale_store_3) =
                mul_set_v4(store_0, store_1, store_2, store_3, v_weight);

            let (scale_store_4, scale_store_5, scale_store_6, scale_store_7) =
                mul_set_v4(store_4, store_5, store_6, store_7, v_weight);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vqmovn_u32(scale_store_0);
            let px_16_hi0 = vqmovn_u32(scale_store_1);
            let px_16_lo1 = vqmovn_u32(scale_store_2);
            let px_16_hi2 = vqmovn_u32(scale_store_3);
            let px_16_lo3 = vqmovn_u32(scale_store_4);
            let px_16_hi4 = vqmovn_u32(scale_store_5);
            let px_16_lo5 = vqmovn_u32(scale_store_6);
            let px_16_hi6 = vqmovn_u32(scale_store_7);
            let px0 = vqmovn_u16(vcombine_u16(px_16_lo0, px_16_hi0));
            let px1 = vqmovn_u16(vcombine_u16(px_16_lo1, px_16_hi2));
            let px2 = vqmovn_u16(vcombine_u16(px_16_lo3, px_16_hi4));
            let px3 = vqmovn_u16(vcombine_u16(px_16_lo5, px_16_hi6));

            vst1q_u8(ptr, vcombine_u8(px0, px1));
            vst1q_u8(ptr.add(16), vcombine_u8(px2, px3));

            cx += 32;
            buf_cx += 32;
        }

        while cx + 16 < end_x {
            let px = cx;

            let mut store_0 = vld1q_u32(buffer.get_unchecked(buf_cx..).as_ptr());
            let mut store_1 = vld1q_u32(buffer.get_unchecked(buf_cx + 4..).as_ptr());
            let mut store_2 = vld1q_u32(buffer.get_unchecked(buf_cx + 8..).as_ptr());
            let mut store_3 = vld1q_u32(buffer.get_unchecked(buf_cx + 12..).as_ptr());

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge = vld1q_u8(s_ptr);
                let lo0 = vmovl_u8(vget_low_u8(edge));
                let hi0 = vmovl_high_u8(edge);

                let i16_l0 = vget_low_u16(lo0);
                let i16_h0 = lo0;
                let i16_l1 = vget_low_u16(hi0);
                let i16_h1 = hi0;

                store_0 = vsubw_u16(store_0, i16_l0);
                store_1 = vsubw_high_u16(store_1, i16_h0);
                store_2 = vsubw_u16(store_2, i16_l1);
                store_3 = vsubw_high_u16(store_3, i16_h1);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge = vld1q_u8(s_ptr);
                let lo0 = vmovl_u8(vget_low_u8(edge));
                let hi0 = vmovl_high_u8(edge);

                let i16_l0 = vget_low_u16(lo0);
                let i16_h0 = lo0;
                let i16_l1 = vget_low_u16(hi0);
                let i16_h1 = hi0;

                store_0 = vaddw_u16(store_0, i16_l0);
                store_1 = vaddw_high_u16(store_1, i16_h0);
                store_2 = vaddw_u16(store_2, i16_l1);
                store_3 = vaddw_high_u16(store_3, i16_h1);
            }

            let px = cx;

            vst1q_u32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
            vst1q_u32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);
            vst1q_u32(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_2);
            vst1q_u32(
                buffer.get_unchecked_mut(buf_cx + 12..).as_mut_ptr(),
                store_3,
            );

            let (scale_store_0, scale_store_1, scale_store_2, scale_store_3) =
                mul_set_v4(store_0, store_1, store_2, store_3, v_weight);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vqmovn_u32(scale_store_0);
            let px_16_hi0 = vqmovn_u32(scale_store_1);
            let px_16_lo1 = vqmovn_u32(scale_store_2);
            let px_16_hi2 = vqmovn_u32(scale_store_3);
            let px0 = vqmovn_u16(vcombine_u16(px_16_lo0, px_16_hi0));
            let px1 = vqmovn_u16(vcombine_u16(px_16_lo1, px_16_hi2));
            vst1q_u8(ptr, vcombine_u8(px0, px1));

            cx += 16;
            buf_cx += 16;
        }

        while cx + 8 < end_x {
            let px = cx;

            let mut store_0 = vld1q_u32(buffer.get_unchecked(buf_cx..).as_ptr());
            let mut store_1 = vld1q_u32(buffer.get_unchecked(buf_cx + 4..).as_ptr());

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge = vld1_u8(s_ptr);
                let lo0 = vmovl_u8(edge);

                let i16_l0 = vget_low_u16(lo0);
                let i16_h0 = lo0;

                store_0 = vsubw_u16(store_0, i16_l0);
                store_1 = vsubw_high_u16(store_1, i16_h0);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge = vld1_u8(s_ptr);
                let lo0 = vmovl_u8(edge);

                let i16_l0 = vget_low_u16(lo0);
                let i16_h0 = lo0;

                store_0 = vaddw_u16(store_0, i16_l0);
                store_1 = vaddw_high_u16(store_1, i16_h0);
            }

            let px = cx;

            vst1q_u32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
            vst1q_u32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);

            let (scale_store_0, scale_store_1) = mul_set_v2(store_0, store_1, v_weight);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vqmovn_u32(scale_store_0);
            let px_16_hi0 = vqmovn_u32(scale_store_1);
            let px0 = vqmovn_u16(vcombine_u16(px_16_lo0, px_16_hi0));
            vst1_u8(ptr, px0);

            cx += 8;
            buf_cx += 8;
        }

        while cx < end_x {
            let px = cx;

            let mut store_0 =
                vld1q_lane_u32::<0>(buffer.get_unchecked(buf_cx..).as_ptr(), vdupq_n_u32(0));

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge = vld1_lane_u8::<0>(s_ptr, vdup_n_u8(0));
                let lo0 = vmovl_u8(edge);

                let i16_l0 = vget_low_u16(lo0);

                store_0 = vsubw_u16(store_0, i16_l0);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge = vld1_lane_u8::<0>(s_ptr, vdup_n_u8(0));
                let lo0 = vmovl_u8(edge);

                let i16_l0 = vget_low_u16(lo0);

                store_0 = vaddw_u16(store_0, i16_l0);
            }

            let px = cx;

            vst1q_lane_u32::<0>(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);

            let scale_store_0 = mul_set(store_0, v_weight);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vqmovn_u32(scale_store_0);
            let px0 = vqmovn_u16(vcombine_u16(px_16_lo0, vdup_n_u16(0)));
            vst1_lane_u8::<0>(ptr, px0);

            cx += 1;
            buf_cx += 1;
        }
    }
}
