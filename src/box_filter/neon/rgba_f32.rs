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

use crate::neon::{load_f32_fast, store_f32};
use crate::unsafe_slice::UnsafeSlice;

pub(crate) fn box_blur_horizontal_pass_neon_rgba_f32<const CN: usize>(
    src: &[f32],
    src_stride: u32,
    dst: &UnsafeSlice<f32>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    box_blur_horizontal_pass_neon_impl_f32::<CN>(
        src, src_stride, dst, dst_stride, width, radius, start_y, end_y,
    );
}

fn box_blur_horizontal_pass_neon_impl_f32<const CN: usize>(
    src: &[f32],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<f32>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = unsafe { vdupq_n_f32(edge_count as f32) };

    let v_weight = unsafe { vdupq_n_f32(1f32 / (radius * 2) as f32) };

    let half_kernel = kernel_size / 2;

    let mut yy = start_y;

    while yy + 4 < end_y {
        let y = yy;
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store_0: float32x4_t;
        let mut store_1: float32x4_t;
        let mut store_2: float32x4_t;
        let mut store_3: float32x4_t;

        unsafe {
            let s_ptr_0 = src.as_ptr().add(y_src_shift);
            let edge_colors_0 = load_f32_fast::<CN>(s_ptr_0);

            let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize);
            let edge_colors_1 = load_f32_fast::<CN>(s_ptr_1);

            let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2);
            let edge_colors_2 = load_f32_fast::<CN>(s_ptr_2);

            let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3);
            let edge_colors_3 = load_f32_fast::<CN>(s_ptr_3);

            store_0 = vmulq_f32(edge_colors_0, v_edge_count);
            store_1 = vmulq_f32(edge_colors_1, v_edge_count);
            store_2 = vmulq_f32(edge_colors_2, v_edge_count);
            store_3 = vmulq_f32(edge_colors_3, v_edge_count);
        }

        unsafe {
            for x in 1..half_kernel as usize {
                let px = x.min(width as usize - 1) * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + px);
                let edge_colors_0 = load_f32_fast::<CN>(s_ptr_0);

                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + px);
                let edge_colors_1 = load_f32_fast::<CN>(s_ptr_1);

                let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px);
                let edge_colors_2 = load_f32_fast::<CN>(s_ptr_2);

                let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px);
                let edge_colors_3 = load_f32_fast::<CN>(s_ptr_3);

                store_0 = vaddq_f32(store_0, edge_colors_0);
                store_1 = vaddq_f32(store_1, edge_colors_1);
                store_2 = vaddq_f32(store_2, edge_colors_2);
                store_3 = vaddq_f32(store_3, edge_colors_3);
            }
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            unsafe {
                let previous_x = (x as i64 - half_kernel as i64).max(0) as usize;
                let previous = previous_x * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + previous);
                let edge_colors_0 = load_f32_fast::<CN>(s_ptr_0);

                let s_ptr_1 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize + previous);
                let edge_colors_1 = load_f32_fast::<CN>(s_ptr_1);

                let s_ptr_2 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 2 + previous);
                let edge_colors_2 = load_f32_fast::<CN>(s_ptr_2);

                let s_ptr_3 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 3 + previous);
                let edge_colors_3 = load_f32_fast::<CN>(s_ptr_3);

                store_0 = vsubq_f32(store_0, edge_colors_0);
                store_1 = vsubq_f32(store_1, edge_colors_1);
                store_2 = vsubq_f32(store_2, edge_colors_2);
                store_3 = vsubq_f32(store_3, edge_colors_3);
            }

            // add next
            unsafe {
                let next_x = (x + half_kernel).min(width - 1) as usize;

                let next = next_x * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + next);
                let edge_colors_0 = load_f32_fast::<CN>(s_ptr_0);

                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + next);
                let edge_colors_1 = load_f32_fast::<CN>(s_ptr_1);

                let s_ptr_2 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 2 + next);
                let edge_colors_2 = load_f32_fast::<CN>(s_ptr_2);

                let s_ptr_3 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 3 + next);
                let edge_colors_3 = load_f32_fast::<CN>(s_ptr_3);

                store_0 = vaddq_f32(store_0, edge_colors_0);
                store_1 = vaddq_f32(store_1, edge_colors_1);
                store_2 = vaddq_f32(store_2, edge_colors_2);
                store_3 = vaddq_f32(store_3, edge_colors_3);
            }

            let px = x as usize * CN;

            unsafe {
                let ss0 = vmulq_f32(store_0, v_weight);
                let ss1 = vmulq_f32(store_1, v_weight);
                let ss2 = vmulq_f32(store_2, v_weight);
                let ss3 = vmulq_f32(store_3, v_weight);

                let bytes_offset_0 = y_dst_shift + px;
                let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
                let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
                let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;

                store_f32::<CN>(unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut _, ss0);
                store_f32::<CN>(unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut _, ss1);
                store_f32::<CN>(unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut _, ss2);
                store_f32::<CN>(unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut _, ss3);
            }
        }

        yy += 4;
    }

    for y in yy..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store: float32x4_t;
        unsafe {
            let s_ptr = src.as_ptr().add(y_src_shift);
            let edge_colors = load_f32_fast::<CN>(s_ptr);
            store = vmulq_f32(edge_colors, v_edge_count);
        }

        unsafe {
            for x in 1..half_kernel as usize {
                let px = x.min(width as usize - 1) * CN;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let edge_colors = load_f32_fast::<CN>(s_ptr);
                store = vaddq_f32(store, edge_colors);
            }
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            unsafe {
                let previous_x = (x as isize - half_kernel as isize).max(0) as usize;
                let previous = previous_x * CN;
                let s_ptr = src.as_ptr().add(y_src_shift + previous);
                let edge_colors = load_f32_fast::<CN>(s_ptr);
                store = vsubq_f32(store, edge_colors);
            }

            // add next
            unsafe {
                let next_x = (x + half_kernel).min(width - 1) as usize;

                let next = next_x * CN;

                let s_ptr = src.as_ptr().add(y_src_shift + next);
                let edge_colors = load_f32_fast::<CN>(s_ptr);
                store = vaddq_f32(store, edge_colors);
            }

            let px = x as usize * CN;

            let scale_store = unsafe { vmulq_f32(store, v_weight) };

            let bytes_offset = y_dst_shift + px;
            unsafe {
                let dst_ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut f32;
                store_f32::<CN>(dst_ptr, scale_store);
            }
        }
    }
}

pub(crate) fn box_blur_vertical_pass_neon_rgba_f32(
    src: &[f32],
    src_stride: u32,
    dst: &UnsafeSlice<f32>,
    dst_stride: u32,
    w: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    unsafe {
        box_blur_vertical_pass_neon_any_rgba_f32(
            src, src_stride, dst, dst_stride, w, height, radius, start_x, end_x,
        );
    }
}

unsafe fn box_blur_vertical_pass_neon_any_rgba_f32(
    src: &[f32],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<f32>,
    dst_stride: u32,
    _: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = vdupq_n_f32(edge_count as f32);

    let half_kernel = kernel_size / 2;

    let start_x = start_x as usize;
    let end_x = end_x as usize;

    let v_weight = vdupq_n_f32(1f32 / (radius * 2) as f32);

    assert!(end_x >= start_x);

    let buf_size = end_x - start_x;

    let buf_cap = buf_size.div_ceil(4) * 4 + 4;

    let mut buffer = vec![0f32; buf_cap];

    let mut cx = start_x;
    let mut buf_cx = 0usize;

    while cx + 16 < end_x {
        let px = cx;

        let mut store_0: float32x4_t;
        let mut store_1: float32x4_t;
        let mut store_2: float32x4_t;
        let mut store_3: float32x4_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1q_f32(s_ptr);
        let edge1 = vld1q_f32(s_ptr.add(4));
        let edge2 = vld1q_f32(s_ptr.add(8));
        let edge3 = vld1q_f32(s_ptr.add(12));

        store_0 = vmulq_f32(edge0, v_edge_count);
        store_1 = vmulq_f32(edge1, v_edge_count);

        store_2 = vmulq_f32(edge2, v_edge_count);
        store_3 = vmulq_f32(edge3, v_edge_count);

        for y in 1..half_kernel.min(height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1q_f32(s_ptr);
            let edge1 = vld1q_f32(s_ptr.add(4));
            let edge2 = vld1q_f32(s_ptr.add(8));
            let edge3 = vld1q_f32(s_ptr.add(12));

            store_0 = vaddq_f32(store_0, edge0);
            store_1 = vaddq_f32(store_1, edge1);
            store_2 = vaddq_f32(store_2, edge2);
            store_3 = vaddq_f32(store_3, edge3);
        }

        vst1q_f32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
        vst1q_f32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);
        vst1q_f32(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_2);
        vst1q_f32(
            buffer.get_unchecked_mut(buf_cx + 12..).as_mut_ptr(),
            store_3,
        );

        cx += 16;
        buf_cx += 16;
    }

    while cx + 8 < end_x {
        let px = cx;

        let mut store_0: float32x4_t;
        let mut store_1: float32x4_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1q_f32(s_ptr);
        let edge1 = vld1q_f32(s_ptr.add(4));

        store_0 = vmulq_f32(edge0, v_edge_count);
        store_1 = vmulq_f32(edge1, v_edge_count);

        for y in 1..half_kernel.min(height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1q_f32(s_ptr);
            let edge1 = vld1q_f32(s_ptr.add(4));

            store_0 = vaddq_f32(store_0, edge0);
            store_1 = vaddq_f32(store_1, edge1);
        }

        vst1q_f32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
        vst1q_f32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);

        cx += 8;
        buf_cx += 8;
    }

    while cx + 4 < end_x {
        let px = cx;

        let mut store_0: float32x4_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1q_f32(s_ptr);

        store_0 = vmulq_f32(edge0, v_edge_count);

        for y in 1..half_kernel.min(height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1q_f32(s_ptr);

            store_0 = vaddq_f32(store_0, edge0);
        }

        vst1q_f32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);

        cx += 4;
        buf_cx += 4;
    }

    while cx < end_x {
        let px = cx;

        let mut store_0: float32x4_t;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = vld1q_lane_f32::<0>(s_ptr, vdupq_n_f32(0.));

        store_0 = vmulq_f32(edge0, v_edge_count);

        for y in 1..half_kernel.min(height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = vld1q_lane_f32::<0>(s_ptr, vdupq_n_f32(0.));
            store_0 = vaddq_f32(store_0, edge0);
        }

        vst1q_lane_f32::<0>(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);

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

        while cx + 16 < end_x {
            let px = cx;

            let mut store_0 = vld1q_f32(buffer.get_unchecked(buf_cx..).as_ptr());
            let mut store_1 = vld1q_f32(buffer.get_unchecked(buf_cx + 4..).as_ptr());
            let mut store_2 = vld1q_f32(buffer.get_unchecked(buf_cx + 8..).as_ptr());
            let mut store_3 = vld1q_f32(buffer.get_unchecked(buf_cx + 12..).as_ptr());

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);

                let edge0 = vld1q_f32(s_ptr);
                let edge1 = vld1q_f32(s_ptr.add(4));
                let edge2 = vld1q_f32(s_ptr.add(8));
                let edge3 = vld1q_f32(s_ptr.add(12));

                store_0 = vsubq_f32(store_0, edge0);
                store_1 = vsubq_f32(store_1, edge1);
                store_2 = vsubq_f32(store_2, edge2);
                store_3 = vsubq_f32(store_3, edge3);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);

                let edge0 = vld1q_f32(s_ptr);
                let edge1 = vld1q_f32(s_ptr.add(4));
                let edge2 = vld1q_f32(s_ptr.add(8));
                let edge3 = vld1q_f32(s_ptr.add(12));

                store_0 = vaddq_f32(store_0, edge0);
                store_1 = vaddq_f32(store_1, edge1);
                store_2 = vaddq_f32(store_2, edge2);
                store_3 = vaddq_f32(store_3, edge3);
            }

            let px = cx;

            vst1q_f32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
            vst1q_f32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);
            vst1q_f32(buffer.get_unchecked_mut(buf_cx + 8..).as_mut_ptr(), store_2);
            vst1q_f32(
                buffer.get_unchecked_mut(buf_cx + 12..).as_mut_ptr(),
                store_3,
            );

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vmulq_f32(store_0, v_weight);
            let px_16_hi0 = vmulq_f32(store_1, v_weight);
            let px_16_lo1 = vmulq_f32(store_2, v_weight);
            let px_16_hi2 = vmulq_f32(store_3, v_weight);

            vst1q_f32(ptr, px_16_lo0);
            vst1q_f32(ptr.add(4), px_16_hi0);
            vst1q_f32(ptr.add(8), px_16_lo1);
            vst1q_f32(ptr.add(12), px_16_hi2);

            cx += 16;
            buf_cx += 16;
        }

        while cx + 8 < end_x {
            let px = cx;

            let mut store_0 = vld1q_f32(buffer.get_unchecked(buf_cx..).as_ptr());
            let mut store_1 = vld1q_f32(buffer.get_unchecked(buf_cx + 4..).as_ptr());

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge0 = vld1q_f32(s_ptr);
                let edge1 = vld1q_f32(s_ptr.add(4));

                store_0 = vsubq_f32(store_0, edge0);
                store_1 = vsubq_f32(store_1, edge1);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge0 = vld1q_f32(s_ptr);
                let edge1 = vld1q_f32(s_ptr.add(4));

                store_0 = vaddq_f32(store_0, edge0);
                store_1 = vaddq_f32(store_1, edge1);
            }

            let px = cx;

            vst1q_f32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);
            vst1q_f32(buffer.get_unchecked_mut(buf_cx + 4..).as_mut_ptr(), store_1);

            let px_16_lo0 = vmulq_f32(store_0, v_weight);
            let px_16_hi0 = vmulq_f32(store_1, v_weight);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();

            vst1q_f32(ptr, px_16_lo0);
            vst1q_f32(ptr.add(4), px_16_hi0);

            cx += 8;
            buf_cx += 8;
        }

        while cx + 4 < end_x {
            let px = cx;

            let mut store_0 = vld1q_f32(buffer.get_unchecked(buf_cx..).as_ptr());

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge = vld1q_f32(s_ptr);

                store_0 = vsubq_f32(store_0, edge);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge = vld1q_f32(s_ptr);

                store_0 = vaddq_f32(store_0, edge);
            }

            let px = cx;

            vst1q_f32(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vmulq_f32(store_0, v_weight);
            vst1q_f32(ptr, px_16_lo0);

            cx += 4;
            buf_cx += 4;
        }

        while cx < end_x {
            let px = cx;

            let mut store_0 =
                vld1q_lane_f32::<0>(buffer.get_unchecked(buf_cx..).as_ptr(), vdupq_n_f32(0.));

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge = vld1q_lane_f32::<0>(s_ptr, vdupq_n_f32(0.));

                store_0 = vsubq_f32(store_0, edge);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge = vld1q_lane_f32::<0>(s_ptr, vdupq_n_f32(0.));
                store_0 = vaddq_f32(store_0, edge);
            }

            let px = cx;

            vst1q_lane_f32::<0>(buffer.get_unchecked_mut(buf_cx..).as_mut_ptr(), store_0);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();
            let px_16_lo0 = vmulq_f32(store_0, v_weight);
            vst1q_lane_f32::<0>(ptr, px_16_lo0);

            cx += 1;
            buf_cx += 1;
        }
    }
}
