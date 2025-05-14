/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::sse::{load_f32, store_f32};
use crate::unsafe_slice::UnsafeSlice;

pub(crate) fn box_blur_horizontal_pass_sse_f32<const CN: usize>(
    src: &[f32],
    src_stride: u32,
    dst: &UnsafeSlice<f32>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let unit = HorizontalExecutionUnit::<CN>::default();
        unit.pass(
            src, src_stride, dst, dst_stride, width, radius, start_y, end_y,
        );
    }
}

#[derive(Default, Copy, Clone)]
struct HorizontalExecutionUnit<const CN: usize> {}

impl<const CN: usize> HorizontalExecutionUnit<CN> {
    #[target_feature(enable = "sse4.1")]
    unsafe fn pass(
        &self,
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
        let v_edge_count = _mm_set1_ps(edge_count as f32);

        let v_weight = _mm_set1_ps(1f32 / (radius * 2) as f32);

        let half_kernel = kernel_size / 2;

        let mut yy = start_y;

        while yy + 4 < end_y {
            let y = yy;
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store_0: __m128;
            let mut store_1: __m128;
            let mut store_2: __m128;
            let mut store_3: __m128;

            unsafe {
                let s_ptr_0 = src.as_ptr().add(y_src_shift);
                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize);
                let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2);
                let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3);

                let edge_colors_0 = load_f32::<CN>(s_ptr_0);
                let edge_colors_1 = load_f32::<CN>(s_ptr_1);
                let edge_colors_2 = load_f32::<CN>(s_ptr_2);
                let edge_colors_3 = load_f32::<CN>(s_ptr_3);

                store_0 = _mm_mul_ps(edge_colors_0, v_edge_count);
                store_1 = _mm_mul_ps(edge_colors_1, v_edge_count);
                store_2 = _mm_mul_ps(edge_colors_2, v_edge_count);
                store_3 = _mm_mul_ps(edge_colors_3, v_edge_count);
            }

            unsafe {
                for x in 1usize..half_kernel as usize {
                    let px = x.min(width as usize - 1) * CN;

                    let s_ptr_0 = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + px);
                    let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px);
                    let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px);

                    let edge_colors_0 = load_f32::<CN>(s_ptr_0);
                    let edge_colors_1 = load_f32::<CN>(s_ptr_1);
                    let edge_colors_2 = load_f32::<CN>(s_ptr_2);
                    let edge_colors_3 = load_f32::<CN>(s_ptr_3);

                    store_0 = _mm_add_ps(store_0, edge_colors_0);
                    store_1 = _mm_add_ps(store_1, edge_colors_1);
                    store_2 = _mm_add_ps(store_2, edge_colors_2);
                    store_3 = _mm_add_ps(store_3, edge_colors_3);
                }
            }

            for x in 0..width {
                // preload edge pixels

                // subtract previous
                unsafe {
                    let previous_x = (x as i64 - half_kernel as i64).max(0) as usize;
                    let previous = previous_x * CN;

                    let s_ptr_0 = src.as_ptr().add(y_src_shift + previous);
                    let s_ptr_1 = src
                        .as_ptr()
                        .add(y_src_shift + src_stride as usize + previous);
                    let s_ptr_2 = src
                        .as_ptr()
                        .add(y_src_shift + src_stride as usize * 2 + previous);
                    let s_ptr_3 = src
                        .as_ptr()
                        .add(y_src_shift + src_stride as usize * 3 + previous);

                    let edge_colors_0 = load_f32::<CN>(s_ptr_0);
                    let edge_colors_1 = load_f32::<CN>(s_ptr_1);
                    let edge_colors_2 = load_f32::<CN>(s_ptr_2);
                    let edge_colors_3 = load_f32::<CN>(s_ptr_3);

                    store_0 = _mm_sub_ps(store_0, edge_colors_0);
                    store_1 = _mm_sub_ps(store_1, edge_colors_1);
                    store_2 = _mm_sub_ps(store_2, edge_colors_2);
                    store_3 = _mm_sub_ps(store_3, edge_colors_3);
                }

                // add next
                unsafe {
                    let next_x = (x + half_kernel).min(width - 1) as usize;

                    let next = next_x * CN;

                    let s_ptr_0 = src.as_ptr().add(y_src_shift + next);
                    let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + next);
                    let s_ptr_2 = src
                        .as_ptr()
                        .add(y_src_shift + src_stride as usize * 2 + next);
                    let s_ptr_3 = src
                        .as_ptr()
                        .add(y_src_shift + src_stride as usize * 3 + next);

                    let edge_colors_0 = load_f32::<CN>(s_ptr_0);
                    let edge_colors_1 = load_f32::<CN>(s_ptr_1);
                    let edge_colors_2 = load_f32::<CN>(s_ptr_2);
                    let edge_colors_3 = load_f32::<CN>(s_ptr_3);

                    store_0 = _mm_add_ps(store_0, edge_colors_0);
                    store_1 = _mm_add_ps(store_1, edge_colors_1);
                    store_2 = _mm_add_ps(store_2, edge_colors_2);
                    store_3 = _mm_add_ps(store_3, edge_colors_3);
                }

                let px = x as usize * CN;

                unsafe {
                    let r0 = _mm_mul_ps(store_0, v_weight);
                    let r1 = _mm_mul_ps(store_1, v_weight);
                    let r2 = _mm_mul_ps(store_2, v_weight);
                    let r3 = _mm_mul_ps(store_3, v_weight);

                    let bytes_offset_0 = y_dst_shift + px;
                    let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
                    let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
                    let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;
                    store_f32::<CN>(unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut _, r0);
                    store_f32::<CN>(unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut _, r1);
                    store_f32::<CN>(unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut _, r2);
                    store_f32::<CN>(unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut _, r3);
                }
            }

            yy += 4;
        }

        for y in yy..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store;

            unsafe {
                let s_ptr = src.as_ptr().add(y_src_shift);
                let edge_colors = load_f32::<CN>(s_ptr);
                store = _mm_mul_ps(edge_colors, v_edge_count);
            }

            unsafe {
                for x in 1usize..half_kernel as usize {
                    let px = x.min(width as usize - 1) * CN;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let edge_colors = load_f32::<CN>(s_ptr);
                    store = _mm_add_ps(store, edge_colors);
                }
            }

            for x in 0..width {
                // preload edge pixels

                // subtract previous
                unsafe {
                    let previous_x = (x as i64 - half_kernel as i64).max(0) as usize;
                    let previous = previous_x * CN;
                    let s_ptr = src.as_ptr().add(y_src_shift + previous);
                    let edge_colors = load_f32::<CN>(s_ptr);
                    store = _mm_sub_ps(store, edge_colors);
                }

                // add next
                unsafe {
                    let next_x = (x + half_kernel).min(width - 1) as usize;

                    let next = next_x * CN;

                    let s_ptr = src.as_ptr().add(y_src_shift + next);
                    let edge_colors = load_f32::<CN>(s_ptr);
                    store = _mm_add_ps(store, edge_colors);
                }

                let px = x as usize * CN;

                unsafe {
                    let r0 = _mm_mul_ps(store, v_weight);
                    let bytes_offset = y_dst_shift + px;
                    let ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut f32;
                    store_f32::<CN>(ptr, r0);
                }
            }
        }
    }
}
