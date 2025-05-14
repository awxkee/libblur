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

use crate::sse::{_mm_mul_ps_epi32, load_u16_s32_fast, store_u16_u32};
use crate::unsafe_slice::UnsafeSlice;

pub(crate) fn box_blur_horizontal_pass_sse16<const CN: usize>(
    src: &[u16],
    src_stride: u32,
    dst: &UnsafeSlice<u16>,
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
        src: &[u16],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u16>,
        dst_stride: u32,
        width: u32,
        radius: u32,
        start_y: u32,
        end_y: u32,
    ) {
        let kernel_size = radius * 2 + 1;
        let edge_count = (kernel_size / 2) + 1;
        let v_edge_count = _mm_set1_epi32(edge_count as i32);

        let v_weight = _mm_set1_ps(1f32 / (radius * 2) as f32);

        let half_kernel = kernel_size / 2;

        let mut yy = start_y;

        while yy + 4 < end_y {
            let y = yy;
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store_0: __m128i;
            let mut store_1: __m128i;
            let mut store_2: __m128i;
            let mut store_3: __m128i;

            unsafe {
                let s_ptr_0 = src.as_ptr().add(y_src_shift);
                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize);
                let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2);
                let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3);

                let edge_colors_0 = load_u16_s32_fast::<CN>(s_ptr_0);
                let edge_colors_1 = load_u16_s32_fast::<CN>(s_ptr_1);
                let edge_colors_2 = load_u16_s32_fast::<CN>(s_ptr_2);
                let edge_colors_3 = load_u16_s32_fast::<CN>(s_ptr_3);

                store_0 = _mm_mullo_epi32(edge_colors_0, v_edge_count);
                store_1 = _mm_mullo_epi32(edge_colors_1, v_edge_count);
                store_2 = _mm_mullo_epi32(edge_colors_2, v_edge_count);
                store_3 = _mm_mullo_epi32(edge_colors_3, v_edge_count);
            }

            unsafe {
                for x in 1usize..half_kernel as usize {
                    let px = x.min(width as usize - 1) * CN;

                    let s_ptr_0 = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + px);
                    let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px);
                    let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px);

                    let edge_colors_0 = load_u16_s32_fast::<CN>(s_ptr_0);
                    let edge_colors_1 = load_u16_s32_fast::<CN>(s_ptr_1);
                    let edge_colors_2 = load_u16_s32_fast::<CN>(s_ptr_2);
                    let edge_colors_3 = load_u16_s32_fast::<CN>(s_ptr_3);

                    store_0 = _mm_add_epi32(store_0, edge_colors_0);
                    store_1 = _mm_add_epi32(store_1, edge_colors_1);
                    store_2 = _mm_add_epi32(store_2, edge_colors_2);
                    store_3 = _mm_add_epi32(store_3, edge_colors_3);
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

                    let edge_colors_0 = load_u16_s32_fast::<CN>(s_ptr_0);
                    let edge_colors_1 = load_u16_s32_fast::<CN>(s_ptr_1);
                    let edge_colors_2 = load_u16_s32_fast::<CN>(s_ptr_2);
                    let edge_colors_3 = load_u16_s32_fast::<CN>(s_ptr_3);

                    store_0 = _mm_sub_epi32(store_0, edge_colors_0);
                    store_1 = _mm_sub_epi32(store_1, edge_colors_1);
                    store_2 = _mm_sub_epi32(store_2, edge_colors_2);
                    store_3 = _mm_sub_epi32(store_3, edge_colors_3);
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

                    let edge_colors_0 = load_u16_s32_fast::<CN>(s_ptr_0);
                    let edge_colors_1 = load_u16_s32_fast::<CN>(s_ptr_1);
                    let edge_colors_2 = load_u16_s32_fast::<CN>(s_ptr_2);
                    let edge_colors_3 = load_u16_s32_fast::<CN>(s_ptr_3);

                    store_0 = _mm_add_epi32(store_0, edge_colors_0);
                    store_1 = _mm_add_epi32(store_1, edge_colors_1);
                    store_2 = _mm_add_epi32(store_2, edge_colors_2);
                    store_3 = _mm_add_epi32(store_3, edge_colors_3);
                }

                let px = x as usize * CN;

                unsafe {
                    let scale_store_ps0 = _mm_cvtepi32_ps(store_0);
                    let scale_store_ps1 = _mm_cvtepi32_ps(store_1);
                    let scale_store_ps2 = _mm_cvtepi32_ps(store_2);
                    let scale_store_ps3 = _mm_cvtepi32_ps(store_3);

                    let r0 = _mm_mul_ps(scale_store_ps0, v_weight);
                    let r1 = _mm_mul_ps(scale_store_ps1, v_weight);
                    let r2 = _mm_mul_ps(scale_store_ps2, v_weight);
                    let r3 = _mm_mul_ps(scale_store_ps3, v_weight);

                    let scale_store0 = _mm_cvtps_epi32(r0);
                    let scale_store1 = _mm_cvtps_epi32(r1);
                    let scale_store2 = _mm_cvtps_epi32(r2);
                    let scale_store3 = _mm_cvtps_epi32(r3);

                    let bytes_offset_0 = y_dst_shift + px;
                    let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
                    let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
                    let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;
                    store_u16_u32::<CN>(
                        unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut _,
                        scale_store0,
                    );
                    store_u16_u32::<CN>(
                        unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut _,
                        scale_store1,
                    );
                    store_u16_u32::<CN>(
                        unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut _,
                        scale_store2,
                    );
                    store_u16_u32::<CN>(
                        unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut _,
                        scale_store3,
                    );
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
                let edge_colors = load_u16_s32_fast::<CN>(s_ptr);
                store = _mm_mullo_epi32(edge_colors, v_edge_count);
            }

            unsafe {
                for x in 1usize..half_kernel as usize {
                    let px = x.min(width as usize - 1) * CN;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let edge_colors = load_u16_s32_fast::<CN>(s_ptr);
                    store = _mm_add_epi32(store, edge_colors);
                }
            }

            for x in 0..width {
                // preload edge pixels

                // subtract previous
                unsafe {
                    let previous_x = (x as i64 - half_kernel as i64).max(0) as usize;
                    let previous = previous_x * CN;
                    let s_ptr = src.as_ptr().add(y_src_shift + previous);
                    let edge_colors = load_u16_s32_fast::<CN>(s_ptr);
                    store = _mm_sub_epi32(store, edge_colors);
                }

                // add next
                unsafe {
                    let next_x = (x + half_kernel).min(width - 1) as usize;

                    let next = next_x * CN;

                    let s_ptr = src.as_ptr().add(y_src_shift + next);
                    let edge_colors = load_u16_s32_fast::<CN>(s_ptr);
                    store = _mm_add_epi32(store, edge_colors);
                }

                let px = x as usize * CN;

                unsafe {
                    let r0 = _mm_mul_ps(_mm_cvtepi32_ps(store), v_weight);
                    let bytes_offset = y_dst_shift + px;
                    let ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u16;
                    store_u16_u32::<CN>(ptr, _mm_cvtps_epi32(r0));
                }
            }
        }
    }
}

pub(crate) fn box_blur_vertical_pass_sse16(
    src: &[u16],
    src_stride: u32,
    dst: &UnsafeSlice<u16>,
    dst_stride: u32,
    w: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    unsafe {
        box_blur_vertical_pass_sse_impl16(
            src, src_stride, dst, dst_stride, w, height, radius, start_x, end_x,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn box_blur_vertical_pass_sse_impl16(
    src: &[u16],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u16>,
    dst_stride: u32,
    _: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = _mm_set1_epi32(edge_count as i32);

    let v_weight = _mm_set1_ps(1f32 / (radius * 2) as f32);

    let half_kernel = kernel_size / 2;

    let mut cx = start_x;

    while cx + 16 < end_x {
        let px = cx as usize;

        let mut store_0: __m128i;
        let mut store_1: __m128i;
        let mut store_2: __m128i;
        let mut store_3: __m128i;

        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge0 = _mm_loadu_si128(s_ptr as *const _);
            let edge1 = _mm_loadu_si128(s_ptr.add(8) as *const _);

            let i16_l0 = _mm_unpacklo_epi16(edge0, _mm_setzero_si128());
            let i16_h0 = _mm_unpackhi_epi16(edge0, _mm_setzero_si128());
            let i16_l1 = _mm_unpacklo_epi16(edge1, _mm_setzero_si128());
            let i16_h1 = _mm_unpackhi_epi16(edge1, _mm_setzero_si128());

            store_0 = _mm_mullo_epi32(i16_l0, v_edge_count);
            store_1 = _mm_mullo_epi32(i16_h0, v_edge_count);

            store_2 = _mm_mullo_epi32(i16_l1, v_edge_count);
            store_3 = _mm_mullo_epi32(i16_h1, v_edge_count);
        }

        unsafe {
            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge0 = _mm_loadu_si128(s_ptr as *const _);
                let edge1 = _mm_loadu_si128(s_ptr.add(8) as *const _);

                let i16_l0 = _mm_unpacklo_epi16(edge0, _mm_setzero_si128());
                let i16_h0 = _mm_unpackhi_epi16(edge0, _mm_setzero_si128());
                let i16_l1 = _mm_unpacklo_epi16(edge1, _mm_setzero_si128());
                let i16_h1 = _mm_unpackhi_epi16(edge1, _mm_setzero_si128());

                store_0 = _mm_add_epi32(store_0, i16_l0);
                store_1 = _mm_add_epi32(store_1, i16_h0);

                store_2 = _mm_add_epi32(store_2, i16_l1);
                store_3 = _mm_add_epi32(store_3, i16_h1);
            }
        }

        unsafe {
            for y in 0..height {
                // preload edge pixels
                let next = (y + half_kernel).min(height - 1) as usize * src_stride as usize;
                let previous =
                    (y as i64 - half_kernel as i64).max(0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge0 = _mm_loadu_si128(s_ptr as *const _);
                    let edge1 = _mm_loadu_si128(s_ptr.add(8) as *const _);

                    let i16_l0 = _mm_unpacklo_epi16(edge0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(edge0, _mm_setzero_si128());
                    let i16_l1 = _mm_unpacklo_epi16(edge1, _mm_setzero_si128());
                    let i16_h1 = _mm_unpackhi_epi16(edge1, _mm_setzero_si128());

                    store_0 = _mm_sub_epi32(store_0, i16_l0);
                    store_1 = _mm_sub_epi32(store_1, i16_h0);

                    store_2 = _mm_sub_epi32(store_2, i16_l1);
                    store_3 = _mm_sub_epi32(store_3, i16_h1);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge0 = _mm_loadu_si128(s_ptr as *const _);
                    let edge1 = _mm_loadu_si128(s_ptr.add(8) as *const _);

                    let i16_l0 = _mm_unpacklo_epi16(edge0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(edge0, _mm_setzero_si128());
                    let i16_l1 = _mm_unpacklo_epi16(edge1, _mm_setzero_si128());
                    let i16_h1 = _mm_unpackhi_epi16(edge1, _mm_setzero_si128());

                    store_0 = _mm_add_epi32(store_0, i16_l0);
                    store_1 = _mm_add_epi32(store_1, i16_h0);

                    store_2 = _mm_add_epi32(store_2, i16_l1);
                    store_3 = _mm_add_epi32(store_3, i16_h1);
                }

                let px = cx as usize;

                let store0_ps = _mm_cvtepi32_ps(store_0);
                let store1_ps = _mm_cvtepi32_ps(store_1);
                let store2_ps = _mm_cvtepi32_ps(store_2);
                let store3_ps = _mm_cvtepi32_ps(store_3);

                let r0 = _mm_mul_ps(store0_ps, v_weight);
                let r1 = _mm_mul_ps(store1_ps, v_weight);
                let r2 = _mm_mul_ps(store2_ps, v_weight);
                let r3 = _mm_mul_ps(store3_ps, v_weight);

                let scale_store_0 = _mm_cvtps_epi32(r0);
                let scale_store_1 = _mm_cvtps_epi32(r1);
                let scale_store_2 = _mm_cvtps_epi32(r2);
                let scale_store_3 = _mm_cvtps_epi32(r3);

                let offset = y_dst_shift + px;
                let ptr = unsafe_dst.slice.get_unchecked(offset).get();

                let set0 = _mm_packus_epi32(scale_store_0, scale_store_1);
                let set1 = _mm_packus_epi32(scale_store_2, scale_store_3);

                _mm_storeu_si128(ptr as *mut _, set0);
                _mm_storeu_si128(ptr.add(8) as *mut _, set1);
            }
        }

        cx += 16;
    }

    while cx + 8 < end_x {
        let px = cx as usize;

        let mut store_0: __m128i;
        let mut store_1: __m128i;

        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge = _mm_loadu_si128(s_ptr as *const _);

            let i16_l0 = _mm_unpacklo_epi16(edge, _mm_setzero_si128());
            let i16_h0 = _mm_unpackhi_epi16(edge, _mm_setzero_si128());

            store_0 = _mm_mullo_epi32(i16_l0, v_edge_count);
            store_1 = _mm_mullo_epi32(i16_h0, v_edge_count);
        }

        unsafe {
            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge = _mm_loadu_si128(s_ptr as *const _);

                let i16_l0 = _mm_unpacklo_epi16(edge, _mm_setzero_si128());
                let i16_h0 = _mm_unpackhi_epi16(edge, _mm_setzero_si128());

                store_0 = _mm_add_epi32(store_0, i16_l0);
                store_1 = _mm_add_epi32(store_1, i16_h0);
            }
        }

        unsafe {
            for y in 0..height {
                // preload edge pixels
                let next =
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge = _mm_loadu_si128(s_ptr as *const _);

                    let i16_l0 = _mm_unpacklo_epi16(edge, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(edge, _mm_setzero_si128());

                    store_0 = _mm_sub_epi32(store_0, i16_l0);
                    store_1 = _mm_sub_epi32(store_1, i16_h0);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge = _mm_loadu_si128(s_ptr as *const _);

                    let i16_l0 = _mm_unpacklo_epi16(edge, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(edge, _mm_setzero_si128());

                    store_0 = _mm_add_epi32(store_0, i16_l0);
                    store_1 = _mm_add_epi32(store_1, i16_h0);
                }

                let px = cx as usize;

                let store0_ps = _mm_cvtepi32_ps(store_0);
                let store1_ps = _mm_cvtepi32_ps(store_1);

                let scale_store_0_ps = _mm_mul_ps(store0_ps, v_weight);
                let scale_store_1_ps = _mm_mul_ps(store1_ps, v_weight);

                let scale_store_0 = _mm_cvtps_epi32(scale_store_0_ps);
                let scale_store_1 = _mm_cvtps_epi32(scale_store_1_ps);

                let offset = y_dst_shift + px;
                let ptr = unsafe_dst.slice.get_unchecked(offset).get();

                let set0 = _mm_packus_epi32(scale_store_0, scale_store_1);

                _mm_storeu_si128(ptr as *mut _, set0);
            }
        }

        cx += 8;
    }

    const TAIL_CN: usize = 1;

    for x in cx..end_x {
        let px = x as usize;

        let mut store;
        {
            let s_ptr = src.as_ptr().add(px);
            let edge_colors = load_u16_s32_fast::<TAIL_CN>(s_ptr);
            store = _mm_mullo_epi32(edge_colors, v_edge_count);
        }

        for y in 1..std::cmp::min(half_kernel, height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);
            let edge_colors = load_u16_s32_fast::<TAIL_CN>(s_ptr);
            store = _mm_add_epi32(store, edge_colors);
        }

        for y in 0..height {
            // preload edge pixels
            let next = std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
            let previous =
                std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
            let y_dst_shift = dst_stride as usize * y as usize;

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge_colors = load_u16_s32_fast::<TAIL_CN>(s_ptr);
                store = _mm_sub_epi32(store, edge_colors);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge_colors = load_u16_s32_fast::<TAIL_CN>(s_ptr);
                store = _mm_add_epi32(store, edge_colors);
            }

            let px = x as usize;

            let scale_store = _mm_mul_ps_epi32(store, v_weight);

            let ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u16;
            store_u16_u32::<TAIL_CN>(ptr, scale_store);
        }
    }
}
