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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::sse::{_mm_mul_ps_epi32, _mm_mul_round_ps, load_u8_s32_fast, store_u8_u32, write_u8};
use crate::unsafe_slice::UnsafeSlice;

pub(crate) fn box_blur_horizontal_pass_sse<T, const CHANNELS: usize>(
    undefined_src: &[T],
    src_stride: u32,
    undefined_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let src: &[u8] = std::mem::transmute(undefined_src);
        let unsafe_dst: &UnsafeSlice<'_, u8> = std::mem::transmute(undefined_unsafe_dst);
        if std::arch::is_x86_feature_detected!("fma") {
            box_blur_horizontal_pass_fma::<CHANNELS>(
                src, src_stride, unsafe_dst, dst_stride, width, radius, start_y, end_y,
            );
        } else {
            box_blur_horizontal_pass_def::<CHANNELS>(
                src, src_stride, unsafe_dst, dst_stride, width, radius, start_y, end_y,
            );
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn box_blur_horizontal_pass_def<const CHANNELS: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    box_blur_horizontal_pass_impl::<CHANNELS, false>(
        src, src_stride, unsafe_dst, dst_stride, width, radius, start_y, end_y,
    )
}

#[target_feature(enable = "sse4.1", enable = "fma")]
unsafe fn box_blur_horizontal_pass_fma<const CHANNELS: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    box_blur_horizontal_pass_impl::<CHANNELS, true>(
        src, src_stride, unsafe_dst, dst_stride, width, radius, start_y, end_y,
    )
}

#[inline(always)]
unsafe fn box_blur_horizontal_pass_impl<const CHANNELS: usize, const FMA: bool>(
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
    let v_edge_count = unsafe { _mm_set1_epi32(edge_count as i32) };

    let v_weight = unsafe { _mm_set1_ps(1f32 / (radius * 2) as f32) };

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

            let edge_colors_0 = load_u8_s32_fast::<CHANNELS>(s_ptr_0);
            let edge_colors_1 = load_u8_s32_fast::<CHANNELS>(s_ptr_1);
            let edge_colors_2 = load_u8_s32_fast::<CHANNELS>(s_ptr_2);
            let edge_colors_3 = load_u8_s32_fast::<CHANNELS>(s_ptr_3);

            store_0 = _mm_madd_epi16(edge_colors_0, v_edge_count);
            store_1 = _mm_madd_epi16(edge_colors_1, v_edge_count);
            store_2 = _mm_madd_epi16(edge_colors_2, v_edge_count);
            store_3 = _mm_madd_epi16(edge_colors_3, v_edge_count);
        }

        unsafe {
            for x in 1..std::cmp::min(half_kernel, width) {
                let px = x as usize * CHANNELS;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + px);
                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + px);
                let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px);
                let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px);

                let edge_colors_0 = load_u8_s32_fast::<CHANNELS>(s_ptr_0);
                let edge_colors_1 = load_u8_s32_fast::<CHANNELS>(s_ptr_1);
                let edge_colors_2 = load_u8_s32_fast::<CHANNELS>(s_ptr_2);
                let edge_colors_3 = load_u8_s32_fast::<CHANNELS>(s_ptr_3);

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
                let previous_x = std::cmp::max(x as i64 - half_kernel as i64, 0) as usize;
                let previous = previous_x * CHANNELS;

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

                let edge_colors_0 = load_u8_s32_fast::<CHANNELS>(s_ptr_0);
                let edge_colors_1 = load_u8_s32_fast::<CHANNELS>(s_ptr_1);
                let edge_colors_2 = load_u8_s32_fast::<CHANNELS>(s_ptr_2);
                let edge_colors_3 = load_u8_s32_fast::<CHANNELS>(s_ptr_3);

                store_0 = _mm_sub_epi32(store_0, edge_colors_0);
                store_1 = _mm_sub_epi32(store_1, edge_colors_1);
                store_2 = _mm_sub_epi32(store_2, edge_colors_2);
                store_3 = _mm_sub_epi32(store_3, edge_colors_3);
            }

            // add next
            unsafe {
                let next_x = std::cmp::min(x + half_kernel, width - 1) as usize;

                let next = next_x * CHANNELS;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + next);
                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + next);
                let s_ptr_2 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 2 + next);
                let s_ptr_3 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 3 + next);

                let edge_colors_0 = load_u8_s32_fast::<CHANNELS>(s_ptr_0);
                let edge_colors_1 = load_u8_s32_fast::<CHANNELS>(s_ptr_1);
                let edge_colors_2 = load_u8_s32_fast::<CHANNELS>(s_ptr_2);
                let edge_colors_3 = load_u8_s32_fast::<CHANNELS>(s_ptr_3);

                store_0 = _mm_add_epi32(store_0, edge_colors_0);
                store_1 = _mm_add_epi32(store_1, edge_colors_1);
                store_2 = _mm_add_epi32(store_2, edge_colors_2);
                store_3 = _mm_add_epi32(store_3, edge_colors_3);
            }

            let px = x as usize * CHANNELS;

            unsafe {
                let scale_store_ps0 = _mm_cvtepi32_ps(store_0);
                let scale_store_ps1 = _mm_cvtepi32_ps(store_1);
                let scale_store_ps2 = _mm_cvtepi32_ps(store_2);
                let scale_store_ps3 = _mm_cvtepi32_ps(store_3);

                let (mut r0, mut r1, mut r2, mut r3);

                if FMA {
                    r0 = _mm_mul_round_ps(scale_store_ps0, v_weight);
                    r1 = _mm_mul_round_ps(scale_store_ps1, v_weight);
                    r2 = _mm_mul_round_ps(scale_store_ps2, v_weight);
                    r3 = _mm_mul_round_ps(scale_store_ps3, v_weight);
                } else {
                    r0 = _mm_mul_ps(scale_store_ps0, v_weight);
                    r1 = _mm_mul_ps(scale_store_ps1, v_weight);
                    r2 = _mm_mul_ps(scale_store_ps2, v_weight);
                    r3 = _mm_mul_ps(scale_store_ps3, v_weight);

                    const RND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

                    r0 = _mm_round_ps::<RND>(r0);
                    r1 = _mm_round_ps::<RND>(r1);
                    r2 = _mm_round_ps::<RND>(r2);
                    r3 = _mm_round_ps::<RND>(r3);
                }

                let scale_store0 = _mm_cvtps_epi32(r0);
                let scale_store1 = _mm_cvtps_epi32(r1);
                let scale_store2 = _mm_cvtps_epi32(r2);
                let scale_store3 = _mm_cvtps_epi32(r3);

                let px_160 = _mm_packus_epi32(scale_store0, _mm_setzero_si128());
                let px_161 = _mm_packus_epi32(scale_store1, _mm_setzero_si128());
                let px_162 = _mm_packus_epi32(scale_store2, _mm_setzero_si128());
                let px_163 = _mm_packus_epi32(scale_store3, _mm_setzero_si128());

                let px_80 = _mm_packus_epi16(px_160, _mm_setzero_si128());
                let px_81 = _mm_packus_epi16(px_161, _mm_setzero_si128());
                let px_82 = _mm_packus_epi16(px_162, _mm_setzero_si128());
                let px_83 = _mm_packus_epi16(px_163, _mm_setzero_si128());

                let bytes_offset_0 = y_dst_shift + px;
                let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
                let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
                let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;
                if CHANNELS == 4 {
                    let dst_ptr_0 = unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut u8;
                    _mm_storeu_si32(dst_ptr_0, px_80);

                    let dst_ptr_1 = unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut u8;
                    _mm_storeu_si32(dst_ptr_1, px_81);

                    let dst_ptr_2 = unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut u8;
                    _mm_storeu_si32(dst_ptr_2, px_82);

                    let dst_ptr_3 = unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut u8;
                    _mm_storeu_si32(dst_ptr_3, px_83);
                } else {
                    let dst_ptr_0 = unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut u8;
                    write_u8::<CHANNELS>(dst_ptr_0, px_80);

                    let dst_ptr_1 = unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut u8;
                    write_u8::<CHANNELS>(dst_ptr_1, px_81);

                    let dst_ptr_2 = unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut u8;
                    write_u8::<CHANNELS>(dst_ptr_2, px_82);

                    let dst_ptr_3 = unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut u8;
                    write_u8::<CHANNELS>(dst_ptr_3, px_83);
                }
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
            let edge_colors = load_u8_s32_fast::<CHANNELS>(s_ptr);
            store = _mm_madd_epi16(edge_colors, v_edge_count);
        }

        unsafe {
            for x in 1..std::cmp::min(half_kernel, width) {
                let px = x as usize * CHANNELS;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let edge_colors = load_u8_s32_fast::<CHANNELS>(s_ptr);
                store = _mm_add_epi32(store, edge_colors);
            }
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            unsafe {
                let previous_x = std::cmp::max(x as i64 - half_kernel as i64, 0) as usize;
                let previous = previous_x * CHANNELS;
                let s_ptr = src.as_ptr().add(y_src_shift + previous);
                let edge_colors = load_u8_s32_fast::<CHANNELS>(s_ptr);
                store = _mm_sub_epi32(store, edge_colors);
            }

            // add next
            unsafe {
                let next_x = std::cmp::min(x + half_kernel, width - 1) as usize;

                let next = next_x * CHANNELS;

                let s_ptr = src.as_ptr().add(y_src_shift + next);
                let edge_colors = load_u8_s32_fast::<CHANNELS>(s_ptr);
                store = _mm_add_epi32(store, edge_colors);
            }

            let px = x as usize * CHANNELS;

            unsafe {
                let mut r0;

                if FMA {
                    r0 = _mm_mul_round_ps(_mm_cvtepi32_ps(store), v_weight);
                } else {
                    r0 = _mm_mul_ps(_mm_cvtepi32_ps(store), v_weight);

                    const RND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

                    r0 = _mm_round_ps::<RND>(r0);
                }

                let bytes_offset = y_dst_shift + px;
                let ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u8;
                store_u8_u32::<CHANNELS>(ptr, _mm_cvtps_epi32(r0));
            }
        }
    }
}

pub(crate) fn box_blur_vertical_pass_sse<T, const CHANNELS: usize>(
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

        box_blur_vertical_pass_sse_impl::<CHANNELS>(
            src, src_stride, unsafe_dst, dst_stride, w, height, radius, start_x, end_x,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn box_blur_vertical_pass_sse_impl<const CHANNELS: usize>(
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
    let v_edge_count = unsafe { _mm_set1_epi32(edge_count as i32) };

    let v_weight = unsafe { _mm_set1_ps(1f32 / (radius * 2) as f32) };

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
            let edge = _mm_loadu_si128(s_ptr as *const _);
            let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());
            let hi0 = _mm_unpackhi_epi8(edge, _mm_setzero_si128());

            let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
            let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());
            let i16_l1 = _mm_unpacklo_epi16(hi0, _mm_setzero_si128());
            let i16_h1 = _mm_unpackhi_epi16(hi0, _mm_setzero_si128());

            store_0 = _mm_madd_epi16(i16_l0, v_edge_count);
            store_1 = _mm_madd_epi16(i16_h0, v_edge_count);

            store_2 = _mm_madd_epi16(i16_l1, v_edge_count);
            store_3 = _mm_madd_epi16(i16_h1, v_edge_count);
        }

        unsafe {
            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge = _mm_loadu_si128(s_ptr as *const _);
                let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());
                let hi0 = _mm_unpackhi_epi8(edge, _mm_setzero_si128());

                let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());
                let i16_l1 = _mm_unpacklo_epi16(hi0, _mm_setzero_si128());
                let i16_h1 = _mm_unpackhi_epi16(hi0, _mm_setzero_si128());

                store_0 = _mm_add_epi32(store_0, i16_l0);
                store_1 = _mm_add_epi32(store_1, i16_h0);

                store_2 = _mm_add_epi32(store_2, i16_l1);
                store_3 = _mm_add_epi32(store_3, i16_h1);
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
                    let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());
                    let hi0 = _mm_unpackhi_epi8(edge, _mm_setzero_si128());

                    let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());
                    let i16_l1 = _mm_unpacklo_epi16(hi0, _mm_setzero_si128());
                    let i16_h1 = _mm_unpackhi_epi16(hi0, _mm_setzero_si128());

                    store_0 = _mm_sub_epi32(store_0, i16_l0);
                    store_1 = _mm_sub_epi32(store_1, i16_h0);

                    store_2 = _mm_sub_epi32(store_2, i16_l1);
                    store_3 = _mm_sub_epi32(store_3, i16_h1);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge = _mm_loadu_si128(s_ptr as *const _);
                    let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());
                    let hi0 = _mm_unpackhi_epi8(edge, _mm_setzero_si128());

                    let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());
                    let i16_l1 = _mm_unpacklo_epi16(hi0, _mm_setzero_si128());
                    let i16_h1 = _mm_unpackhi_epi16(hi0, _mm_setzero_si128());

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

                let scale_store_0_ps = _mm_mul_ps(store0_ps, v_weight);
                let scale_store_1_ps = _mm_mul_ps(store1_ps, v_weight);
                let scale_store_2_ps = _mm_mul_ps(store2_ps, v_weight);
                let scale_store_3_ps = _mm_mul_ps(store3_ps, v_weight);

                const RND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

                let r0 = _mm_round_ps::<RND>(scale_store_0_ps);
                let r1 = _mm_round_ps::<RND>(scale_store_1_ps);
                let r2 = _mm_round_ps::<RND>(scale_store_2_ps);
                let r3 = _mm_round_ps::<RND>(scale_store_3_ps);

                let scale_store_0 = _mm_cvtps_epi32(r0);
                let scale_store_1 = _mm_cvtps_epi32(r1);
                let scale_store_2 = _mm_cvtps_epi32(r2);
                let scale_store_3 = _mm_cvtps_epi32(r3);

                let offset = y_dst_shift + px;
                let ptr = unsafe_dst.slice.get_unchecked(offset).get();

                let set0 = _mm_packus_epi32(scale_store_0, scale_store_1);
                let set1 = _mm_packus_epi32(scale_store_2, scale_store_3);

                let full_set = _mm_packus_epi16(set0, set1);
                _mm_storeu_si128(ptr as *mut _, full_set);
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
            let edge = _mm_loadu_si64(s_ptr as *const _);
            let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

            let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
            let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

            store_0 = _mm_madd_epi16(i16_l0, v_edge_count);
            store_1 = _mm_madd_epi16(i16_h0, v_edge_count);
        }

        unsafe {
            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge = _mm_loadu_si64(s_ptr as *const _);
                let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

                let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

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
                    let edge = _mm_loadu_si64(s_ptr as *const _);
                    let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

                    let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

                    store_0 = _mm_sub_epi32(store_0, i16_l0);
                    store_1 = _mm_sub_epi32(store_1, i16_h0);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge = _mm_loadu_si64(s_ptr as *const _);
                    let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

                    let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

                    store_0 = _mm_add_epi32(store_0, i16_l0);
                    store_1 = _mm_add_epi32(store_1, i16_h0);
                }

                let px = cx as usize;

                let store0_ps = _mm_cvtepi32_ps(store_0);
                let store1_ps = _mm_cvtepi32_ps(store_1);

                let scale_store_0_ps = _mm_mul_ps(store0_ps, v_weight);
                let scale_store_1_ps = _mm_mul_ps(store1_ps, v_weight);

                const RND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

                let scale_store_0_rnd = _mm_round_ps::<RND>(scale_store_0_ps);
                let scale_store_1_rnd = _mm_round_ps::<RND>(scale_store_1_ps);

                let scale_store_0 = _mm_cvtps_epi32(scale_store_0_rnd);
                let scale_store_1 = _mm_cvtps_epi32(scale_store_1_rnd);

                let offset = y_dst_shift + px;
                let ptr = unsafe_dst.slice.get_unchecked(offset).get();

                let set0 = _mm_packus_epi32(scale_store_0, scale_store_1);

                let full_set = _mm_packus_epi16(set0, _mm_setzero_si128());
                _mm_storeu_si64(ptr as *mut _, full_set);
            }
        }

        cx += 8;
    }

    const TAIL_CN: usize = 1;

    for x in cx..end_x {
        let px = x as usize;

        let mut store;
        {
            let s_ptr = unsafe { src.as_ptr().add(px) };
            let edge_colors = unsafe { load_u8_s32_fast::<TAIL_CN>(s_ptr) };
            store = unsafe { _mm_madd_epi16(edge_colors, v_edge_count) };
        }

        for y in 1..std::cmp::min(half_kernel, height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
            let edge_colors = unsafe { load_u8_s32_fast::<TAIL_CN>(s_ptr) };
            store = unsafe { _mm_add_epi32(store, edge_colors) };
        }

        for y in 0..height {
            // preload edge pixels
            let next = std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
            let previous =
                std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
            let y_dst_shift = dst_stride as usize * y as usize;

            // subtract previous
            {
                let s_ptr = unsafe { src.as_ptr().add(previous + px) };
                let edge_colors = unsafe { load_u8_s32_fast::<TAIL_CN>(s_ptr) };
                store = unsafe { _mm_sub_epi32(store, edge_colors) };
            }

            // add next
            {
                let s_ptr = unsafe { src.as_ptr().add(next + px) };
                let edge_colors = unsafe { load_u8_s32_fast::<TAIL_CN>(s_ptr) };
                store = unsafe { _mm_add_epi32(store, edge_colors) };
            }

            let px = x as usize;

            let scale_store = unsafe { _mm_mul_ps_epi32(store, v_weight) };

            unsafe {
                let ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8;
                store_u8_u32::<TAIL_CN>(ptr, scale_store);
            }
        }
    }
}
