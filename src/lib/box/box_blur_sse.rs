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

use crate::sse::{_mm_mul_ps_epi32, load_u8_s32_fast, store_u8_u32};
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
        box_blur_horizontal_pass_impl::<T, CHANNELS>(
            undefined_src,
            src_stride,
            undefined_unsafe_dst,
            dst_stride,
            width,
            radius,
            start_y,
            end_y,
        );
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn box_blur_horizontal_pass_impl<T, const CHANNELS: usize>(
    undefined_src: &[T],
    src_stride: u32,
    undefined_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undefined_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undefined_unsafe_dst) };

    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = unsafe { _mm_set1_epi32(edge_count as i32) };

    let v_weight = unsafe { _mm_set1_ps(1f32 / (radius * 2) as f32) };

    let half_kernel = kernel_size / 2;

    for y in start_y..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store;
        {
            let s_ptr = unsafe { src.as_ptr().add(y_src_shift) };
            let edge_colors = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
            store = unsafe { _mm_mullo_epi32(edge_colors, v_edge_count) };
        }

        for x in 1..std::cmp::min(half_kernel, width) {
            let px = x as usize * CHANNELS;
            let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
            let edge_colors = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
            store = unsafe { _mm_add_epi32(store, edge_colors) };
        }

        for x in 0..width {
            // preload edge pixels

            // subtract previous
            {
                let previous_x = std::cmp::max(x as i64 - half_kernel as i64, 0) as usize;
                let previous = previous_x * CHANNELS;
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + previous) };
                let edge_colors = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
                store = unsafe { _mm_sub_epi32(store, edge_colors) };
            }

            // add next
            {
                let next_x = std::cmp::min(x + half_kernel, width - 1) as usize;

                let next = next_x * CHANNELS;

                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + next) };
                let edge_colors = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
                store = unsafe { _mm_add_epi32(store, edge_colors) };
            }

            let px = x as usize * CHANNELS;

            if CHANNELS == 3 {
                store = _mm_insert_epi32::<3>(store, 0);
            }

            let scale_store = unsafe { _mm_mul_ps_epi32(store, v_weight) };

            let bytes_offset = y_dst_shift + px;
            unsafe {
                let ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u8;
                store_u8_u32::<CHANNELS>(ptr, scale_store);
            }
        }
    }
}

pub(crate) fn box_blur_vertical_pass_sse<T, const CHANNELS: usize>(
    undefined_src: &[T],
    src_stride: u32,
    undefined_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    _: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undefined_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undefined_unsafe_dst) };

    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = unsafe { _mm_set1_epi32(edge_count as i32) };

    let v_weight = unsafe { _mm_set1_ps(1f32 / (radius * 2) as f32) };

    let half_kernel = kernel_size / 2;

    let mut cx = start_x;

    for x in (cx..end_x.saturating_sub(2)).step_by(2) {
        let px = x as usize * CHANNELS;

        let mut store_0;
        let mut store_1;

        {
            let s_ptr = unsafe { src.as_ptr().add(px) };
            let edge_colors_0 = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
            let edge_colors_1 = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr.add(CHANNELS)) };
            store_0 = unsafe { _mm_mullo_epi32(edge_colors_0, v_edge_count) };
            store_1 = unsafe { _mm_mullo_epi32(edge_colors_1, v_edge_count) };
        }

        for y in 1..std::cmp::min(half_kernel, height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
            let edge_colors_0 = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
            let edge_colors_1 = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr.add(CHANNELS)) };
            store_0 = unsafe { _mm_add_epi32(store_0, edge_colors_0) };
            store_1 = unsafe { _mm_add_epi32(store_1, edge_colors_1) };
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
                let edge_colors_0 = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
                let edge_colors_1 = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr.add(CHANNELS)) };
                store_0 = unsafe { _mm_sub_epi32(store_0, edge_colors_0) };
                store_1 = unsafe { _mm_sub_epi32(store_1, edge_colors_1) };
            }

            // add next
            {
                let s_ptr = unsafe { src.as_ptr().add(next + px) };
                let edge_colors_0 = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
                let edge_colors_1 = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr.add(CHANNELS)) };
                store_0 = unsafe { _mm_add_epi32(store_0, edge_colors_0) };
                store_1 = unsafe { _mm_add_epi32(store_1, edge_colors_1) };
            }

            let px = x as usize * CHANNELS;

            if CHANNELS == 3 {
                store_0 = unsafe { _mm_insert_epi32::<3>(store_0, 0) };
                store_1 = unsafe { _mm_insert_epi32::<3>(store_1, 0) };
            }

            let scale_store_0 = unsafe { _mm_mul_ps_epi32(store_0, v_weight) };
            let scale_store_1 = unsafe { _mm_mul_ps_epi32(store_1, v_weight) };

            if CHANNELS == 3 {
                let offset = y_dst_shift + px;

                unsafe {
                    let ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut u8;
                    store_u8_u32::<CHANNELS>(ptr, scale_store_0);
                }

                unsafe {
                    let ptr = unsafe_dst.slice.as_ptr().add(offset + CHANNELS) as *mut u8;
                    store_u8_u32::<CHANNELS>(ptr, scale_store_1);
                }
            } else {
                let px_16 = unsafe { _mm_packus_epi32(scale_store_0, scale_store_1) };
                let px_8 = unsafe { _mm_packus_epi16(px_16, px_16) };
                let unsafe_offset = y_dst_shift + px;
                let ptr = unsafe { unsafe_dst.slice.get_unchecked(unsafe_offset).get() };
                unsafe {
                    _mm_storeu_si64(ptr, px_8);
                }
            }
        }

        cx = x;
    }

    for x in cx..end_x {
        let px = x as usize * CHANNELS;

        let mut store;
        {
            let s_ptr = unsafe { src.as_ptr().add(px) };
            let edge_colors = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
            store = unsafe { _mm_mullo_epi32(edge_colors, v_edge_count) };
        }

        for y in 1..std::cmp::min(half_kernel, height) {
            let y_src_shift = y as usize * src_stride as usize;
            let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
            let edge_colors = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
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
                let edge_colors = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
                store = unsafe { _mm_sub_epi32(store, edge_colors) };
            }

            // add next
            {
                let s_ptr = unsafe { src.as_ptr().add(next + px) };
                let edge_colors = unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
                store = unsafe { _mm_add_epi32(store, edge_colors) };
            }

            let px = x as usize * CHANNELS;

            if CHANNELS == 3 {
                store = unsafe { _mm_insert_epi32::<3>(store, 0) };
            }

            let scale_store = unsafe { _mm_mul_ps_epi32(store, v_weight) };

            unsafe {
                let ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8;
                store_u8_u32::<CHANNELS>(ptr, scale_store);
            }
        }
    }
}
