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

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse4.1"))]
pub mod sse_support {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use crate::sse_utils::sse_utils::load_u8_s32_fast;
    use crate::unsafe_slice::UnsafeSlice;

    pub(crate) fn box_blur_horizontal_pass_sse<const CHANNELS: usize>(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        radius: u32,
        start_y: u32,
        end_y: u32,
    ) {
        let eraser_store: [i32; 4] = if CHANNELS == 3 { [1i32, 1i32, 1i32, 0i32] } else { [1i32, 1i32, 1i32, 1i32] };
        let eraser = unsafe { _mm_loadu_si128(eraser_store.as_ptr() as *const __m128i) };

        let kernel_scale = 1f32 / (radius * 2) as f32;
        let v_kernel_scale = unsafe { _mm_set1_ps(kernel_scale) };
        let kernel_size = radius * 2 + 1;
        let edge_count = (kernel_size / 2) + 1;
        let v_edge_count = unsafe { _mm_set1_epi32(edge_count as i32) };

        let half_kernel = kernel_size / 2;

        for y in start_y..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store;
            {
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift) };
                let edge_colors =
                    unsafe { load_u8_s32_fast::<CHANNELS>(s_ptr) };
                store = unsafe { _mm_mullo_epi32(edge_colors, v_edge_count) };
            }

            for x in 1..std::cmp::min(half_kernel, width) {
                let px = x as usize * CHANNELS;
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                let edge_colors = unsafe {
                    load_u8_s32_fast::<CHANNELS>(s_ptr)
                };
                store = unsafe { _mm_add_epi32(store, edge_colors) };
            }

            for x in 0..width {
                // preload edge pixels

                // subtract previous
                {
                    let previous_x = std::cmp::max(x as i64 - half_kernel as i64, 0) as usize;
                    let previous = previous_x * CHANNELS;
                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + previous) };
                    let edge_colors = unsafe {
                        load_u8_s32_fast::<CHANNELS>(s_ptr)
                    };
                    store = unsafe { _mm_sub_epi32(store, edge_colors) };
                }

                // add next
                {
                    let next_x =
                        std::cmp::min(x + half_kernel, width - 1) as usize;

                    let next = next_x * CHANNELS;

                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + next) };
                    let edge_colors = unsafe {
                        load_u8_s32_fast::<CHANNELS>(s_ptr)
                    };
                    store = unsafe { _mm_add_epi32(store, edge_colors) };
                }

                let px = x as usize * CHANNELS;

                if CHANNELS == 3 {
                    store = unsafe { _mm_mullo_epi32(store, eraser) };
                }

                let scale_store =
                    unsafe { _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(store), v_kernel_scale)) };
                let px_16 = unsafe { _mm_packus_epi32(scale_store, scale_store) };
                let px_8 = unsafe { _mm_packus_epi16(px_16, px_16) };
                let pixel = unsafe { _mm_extract_epi32::<0>(px_8) };
                let pixel_bytes = pixel.to_le_bytes();

                unsafe {
                    let unsafe_offset = y_dst_shift + px;
                    unsafe_dst.write(unsafe_offset, pixel_bytes[0]);
                    unsafe_dst.write(unsafe_offset + 1, pixel_bytes[1]);
                    unsafe_dst.write(unsafe_offset + 2, pixel_bytes[2]);
                    if CHANNELS == 4 {
                        unsafe_dst.write(unsafe_offset + 3, pixel_bytes[3]);
                    }
                }
            }
        }
    }

    pub(crate) fn box_blur_vertical_pass_sse<const CHANNELS: usize>(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        _: u32,
        height: u32,
        radius: u32,
        start_x: u32,
        end_x: u32,
    ) {
        let eraser_store: [i32; 4] = if CHANNELS == 3 { [1i32, 1i32, 1i32, 0i32] } else { [1i32, 1i32, 1i32, 1i32] };
        let eraser = unsafe { _mm_loadu_si128(eraser_store.as_ptr() as *const __m128i) };

        let kernel_scale = 1f32 / (radius * 2) as f32;
        let v_kernel_scale = unsafe { _mm_set1_ps(kernel_scale) };
        let kernel_size = radius * 2 + 1;
        let edge_count = (kernel_size / 2) + 1;
        let v_edge_count = unsafe { _mm_set1_epi32(edge_count as i32) };

        let half_kernel = kernel_size / 2;

        for x in start_x..end_x {
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
                let next =
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
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
                    store = unsafe { _mm_mullo_epi32(store, eraser) };
                }

                let scale_store =
                    unsafe { _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(store), v_kernel_scale)) };
                let px_16 = unsafe { _mm_packus_epi32(scale_store, scale_store) };
                let px_8 = unsafe { _mm_packus_epi16(px_16, px_16) };

                let pixel = unsafe { _mm_extract_epi32::<0>(px_8) };
                let pixel_bytes = pixel.to_le_bytes();

                unsafe {
                    let unsafe_offset = y_dst_shift + px;
                    unsafe_dst.write(unsafe_offset, pixel_bytes[0]);
                    unsafe_dst.write(unsafe_offset + 1, pixel_bytes[1]);
                    unsafe_dst.write(unsafe_offset + 2, pixel_bytes[2]);
                    if CHANNELS == 4 {
                        unsafe_dst.write(unsafe_offset + 3, pixel_bytes[3]);
                    }
                }
            }
        }
    }
}

#[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse4.1")))]
pub mod sse_support {
    use crate::FastBlurChannels;
    use crate::unsafe_slice::UnsafeSlice;

    #[allow(dead_code)]
    pub(crate) fn box_blur_horizontal_pass_sse(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _dst_stride: u32,
        _width: u32,
        _radius: u32,
        _box_channels: FastBlurChannels,
        _start_y: u32,
        _end_y: u32,
        _channels: FastBlurChannels,
    ) {}

    #[allow(dead_code)]
    pub(crate) fn box_blur_vertical_pass_sse(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _dst_stride: u32,
        _width: u32,
        _height: u32,
        _radius: u32,
        _start_y: u32,
        _end_y: u32,
        _channels: FastBlurChannels,
    ) {}
}
