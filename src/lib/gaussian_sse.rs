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
    use crate::sse_utils::sse_utils::{_mm_prefer_fma_ps, load_u8_f32_fast};

    use crate::unsafe_slice::UnsafeSlice;

    pub fn gaussian_blur_horizontal_pass_impl_sse<const CHANNEL_CONFIGURATION: usize>(
        src: &[u8],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        kernel_size: usize,
        kernel: &[f32],
        start_y: u32,
        end_y: u32,
    ) {
        let half_kernel = (kernel_size / 2) as i32;

        let shuffle_rgb = unsafe {
            _mm_setr_epi8(0, 1, 2, -1,
                          3, 4, 5, -1,
                          6, 7, 8, -1,
                          9, 10, 11, -1)
        };

        let zeros_si = unsafe { _mm_setzero_si128() };

        for y in start_y..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store = unsafe { _mm_set1_ps(0f32) };

                let mut r = -half_kernel;

                unsafe {
                    while r + 4 <= half_kernel && x as i64 + r as i64 + 6 < width as i64 {
                        let px =
                            std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                                as usize
                                * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let mut pixel_colors = _mm_loadu_si128(s_ptr as *const __m128i);
                        if CHANNEL_CONFIGURATION == 3 {
                            pixel_colors = _mm_shuffle_epi8(pixel_colors, shuffle_rgb);
                        }
                        let mut pixel_colors_u16 = _mm_unpacklo_epi8(pixel_colors, zeros_si);
                        let mut pixel_colors_u32 = _mm_unpacklo_epi16(pixel_colors_u16, zeros_si);
                        let mut pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);
                        let mut weight = *kernel.get_unchecked((r + half_kernel) as usize);
                        let mut f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        pixel_colors_u32 = _mm_unpackhi_epi16(pixel_colors_u16, zeros_si);
                        pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);

                        weight = *kernel.get_unchecked((r + half_kernel + 1) as usize);
                        f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        pixel_colors_u16 = _mm_unpackhi_epi8(pixel_colors, zeros_si);
                        pixel_colors_u32 = _mm_unpacklo_epi16(pixel_colors_u16, zeros_si);
                        let mut pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);
                        let mut weight = *kernel.get_unchecked((r + half_kernel + 2) as usize);
                        let mut f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        pixel_colors_u32 = _mm_unpackhi_epi16(pixel_colors_u16, zeros_si);
                        pixel_colors_f32 = _mm_cvtepi32_ps(pixel_colors_u32);

                        weight = *kernel.get_unchecked((r + half_kernel + 3) as usize);
                        f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        r += 4;
                    }
                }

                unsafe {
                    while r <= half_kernel {
                        let current_x = std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                        let px = current_x * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_f32 = load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                        let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                        let f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        r += 1;
                    }
                }

                let px = x as usize * CHANNEL_CONFIGURATION;

                const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                let px_32 = unsafe { _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store)) };
                let px_16 = unsafe { _mm_packus_epi32(px_32, px_32) };
                let px_8 = unsafe { _mm_packus_epi16(px_16, px_16) };
                let pixel = unsafe { _mm_extract_epi32::<0>(px_8) };
                let pixel_bytes = pixel.to_le_bytes();

                unsafe {
                    let unsafe_offset = y_dst_shift + px;
                    unsafe_dst.write(unsafe_offset, pixel_bytes[0]);
                    unsafe_dst.write(unsafe_offset + 1, pixel_bytes[1]);
                    unsafe_dst.write(unsafe_offset + 2, pixel_bytes[2]);
                    if CHANNEL_CONFIGURATION == 4 {
                        unsafe_dst.write(unsafe_offset + 3, pixel_bytes[3]);
                    }
                }
            }
        }
    }

    pub fn gaussian_blur_vertical_pass_impl_sse<const CHANNEL_CONFIGURATION: usize>(
        src: &[u8],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: usize,
        kernel: &Vec<f32>,
        start_y: u32,
        end_y: u32,
    ) {
        let half_kernel = (kernel_size / 2) as i32;

        for y in start_y..end_y {
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store = unsafe { _mm_set1_ps(0f32) };

                let mut r = -half_kernel;

                let px = x as usize * CHANNEL_CONFIGURATION;

                unsafe {
                    while r <= half_kernel {
                        let py =
                            std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                        let y_src_shift = py as usize * src_stride as usize;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_f32 =
                            load_u8_f32_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                        let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                        let f_weight = _mm_set1_ps(weight);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                        r += 1;
                    }
                }

                const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                let px_32 = unsafe { _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store)) };
                let px_16 = unsafe { _mm_packus_epi32(px_32, px_32) };
                let px_8 = unsafe { _mm_packus_epi16(px_16, px_16) };
                let pixel = unsafe { _mm_extract_epi32::<0>(px_8) };
                let pixel_bytes = pixel.to_le_bytes();

                unsafe {
                    let unsafe_offset = y_dst_shift + px;
                    unsafe_dst.write(unsafe_offset, pixel_bytes[0]);
                    unsafe_dst.write(unsafe_offset + 1, pixel_bytes[1]);
                    unsafe_dst.write(unsafe_offset + 2, pixel_bytes[2]);
                    if CHANNEL_CONFIGURATION == 4 {
                        unsafe_dst.write(unsafe_offset + 3, pixel_bytes[3]);
                    }
                }
            }
        }
    }
}

#[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse4.1")))]
pub mod sse_support {
    use crate::unsafe_slice::UnsafeSlice;

    #[allow(dead_code)]
    pub fn gaussian_blur_vertical_pass_impl_sse(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _dst_stride: u32,
        _width: u32,
        _height: u32,
        _kernel_size: usize,
        _kernel: &Vec<f32>,
        _start_y: u32,
        _end_y: u32,
    ) {}

    #[allow(dead_code)]
    pub fn gaussian_blur_horizontal_pass_impl_sse(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _dst_stride: u32,
        _width: u32,
        _kernel_size: usize,
        _kernel: &Vec<f32>,
        _start_y: u32,
        _end_y: u32,
    ) {}
}
