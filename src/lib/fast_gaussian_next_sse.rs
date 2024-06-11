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

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
pub mod sse_support {
    use crate::sse_utils::sse_utils::load_u8_s32_fast;
    use crate::unsafe_slice::UnsafeSlice;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    pub(crate) fn fast_gaussian_next_vertical_pass_sse_u8<const CHANNELS_COUNT: usize>(
        bytes: &UnsafeSlice<u8>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
    ) {
        let mut buffer: [[i32; 4]; 1024] = [[0; 4]; 1024];

        let height_wide = height as i64;

        let threes = unsafe { _mm_set1_epi32(3) };

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let f_weight = unsafe { _mm_set1_ps(weight) };
        for x in start..std::cmp::min(width, end) {
            let mut diffs = unsafe { _mm_set1_epi32(0) };
            let mut ders = unsafe { _mm_set1_epi32(0) };
            let mut summs = unsafe { _mm_set1_epi32(0) };

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;

                if y >= 0 {
                    let current_px = ((std::cmp::max(x, 0)) * CHANNELS_COUNT as u32) as usize;
                    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                    let prepared_px_s32 = unsafe {
                        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
                            _mm_cvtepi32_ps(summs),
                            f_weight,
                        )))
                    };
                    let prepared_u16 =
                        unsafe { _mm_packus_epi32(prepared_px_s32, prepared_px_s32) };
                    let prepared_u8 = unsafe { _mm_packus_epi16(prepared_u16, prepared_u16) };

                    let pixel = unsafe { _mm_extract_epi32::<0>(prepared_u8) };

                    let bytes_offset = current_y + current_px;

                    if CHANNELS_COUNT == 4 {
                        unsafe {
                            let dst_ptr =
                                (bytes.slice.as_ptr() as *mut u8).add(bytes_offset) as *mut i32;
                            *dst_ptr = pixel;
                        }
                    } else {
                        let bits = pixel.to_le_bytes();

                        unsafe {
                            bytes.write(bytes_offset, bits[0]);
                            bytes.write(bytes_offset + 1, bits[1]);
                            bytes.write(bytes_offset + 2, bits[2]);
                        }
                    }

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let buf_ptr = unsafe { buffer.get_unchecked_mut(d_arr_index).as_mut_ptr() };
                    let stored = unsafe { _mm_loadu_si128(buf_ptr as *const __m128i) };

                    let buf_ptr_1 = buffer[d_arr_index_1].as_mut_ptr();
                    let stored_1 = unsafe { _mm_loadu_si128(buf_ptr_1 as *const __m128i) };

                    let buf_ptr_2 = buffer[d_arr_index_2].as_mut_ptr();
                    let stored_2 = unsafe { _mm_loadu_si128(buf_ptr_2 as *const __m128i) };

                    let new_diff = unsafe {
                        _mm_sub_epi32(
                            _mm_mullo_epi32(_mm_sub_epi32(stored, stored_1), threes),
                            stored_2,
                        )
                    };
                    diffs = unsafe { _mm_add_epi32(diffs, new_diff) };
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = unsafe { buffer.get_unchecked_mut(arr_index).as_mut_ptr() };
                    let stored = unsafe { _mm_loadu_si128(buf_ptr as *const __m128i) };

                    let buf_ptr_1 = unsafe { buffer.get_unchecked_mut(arr_index_1).as_mut_ptr() };
                    let stored_1 = unsafe { _mm_loadu_si128(buf_ptr_1 as *const __m128i) };

                    let new_diff =
                        unsafe { _mm_mullo_epi32(_mm_sub_epi32(stored, stored_1), threes) };

                    diffs = unsafe { _mm_add_epi32(diffs, new_diff) };
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = unsafe { buffer.get_unchecked_mut(arr_index).as_mut_ptr() };
                    let stored = unsafe { _mm_loadu_si128(buf_ptr as *const __m128i) };
                    diffs = unsafe { _mm_sub_epi32(diffs, _mm_mullo_epi32(stored, threes)) };
                }

                let next_row_y = (std::cmp::min(
                    std::cmp::max(y + ((3 * radius_64) >> 1), 0),
                    height_wide - 1,
                ) as usize)
                    * (stride as usize);
                let next_row_x = (x * CHANNELS_COUNT as u32) as usize;

                let s_ptr = unsafe { bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut u8 };

                let pixel_color = unsafe { load_u8_s32_fast::<CHANNELS_COUNT>(s_ptr) };

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                let buf_ptr = unsafe { buffer.get_unchecked_mut(arr_index).as_mut_ptr() };

                diffs = unsafe { _mm_add_epi32(diffs, pixel_color) };
                ders = unsafe { _mm_add_epi32(ders, diffs) };
                summs = unsafe { _mm_add_epi32(summs, ders) };
                unsafe {
                    _mm_storeu_si128(buf_ptr as *mut __m128i, pixel_color);
                }
            }
        }
    }

    pub(crate) fn fast_gaussian_next_horizontal_pass_sse_u8<const CHANNELS_COUNT: usize>(
        bytes: &UnsafeSlice<u8>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
    ) {
        let mut buffer: [[i32; 4]; 1024] = [[0; 4]; 1024];

        let width_wide = width as i64;

        let threes = unsafe { _mm_set1_epi32(3) };

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let f_weight = unsafe { _mm_set1_ps(weight) };
        for y in start..std::cmp::min(height, end) {
            let mut diffs = unsafe { _mm_set1_epi32(0) };
            let mut ders = unsafe { _mm_set1_epi32(0) };
            let mut summs = unsafe { _mm_set1_epi32(0) };

            let current_y = ((y as i64) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * CHANNELS_COUNT;

                    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                    let prepared_px_s32 = unsafe {
                        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
                            _mm_cvtepi32_ps(summs),
                            f_weight,
                        )))
                    };
                    let prepared_u16 =
                        unsafe { _mm_packus_epi32(prepared_px_s32, prepared_px_s32) };
                    let prepared_u8 = unsafe { _mm_packus_epi16(prepared_u16, prepared_u16) };

                    let pixel = unsafe { _mm_extract_epi32::<0>(prepared_u8) };

                    let bytes_offset = current_y + current_px;

                    if CHANNELS_COUNT == 4 {
                        unsafe {
                            let dst_ptr =
                                (bytes.slice.as_ptr() as *mut u8).add(bytes_offset) as *mut i32;
                            *dst_ptr = pixel;
                        }
                    } else {
                        let bits = pixel.to_le_bytes();

                        unsafe {
                            bytes.write(bytes_offset, bits[0]);
                            bytes.write(bytes_offset + 1, bits[1]);
                            bytes.write(bytes_offset + 2, bits[2]);
                        }
                    }

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let buf_ptr = unsafe { buffer.get_unchecked_mut(d_arr_index).as_mut_ptr() };
                    let stored = unsafe { _mm_loadu_si128(buf_ptr as *const __m128i) };

                    let buf_ptr_1 = unsafe { buffer.get_unchecked_mut(d_arr_index_1).as_mut_ptr() };
                    let stored_1 = unsafe { _mm_loadu_si128(buf_ptr_1 as *const __m128i) };

                    let buf_ptr_2 = unsafe { buffer.get_unchecked_mut(d_arr_index_2).as_mut_ptr() };
                    let stored_2 = unsafe { _mm_loadu_si128(buf_ptr_2 as *const __m128i) };

                    let new_diff = unsafe {
                        _mm_sub_epi32(
                            _mm_mullo_epi32(_mm_sub_epi32(stored, stored_1), threes),
                            stored_2,
                        )
                    };
                    diffs = unsafe { _mm_add_epi32(diffs, new_diff) };
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let stored = unsafe { _mm_loadu_si128(buf_ptr as *const __m128i) };

                    let buf_ptr_1 = buffer[arr_index_1].as_mut_ptr();
                    let stored_1 = unsafe { _mm_loadu_si128(buf_ptr_1 as *const __m128i) };

                    let new_diff =
                        unsafe { _mm_mullo_epi32(_mm_sub_epi32(stored, stored_1), threes) };

                    diffs = unsafe { _mm_add_epi32(diffs, new_diff) };
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let stored = unsafe { _mm_loadu_si128(buf_ptr as *const __m128i) };
                    diffs = unsafe { _mm_sub_epi32(diffs, _mm_mullo_epi32(stored, threes)) };
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x =
                    std::cmp::min(std::cmp::max(x + 3 * radius_64 / 2, 0), width_wide - 1) as u32;
                let next_row_px = next_row_x as usize * CHANNELS_COUNT;

                let s_ptr =
                    unsafe { bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut u8 };

                let pixel_color = unsafe { load_u8_s32_fast::<CHANNELS_COUNT>(s_ptr) };

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr = unsafe { buffer.get_unchecked_mut(arr_index).as_mut_ptr() };

                diffs = unsafe { _mm_add_epi32(diffs, pixel_color) };
                ders = unsafe { _mm_add_epi32(ders, diffs) };
                summs = unsafe { _mm_add_epi32(summs, ders) };
                unsafe {
                    _mm_storeu_si128(buf_ptr as *mut __m128i, pixel_color);
                }
            }
        }
    }
}

#[cfg(not(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
)))]
pub mod sse_support {
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    #[allow(dead_code)]
    pub(crate) fn fast_gaussian_next_vertical_pass_sse_u8(
        _bytes: &UnsafeSlice<u8>,
        _stride: u32,
        _width: u32,
        _height: u32,
        _radius: u32,
        _start: u32,
        _end: u32,
        _channels: FastBlurChannels,
    ) {
    }

    #[allow(dead_code)]
    pub(crate) fn fast_gaussian_next_horizontal_pass_sse_u8(
        _bytes: &UnsafeSlice<u8>,
        _stride: u32,
        _width: u32,
        _height: u32,
        _radius: u32,
        _start: u32,
        _end: u32,
        _channels: FastBlurChannels,
    ) {
    }
}
