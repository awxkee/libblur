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

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub(crate) mod fast_gaussian_f32_impl {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vgetq_lane_f32, vld1q_f32, vmulq_f32, vmulq_n_f32,
        vst1q_f32, vsubq_f32,
    };

    use crate::neon_utils::neon_utils::load_f32;
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    pub(crate) fn fast_gaussian_vertical_pass_f32(
        bytes: &UnsafeSlice<f32>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        channels: FastBlurChannels,
    ) {
        let mut buffer: [[f32; 4]; 1024] = [[0f32; 4]; 1024];

        let safe_pixel_count_x = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 2,
        };

        let height_wide = height as i64;

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32));
        let f_weight = unsafe { vdupq_n_f32(weight) };
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for x in start..std::cmp::min(width, end) {
            let mut diffs: float32x4_t = unsafe { vdupq_n_f32(0f32) };
            let mut summs: float32x4_t = unsafe { vdupq_n_f32(0f32) };

            let start_y = 0 - 2 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;

                if y >= 0 {
                    let current_px = ((std::cmp::max(x, 0)) * channels_count) as usize;

                    let prepared_px = unsafe { vmulq_f32(summs, f_weight) };

                    let new_r = unsafe { vgetq_lane_f32::<0>(prepared_px) };
                    let new_g = unsafe { vgetq_lane_f32::<1>(prepared_px) };
                    let new_b = unsafe { vgetq_lane_f32::<2>(prepared_px) };

                    unsafe {
                        bytes.write(current_y + current_px, new_r);
                        bytes.write(current_y + current_px + 1, new_g);
                        bytes.write(current_y + current_px + 2, new_b);
                    }

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let d_buf_ptr = buffer[d_arr_index].as_mut_ptr();
                    let mut d_stored = unsafe { vld1q_f32(d_buf_ptr) };
                    d_stored = unsafe { vmulq_n_f32(d_stored, 2f32) };

                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let a_stored = unsafe { vld1q_f32(buf_ptr) };

                    diffs = unsafe { vaddq_f32(diffs, vsubq_f32(a_stored, d_stored)) };
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let mut stored = unsafe { vld1q_f32(buf_ptr) };
                    stored = unsafe { vmulq_n_f32(stored, 2f32) };
                    diffs = unsafe { vsubq_f32(diffs, stored) };
                }

                let next_row_y = (std::cmp::min(std::cmp::max(y + radius_64, 0), height_wide - 1)
                    as usize)
                    * (stride as usize);
                let next_row_x = (x * channels_count) as usize;

                let s_ptr =
                    unsafe { bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut f32 };
                let pixel_color = load_f32(
                    s_ptr,
                    x as i64 + safe_pixel_count_x < width as i64,
                    channels_count as usize,
                );

                let arr_index = ((y + radius_64) & 1023) as usize;
                let buf_ptr = buffer[arr_index].as_mut_ptr();

                diffs = unsafe { vaddq_f32(diffs, pixel_color) };
                summs = unsafe { vaddq_f32(summs, diffs) };
                unsafe {
                    vst1q_f32(buf_ptr, pixel_color);
                }
            }
        }
    }

    pub(crate) fn fast_gaussian_horizontal_pass_f32(
        bytes: &UnsafeSlice<f32>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        channels: FastBlurChannels,
    ) {
        let mut buffer: [[f32; 4]; 1024] = [[0f32; 4]; 1024];

        let safe_pixel_count_x = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 2,
        };

        let radius_64 = radius as i64;
        let width_wide = width as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32));
        let f_weight = unsafe { vdupq_n_f32(weight) };
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for y in start..std::cmp::min(height, end) {
            let mut diffs: float32x4_t = unsafe { vdupq_n_f32(0f32) };
            let mut summs: float32x4_t = unsafe { vdupq_n_f32(0f32) };

            let current_y = ((y as i64) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = ((std::cmp::max(x, 0) as u32) * channels_count) as usize;

                    let prepared_px = unsafe { vmulq_f32(summs, f_weight) };

                    let new_r = unsafe { vgetq_lane_f32::<0>(prepared_px) };
                    let new_g = unsafe { vgetq_lane_f32::<1>(prepared_px) };
                    let new_b = unsafe { vgetq_lane_f32::<2>(prepared_px) };

                    unsafe {
                        bytes.write(current_y + current_px, new_r);
                        bytes.write(current_y + current_px + 1, new_g);
                        bytes.write(current_y + current_px + 2, new_b);
                    }

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let d_buf_ptr = buffer[d_arr_index].as_mut_ptr();
                    let mut d_stored = unsafe { vld1q_f32(d_buf_ptr) };
                    d_stored = unsafe { vmulq_n_f32(d_stored, 2f32) };

                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let a_stored = unsafe { vld1q_f32(buf_ptr) };

                    diffs = unsafe { vaddq_f32(diffs, vsubq_f32(a_stored, d_stored)) };
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let mut stored = unsafe { vld1q_f32(buf_ptr) };
                    stored = unsafe { vmulq_n_f32(stored, 2f32) };
                    diffs = unsafe { vsubq_f32(diffs, stored) };
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x = std::cmp::min(std::cmp::max(x + radius_64, 0), width_wide - 1) as u32;
                let next_row_px = (next_row_x * channels_count) as usize;

                let s_ptr =
                    unsafe { bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut f32 };
                let pixel_color = load_f32(
                    s_ptr,
                    (next_row_x as i64) + (safe_pixel_count_x as i64) < width as i64,
                    channels_count as usize,
                );

                let arr_index = ((x + radius_64) & 1023) as usize;
                let buf_ptr = buffer[arr_index].as_mut_ptr();

                diffs = unsafe { vaddq_f32(diffs, pixel_color) };
                summs = unsafe { vaddq_f32(summs, diffs) };
                unsafe {
                    vst1q_f32(buf_ptr, pixel_color);
                }
            }
        }
    }
}

#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
#[cfg(not(target_feature = "neon"))]
pub(crate) mod fast_gaussian_f32_impl {
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    pub(crate) fn fast_gaussian_vertical_pass_f32(
        bytes: &UnsafeSlice<f32>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        channels: FastBlurChannels,
    ) {
        let mut buffer_r: [f32; 1024] = [0f32; 1024];
        let mut buffer_g: [f32; 1024] = [0f32; 1024];
        let mut buffer_b: [f32; 1024] = [0f32; 1024];
        let radius_64 = radius as i64;
        let height_wide = height as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32));
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for x in start..std::cmp::min(width, end) {
            let mut dif_r: f32 = 0f32;
            let mut sum_r: f32 = 0f32;
            let mut dif_g: f32 = 0f32;
            let mut sum_g: f32 = 0f32;
            let mut dif_b: f32 = 0f32;
            let mut sum_b: f32 = 0f32;

            let current_px = (x * channels_count) as usize;

            let start_y = 0 - 2 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;
                if y >= 0 {
                    let new_r = sum_r * weight;
                    let new_g = sum_g * weight;
                    let new_b = sum_b * weight;

                    unsafe {
                        bytes.write(current_y + current_px, new_r);
                        bytes.write(current_y + current_px + 1, new_g);
                        bytes.write(current_y + current_px + 2, new_b);
                    }

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;
                    dif_r += buffer_r[arr_index] - 2f32 * buffer_r[d_arr_index];
                    dif_g += buffer_g[arr_index] - 2f32 * buffer_g[d_arr_index];
                    dif_b += buffer_b[arr_index] - 2f32 * buffer_b[d_arr_index];
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    dif_r -= 2f32 * buffer_r[arr_index];
                    dif_g -= 2f32 * buffer_g[arr_index];
                    dif_b -= 2f32 * buffer_b[arr_index];
                }

                let next_row_y = (std::cmp::min(std::cmp::max(y + radius_64, 0), height_wide - 1)
                    as usize)
                    * (stride as usize);
                let next_row_x = (x * channels_count) as usize;

                let px_idx = next_row_y + next_row_x;

                let rf32 = bytes[px_idx];
                let gf32 = bytes[px_idx + 1];
                let bf32 = bytes[px_idx + 2];

                let arr_index = ((y + radius_64) & 1023) as usize;

                dif_r += rf32;
                sum_r += dif_r;
                buffer_r[arr_index] = rf32;

                dif_g += gf32;
                sum_g += dif_g;
                buffer_g[arr_index] = gf32;

                dif_b += bf32;
                sum_b += dif_b;
                buffer_b[arr_index] = bf32;
            }
        }
    }

    pub(crate) fn fast_gaussian_horizontal_pass_f32(
        bytes: &UnsafeSlice<f32>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        channels: FastBlurChannels,
    ) {
        let mut buffer_r: [f32; 1024] = [0f32; 1024];
        let mut buffer_g: [f32; 1024] = [0f32; 1024];
        let mut buffer_b: [f32; 1024] = [0f32; 1024];
        let radius_64 = radius as i64;
        let width_wide = width as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32));
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for y in start..std::cmp::min(height, end) {
            let mut dif_r: f32 = 0f32;
            let mut sum_r: f32 = 0f32;
            let mut dif_g: f32 = 0f32;
            let mut sum_g: f32 = 0f32;
            let mut dif_b: f32 = 0f32;
            let mut sum_b: f32 = 0f32;

            let current_y = ((y as i64) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = ((std::cmp::max(x, 0) as u32) * channels_count) as usize;
                    let new_r = sum_r * weight;
                    let new_g = sum_g * weight;
                    let new_b = sum_b * weight;

                    unsafe {
                        bytes.write(current_y + current_px, new_r);
                        bytes.write(current_y + current_px + 1, new_g);
                        bytes.write(current_y + current_px + 2, new_b);
                    }

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;
                    dif_r += buffer_r[arr_index] - 2f32 * buffer_r[d_arr_index];
                    dif_g += buffer_g[arr_index] - 2f32 * buffer_g[d_arr_index];
                    dif_b += buffer_b[arr_index] - 2f32 * buffer_b[d_arr_index];
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    dif_r -= 2f32 * buffer_r[arr_index];
                    dif_g -= 2f32 * buffer_g[arr_index];
                    dif_b -= 2f32 * buffer_b[arr_index];
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x = ((std::cmp::min(std::cmp::max(x + radius_64, 0), width_wide - 1)
                    as u32)
                    * channels_count) as usize;

                let rf32 = bytes[next_row_y + next_row_x];
                let gf32 = bytes[next_row_y + next_row_x + 1];
                let bf32 = bytes[next_row_y + next_row_x + 2];

                let arr_index = ((x + radius_64) & 1023) as usize;

                dif_r += rf32;
                sum_r += dif_r;
                buffer_r[arr_index] = rf32;

                dif_g += gf32;
                sum_g += dif_g;
                buffer_g[arr_index] = gf32;

                dif_b += bf32;
                sum_b += dif_b;
                buffer_b[arr_index] = bf32;
            }
        }
    }
}

pub(crate) mod fast_gaussian_f32 {
    use crate::fast_gaussian_f32::fast_gaussian_f32_impl;
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    pub(crate) fn fast_gaussian_impl_f32(
        bytes: &mut [f32],
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        channels: FastBlurChannels,
    ) {
        let unsafe_image = UnsafeSlice::new(bytes);
        let thread_count = std::cmp::max(std::cmp::min(width * height / (256 * 256), 12), 1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count as usize)
            .build()
            .unwrap();
        pool.scope(|scope| {
            let segment_size = width / thread_count;

            for i in 0..thread_count {
                let start_x = i * segment_size;
                let mut end_x = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_x = width;
                }
                scope.spawn(move |_| {
                    fast_gaussian_f32_impl::fast_gaussian_vertical_pass_f32(
                        &unsafe_image,
                        stride,
                        width,
                        height,
                        radius,
                        start_x,
                        end_x,
                        channels,
                    );
                });
            }
        });
        pool.scope(|scope| {
            let segment_size = height / thread_count;

            for i in 0..thread_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_y = height;
                }

                scope.spawn(move |_| {
                    fast_gaussian_f32_impl::fast_gaussian_horizontal_pass_f32(
                        &unsafe_image,
                        stride,
                        width,
                        height,
                        radius,
                        start_y,
                        end_y,
                        channels,
                    );
                });
            }
        });
    }
}
