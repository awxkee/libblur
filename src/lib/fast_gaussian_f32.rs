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

use crate::unsafe_slice::UnsafeSlice;
use crate::FastBlurChannels;

pub(crate) fn fast_gaussian_vertical_pass_f32<const CHANNELS_COUNT: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
) {
    let channels: FastBlurChannels = CHANNELS_COUNT.into();
    let mut buffer_r: [f32; 1024] = [0f32; 1024];
    let mut buffer_g: [f32; 1024] = [0f32; 1024];
    let mut buffer_b: [f32; 1024] = [0f32; 1024];
    let mut buffer_a: [f32; 1024] = [0f32; 1024];
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
        let mut dif_a: f32 = 0f32;
        let mut sum_a: f32 = 0f32;

        let current_px = (x * channels_count) as usize;

        let start_y = 0 - 2 * radius as i64;
        for y in start_y..height_wide {
            let current_y = (y * (stride as i64)) as usize;
            if y >= 0 {
                let new_r = sum_r * weight;
                let new_g = sum_g * weight;
                let new_b = sum_b * weight;

                unsafe {
                    let offset = current_y + current_px;
                    bytes.write(offset, new_r);
                    bytes.write(offset + 1, new_g);
                    bytes.write(offset + 2, new_b);
                    if CHANNELS_COUNT == 4 {
                        let new_a = sum_a * weight;
                        bytes.write(offset + 3, new_a);
                    }
                }

                let arr_index = ((y - radius_64) & 1023) as usize;
                let d_arr_index = (y & 1023) as usize;
                unsafe {
                    dif_r += (*buffer_r.get_unchecked(arr_index))
                        - 2f32 * (*buffer_r.get_unchecked(d_arr_index));
                    dif_g += (*buffer_g.get_unchecked(arr_index))
                        - 2f32 * (*buffer_g.get_unchecked(d_arr_index));
                    dif_b += (*buffer_b.get_unchecked(arr_index))
                        - 2f32 * (*buffer_b.get_unchecked(d_arr_index));
                    dif_a += (*buffer_a.get_unchecked(arr_index))
                        - 2f32 * (*buffer_a.get_unchecked(d_arr_index));
                }
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;
                unsafe {
                    dif_r -= 2f32 * (*buffer_r.get_unchecked(arr_index));
                    dif_g -= 2f32 * (*buffer_g.get_unchecked(arr_index));
                    dif_b -= 2f32 * (*buffer_b.get_unchecked(arr_index));
                    dif_a -= 2f32 * (*buffer_a.get_unchecked(arr_index));
                }
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
            unsafe {
                *buffer_r.get_unchecked_mut(arr_index) = rf32;
            }

            dif_g += gf32;
            sum_g += dif_g;
            unsafe {
                *buffer_g.get_unchecked_mut(arr_index) = gf32;
            }

            dif_b += bf32;
            sum_b += dif_b;
            unsafe {
                *buffer_b.get_unchecked_mut(arr_index) = bf32;
            }

            if CHANNELS_COUNT == 4 {
                let af32 = bytes[px_idx + 3];
                dif_a += af32;
                sum_a += dif_a;
                unsafe {
                    *buffer_a.get_unchecked_mut(arr_index) = af32;
                }
            }
        }
    }
}

pub(crate) fn fast_gaussian_horizontal_pass_f32<const CHANNELS_COUNT: usize>(
    bytes: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
) {
    let channels: FastBlurChannels = CHANNELS_COUNT.into();
    let mut buffer_r: [f32; 1024] = [0f32; 1024];
    let mut buffer_g: [f32; 1024] = [0f32; 1024];
    let mut buffer_b: [f32; 1024] = [0f32; 1024];
    let mut buffer_a: [f32; 1024] = [0f32; 1024];
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
        let mut dif_a: f32 = 0f32;
        let mut sum_a: f32 = 0f32;

        let current_y = ((y as i64) * (stride as i64)) as usize;

        let start_x = 0 - 2 * radius_64;
        for x in start_x..(width as i64) {
            if x >= 0 {
                let current_px = ((std::cmp::max(x, 0) as u32) * channels_count) as usize;
                let new_r = sum_r * weight;
                let new_g = sum_g * weight;
                let new_b = sum_b * weight;

                unsafe {
                    let offset = current_y + current_px;
                    bytes.write(offset, new_r);
                    bytes.write(offset + 1, new_g);
                    bytes.write(offset + 2, new_b);
                    if CHANNELS_COUNT == 4 {
                        let new_a = sum_a * weight;
                        bytes.write(offset + 3, new_a);
                    }
                }

                let arr_index = ((x - radius_64) & 1023) as usize;
                let d_arr_index = (x & 1023) as usize;
                unsafe {
                    dif_r += (*buffer_r.get_unchecked(arr_index))
                        - 2f32 * (*buffer_r.get_unchecked(d_arr_index));
                    dif_g += (*buffer_g.get_unchecked(arr_index))
                        - 2f32 * (*buffer_g.get_unchecked(d_arr_index));
                    dif_b += (*buffer_b.get_unchecked(arr_index))
                        - 2f32 * (*buffer_b.get_unchecked(d_arr_index));
                    dif_a += (*buffer_a.get_unchecked(arr_index))
                        - 2f32 * (*buffer_a.get_unchecked(d_arr_index));
                }
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                unsafe {
                    dif_r -= 2f32 * (*buffer_r.get_unchecked(arr_index));
                    dif_g -= 2f32 * (*buffer_g.get_unchecked(arr_index));
                    dif_b -= 2f32 * (*buffer_b.get_unchecked(arr_index));
                    dif_a -= 2f32 * (*buffer_a.get_unchecked(arr_index));
                }
            }

            let next_row_y = (y as usize) * (stride as usize);
            let next_row_x = ((std::cmp::min(std::cmp::max(x + radius_64, 0), width_wide - 1)
                as u32)
                * channels_count) as usize;

            let src_offset = next_row_y + next_row_x;

            let rf32 = bytes[src_offset];
            let gf32 = bytes[src_offset + 1];
            let bf32 = bytes[src_offset + 2];

            let arr_index = ((x + radius_64) & 1023) as usize;

            dif_r += rf32;
            sum_r += dif_r;
            unsafe {
                *buffer_r.get_unchecked_mut(arr_index) = rf32;
            }

            dif_g += gf32;
            sum_g += dif_g;
            unsafe {
                *buffer_g.get_unchecked_mut(arr_index) = gf32;
            }

            dif_b += bf32;
            sum_b += dif_b;
            unsafe {
                *buffer_b.get_unchecked_mut(arr_index) = bf32;
            }

            if CHANNELS_COUNT == 4 {
                let af32 = bytes[src_offset + 3];
                dif_a += af32;
                sum_a += dif_a;
                unsafe {
                    *buffer_a.get_unchecked_mut(arr_index) = af32;
                }
            }
        }
    }
}

pub(crate) mod fast_gaussian_f32 {
    use crate::fast_gaussian_f32::*;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    use crate::neon::{
        fast_gaussian_horizontal_pass_neon_f32, fast_gaussian_vertical_pass_neon_f32,
    };
    use crate::unsafe_slice::UnsafeSlice;
    use crate::{FastBlurChannels, ThreadingPolicy};

    pub(crate) fn fast_gaussian_impl_f32<const CHANNELS_COUNT: usize>(
        bytes: &mut [f32],
        width: u32,
        height: u32,
        radius: u32,
        threading_policy: ThreadingPolicy,
    ) {
        let threads_count = threading_policy.get_threads_count(width, height) as u32;
        let channels: FastBlurChannels = CHANNELS_COUNT.into();
        let mut _dispatcher_vertical: fn(
            bytes: &UnsafeSlice<f32>,
            stride: u32,
            width: u32,
            height: u32,
            radius: u32,
            start: u32,
            end: u32,
        ) = fast_gaussian_vertical_pass_f32::<CHANNELS_COUNT>;
        let mut _dispatcher_horizontal: fn(
            bytes: &UnsafeSlice<f32>,
            stride: u32,
            width: u32,
            height: u32,
            radius: u32,
            start: u32,
            end: u32,
        ) = fast_gaussian_horizontal_pass_f32::<CHANNELS_COUNT>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_vertical = fast_gaussian_vertical_pass_neon_f32::<CHANNELS_COUNT>;
            _dispatcher_horizontal = fast_gaussian_horizontal_pass_neon_f32::<CHANNELS_COUNT>;
        }
        let stride = channels.get_channels() as u32 * width;
        let unsafe_image = UnsafeSlice::new(bytes);
        if threads_count == 1 {
            _dispatcher_vertical(&unsafe_image, stride, width, height, radius, 0, width);
            _dispatcher_horizontal(&unsafe_image, stride, width, height, radius, 0, height);
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads_count as usize)
            .build()
            .unwrap();
        pool.scope(|scope| {
            let segment_size = width / threads_count;

            for i in 0..threads_count {
                let start_x = i * segment_size;
                let mut end_x = (i + 1) * segment_size;
                if i == threads_count - 1 {
                    end_x = width;
                }
                scope.spawn(move |_| {
                    _dispatcher_vertical(
                        &unsafe_image,
                        stride,
                        width,
                        height,
                        radius,
                        start_x,
                        end_x,
                    );
                });
            }
        });
        pool.scope(|scope| {
            let segment_size = height / threads_count;

            for i in 0..threads_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == threads_count - 1 {
                    end_y = height;
                }

                scope.spawn(move |_| {
                    _dispatcher_horizontal(
                        &unsafe_image,
                        stride,
                        width,
                        height,
                        radius,
                        start_y,
                        end_y,
                    );
                });
            }
        });
    }
}
