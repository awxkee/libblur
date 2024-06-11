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

use crate::channels_configuration::FastBlurChannels;
use crate::fast_gaussian_f16::fast_gaussian_f16;
use crate::fast_gaussian_f32::fast_gaussian_f32;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::fast_gaussian_neon::neon_support;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::fast_gaussian_sse::sse_support;
use crate::mul_table::{MUL_TABLE_DOUBLE, SHR_TABLE_DOUBLE};
use crate::threading_policy::ThreadingPolicy;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::cast::FromPrimitive;

fn fast_gaussian_vertical_pass<
    T: FromPrimitive + Default + Into<i32>,
    const CHANNELS_CONFIGURATION: usize,
>(
    bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    if std::any::type_name::<T>() == "u8" {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(bytes) };
            neon_support::fast_gaussian_vertical_pass_neon_u8::<CHANNELS_CONFIGURATION>(
                slice, stride, width, height, radius, start, end,
            );
            return;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(bytes) };
            sse_support::fast_gaussian_vertical_pass_sse_u8::<CHANNELS_CONFIGURATION>(
                slice, stride, width, height, radius, start, end,
            );
            return;
        }
    }
    let mut buffer_r: [i32; 1024] = [0; 1024];
    let mut buffer_g: [i32; 1024] = [0; 1024];
    let mut buffer_b: [i32; 1024] = [0; 1024];
    let mut buffer_a: [i32; 1024] = [0; 1024];
    let radius_64 = radius as i64;
    let height_wide = height as i64;
    let mul_value = MUL_TABLE_DOUBLE[radius as usize];
    let shr_value = SHR_TABLE_DOUBLE[radius as usize];
    let initial = ((radius * radius) >> 1) as i32;
    for x in start..std::cmp::min(width, end) {
        let mut dif_r: i32 = 0;
        let mut sum_r: i32 = initial;
        let mut dif_g: i32 = 0;
        let mut sum_g: i32 = initial;
        let mut dif_b: i32 = 0;
        let mut sum_b: i32 = initial;
        let mut dif_a: i32 = 0;
        let mut sum_a: i32 = initial;

        let current_px = (x * CHANNELS_CONFIGURATION as u32) as usize;

        let start_y = 0 - 2 * radius as i64;
        for y in start_y..height_wide {
            let current_y = (y * (stride as i64)) as usize;
            if y >= 0 {
                let new_r =
                    T::from_u32((sum_r * mul_value) as u32 >> shr_value).unwrap_or_default();
                let new_g =
                    T::from_u32((sum_g * mul_value) as u32 >> shr_value).unwrap_or_default();
                let new_b =
                    T::from_u32((sum_b * mul_value) as u32 >> shr_value).unwrap_or_default();

                unsafe {
                    bytes.write(current_y + current_px, new_r);
                    bytes.write(current_y + current_px + 1, new_g);
                    bytes.write(current_y + current_px + 2, new_b);
                    if CHANNELS_CONFIGURATION == 4 {
                        let new_a = T::from_u32((sum_a * mul_value) as u32 >> shr_value)
                            .unwrap_or_default();
                        bytes.write(current_y + current_px + 3, new_a);
                    }
                }

                let arr_index = ((y - radius_64) & 1023) as usize;
                let d_arr_index = (y & 1023) as usize;
                dif_r += unsafe { *buffer_r.get_unchecked(arr_index) }
                    - 2 * unsafe { *buffer_r.get_unchecked(d_arr_index) };
                dif_g += unsafe { *buffer_g.get_unchecked(arr_index) }
                    - 2 * unsafe { *buffer_g.get_unchecked(d_arr_index) };
                dif_b += unsafe { *buffer_b.get_unchecked(arr_index) }
                    - 2 * unsafe { *buffer_b.get_unchecked(d_arr_index) };
                if CHANNELS_CONFIGURATION == 4 {
                    dif_a += unsafe { *buffer_a.get_unchecked(arr_index) }
                        - 2 * unsafe { *buffer_a.get_unchecked(d_arr_index) };
                }
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;
                dif_r -= 2 * unsafe { *buffer_r.get_unchecked(arr_index) };
                dif_g -= 2 * unsafe { *buffer_g.get_unchecked(arr_index) };
                dif_b -= 2 * unsafe { *buffer_b.get_unchecked(arr_index) };
                if CHANNELS_CONFIGURATION == 4 {
                    dif_a -= 2 * unsafe { *buffer_a.get_unchecked(arr_index) };
                }
            }

            let next_row_y = (std::cmp::min(std::cmp::max(y + radius_64, 0), height_wide - 1)
                as usize)
                * (stride as usize);
            let next_row_x = (x * CHANNELS_CONFIGURATION as u32) as usize;

            let px_idx = next_row_y + next_row_x;

            let ur8 = bytes[px_idx];
            let ug8 = bytes[px_idx + 1];
            let ub8 = bytes[px_idx + 2];

            let arr_index = ((y + radius_64) & 1023) as usize;

            dif_r += ur8.into();
            sum_r += dif_r;
            unsafe {
                *buffer_r.get_unchecked_mut(arr_index) = ur8.into();
            }

            dif_g += ug8.into();
            sum_g += dif_g;
            unsafe {
                *buffer_g.get_unchecked_mut(arr_index) = ug8.into();
            }

            dif_b += ub8.into();
            sum_b += dif_b;
            unsafe {
                *buffer_b.get_unchecked_mut(arr_index) = ub8.into();
            }

            if CHANNELS_CONFIGURATION == 4 {
                let ua8 = bytes[px_idx + 3];
                dif_a += ua8.into();
                sum_a += dif_a;
                unsafe {
                    *buffer_a.get_unchecked_mut(arr_index) = ua8.into();
                }
            }
        }
    }
}

fn fast_gaussian_horizontal_pass<
    T: FromPrimitive + Default + Into<i32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    if std::any::type_name::<T>() == "u8" {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(bytes) };
            neon_support::fast_gaussian_horizontal_pass_neon_u8::<CHANNEL_CONFIGURATION>(
                slice, stride, width, height, radius, start, end,
            );
            return;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(bytes) };
            sse_support::fast_gaussian_horizontal_pass_sse_u8::<CHANNEL_CONFIGURATION>(
                slice, stride, width, height, radius, start, end,
            );
            return;
        }
    }
    let channels: FastBlurChannels = CHANNEL_CONFIGURATION.into();
    let mut buffer_r: [i32; 1024] = [0; 1024];
    let mut buffer_g: [i32; 1024] = [0; 1024];
    let mut buffer_b: [i32; 1024] = [0; 1024];
    let mut buffer_a: [i32; 1024] = [0; 1024];
    let radius_64 = radius as i64;
    let width_wide = width as i64;
    let mul_value = MUL_TABLE_DOUBLE[radius as usize];
    let shr_value = SHR_TABLE_DOUBLE[radius as usize];
    let channels_count = match channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    };
    let initial_sum = ((radius * radius) >> 1) as i32;
    for y in start..std::cmp::min(height, end) {
        let mut dif_r: i32 = 0;
        let mut sum_r: i32 = initial_sum;
        let mut dif_g: i32 = 0;
        let mut sum_g: i32 = initial_sum;
        let mut dif_b: i32 = 0;
        let mut sum_b: i32 = initial_sum;
        let mut dif_a: i32 = 0;
        let mut sum_a: i32 = initial_sum;

        let current_y = ((y as i64) * (stride as i64)) as usize;

        let start_x = 0 - 2 * radius_64;
        for x in start_x..(width as i64) {
            if x >= 0 {
                let current_px = ((std::cmp::max(x, 0) as u32) * channels_count) as usize;
                let new_r =
                    T::from_u32((sum_r * mul_value) as u32 >> shr_value).unwrap_or_default();
                let new_g =
                    T::from_u32((sum_g * mul_value) as u32 >> shr_value).unwrap_or_default();
                let new_b =
                    T::from_u32((sum_b * mul_value) as u32 >> shr_value).unwrap_or_default();

                unsafe {
                    let offset = current_y + current_px;
                    bytes.write(offset, new_r);
                    bytes.write(offset + 1, new_g);
                    bytes.write(offset + 2, new_b);
                    if CHANNEL_CONFIGURATION == 4 {
                        let new_a = T::from_u32((sum_a * mul_value) as u32 >> shr_value)
                            .unwrap_or_default();
                        bytes.write(offset + 3, new_a);
                    }
                }

                let arr_index = ((x - radius_64) & 1023) as usize;
                let d_arr_index = (x & 1023) as usize;
                dif_r += unsafe { *buffer_r.get_unchecked(arr_index) }
                    - 2 * unsafe { *buffer_r.get_unchecked(d_arr_index) };
                dif_g += unsafe { *buffer_g.get_unchecked(arr_index) }
                    - 2 * unsafe { *buffer_g.get_unchecked(d_arr_index) };
                dif_b += unsafe { *buffer_b.get_unchecked(arr_index) }
                    - 2 * unsafe { *buffer_b.get_unchecked(d_arr_index) };
                if CHANNEL_CONFIGURATION == 4 {
                    dif_a += unsafe { *buffer_a.get_unchecked(arr_index) }
                        - 2 * unsafe { *buffer_a.get_unchecked(d_arr_index) };
                }
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                dif_r -= 2 * unsafe { *buffer_r.get_unchecked(arr_index) };
                dif_g -= 2 * unsafe { *buffer_g.get_unchecked(arr_index) };
                dif_b -= 2 * unsafe { *buffer_b.get_unchecked(arr_index) };
                if CHANNEL_CONFIGURATION == 4 {
                    dif_a -= 2 * unsafe { *buffer_a.get_unchecked(arr_index) };
                }
            }

            let next_row_y = (y as usize) * (stride as usize);
            let next_row_x = ((std::cmp::min(std::cmp::max(x + radius_64, 0), width_wide - 1)
                as u32)
                * channels_count) as usize;

            let bytes_offset = next_row_y + next_row_x;

            let ur8 = bytes[bytes_offset];
            let ug8 = bytes[bytes_offset + 1];
            let ub8 = bytes[bytes_offset + 2];

            let arr_index = ((x + radius_64) & 1023) as usize;

            dif_r += ur8.into();
            sum_r += dif_r;
            unsafe {
                *buffer_r.get_unchecked_mut(arr_index) = ur8.into();
            }

            dif_g += ug8.into();
            sum_g += dif_g;
            unsafe {
                *buffer_g.get_unchecked_mut(arr_index) = ug8.into();
            }

            dif_b += ub8.into();
            sum_b += dif_b;
            unsafe {
                *buffer_b.get_unchecked_mut(arr_index) = ub8.into();
            }

            if CHANNEL_CONFIGURATION == 4 {
                let ua8 = bytes[bytes_offset + 3];
                dif_a += ua8.into();
                sum_a += dif_a;
                unsafe {
                    *buffer_a.get_unchecked_mut(arr_index) = ua8.into();
                }
            }
        }
    }
}

fn fast_gaussian_impl<
    T: FromPrimitive + Default + Into<i32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    bytes: &mut [T],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    threading_policy: ThreadingPolicy,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let unsafe_image = UnsafeSlice::new(bytes);
    let thread_count = threading_policy.get_threads_count(width, height) as u32;
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
                fast_gaussian_vertical_pass::<T, CHANNEL_CONFIGURATION>(
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
        let segment_size = height / thread_count;

        for i in 0..thread_count {
            let start_y = i * segment_size;
            let mut end_y = (i + 1) * segment_size;
            if i == thread_count - 1 {
                end_y = height;
            }
            scope.spawn(move |_| {
                fast_gaussian_horizontal_pass::<T, CHANNEL_CONFIGURATION>(
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

/// Fast gaussian approximation.
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `radius` - Radius more than 319 is not supported. To use larger radius convert image to f32 and use function for f32
/// O(1) complexity.
#[no_mangle]
pub fn fast_gaussian(
    bytes: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    let acq_radius = std::cmp::min(radius, 319);
    match channels {
        FastBlurChannels::Channels3 => {
            fast_gaussian_impl::<u8, 3>(bytes, stride, width, height, acq_radius, threading_policy);
        }
        FastBlurChannels::Channels4 => {
            fast_gaussian_impl::<u8, 4>(bytes, stride, width, height, acq_radius, threading_policy);
        }
    }
}

/// Fast gaussian approximation.
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `radius` - Radius more than 255 is not supported. To use larger radius convert image to f32 and use function for f32
/// O(1) complexity.
#[no_mangle]
pub fn fast_gaussian_u16(
    bytes: &mut [u16],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    let acq_radius = std::cmp::min(radius, 255);
    match channels {
        FastBlurChannels::Channels3 => {
            fast_gaussian_impl::<u16, 3>(
                bytes,
                stride,
                width,
                height,
                acq_radius,
                threading_policy,
            );
        }
        FastBlurChannels::Channels4 => {
            fast_gaussian_impl::<u16, 4>(
                bytes,
                stride,
                width,
                height,
                acq_radius,
                threading_policy,
            );
        }
    }
}

/// Fast gaussian approximation.
/// O(1) complexity.
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `radius` - almost any radius is supported
#[no_mangle]
pub fn fast_gaussian_f32(
    bytes: &mut [f32],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    fast_gaussian_f32::fast_gaussian_impl_f32(bytes, stride, width, height, radius, channels);
}

/// Fast gaussian approximation.
/// O(1) complexity.
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `radius` - almost any radius is supported
#[no_mangle]
pub fn fast_gaussian_f16(
    bytes: &mut [u16],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    fast_gaussian_f16::fast_gaussian_impl_f16(bytes, stride, width, height, radius, channels);
}
