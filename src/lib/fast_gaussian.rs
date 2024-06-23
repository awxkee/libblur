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
use crate::mul_table::{MUL_TABLE_DOUBLE, SHR_TABLE_DOUBLE};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{fast_gaussian_horizontal_pass_neon_u8, fast_gaussian_vertical_pass_neon_u8};
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::{fast_gaussian_horizontal_pass_sse_u8, fast_gaussian_vertical_pass_sse_u8};
use crate::threading_policy::ThreadingPolicy;
use crate::unsafe_slice::UnsafeSlice;
use colorutils_rs::{
    linear_to_rgb, linear_to_rgba, rgb_to_linear, rgba_to_linear, TransferFunction,
};
use num_traits::cast::FromPrimitive;

const BASE_RADIUS_I64_CUTOFF: u32 = 180;

fn fast_gaussian_vertical_pass<T, J, const CHANNELS_CONFIGURATION: usize>(
    bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
) where
    T: std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + FromPrimitive
        + Default
        + Into<i32>
        + Into<J>,
    J: Copy
        + FromPrimitive
        + Default
        + Into<i64>
        + std::ops::Mul<Output = J>
        + std::ops::Sub<Output = J>
        + std::ops::Add<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + From<i32>,
    i64: From<T>,
{
    let mut buffer_r: [J; 1024] = [J::from_i32(0i32).unwrap(); 1024];
    let mut buffer_g: [J; 1024] = [J::from_i32(0i32).unwrap(); 1024];
    let mut buffer_b: [J; 1024] = [J::from_i32(0i32).unwrap(); 1024];
    let mut buffer_a: [J; 1024] = [J::from_i32(0i32).unwrap(); 1024];
    let radius_64 = radius as i64;
    let height_wide = height as i64;
    let mul_value = MUL_TABLE_DOUBLE[radius as usize] as i64;
    let shr_value = SHR_TABLE_DOUBLE[radius as usize] as u64;
    let initial = J::from_i32(((radius * radius) >> 1) as i32).unwrap();
    for x in start..std::cmp::min(width, end) {
        let mut dif_r: J = J::from_i32(0i32).unwrap();
        let mut sum_r: J = initial;
        let mut dif_g: J = J::from_i32(0i32).unwrap();
        let mut sum_g: J = initial;
        let mut dif_b: J = J::from_i32(0i32).unwrap();
        let mut sum_b: J = initial;
        let mut dif_a: J = J::from_i32(0i32).unwrap();
        let mut sum_a: J = initial;

        let current_px = (x * CHANNELS_CONFIGURATION as u32) as usize;

        let start_y = 0 - 2 * radius as i64;
        for y in start_y..height_wide {
            let current_y = (y * (stride as i64)) as usize;
            if y >= 0 {
                let sum_r_i64: i64 = sum_r.into();
                let new_r = T::from_u64((sum_r_i64 * mul_value) as u64 >> shr_value).unwrap();
                let sum_g_i64: i64 = sum_g.into();
                let new_g = T::from_u64((sum_g_i64 * mul_value) as u64 >> shr_value).unwrap();
                let sum_b_i64: i64 = sum_b.into();
                let new_b = T::from_u64((sum_b_i64 * mul_value) as u64 >> shr_value).unwrap();

                unsafe {
                    bytes.write(current_y + current_px, new_r);
                    bytes.write(current_y + current_px + 1, new_g);
                    bytes.write(current_y + current_px + 2, new_b);
                    if CHANNELS_CONFIGURATION == 4 {
                        let sum_a_i64: i64 = sum_a.into();
                        let new_a =
                            T::from_u64((sum_a_i64 * mul_value) as u64 >> shr_value).unwrap();
                        bytes.write(current_y + current_px + 3, new_a);
                    }
                }

                let arr_index = ((y - radius_64) & 1023) as usize;
                let d_arr_index = (y & 1023) as usize;
                let twos: J = 2i32.into();
                dif_r += unsafe { *buffer_r.get_unchecked(arr_index) }
                    - twos * unsafe { *buffer_r.get_unchecked(d_arr_index) };
                dif_g += unsafe { *buffer_g.get_unchecked(arr_index) }
                    - twos * unsafe { *buffer_g.get_unchecked(d_arr_index) };
                dif_b += unsafe { *buffer_b.get_unchecked(arr_index) }
                    - twos * unsafe { *buffer_b.get_unchecked(d_arr_index) };
                if CHANNELS_CONFIGURATION == 4 {
                    dif_a += unsafe { *buffer_a.get_unchecked(arr_index) }
                        - twos * unsafe { *buffer_a.get_unchecked(d_arr_index) };
                }
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;
                let twos: J = 2i32.into();
                dif_r -= twos * unsafe { *buffer_r.get_unchecked(arr_index) };
                dif_g -= twos * unsafe { *buffer_g.get_unchecked(arr_index) };
                dif_b -= twos * unsafe { *buffer_b.get_unchecked(arr_index) };
                if CHANNELS_CONFIGURATION == 4 {
                    dif_a -= twos * unsafe { *buffer_a.get_unchecked(arr_index) };
                }
            }

            let next_row_y = (std::cmp::min(std::cmp::max(y + radius_64, 0), height_wide - 1)
                as usize)
                * (stride as usize);
            let next_row_x = (x * CHANNELS_CONFIGURATION as u32) as usize;

            let px_idx = next_row_y + next_row_x;

            let ur8: J = bytes[px_idx].into();
            let ug8: J = bytes[px_idx + 1].into();
            let ub8: J = bytes[px_idx + 2].into();

            let arr_index = ((y + radius_64) & 1023) as usize;

            dif_r += ur8;
            sum_r += dif_r;
            unsafe {
                *buffer_r.get_unchecked_mut(arr_index) = ur8;
            }

            dif_g += ug8;
            sum_g += dif_g;
            unsafe {
                *buffer_g.get_unchecked_mut(arr_index) = ug8;
            }

            dif_b += ub8;
            sum_b += dif_b;
            unsafe {
                *buffer_b.get_unchecked_mut(arr_index) = ub8;
            }

            if CHANNELS_CONFIGURATION == 4 {
                let ua8: J = bytes[px_idx + 3].into();
                dif_a += ua8;
                sum_a += dif_a;
                unsafe {
                    *buffer_a.get_unchecked_mut(arr_index) = ua8;
                }
            }
        }
    }
}

fn fast_gaussian_horizontal_pass<
    T: FromPrimitive + Default + Into<i32> + Send + Sync,
    J,
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
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + Into<J>,
    J: Copy
        + FromPrimitive
        + Default
        + Into<i64>
        + std::ops::Mul<Output = J>
        + std::ops::Sub<Output = J>
        + std::ops::Add<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + TryFrom<u8>,
    i64: From<T>,
{
    let channels: FastBlurChannels = CHANNEL_CONFIGURATION.into();
    let mut buffer_r: [J; 1024] = [J::from_i32(0i32).unwrap(); 1024];
    let mut buffer_g: [J; 1024] = [J::from_i32(0i32).unwrap(); 1024];
    let mut buffer_b: [J; 1024] = [J::from_i32(0i32).unwrap(); 1024];
    let mut buffer_a: [J; 1024] = [J::from_i32(0i32).unwrap(); 1024];
    let radius_64 = radius as i64;
    let width_wide = width as i64;
    let mul_value = MUL_TABLE_DOUBLE[radius as usize] as i64;
    let shr_value = SHR_TABLE_DOUBLE[radius as usize] as i64;
    let channels_count = match channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    };
    let initial = J::from_i32(((radius * radius) >> 1) as i32).unwrap();
    for y in start..std::cmp::min(height, end) {
        let mut dif_r: J = J::from_i32(0i32).unwrap();
        let mut sum_r: J = initial;
        let mut dif_g: J = J::from_i32(0i32).unwrap();
        let mut sum_g: J = initial;
        let mut dif_b: J = J::from_i32(0i32).unwrap();
        let mut sum_b: J = initial;
        let mut dif_a: J = J::from_i32(0i32).unwrap();
        let mut sum_a: J = initial;

        let current_y = ((y as i64) * (stride as i64)) as usize;

        let start_x = 0 - 2 * radius_64;
        for x in start_x..(width as i64) {
            if x >= 0 {
                let current_px = ((std::cmp::max(x, 0) as u32) * channels_count) as usize;
                let sum_r_i64: i64 = sum_r.into();
                let new_r =
                    T::from_u64((sum_r_i64 * mul_value) as u64 >> shr_value).unwrap_or_default();
                let sum_g_i64: i64 = sum_g.into();
                let new_g =
                    T::from_u64((sum_g_i64 * mul_value) as u64 >> shr_value).unwrap_or_default();
                let sum_b_i64: i64 = sum_b.into();
                let new_b =
                    T::from_u64((sum_b_i64 * mul_value) as u64 >> shr_value).unwrap_or_default();

                unsafe {
                    let offset = current_y + current_px;
                    bytes.write(offset, new_r);
                    bytes.write(offset + 1, new_g);
                    bytes.write(offset + 2, new_b);
                    if CHANNEL_CONFIGURATION == 4 {
                        let sum_a_i64: i64 = sum_a.into();
                        let new_a = T::from_u64((sum_a_i64 * mul_value) as u64 >> shr_value)
                            .unwrap_or_default();
                        bytes.write(offset + 3, new_a);
                    }
                }

                let arr_index = ((x - radius_64) & 1023) as usize;
                let d_arr_index = (x & 1023) as usize;
                dif_r += unsafe { *buffer_r.get_unchecked(arr_index) }
                    - J::from_i32(2).unwrap() * unsafe { *buffer_r.get_unchecked(d_arr_index) };
                dif_g += unsafe { *buffer_g.get_unchecked(arr_index) }
                    - J::from_i32(2).unwrap() * unsafe { *buffer_g.get_unchecked(d_arr_index) };
                dif_b += unsafe { *buffer_b.get_unchecked(arr_index) }
                    - J::from_i32(2).unwrap() * unsafe { *buffer_b.get_unchecked(d_arr_index) };
                if CHANNEL_CONFIGURATION == 4 {
                    dif_a += unsafe { *buffer_a.get_unchecked(arr_index) }
                        - J::from_i32(2).unwrap() * unsafe { *buffer_a.get_unchecked(d_arr_index) };
                }
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                dif_r -= J::from_i32(2).unwrap() * unsafe { *buffer_r.get_unchecked(arr_index) };
                dif_g -= J::from_i32(2).unwrap() * unsafe { *buffer_g.get_unchecked(arr_index) };
                dif_b -= J::from_i32(2).unwrap() * unsafe { *buffer_b.get_unchecked(arr_index) };
                if CHANNEL_CONFIGURATION == 4 {
                    dif_a -=
                        J::from_i32(2).unwrap() * unsafe { *buffer_a.get_unchecked(arr_index) };
                }
            }

            let next_row_y = (y as usize) * (stride as usize);
            let next_row_x = ((std::cmp::min(std::cmp::max(x + radius_64, 0), width_wide - 1)
                as u32)
                * channels_count) as usize;

            let bytes_offset = next_row_y + next_row_x;

            let ur8: J = bytes[bytes_offset].into();
            let ug8: J = bytes[bytes_offset + 1].into();
            let ub8: J = bytes[bytes_offset + 2].into();

            let arr_index = ((x + radius_64) & 1023) as usize;

            dif_r += ur8;
            sum_r += dif_r;
            unsafe {
                *buffer_r.get_unchecked_mut(arr_index) = ur8;
            }

            dif_g += ug8;
            sum_g += dif_g;
            unsafe {
                *buffer_g.get_unchecked_mut(arr_index) = ug8;
            }

            dif_b += ub8;
            sum_b += dif_b;
            unsafe {
                *buffer_b.get_unchecked_mut(arr_index) = ub8;
            }

            if CHANNEL_CONFIGURATION == 4 {
                let ua8: J = bytes[bytes_offset + 3].into();
                dif_a += ua8;
                sum_a += dif_a;
                unsafe {
                    *buffer_a.get_unchecked_mut(arr_index) = ua8;
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
    i64: From<T>,
    i32: From<T>,
{
    let unsafe_image = UnsafeSlice::new(bytes);
    let thread_count = threading_policy.get_threads_count(width, height) as u32;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();
    let mut _dispatcher_vertical: fn(
        bytes: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
    ) = if BASE_RADIUS_I64_CUTOFF > radius {
        fast_gaussian_vertical_pass::<T, i32, CHANNEL_CONFIGURATION>
    } else {
        fast_gaussian_vertical_pass::<T, i64, CHANNEL_CONFIGURATION>
    };
    let mut _dispatcher_horizontal: fn(&UnsafeSlice<T>, u32, u32, u32, u32, u32, u32) =
        if BASE_RADIUS_I64_CUTOFF > radius {
            fast_gaussian_horizontal_pass::<T, i32, CHANNEL_CONFIGURATION>
        } else {
            fast_gaussian_horizontal_pass::<T, i64, CHANNEL_CONFIGURATION>
        };
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if std::any::type_name::<T>() == "u8" {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical =
                    fast_gaussian_vertical_pass_neon_u8::<T, CHANNEL_CONFIGURATION>;
                _dispatcher_horizontal =
                    fast_gaussian_horizontal_pass_neon_u8::<T, CHANNEL_CONFIGURATION>;
            }
        }
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        if std::any::type_name::<T>() == "u8" {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical =
                    fast_gaussian_vertical_pass_sse_u8::<T, CHANNEL_CONFIGURATION>;
                _dispatcher_horizontal =
                    fast_gaussian_horizontal_pass_sse_u8::<T, CHANNEL_CONFIGURATION>;
            }
        }
    }
    if thread_count == 1 {
        _dispatcher_vertical(&unsafe_image, stride, width, height, radius, 0, width);
        _dispatcher_horizontal(&unsafe_image, stride, width, height, radius, 0, height);
    } else {
        pool.scope(|scope| {
            let segment_size = width / thread_count;

            for i in 0..thread_count {
                let start_x = i * segment_size;
                let mut end_x = (i + 1) * segment_size;
                if i == thread_count - 1 {
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
            let segment_size = height / thread_count;

            for i in 0..thread_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == thread_count - 1 {
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

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for u8 image, limited to 319 radius, sometimes on the very bright images may start ringing on a very large radius.
/// Approximation based on binomial filter. Algorithm is close to stack blur with better results and a little slower speed
/// Results better than in stack blur however this a little slower.
/// This is a very fast approximation using i32 accumulator size with radius less that *BASE_RADIUS_I64_CUTOFF*,
/// after it to avoid overflowing fallback to i64 accumulator will be used with some computational slowdown with factor ~2
/// O(1) complexity.
///
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - Radius more than 319 is not supported. To use larger radius convert image to f32 and use function for f32
/// * `channels` - Count of channels of the image, only 3 and 4 is supported, alpha position, and channels order does not matter
/// * `threading_policy` - Threads usage policy
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
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

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for u16 image, limited to 319 radius, sometimes on the very bright images may start ringing on a very large radius.
/// Approximation based on binomial filter. Algorithm is close to stack blur with better results and a little slower speed
/// O(1) complexity.
///
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - Radius more than 255 is not supported. To use larger radius convert image to f32 and use function for f32
/// * `channels` - Count of channels in the image
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
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

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for f32 image. No limitations are expected.
/// Approximation based on binomial filter. Algorithm is close to stack blur with better results and a little slower speed
/// O(1) complexity.
///
/// # Arguments
///
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - almost any radius is supported
/// * `channels` - Count of channels in the image
/// * `transfer_function` - Transfer function in linear colorspace
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_f32(
    bytes: &mut [f32],
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    match channels {
        FastBlurChannels::Channels3 => {
            fast_gaussian_f32::fast_gaussian_impl_f32::<3>(
                bytes,
                width,
                height,
                radius,
                threading_policy,
            );
        }
        FastBlurChannels::Channels4 => {
            fast_gaussian_f32::fast_gaussian_impl_f32::<4>(
                bytes,
                width,
                height,
                radius,
                threading_policy,
            );
        }
    }
}

/// Performs gaussian approximation on the image in linear colorspace
///
/// This is fast approximation that first converts in linear colorspace, performs blur and converts back,
/// operation will be performed in f32 so its cost is significant
/// O(1) complexity.
///
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - Almost any reasonable radius is supported
/// * `channels` - Count of channels of the image, only 3 and 4 is supported, alpha position, and channels order does not matter
/// * `threading_policy` - Threads usage policy
/// * `transfer_function` - Transfer function in linear colorspace
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_in_linear(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
    transfer_function: TransferFunction,
) {
    let mut linear_data: Vec<f32> =
        vec![0f32; width as usize * height as usize * channels.get_channels()];

    let forward_transformer = match channels {
        FastBlurChannels::Channels3 => rgb_to_linear,
        FastBlurChannels::Channels4 => rgba_to_linear,
    };

    let inverse_transformer = match channels {
        FastBlurChannels::Channels3 => linear_to_rgb,
        FastBlurChannels::Channels4 => linear_to_rgba,
    };

    forward_transformer(
        &in_place,
        stride,
        &mut linear_data,
        width * std::mem::size_of::<f32>() as u32 * channels.get_channels() as u32,
        width,
        height,
        transfer_function,
    );

    fast_gaussian_f32(
        &mut linear_data,
        width,
        height,
        radius,
        channels,
        threading_policy,
    );

    inverse_transformer(
        &linear_data,
        width * std::mem::size_of::<f32>() as u32 * channels.get_channels() as u32,
        in_place,
        stride,
        width,
        height,
        transfer_function,
    );
}

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for f32 image. No limitations are expected.
/// Approximation based on binomial filter. Algorithm is close to stack blur with better results and a little slower speed
/// O(1) complexity.
///
/// # Arguments
///
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - almost any radius is supported
/// * `channels` - Count of channels in the image
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_f16(
    bytes: &mut [u16],
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    let stride = width * channels.get_channels() as u32;
    fast_gaussian_f16::fast_gaussian_impl_f16(bytes, stride, width, height, radius, channels);
}
