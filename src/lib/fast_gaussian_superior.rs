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

use crate::{FastBlurChannels, ThreadingPolicy};

use num_traits::{AsPrimitive, FromPrimitive, ToPrimitive};

use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;

macro_rules! update_differences_inside {
    ($dif:expr, $buffer:expr, $d_idx:expr, $d_idx_1:expr, $d_idx_2:expr, $d_idx_3:expr) => {{
        $dif += -4 * ((*$buffer.get_unchecked($d_idx)) + (*$buffer.get_unchecked($d_idx_1)))
            + 6 * (*$buffer.get_unchecked($d_idx_2))
            + (*$buffer.get_unchecked($d_idx_3));
    }};
}

macro_rules! update_differences_3_rad {
    ($dif:expr, $buffer:expr, $d_idx:expr) => {{
        $dif -= 4 * unsafe { *$buffer.get_unchecked($d_idx) };
    }};
}

macro_rules! update_differences_2_rad {
    ($dif:expr, $buffer:expr, $d_idx:expr) => {{
        $dif += 6 * unsafe { *$buffer.get_unchecked($d_idx) };
    }};
}

macro_rules! update_differences_1_rad {
    ($dif:expr, $buffer:expr, $d_idx:expr) => {{
        $dif -= 4 * unsafe { *$buffer.get_unchecked($d_idx) };
    }};
}

macro_rules! update_sum_in {
    ($bytes:expr, $bytes_offset:expr, $dif:expr, $der_2:expr, $der_1:expr, $sum:expr, $buffer:expr, $arr_index:expr) => {{
        let ug8 = $bytes[$bytes_offset];
        $dif += ug8.into();
        $der_2 += $dif;
        $der_1 += $der_2;
        $sum += $der_1;
        unsafe {
            *$buffer.get_unchecked_mut($arr_index) = ug8.into();
        }
    }};
}

fn fast_gaussian_vertical_pass<
    T: FromPrimitive
        + ToPrimitive
        + Default
        + Into<i64>
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static,
    const CHANNELS_COUNT: usize,
>(
    bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
) where
    f64: ToStorage<T> + AsPrimitive<T>,
{
    let mut buffer_r: [i64; 2048] = [0; 2048];
    let mut buffer_g: [i64; 2048] = [0; 2048];
    let mut buffer_b: [i64; 2048] = [0; 2048];
    let mut buffer_a: [i64; 2048] = [0; 2048];
    let radius_64 = radius as i64;
    let height_wide = height as i64;
    let radius_2d = (radius as f64) * (radius as f64);
    let weight = 1.0f64 / (radius_2d * radius_2d);
    for x in start..std::cmp::min(width, end) {
        let mut dif_r: i64 = 0;
        let mut der_1_r: i64 = 0;
        let mut der_2_r: i64 = 0;
        let mut sum_r: i64 = 0;
        let mut dif_g: i64 = 0;
        let mut der_1_g: i64 = 0;
        let mut der_2_g: i64 = 0;
        let mut sum_g: i64 = 0;
        let mut dif_b: i64 = 0;
        let mut der_1_b: i64 = 0;
        let mut der_2_b: i64 = 0;
        let mut sum_b: i64 = 0;
        let mut dif_a: i64 = 0;
        let mut der_1_a: i64 = 0;
        let mut der_2_a: i64 = 0;
        let mut sum_a: i64 = 0;

        let current_px = x as usize * CHANNELS_COUNT;

        let start_y = 0i64 - 4i64 * radius as i64;
        for y in start_y..height_wide {
            let current_y = (y * (stride as i64)) as usize;
            if y >= 0 {
                let new_r = ((sum_r as f64) * weight).to_();
                let new_g = if CHANNELS_COUNT > 1 {
                    ((sum_g as f64) * weight).to_()
                } else {
                    0f64.to_()
                };
                let new_b = if CHANNELS_COUNT > 2 {
                    ((sum_b as f64) * weight).to_()
                } else {
                    0f64.as_()
                };

                let bytes_offset = current_y + current_px;

                unsafe {
                    bytes.write(bytes_offset, new_r);
                    if CHANNELS_COUNT > 1 {
                        bytes.write(bytes_offset + 1, new_g);
                    }
                    if CHANNELS_COUNT > 2 {
                        bytes.write(bytes_offset + 2, new_b);
                    }
                    if CHANNELS_COUNT == 4 {
                        let new_a = ((sum_a as f64) * weight).to_();
                        bytes.write(bytes_offset + 3, new_a);
                    }
                }

                let idx3 = (y & 2047) as usize;
                let idx2 = ((y + radius_64) & 2047) as usize;
                let idx1 = ((y - radius_64) & 2047) as usize;
                let idx4 = ((y - 2 * radius_64) & 2047) as usize;

                unsafe {
                    update_differences_inside!(dif_r, buffer_r, idx1, idx2, idx3, idx4);
                    if CHANNELS_COUNT > 1 {
                        update_differences_inside!(dif_g, buffer_g, idx1, idx2, idx3, idx4);
                    }
                    if CHANNELS_COUNT > 2 {
                        update_differences_inside!(dif_b, buffer_b, idx1, idx2, idx3, idx4);
                    }
                    if CHANNELS_COUNT == 4 {
                        update_differences_inside!(dif_a, buffer_a, idx1, idx2, idx3, idx4);
                    }
                };
            } else {
                if y + 3 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 2047) as usize;
                    update_differences_3_rad!(dif_r, buffer_r, arr_index);
                    if CHANNELS_COUNT > 1 {
                        update_differences_3_rad!(dif_g, buffer_g, arr_index);
                    }
                    if CHANNELS_COUNT > 2 {
                        update_differences_3_rad!(dif_b, buffer_b, arr_index);
                    }
                    if CHANNELS_COUNT == 4 {
                        update_differences_3_rad!(dif_a, buffer_a, arr_index);
                    }
                }
                if y + 2 * radius_64 >= 0 {
                    let arr_index = (y & 2047) as usize;
                    update_differences_2_rad!(dif_r, buffer_r, arr_index);
                    if CHANNELS_COUNT > 1 {
                        update_differences_2_rad!(dif_g, buffer_g, arr_index);
                    }
                    if CHANNELS_COUNT > 2 {
                        update_differences_2_rad!(dif_b, buffer_b, arr_index);
                    }
                    if CHANNELS_COUNT == 4 {
                        update_differences_2_rad!(dif_a, buffer_a, arr_index);
                    }
                }
                if y + radius_64 >= 0 {
                    let arr_index = ((y - radius_64) & 2047) as usize;
                    update_differences_1_rad!(dif_r, buffer_r, arr_index);
                    if CHANNELS_COUNT > 1 {
                        update_differences_1_rad!(dif_g, buffer_g, arr_index);
                    }
                    if CHANNELS_COUNT > 2 {
                        update_differences_1_rad!(dif_b, buffer_b, arr_index);
                    }
                    if CHANNELS_COUNT == 4 {
                        update_differences_1_rad!(dif_a, buffer_a, arr_index);
                    }
                }
            }

            let next_row_y =
                (std::cmp::min(std::cmp::max(y + 2 * radius_64 - 1, 0), height_wide - 1) as usize)
                    * (stride as usize);
            let next_row_x = x as usize * CHANNELS_COUNT;

            let px_idx = next_row_y + next_row_x;

            let a_idx = ((y + 2 * radius_64) & 2047) as usize;

            update_sum_in!(bytes, px_idx, dif_r, der_2_r, der_1_r, sum_r, buffer_r, a_idx);

            if CHANNELS_COUNT > 1 {
                update_sum_in!(
                    bytes,
                    px_idx + 1,
                    dif_g,
                    der_2_g,
                    der_1_g,
                    sum_g,
                    buffer_g,
                    a_idx
                );
            }

            if CHANNELS_COUNT > 2 {
                update_sum_in!(
                    bytes,
                    px_idx + 2,
                    dif_b,
                    der_2_b,
                    der_1_b,
                    sum_b,
                    buffer_b,
                    a_idx
                );
            }

            if CHANNELS_COUNT == 4 {
                update_sum_in!(
                    bytes,
                    px_idx + 3,
                    dif_a,
                    der_2_a,
                    der_1_a,
                    sum_a,
                    buffer_a,
                    a_idx
                );
            }
        }
    }
}

fn fast_gaussian_horizontal_pass<
    T: FromPrimitive
        + ToPrimitive
        + Default
        + Into<i64>
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static,
    const CHANNELS_COUNT: usize,
>(
    bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
) where
    f64: ToStorage<T> + AsPrimitive<T>,
{
    let mut buffer_r: [i64; 2048] = [0; 2048];
    let mut buffer_g: [i64; 2048] = [0; 2048];
    let mut buffer_b: [i64; 2048] = [0; 2048];
    let mut buffer_a: [i64; 2048] = [0; 2048];
    let radius_64 = radius as i64;
    let width_wide = width as i64;
    let radius_2d = (radius as f64) * (radius as f64);
    let weight = 1.0f64 / (radius_2d * radius_2d);
    for y in start..std::cmp::min(height, end) {
        let mut dif_r: i64 = 0;
        let mut der_1_r: i64 = 0;
        let mut der_2_r: i64 = 0;
        let mut sum_r: i64 = 0;
        let mut dif_g: i64 = 0;
        let mut der_1_g: i64 = 0;
        let mut der_2_g: i64 = 0;
        let mut sum_g: i64 = 0;
        let mut dif_b: i64 = 0;
        let mut der_1_b: i64 = 0;
        let mut der_2_b: i64 = 0;
        let mut sum_b: i64 = 0;
        let mut dif_a: i64 = 0;
        let mut der_1_a: i64 = 0;
        let mut der_2_a: i64 = 0;
        let mut sum_a: i64 = 0;

        let current_y = ((y as i64) * (stride as i64)) as usize;

        for x in (0i64 - 4i64 * radius_64)..(width as i64) {
            if x >= 0 {
                let current_px = (std::cmp::max(x, 0) as u32) as usize * CHANNELS_COUNT;
                let new_r = ((sum_r as f64) * weight).to_();
                let new_g = if CHANNELS_COUNT > 1 {
                    ((sum_g as f64) * weight).to_()
                } else {
                    0f64.to_()
                };
                let new_b = if CHANNELS_COUNT > 2 {
                    ((sum_b as f64) * weight).to_()
                } else {
                    0f64.as_()
                };

                let bytes_offset = current_y + current_px;

                unsafe {
                    bytes.write(bytes_offset, new_r);
                    if CHANNELS_COUNT > 1 {
                        bytes.write(bytes_offset + 1, new_g);
                    }
                    if CHANNELS_COUNT > 2 {
                        bytes.write(bytes_offset + 2, new_b);
                    }
                    if CHANNELS_COUNT == 4 {
                        let new_a = ((sum_a as f64) * weight).to_();
                        bytes.write(bytes_offset + 3, new_a);
                    }
                }

                let idx3 = (x & 2047) as usize;
                let idx2 = ((x + radius_64) & 2047) as usize;
                let idx1 = ((x - radius_64) & 2047) as usize;
                let idx4 = ((x - 2 * radius_64) & 2047) as usize;

                unsafe {
                    update_differences_inside!(dif_r, buffer_r, idx1, idx2, idx3, idx4);
                    if CHANNELS_COUNT > 1 {
                        update_differences_inside!(dif_g, buffer_g, idx1, idx2, idx3, idx4);
                    }
                    if CHANNELS_COUNT > 2 {
                        update_differences_inside!(dif_b, buffer_b, idx1, idx2, idx3, idx4);
                    }
                    if CHANNELS_COUNT == 4 {
                        update_differences_inside!(dif_a, buffer_a, idx1, idx2, idx3, idx4);
                    }
                }
            } else {
                if x + 3 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 2047) as usize;
                    update_differences_3_rad!(dif_r, buffer_r, arr_index);
                    if CHANNELS_COUNT > 1 {
                        update_differences_3_rad!(dif_g, buffer_g, arr_index);
                    }
                    if CHANNELS_COUNT > 2 {
                        update_differences_3_rad!(dif_b, buffer_b, arr_index);
                    }
                    if CHANNELS_COUNT == 4 {
                        update_differences_3_rad!(dif_a, buffer_a, arr_index);
                    }
                }
                if x + 2 * radius_64 >= 0 {
                    let arr_index = (x & 2047) as usize;
                    update_differences_2_rad!(dif_r, buffer_r, arr_index);
                    if CHANNELS_COUNT > 1 {
                        update_differences_2_rad!(dif_g, buffer_g, arr_index);
                    }
                    if CHANNELS_COUNT > 2 {
                        update_differences_2_rad!(dif_b, buffer_b, arr_index);
                    }
                    if CHANNELS_COUNT == 4 {
                        update_differences_2_rad!(dif_a, buffer_a, arr_index);
                    }
                }
                if x + radius_64 >= 0 {
                    let arr_index = ((x - radius_64) & 2047) as usize;
                    update_differences_1_rad!(dif_r, buffer_r, arr_index);
                    if CHANNELS_COUNT > 1 {
                        update_differences_1_rad!(dif_g, buffer_g, arr_index);
                    }
                    if CHANNELS_COUNT > 2 {
                        update_differences_1_rad!(dif_b, buffer_b, arr_index);
                    }
                    if CHANNELS_COUNT == 4 {
                        update_differences_1_rad!(dif_a, buffer_a, arr_index);
                    }
                }
            }

            let next_row_y = (y as usize) * (stride as usize);
            let next_row_x = (std::cmp::min(std::cmp::max(x + 2 * radius_64 - 1, 0), width_wide - 1)
                as u32) as usize
                * CHANNELS_COUNT;

            let px_idx = next_row_y + next_row_x;

            let a_idx = ((x + 2 * radius_64) & 2047) as usize;

            update_sum_in!(bytes, px_idx, dif_r, der_2_r, der_1_r, sum_r, buffer_r, a_idx);

            if CHANNELS_COUNT > 1 {
                update_sum_in!(
                    bytes,
                    px_idx + 1,
                    dif_g,
                    der_2_g,
                    der_1_g,
                    sum_g,
                    buffer_g,
                    a_idx
                );
            }

            if CHANNELS_COUNT > 2 {
                update_sum_in!(
                    bytes,
                    px_idx + 2,
                    dif_b,
                    der_2_b,
                    der_1_b,
                    sum_b,
                    buffer_b,
                    a_idx
                );
            }

            if CHANNELS_COUNT == 4 {
                update_sum_in!(
                    bytes,
                    px_idx + 3,
                    dif_a,
                    der_2_a,
                    der_1_a,
                    sum_a,
                    buffer_a,
                    a_idx
                );
            }
        }
    }
}

pub(crate) fn fast_gaussian_impl<
    T: FromPrimitive
        + ToPrimitive
        + Default
        + Into<i64>
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static,
    const CHANNELS_COUNT: usize,
>(
    bytes: &mut [T],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    threading_policy: ThreadingPolicy,
) where
    f64: ToStorage<T> + AsPrimitive<T>,
{
    let unsafe_image = UnsafeSlice::new(bytes);
    let thread_count = threading_policy.get_threads_count(width, height) as u32;
    if thread_count == 1 {
        fast_gaussian_vertical_pass::<T, CHANNELS_COUNT>(
            &unsafe_image,
            stride,
            width,
            height,
            radius,
            0,
            width,
        );
        fast_gaussian_horizontal_pass::<T, CHANNELS_COUNT>(
            &unsafe_image,
            stride,
            width,
            height,
            radius,
            0,
            height,
        );
    } else {
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
                    fast_gaussian_vertical_pass::<T, CHANNELS_COUNT>(
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
                    fast_gaussian_horizontal_pass::<T, CHANNELS_COUNT>(
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

/// Fast gaussian approximation. This is almost gaussian blur. Significantly slower than alternatives.
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `radius` - Radius more than ~256 is not supported.
///
/// O(1) complexity.
pub fn fast_gaussian_superior(
    bytes: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    let acq_radius = std::cmp::min(radius, 256);
    let _dispatcher = match channels {
        FastBlurChannels::Plane => fast_gaussian_impl::<u8, 1>,
        FastBlurChannels::Channels3 => fast_gaussian_impl::<u8, 3>,
        FastBlurChannels::Channels4 => fast_gaussian_impl::<u8, 4>,
    };
    _dispatcher(bytes, stride, width, height, acq_radius, threading_policy);
}
