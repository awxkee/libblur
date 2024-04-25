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

#[allow(unused_imports)]
use crate::box_blur_neon::neon_support;
use crate::channels_configuration::FastBlurChannels;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::cast::FromPrimitive;
use std::thread;

#[allow(unused_variables)]
#[allow(unused_imports)]
fn box_blur_horizontal_pass_impl<T: FromPrimitive + Default + Into<u32> + Send + Sync>(
    src: &Vec<T>,
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    box_channels: FastBlurChannels,
    start_y: u32,
    end_y: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    if std::any::type_name::<T>() == "u8" {
        #[cfg(target_arch = "aarch64")]
        {
            let u8_slice: &Vec<u8> = unsafe { std::mem::transmute(src) };
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(unsafe_dst) };
            neon_support::box_blur_horizontal_pass_4channels_u8_impl(
                u8_slice,
                src_stride,
                slice,
                dst_stride,
                width,
                radius,
                start_y,
                end_y,
                box_channels,
            );
            return;
        }
    }
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let half_kernel = kernel_size / 2;
    let kernel_scale = 1f32 / (radius * 2) as f32;
    let channels_count = match box_channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    } as usize;

    for y in start_y..end_y {
        let mut kernel: [u32; 4] = [0; 4];
        let y_src_shift = (y * src_stride) as usize;
        let y_dst_shift = (y * dst_stride) as usize;
        // replicate edge
        kernel[0] = (src[y_src_shift].into()) * edge_count;
        kernel[1] = (src[y_src_shift + 1].into()) * edge_count;
        kernel[2] = (src[y_src_shift + 2].into()) * edge_count;
        match box_channels {
            FastBlurChannels::Channels3 => {}
            FastBlurChannels::Channels4 => {
                kernel[3] = (src[y_src_shift + 3].into()) * edge_count;
            }
        }

        for x in 1..std::cmp::min(half_kernel, width) {
            let px = x as usize * channels_count;
            kernel[0] += src[y_src_shift + px].into();
            kernel[1] += src[y_src_shift + px + 1].into();
            kernel[2] += src[y_src_shift + px + 2].into();
            match box_channels {
                FastBlurChannels::Channels3 => {}
                FastBlurChannels::Channels4 => {
                    kernel[3] += src[y_src_shift + px + 3].into();
                }
            }
        }

        for x in 0..width {
            let next = std::cmp::min(x + half_kernel, width - 1) as usize * channels_count;
            let previous =
                std::cmp::max(x as i64 - half_kernel as i64, 0) as usize * channels_count;
            let px = x as usize * channels_count;
            // Prune previous and add next and compute mean

            kernel[0] += src[y_src_shift + next].into();
            kernel[1] += src[y_src_shift + next + 1].into();
            kernel[2] += src[y_src_shift + next + 2].into();

            kernel[0] -= src[y_src_shift + previous].into();
            kernel[1] -= src[y_src_shift + previous + 1].into();
            kernel[2] -= src[y_src_shift + previous + 2].into();

            match box_channels {
                FastBlurChannels::Channels3 => {}
                FastBlurChannels::Channels4 => {
                    kernel[3] += src[y_src_shift + next + 3].into();
                    kernel[3] -= src[y_src_shift + previous + 3].into();
                }
            }

            unsafe {
                unsafe_dst.write(
                    y_dst_shift + px + 0,
                    T::from_f32(kernel[0] as f32 * kernel_scale).unwrap_or_default(),
                );
                unsafe_dst.write(
                    y_dst_shift + px + 1,
                    T::from_f32(kernel[1] as f32 * kernel_scale).unwrap_or_default(),
                );
                unsafe_dst.write(
                    y_dst_shift + px + 2,
                    T::from_f32(kernel[2] as f32 * kernel_scale).unwrap_or_default(),
                );

                match box_channels {
                    FastBlurChannels::Channels3 => {}
                    FastBlurChannels::Channels4 => {
                        unsafe_dst.write(
                            y_dst_shift + px + 3,
                            T::from_f32(kernel[3] as f32 * kernel_scale).unwrap_or_default(),
                        );
                    }
                }
            }
        }
    }
}

fn box_blur_horizontal_pass<T: FromPrimitive + Default + Into<u32> + Send + Sync>(
    src: &Vec<T>,
    src_stride: u32,
    dst: &mut Vec<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    box_channels: FastBlurChannels,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let unsafe_dst = UnsafeSlice::new(dst);
    thread::scope(|scope| {
        let thread_count = std::cmp::max(std::cmp::min(width * height / (256 * 256), 12), 1);
        let segment_size = height / thread_count;
        let mut handles = vec![];
        for i in 0..thread_count {
            let start_y = i * segment_size;
            let mut end_y = (i + 1) * segment_size;
            if i == thread_count - 1 {
                end_y = height;
            }

            let handle = scope.spawn(move || {
                box_blur_horizontal_pass_impl(
                    src,
                    src_stride,
                    &unsafe_dst,
                    dst_stride,
                    width,
                    radius,
                    box_channels,
                    start_y,
                    end_y,
                );
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

#[allow(unused_variables)]
#[allow(unused_imports)]
fn box_blur_vertical_pass_impl<T: FromPrimitive + Default + Into<u32> + Sync + Send + Copy>(
    src: &Vec<T>,
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    box_channels: FastBlurChannels,
    start_x: u32,
    end_x: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    if std::any::type_name::<T>() == "u8" {
        #[cfg(target_arch = "aarch64")]
        {
            let u8_slice: &Vec<u8> = unsafe { std::mem::transmute(src) };
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(unsafe_dst) };
            neon_support::box_blur_vertical_pass_4channels_u8_impl(
                u8_slice,
                src_stride,
                slice,
                dst_stride,
                width,
                height,
                radius,
                start_x,
                end_x,
                box_channels,
            );
            return;
        }
    }
    let kernel_size = radius * 2 + 1;
    let kernel_scale = 1f32 / (radius * 2) as f32;
    let edge_count = (kernel_size / 2) + 1;
    let half_kernel = kernel_size / 2;
    let channels_count = match box_channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    };

    for x in start_x..end_x {
        let mut kernel: [u32; 4] = [0; 4];
        // replicate edge
        let px = x as usize * channels_count;
        kernel[0] = (src[px].into()) * edge_count;
        kernel[1] = (src[px + 1].into()) * edge_count;
        kernel[2] = (src[px + 2].into()) * edge_count;
        match box_channels {
            FastBlurChannels::Channels3 => {}
            FastBlurChannels::Channels4 => {
                kernel[3] = (src[px + 3].into()) * edge_count;
            }
        }

        for y in 1..std::cmp::min(half_kernel, height) {
            let y_src_shift = y as usize * src_stride as usize;
            kernel[0] += src[y_src_shift + px].into();
            kernel[1] += src[y_src_shift + px + 1].into();
            kernel[2] += src[y_src_shift + px + 2].into();
            match box_channels {
                FastBlurChannels::Channels3 => {}
                FastBlurChannels::Channels4 => {
                    kernel[3] += src[y_src_shift + px + 3].into();
                }
            }
        }

        for y in 0..height {
            let next = std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
            let previous =
                std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
            let y_dst_shift = dst_stride as usize * y as usize;
            // Prune previous and add next and compute mean

            kernel[0] += src[next + px].into();
            kernel[1] += src[next + px + 1].into();
            kernel[2] += src[next + px + 2].into();

            kernel[0] -= src[previous + px].into();
            kernel[1] -= src[previous + px + 1].into();
            kernel[2] -= src[previous + px + 2].into();

            match box_channels {
                FastBlurChannels::Channels3 => {}
                FastBlurChannels::Channels4 => {
                    kernel[3] += src[next + px + 3].into();
                    kernel[3] -= src[previous + px + 3].into();
                }
            }

            unsafe {
                unsafe_dst.write(
                    y_dst_shift + px + 0,
                    T::from_f32(kernel[0] as f32 * kernel_scale).unwrap_or_default(),
                );
                unsafe_dst.write(
                    y_dst_shift + px + 1,
                    T::from_f32(kernel[1] as f32 * kernel_scale).unwrap_or_default(),
                );
                unsafe_dst.write(
                    y_dst_shift + px + 2,
                    T::from_f32(kernel[2] as f32 * kernel_scale).unwrap_or_default(),
                );
                match box_channels {
                    FastBlurChannels::Channels3 => {}
                    FastBlurChannels::Channels4 => {
                        unsafe_dst.write(
                            y_dst_shift + px + 3,
                            T::from_f32(kernel[3] as f32 * kernel_scale).unwrap_or_default(),
                        );
                    }
                }
            }
        }
    }
}

fn box_blur_vertical_pass<T: FromPrimitive + Default + Into<u32> + Sync + Send + Copy>(
    src: &Vec<T>,
    src_stride: u32,
    dst: &mut Vec<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    box_channels: FastBlurChannels,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let unsafe_dst = UnsafeSlice::new(dst);

    thread::scope(|scope| {
        let thread_count = std::cmp::max(std::cmp::min(width * height / (256 * 256), 12), 1);
        let segment_size = width / thread_count;
        let mut handles = vec![];
        for i in 0..thread_count {
            let start_x = i * segment_size;
            let mut end_x = (i + 1) * segment_size;
            if i == thread_count - 1 {
                end_x = width;
            }

            let handle = scope.spawn(move || {
                box_blur_vertical_pass_impl(
                    src,
                    src_stride,
                    &unsafe_dst,
                    dst_stride,
                    width,
                    height,
                    radius,
                    box_channels,
                    start_x,
                    end_x,
                );
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

fn box_blur_impl<T: FromPrimitive + Default + Into<u32> + Sync + Send + Copy>(
    src: &Vec<T>,
    src_stride: u32,
    dst: &mut Vec<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let mut transient: Vec<T> = Vec::with_capacity(dst_stride as usize * height as usize);
    transient.resize(
        dst_stride as usize * height as usize,
        T::from_u32(0).unwrap_or_default(),
    );
    box_blur_horizontal_pass(
        src,
        src_stride,
        &mut transient,
        dst_stride,
        width,
        height,
        radius,
        channels,
    );
    box_blur_vertical_pass(
        &transient, src_stride, dst, dst_stride, width, height, radius, channels,
    );
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn box_blur(
    src: &Vec<u8>,
    src_stride: u32,
    dst: &mut Vec<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    box_blur_impl(
        src, src_stride, dst, dst_stride, width, height, radius, channels,
    );
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn box_blur_u16(
    src: &Vec<u16>,
    src_stride: u32,
    dst: &mut Vec<u16>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    box_blur_impl(
        src, src_stride, dst, dst_stride, width, height, radius, channels,
    );
}

fn tent_blur_impl<T: FromPrimitive + Default + Into<u32> + Sync + Send + Copy>(
    src: &Vec<T>,
    src_stride: u32,
    dst: &mut Vec<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let mut transient: Vec<T> = Vec::with_capacity(dst_stride as usize * height as usize);
    transient.resize(
        dst_stride as usize * height as usize,
        T::from_u32(0).unwrap_or_default(),
    );
    box_blur_impl(
        src,
        src_stride,
        &mut transient,
        dst_stride,
        width,
        height,
        radius,
        channels,
    );
    box_blur_impl(
        &transient, src_stride, dst, dst_stride, width, height, radius, channels,
    );
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn tent_blur(
    src: &Vec<u8>,
    src_stride: u32,
    dst: &mut Vec<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    tent_blur_impl(
        src, src_stride, dst, dst_stride, width, height, radius, channels,
    );
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn tent_blur_u16(
    src: &Vec<u16>,
    src_stride: u32,
    dst: &mut Vec<u16>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    tent_blur_impl(
        src, src_stride, dst, dst_stride, width, height, radius, channels,
    );
}

fn gaussian_box_blur_impl<T: FromPrimitive + Default + Into<u32> + Sync + Send + Copy>(
    src: &Vec<T>,
    src_stride: u32,
    dst: &mut Vec<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    box_channels: FastBlurChannels,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let mut transient: Vec<T> = Vec::with_capacity(dst_stride as usize * height as usize);
    transient.resize(
        dst_stride as usize * height as usize,
        T::from_u32(0).unwrap_or_default(),
    );
    let mut transient2: Vec<T> = Vec::with_capacity(dst_stride as usize * height as usize);
    transient2.resize(
        dst_stride as usize * height as usize,
        T::from_u32(0).unwrap_or_default(),
    );
    box_blur_impl(
        &src,
        src_stride,
        &mut transient,
        dst_stride,
        width,
        height,
        radius,
        box_channels,
    );
    box_blur_impl(
        &transient,
        src_stride,
        &mut transient2,
        dst_stride,
        width,
        height,
        radius,
        box_channels,
    );
    box_blur_impl(
        &transient2,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        radius,
        box_channels,
    );
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn gaussian_box_blur(
    src: &Vec<u8>,
    src_stride: u32,
    dst: &mut Vec<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    gaussian_box_blur_impl(
        src, src_stride, dst, dst_stride, width, height, radius, channels,
    );
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn gaussian_box_blur_u16(
    src: &Vec<u16>,
    src_stride: u32,
    dst: &mut Vec<u16>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    gaussian_box_blur_impl(
        src, src_stride, dst, dst_stride, width, height, radius, channels,
    );
}
