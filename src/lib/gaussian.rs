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
use crate::gaussian_neon::{gaussian_blur_horizontal_pass_impl_neon_3channels_u8, gaussian_blur_vertical_pass_impl_neon_3channels_u8};
use crate::unsafe_slice::UnsafeSlice;
use crate::FastBlurChannels::Channels3;
use num_traits::cast::FromPrimitive;
use std::thread;

fn get_gaussian_kernel_1d(width: u32, sigma: f32) -> Vec<f32> {
    let mut sum_norm: f32 = 0f32;
    let mut kernel: Vec<f32> = Vec::with_capacity(width as usize);
    let scale = 1f32 / (f32::sqrt(2f32 * std::f32::consts::PI) * sigma);
    let mean = (width / 2) as f32;

    for x in 0..width {
        let new_weight = f32::exp(-0.5f32 * f32::powf((x as f32 - mean) / sigma, 2.0f32)) * scale;
        kernel.push(new_weight);
        sum_norm += new_weight;
    }

    if sum_norm != 0f32 {
        for x in 0..width as usize {
            kernel[x] = kernel[x] / sum_norm;
        }
    }

    return kernel;
}

fn gaussian_blur_horizontal_pass_impl<T: FromPrimitive + Default + Into<f32> + Send + Sync>(
    src: &Vec<T>,
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    gaussian_channels: FastBlurChannels,
    kernel: &Vec<f32>,
    start_y: u32,
    end_y: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    #[cfg(target_arch = "aarch64")]
    match gaussian_channels {
        Channels3 => {
            #[cfg(target_arch = "aarch64")]
            if std::any::type_name::<T>() == "u8" {
                let u8_slice: &Vec<u8> = unsafe { std::mem::transmute(src) };
                let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(unsafe_dst) };
                gaussian_blur_horizontal_pass_impl_neon_3channels_u8(
                    u8_slice,
                    src_stride,
                    slice,
                    dst_stride,
                    width,
                    kernel_size,
                    kernel,
                    start_y,
                    end_y,
                );
            }
            return;
        }
        FastBlurChannels::Channels4 => {}
    }
    let half_kernel = (kernel_size / 2) as i32;
    let channels_count = match gaussian_channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    };
    for y in start_y..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let mut weights: [f32; 4] = [0f32; 4];
            for r in -half_kernel..=half_kernel {
                let px = std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                    as usize
                    * channels_count;
                let weight = kernel[(r + half_kernel) as usize];
                weights[0] += (src[y_src_shift + px].into()) * weight;
                weights[1] += (src[y_src_shift + px + 1].into()) * weight;
                weights[2] += (src[y_src_shift + px + 2].into()) * weight;
                match gaussian_channels {
                    FastBlurChannels::Channels3 => {}
                    FastBlurChannels::Channels4 => {
                        weights[3] += (src[y_src_shift + px + 3].into()) * weight;
                    }
                }
            }

            let px = x as usize * channels_count;

            unsafe {
                unsafe_dst.write(
                    y_dst_shift + px,
                    T::from_f32(weights[0]).unwrap_or_default(),
                );
                unsafe_dst.write(
                    y_dst_shift + px + 1,
                    T::from_f32(weights[1]).unwrap_or_default(),
                );
                unsafe_dst.write(
                    y_dst_shift + px + 2,
                    T::from_f32(weights[2]).unwrap_or_default(),
                );
                match gaussian_channels {
                    FastBlurChannels::Channels3 => {}
                    FastBlurChannels::Channels4 => {
                        unsafe_dst.write(
                            y_dst_shift + px + 3,
                            T::from_f32(weights[3]).unwrap_or_default(),
                        );
                    }
                }
            }
        }
    }
}

fn gaussian_blur_horizontal_pass<T: FromPrimitive + Default + Into<f32> + Send + Sync>(
    src: &Vec<T>,
    src_stride: u32,
    dst: &mut Vec<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    gaussian_channels: FastBlurChannels,
    kernel: &Vec<f32>,
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
                gaussian_blur_horizontal_pass_impl(
                    src,
                    src_stride,
                    &unsafe_dst,
                    dst_stride,
                    width,
                    kernel_size,
                    gaussian_channels,
                    kernel,
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

fn gaussian_blur_vertical_pass_impl<T: FromPrimitive + Default + Into<f32> + Send + Sync>(
    src: &Vec<T>,
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    gaussian_channels: FastBlurChannels,
    kernel: &Vec<f32>,
    start_y: u32,
    end_y: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    #[cfg(target_arch = "aarch64")]
    match gaussian_channels {
        Channels3 => {
            #[cfg(target_arch = "aarch64")]
            if std::any::type_name::<T>() == "u8" {
                let u8_slice: &Vec<u8> = unsafe { std::mem::transmute(src) };
                let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(unsafe_dst) };
                gaussian_blur_vertical_pass_impl_neon_3channels_u8(
                    u8_slice,
                    src_stride,
                    slice,
                    dst_stride,
                    width,
                    height,
                    kernel_size,
                    kernel,
                    start_y,
                    end_y,
                );
            }
            return;
        }
        FastBlurChannels::Channels4 => {}
    }
    let half_kernel = (kernel_size / 2) as i32;
    let channels_count = match gaussian_channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    };
    for y in start_y..end_y {
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let px = x as usize * channels_count;
            let mut weights: [f32; 4] = [0f32; 4];
            for r in -half_kernel..=half_kernel {
                let py = std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                let y_src_shift = py as usize * src_stride as usize;
                let weight = kernel[(r + half_kernel) as usize];
                weights[0] += (src[y_src_shift + px].into()) * weight;
                weights[1] += (src[y_src_shift + px + 1].into()) * weight;
                weights[2] += (src[y_src_shift + px + 2].into()) * weight;
                match gaussian_channels {
                    FastBlurChannels::Channels3 => {}
                    FastBlurChannels::Channels4 => {
                        weights[3] += (src[y_src_shift + px + 3].into()) * weight;
                    }
                }
            }

            unsafe {
                unsafe_dst.write(
                    y_dst_shift + px,
                    T::from_f32(weights[0]).unwrap_or_default(),
                );
                unsafe_dst.write(
                    y_dst_shift + px + 1,
                    T::from_f32(weights[1]).unwrap_or_default(),
                );
                unsafe_dst.write(
                    y_dst_shift + px + 2,
                    T::from_f32(weights[2]).unwrap_or_default(),
                );
                match gaussian_channels {
                    FastBlurChannels::Channels3 => {}
                    FastBlurChannels::Channels4 => {
                        unsafe_dst.write(
                            y_dst_shift + px + 3,
                            T::from_f32(weights[3]).unwrap_or_default(),
                        );
                    }
                }
            }
        }
    }
}

fn gaussian_blur_vertical_pass<T: FromPrimitive + Default + Into<f32> + Send + Sync>(
    src: &Vec<T>,
    src_stride: u32,
    dst: &mut Vec<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    gaussian_channels: FastBlurChannels,
    kernel: &Vec<f32>,
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
                gaussian_blur_vertical_pass_impl(
                    src,
                    src_stride,
                    &unsafe_dst,
                    dst_stride,
                    width,
                    height,
                    kernel_size,
                    gaussian_channels,
                    kernel,
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

fn gaussian_blur_impl<T: FromPrimitive + Default + Into<f32> + Send + Sync>(
    src: &Vec<T>,
    src_stride: u32,
    dst: &mut Vec<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    box_channels: FastBlurChannels,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
    if kernel_size % 2 == 0 {
        panic!("kernel size must be odd");
    }
    let mut transient: Vec<T> = Vec::with_capacity(dst_stride as usize * height as usize);
    transient.resize(
        dst_stride as usize * height as usize,
        T::from_u32(0).unwrap_or_default(),
    );
    gaussian_blur_horizontal_pass(
        &src,
        src_stride,
        &mut transient,
        dst_stride,
        width,
        height,
        kernel.len(),
        box_channels,
        &kernel,
    );
    gaussian_blur_vertical_pass(
        &transient,
        dst_stride,
        dst,
        dst_stride,
        width,
        height,
        kernel.len(),
        box_channels,
        &kernel,
    );
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn gaussian_blur(
    src: &Vec<u8>,
    src_stride: u32,
    dst: &mut Vec<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
) {
    gaussian_blur_impl(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        kernel_size,
        sigma,
        channels,
    );
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn gaussian_blur_u16(
    src: &Vec<u16>,
    src_stride: u32,
    dst: &mut Vec<u16>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
) {
    gaussian_blur_impl(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        kernel_size,
        sigma,
        channels,
    );
}
