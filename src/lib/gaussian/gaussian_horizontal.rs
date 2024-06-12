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

use crate::gaussian::gaussian_filter::GaussianFilter;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::gaussian::gaussian_neon::neon_support;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::gaussian::gaussian_neon_filter::neon_gaussian_filter::gaussian_blur_horizontal_pass_filter_neon;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::gaussian::gaussian_sse::sse_support::gaussian_blur_horizontal_pass_impl_sse;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::FromPrimitive;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::gaussian::gaussian_sse_filter::sse_filter::gaussian_blur_horizontal_pass_filter_sse;

pub(crate) fn gaussian_blur_horizontal_pass_impl<
    T: FromPrimitive + Default + Into<f32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &Vec<f32>,
    start_y: u32,
    end_y: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    if std::any::type_name::<T>() == "u8" {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let u8_slice: &[u8] = unsafe { std::mem::transmute(src) };
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(unsafe_dst) };
            neon_support::gaussian_blur_horizontal_pass_neon::<CHANNEL_CONFIGURATION>(
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
            return;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            let u8_slice: &[u8] = unsafe { std::mem::transmute(src) };
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(unsafe_dst) };
            gaussian_blur_horizontal_pass_impl_sse::<CHANNEL_CONFIGURATION>(
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
            return;
        }
    }
    gaussian_blur_horizontal_pass_impl_c::<T, CHANNEL_CONFIGURATION>(
        src,
        src_stride,
        unsafe_dst,
        dst_stride,
        width,
        kernel_size,
        kernel,
        start_y,
        end_y,
    );
}

pub(crate) fn gaussian_blur_horizontal_pass_impl_c<
    T: FromPrimitive + Default + Into<f32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &Vec<f32>,
    start_y: u32,
    end_y: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let half_kernel = (kernel_size / 2) as i32;
    for y in start_y..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let mut weights: [f32; 4] = [0f32; 4];
            for r in -half_kernel..=half_kernel {
                let px = std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                    as usize
                    * CHANNEL_CONFIGURATION;
                let weight = unsafe { *kernel.get_unchecked((r + half_kernel) as usize) };
                weights[0] += (unsafe { *src.get_unchecked(y_src_shift + px) }.into()) * weight;
                weights[1] += (unsafe { *src.get_unchecked(y_src_shift + px + 1) }.into()) * weight;
                weights[2] += (unsafe { *src.get_unchecked(y_src_shift + px + 2) }.into()) * weight;
                if CHANNEL_CONFIGURATION == 4 {
                    weights[3] +=
                        (unsafe { *src.get_unchecked(y_src_shift + px + 3) }.into()) * weight;
                }
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

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
                if CHANNEL_CONFIGURATION == 4 {
                    unsafe_dst.write(
                        y_dst_shift + px + 3,
                        T::from_f32(weights[3]).unwrap_or_default(),
                    );
                }
            }
        }
    }
}

pub(crate) fn gaussian_blur_horizontal_pass_impl_clip_edge<
    T: FromPrimitive + Default + Into<f32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &Vec<GaussianFilter>,
    start_y: u32,
    end_y: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    if std::any::type_name::<T>() == "u8" {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let u8_slice: &[u8] = unsafe { std::mem::transmute(src) };
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(unsafe_dst) };
            gaussian_blur_horizontal_pass_filter_neon::<CHANNEL_CONFIGURATION>(
                u8_slice, src_stride, slice, dst_stride, width, filter, start_y, end_y,
            );
            return;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            let u8_slice: &[u8] = unsafe { std::mem::transmute(src) };
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(unsafe_dst) };
            gaussian_blur_horizontal_pass_filter_sse::<CHANNEL_CONFIGURATION>(
                u8_slice,
                src_stride,
                slice,
                dst_stride,
                width,
                filter,
                start_y,
                end_y,
            );
            return;
        }
    }
    for y in start_y..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let mut weights: [f32; 4] = [0f32; 4];

            let current_filter = unsafe { filter.get_unchecked(x as usize) };
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            for j in 0..current_filter.size {
                let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                let weight = unsafe { *filter_weights.get_unchecked(j) };
                weights[0] += (unsafe { *src.get_unchecked(y_src_shift + px) }.into()) * weight;
                weights[1] += (unsafe { *src.get_unchecked(y_src_shift + px + 1) }.into()) * weight;
                weights[2] += (unsafe { *src.get_unchecked(y_src_shift + px + 2) }.into()) * weight;
                if CHANNEL_CONFIGURATION == 4 {
                    weights[3] +=
                        (unsafe { *src.get_unchecked(y_src_shift + px + 3) }.into()) * weight;
                }
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

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
                if CHANNEL_CONFIGURATION == 4 {
                    unsafe_dst.write(
                        y_dst_shift + px + 3,
                        T::from_f32(weights[3]).unwrap_or_default(),
                    );
                }
            }
        }
    }
}