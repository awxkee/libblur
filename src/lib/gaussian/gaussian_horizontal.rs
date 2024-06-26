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
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, reflect_101, reflect_index, EdgeMode};
use num_traits::{AsPrimitive, FromPrimitive};

pub(crate) fn gaussian_blur_horizontal_pass_impl<
    T: FromPrimitive + Default + Into<f32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
    const EDGE_MODE: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + 'static,
    f32: AsPrimitive<T> + ToStorage<T>,
{
    gaussian_blur_horizontal_pass_impl_c::<T, CHANNEL_CONFIGURATION, EDGE_MODE>(
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
    const EDGE_MODE: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + 'static,
    f32: AsPrimitive<T> + ToStorage<T>,
{
    let edge_mode: EdgeMode = EDGE_MODE.into();
    let half_kernel = (kernel_size / 2) as i32;
    for y in start_y..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let (mut weight0, mut weight1, mut weight2, mut weight3) = (0f32, 0f32, 0f32, 0f32);
            for r in -half_kernel..=half_kernel {
                let px = clamp_edge!(edge_mode, x as i64 + r as i64, 0, (width - 1) as i64)
                    * CHANNEL_CONFIGURATION;
                let y_offset = y_src_shift + px;
                let weight = unsafe { *kernel.get_unchecked((r + half_kernel) as usize) };
                weight0 += (unsafe { *src.get_unchecked(y_offset) }.into()) * weight;
                weight1 += (unsafe { *src.get_unchecked(y_offset + 1) }.into()) * weight;
                weight2 += (unsafe { *src.get_unchecked(y_offset + 2) }.into()) * weight;
                if CHANNEL_CONFIGURATION == 4 {
                    weight3 += (unsafe { *src.get_unchecked(y_offset + 3) }.into()) * weight;
                }
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

            unsafe {
                let bytes_offset = y_dst_shift + px;
                unsafe_dst.write(bytes_offset, weight0.to_());
                unsafe_dst.write(bytes_offset + 1, weight1.to_());
                unsafe_dst.write(bytes_offset + 2, weight2.to_());
                if CHANNEL_CONFIGURATION == 4 {
                    unsafe_dst.write(bytes_offset + 3, weight3.to_());
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
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + 'static,
    f32: AsPrimitive<T> + ToStorage<T>,
{
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
                let y_offset = y_src_shift + px;
                weights[0] += (unsafe { *src.get_unchecked(y_offset) }.into()) * weight;
                weights[1] += (unsafe { *src.get_unchecked(y_offset + 1) }.into()) * weight;
                weights[2] += (unsafe { *src.get_unchecked(y_offset + 2) }.into()) * weight;
                if CHANNEL_CONFIGURATION == 4 {
                    weights[3] += (unsafe { *src.get_unchecked(y_offset + 3) }.into()) * weight;
                }
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

            unsafe {
                unsafe_dst.write(y_dst_shift + px, weights[0].to_());
                unsafe_dst.write(y_dst_shift + px + 1, weights[1].to_());
                unsafe_dst.write(y_dst_shift + px + 2, weights[2].to_());
                if CHANNEL_CONFIGURATION == 4 {
                    unsafe_dst.write(y_dst_shift + px + 3, weights[3].to_());
                }
            }
        }
    }
}
