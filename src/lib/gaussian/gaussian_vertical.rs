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

pub fn gaussian_blur_vertical_pass_c_impl<
    T: FromPrimitive + Default + Into<f32> + Send + Sync + Copy + 'static,
    const CHANNEL_CONFIGURATION: usize,
    const EDGE_MODE: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) where
    f32: AsPrimitive<T> + ToStorage<T>,
{
    let total_length = width as usize * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let mut _cx = 0usize;

        while _cx + 32 < total_length {
            gaussian_vertical_row::<T, 32, EDGE_MODE>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 32;
        }

        while _cx + 16 < total_length {
            gaussian_vertical_row::<T, 16, EDGE_MODE>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 16;
        }

        while _cx + 8 < total_length {
            gaussian_vertical_row::<T, 8, EDGE_MODE>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 8;
        }

        while _cx + 4 < total_length {
            gaussian_vertical_row::<T, 4, EDGE_MODE>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 4;
        }

        while _cx < total_length {
            gaussian_vertical_row::<T, 1, EDGE_MODE>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 1;
        }
    }
}

#[inline]
pub fn gaussian_vertical_row<
    T: FromPrimitive + Default + Into<f32> + Send + Sync + Copy + 'static,
    const ROW_SIZE: usize,
    const EDGE_MODE: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    _: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    x: u32,
    y: u32,
) where
    f32: AsPrimitive<T> + ToStorage<T>,
{
    let edge_mode: EdgeMode = EDGE_MODE.into();
    let half_kernel = (kernel_size / 2) as i32;
    let mut weights: [f32; ROW_SIZE] = [0f32; ROW_SIZE];
    for r in -half_kernel..=half_kernel {
        let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0, (height - 1) as i64);
        let y_src_shift = py * src_stride as usize;
        let weight = unsafe { *kernel.get_unchecked((r + half_kernel) as usize) };
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe {
                let v = *src.get_unchecked(y_src_shift + px);
                let w0 = weights.get_unchecked_mut(i);
                *w0 = *w0 + v.into() * weight;
            }
        }
    }
    let y_dst_shift = y as usize * dst_stride as usize;
    unsafe {
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe_dst.write(y_dst_shift + px, (*weights.get_unchecked(i)).as_());
        }
    }
}

pub fn gaussian_blur_vertical_pass_clip_edge_impl<
    T: FromPrimitive + Default + Into<f32> + Send + Sync + Copy + 'static,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter>,
    start_y: u32,
    end_y: u32,
) where
    f32: ToStorage<T>,
{
    let total_length = width as usize * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let mut _cx = 0usize;

        while _cx + 32 < total_length {
            gaussian_vertical_row_clip_edge::<T, 32>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 32;
        }

        while _cx + 16 < total_length {
            gaussian_vertical_row_clip_edge::<T, 16>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 16;
        }

        while _cx + 8 < total_length {
            gaussian_vertical_row_clip_edge::<T, 8>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 8;
        }

        while _cx + 4 < total_length {
            gaussian_vertical_row_clip_edge::<T, 4>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 4;
        }

        while _cx < total_length {
            gaussian_vertical_row_clip_edge::<T, 1>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 1;
        }
    }
}

#[inline]
pub fn gaussian_vertical_row_clip_edge<
    T: FromPrimitive + Default + Into<f32> + Send + Sync + Copy + 'static,
    const ROW_SIZE: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    _: u32,
    _: u32,
    filter: &Vec<GaussianFilter>,
    x: u32,
    y: u32,
) where
    f32: ToStorage<T>,
{
    let mut weights: [f32; ROW_SIZE] = [0f32; ROW_SIZE];
    let current_filter = unsafe { filter.get_unchecked(y as usize) };
    let filter_start = current_filter.start;
    let filter_weights = &current_filter.filter;
    for j in 0..current_filter.size {
        let py = filter_start + j;
        let y_src_shift = py * src_stride as usize;
        let weight = unsafe { *filter_weights.get_unchecked(j) };
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe {
                let v = *src.get_unchecked(y_src_shift + px);
                let w0 = weights.get_unchecked_mut(i);
                *w0 = *w0 + v.into() * weight;
            }
        }
    }
    let y_dst_shift = y as usize * dst_stride as usize;
    unsafe {
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe_dst.write(y_dst_shift + px, (*weights.get_unchecked(i)).to_());
        }
    }
}
