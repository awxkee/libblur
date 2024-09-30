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

use crate::gaussian::gaussian_approx::{PRECISION, ROUNDING_APPROX};
use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, reflect_101, reflect_index, EdgeMode};

#[allow(clippy::too_many_arguments)]
pub(crate) fn gaussian_blur_vertical_pass_c_approx<const CHANNEL_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[i16],
    start_y: u32,
    end_y: u32,
    edge_mode: EdgeMode,
) {
    let total_length = width as usize * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let mut _cx = 0usize;

        while _cx + 32 < total_length {
            gaussian_vertical_row_ap::<32>(
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
                edge_mode,
            );
            _cx += 32;
        }

        while _cx + 16 < total_length {
            gaussian_vertical_row_ap::<16>(
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
                edge_mode,
            );
            _cx += 16;
        }

        while _cx + 8 < total_length {
            gaussian_vertical_row_ap::<8>(
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
                edge_mode,
            );
            _cx += 8;
        }

        while _cx + 4 < total_length {
            gaussian_vertical_row_ap::<4>(
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
                edge_mode,
            );
            _cx += 4;
        }

        while _cx < total_length {
            gaussian_vertical_row_ap::<1>(
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
                edge_mode,
            );
            _cx += 1;
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn gaussian_vertical_row_ap<const ROW_SIZE: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    _: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[i16],
    x: u32,
    y: u32,
    edge_mode: EdgeMode,
) {
    let half_kernel = (kernel_size / 2) as i32;
    let mut weights: [i32; ROW_SIZE] = [ROUNDING_APPROX; ROW_SIZE];
    for r in -half_kernel..=half_kernel {
        let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0, (height - 1) as i64);
        let y_src_shift = py * src_stride as usize;
        let weight = unsafe { *kernel.get_unchecked((r + half_kernel) as usize) };
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe {
                let v = *src.get_unchecked(y_src_shift + px);
                let w0 = weights.get_unchecked_mut(i);
                *w0 += v as i32 * weight as i32;
            }
        }
    }
    let y_dst_shift = y as usize * dst_stride as usize;
    unsafe {
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe_dst.write(
                y_dst_shift + px,
                ((*weights.get_unchecked(i)) >> PRECISION).min(255) as u8,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gaussian_blur_vertical_pass_clip_edge_approx<const CHANNEL_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &[GaussianFilter<i16>],
    start_y: u32,
    end_y: u32,
) {
    let total_length = width as usize * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let mut _cx = 0usize;

        while _cx + 32 < total_length {
            gaussian_vertical_row_clip_edge::<32>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 32;
        }

        while _cx + 16 < total_length {
            gaussian_vertical_row_clip_edge::<16>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 16;
        }

        while _cx + 8 < total_length {
            gaussian_vertical_row_clip_edge::<8>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 8;
        }

        while _cx + 4 < total_length {
            gaussian_vertical_row_clip_edge::<4>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 4;
        }

        while _cx < total_length {
            gaussian_vertical_row_clip_edge::<1>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 1;
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn gaussian_vertical_row_clip_edge<const ROW_SIZE: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    _: u32,
    _: u32,
    filter: &[GaussianFilter<i16>],
    x: u32,
    y: u32,
) {
    let mut weights: [i32; ROW_SIZE] = [ROUNDING_APPROX; ROW_SIZE];
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
                *w0 += v as i32 * weight as i32;
            }
        }
    }
    let y_dst_shift = y as usize * dst_stride as usize;
    unsafe {
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe_dst.write(
                y_dst_shift + px,
                ((*weights.get_unchecked(i)) >> PRECISION).min(255) as u8,
            );
        }
    }
}
