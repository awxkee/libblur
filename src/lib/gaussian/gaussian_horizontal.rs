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
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static
        + AsPrimitive<f32>,
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

macro_rules! accumulate_4_c_u8 {
    ($channels: expr, $w0: expr, $w1: expr, $w2: expr, $w3: expr, $weight:expr, $src: expr, $offset:expr) => {{
        $w0 += (unsafe { *$src.get_unchecked($offset) }.as_()) * $weight;
        if $channels > 1 {
            $w1 += (unsafe { *$src.get_unchecked($offset + 1) }.as_()) * $weight;
        }
        if $channels > 2 {
            $w2 += (unsafe { *$src.get_unchecked($offset + 2) }.as_()) * $weight;
        }
        if $channels == 4 {
            $w3 += (unsafe { *$src.get_unchecked($offset + 3) }.as_()) * $weight;
        }
    }};
}

macro_rules! save_4_weights {
    ($channels: expr, $w0: expr, $w1: expr, $w2: expr, $w3: expr, $dst: expr, $offset:expr) => {{
        unsafe {
            $dst.write($offset, $w0.to_());
            if $channels > 1 {
                $dst.write($offset + 1, $w1.to_());
            }
            if $channels > 2 {
                $dst.write($offset + 2, $w2.to_());
            }
            if $channels == 4 {
                $dst.write($offset + 3, $w3.to_());
            }
        }
    }};
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gaussian_blur_horizontal_pass_impl_c<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static
        + AsPrimitive<f32>,
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
    f32: AsPrimitive<T> + ToStorage<T>,
{
    let edge_mode: EdgeMode = EDGE_MODE.into();
    let half_kernel = (kernel_size / 2) as i32;
    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(4)).step_by(4) {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let (mut w0, mut w1, mut w2, mut w3) = (0f32, 0f32, 0f32, 0f32);
            let (mut w4, mut w5, mut w6, mut w7) = (0f32, 0f32, 0f32, 0f32);
            let (mut w8, mut w9, mut w10, mut w11) = (0f32, 0f32, 0f32, 0f32);
            let (mut w12, mut w13, mut w14, mut w15) = (0f32, 0f32, 0f32, 0f32);
            for r in -half_kernel..=half_kernel {
                let px = clamp_edge!(edge_mode, x as i64 + r as i64, 0, (width - 1) as i64)
                    * CHANNEL_CONFIGURATION;
                let y_offset = y_src_shift + px;
                let y_offset_1 = y_offset + src_stride as usize;
                let y_offset_2 = y_offset_1 + src_stride as usize;
                let y_offset_3 = y_offset_2 + src_stride as usize;

                let weight = unsafe { *kernel.get_unchecked((r + half_kernel) as usize) };
                accumulate_4_c_u8!(CHANNEL_CONFIGURATION, w0, w1, w2, w3, weight, src, y_offset);
                accumulate_4_c_u8!(
                    CHANNEL_CONFIGURATION,
                    w4,
                    w5,
                    w6,
                    w7,
                    weight,
                    src,
                    y_offset_1
                );
                accumulate_4_c_u8!(
                    CHANNEL_CONFIGURATION,
                    w8,
                    w9,
                    w10,
                    w11,
                    weight,
                    src,
                    y_offset_2
                );
                accumulate_4_c_u8!(
                    CHANNEL_CONFIGURATION,
                    w12,
                    w13,
                    w14,
                    w15,
                    weight,
                    src,
                    y_offset_3
                );
            }

            let px = x as usize * CHANNEL_CONFIGURATION;
            let bytes_offset = y_dst_shift + px;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w0,
                w1,
                w2,
                w3,
                unsafe_dst,
                bytes_offset
            );
            let bytes_offset_1 = bytes_offset + dst_stride as usize;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w4,
                w5,
                w6,
                w7,
                unsafe_dst,
                bytes_offset_1
            );
            let bytes_offset_2 = bytes_offset_1 + dst_stride as usize;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w8,
                w9,
                w10,
                w11,
                unsafe_dst,
                bytes_offset_2
            );
            let bytes_offset_3 = bytes_offset_2 + dst_stride as usize;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w12,
                w13,
                w14,
                w15,
                unsafe_dst,
                bytes_offset_3
            );
        }

        _cy = y;
    }

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let (mut w0, mut w1, mut w2, mut w3) = (0f32, 0f32, 0f32, 0f32);
            let (mut w4, mut w5, mut w6, mut w7) = (0f32, 0f32, 0f32, 0f32);
            for r in -half_kernel..=half_kernel {
                let px = clamp_edge!(edge_mode, x as i64 + r as i64, 0, (width - 1) as i64)
                    * CHANNEL_CONFIGURATION;
                let y_offset = y_src_shift + px;
                let y_offset_next = y_offset + src_stride as usize;

                let weight = unsafe { *kernel.get_unchecked((r + half_kernel) as usize) };
                accumulate_4_c_u8!(CHANNEL_CONFIGURATION, w0, w1, w2, w3, weight, src, y_offset);
                accumulate_4_c_u8!(
                    CHANNEL_CONFIGURATION,
                    w4,
                    w5,
                    w6,
                    w7,
                    weight,
                    src,
                    y_offset_next
                );
            }

            let px = x as usize * CHANNEL_CONFIGURATION;
            let bytes_offset = y_dst_shift + px;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w0,
                w1,
                w2,
                w3,
                unsafe_dst,
                bytes_offset
            );
            let bytes_offset_1 = bytes_offset + dst_stride as usize;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w4,
                w5,
                w6,
                w7,
                unsafe_dst,
                bytes_offset_1
            );
        }

        _cy = y;
    }

    for y in _cy..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let (mut w0, mut w1, mut w2, mut w3) = (0f32, 0f32, 0f32, 0f32);
            for r in -half_kernel..=half_kernel {
                let px = clamp_edge!(edge_mode, x as i64 + r as i64, 0, (width - 1) as i64)
                    * CHANNEL_CONFIGURATION;
                let y_offset = y_src_shift + px;
                let weight = unsafe { *kernel.get_unchecked((r + half_kernel) as usize) };
                accumulate_4_c_u8!(CHANNEL_CONFIGURATION, w0, w1, w2, w3, weight, src, y_offset);
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

            let bytes_offset = y_dst_shift + px;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w0,
                w1,
                w2,
                w3,
                unsafe_dst,
                bytes_offset
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gaussian_blur_horizontal_pass_impl_clip_edge<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static
        + AsPrimitive<f32>,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &[GaussianFilter<f32>],
    start_y: u32,
    end_y: u32,
) where
    f32: AsPrimitive<T> + ToStorage<T>,
{
    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(4)).step_by(4) {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let (mut w0, mut w1, mut w2, mut w3) = (0f32, 0f32, 0f32, 0f32);
            let (mut w4, mut w5, mut w6, mut w7) = (0f32, 0f32, 0f32, 0f32);
            let (mut w8, mut w9, mut w10, mut w11) = (0f32, 0f32, 0f32, 0f32);
            let (mut w12, mut w13, mut w14, mut w15) = (0f32, 0f32, 0f32, 0f32);

            let current_filter = unsafe { filter.get_unchecked(x as usize) };
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            for j in 0..current_filter.size {
                let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                let weight = unsafe { *filter_weights.get_unchecked(j) };
                let y_offset = y_src_shift + px;
                let y_offset_1 = y_offset + src_stride as usize;
                let y_offset_2 = y_offset_1 + src_stride as usize;
                let y_offset_3 = y_offset_2 + src_stride as usize;

                accumulate_4_c_u8!(CHANNEL_CONFIGURATION, w0, w1, w2, w3, weight, src, y_offset);
                accumulate_4_c_u8!(
                    CHANNEL_CONFIGURATION,
                    w4,
                    w5,
                    w6,
                    w7,
                    weight,
                    src,
                    y_offset_1
                );
                accumulate_4_c_u8!(
                    CHANNEL_CONFIGURATION,
                    w8,
                    w9,
                    w10,
                    w11,
                    weight,
                    src,
                    y_offset_2
                );
                accumulate_4_c_u8!(
                    CHANNEL_CONFIGURATION,
                    w12,
                    w13,
                    w14,
                    w15,
                    weight,
                    src,
                    y_offset_3
                );
            }

            let px = x as usize * CHANNEL_CONFIGURATION;
            let bytes_offset = y_dst_shift + px;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w0,
                w1,
                w2,
                w3,
                unsafe_dst,
                bytes_offset
            );
            let bytes_offset_1 = bytes_offset + dst_stride as usize;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w4,
                w5,
                w6,
                w7,
                unsafe_dst,
                bytes_offset_1
            );
            let bytes_offset_2 = bytes_offset_1 + dst_stride as usize;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w8,
                w9,
                w10,
                w11,
                unsafe_dst,
                bytes_offset_2
            );
            let bytes_offset_3 = bytes_offset_2 + dst_stride as usize;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w12,
                w13,
                w14,
                w15,
                unsafe_dst,
                bytes_offset_3
            );
        }
        _cy = y;
    }

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let (mut w0, mut w1, mut w2, mut w3) = (0f32, 0f32, 0f32, 0f32);
            let (mut w4, mut w5, mut w6, mut w7) = (0f32, 0f32, 0f32, 0f32);

            let current_filter = unsafe { filter.get_unchecked(x as usize) };
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            for j in 0..current_filter.size {
                let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                let weight = unsafe { *filter_weights.get_unchecked(j) };
                let y_offset = y_src_shift + px;
                let y_offset_next = y_offset + src_stride as usize;

                accumulate_4_c_u8!(CHANNEL_CONFIGURATION, w0, w1, w2, w3, weight, src, y_offset);
                accumulate_4_c_u8!(
                    CHANNEL_CONFIGURATION,
                    w4,
                    w5,
                    w6,
                    w7,
                    weight,
                    src,
                    y_offset_next
                );
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

            let bytes_offset = y_dst_shift + px;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w0,
                w1,
                w2,
                w3,
                unsafe_dst,
                bytes_offset
            );
            let bytes_offset = bytes_offset + dst_stride as usize;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w4,
                w5,
                w6,
                w7,
                unsafe_dst,
                bytes_offset
            );
        }
        _cy = y;
    }

    for y in _cy..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;
        for x in 0..width {
            let (mut w0, mut w1, mut w2, mut w3) = (0f32, 0f32, 0f32, 0f32);

            let current_filter = unsafe { filter.get_unchecked(x as usize) };
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            for j in 0..current_filter.size {
                let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                let weight = unsafe { *filter_weights.get_unchecked(j) };
                let y_offset = y_src_shift + px;
                accumulate_4_c_u8!(CHANNEL_CONFIGURATION, w0, w1, w2, w3, weight, src, y_offset);
            }

            let px = x as usize * CHANNEL_CONFIGURATION;

            let bytes_offset = y_dst_shift + px;
            save_4_weights!(
                CHANNEL_CONFIGURATION,
                w0,
                w1,
                w2,
                w3,
                unsafe_dst,
                bytes_offset
            );
        }
    }
}
