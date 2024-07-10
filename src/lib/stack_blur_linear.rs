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
use colorutils_rs::linear_to_planar::linear_to_plane;
use colorutils_rs::planar_to_linear::plane_to_linear;
use colorutils_rs::{
    linear_to_rgb, linear_to_rgba, rgb_to_linear, rgba_to_linear, TransferFunction,
};
use std::mem::size_of;

/// Stack blur that will be performed in linear color space
///
/// Blurs in linear color space produces most pleasant results and mathematically correct to do so however significantly slower.
/// This method is significantly slower than regular `u8` stack blur in perceptual colorspace.
/// This is a very fast approximation using f32 accumulator size with radius less that *BASE_RADIUS_F64_CUTOFF*,
/// after it to avoid overflowing fallback to f64 accumulator will be used with some computational slowdown with factor ~1.5-2
///
/// # Arguments
/// * `in_place` - mutable buffer contains image data that will be used as a source and destination
/// * `stride` - Bytes per lane, default is width * channels_count if not aligned
/// * `width` - image width
/// * `height` - image height
/// * `radius` - since f32 accumulator is used under the hood radius almost is not limited
/// * `channels` - Count of channels of the image, only 3 and 4 is supported, alpha position, and channels order does not matter
/// * `threading_policy` - Threads usage policy
/// * `transfer_function` - Transfer function in linear colorspace
///
/// # Complexity
/// O(1) complexity.
pub fn stack_blur_in_linear(
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
        FastBlurChannels::Plane => plane_to_linear,
        FastBlurChannels::Channels3 => rgb_to_linear,
        FastBlurChannels::Channels4 => rgba_to_linear,
    };

    let inverse_transformer = match channels {
        FastBlurChannels::Plane => linear_to_plane,
        FastBlurChannels::Channels3 => linear_to_rgb,
        FastBlurChannels::Channels4 => linear_to_rgba,
    };

    forward_transformer(
        &in_place,
        stride,
        &mut linear_data,
        width * size_of::<f32>() as u32 * channels.get_channels() as u32,
        width,
        height,
        transfer_function,
    );

    crate::stack_blur_f32(
        &mut linear_data,
        width,
        height,
        radius,
        channels,
        threading_policy,
    );

    inverse_transformer(
        &linear_data,
        width * size_of::<f32>() as u32 * channels.get_channels() as u32,
        in_place,
        stride,
        width,
        height,
        transfer_function,
    );
}
