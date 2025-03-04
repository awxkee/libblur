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

use crate::{BlurError, BlurImageMut, FastBlurChannels, ThreadingPolicy};
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
/// * `image` - mutable buffer contains image data that will be used as a source and destination.
/// * `radius` - since f32 accumulator is used under the hood radius almost is not limited.
/// * `threading_policy` - Threads usage policy.
/// * `transfer_function` - Transfer function in linear colorspace.
///
/// # Complexity
/// O(1) complexity.
#[allow(clippy::too_many_arguments)]
pub fn stack_blur_in_linear(
    image: &mut BlurImageMut<u8>,
    radius: u32,
    threading_policy: ThreadingPolicy,
    transfer_function: TransferFunction,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let radius = radius.max(1);
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;

    let mut linear_image = BlurImageMut::<f32>::alloc(width, height, channels);

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
        image.data.borrow_mut(),
        stride,
        linear_image.data.borrow_mut(),
        width * size_of::<f32>() as u32 * channels.channels() as u32,
        width,
        height,
        transfer_function,
    );

    crate::stack_blur_f32(&mut linear_image, radius, threading_policy)?;

    inverse_transformer(
        linear_image.data.borrow(),
        width * size_of::<f32>() as u32 * channels.channels() as u32,
        image.data.borrow_mut(),
        stride,
        width,
        height,
        transfer_function,
    );
    Ok(())
}
