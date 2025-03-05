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

use crate::{
    gaussian_blur_f32, sigma_size, BlurError, BlurImage, BlurImageMut, EdgeMode, FastBlurChannels,
    ThreadingPolicy,
};
use colorutils_rs::linear_to_planar::linear_to_plane;
use colorutils_rs::planar_to_linear::plane_to_linear;
use colorutils_rs::{
    linear_to_rgb, linear_to_rgba, rgb_to_linear, rgba_to_linear, TransferFunction,
};
use std::mem::size_of;

/// Performs gaussian blur on the image in linear colorspace
///
/// This is performing gaussian blur in linear colorspace, it is mathematically correct to do so.
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// O(R) complexity.
///
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count * size_of(PixelType) if not aligned
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
/// * `transfer_function` - Transfer function in linear colorspace
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_blur_in_linear(
    src: &BlurImage<u8>,
    dst: &mut BlurImageMut<u8>,
    kernel_size: u32,
    sigma: f32,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
    transfer_function: TransferFunction,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    let mut linear_data = BlurImageMut::alloc(src.width, src.height, src.channels);
    let mut linear_data_1 = BlurImageMut::alloc(src.width, src.height, src.channels);

    let forward_transformer = match src.channels {
        FastBlurChannels::Plane => plane_to_linear,
        FastBlurChannels::Channels3 => rgb_to_linear,
        FastBlurChannels::Channels4 => rgba_to_linear,
    };

    let inverse_transformer = match src.channels {
        FastBlurChannels::Plane => linear_to_plane,
        FastBlurChannels::Channels3 => linear_to_rgb,
        FastBlurChannels::Channels4 => linear_to_rgba,
    };

    let width = src.width;
    let height = src.height;
    let channels = src.channels;

    forward_transformer(
        src.data.as_ref(),
        src.row_stride(),
        linear_data.data.borrow_mut(),
        width * size_of::<f32>() as u32 * channels.channels() as u32,
        width,
        height,
        transfer_function,
    );

    let sigma = if sigma <= 0. {
        sigma_size(kernel_size as f32)
    } else {
        sigma
    };

    let lin_data = linear_data.to_immutable_ref();

    gaussian_blur_f32(
        &lin_data,
        &mut linear_data_1,
        kernel_size,
        sigma,
        edge_mode,
        threading_policy,
    )?;

    let dst_stride = dst.row_stride();

    inverse_transformer(
        linear_data_1.data.borrow(),
        width * size_of::<f32>() as u32 * channels.channels() as u32,
        dst.data.borrow_mut(),
        dst_stride,
        width,
        height,
        transfer_function,
    );
    Ok(())
}
