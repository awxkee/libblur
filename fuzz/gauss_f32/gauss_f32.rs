/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#![no_main]

use arbitrary::Arbitrary;
use libblur::{
    BlurImage, BlurImageMut, EdgeMode, EdgeMode2D, FastBlurChannels, GaussianBlurParams,
    IeeeBinaryConvolutionMode, ThreadingPolicy,
};
use libfuzzer_sys::fuzz_target;

#[derive(Clone, Debug, Arbitrary)]
pub struct SrcImage {
    pub src_width: u16,
    pub src_height: u16,
    pub value: u8,
    pub edge_mode: u8,
    pub channels: u8,
    pub x_kernel_size: u8,
    pub y_kernel_size: u8,
    pub double_precision: bool,
}

fuzz_target!(|data: SrcImage| {
    if data.src_width > 250 || data.src_height > 250 {
        return;
    }
    if data.x_kernel_size > 45 || data.x_kernel_size == 0 {
        return;
    }
    if data.y_kernel_size > 45 || data.y_kernel_size == 0 {
        return;
    }
    let edge_mode = match data.edge_mode % 4 {
        0 => EdgeMode::Clamp,
        1 => EdgeMode::Wrap,
        2 => EdgeMode::Reflect,
        _ => EdgeMode::Reflect101,
    };
    let channels = match data.channels % 3 {
        0 => FastBlurChannels::Channels4,
        1 => FastBlurChannels::Channels3,
        _ => FastBlurChannels::Plane,
    };
    fuzz_f32(
        data.src_width as usize,
        data.src_height as usize,
        data.x_kernel_size as usize,
        data.y_kernel_size as usize,
        channels,
        edge_mode,
        data.double_precision,
    );
});

fn fuzz_f32(
    width: usize,
    height: usize,
    x_kernel_size: usize,
    y_kernel_size: usize,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    double_precision: bool,
) {
    if width == 0 || height == 0 || x_kernel_size == 0 || y_kernel_size == 0 {
        return;
    }
    let src_image = BlurImage::alloc(width as u32, height as u32, channels);
    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);
    libblur::gaussian_blur_f32(
        &src_image,
        &mut dst_image,
        GaussianBlurParams::new_asymmetric_from_kernels(x_kernel_size as f64, y_kernel_size as f64),
        EdgeMode2D::new(edge_mode),
        ThreadingPolicy::Single,
        if double_precision {
            IeeeBinaryConvolutionMode::Zealous
        } else {
            IeeeBinaryConvolutionMode::Normal
        },
    )
    .unwrap();
}
