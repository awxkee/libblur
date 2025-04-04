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

use libblur::{
    filter_1d_approx, filter_1d_exact, gaussian_kernel_1d, sigma_size, BlurImage, BlurImageMut,
    ConvolutionMode, EdgeMode, FastBlurChannels, Scalar, ThreadingPolicy,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (u8, u8, u8, u8)| {
    let edge_mode = match data.3 % 4 {
        0 => EdgeMode::Clamp,
        1 => EdgeMode::Wrap,
        2 => EdgeMode::Reflect,
        _ => EdgeMode::Reflect101,
    };
    fuzz_8bit(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels4,
        edge_mode,
    );
    fuzz_8bit_non_symmetry(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels4,
        edge_mode,
    );
    fuzz_8bit(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels3,
        edge_mode,
    );
    fuzz_8bit_non_symmetry(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels3,
        edge_mode,
    );
    fuzz_8bit(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Plane,
        edge_mode,
    );
    fuzz_8bit_non_symmetry(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Plane,
        edge_mode,
    );
});

fn fuzz_8bit(
    width: usize,
    height: usize,
    radius: usize,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
) {
    if width == 0 || height == 0 || radius == 0 {
        return;
    }
    let src_image = BlurImage::alloc(width as u32, height as u32, channels);
    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);

    libblur::gaussian_blur(
        &src_image,
        &mut dst_image,
        radius as u32 * 2 + 1,
        0.,
        edge_mode,
        ThreadingPolicy::Single,
        ConvolutionMode::FixedPoint,
    )
    .unwrap();

    libblur::gaussian_blur(
        &src_image,
        &mut dst_image,
        radius as u32 * 2 + 1,
        0.,
        edge_mode,
        ThreadingPolicy::Single,
        ConvolutionMode::Exact,
    )
    .unwrap();
}

fn fuzz_8bit_non_symmetry(
    width: usize,
    height: usize,
    radius: usize,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
) {
    if width == 0 || height == 0 || radius == 0 {
        return;
    }
    let src_image = BlurImage::alloc(width as u32, height as u32, channels);
    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);

    let kernel_size = radius * 2 + 1;

    let kernel = gaussian_kernel_1d(kernel_size as u32, sigma_size(kernel_size as f32));

    match channels {
        FastBlurChannels::Plane => {
            filter_1d_exact::<u8, f32, 1>(
                &src_image,
                &mut dst_image,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
            filter_1d_approx::<u8, f32, i32, 1>(
                &src_image,
                &mut dst_image,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        }
        FastBlurChannels::Channels3 => {
            filter_1d_exact::<u8, f32, 3>(
                &src_image,
                &mut dst_image,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
            filter_1d_approx::<u8, f32, i32, 3>(
                &src_image,
                &mut dst_image,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        }
        FastBlurChannels::Channels4 => {
            filter_1d_exact::<u8, f32, 4>(
                &src_image,
                &mut dst_image,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
            filter_1d_approx::<u8, f32, i32, 4>(
                &src_image,
                &mut dst_image,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        }
    }
}
