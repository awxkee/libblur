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
    filter_1d_approx, filter_1d_exact, filter_1d_rgb_approx, filter_1d_rgb_exact,
    filter_1d_rgba_approx, filter_1d_rgba_exact, get_gaussian_kernel_1d, get_sigma_size, EdgeMode,
    FastBlurChannels, GaussianPreciseLevel, ImageSize, Scalar, ThreadingPolicy,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (u8, u8, u8)| {
    fuzz_8bit(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels4,
    );
    fuzz_8bit_non_symmetry(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels4,
    );
    fuzz_8bit(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels3,
    );
    fuzz_8bit_non_symmetry(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels3,
    );
    fuzz_8bit(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Plane,
    );
    fuzz_8bit_non_symmetry(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Plane,
    );
});

fn fuzz_8bit(width: usize, height: usize, radius: usize, channels: FastBlurChannels) {
    if width == 0 || height == 0 || radius == 0 {
        return;
    }
    let src_image = vec![15u8; width * height * channels.get_channels()];
    let mut dst_image = vec![0u8; width * height * channels.get_channels()];

    libblur::gaussian_blur(
        &src_image,
        &mut dst_image,
        width as u32,
        height as u32,
        radius as u32 * 2 + 1,
        0.,
        channels,
        EdgeMode::Clamp,
        ThreadingPolicy::Single,
        GaussianPreciseLevel::INTEGRAL,
    );

    libblur::gaussian_blur(
        &src_image,
        &mut dst_image,
        width as u32,
        height as u32,
        radius as u32 * 2 + 1,
        0.,
        channels,
        EdgeMode::Clamp,
        ThreadingPolicy::Single,
        GaussianPreciseLevel::EXACT,
    );
}

fn fuzz_8bit_non_symmetry(width: usize, height: usize, radius: usize, channels: FastBlurChannels) {
    if width == 0 || height == 0 || radius == 0 {
        return;
    }
    let src_image = vec![15u8; width * height * channels.get_channels()];
    let mut dst_image = vec![0u8; width * height * channels.get_channels()];

    let kernel_size = radius * 2 + 1;

    let kernel = get_gaussian_kernel_1d(kernel_size as u32, get_sigma_size(kernel_size));

    match channels {
        FastBlurChannels::Plane => {
            filter_1d_exact(
                &src_image,
                &mut dst_image,
                ImageSize::new(width, height),
                &kernel,
                &kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
            filter_1d_approx::<u8, f32, i32>(
                &src_image,
                &mut dst_image,
                ImageSize::new(width, height),
                &kernel,
                &kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        }
        FastBlurChannels::Channels3 => {
            filter_1d_rgb_exact(
                &src_image,
                &mut dst_image,
                ImageSize::new(width, height),
                &kernel,
                &kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
            filter_1d_rgb_approx::<u8, f32, i32>(
                &src_image,
                &mut dst_image,
                ImageSize::new(width, height),
                &kernel,
                &kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        }
        FastBlurChannels::Channels4 => {
            filter_1d_rgba_exact(
                &src_image,
                &mut dst_image,
                ImageSize::new(width, height),
                &kernel,
                &kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
            filter_1d_rgba_approx::<u8, f32, i32>(
                &src_image,
                &mut dst_image,
                ImageSize::new(width, height),
                &kernel,
                &kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        }
    }
}
