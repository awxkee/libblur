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
    filter_1d_exact, gaussian_kernel_1d_f64, sigma_size_d, BlurImage, BlurImageMut, EdgeMode,
    FastBlurChannels, Scalar, ThreadingPolicy,
};
use libfuzzer_sys::fuzz_target;

#[derive(Clone, Debug, Arbitrary)]
pub struct SrcImage {
    pub src_width: u16,
    pub src_height: u16,
    pub value: u8,
    pub edge_mode: u8,
    pub channels: u8,
    pub kernel_size: u8,
    pub double_precision: bool,
}

fuzz_target!(|data: SrcImage| {
    if data.src_width > 250 || data.src_height > 250 {
        return;
    }
    if data.kernel_size % 2 == 0 || data.kernel_size > 45 || data.kernel_size == 0 {
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
        data.kernel_size as usize,
        channels,
        edge_mode,
        data.double_precision,
    );
});

fn fuzz_f32(
    width: usize,
    height: usize,
    kernel_size: usize,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    double_precision: bool,
) {
    if width == 0 || height == 0 || kernel_size == 0 {
        return;
    }
    let src_image = BlurImage::alloc(width as u32, height as u32, channels);
    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);
    if double_precision {
        let kernel = gaussian_kernel_1d_f64(kernel_size as u32, sigma_size_d(kernel_size as f64));

        match channels {
            FastBlurChannels::Plane => {
                filter_1d_exact::<f32, f64, 1>(
                    &src_image,
                    &mut dst_image,
                    &kernel,
                    &kernel,
                    EdgeMode::Clamp,
                    Scalar::default(),
                    ThreadingPolicy::Single,
                )
                .unwrap();
            }
            FastBlurChannels::Channels3 => {
                filter_1d_exact::<f32, f64, 3>(
                    &src_image,
                    &mut dst_image,
                    &kernel,
                    &kernel,
                    EdgeMode::Clamp,
                    Scalar::default(),
                    ThreadingPolicy::Single,
                )
                .unwrap();
            }
            FastBlurChannels::Channels4 => {
                filter_1d_exact::<f32, f64, 4>(
                    &src_image,
                    &mut dst_image,
                    &kernel,
                    &kernel,
                    EdgeMode::Clamp,
                    Scalar::default(),
                    ThreadingPolicy::Single,
                )
                .unwrap();
            }
        }
    } else {
        libblur::gaussian_blur_f32(
            &src_image,
            &mut dst_image,
            kernel_size as u32,
            0.,
            edge_mode,
            ThreadingPolicy::Single,
        )
        .unwrap();
    }
}
