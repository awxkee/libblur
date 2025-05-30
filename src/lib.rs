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
#![allow(clippy::too_many_arguments, clippy::int_plus_one, stable_features)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    all(feature = "nightly_fcma", target_arch = "aarch64"),
    feature(stdarch_neon_fcma)
)]
#![cfg_attr(
    all(
        feature = "nightly_avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    feature(cfg_version)
)]
#![cfg_attr(
    all(
        feature = "nightly_avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    feature(avx512_target_feature)
)]
#![cfg_attr(
    all(
        feature = "nightly_avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    feature(stdarch_x86_avx512)
)]
#![cfg_attr(
    all(
        feature = "nightly_avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    feature(x86_amx_intrinsics)
)]

#[cfg(feature = "fft")]
#[cfg_attr(docsrs, doc(cfg(feature = "fft")))]
mod adaptive_blur;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod box_filter;
mod channels_configuration;
mod edge_mode;
mod fast_bilateral_filter;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod fast_bilateral_image;
mod fast_gaussian;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod fast_gaussian_image;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod fast_gaussian_image_next;
mod fast_gaussian_next;
mod filter1d;
mod filter2d;
mod gamma_curves;
mod gaussian;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod gaussian_blur_image;
mod image;
mod image_linearization;
mod img_size;
mod laplacian;
mod lens;
mod median_blur;
mod mlaf;
mod motion_blur;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod safe_math;
mod sobel;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod sse;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod stack_blur_image;
mod stackblur;
mod threading_policy;
mod to_storage;
mod unsafe_slice;
mod util;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;

#[cfg(feature = "fft")]
#[cfg_attr(docsrs, doc(cfg(feature = "fft")))]
pub use adaptive_blur::adaptive_blur;
pub use box_filter::{
    box_blur, box_blur_f32, box_blur_u16, gaussian_box_blur, gaussian_box_blur_f32,
    gaussian_box_blur_u16, tent_blur, tent_blur_f32, tent_blur_u16, BoxBlurParameters,
    CLTParameters,
};
pub use channels_configuration::FastBlurChannels;
pub use edge_mode::*;
pub use fast_bilateral_filter::{
    fast_bilateral_filter, fast_bilateral_filter_f32, fast_bilateral_filter_u16,
};
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use fast_bilateral_image::fast_bilateral_filter_image;
pub use fast_gaussian::{fast_gaussian, fast_gaussian_f16, fast_gaussian_f32, fast_gaussian_u16};
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use fast_gaussian_image::fast_gaussian_blur_image;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use fast_gaussian_image_next::fast_gaussian_next_blur_image;
pub use fast_gaussian_next::{
    fast_gaussian_next, fast_gaussian_next_f16, fast_gaussian_next_f32, fast_gaussian_next_u16,
};
pub use filter1d::{
    filter_1d_approx, filter_1d_complex, filter_1d_exact, make_arena, Arena, ArenaPads, KernelShape,
};
#[cfg(feature = "fft")]
#[cfg_attr(docsrs, doc(cfg(feature = "fft")))]
pub use filter2d::{
    fft_next_good_size, filter_2d_fft, filter_2d_fft_complex, filter_2d_rgb_fft,
    filter_2d_rgb_fft_complex, filter_2d_rgba_fft, filter_2d_rgba_fft_complex,
};
pub use filter2d::{filter_2d, filter_2d_arbitrary, filter_2d_rgb, filter_2d_rgba};
pub use gamma_curves::TransferFunction;
pub use gaussian::{
    gaussian_blur, gaussian_blur_f16, gaussian_blur_f32, gaussian_blur_u16, gaussian_kernel_1d,
    gaussian_kernel_1d_f64, sigma_size, sigma_size_d, ConvolutionMode, GaussianBlurParams,
    IeeeBinaryConvolutionMode,
};
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use gaussian_blur_image::gaussian_blur_image;
pub use image::{BlurImage, BlurImageMut, BufferStore};
pub use img_size::ImageSize;
pub use laplacian::{laplacian, laplacian_kernel};
pub use lens::lens_kernel;
pub use median_blur::median_blur;
pub use motion_blur::{generate_motion_kernel, motion_blur};
pub use sobel::sobel;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use stack_blur_image::stack_blur_image;
pub use stackblur::stack_blur::stack_blur;
pub use stackblur::stack_blur_f16::stack_blur_f16;
pub use stackblur::stack_blur_f32::stack_blur_f32;
pub use stackblur::stack_blur_u16;
pub use threading_policy::ThreadingPolicy;
pub use util::{BlurError, MismatchedSize};

/// Asymmetric radius container
#[derive(Copy, Clone, Default, PartialOrd, PartialEq, Debug)]
pub struct AnisotropicRadius {
    pub x_axis: u32,
    pub y_axis: u32,
}

impl AnisotropicRadius {
    pub fn new(radius: u32) -> AnisotropicRadius {
        AnisotropicRadius {
            x_axis: radius,
            y_axis: radius,
        }
    }

    pub fn create(x: u32, y: u32) -> AnisotropicRadius {
        AnisotropicRadius {
            x_axis: x,
            y_axis: y,
        }
    }

    pub fn clamp(&self, min: u32, max: u32) -> AnisotropicRadius {
        AnisotropicRadius {
            x_axis: self.x_axis.clamp(min, max),
            y_axis: self.y_axis.clamp(min, max),
        }
    }

    pub fn max(&self, max: u32) -> AnisotropicRadius {
        AnisotropicRadius {
            x_axis: self.x_axis.max(max),
            y_axis: self.y_axis.max(max),
        }
    }
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "fft")]
    fn gaussian_kernel_9x9(sigma: f32) -> Vec<f32> {
        let mut kernel = [[0.0f32; 9]; 9];
        let mut sum = 0.0;

        let radius = 4; // (9 - 1) / 2

        let two_sigma_sq = 2.0 * sigma * sigma;
        let norm = 1.0 / (std::f32::consts::PI * two_sigma_sq);

        for y in -radius..=radius {
            for x in -radius..=radius {
                let value = norm * ((-(x * x + y * y) as f32) / two_sigma_sq).exp();
                kernel[(y + radius) as usize][(x + radius) as usize] = value;
                sum += value;
            }
        }

        // Normalize
        for row in &mut kernel {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }

        kernel
            .iter()
            .flat_map(|x| x)
            .map(|&x| x)
            .collect::<Vec<f32>>()
    }

    #[cfg(feature = "fft")]
    #[test]
    fn test_fft_rgb() {
        use super::*;
        let width: usize = 188;
        let height: usize = 188;
        let src = vec![126u8; width * height * 3];
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();

        let kernel = gaussian_kernel_9x9(3.);

        filter_2d_rgb_fft::<u8, f32, f32>(
            &src_image,
            &mut dst,
            &kernel,
            KernelShape::new(9, 9),
            EdgeMode::Clamp,
            Scalar::default(),
            ThreadingPolicy::Single,
        )
        .unwrap();
        for (i, &cn) in dst.data.borrow_mut().iter().enumerate() {
            let diff = (cn as i32 - 126).abs();
            assert!(
                diff <= 3,
                "Diff expected to be less than 3 but it was {diff} at {i}"
            );
        }
    }
}
