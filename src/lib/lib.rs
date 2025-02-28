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
#![allow(clippy::too_many_arguments)]
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

#[cfg(feature = "fft")]
mod adaptive_blur;
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
mod avx;
mod box_filter;
mod channels_configuration;
mod cpu_features;
mod edge_mode;
mod fast_bilateral_filter;
#[cfg(feature = "image")]
mod fast_bilateral_image;
mod fast_gaussian;
#[cfg(feature = "image")]
mod fast_gaussian_image;
#[cfg(feature = "image")]
mod fast_gaussian_image_next;
mod fast_gaussian_next;
mod fast_gaussian_superior;
mod filter1d;
mod filter2d;
mod gaussian;
#[cfg(feature = "image")]
mod gaussian_blur_image;
mod image;
mod img_size;
mod laplacian;
mod median_blur;
mod mlaf;
mod motion_blur;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod sobel;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod sse;
#[cfg(feature = "image")]
mod stack_blur_image;
mod stack_blur_linear;
mod stackblur;
mod threading_policy;
mod to_storage;
mod unsafe_slice;
mod util;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;

#[cfg(feature = "fft")]
pub use adaptive_blur::adaptive_blur;
pub use box_filter::box_blur;
pub use box_filter::box_blur_f32;
pub use box_filter::box_blur_in_linear;
pub use box_filter::box_blur_u16;
pub use box_filter::gaussian_box_blur;
pub use box_filter::gaussian_box_blur_f32;
pub use box_filter::gaussian_box_blur_in_linear;
pub use box_filter::gaussian_box_blur_u16;
pub use box_filter::tent_blur;
pub use box_filter::tent_blur_f32;
pub use box_filter::tent_blur_in_linear;
pub use box_filter::tent_blur_u16;
pub use channels_configuration::FastBlurChannels;
pub use colorutils_rs::TransferFunction;
pub use edge_mode::*;
pub use fast_bilateral_filter::{
    fast_bilateral_filter, fast_bilateral_filter_f32, fast_bilateral_filter_u16,
};
#[cfg(feature = "image")]
pub use fast_bilateral_image::fast_bilateral_filter_image;
pub use fast_gaussian::{
    fast_gaussian, fast_gaussian_f16, fast_gaussian_f32, fast_gaussian_in_linear, fast_gaussian_u16,
};
#[cfg(feature = "image")]
pub use fast_gaussian_image::fast_gaussian_blur_image;
#[cfg(feature = "image")]
pub use fast_gaussian_image_next::fast_gaussian_next_blur_image;
pub use fast_gaussian_next::fast_gaussian_next;
pub use fast_gaussian_next::fast_gaussian_next_f16;
pub use fast_gaussian_next::fast_gaussian_next_f32;
pub use fast_gaussian_next::fast_gaussian_next_in_linear;
pub use fast_gaussian_next::fast_gaussian_next_u16;
pub use fast_gaussian_superior::fast_gaussian_superior;
pub use filter1d::{
    filter_1d_approx, filter_1d_exact, filter_1d_rgb_approx, filter_1d_rgb_exact,
    filter_1d_rgba_approx, filter_1d_rgba_exact, make_arena, Arena, ArenaPads, KernelShape,
};
#[cfg(feature = "fft")]
pub use filter2d::{fft_next_good_size, filter_2d_fft, filter_2d_rgb_fft, filter_2d_rgba_fft};
pub use filter2d::{filter_2d, filter_2d_arbitrary, filter_2d_rgb, filter_2d_rgba};
pub use gaussian::gaussian_blur;
pub use gaussian::gaussian_blur_f16;
pub use gaussian::gaussian_blur_f32;
pub use gaussian::gaussian_blur_in_linear;
pub use gaussian::gaussian_blur_u16;
pub use gaussian::gaussian_kernel_1d;
pub use gaussian::sigma_size;
pub use gaussian::ConvolutionMode;
#[cfg(feature = "image")]
pub use gaussian_blur_image::gaussian_blur_image;
pub use image::{BlurImage, BlurImageMut, BufferStore};
pub use img_size::ImageSize;
pub use laplacian::{laplacian, laplacian_kernel};
pub use median_blur::median_blur;
pub use motion_blur::{generate_motion_kernel, motion_blur};
pub use sobel::sobel;
#[cfg(feature = "image")]
pub use stack_blur_image::stack_blur_image;
pub use stack_blur_linear::stack_blur_in_linear;
pub use stackblur::stack_blur::stack_blur;
pub use stackblur::stack_blur_f16::stack_blur_f16;
pub use stackblur::stack_blur_f32::stack_blur_f32;
pub use stackblur::stack_blur_u16;
pub use threading_policy::ThreadingPolicy;
pub use util::{BlurError, MismatchedSize};
