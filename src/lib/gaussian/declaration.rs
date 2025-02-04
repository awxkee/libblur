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

use crate::channels_configuration::FastBlurChannels;
use crate::edge_mode::EdgeMode;
use crate::gaussian::gaussian_kernel::get_gaussian_kernel_1d;
use crate::gaussian::gaussian_precise_level::GaussianPreciseLevel;
use crate::{
    filter_1d_approx, filter_1d_exact, filter_1d_rgb_approx, filter_1d_rgb_exact,
    filter_1d_rgba_approx, filter_1d_rgba_exact, get_sigma_size, ImageSize, Scalar,
    ThreadingPolicy,
};
use half::f16;
use crate::gaussian::gaussian_util::get_kernel_size;

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// O(R) complexity.
///
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count * size_of(PixelType) if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted. If zero, then sigma must be set.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
/// * `precise_level` - Gaussian precise level
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
#[allow(clippy::too_many_arguments)]
pub fn gaussian_blur(
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
    precise_level: GaussianPreciseLevel,
) {
    assert!(kernel_size != 0 || sigma > 0.0, "Either sigma or kernel size must be set");
    if kernel_size != 0 {
        assert_ne!(kernel_size % 2, 0, "Kernel size must be odd");
    }
    let sigma = if sigma <= 0. {
        get_sigma_size(kernel_size as usize)
    } else {
        sigma
    };
    let kernel_size = if kernel_size == 0 {
        get_kernel_size(sigma)
    } else {
        kernel_size
    };
    let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
    match precise_level {
        GaussianPreciseLevel::EXACT => {
            let _dispatcher = match channels {
                FastBlurChannels::Plane => filter_1d_exact::<u8, f32>,
                FastBlurChannels::Channels3 => filter_1d_rgb_exact::<u8, f32>,
                FastBlurChannels::Channels4 => filter_1d_rgba_exact::<u8, f32>,
            };
            _dispatcher(
                src,
                dst,
                ImageSize::new(width as usize, height as usize),
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
        GaussianPreciseLevel::INTEGRAL => {
            let _dispatcher = match channels {
                FastBlurChannels::Plane => filter_1d_approx::<u8, f32, i32>,
                FastBlurChannels::Channels3 => filter_1d_rgb_approx::<u8, f32, i32>,
                FastBlurChannels::Channels4 => filter_1d_rgba_approx::<u8, f32, i32>,
            };
            _dispatcher(
                src,
                dst,
                ImageSize::new(width as usize, height as usize),
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
    }
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// O(R) complexity.
///
/// # Arguments
///
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted. If zero, then sigma must be set.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_blur_u16(
    src: &[u16],
    dst: &mut [u16],
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) {
    assert!(kernel_size != 0 || sigma > 0.0, "Either sigma or kernel size must be set");
    if kernel_size != 0 {
        assert_ne!(kernel_size % 2, 0, "Kernel size must be odd");
    }
    let sigma = if sigma <= 0. {
        get_sigma_size(kernel_size as usize)
    } else {
        sigma
    };
    let kernel_size = if kernel_size == 0 {
        get_kernel_size(sigma)
    } else {
        kernel_size
    };
    let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
    let _dispatcher = match channels {
        FastBlurChannels::Plane => filter_1d_exact::<u16, f32>,
        FastBlurChannels::Channels3 => filter_1d_rgb_exact::<u16, f32>,
        FastBlurChannels::Channels4 => filter_1d_rgba_exact::<u16, f32>,
    };
    _dispatcher(
        src,
        dst,
        ImageSize::new(width as usize, height as usize),
        &kernel,
        &kernel,
        edge_mode,
        Scalar::default(),
        threading_policy,
    )
    .unwrap();
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// O(R) complexity.
///
/// # Arguments
///
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted. If zero, then sigma must be set.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
#[allow(clippy::too_many_arguments)]
pub fn gaussian_blur_f32(
    src: &[f32],
    dst: &mut [f32],
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) {
    assert!(kernel_size != 0 || sigma > 0.0, "Either sigma or kernel size must be set");
    if kernel_size != 0 {
        assert_ne!(kernel_size % 2, 0, "Kernel size must be odd");
    }
    let sigma = if sigma <= 0. {
        get_sigma_size(kernel_size as usize)
    } else {
        sigma
    };
    let kernel_size = if kernel_size == 0 {
        get_kernel_size(sigma)
    } else {
        kernel_size
    };
    let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
    let _dispatcher = match channels {
        FastBlurChannels::Plane => filter_1d_exact::<f32, f32>,
        FastBlurChannels::Channels3 => filter_1d_rgb_exact::<f32, f32>,
        FastBlurChannels::Channels4 => filter_1d_rgba_exact::<f32, f32>,
    };
    _dispatcher(
        src,
        dst,
        ImageSize::new(width as usize, height as usize),
        &kernel,
        &kernel,
        edge_mode,
        Scalar::default(),
        threading_policy,
    )
    .unwrap();
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// O(R) complexity.
///
/// # Arguments
///
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted. If zero, then sigma must be set.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_blur_f16(
    src: &[f16],
    dst: &mut [f16],
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) {
    assert!(kernel_size != 0 || sigma > 0.0, "Either sigma or kernel size must be set");
    if kernel_size != 0 {
        assert_ne!(kernel_size % 2, 0, "Kernel size must be odd");
    }
    let sigma = if sigma <= 0. {
        get_sigma_size(kernel_size as usize)
    } else {
        sigma
    };
    let kernel_size = if kernel_size == 0 {
        get_kernel_size(sigma)
    } else {
        kernel_size
    };
    let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
    let _dispatcher = match channels {
        FastBlurChannels::Plane => filter_1d_exact::<f16, f32>,
        FastBlurChannels::Channels3 => filter_1d_rgb_exact::<f16, f32>,
        FastBlurChannels::Channels4 => filter_1d_rgba_exact::<f16, f32>,
    };
    _dispatcher(
        src,
        dst,
        ImageSize::new(width as usize, height as usize),
        &kernel,
        &kernel,
        edge_mode,
        Scalar::default(),
        threading_policy,
    )
    .unwrap();
}
