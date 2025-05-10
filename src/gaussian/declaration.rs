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
use crate::gaussian::gaussian_kernel::gaussian_kernel_1d;
use crate::gaussian::gaussian_util::kernel_size as get_kernel_size;
use crate::{
    filter_1d_approx, filter_1d_exact, sigma_size, BlurError, BlurImage, BlurImageMut,
    ConvolutionMode, Scalar, ThreadingPolicy,
};
use half::f16;

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// This is always perform accurate pure analytical gaussian filter.
/// O(R) complexity.
///
/// # Arguments
///
/// * `src` - Source image.
/// * `dst` - Destination image.
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted. If zero, then sigma must be set.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
/// * `hint` - see [ConvolutionMode] for more info
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided.
/// Panics if sigma = 0.8 and kernel size = 0.
#[allow(clippy::too_many_arguments)]
pub fn gaussian_blur(
    src: &BlurImage<u8>,
    dst: &mut BlurImageMut<u8>,
    kernel_size: u32,
    sigma: f32,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
    hint: ConvolutionMode,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    assert!(
        kernel_size != 0 || sigma > 0.0,
        "Either sigma or kernel size must be set"
    );
    if kernel_size != 0 {
        assert_ne!(kernel_size % 2, 0, "Kernel size must be odd");
    }
    let sigma = if sigma <= 0. {
        sigma_size(kernel_size as f32)
    } else {
        sigma
    };
    let kernel_size = if kernel_size == 0 {
        get_kernel_size(sigma)
    } else {
        kernel_size
    };
    let kernel = gaussian_kernel_1d(kernel_size, sigma);
    match hint {
        ConvolutionMode::Exact => {
            let _dispatcher = match src.channels {
                FastBlurChannels::Plane => filter_1d_exact::<u8, f32, 1>,
                FastBlurChannels::Channels3 => filter_1d_exact::<u8, f32, 3>,
                FastBlurChannels::Channels4 => filter_1d_exact::<u8, f32, 4>,
            };
            _dispatcher(
                src,
                dst,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )?;
        }
        ConvolutionMode::FixedPoint => {
            let _dispatcher = match src.channels {
                FastBlurChannels::Plane => filter_1d_approx::<u8, f32, i32, 1>,
                FastBlurChannels::Channels3 => filter_1d_approx::<u8, f32, i32, 3>,
                FastBlurChannels::Channels4 => filter_1d_approx::<u8, f32, i32, 4>,
            };
            _dispatcher(
                src,
                dst,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )?;
        }
    }
    Ok(())
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// This is always perform accurate pure analytical gaussian filter.
/// O(R) complexity.
///
/// # Arguments
///
/// * `src` - Source image.
/// * `dst` - Destination image.
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted. If zero, then sigma must be set.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
/// * `hint` - see [ConvolutionMode] for more info
///
/// This method always clamp into [0, 65535], if other bit-depth is used
/// consider additional clamp into required range.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided.
/// Panics if sigma = 0.8 and kernel size = 0.
pub fn gaussian_blur_u16(
    src: &BlurImage<u16>,
    dst: &mut BlurImageMut<u16>,
    kernel_size: u32,
    sigma: f32,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
    hint: ConvolutionMode,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    assert!(
        kernel_size != 0 || sigma > 0.0,
        "Either sigma or kernel size must be set"
    );
    if kernel_size != 0 {
        assert_ne!(kernel_size % 2, 0, "Kernel size must be odd");
    }
    let sigma = if sigma <= 0. {
        sigma_size(kernel_size as f32)
    } else {
        sigma
    };
    let kernel_size = if kernel_size == 0 {
        get_kernel_size(sigma)
    } else {
        kernel_size
    };
    let kernel = gaussian_kernel_1d(kernel_size, sigma);
    match hint {
        ConvolutionMode::Exact => {
            let _dispatcher = match src.channels {
                FastBlurChannels::Plane => filter_1d_exact::<u16, f32, 1>,
                FastBlurChannels::Channels3 => filter_1d_exact::<u16, f32, 3>,
                FastBlurChannels::Channels4 => filter_1d_exact::<u16, f32, 4>,
            };
            _dispatcher(
                src,
                dst,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
        }
        ConvolutionMode::FixedPoint => {
            use crate::filter1d::filter_1d_approx;
            let _dispatcher = match src.channels {
                FastBlurChannels::Plane => filter_1d_approx::<u16, f32, u32, 1>,
                FastBlurChannels::Channels3 => filter_1d_approx::<u16, f32, u32, 3>,
                FastBlurChannels::Channels4 => filter_1d_approx::<u16, f32, u32, 4>,
            };
            _dispatcher(
                src,
                dst,
                &kernel,
                &kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
        }
    }
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// This is always perform accurate pure analytical gaussian filter.
/// O(R) complexity.
///
/// # Arguments
///
/// * `src` - Source image.
/// * `dst` - Destination image.
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted. If zero, then sigma must be set.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided.
/// Panics if sigma = 0.8 and kernel size = 0.
pub fn gaussian_blur_f32(
    src: &BlurImage<f32>,
    dst: &mut BlurImageMut<f32>,
    kernel_size: u32,
    sigma: f32,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    assert!(
        kernel_size != 0 || sigma > 0.0,
        "Either sigma or kernel size must be set"
    );
    if kernel_size != 0 {
        assert_ne!(kernel_size % 2, 0, "Kernel size must be odd");
    }
    let sigma = if sigma <= 0. {
        sigma_size(kernel_size as f32)
    } else {
        sigma
    };
    let kernel_size = if kernel_size == 0 {
        get_kernel_size(sigma)
    } else {
        kernel_size
    };
    let kernel = gaussian_kernel_1d(kernel_size, sigma);
    let _dispatcher = match src.channels {
        FastBlurChannels::Plane => filter_1d_exact::<f32, f32, 1>,
        FastBlurChannels::Channels3 => filter_1d_exact::<f32, f32, 3>,
        FastBlurChannels::Channels4 => filter_1d_exact::<f32, f32, 4>,
    };
    _dispatcher(
        src,
        dst,
        &kernel,
        &kernel,
        edge_mode,
        Scalar::default(),
        threading_policy,
    )
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// This is always perform accurate pure analytical gaussian filter.
/// O(R) complexity.
///
/// # Arguments
///
/// * `src` - Source image.
/// * `dst` - Destination image.
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted. If zero, then sigma must be set.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
/// Panics if sigma = 0.8 and kernel size = 0.
pub fn gaussian_blur_f16(
    src: &BlurImage<f16>,
    dst: &mut BlurImageMut<f16>,
    kernel_size: u32,
    sigma: f32,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    assert!(
        kernel_size != 0 || sigma > 0.0,
        "Either sigma or kernel size must be set"
    );
    if kernel_size != 0 {
        assert_ne!(kernel_size % 2, 0, "Kernel size must be odd");
    }
    let sigma = if sigma <= 0. {
        sigma_size(kernel_size as f32)
    } else {
        sigma
    };
    let kernel_size = if kernel_size == 0 {
        get_kernel_size(sigma)
    } else {
        kernel_size
    };
    let kernel = gaussian_kernel_1d(kernel_size, sigma);
    let _dispatcher = match src.channels {
        FastBlurChannels::Plane => filter_1d_exact::<f16, f32, 1>,
        FastBlurChannels::Channels3 => filter_1d_exact::<f16, f32, 3>,
        FastBlurChannels::Channels4 => filter_1d_exact::<f16, f32, 4>,
    };
    _dispatcher(
        src,
        dst,
        &kernel,
        &kernel,
        edge_mode,
        Scalar::default(),
        threading_policy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gaussian_kernel_1d_f64, sigma_size_d};

    macro_rules! compare_u8_stat {
        ($dst: expr) => {
            for (i, cn) in $dst.data.borrow_mut().chunks_exact(3).enumerate() {
                let diff0 = (cn[0] as i32 - 126).abs();
                assert!(
                    diff0 <= 3,
                    "Diff expected to be less than 3, but it was {diff0} at {i} in channel 0"
                );
                let diff1 = (cn[1] as i32 - 66).abs();
                assert!(
                    diff1 <= 3,
                    "Diff expected to be less than 3, but it was {diff1} at {i} in channel 1"
                );
                let diff2 = (cn[2] as i32 - 77).abs();
                assert!(
                    diff2 <= 3,
                    "Diff expected to be less than 3, but it was {diff2} at {i} in channel 2"
                );
            }
        };
    }

    #[test]
    fn test_gauss_u8_q_k5() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![126; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 126;
            dst[1] = 66;
            dst[2] = 77;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        gaussian_blur(
            &src_image,
            &mut dst,
            5,
            0.,
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            ConvolutionMode::FixedPoint,
        )
        .unwrap();
        compare_u8_stat!(dst);
    }

    #[test]
    fn test_gauss_u8_q_k3() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![126; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 126;
            dst[1] = 66;
            dst[2] = 77;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        gaussian_blur(
            &src_image,
            &mut dst,
            3,
            0.,
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            ConvolutionMode::FixedPoint,
        )
        .unwrap();
        compare_u8_stat!(dst);
    }

    #[test]
    fn test_gauss_u8_q_k7() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![126; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 126;
            dst[1] = 66;
            dst[2] = 77;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        gaussian_blur(
            &src_image,
            &mut dst,
            7,
            0.,
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            ConvolutionMode::FixedPoint,
        )
        .unwrap();
        compare_u8_stat!(dst);
    }

    #[test]
    fn test_gauss_u8_fp_k5() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![126; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 126;
            dst[1] = 66;
            dst[2] = 77;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        gaussian_blur(
            &src_image,
            &mut dst,
            5,
            0.,
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            ConvolutionMode::Exact,
        )
        .unwrap();
        compare_u8_stat!(dst);
    }

    #[test]
    fn test_gauss_u8_q_k31() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![126; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 126;
            dst[1] = 66;
            dst[2] = 77;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        gaussian_blur(
            &src_image,
            &mut dst,
            31,
            0.,
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            ConvolutionMode::FixedPoint,
        )
        .unwrap();
        compare_u8_stat!(dst);
    }

    #[test]
    fn test_gauss_u8_fp_k31() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![126; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 126;
            dst[1] = 66;
            dst[2] = 77;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        gaussian_blur(
            &src_image,
            &mut dst,
            31,
            0.,
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            ConvolutionMode::Exact,
        )
        .unwrap();
        compare_u8_stat!(dst);
    }

    macro_rules! compare_u16_stat {
        ($dst: expr) => {
            for (i, cn) in $dst.data.borrow_mut().chunks_exact(3).enumerate() {
                let diff0 = (cn[0] as i32 - 17234i32).abs();
                assert!(
                    diff0 <= 16,
                    "Diff expected to be less than 16, but it was {diff0} at {i} in channel 0"
                );
                let diff1 = (cn[1] as i32 - 5322).abs();
                assert!(
                    diff1 <= 16,
                    "Diff expected to be less than 16, but it was {diff1} at {i} in channel 1"
                );
                let diff2 = (cn[2] as i32 - 7652).abs();
                assert!(
                    diff2 <= 16,
                    "Diff expected to be less than 16, but it was {diff2} at {i} in channel 2"
                );
            }
        };
    }

    #[test]
    fn test_gauss_u16_q_k31() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![17234u16; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 17234u16;
            dst[1] = 5322;
            dst[2] = 7652;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        gaussian_blur_u16(
            &src_image,
            &mut dst,
            31,
            0.,
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            ConvolutionMode::FixedPoint,
        )
        .unwrap();
        compare_u16_stat!(dst);
    }

    #[test]
    fn test_gauss_u16_fp_k31() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![17234u16; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 17234u16;
            dst[1] = 5322;
            dst[2] = 7652;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        gaussian_blur_u16(
            &src_image,
            &mut dst,
            31,
            0.,
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            ConvolutionMode::Exact,
        )
        .unwrap();
        compare_u16_stat!(dst);
    }

    macro_rules! compare_f32_stat {
        ($dst: expr) => {
            for (i, cn) in $dst.data.borrow_mut().chunks_exact(3).enumerate() {
                let diff0 = (cn[0] as f32 - 0.532).abs();
                assert!(
                    diff0 <= 1e-4,
                    "Diff expected to be less than 1e-4, but it was {diff0} at {i} in channel 0"
                );
                let diff1 = (cn[1] as f32 - 0.123).abs();
                assert!(
                    diff1 <= 1e-4,
                    "Diff expected to be less than 1e-4, but it was {diff1} at {i} in channel 1"
                );
                let diff2 = (cn[2] as f32 - 0.654).abs();
                assert!(
                    diff2 <= 1e-4,
                    "Diff expected to be less than 1e-4, but it was {diff2} at {i} in channel 2"
                );
            }
        };
    }

    #[test]
    fn test_gauss_f32_k31() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![0.532; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 0.532;
            dst[1] = 0.123;
            dst[2] = 0.654;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        gaussian_blur_f32(
            &src_image,
            &mut dst,
            31,
            0.,
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
        )
        .unwrap();
        compare_f32_stat!(dst);
    }

    #[test]
    fn test_gauss_f32_f64_k31() {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![0.532; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 0.532;
            dst[1] = 0.123;
            dst[2] = 0.654;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );

        let kernel = gaussian_kernel_1d_f64(31, sigma_size_d(2.5));

        let mut dst = BlurImageMut::default();
        filter_1d_exact::<f32, f64, 3>(
            &src_image,
            &mut dst,
            &kernel,
            &kernel,
            EdgeMode::Clamp,
            Scalar::default(),
            ThreadingPolicy::Adaptive,
        )
        .unwrap();
        compare_f32_stat!(dst);
    }
}
