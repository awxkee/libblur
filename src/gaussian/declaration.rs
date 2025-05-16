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
use crate::gaussian::gaussian_hint::IeeeBinaryConvolutionMode;
use crate::gaussian::gaussian_kernel::gaussian_kernel_1d;
use crate::gaussian::gaussian_util::{kernel_size as get_kernel_size, kernel_size_d};
use crate::{
    filter_1d_approx, filter_1d_exact, gaussian_kernel_1d_f64, sigma_size, sigma_size_d, BlurError,
    BlurImage, BlurImageMut, ConvolutionMode, Scalar, ThreadingPolicy,
};
use half::f16;

#[derive(Copy, Clone, Debug)]
pub struct GaussianBlurParams {
    /// X-axis kernel size
    pub x_kernel: u32,
    /// X-axis sigma
    pub x_sigma: f64,
    /// Y-axis kernel size
    pub y_kernel: u32,
    /// Y-axis sigma
    pub y_sigma: f64,
}

#[inline]
fn round_to_nearest_odd(x: f64) -> i64 {
    let n = x.round() as i64;
    if n % 2 != 0 {
        n
    } else {
        // Check which odd integer is closer
        let lower = n - 1;
        let upper = n + 1;

        let dist_lower = (x - lower as f64).abs();
        let dist_upper = (x - upper as f64).abs();

        if dist_lower <= dist_upper {
            lower
        } else {
            upper
        }
    }
}
impl GaussianBlurParams {
    /// Kernel expected to be odd.
    /// Sigma must be > 0.
    pub fn new(kernel: u32, sigma: f64) -> GaussianBlurParams {
        GaussianBlurParams {
            x_kernel: kernel,
            x_sigma: sigma,
            y_kernel: kernel,
            y_sigma: sigma,
        }
    }

    /// Sigma must be > 0 and not equal to `0.8`.
    pub fn new_from_sigma(sigma: f64) -> GaussianBlurParams {
        assert!(sigma > 0.);
        let kernel_size = kernel_size_d(sigma);
        Self::new(kernel_size, sigma)
    }

    /// Kernel must be > 0.
    /// Kernel will be rounded to nearest odd, it is safe to pass any kernel here.
    pub fn new_from_kernel(kernel: f64) -> GaussianBlurParams {
        assert!(kernel > 0.);
        let sigma = sigma_size_d(kernel);
        Self::new(round_to_nearest_odd(kernel) as u32, sigma)
    }

    /// Kernel must be > 0.
    /// Kernel will be rounded to nearest odd, it is safe to pass any kernel here.
    pub fn new_asymmetric_from_kernels(x_kernel: f64, y_kernel: f64) -> GaussianBlurParams {
        assert!(x_kernel > 0.);
        assert!(y_kernel > 0.);
        let x_sigma = sigma_size_d(x_kernel);
        let y_sigma = sigma_size_d(y_kernel);
        Self::new_asymmetric(
            round_to_nearest_odd(x_kernel) as u32,
            x_sigma,
            round_to_nearest_odd(y_kernel) as u32,
            y_sigma,
        )
    }

    /// Kernel expected to be odd.
    /// Sigma must be > 0.
    pub fn new_asymmetric(
        x_kernel: u32,
        x_sigma: f64,
        y_kernel: u32,
        y_sigma: f64,
    ) -> GaussianBlurParams {
        GaussianBlurParams {
            x_kernel,
            x_sigma,
            y_kernel,
            y_sigma,
        }
    }

    /// Sigma must be > 0 and not equal to `0.8`.
    pub fn new_asymmetric_from_sigma(x_sigma: f64, y_sigma: f64) -> GaussianBlurParams {
        GaussianBlurParams {
            x_kernel: kernel_size_d(x_sigma),
            x_sigma,
            y_kernel: kernel_size_d(y_sigma),
            y_sigma,
        }
    }

    fn make_f32_kernel(&self, kernel_size: u32, sigma: f32) -> Vec<f32> {
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
        gaussian_kernel_1d(kernel_size, sigma)
    }

    fn make_f64_kernel(&self, kernel_size: u32, sigma: f64) -> Vec<f64> {
        assert!(
            kernel_size != 0 || sigma > 0.0,
            "Either sigma or kernel size must be set"
        );
        if kernel_size != 0 {
            assert_ne!(kernel_size % 2, 0, "Kernel size must be odd");
        }
        let sigma = if sigma <= 0. {
            sigma_size_d(kernel_size as f64)
        } else {
            sigma
        };
        let kernel_size = if kernel_size == 0 {
            kernel_size_d(sigma)
        } else {
            kernel_size
        };
        gaussian_kernel_1d_f64(kernel_size, sigma)
    }

    fn make_f32_kernels(&self) -> (Vec<f32>, Vec<f32>) {
        let vx_kernel = self.make_f32_kernel(self.x_kernel, self.x_sigma as f32);
        let vy_kernel = self.make_f32_kernel(self.y_kernel, self.y_sigma as f32);
        (vx_kernel, vy_kernel)
    }

    fn make_f64_kernels(&self) -> (Vec<f64>, Vec<f64>) {
        let vx_kernel = self.make_f64_kernel(self.x_kernel, self.x_sigma);
        let vy_kernel = self.make_f64_kernel(self.y_kernel, self.y_sigma);
        (vx_kernel, vy_kernel)
    }

    fn validate(&self) -> Result<(), BlurError> {
        if self.x_sigma < 0. || self.y_sigma < 0. {
            return Err(BlurError::NegativeOrZeroSigma);
        }
        if self.x_kernel > 0 && self.x_kernel % 2 == 0 {
            return Err(BlurError::OddKernel(self.x_kernel as usize));
        }
        if self.y_kernel > 0 && self.y_kernel % 2 == 0 {
            return Err(BlurError::OddKernel(self.y_kernel as usize));
        }
        if self.x_sigma == 0. && self.x_kernel == 0 {
            return Err(BlurError::InvalidArguments);
        }
        if self.y_sigma == 0. && self.y_kernel == 0 {
            return Err(BlurError::InvalidArguments);
        }
        Ok(())
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
/// * `params` - See [GaussianBlurParams] for more info.
/// * `edge_mode` - Rule to handle edge mode, sse [EdgeMode] for more info.
/// * `threading_policy` - Threading policy according to [ThreadingPolicy].
/// * `hint` - see [ConvolutionMode] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided.
/// Panics if sigma = 0.8 and kernel size = 0.
pub fn gaussian_blur(
    src: &BlurImage<u8>,
    dst: &mut BlurImageMut<u8>,
    params: GaussianBlurParams,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
    hint: ConvolutionMode,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    params.validate()?;
    let (x_kernel, y_kernel) = params.make_f32_kernels();
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
                &x_kernel,
                &y_kernel,
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
                &x_kernel,
                &y_kernel,
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
/// * `params` - See [GaussianBlurParams] for more info.
/// * `edge_mode` - Rule to handle edge mode, sse [EdgeMode] for more info.
/// * `threading_policy` - Threading policy according to [ThreadingPolicy].
/// * `hint` - see [ConvolutionMode] for more info.
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
    params: GaussianBlurParams,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
    hint: ConvolutionMode,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    params.validate()?;
    let (x_kernel, y_kernel) = params.make_f32_kernels();
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
                &x_kernel,
                &y_kernel,
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
                &x_kernel,
                &y_kernel,
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
/// * `params` - See [GaussianBlurParams] for more info.
/// * `edge_mode` - Rule to handle edge mode, sse [EdgeMode] for more info.
/// * `threading_policy` - Threading policy according to [ThreadingPolicy].
/// * `convolution_mode` - See [IeeeBinaryConvolutionMode] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided.
/// Panics if sigma = 0.8 and kernel size = 0.
pub fn gaussian_blur_f32(
    src: &BlurImage<f32>,
    dst: &mut BlurImageMut<f32>,
    params: GaussianBlurParams,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
    convolution_mode: IeeeBinaryConvolutionMode,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    params.validate()?;
    match convolution_mode {
        IeeeBinaryConvolutionMode::Normal => {
            let (x_kernel, y_kernel) = params.make_f32_kernels();
            let _dispatcher = match src.channels {
                FastBlurChannels::Plane => filter_1d_exact::<f32, f32, 1>,
                FastBlurChannels::Channels3 => filter_1d_exact::<f32, f32, 3>,
                FastBlurChannels::Channels4 => filter_1d_exact::<f32, f32, 4>,
            };
            _dispatcher(
                src,
                dst,
                &x_kernel,
                &y_kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
        }
        IeeeBinaryConvolutionMode::Zealous => {
            let (x_kernel, y_kernel) = params.make_f64_kernels();
            let _dispatcher = match src.channels {
                FastBlurChannels::Plane => filter_1d_exact::<f32, f64, 1>,
                FastBlurChannels::Channels3 => filter_1d_exact::<f32, f64, 3>,
                FastBlurChannels::Channels4 => filter_1d_exact::<f32, f64, 4>,
            };
            _dispatcher(
                src,
                dst,
                &x_kernel,
                &y_kernel,
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
/// * `params` - See [GaussianBlurParams] for more info.
/// * `edge_mode` - Rule to handle edge mode, sse [EdgeMode] for more info.
/// * `threading_policy` - Threading policy according to [ThreadingPolicy].
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
/// Panics if sigma = 0.8 and kernel size = 0.
pub fn gaussian_blur_f16(
    src: &BlurImage<f16>,
    dst: &mut BlurImageMut<f16>,
    params: GaussianBlurParams,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    params.validate()?;
    let (x_kernel, y_kernel) = params.make_f32_kernels();
    let _dispatcher = match src.channels {
        FastBlurChannels::Plane => filter_1d_exact::<f16, f32, 1>,
        FastBlurChannels::Channels3 => filter_1d_exact::<f16, f32, 3>,
        FastBlurChannels::Channels4 => filter_1d_exact::<f16, f32, 4>,
    };
    _dispatcher(
        src,
        dst,
        &x_kernel,
        &y_kernel,
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
            GaussianBlurParams::new_from_kernel(5.),
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
            GaussianBlurParams::new_from_kernel(3.),
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            ConvolutionMode::FixedPoint,
        )
        .unwrap();
        println!("{}", dst.data.borrow_mut()[0]);
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
            GaussianBlurParams::new_from_kernel(7.),
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
            GaussianBlurParams::new_from_kernel(5.),
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
            GaussianBlurParams::new_from_kernel(31.),
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
            GaussianBlurParams::new_from_kernel(31.),
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
            GaussianBlurParams::new_from_kernel(31.),
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
            GaussianBlurParams::new_from_kernel(31.),
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
            GaussianBlurParams::new_from_kernel(31.),
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            IeeeBinaryConvolutionMode::Normal,
        )
        .unwrap();
        compare_f32_stat!(dst);
        dst.data.borrow_mut().fill(0.);
        gaussian_blur_f32(
            &src_image,
            &mut dst,
            GaussianBlurParams::new_from_kernel(31.),
            EdgeMode::Clamp,
            ThreadingPolicy::Single,
            IeeeBinaryConvolutionMode::Zealous,
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
