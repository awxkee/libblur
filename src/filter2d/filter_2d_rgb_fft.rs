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
use crate::filter2d::filter_2d_fft::{filter_2d_fft_impl, FftTranspose};
use crate::filter2d::gather_channel::{gather_channel, squash_channel};
use crate::filter2d::mul_spectrum::SpectrumMultiplier;
use crate::filter2d::scan_se_2d::scan_se_2d_complex;
use crate::to_storage::ToStorage;
use crate::{
    BlurError, BlurImage, BlurImageMut, EdgeMode, FastBlurChannels, KernelShape, Scalar,
    ThreadingPolicy,
};
use num_traits::AsPrimitive;
use rustfft::num_complex::Complex;
use rustfft::FftNum;
use std::fmt::Debug;
use std::ops::Mul;

/// Performs 2D non-separable convolution on RGB image using FFT.
///
/// This method does convolution using spectrum multiplication via fft.
///
/// # Arguments
///
/// * `image`: RGB image.
/// * `destination`: Destination RGB image.
/// * `kernel` - Kernel.
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info.
/// * `border_mode`: See [EdgeMode] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `FftIntermediate`: Intermediate internal type for fft, only `f32` and `f64` is supported.
///
/// returns: Result<(), String>
///
pub fn filter_2d_rgb_fft<T, F, FftIntermediate>(
    src: &BlurImage<T>,
    dst: &mut BlurImageMut<T>,
    kernel: &[F],
    kernel_shape: KernelShape,
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + AsPrimitive<FftIntermediate> + Debug,
    F: ToStorage<T> + Mul<F> + Send + Sync + PartialEq + AsPrimitive<FftIntermediate>,
    FftIntermediate: FftNum
        + Default
        + Mul<FftIntermediate>
        + ToStorage<T>
        + SpectrumMultiplier<FftIntermediate>
        + FftTranspose<FftIntermediate>,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T> + AsPrimitive<FftIntermediate>,
{
    let complex_kernel = kernel
        .iter()
        .map(|&x| Complex {
            re: x.as_(),
            im: 0.0f64.as_(),
        })
        .collect::<Vec<_>>();
    filter_2d_rgb_fft_complex::<T, FftIntermediate>(
        src,
        dst,
        &complex_kernel,
        kernel_shape,
        border_mode,
        border_constant,
        threading_policy,
    )
}

/// Performs 2D non-separable convolution on RGB image using FFT with complex kernel.
///
/// This method does convolution using spectrum multiplication via fft.
///
/// # Arguments
///
/// * `image`: RGB image.
/// * `destination`: Destination RGB image.
/// * `kernel` - Kernel.
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info.
/// * `border_mode`: See [EdgeMode] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `FftIntermediate`: Intermediate internal type for fft, only `f32` and `f64` is supported.
///
/// returns: Result<(), String>
///
pub fn filter_2d_rgb_fft_complex<T, FftIntermediate>(
    src: &BlurImage<T>,
    dst: &mut BlurImageMut<T>,
    kernel: &[Complex<FftIntermediate>],
    kernel_shape: KernelShape,
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy + Default + Send + Sync + AsPrimitive<FftIntermediate> + Debug,
    FftIntermediate: FftNum
        + Default
        + Mul<FftIntermediate>
        + ToStorage<T>
        + SpectrumMultiplier<FftIntermediate>
        + FftTranspose<FftIntermediate>,
    f64: AsPrimitive<T> + AsPrimitive<FftIntermediate>,
{
    if src.channels != FastBlurChannels::Channels3 {
        return Err(BlurError::FftChannelsNotSupported);
    }

    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;

    let analyzed_se = scan_se_2d_complex(kernel, kernel_shape);
    if analyzed_se.is_empty() {
        let dst_stride = dst.row_stride() as usize;
        for (src, dst) in src
            .data
            .chunks_exact(src.row_stride() as usize)
            .zip(dst.data.borrow_mut().chunks_exact_mut(dst_stride))
        {
            for (src, dst) in src.iter().zip(dst.iter_mut()) {
                *dst = *src;
            }
        }
        return Ok(());
    }

    let thread_count = threading_policy.thread_count(src.width, src.height);
    let pool = novtb::ThreadPool::new(thread_count);

    let image_size = src.size();

    let mut working_channel = BlurImageMut::alloc(
        image_size.width as u32,
        image_size.height as u32,
        FastBlurChannels::Plane,
    );

    let mut channel = BlurImageMut::<T>::alloc(
        image_size.width as u32,
        image_size.height as u32,
        FastBlurChannels::Plane,
    );

    gather_channel::<T, 3>(src, &mut channel, 0);
    filter_2d_fft_impl(
        &channel.to_immutable_ref(),
        &mut working_channel,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[0]),
        &pool,
    )?;
    squash_channel::<T, 3>(dst, &working_channel.to_immutable_ref(), 0);

    gather_channel::<T, 3>(src, &mut channel, 1);
    filter_2d_fft_impl(
        &channel.to_immutable_ref(),
        &mut working_channel,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[1]),
        &pool,
    )?;
    squash_channel::<T, 3>(dst, &working_channel.to_immutable_ref(), 1);

    gather_channel::<T, 3>(src, &mut channel, 2);
    filter_2d_fft_impl(
        &channel.to_immutable_ref(),
        &mut working_channel,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[2]),
        &pool,
    )?;
    squash_channel::<T, 3>(dst, &working_channel.to_immutable_ref(), 2);

    Ok(())
}
