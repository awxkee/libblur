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
use crate::filter2d::filter_2d_fft::filter_2d_fft;
use crate::filter2d::gather_channel::{gather_channel, squash_channel};
use crate::to_storage::ToStorage;
use crate::{BlurError, BlurImage, BlurImageMut, EdgeMode, FastBlurChannels, KernelShape, Scalar};
use num_traits::AsPrimitive;
use rustfft::FftNum;
use std::fmt::Debug;
use std::ops::Mul;

/// Performs 2D non-separable approximated convolution on RGBA image using FFT.
///
/// This method does convolution using spectrum multiplication via fft.
///
/// # Arguments
///
/// * `image`: RGBA image.
/// * `destination`: Destination RGBA image.
/// * `kernel`: Kernel`
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info.
/// * `border_mode`: See [EdgeMode] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `FftIntermediate`: Intermediate internal type for fft, only `f32` and `f64` is supported.
///
/// returns: Result<(), String>
///
pub fn filter_2d_rgba_fft<T, F, FftIntermediate>(
    src: &BlurImage<T>,
    dst: &mut BlurImageMut<T>,
    kernel: &[F],
    kernel_shape: KernelShape,
    border_mode: EdgeMode,
    border_constant: Scalar,
) -> Result<(), BlurError>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + AsPrimitive<FftIntermediate> + Debug,
    F: ToStorage<T> + Mul<F> + Send + Sync + PartialEq + AsPrimitive<FftIntermediate>,
    FftIntermediate: FftNum + Default + Mul<FftIntermediate> + ToStorage<T>,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T> + AsPrimitive<FftIntermediate>,
{
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;

    let image_size = src.size();

    let mut working_channel = BlurImageMut::alloc(
        image_size.width as u32,
        image_size.height as u32,
        FastBlurChannels::Plane,
    );

    let chanel_first = gather_channel::<T, 4>(src, 0);
    filter_2d_fft(
        &chanel_first.to_immutable_ref(),
        &mut working_channel,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[0]),
    )?;
    squash_channel::<T, 4>(dst, &working_channel.to_immutable_ref(), 0);

    let chanel_second = gather_channel::<T, 4>(src, 1);
    filter_2d_fft(
        &chanel_second.to_immutable_ref(),
        &mut working_channel,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[1]),
    )?;
    squash_channel::<T, 4>(dst, &working_channel.to_immutable_ref(), 1);

    let chanel_third = gather_channel::<T, 4>(src, 2);
    filter_2d_fft(
        &chanel_third.to_immutable_ref(),
        &mut working_channel,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[2]),
    )?;
    squash_channel::<T, 4>(dst, &working_channel.to_immutable_ref(), 2);

    let chanel_fourth = gather_channel::<T, 4>(src, 3);
    filter_2d_fft(
        &chanel_fourth.to_immutable_ref(),
        &mut working_channel,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[3]),
    )?;
    squash_channel::<T, 4>(dst, &working_channel.to_immutable_ref(), 3);

    Ok(())
}
