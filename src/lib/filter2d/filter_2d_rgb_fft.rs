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
use crate::util::check_slice_size;
use crate::{BlurError, EdgeMode, ImageSize, KernelShape, Scalar};
use num_traits::AsPrimitive;
use rustfft::FftNum;
use std::ops::Mul;

/// Performs 2D separable approximated convolution on RGB image.
///
/// This method does convolution using spectrum multiplication via fft.
///
/// # Arguments
///
/// * `image`: RGB image.
/// * `destination`: Destination RGB image.
/// * `image_size`: Image size see [ImageSize].
/// * `kernel` - Kernel.
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info.
/// * `border_mode`: See [EdgeMode] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `FftIntermediate`: Intermediate internal type for fft, only `f32` and `f64` is supported.
///
/// returns: Result<(), String>
///
pub fn filter_2d_rgb_fft<T, F, FftIntermediate>(
    src: &[T],
    dst: &mut [T],
    image_size: ImageSize,
    kernel: &[F],
    kernel_shape: KernelShape,
    border_mode: EdgeMode,
    border_constant: Scalar,
) -> Result<(), BlurError>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + AsPrimitive<FftIntermediate>,
    F: ToStorage<T> + Mul<F> + Send + Sync + PartialEq + AsPrimitive<FftIntermediate>,
    FftIntermediate: FftNum + Default + Mul<FftIntermediate> + ToStorage<T>,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T> + AsPrimitive<FftIntermediate>,
{
    check_slice_size(
        src,
        image_size.width * 3,
        image_size.width,
        image_size.height,
        3,
    )?;
    check_slice_size(
        dst,
        image_size.width * 3,
        image_size.width,
        image_size.height,
        3,
    )?;

    let mut working_channel = vec![T::default(); image_size.width * image_size.height];

    let mut chanel_first = gather_channel::<T, 3>(src, image_size, 0);
    filter_2d_fft(
        &chanel_first,
        &mut working_channel,
        image_size,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[0]),
    )?;
    squash_channel::<T, 3>(dst, &working_channel, 0);
    chanel_first.resize(0, T::default());

    let mut chanel_second = gather_channel::<T, 3>(src, image_size, 1);
    filter_2d_fft(
        &chanel_second,
        &mut working_channel,
        image_size,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[1]),
    )?;
    squash_channel::<T, 3>(dst, &working_channel, 1);
    chanel_second.resize(0, T::default());

    let mut chanel_third = gather_channel::<T, 3>(src, image_size, 2);
    filter_2d_fft(
        &chanel_third,
        &mut working_channel,
        image_size,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[2]),
    )?;
    squash_channel::<T, 3>(dst, &working_channel, 2);
    chanel_third.resize(0, T::default());

    Ok(())
}
