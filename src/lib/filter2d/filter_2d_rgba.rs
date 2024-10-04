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

use crate::filter1d::KernelShape;
use crate::filter2d::filter_2d_handler::Filter2dHandler;
use crate::filter2d::gather_channel::{gather_channel, squash_channel};
use crate::to_storage::ToStorage;
use crate::{filter_2d, EdgeMode, ImageSize, Scalar, ThreadingPolicy};
use num_traits::{AsPrimitive, MulAdd};
use std::ops::Mul;

/// This performs direct 2D convolution on RGBA image
///
/// # Arguments
///
/// * `src`: Source RGBA image
/// * `dst`: Destination RGBA image
/// * `image_size`: Image size
/// * `kernel`: Kernel
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info
/// * `border_mode`: Border handling mode see [EdgeMode] for more info
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value
/// * `threading_policy`: See [ThreadingPolicy] for more info
///
/// returns: Result<(), String>
///
/// # Examples
///
/// See [crate::motion_blur] for example
///
pub fn filter_2d_rgba<T, F>(
    src: &[T],
    dst: &mut [T],
    image_size: ImageSize,
    kernel: &[F],
    kernel_shape: KernelShape,
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), String>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + Filter2dHandler<T, F>,
    F: ToStorage<T> + Mul<F> + MulAdd<F, Output = F> + Send + Sync + PartialEq,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T>,
{
    if src.len() != image_size.height * image_size.width * 4 {
        return Err(format!(
            "Image size expected to be {} but it was {}",
            image_size.height * image_size.width * 4,
            src.len()
        ));
    }
    if dst.len() != image_size.height * image_size.width * 4 {
        return Err(format!(
            "Image size expected to be {} but it was {}",
            image_size.height * image_size.width * 4,
            src.len()
        ));
    }

    let mut working_channel = vec![T::default(); image_size.width * image_size.height];

    let mut chanel_first = gather_channel::<T, 4>(src, image_size, 0);
    filter_2d(
        &chanel_first,
        &mut working_channel,
        image_size,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[0]),
        threading_policy,
    )?;
    squash_channel::<T, 4>(dst, &working_channel, 0);
    chanel_first.resize(0, T::default());

    let mut chanel_second = gather_channel::<T, 4>(src, image_size, 1);
    filter_2d(
        &chanel_second,
        &mut working_channel,
        image_size,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[1]),
        threading_policy,
    )?;
    squash_channel::<T, 4>(dst, &working_channel, 1);
    chanel_second.resize(0, T::default());

    let mut chanel_third = gather_channel::<T, 4>(src, image_size, 2);
    filter_2d(
        &chanel_third,
        &mut working_channel,
        image_size,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[2]),
        threading_policy,
    )?;
    squash_channel::<T, 4>(dst, &working_channel, 2);
    chanel_third.resize(0, T::default());

    let mut chanel_fourth = gather_channel::<T, 4>(src, image_size, 3);
    filter_2d(
        &chanel_fourth,
        &mut working_channel,
        image_size,
        kernel,
        kernel_shape,
        border_mode,
        Scalar::dup(border_constant[3]),
        threading_policy,
    )?;
    squash_channel::<T, 4>(dst, &working_channel, 3);
    chanel_fourth.resize(0, T::default());

    Ok(())
}
