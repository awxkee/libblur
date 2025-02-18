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
use crate::to_storage::ToStorage;
use crate::{filter_2d, BlurError, EdgeMode, ImageSize, Scalar, ThreadingPolicy};
use num_traits::{AsPrimitive, MulAdd};
use std::ops::Mul;

/// This performs direct 2D convolution on RGBA image
///
/// # Arguments
///
/// * `src`: Source RGBA image.
/// * `src_stride`: Source image stride
/// * `dst`: Destination RGBA image.
/// * `dst_stride`: Destination image stride.
/// * `image_size`: Image size.
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
    src_stride: usize,
    dst: &mut [T],
    dst_stride: usize,
    image_size: ImageSize,
    kernel: &[F],
    kernel_shape: KernelShape,
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + Filter2dHandler<T, F>,
    F: ToStorage<T> + Mul<F> + MulAdd<F, Output = F> + Send + Sync + PartialEq,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T>,
{
    filter_2d::<T, F, 4>(
        src,
        src_stride,
        dst,
        dst_stride,
        image_size,
        kernel,
        kernel_shape,
        border_mode,
        border_constant,
        threading_policy,
    )
}
