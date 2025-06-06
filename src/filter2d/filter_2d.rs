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
use crate::filter1d::{make_arena, ArenaPads, KernelShape};
use crate::filter2d::filter_2d_handler::Filter2dHandler;
use crate::filter2d::scan_point_2d::ScanPoint2d;
use crate::filter2d::scan_se_2d::scan_se_2d;
use crate::to_storage::ToStorage;
use crate::{
    BlurError, BlurImage, BlurImageMut, EdgeMode, FastBlurChannels, ImageSize, MismatchedSize,
    Scalar, ThreadingPolicy,
};
use novtb::{ParallelZonedIterator, TbSliceMut};
use num_traits::{AsPrimitive, MulAdd};
use std::fmt::Debug;
use std::ops::Mul;
use std::sync::Arc;

/// This performs direct 2D convolution on image.
///
/// # Arguments
///
/// * `CN`: channels count.
/// * `src`: Source planar image.
/// * `src_stride`: Source image stride.
/// * `dst`: Destination image.
/// * `dst_stride`: Destination image stride.
/// * `image_size`: Image size.
/// * `kernel`: Kernel.
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info.
/// * `border_mode`: Border handling mode see [EdgeMode] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `threading_policy`: See [ThreadingPolicy] for more info.
///
/// returns: Result<(), String>
///
/// # Examples
///
/// See [crate::motion_blur] for example
///
pub fn filter_2d<T, F>(
    src: &BlurImage<T>,
    dst: &mut BlurImageMut<T>,
    kernel: &[F],
    kernel_shape: KernelShape,
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + Filter2dHandler<T, F> + Debug,
    F: ToStorage<T> + Mul<F> + MulAdd<F, Output = F> + Send + Sync + PartialEq + AsPrimitive<f64>,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T>,
{
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    let channels = src.channels;
    match channels {
        FastBlurChannels::Plane => filter_2d_arbitrary::<T, F, 1>(
            src,
            dst,
            kernel,
            kernel_shape,
            border_mode,
            border_constant,
            threading_policy,
        ),
        FastBlurChannels::Channels3 => filter_2d_arbitrary::<T, F, 3>(
            src,
            dst,
            kernel,
            kernel_shape,
            border_mode,
            border_constant,
            threading_policy,
        ),
        FastBlurChannels::Channels4 => filter_2d_arbitrary::<T, F, 4>(
            src,
            dst,
            kernel,
            kernel_shape,
            border_mode,
            border_constant,
            threading_policy,
        ),
    }
}

/// This performs direct 2D convolution on image.
///
/// # Arguments
///
/// * `CN`: channels count.
/// * `src`: Source planar image.
/// * `dst`: Destination image.
/// * `image_size`: Image size.
/// * `kernel`: Kernel.
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info.
/// * `border_mode`: Border handling mode see [EdgeMode] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `threading_policy`: See [ThreadingPolicy] for more info.
///
/// returns: Result<(), String>
///
/// # Examples
///
/// See [crate::motion_blur] for example
///
pub fn filter_2d_arbitrary<T, F, const CN: usize>(
    src: &BlurImage<T>,
    dst: &mut BlurImageMut<T>,
    kernel: &[F],
    kernel_shape: KernelShape,
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + Filter2dHandler<T, F> + Debug,
    F: ToStorage<T> + Mul<F> + MulAdd<F, Output = F> + Send + Sync + PartialEq + AsPrimitive<f64>,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T>,
{
    src.check_layout_channels(CN)?;
    dst.check_layout_channels(CN, Some(src))?;
    let kernel_width = kernel_shape.width;
    let kernel_height = kernel_shape.height;
    if kernel_height * kernel_width != kernel.len() {
        return Err(BlurError::KernelSizeMismatch(MismatchedSize {
            expected: kernel_height * kernel_width,
            received: kernel.len(),
        }));
    }

    let analyzed_se = scan_se_2d(kernel, kernel_shape);

    let dst_stride = dst.stride as usize;

    if analyzed_se.is_empty() {
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

    let image_size = ImageSize::new(src.width as usize, src.height as usize);

    let (arena_source, arena) = make_arena::<T, CN>(
        src.data.as_ref(),
        src.row_stride() as usize,
        image_size,
        ArenaPads::from_kernel_shape(kernel_shape),
        border_mode,
        border_constant,
    )?;

    let thread_count =
        threading_policy.thread_count(image_size.width as u32, image_size.height as u32) as u32;
    let pool = novtb::ThreadPool::new(thread_count as usize);

    let arena_source_slice = arena_source.as_slice();
    let kernel_slice = analyzed_se.as_slice();

    let fp_executor_test = T::get_fp_executor();

    if T::FIXED_POINT_REPRESENTABLE && fp_executor_test.is_some() {
        let filter = Arc::new(T::get_fp_executor().unwrap());
        const S: f64 = ((1 << 15) - 1) as f64;
        let fp_slice = kernel_slice
            .iter()
            .map(|&x| {
                let w: f64 = x.weight.as_();
                ScanPoint2d::<i16> {
                    x: x.x,
                    y: x.y,
                    weight: (w * S).round().min(i16::MAX as f64).max(i16::MIN as f64) as i16,
                }
            })
            .collect::<Vec<ScanPoint2d<i16>>>();

        dst.data
            .borrow_mut()
            .tb_par_chunks_exact_mut(dst_stride)
            .for_each_enumerated(&pool, |y, row| {
                let row = &mut row[..image_size.width * CN];
                filter(arena, arena_source_slice, row, image_size, &fp_slice, y);
            });
    } else {
        let filter = Arc::new(T::get_executor());
        dst.data
            .borrow_mut()
            .tb_par_chunks_exact_mut(dst_stride)
            .for_each_enumerated(&pool, |y, row| {
                let row = &mut row[..image_size.width * CN];
                filter(arena, arena_source_slice, row, image_size, kernel_slice, y);
            });
    }
    Ok(())
}
