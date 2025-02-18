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
use crate::filter2d::scan_se_2d::scan_se_2d;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::util::check_slice_size;
use crate::{BlurError, EdgeMode, ImageSize, MismatchedSize, Scalar, ThreadingPolicy};
use num_traits::{AsPrimitive, MulAdd};
use std::ops::Mul;
use std::sync::Arc;

/// This performs direct 2D convolution on planar image
///
/// # Arguments
///
/// * `CN`: channels count
/// * `src`: Source planar image
/// * `dst`: Destination image
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
pub fn filter_2d<T, F, const CN: usize>(
    src: &[T],
    dst: &mut [T],
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
    check_slice_size(
        src,
        image_size.width * CN,
        image_size.width,
        image_size.height,
        CN,
    )?;
    check_slice_size(
        dst,
        image_size.width * CN,
        image_size.width,
        image_size.height,
        CN,
    )?;

    if src.len() != dst.len() {
        return Err(BlurError::ImagesMustMatch);
    }

    let kernel_width = kernel_shape.width;
    let kernel_height = kernel_shape.height;
    if kernel_height * kernel_width != kernel.len() {
        return Err(BlurError::KernelSizeMismatch(MismatchedSize {
            expected: kernel_height * kernel_width,
            received: kernel.len(),
        }));
    }

    let height = image_size.height;

    let analyzed_se = scan_se_2d(kernel, kernel_shape);

    if analyzed_se.is_empty() {
        for (src, dst) in src.iter().zip(dst.iter_mut()) {
            *dst = *src;
        }
        return Ok(());
    }

    let filter = Arc::new(T::get_executor());

    let (arena_source, arena) = make_arena::<T, CN>(
        src,
        image_size.width * CN,
        image_size,
        ArenaPads::from_kernel_shape(kernel_shape),
        border_mode,
        border_constant,
    )?;

    let thread_count =
        threading_policy.thread_count(image_size.width as u32, image_size.height as u32) as u32;
    let pool = if thread_count == 1 {
        None
    } else {
        let hold = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count as usize)
            .build()
            .unwrap();
        Some(hold)
    };

    let arena_source_slice = arena_source.as_slice();
    let kernel_slice = analyzed_se.as_slice();

    if let Some(pool) = &pool {
        pool.scope(|scope| {
            let unsafe_slice = UnsafeSlice::new(dst);

            for y in 0..height {
                let cloned_filter = filter.clone();
                scope.spawn(move |_| {
                    cloned_filter(
                        arena,
                        arena_source_slice,
                        &unsafe_slice,
                        image_size,
                        kernel_slice,
                        y,
                    );
                });
            }
        })
    } else {
        for y in 0..height {
            let unsafe_slice = UnsafeSlice::new(dst);
            filter(
                arena,
                arena_source_slice,
                &unsafe_slice,
                image_size,
                kernel_slice,
                y,
            );
        }
    }

    Ok(())
}
