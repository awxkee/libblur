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
use crate::fast_divide::{DividerIsize, RemEuclidFast};
use crate::filter1d::{make_arena, ArenaPads};
use crate::filter2d::fft_utils::fft_next_good_size;
use crate::filter2d::scan_se_2d::scan_se_2d_complex;
use crate::to_storage::ToStorage;
use crate::{
    BlurError, BlurImage, BlurImageMut, EdgeMode2D, FastBlurChannels, FftNumber, KernelShape,
    MismatchedSize, Scalar, ThreadingPolicy,
};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::fmt::Debug;
use std::ops::Mul;
use zaft::FftDirection;

/// Performs 2D non-separable convolution on single plane image using FFT.
///
/// This method does convolution using spectrum multiplication via fft.
///
/// # Arguments
///
/// * `image`: Single plane image.
/// * `destination`: Destination image.
/// * `kernel`: Kernel.
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info.
/// * `edge_modes`: See [EdgeMode] and [EdgeMode2D] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `FftIntermediate`: Intermediate internal type for fft, only `f32` and `f64` is supported.
///
/// returns: Result<(), String>
///
pub fn filter_2d_fft<T, F, FftIntermediate>(
    src: &BlurImage<T>,
    dst: &mut BlurImageMut<T>,
    kernel: &[F],
    kernel_shape: KernelShape,
    edge_modes: EdgeMode2D,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + AsPrimitive<FftIntermediate> + Debug,
    F: ToStorage<T> + Mul<F> + Send + Sync + PartialEq + AsPrimitive<FftIntermediate>,
    FftIntermediate: FftNumber + ToStorage<T>,
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
    filter_2d_fft_complex::<T, FftIntermediate>(
        src,
        dst,
        &complex_kernel,
        kernel_shape,
        edge_modes,
        border_constant,
        threading_policy,
    )
}

/// Performs 2D non-separable convolution on single plane image using FFT with complex kernel.
///
/// This method does convolution using spectrum multiplication via fft.
///
/// # Arguments
///
/// * `image`: Single plane image.
/// * `destination`: Destination image.
/// * `kernel`: Kernel.
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info.
/// * `edge_modes`: See [EdgeMode] and [EdgeMode2D] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `FftIntermediate`: Intermediate internal type for fft, only `f32` and `f64` is supported.
///
/// returns: Result<(), String>
///
pub fn filter_2d_fft_complex<T, FftIntermediate>(
    src: &BlurImage<T>,
    dst: &mut BlurImageMut<T>,
    kernel: &[Complex<FftIntermediate>],
    kernel_shape: KernelShape,
    edge_modes: EdgeMode2D,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy + Default + Send + Sync + AsPrimitive<FftIntermediate> + Debug,
    FftIntermediate: FftNumber + ToStorage<T>,
    f64: AsPrimitive<T> + AsPrimitive<FftIntermediate>,
{
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;

    if src.channels != FastBlurChannels::Plane {
        return Err(BlurError::FftChannelsNotSupported);
    }

    let thread_count = threading_policy.thread_count(src.width, src.height);
    let pool = novtb::ThreadPool::new(thread_count);

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

    filter_2d_fft_impl::<T, FftIntermediate>(
        src,
        dst,
        kernel,
        kernel_shape,
        edge_modes,
        border_constant,
        &pool,
    )
}

pub(crate) fn filter_2d_fft_impl<T, FftIntermediate>(
    src: &BlurImage<T>,
    dst: &mut BlurImageMut<T>,
    kernel: &[Complex<FftIntermediate>],
    kernel_shape: KernelShape,
    edge_modes: EdgeMode2D,
    border_constant: Scalar,
    pool: &novtb::ThreadPool,
) -> Result<(), BlurError>
where
    T: Copy + Default + Send + Sync + AsPrimitive<FftIntermediate> + Debug,
    FftIntermediate: FftNumber + ToStorage<T>,
    f64: AsPrimitive<T> + AsPrimitive<FftIntermediate>,
{
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;

    if src.channels != FastBlurChannels::Plane {
        return Err(BlurError::FftChannelsNotSupported);
    }

    let kernel_width = kernel_shape.width;
    let kernel_height = kernel_shape.height;
    if kernel_height * kernel_width != kernel.len() {
        return Err(BlurError::KernelSizeMismatch(MismatchedSize {
            expected: kernel_height * kernel_width,
            received: kernel.len(),
        }));
    }

    let image_size = src.size();

    let best_width = fft_next_good_size(image_size.width + kernel_shape.width);
    let best_height = fft_next_good_size(image_size.height + kernel_shape.height);

    let arena_pad_left = (best_width - image_size.width) / 2;
    let arena_pad_right = best_width - image_size.width - arena_pad_left;
    let arena_pad_top = (best_height - image_size.height) / 2;
    let arena_pad_bottom = best_height - image_size.height - arena_pad_top;

    let (arena_v_src, _) = make_arena::<T, 1>(
        src.data.as_ref(),
        src.row_stride() as usize,
        image_size,
        ArenaPads::new(
            arena_pad_left,
            arena_pad_top,
            arena_pad_right,
            arena_pad_bottom,
        ),
        edge_modes,
        border_constant,
    )?;

    let mut arena_source = arena_v_src
        .iter()
        .map(|&v| Complex::<FftIntermediate> {
            re: v.as_(),
            im: 0f64.as_(),
        })
        .collect::<Vec<Complex<FftIntermediate>>>();

    let mut kernel_arena = vec![Complex::<FftIntermediate>::default(); best_height * best_width];

    let shift_x = kernel_width as isize / 2;
    let shift_y = kernel_height as isize / 2;

    if best_height - 1 <= 1 || best_width - 1 <= 1 {
        // fast divide do not support <= 1
        kernel
            .chunks_exact(kernel_shape.width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, item) in row.iter().enumerate() {
                    let new_y =
                        (y as isize - shift_y).rem_euclid(best_height as isize - 1) as usize;
                    let new_x = (x as isize - shift_x).rem_euclid(best_width as isize - 1) as usize;
                    kernel_arena[new_y * best_width + new_x] = *item;
                }
            });
    } else {
        let divider_height = DividerIsize::new(best_height as isize - 1);
        let divider_width = DividerIsize::new(best_width as isize - 1);
        kernel
            .chunks_exact(kernel_shape.width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, item) in row.iter().enumerate() {
                    let new_y = (y as isize - shift_y).rem_euclid_fast(&divider_height) as usize;
                    let new_x = (x as isize - shift_x).rem_euclid_fast(&divider_width) as usize;
                    kernel_arena[new_y * best_width + new_x] = *item;
                }
            });
    }

    let fft_forward = FftIntermediate::make_fft(
        best_width,
        best_height,
        FftDirection::Forward,
        pool.thread_count(),
    )?;

    let mut scratch =
        vec![Complex::<FftIntermediate>::default(); fft_forward.required_scratch_size()];

    fft_forward
        .execute_with_scratch(&mut arena_source, &mut scratch)
        .map_err(|_| BlurError::FftChannelsNotSupported)?;
    fft_forward
        .execute_with_scratch(&mut kernel_arena, &mut scratch)
        .map_err(|_| BlurError::FftChannelsNotSupported)?;

    let norm_factor = 1f64 / (best_width * best_height) as f64;

    FftIntermediate::mul_spectrum(
        &mut kernel_arena,
        &arena_source,
        best_width,
        best_height,
        norm_factor.as_(),
    );

    arena_source.resize(0, Complex::<FftIntermediate>::default());

    let fft_inverse = FftIntermediate::make_fft(
        best_width,
        best_height,
        FftDirection::Inverse,
        pool.thread_count(),
    )?;

    fft_inverse
        .execute_with_scratch(&mut kernel_arena, &mut scratch)
        .map_err(|_| BlurError::FftChannelsNotSupported)?;

    let dst_stride = dst.row_stride() as usize;

    for (dst_chunk, src_chunk) in dst.data.borrow_mut().chunks_exact_mut(dst_stride).zip(
        kernel_arena
            .chunks_exact_mut(best_width)
            .skip(arena_pad_top),
    ) {
        for (dst, src) in dst_chunk
            .iter_mut()
            .zip(src_chunk.iter().skip(arena_pad_left))
        {
            *dst = src.re.to_();
        }
    }

    Ok(())
}
