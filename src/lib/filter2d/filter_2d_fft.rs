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
#![forbid(unsafe_code)]
use crate::filter1d::{make_arena, ArenaPads};
use crate::filter2d::fft_utils::fft_next_good_size;
use crate::filter2d::mul_spectrum::mul_spectrum_in_place;
use crate::filter2d::scan_se_2d::scan_se_2d;
use crate::to_storage::ToStorage;
use crate::util::check_slice_size;
use crate::{BlurError, EdgeMode, ImageSize, KernelShape, MismatchedSize, Scalar};
use fast_transpose::{transpose_arbitrary, FlipMode, FlopMode};
use num_traits::AsPrimitive;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use rustfft::num_complex::Complex;
use rustfft::{FftNum, FftPlanner};
use std::ops::Mul;

fn transpose<T: Copy + Default>(
    matrix: &[T],
    width: usize,
    height: usize,
    flop_mode: FlopMode,
) -> Vec<T> {
    if matrix.is_empty() {
        return Vec::new();
    }

    let mut transposed = vec![T::default(); width * height];

    transpose_arbitrary(
        matrix,
        width,
        &mut transposed,
        height,
        width,
        height,
        FlipMode::NoFlip,
        flop_mode,
    )
    .unwrap();

    transposed
}

/// Performs 2D separable approximated convolution on single plane image.
///
/// This method does convolution using spectrum multiplication via fft.
///
/// # Arguments
///
/// * `image`: Single plane image.
/// * `destination`: Destination image.
/// * `image_size`: Image size see [ImageSize].
/// * `kernel`: Kernel.
/// * `kernel_shape`: Kernel size, see [KernelShape] for more info.
/// * `border_mode`: See [EdgeMode] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `FftIntermediate`: Intermediate internal type for fft, only `f32` and `f64` is supported.
///
/// returns: Result<(), String>
///
pub fn filter_2d_fft<T, F, FftIntermediate>(
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
        image_size.width,
        image_size.width,
        image_size.height,
        1,
    )?;
    check_slice_size(
        dst,
        image_size.width,
        image_size.width,
        image_size.height,
        1,
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

    let width = image_size.width;

    let analyzed_se = scan_se_2d(kernel, kernel_shape);

    if analyzed_se.is_empty() {
        for (src, dst) in src.iter().zip(dst.iter_mut()) {
            *dst = *src;
        }
        return Ok(());
    }

    let best_width = fft_next_good_size(image_size.width + kernel_shape.width);
    let best_height = fft_next_good_size(image_size.height + kernel_shape.height);

    let arena_pad_left = (best_width - image_size.width) / 2;
    let arena_pad_right = best_width - image_size.width - arena_pad_left;
    let arena_pad_top = (best_height - image_size.height) / 2;
    let arena_pad_bottom = best_height - image_size.height - arena_pad_top;

    let (arena_v_src, _) = make_arena::<T, 1>(
        src,
        image_size.width,
        image_size,
        ArenaPads::new(
            arena_pad_left,
            arena_pad_top,
            arena_pad_right,
            arena_pad_bottom,
        ),
        border_mode,
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

    let shift_x = kernel_width as i64 / 2;
    let shift_y = kernel_height as i64 / 2;

    kernel
        .chunks_exact(kernel_shape.width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, item) in row.iter().enumerate() {
                let new_y = (y as i64 - shift_y).rem_euclid(best_height as i64 - 1) as usize;
                let new_x = (x as i64 - shift_x).rem_euclid(best_width as i64 - 1) as usize;
                kernel_arena[new_y * best_width + new_x] = Complex {
                    re: item.as_(),
                    im: 0f64.as_(),
                };
            }
        });

    let mut fft_planner = FftPlanner::<FftIntermediate>::new();
    let rows_planner = fft_planner.plan_fft_forward(best_width);
    let columns_planner = fft_planner.plan_fft_forward(best_height);

    arena_source
        .par_chunks_exact_mut(best_width)
        .for_each(|row| {
            rows_planner.process(row);
        });

    kernel_arena
        .par_chunks_exact_mut(best_width)
        .for_each(|row| {
            rows_planner.process(row);
        });

    arena_source = transpose(&arena_source, best_width, best_height, FlopMode::Flop);
    kernel_arena = transpose(&kernel_arena, best_width, best_height, FlopMode::Flop);

    arena_source
        .par_chunks_exact_mut(best_height)
        .for_each(|column| {
            columns_planner.process(column);
        });

    kernel_arena
        .par_chunks_exact_mut(best_height)
        .for_each(|column| {
            columns_planner.process(column);
        });

    mul_spectrum_in_place(&mut kernel_arena, &arena_source, best_width, best_height);

    arena_source.resize(0, Complex::<FftIntermediate>::default());

    let rows_inverse_planner = fft_planner.plan_fft_inverse(best_width);
    let columns_inverse_planner = fft_planner.plan_fft_inverse(best_height);

    kernel_arena
        .par_chunks_exact_mut(best_height)
        .for_each(|column| {
            columns_inverse_planner.process(column);
        });

    kernel_arena = transpose(&kernel_arena, best_height, best_width, FlopMode::Flop);

    kernel_arena
        .par_chunks_exact_mut(best_width)
        .for_each(|row| {
            rows_inverse_planner.process(row);
        });

    for (dst_chunk, src_chunk) in dst.chunks_exact_mut(width).zip(
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
