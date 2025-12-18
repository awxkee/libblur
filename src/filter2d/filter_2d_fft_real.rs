/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::filter2d::fft_utils::fft_next_good_size_real;
use crate::filter2d::mul_spectrum::SpectrumMultiplier;
use crate::to_storage::ToStorage;
use crate::{
    BlurError, BlurImage, BlurImageMut, EdgeMode2D, FastBlurChannels, KernelShape, MismatchedSize,
    Scalar,
};
use num_complex::Complex;
use num_traits::{AsPrimitive, Num};
use std::fmt::Debug;
use std::ops::Mul;
use std::sync::Arc;
use zaft::{
    FftDirection, TwoDimensionalExecutorC2R, TwoDimensionalExecutorR2C, TwoDimensionalFftExecutor,
    Zaft,
};

pub trait FftFactory<T> {
    fn make_r2c_executor(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorR2C<T> + Send + Sync>, BlurError>;
    fn make_c2r_executor(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorC2R<T> + Send + Sync>, BlurError>;
    fn make_fft(
        width: usize,
        height: usize,
        direction: FftDirection,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalFftExecutor<T> + Send + Sync>, BlurError>;
}

impl FftFactory<f32> for f32 {
    fn make_c2r_executor(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorC2R<f32> + Send + Sync>, BlurError> {
        Zaft::make_2d_c2r_fft_f32(width, height, thread_count)
            .map_err(|_| BlurError::FftChannelsNotSupported)
    }

    fn make_r2c_executor(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorR2C<f32> + Send + Sync>, BlurError> {
        Zaft::make_2d_r2c_fft_f32(width, height, thread_count)
            .map_err(|_| BlurError::FftChannelsNotSupported)
    }

    fn make_fft(
        width: usize,
        height: usize,
        direction: FftDirection,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalFftExecutor<f32> + Send + Sync>, BlurError> {
        Zaft::make_2d_c2c_fft_f32(width, height, direction, thread_count)
            .map_err(|_| BlurError::FftChannelsNotSupported)
    }
}

impl FftFactory<f64> for f64 {
    fn make_c2r_executor(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorC2R<f64> + Send + Sync>, BlurError> {
        Zaft::make_2d_c2r_fft_f64(width, height, thread_count)
            .map_err(|_| BlurError::FftChannelsNotSupported)
    }

    fn make_r2c_executor(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorR2C<f64> + Send + Sync>, BlurError> {
        Zaft::make_2d_r2c_fft_f64(width, height, thread_count)
            .map_err(|_| BlurError::FftChannelsNotSupported)
    }

    fn make_fft(
        width: usize,
        height: usize,
        direction: FftDirection,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalFftExecutor<f64> + Send + Sync>, BlurError> {
        Zaft::make_2d_c2c_fft_f64(width, height, direction, thread_count)
            .map_err(|_| BlurError::FftChannelsNotSupported)
    }
}

pub trait FftNumber:
    Copy
    + Default
    + Send
    + Sync
    + Default
    + Mul<Self>
    + SpectrumMultiplier<Self>
    + Debug
    + Num
    + FftFactory<Self>
    + PartialEq
    + PartialOrd
    + 'static
{
}

impl FftNumber for f32 {}

impl FftNumber for f64 {}

pub(crate) fn filter_2d_fft_real_impl<T, F>(
    src: &BlurImage<T>,
    dst: &mut BlurImageMut<T>,
    kernel: &[F],
    kernel_shape: KernelShape,
    edge_modes: EdgeMode2D,
    border_constant: Scalar,
    thread_count: usize,
) -> Result<(), BlurError>
where
    T: AsPrimitive<F> + Copy + Default + Send + Sync + Default + Debug,
    F: FftNumber + ToStorage<T>,
    f64: AsPrimitive<T> + AsPrimitive<F>,
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

    let best_width = fft_next_good_size_real(image_size.width + kernel_shape.width);
    let best_height = fft_next_good_size_real(image_size.height + kernel_shape.height);

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

    let mut kernel_arena = vec![F::default(); best_height * best_width];

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

    let complex_plane_width = (best_width / 2) + 1;
    let complex_plane_size = ((best_width / 2) + 1) * best_height;
    let complex_plane_height = best_height;

    let fft_2d_r2c = F::make_r2c_executor(best_width, best_height, thread_count)?;

    let mut scratch = vec![Complex::<F>::default(); fft_2d_r2c.required_scratch_size()];

    let mut arena_dst = vec![Complex::<F>::default(); complex_plane_size];
    let mut kernel_dst = vec![Complex::<F>::default(); complex_plane_size];

    let arena_source = arena_v_src.iter().map(|&v| v.as_()).collect::<Vec<F>>();

    fft_2d_r2c
        .execute_with_scratch(&arena_source, &mut arena_dst, &mut scratch)
        .map_err(|_| BlurError::FftChannelsNotSupported)?;
    fft_2d_r2c
        .execute_with_scratch(&kernel_arena, &mut kernel_dst, &mut scratch)
        .map_err(|_| BlurError::FftChannelsNotSupported)?;

    let norm_factor = 1f64 / (best_width * best_height) as f64;

    F::mul_spectrum(
        &mut kernel_dst,
        &arena_dst,
        complex_plane_width,
        complex_plane_height,
        norm_factor.as_(),
    );

    arena_dst.resize(0, Complex::<F>::default());

    let fft_2d_c2r = F::make_c2r_executor(best_width, best_height, thread_count)?;

    fft_2d_c2r
        .execute_with_scratch(&mut kernel_dst, &mut kernel_arena, &mut scratch)
        .map_err(|_| BlurError::FftChannelsNotSupported)?;

    let dst_stride = dst.row_stride() as usize;

    for (dst_chunk, src_chunk) in dst
        .data
        .borrow_mut()
        .chunks_exact_mut(dst_stride)
        .zip(kernel_arena.chunks_exact(best_width).skip(arena_pad_top))
    {
        for (dst, src) in dst_chunk
            .iter_mut()
            .zip(src_chunk.iter().skip(arena_pad_left))
        {
            *dst = src.to_();
        }
    }

    Ok(())
}
