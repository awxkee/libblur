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
use crate::edge_mode::clamp_edge;
use crate::filter1d::arena::{make_arena_columns, make_arena_row, write_arena_row, Arena};
use crate::filter1d::filter::create_brows;
use crate::filter1d::filter_complex_dispatch::ComplexDispatch;
use crate::filter1d::filter_element::KernelShape;
use crate::filter1d::filter_scan::is_symmetric_1d;
use crate::safe_math::{SafeAdd, SafeMul};
use crate::to_storage::ToStorage;
use crate::{BlurError, BlurImage, BlurImageMut, EdgeMode, Scalar, ThreadingPolicy};
use novtb::{ParallelZonedIterator, TbSliceMut};
use num_complex::Complex;
use num_traits::{AsPrimitive, FromPrimitive, MulAdd, Num};
use std::fmt::Debug;
use std::ops::Mul;

/// Performs 2D separable convolution on the image with complex domain kernel.
///
/// This method does exact convolution on the image without any approximations using required
/// intermediate type based on kernel data type.
///
/// # Arguments
///
/// * `image`: Source image.
/// * `destination`: Destination image.
/// * `row_kernel`: Row kernel, *size must be odd*!
/// * `column_kernel`: Column kernel, *size must be odd*!
/// * `border_mode`: See [EdgeMode] for more info
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value
/// * `threading_policy`: See [ThreadingPolicy] for more info
///
/// returns: Result<(), String>
///
/// # Examples
///
/// See [crate::gaussian_blur] for example
///
pub fn filter_1d_complex<T, F, const N: usize>(
    image: &BlurImage<T>,
    destination: &mut BlurImageMut<T>,
    row_kernel: &[Complex<F>],
    column_kernel: &[Complex<F>],
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + ComplexDispatch<T, F> + Debug,
    F: ToStorage<T>
        + Mul<F>
        + MulAdd<F, Output = F>
        + Send
        + Sync
        + PartialEq
        + Default
        + FromPrimitive
        + Num,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T>,
{
    const SMALL_KERNEL_CUTOFF: usize = 61;
    if column_kernel.len() <= SMALL_KERNEL_CUTOFF {
        return filter_1d_complex_sliding_buffer::<T, F, N>(
            image,
            destination,
            row_kernel,
            column_kernel,
            border_mode,
            border_constant,
            threading_policy,
        );
    }

    image.check_layout_channels(N)?;
    destination.check_layout_channels(N, Some(image))?;
    image.only_size_matches_mut(destination)?;

    if row_kernel.len() & 1 == 0 {
        return Err(BlurError::OddKernel(row_kernel.len()));
    }
    if column_kernel.len() & 1 == 0 {
        return Err(BlurError::OddKernel(column_kernel.len()));
    }

    _ = column_kernel.len().safe_mul(image.height as usize)?;

    let pad_w = (row_kernel.len() / 2).max(1);
    _ = (image.width as usize)
        .safe_mul(N)?
        .safe_add(pad_w.safe_mul(2 * N)?);

    _ = (image.width as usize)
        .safe_mul(image.height as usize)?
        .safe_mul(N)?;

    _ = (destination.stride as usize).safe_mul(3)?;

    let is_column_kernel_symmetrical = is_symmetric_1d::<Complex<F>>(column_kernel);
    let is_row_kernel_symmetrical = is_symmetric_1d::<Complex<F>>(row_kernel);

    let image_size = image.size();

    let thread_count =
        threading_policy.thread_count(image_size.width as u32, image_size.height as u32) as u32;
    let pool = novtb::ThreadPool::new(thread_count as usize);

    let mut transient_image =
        vec![Complex::<F>::default(); image_size.width * image_size.height * N];

    let row_handler = T::row_dispatch(is_row_kernel_symmetrical);
    transient_image
        .tb_par_chunks_exact_mut(image_size.width * N)
        .for_each_enumerated(&pool, |y, dst_row| {
            let pad_w = row_kernel.len() / 2;
            let (row, arena_width) = make_arena_row::<T, N>(
                image,
                y,
                KernelShape::new(row_kernel.len(), 0),
                border_mode,
                border_constant,
            )
            .unwrap();

            row_handler(
                Arena::new(arena_width, 1, pad_w, 0, N),
                &row,
                dst_row,
                image_size,
                row_kernel,
            );
        });

    let column_kernel_shape = KernelShape::new(0, column_kernel.len());

    let column_arena_k = make_arena_columns::<Complex<F>, N>(
        transient_image.as_slice(),
        image_size,
        column_kernel_shape,
        border_mode,
        border_constant,
    )?;

    let top_pad = column_arena_k.top_pad.as_slice();
    let bottom_pad = column_arena_k.bottom_pad.as_slice();

    let pad_h = column_kernel_shape.height / 2;

    let transient_image_slice = transient_image.as_slice();

    let column_handler = T::column_dispatch(is_column_kernel_symmetrical);

    let src_stride = image_size.width * N;
    let dst_stride = destination.row_stride() as usize;

    let mut _dest_slice = destination.data.borrow_mut();

    _dest_slice
        .tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(&pool, |y, row| {
            let brows = create_brows(
                image_size,
                column_kernel_shape,
                top_pad,
                bottom_pad,
                pad_h,
                transient_image_slice,
                src_stride,
                y,
            );

            let brows_slice = brows.as_slice();
            let row = &mut row[..image_size.width * N];

            column_handler(
                Arena::new(image_size.width, pad_h, 0, pad_h, N),
                brows_slice,
                row,
                image_size,
                column_kernel,
            );
        });

    Ok(())
}

fn filter_1d_complex_sliding_buffer<T, F, const N: usize>(
    image: &BlurImage<T>,
    destination: &mut BlurImageMut<T>,
    row_kernel: &[Complex<F>],
    column_kernel: &[Complex<F>],
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy + AsPrimitive<F> + Default + Send + Sync + ComplexDispatch<T, F> + Debug,
    F: ToStorage<T> + Mul<F> + MulAdd<F, Output = F> + Send + Sync + PartialEq + Default,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T>,
{
    image.check_layout_channels(N)?;
    destination.check_layout_channels(N, Some(image))?;
    image.only_size_matches_mut(destination)?;
    if row_kernel.len() & 1 == 0 {
        return Err(BlurError::OddKernel(row_kernel.len()));
    }
    if column_kernel.len() & 1 == 0 {
        return Err(BlurError::OddKernel(column_kernel.len()));
    }

    _ = column_kernel.len().safe_mul(image.height as usize)?;

    let pad_w = (row_kernel.len() / 2).max(1);
    _ = (image.width as usize)
        .safe_mul(N)?
        .safe_add(pad_w.safe_mul(2 * N)?);

    let is_column_kernel_symmetrical = is_symmetric_1d::<Complex<F>>(column_kernel);
    let is_row_kernel_symmetrical = is_symmetric_1d::<Complex<F>>(row_kernel);

    let image_size = image.size();

    let thread_count =
        threading_policy.thread_count(image_size.width as u32, image_size.height as u32) as u32;

    let pool = novtb::ThreadPool::new(thread_count as usize);

    let tile_size = (image_size.height as u32 / thread_count).clamp(1, image_size.height as u32);

    let row_handler = T::row_dispatch(is_row_kernel_symmetrical);
    let column_handler = T::column_dispatch(is_column_kernel_symmetrical);

    let row_stride = image_size.width * N;

    let dest_stride = destination.row_stride() as usize;

    if thread_count > 1 {
        destination
            .data
            .borrow_mut()
            .tb_par_chunks_mut(dest_stride * tile_size as usize)
            .for_each_enumerated(&pool, |cy, dst_rows| {
                let source_y = cy * tile_size as usize;
                let mut buffer = vec![Complex::<F>::default(); row_stride * column_kernel.len()];

                let pad_w = row_kernel.len() / 2;
                let mut row_buffer = vec![T::default(); image_size.width * N + pad_w * 2 * N];

                let column_kernel_len = column_kernel.len();

                let mut start_ky = column_kernel_len / 2 + 1;

                start_ky %= column_kernel_len;

                let half_kernel = column_kernel_len / 2;

                // preload top edge
                if source_y == 0 {
                    let pad_w = row_kernel.len() / 2;
                    if row_buffer.is_empty() {
                        row_buffer = vec![T::default(); image_size.width * N + pad_w * 2 * N];
                    }
                    write_arena_row::<T, N>(
                        &mut row_buffer,
                        image,
                        0,
                        KernelShape::new(row_kernel.len(), 0),
                        border_mode,
                        border_constant,
                    )
                    .unwrap();
                    row_handler(
                        Arena::new(image_size.width, 1, row_kernel.len() / 2, 0, N),
                        &row_buffer,
                        &mut buffer[..row_stride],
                        image_size,
                        row_kernel,
                    );

                    let (src_row, rest) = buffer.split_at_mut(row_stride);
                    for dst in rest.chunks_exact_mut(row_stride).take(half_kernel) {
                        for (dst, src) in dst.iter_mut().zip(src_row.iter()) {
                            *dst = *src;
                        }
                    }
                } else {
                    for src_y in 0..=half_kernel {
                        let s_y = clamp_edge!(
                            border_mode,
                            src_y as i64 + source_y as i64 - half_kernel as i64 - 1,
                            0i64,
                            image_size.height as i64
                        );
                        let pad_w = row_kernel.len() / 2;
                        if row_buffer.is_empty() {
                            row_buffer = vec![T::default(); image_size.width * N + pad_w * 2 * N];
                        }
                        write_arena_row::<T, N>(
                            &mut row_buffer,
                            image,
                            s_y,
                            KernelShape::new(row_kernel.len(), 0),
                            border_mode,
                            border_constant,
                        )
                        .unwrap();
                        row_handler(
                            Arena::new(image_size.width, 1, row_kernel.len() / 2, 0, N),
                            &row_buffer,
                            &mut buffer[src_y * row_stride..(src_y + 1) * row_stride],
                            image_size,
                            row_kernel,
                        );
                    }
                }

                let rows_count = dst_rows.len() / dest_stride;

                for (y, dy) in
                    (source_y..source_y + rows_count + half_kernel).zip(0..rows_count + half_kernel)
                {
                    let new_y = if y < image_size.height {
                        y
                    } else {
                        clamp_edge!(border_mode, y as i64, 0i64, image_size.height as i64)
                    };

                    write_arena_row::<T, N>(
                        &mut row_buffer,
                        image,
                        new_y,
                        KernelShape::new(row_kernel.len(), 0),
                        border_mode,
                        border_constant,
                    )
                    .unwrap();
                    row_handler(
                        Arena::new(image_size.width, 1, row_kernel.len() / 2, 0, N),
                        &row_buffer,
                        &mut buffer[start_ky * row_stride..(start_ky + 1) * row_stride],
                        image_size,
                        row_kernel,
                    );

                    if dy >= half_kernel {
                        let mut brows = vec![buffer.as_slice(); column_kernel_len];

                        for (i, brow) in brows.iter_mut().enumerate() {
                            let ky = (i + start_ky + 1) % column_kernel_len;
                            *brow = &buffer[ky * row_stride..(ky + 1) * row_stride];
                        }

                        let dy = dy - half_kernel;

                        let dst = &mut dst_rows[dy * dest_stride..(dy + 1) * dest_stride];

                        column_handler(
                            Arena::new(image_size.width, half_kernel, 0, half_kernel, N),
                            &brows,
                            dst,
                            image_size,
                            column_kernel,
                        );
                    }

                    start_ky += 1;
                    start_ky %= column_kernel_len;
                }
            });
    } else {
        let mut buffer = vec![Complex::<F>::default(); row_stride * column_kernel.len()];

        let pad_w = row_kernel.len() / 2;
        let mut row_buffer = vec![T::default(); image_size.width * N + pad_w * 2 * N];

        // preload top edge
        write_arena_row::<T, N>(
            &mut row_buffer,
            image,
            0,
            KernelShape::new(row_kernel.len(), 0),
            border_mode,
            border_constant,
        )?;
        row_handler(
            Arena::new(image_size.width, 1, row_kernel.len() / 2, 0, N),
            &row_buffer,
            &mut buffer[..row_stride],
            image_size,
            row_kernel,
        );

        let column_kernel_len = column_kernel.len();

        let half_kernel = column_kernel_len / 2;

        let (src_row, rest) = buffer.split_at_mut(row_stride);
        for dst in rest.chunks_exact_mut(row_stride).take(half_kernel) {
            for (dst, src) in dst.iter_mut().zip(src_row.iter()) {
                *dst = *src;
            }
        }

        let mut start_ky = column_kernel_len / 2 + 1;

        start_ky %= column_kernel_len;

        for y in 1..image_size.height + half_kernel {
            let new_y = if y < image_size.height {
                y
            } else {
                clamp_edge!(border_mode, y as i64, 0i64, image_size.height as i64)
            };

            write_arena_row::<T, N>(
                &mut row_buffer,
                image,
                new_y,
                KernelShape::new(row_kernel.len(), 0),
                border_mode,
                border_constant,
            )?;
            row_handler(
                Arena::new(image_size.width, 1, row_kernel.len() / 2, 0, N),
                &row_buffer,
                &mut buffer[start_ky * row_stride..(start_ky + 1) * row_stride],
                image_size,
                row_kernel,
            );

            if y >= half_kernel {
                let mut brows = vec![buffer.as_slice(); column_kernel_len];

                for (i, brow) in brows.iter_mut().enumerate() {
                    let ky = (i + start_ky + 1) % column_kernel_len;
                    *brow = &buffer[ky * row_stride..(ky + 1) * row_stride];
                }

                let dy = y - half_kernel;

                let dst =
                    &mut destination.data.borrow_mut()[dy * dest_stride..(dy + 1) * dest_stride];

                column_handler(
                    Arena::new(image_size.width, half_kernel, 0, half_kernel, N),
                    &brows,
                    dst,
                    image_size,
                    column_kernel,
                );
            }

            start_ky += 1;
            start_ky %= column_kernel_len;
        }
    }

    Ok(())
}
