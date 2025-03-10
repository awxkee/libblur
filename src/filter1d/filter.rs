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
use crate::filter1d::arena::{make_arena_columns, make_arena_row, Arena};
use crate::filter1d::filter_1d_column_handler::{
    Filter1DColumnHandler, Filter1DColumnHandlerMultipleRows, FilterBrows,
};
use crate::filter1d::filter_1d_row_handler::Filter1DRowHandler;
use crate::filter1d::filter_element::KernelShape;
use crate::filter1d::filter_scan::{is_symmetric_1d, scan_se_1d};
use crate::filter1d::region::FilterRegion;
use crate::to_storage::ToStorage;
use crate::{BlurError, BlurImage, BlurImageMut, EdgeMode, ImageSize, Scalar, ThreadingPolicy};
use num_traits::{AsPrimitive, MulAdd};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use std::fmt::Debug;
use std::ops::Mul;

/// Performs 2D separable convolution on single plane image
///
/// This method does exact convolution on the image without any approximations using required
/// intermediate type based on kernel data type.
///
/// # Arguments
///
/// * `image`: Source image
/// * `destination`: Destination image
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
pub fn filter_1d_exact<T, F, const N: usize>(
    image: &BlurImage<T>,
    destination: &mut BlurImageMut<T>,
    row_kernel: &[F],
    column_kernel: &[F],
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    T: Copy
        + AsPrimitive<F>
        + Default
        + Send
        + Sync
        + Filter1DRowHandler<T, F>
        + Filter1DColumnHandler<T, F>
        + Debug
        + Filter1DColumnHandlerMultipleRows<T, F>,
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

    let scanned_row_kernel = scan_se_1d(row_kernel);
    let scanned_row_kernel_slice = scanned_row_kernel.as_slice();
    let scanned_column_kernel = scan_se_1d(column_kernel);
    let scanned_column_kernel_slice = scanned_column_kernel.as_slice();
    let is_column_kernel_symmetrical = is_symmetric_1d(column_kernel);
    let is_row_kernel_symmetrical = is_symmetric_1d(row_kernel);

    let image_size = image.size();

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

    let mut transient_image = vec![T::default(); image_size.width * image_size.height * N];

    if let Some(pool) = &pool {
        let row_handler = T::get_row_handler::<N>(is_row_kernel_symmetrical);
        pool.install(|| {
            transient_image
                .par_chunks_exact_mut(image_size.width * N)
                .enumerate()
                .for_each(|(y, dst_row)| {
                    let pad_w = scanned_row_kernel.len() / 2;
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
                        FilterRegion::new(y, y + 1),
                        scanned_row_kernel_slice,
                    );
                });
        });
    } else {
        let row_handler = T::get_row_handler::<N>(is_row_kernel_symmetrical);
        transient_image
            .chunks_exact_mut(image_size.width * N)
            .enumerate()
            .for_each(|(y, dst_row)| {
                let pad_w = scanned_row_kernel.len() / 2;
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
                    FilterRegion::new(y, y + 1),
                    scanned_row_kernel_slice,
                );
            });
    }

    let column_kernel_shape = KernelShape::new(0, scanned_column_kernel_slice.len());

    let column_arena_k = make_arena_columns::<T, N>(
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

    if let Some(pool) = &pool {
        pool.install(|| {
            let column_handler = T::get_column_handler(is_column_kernel_symmetrical);
            let column_multiple_rows =
                T::get_column_handler_multiple_rows(is_column_kernel_symmetrical);

            let src_stride = image_size.width * N;
            let dst_stride = destination.row_stride() as usize;

            let mut dest_slice = destination.data.borrow_mut();

            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            if let Some(column_multiple_rows) = column_multiple_rows {
                dest_slice
                    .par_chunks_exact_mut(dst_stride * 3)
                    .enumerate()
                    .for_each(|(y, row)| {
                        let y = y * 3;
                        let brows0 = create_brows(
                            image_size,
                            column_kernel_shape,
                            &top_pad,
                            &bottom_pad,
                            pad_h,
                            &transient_image_slice,
                            src_stride,
                            y,
                        );

                        let brows1 = create_brows(
                            image_size,
                            column_kernel_shape,
                            &top_pad,
                            &bottom_pad,
                            pad_h,
                            &transient_image_slice,
                            src_stride,
                            y + 1,
                        );

                        let brows2 = create_brows(
                            image_size,
                            column_kernel_shape,
                            &top_pad,
                            &bottom_pad,
                            pad_h,
                            &transient_image_slice,
                            src_stride,
                            y + 2,
                        );

                        let brows_slice0 = brows0.as_slice();
                        let brows_slice1 = brows1.as_slice();
                        let brows_slice2 = brows2.as_slice();

                        let brows_vec = vec![brows_slice0, brows_slice1, brows_slice2];

                        let brows = FilterBrows { brows: brows_vec };

                        column_multiple_rows(
                            Arena::new(image_size.width, pad_h, 0, pad_h, N),
                            brows,
                            row,
                            image_size,
                            dst_stride,
                            scanned_column_kernel_slice,
                        );
                    });
                dest_slice = dest_slice.chunks_exact_mut(dst_stride * 3).into_remainder();
            }

            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            if let Some(column_multiple_rows) = column_multiple_rows {
                dest_slice
                    .par_chunks_exact_mut(dst_stride * 2)
                    .enumerate()
                    .for_each(|(y, row)| {
                        let y = y * 2;
                        let brows0 = create_brows(
                            image_size,
                            column_kernel_shape,
                            &top_pad,
                            &bottom_pad,
                            pad_h,
                            &transient_image_slice,
                            src_stride,
                            y,
                        );

                        let brows1 = create_brows(
                            image_size,
                            column_kernel_shape,
                            &top_pad,
                            &bottom_pad,
                            pad_h,
                            &transient_image_slice,
                            src_stride,
                            y + 1,
                        );

                        let brows_slice0 = brows0.as_slice();
                        let brows_slice1 = brows1.as_slice();

                        let brows_vec = vec![brows_slice0, brows_slice1];

                        let brows = FilterBrows { brows: brows_vec };

                        column_multiple_rows(
                            Arena::new(image_size.width, pad_h, 0, pad_h, N),
                            brows,
                            row,
                            image_size,
                            dst_stride,
                            scanned_column_kernel_slice,
                        );
                    });
                dest_slice = dest_slice.chunks_exact_mut(dst_stride * 2).into_remainder();
            }

            dest_slice
                .par_chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(y, row)| {
                    let brows = create_brows(
                        image_size,
                        column_kernel_shape,
                        &top_pad,
                        &bottom_pad,
                        pad_h,
                        &transient_image_slice,
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
                        FilterRegion::new(y, y + 1),
                        scanned_column_kernel_slice,
                    );
                });
        });
    } else {
        let column_handler = T::get_column_handler(is_column_kernel_symmetrical);
        let column_multiple_rows =
            T::get_column_handler_multiple_rows(is_column_kernel_symmetrical);

        let src_stride = image_size.width * N;
        let dst_stride = destination.row_stride() as usize;

        let mut dest_slice = destination.data.borrow_mut();

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        if let Some(column_multiple_rows) = column_multiple_rows {
            dest_slice
                .chunks_exact_mut(dst_stride * 2)
                .enumerate()
                .for_each(|(y, row)| {
                    let y = y * 2;
                    let brows0 = create_brows(
                        image_size,
                        column_kernel_shape,
                        &top_pad,
                        &bottom_pad,
                        pad_h,
                        &transient_image_slice,
                        src_stride,
                        y,
                    );

                    let brows1 = create_brows(
                        image_size,
                        column_kernel_shape,
                        &top_pad,
                        &bottom_pad,
                        pad_h,
                        &transient_image_slice,
                        src_stride,
                        y + 1,
                    );

                    let brows_slice0 = brows0.as_slice();
                    let brows_slice1 = brows1.as_slice();

                    let brows_vec = vec![brows_slice0, brows_slice1];

                    let brows = FilterBrows { brows: brows_vec };

                    column_multiple_rows(
                        Arena::new(image_size.width, pad_h, 0, pad_h, N),
                        brows,
                        row,
                        image_size,
                        dst_stride,
                        scanned_column_kernel_slice,
                    );
                });
            dest_slice = dest_slice.chunks_exact_mut(dst_stride * 2).into_remainder();
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if let Some(column_multiple_rows) = column_multiple_rows {
            dest_slice
                .chunks_exact_mut(dst_stride * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    let y = y * 3;
                    let brows0 = create_brows(
                        image_size,
                        column_kernel_shape,
                        &top_pad,
                        &bottom_pad,
                        pad_h,
                        &transient_image_slice,
                        src_stride,
                        y,
                    );

                    let brows1 = create_brows(
                        image_size,
                        column_kernel_shape,
                        &top_pad,
                        &bottom_pad,
                        pad_h,
                        &transient_image_slice,
                        src_stride,
                        y + 1,
                    );

                    let brows2 = create_brows(
                        image_size,
                        column_kernel_shape,
                        &top_pad,
                        &bottom_pad,
                        pad_h,
                        &transient_image_slice,
                        src_stride,
                        y + 2,
                    );

                    let brows_slice0 = brows0.as_slice();
                    let brows_slice1 = brows1.as_slice();
                    let brows_slice2 = brows2.as_slice();

                    let brows_vec = vec![brows_slice0, brows_slice1, brows_slice2];

                    let brows = FilterBrows { brows: brows_vec };

                    column_multiple_rows(
                        Arena::new(image_size.width, pad_h, 0, pad_h, N),
                        brows,
                        row,
                        image_size,
                        dst_stride,
                        scanned_column_kernel_slice,
                    );
                });
            dest_slice = dest_slice.chunks_exact_mut(dst_stride * 3).into_remainder();
        }

        dest_slice
            .chunks_exact_mut(dst_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let brows = create_brows(
                    image_size,
                    column_kernel_shape,
                    &top_pad,
                    &bottom_pad,
                    pad_h,
                    &transient_image_slice,
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
                    FilterRegion::new(y, y + 1),
                    scanned_column_kernel_slice,
                );
            });
    }

    Ok(())
}

fn create_brows<'a, T>(
    image_size: ImageSize,
    column_kernel_shape: KernelShape,
    top_pad: &'a [T],
    bottom_pad: &'a [T],
    pad_h: usize,
    transient_image_slice: &'a [T],
    src_stride: usize,
    y: usize,
) -> Vec<&'a [T]> {
    let mut brows: Vec<&[T]> = vec![&transient_image_slice[0..]; column_kernel_shape.height];

    for (k, row) in (0..column_kernel_shape.height).zip(brows.iter_mut()) {
        if (y as i64 - pad_h as i64 + k as i64) < 0 {
            *row = &top_pad[(pad_h - k - 1) * src_stride..];
        } else if (y as i64 - pad_h as i64 + k as i64) as usize >= image_size.height {
            *row = &bottom_pad[(k - pad_h - 1) * src_stride..];
        } else {
            let fy = (y as i64 + k as i64 - pad_h as i64) as usize;
            let start_offset = src_stride * fy;
            *row = &transient_image_slice[start_offset..(start_offset + src_stride)];
        }
    }
    brows
}
