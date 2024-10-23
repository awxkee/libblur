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
use crate::filter1d::arena::{make_arena_columns, make_arena_row, Arena};
use crate::filter1d::filter_1d_column_handler::Filter1DColumnHandler;
use crate::filter1d::filter_1d_rgba_row_handler::Filter1DRgbaRowHandler;
use crate::filter1d::filter_element::KernelShape;
use crate::filter1d::filter_scan::{is_symmetric_1d, scan_se_1d};
use crate::filter1d::region::FilterRegion;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::{EdgeMode, ImageSize, Scalar, ThreadingPolicy};
use num_traits::{AsPrimitive, MulAdd};
use std::ops::Mul;

/// Performs 2D separable convolution on RGBA image
///
/// This method does exact convolution on the image without any approximations using required
/// intermediate type based on kernel data type.
/// Note, in most cases for convolution alpha must be premultiplied.
///
/// # Arguments
///
/// * `image`: Single RGBA image
/// * `destination`: Destination RGBA image
/// * `image_size`: Image size see [ImageSize]
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
pub fn filter_1d_rgba_exact<T, F>(
    image: &[T],
    destination: &mut [T],
    image_size: ImageSize,
    row_kernel: &[F],
    column_kernel: &[F],
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), String>
where
    T: Copy
        + AsPrimitive<F>
        + Default
        + Send
        + Sync
        + Filter1DRgbaRowHandler<T, F>
        + Filter1DColumnHandler<T, F>,
    F: ToStorage<T> + Mul<F> + MulAdd<F, Output = F> + Send + Sync + PartialEq + Default,
    i32: AsPrimitive<F>,
    f64: AsPrimitive<T>,
{
    if image.len() != 4 * image_size.width * image_size.height {
        return Err(format!(
            "Can't create arena, expected image with size {} but got {}",
            4 * image_size.width * image_size.height,
            image.len()
        ));
    }
    if destination.len() != 4 * image_size.width * image_size.height {
        return Err(format!(
            "Can't create arena, expected image with size {} but got {}",
            4 * image_size.width * image_size.height,
            destination.len()
        ));
    }
    if row_kernel.len() & 1 == 0 {
        return Err(String::from("Row kernel length must be odd"));
    }
    if column_kernel.len() & 1 == 0 {
        return Err(String::from("Column kernel length must be odd"));
    }

    let scanned_row_kernel = scan_se_1d(row_kernel);
    let scanned_row_kernel_slice = scanned_row_kernel.as_slice();
    let scanned_column_kernel = scan_se_1d(column_kernel);
    let scanned_column_kernel_slice = scanned_column_kernel.as_slice();
    let is_column_kernel_symmetrical = is_symmetric_1d(column_kernel);
    let is_row_column_kernel_symmetrical = is_symmetric_1d(row_kernel);

    let thread_count = threading_policy
        .get_threads_count(image_size.width as u32, image_size.height as u32)
        as u32;
    let pool = if thread_count == 1 {
        None
    } else {
        let hold = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count as usize)
            .build()
            .unwrap();
        Some(hold)
    };

    const N: usize = 4;

    let mut transient_image = vec![T::default(); image_size.width * image_size.height * N];

    if let Some(pool) = &pool {
        let row_handler = T::get_rgba_row_handler(is_row_column_kernel_symmetrical);
        pool.scope(|scope| {
            let transient_cell = UnsafeSlice::new(transient_image.as_mut_slice());

            let pad_w = scanned_row_kernel.len() / 2;

            for y in 0..image_size.height {
                scope.spawn(move |_| {
                    let (row, arena_width) = make_arena_row::<T, N>(
                        image,
                        y,
                        image_size,
                        KernelShape::new(row_kernel.len(), 0),
                        border_mode,
                        border_constant,
                    )
                    .unwrap();

                    row_handler(
                        Arena::new(arena_width, 1, pad_w, 0, N),
                        &row,
                        &transient_cell,
                        image_size,
                        FilterRegion::new(y, y + 1),
                        scanned_row_kernel_slice,
                    );
                });
            }
        });
    } else {
        let pad_w = scanned_row_kernel.len() / 2;

        let row_handler = T::get_rgba_row_handler(is_row_column_kernel_symmetrical);
        let transient_cell = UnsafeSlice::new(transient_image.as_mut_slice());

        for y in 0..image_size.height {
            let (row, arena_width) = make_arena_row::<T, N>(
                image,
                y,
                image_size,
                KernelShape::new(row_kernel.len(), 0),
                border_mode,
                border_constant,
            )?;

            row_handler(
                Arena::new(arena_width, 1, pad_w, 0, N),
                &row,
                &transient_cell,
                image_size,
                FilterRegion::new(y, y + 1),
                scanned_row_kernel_slice,
            );
        }
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
        pool.scope(|scope| {
            let transient_cell_0 = UnsafeSlice::new(destination);
            let column_handler = T::get_column_handler(is_column_kernel_symmetrical);

            let src_stride = image_size.width * N;

            for y in 0..image_size.height {
                scope.spawn(move |_| {
                    let mut brows: Vec<&[T]> = vec![&image[0..]; column_kernel_shape.height];

                    for k in 0..column_kernel_shape.height {
                        if (y as i64 - pad_h as i64 + k as i64) < 0 {
                            brows[k] = &top_pad[(pad_h - k - 1) * src_stride..];
                        } else if (y as i64 - pad_h as i64 + k as i64) as usize >= image_size.height
                        {
                            brows[k] = &bottom_pad[(k - pad_h - 1) * src_stride..];
                        } else {
                            let fy = (y as i64 + k as i64 - pad_h as i64) as usize;
                            let start_offset = src_stride * fy;

                            brows[k] =
                                &transient_image_slice[start_offset..(start_offset + src_stride)];
                        }
                    }

                    let brows_slice = brows.as_slice();

                    column_handler(
                        Arena::new(image_size.width, pad_h, 0, pad_h, N),
                        brows_slice,
                        &transient_cell_0,
                        image_size,
                        FilterRegion::new(y, y + 1),
                        scanned_column_kernel_slice,
                    );
                });
            }
        });
    } else {
        let column_handler = T::get_column_handler(is_column_kernel_symmetrical);
        let final_cell_0 = UnsafeSlice::new(destination);
        let src_stride = image_size.width * N;
        for y in 0..image_size.height {
            let mut brows: Vec<&[T]> = vec![&image[0..]; column_kernel_shape.height];

            for k in 0..column_kernel_shape.height {
                if (y as i64 - pad_h as i64 + k as i64) < 0 {
                    brows[k] = &top_pad[(pad_h - k - 1) * src_stride..];
                } else if (y as i64 - pad_h as i64 + k as i64) as usize >= image_size.height {
                    brows[k] = &bottom_pad[(k - pad_h - 1) * src_stride..];
                } else {
                    let fy = (y as i64 + k as i64 - pad_h as i64) as usize;
                    let start_offset = src_stride * fy;

                    brows[k] = &transient_image_slice[start_offset..(start_offset + src_stride)];
                }
            }

            let brows_slice = brows.as_slice();

            column_handler(
                Arena::new(image_size.width, pad_h, 0, pad_h, N),
                brows_slice,
                &final_cell_0,
                image_size,
                FilterRegion::new(y, y + 1),
                scanned_column_kernel_slice,
            );
        }
    }

    Ok(())
}
