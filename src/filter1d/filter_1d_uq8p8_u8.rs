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
use crate::filter1d::filter::create_brows;
use crate::filter1d::filter_1d_column_handler::FilterBrows;
use crate::filter1d::filter_element::KernelShape;
use crate::filter1d::filter_scan::{scan_se_1d, ScanPoint1d};
use crate::filter1d::region::FilterRegion;
use crate::{BlurError, BlurImage, BlurImageMut, EdgeMode, ImageSize, Scalar, ThreadingPolicy};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;

fn get_column_handler_multiple_rows() -> Option<
    fn(
        arena: Arena,
        FilterBrows<u16>,
        dst: &mut [u16],
        image_size: ImageSize,
        dst_stride: usize,
        scanned_kernel: &[ScanPoint1d<u16>],
    ),
> {
    None
}

fn get_column_handler() -> fn(
    arena: Arena,
    &[&[u16]],
    dst: &mut [u16],
    image_size: ImageSize,
    FilterRegion,
    scanned_kernel: &[ScanPoint1d<u16>],
) {
    use crate::filter1d::filter_uq1p15_u8::filter_column_symmetric_approx_uq8p8_u8;
    filter_column_symmetric_approx_uq8p8_u8
}

fn get_row_handler<const N: usize>() -> fn(
    arena_src: &[u16],
    dst: &mut [u16],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<u16>],
) {
    use crate::filter1d::filter_uq1p15_u8::filter_row_symmetric_approx_uq8p8_u8;
    filter_row_symmetric_approx_uq8p8_u8::<N>
}

/// Performs 2D separable approximated convolution on single plane image
///
/// This method does approximate convolution in fixed point.
///
/// # Arguments
///
/// * `image`: Single plane image
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
pub(crate) fn filter_1d_uq1p15_u16<const N: usize>(
    image: &BlurImage<u16>,
    destination: &mut BlurImageMut<u16>,
    row_kernel: &[f32],
    column_kernel: &[f32],
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout_channels(N)?;
    destination.check_layout_channels(N, Some(image))?;
    image.only_size_matches_mut(destination)?;
    if row_kernel.len() & 1 == 0 {
        return Err(BlurError::OddKernel(row_kernel.len()));
    }
    if column_kernel.len() & 1 == 0 {
        return Err(BlurError::OddKernel(column_kernel.len()));
    }

    let initial_scale: f32 = ((1 << 15) - 1) as f32;

    let scaled_row_kernel = row_kernel
        .iter()
        .map(|&x| (x * initial_scale).round().min(i16::MAX as f32).max(0.) as u16)
        .collect::<Vec<u16>>();
    let scaled_column_kernel = column_kernel
        .iter()
        .map(|&x| (x * initial_scale).round().min(i16::MAX as f32).max(0.) as u16)
        .collect::<Vec<u16>>();

    let scanned_row_kernel = scan_se_1d::<u16>(&scaled_row_kernel);
    let scanned_row_kernel_slice = scanned_row_kernel.as_slice();
    let scanned_column_kernel = scan_se_1d::<u16>(&scaled_column_kernel);
    let scanned_column_kernel_slice = scanned_column_kernel.as_slice();

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

    let mut transient_image = vec![u16::default(); image_size.width * image_size.height * N];

    if let Some(pool) = &pool {
        let row_handler = get_row_handler::<N>();
        pool.install(|| {
            transient_image
                .par_chunks_exact_mut(image_size.width * N)
                .enumerate()
                .for_each(|(y, dst_row)| {
                    let (row, _) = make_arena_row::<u16, N>(
                        image,
                        y,
                        KernelShape::new(row_kernel.len(), 0),
                        border_mode,
                        border_constant,
                    )
                    .unwrap();

                    row_handler(
                        &row,
                        dst_row,
                        image_size,
                        FilterRegion::new(y, y + 1),
                        scanned_row_kernel_slice,
                    );
                });
        });
    } else {
        let row_handler = get_row_handler::<N>();
        transient_image
            .chunks_exact_mut(image_size.width * N)
            .enumerate()
            .for_each(|(y, dst_row)| {
                let (row, _) = make_arena_row::<u16, N>(
                    image,
                    y,
                    KernelShape::new(row_kernel.len(), 0),
                    border_mode,
                    border_constant,
                )
                .unwrap();

                row_handler(
                    &row,
                    dst_row,
                    image_size,
                    FilterRegion::new(y, y + 1),
                    scanned_row_kernel_slice,
                );
            });
    }

    let column_kernel_shape = KernelShape::new(0, scanned_column_kernel_slice.len());

    let column_arena_k = make_arena_columns::<u16, N>(
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
            let column_handler = get_column_handler();
            let src_stride = image_size.width * N;
            let dst_stride = destination.row_stride() as usize;

            let mut _dest_slice = destination.data.borrow_mut();

            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            if let Some(handler) = get_column_handler_multiple_rows() {
                _dest_slice
                    .par_chunks_exact_mut(dst_stride * 3)
                    .enumerate()
                    .for_each(|(y, row)| {
                        use crate::filter1d::filter_1d_column_handler::FilterBrows;
                        let y = y * 3;
                        let brows0 = create_brows(
                            image_size,
                            column_kernel_shape,
                            top_pad,
                            bottom_pad,
                            pad_h,
                            transient_image_slice,
                            src_stride,
                            y,
                        );

                        let brows1 = create_brows(
                            image_size,
                            column_kernel_shape,
                            top_pad,
                            bottom_pad,
                            pad_h,
                            transient_image_slice,
                            src_stride,
                            y + 1,
                        );

                        let brows2 = create_brows(
                            image_size,
                            column_kernel_shape,
                            top_pad,
                            bottom_pad,
                            pad_h,
                            transient_image_slice,
                            src_stride,
                            y + 2,
                        );

                        let brows_slice0 = brows0.as_slice();
                        let brows_slice1 = brows1.as_slice();
                        let brows_slice2 = brows2.as_slice();

                        let brows_vec = vec![brows_slice0, brows_slice1, brows_slice2];

                        let brows = FilterBrows { brows: brows_vec };

                        handler(
                            Arena::new(image_size.width, pad_h, 0, pad_h, N),
                            brows,
                            row,
                            image_size,
                            dst_stride,
                            scanned_column_kernel_slice,
                        );
                    });
                _dest_slice = _dest_slice
                    .chunks_exact_mut(dst_stride * 3)
                    .into_remainder();
            }

            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            if let Some(handler) = get_column_handler_multiple_rows() {
                _dest_slice
                    .par_chunks_exact_mut(dst_stride * 2)
                    .enumerate()
                    .for_each(|(y, row)| {
                        use crate::filter1d::filter_1d_column_handler::FilterBrows;
                        let y = y * 2;
                        let brows0 = create_brows(
                            image_size,
                            column_kernel_shape,
                            top_pad,
                            bottom_pad,
                            pad_h,
                            transient_image_slice,
                            src_stride,
                            y,
                        );

                        let brows1 = create_brows(
                            image_size,
                            column_kernel_shape,
                            top_pad,
                            bottom_pad,
                            pad_h,
                            transient_image_slice,
                            src_stride,
                            y + 1,
                        );

                        let brows_slice0 = brows0.as_slice();
                        let brows_slice1 = brows1.as_slice();

                        let brows_vec = vec![brows_slice0, brows_slice1];

                        let brows = FilterBrows { brows: brows_vec };

                        handler(
                            Arena::new(image_size.width, pad_h, 0, pad_h, N),
                            brows,
                            row,
                            image_size,
                            dst_stride,
                            scanned_column_kernel_slice,
                        );
                    });
                _dest_slice = _dest_slice
                    .chunks_exact_mut(dst_stride * 2)
                    .into_remainder();
            }

            _dest_slice
                .par_chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(y, row)| {
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
                        FilterRegion::new(y, y + 1),
                        scanned_column_kernel_slice,
                    );
                });
        });
    } else {
        let column_handler = get_column_handler();
        let src_stride = image_size.width * N;
        let dst_stride = destination.row_stride() as usize;

        let mut _dest_slice = destination.data.borrow_mut();

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if let Some(handler) = get_column_handler_multiple_rows() {
            _dest_slice
                .chunks_exact_mut(dst_stride * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    use crate::filter1d::filter_1d_column_handler::FilterBrows;
                    let y = y * 3;
                    let brows0 = create_brows(
                        image_size,
                        column_kernel_shape,
                        top_pad,
                        bottom_pad,
                        pad_h,
                        transient_image_slice,
                        src_stride,
                        y,
                    );

                    let brows1 = create_brows(
                        image_size,
                        column_kernel_shape,
                        top_pad,
                        bottom_pad,
                        pad_h,
                        transient_image_slice,
                        src_stride,
                        y + 1,
                    );

                    let brows2 = create_brows(
                        image_size,
                        column_kernel_shape,
                        top_pad,
                        bottom_pad,
                        pad_h,
                        transient_image_slice,
                        src_stride,
                        y + 2,
                    );

                    let brows_slice0 = brows0.as_slice();
                    let brows_slice1 = brows1.as_slice();
                    let brows_slice2 = brows2.as_slice();

                    let brows_vec = vec![brows_slice0, brows_slice1, brows_slice2];

                    let brows = FilterBrows { brows: brows_vec };

                    handler(
                        Arena::new(image_size.width, pad_h, 0, pad_h, N),
                        brows,
                        row,
                        image_size,
                        dst_stride,
                        scanned_column_kernel_slice,
                    );
                });
            _dest_slice = _dest_slice
                .chunks_exact_mut(dst_stride * 3)
                .into_remainder();
        }

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        if let Some(handler) = get_column_handler_multiple_rows() {
            _dest_slice
                .chunks_exact_mut(dst_stride * 2)
                .enumerate()
                .for_each(|(y, row)| {
                    use crate::filter1d::filter_1d_column_handler::FilterBrows;
                    let y = y * 2;
                    let brows0 = create_brows(
                        image_size,
                        column_kernel_shape,
                        top_pad,
                        bottom_pad,
                        pad_h,
                        transient_image_slice,
                        src_stride,
                        y,
                    );

                    let brows1 = create_brows(
                        image_size,
                        column_kernel_shape,
                        top_pad,
                        bottom_pad,
                        pad_h,
                        transient_image_slice,
                        src_stride,
                        y + 1,
                    );

                    let brows_slice0 = brows0.as_slice();
                    let brows_slice1 = brows1.as_slice();

                    let brows_vec = vec![brows_slice0, brows_slice1];

                    let brows = FilterBrows { brows: brows_vec };

                    handler(
                        Arena::new(image_size.width, pad_h, 0, pad_h, N),
                        brows,
                        row,
                        image_size,
                        dst_stride,
                        scanned_column_kernel_slice,
                    );
                });
            _dest_slice = _dest_slice
                .chunks_exact_mut(dst_stride * 2)
                .into_remainder();
        }

        _dest_slice
            .chunks_exact_mut(dst_stride)
            .enumerate()
            .for_each(|(y, row)| {
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
                    FilterRegion::new(y, y + 1),
                    scanned_column_kernel_slice,
                );
            });
    }

    Ok(())
}
