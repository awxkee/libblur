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
use crate::filter1d::arena::make_arena;
use crate::filter1d::filter_1d_column_handler_approx::Filter1DColumnHandlerApprox;
use crate::filter1d::filter_1d_rgba_row_handler_approx::Filter1DRgbaRowHandlerApprox;
use crate::filter1d::filter_element::KernelShape;
use crate::filter1d::filter_scan::scan_se_1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::to_approx_storage::{ApproxLevel, ToApproxStorage};
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::{EdgeMode, ImageSize, ThreadingPolicy};
use num_traits::{AsPrimitive, Float, MulAdd};
use std::ops::{Add, Mul, Shl, Shr};

pub fn filter_2d_rgba_approx<T, F, I>(
    image: &[T],
    destination: &mut [T],
    image_size: ImageSize,
    row_kernel: &[F],
    column_kernel: &[F],
    border_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) -> Result<(), String>
where
    T: Copy
        + AsPrimitive<F>
        + Default
        + Send
        + Sync
        + Filter1DRgbaRowHandlerApprox<T, I>
        + Filter1DColumnHandlerApprox<T, I>,
    F: ToStorage<T> + Mul<F> + MulAdd<F, Output = F> + Send + Sync + AsPrimitive<I> + Float,
    I: Copy
        + Mul<Output = I>
        + Add<Output = I>
        + Shr<I, Output = I>
        + Default
        + 'static
        + ToApproxStorage<T>
        + AsPrimitive<F>
        + PartialEq
        + Sync
        + Send
        + ApproxLevel
        + Shl<Output = I>,
    i32: AsPrimitive<F> + AsPrimitive<I>,
    i64: AsPrimitive<I> + AsPrimitive<F>,
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
    if border_mode == EdgeMode::KernelClip {
        return Err(String::from(
            "Border mode KernelClip is not supported in filter 1d",
        ));
    }

    let one_i: I = 1.as_();
    let base_level: I = one_i << I::approx_level().as_();
    let initial_scale: F = base_level.as_();

    let scaled_row_kernel = row_kernel
        .iter()
        .map(|&x| (x * initial_scale).round().as_())
        .collect::<Vec<I>>();
    let scaled_column_kernel = column_kernel
        .iter()
        .map(|&x| (x * initial_scale).round().as_())
        .collect::<Vec<I>>();

    let scanned_row_kernel = unsafe { scan_se_1d::<I>(&scaled_row_kernel) };
    let scanned_row_kernel_slice = scanned_row_kernel.as_slice();
    let scanned_column_kernel = unsafe { scan_se_1d::<I>(&scaled_column_kernel) };
    let scanned_column_kernel_slice = scanned_column_kernel.as_slice();

    let (mut row_arena_src, arena) = make_arena::<T, 4>(
        image,
        image_size,
        KernelShape::new(row_kernel.len(), 0),
        border_mode,
    )?;
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

    let row_arena_src_slice = row_arena_src.as_slice();

    if let Some(pool) = &pool {
        let row_handler = T::get_rgba_row_handler();
        pool.scope(|scope| {
            let transient_cell = UnsafeSlice::new(destination);

            let segment_size = image_size.height as u32 / thread_count;
            for i in 0..thread_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_y = image_size.height as u32;
                }

                let copied_arena = arena;

                scope.spawn(move |_| {
                    row_handler(
                        copied_arena,
                        row_arena_src_slice,
                        &transient_cell,
                        image_size,
                        FilterRegion::new(start_y as usize, end_y as usize),
                        scanned_row_kernel_slice,
                    );
                });
            }
        });
    } else {
        let row_handler = T::get_rgba_row_handler();
        let transient_cell = UnsafeSlice::new(destination);
        row_handler(
            arena,
            row_arena_src_slice,
            &transient_cell,
            image_size,
            FilterRegion::new(0usize, image_size.height),
            scanned_row_kernel_slice,
        );
    }

    row_arena_src.clear();

    let (mut column_arena_src, column_arena) = make_arena::<T, 4>(
        destination,
        image_size,
        KernelShape::new(0, scanned_column_kernel_slice.len()),
        border_mode,
    )?;
    let column_arena_src_slice = column_arena_src.as_slice();

    if let Some(pool) = &pool {
        pool.scope(|scope| {
            let transient_cell_0 = UnsafeSlice::new(destination);
            let column_handler = T::get_column_handler();
            let copied_arena = column_arena;
            let segment_size = image_size.height as u32 / thread_count;
            for i in 0..thread_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_y = image_size.height as u32;
                }

                scope.spawn(move |_| {
                    column_handler(
                        copied_arena,
                        column_arena_src_slice,
                        &transient_cell_0,
                        image_size,
                        FilterRegion::new(start_y as usize, end_y as usize),
                        scanned_column_kernel_slice,
                    );
                });
            }
        });
    } else {
        let column_handler = T::get_column_handler();
        let final_cell_0 = UnsafeSlice::new(destination);
        column_handler(
            column_arena,
            column_arena_src_slice,
            &final_cell_0,
            image_size,
            FilterRegion::new(0usize, image_size.height),
            scanned_column_kernel_slice,
        );
    }

    column_arena_src.clear();
    Ok(())
}
