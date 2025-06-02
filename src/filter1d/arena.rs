/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::edge_mode::{reflect_index, reflect_index_101};
use crate::filter1d::arena_roi::copy_roi;
use crate::filter1d::filter_element::KernelShape;
use crate::img_size::ImageSize;
use crate::util::check_slice_size;
use crate::{BlurError, BlurImage, EdgeMode, Scalar};
use num_traits::{AsPrimitive, FromPrimitive};
use std::fmt::Debug;

#[derive(Copy, Clone)]
pub struct Arena {
    pub width: usize,
    #[allow(dead_code)]
    pub height: usize,
    pub pad_w: usize,
    pub pad_h: usize,
    pub components: usize,
}

impl Arena {
    pub fn new(
        arena_width: usize,
        arena_height: usize,
        arena_pad_w: usize,
        arena_pad_h: usize,
        components: usize,
    ) -> Arena {
        Arena {
            width: arena_width,
            height: arena_height,
            pad_w: arena_pad_w,
            pad_h: arena_pad_h,
            components,
        }
    }
}

#[derive(Copy, Clone)]
pub struct ArenaPads {
    pub pad_left: usize,
    pub pad_top: usize,
    pub pad_right: usize,
    pub pad_bottom: usize,
}

impl ArenaPads {
    pub fn constant(v: usize) -> ArenaPads {
        ArenaPads::new(v, v, v, v)
    }

    pub fn new(pad_left: usize, pad_top: usize, pad_right: usize, pad_bottom: usize) -> ArenaPads {
        ArenaPads {
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        }
    }

    pub fn from_kernel_shape(kernel_shape: KernelShape) -> ArenaPads {
        let pad_w = kernel_shape.width / 2;
        let pad_h = kernel_shape.height / 2;
        ArenaPads::new(pad_w, pad_h, pad_w, pad_h)
    }
}

/// Pads an image with chosen border strategy
pub fn make_arena<T, const CN: usize>(
    image: &[T],
    image_stride: usize,
    image_size: ImageSize,
    pads: ArenaPads,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<(Vec<T>, Arena), BlurError>
where
    T: Default + Copy + Send + Sync + 'static,
    f64: AsPrimitive<T>,
{
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                make_arena_avx2::<T, CN>(image, image_stride, image_size, pads, border_mode, scalar)
            };
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return unsafe {
                make_arena_sse4_1::<T, CN>(
                    image,
                    image_stride,
                    image_size,
                    pads,
                    border_mode,
                    scalar,
                )
            };
        }
    }
    make_arena_exec::<T, CN>(image, image_stride, image_size, pads, border_mode, scalar)
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
#[target_feature(enable = "avx2")]
unsafe fn make_arena_avx2<T, const CN: usize>(
    image: &[T],
    image_stride: usize,
    image_size: ImageSize,
    pads: ArenaPads,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<(Vec<T>, Arena), BlurError>
where
    T: Default + Copy + Send + Sync + 'static,
    f64: AsPrimitive<T>,
{
    make_arena_exec::<T, CN>(image, image_stride, image_size, pads, border_mode, scalar)
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
#[target_feature(enable = "sse4.1")]
unsafe fn make_arena_sse4_1<T, const CN: usize>(
    image: &[T],
    image_stride: usize,
    image_size: ImageSize,
    pads: ArenaPads,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<(Vec<T>, Arena), BlurError>
where
    T: Default + Copy + Send + Sync + 'static,
    f64: AsPrimitive<T>,
{
    make_arena_exec::<T, CN>(image, image_stride, image_size, pads, border_mode, scalar)
}

/// Pads an image with chosen border strategy
#[inline(always)]
fn make_arena_exec<T, const CN: usize>(
    image: &[T],
    image_stride: usize,
    image_size: ImageSize,
    pads: ArenaPads,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<(Vec<T>, Arena), BlurError>
where
    T: Default + Copy + Send + Sync + 'static,
    f64: AsPrimitive<T>,
{
    check_slice_size(image, image_stride, image_size.width, image_size.height, CN)?;

    let new_height = image_size.height + pads.pad_top + pads.pad_bottom;
    let new_width = image_size.width + pads.pad_left + pads.pad_right;

    let height = image_size.height;
    let width = image_size.width;

    let mut padded_image = vec![T::default(); new_height * new_width * CN];

    let old_stride = image_stride;
    let new_stride = new_width * CN;

    let offset = pads.pad_top * new_stride + pads.pad_left * CN;
    copy_roi(&mut padded_image[offset..], image, new_stride, old_stride);

    let filling_ranges = [
        (0..pads.pad_top, 0..new_width), // Top outer
        (
            pads.pad_top..(new_height - pads.pad_bottom),
            0..pads.pad_left,
        ), // Left outer
        ((height + pads.pad_top)..new_height, 0..new_width), // Bottom outer
        (
            pads.pad_top..(new_height - pads.pad_bottom),
            (width + pads.pad_left)..new_width,
        ), // Right outer,
    ];

    let pad_w = pads.pad_left;
    let pad_h = pads.pad_top;

    match border_mode {
        EdgeMode::Clamp => {
            for ranges in filling_ranges.iter() {
                for (i, dst) in ranges.0.clone().zip(
                    padded_image
                        .chunks_exact_mut(new_stride)
                        .skip(ranges.0.start),
                ) {
                    for (j, dst) in ranges
                        .1
                        .clone()
                        .zip(dst.chunks_exact_mut(CN).skip(ranges.1.start))
                    {
                        let y = i.saturating_sub(pad_h).min(height - 1);
                        let x = j.saturating_sub(pad_w).min(width - 1);

                        let v_src = y * old_stride + x * CN;
                        let src_iter = &image[v_src..(v_src + CN)];
                        for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                            *dst = *src;
                        }
                    }
                }
            }
        }
        EdgeMode::Wrap => {
            for ranges in filling_ranges.iter() {
                for (i, dst) in ranges.0.clone().zip(
                    padded_image
                        .chunks_exact_mut(new_stride)
                        .skip(ranges.0.start),
                ) {
                    for (j, dst) in ranges
                        .1
                        .clone()
                        .zip(dst.chunks_exact_mut(CN).skip(ranges.1.start))
                    {
                        let y = (i as i64 - pad_h as i64).rem_euclid(height as i64) as usize;
                        let x = (j as i64 - pad_w as i64).rem_euclid(width as i64) as usize;
                        let v_src = y * old_stride + x * CN;
                        let src_iter = &image[v_src..(v_src + CN)];
                        for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                            *dst = *src;
                        }
                    }
                }
            }
        }
        EdgeMode::Reflect => {
            for ranges in filling_ranges.iter() {
                for (i, dst) in ranges.0.clone().zip(
                    padded_image
                        .chunks_exact_mut(new_stride)
                        .skip(ranges.0.start),
                ) {
                    for (j, dst) in ranges
                        .1
                        .clone()
                        .zip(dst.chunks_exact_mut(CN).skip(ranges.1.start))
                    {
                        let y = reflect_index(i as isize - pad_h as isize, height as isize);
                        let x = reflect_index(j as isize - pad_w as isize, width as isize);
                        let v_src = y * old_stride + x * CN;
                        let src_iter = &image[v_src..(v_src + CN)];
                        for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                            *dst = *src;
                        }
                    }
                }
            }
        }
        EdgeMode::Reflect101 => {
            for ranges in filling_ranges.iter() {
                for (i, dst) in ranges.0.clone().zip(
                    padded_image
                        .chunks_exact_mut(new_stride)
                        .skip(ranges.0.start),
                ) {
                    for (j, dst) in ranges
                        .1
                        .clone()
                        .zip(dst.chunks_exact_mut(CN).skip(ranges.1.start))
                    {
                        let y = reflect_index_101(i as isize - pad_h as isize, height as isize);
                        let x = reflect_index_101(j as isize - pad_w as isize, width as isize);
                        let v_src = y * old_stride + x * CN;
                        let src_iter = &image[v_src..(v_src + CN)];
                        for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                            *dst = *src;
                        }
                    }
                }
            }
        }
        EdgeMode::Constant => {
            for ranges in filling_ranges.iter() {
                for (_, dst) in ranges.0.clone().zip(
                    padded_image
                        .chunks_exact_mut(new_stride)
                        .skip(ranges.0.start),
                ) {
                    for (_, dst) in ranges
                        .1
                        .clone()
                        .zip(dst.chunks_exact_mut(CN).skip(ranges.1.start))
                    {
                        for (y, dst) in dst.iter_mut().enumerate() {
                            *dst = scalar[y].as_();
                        }
                    }
                }
            }
        }
    }
    Ok((
        padded_image,
        Arena::new(new_width, new_height, pad_w, pad_h, CN),
    ))
}

/// Pads an image with chosen border strategy
pub fn make_arena_row<T, const CN: usize>(
    image: &BlurImage<T>,
    source_y: usize,
    kernel_size: KernelShape,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<(Vec<T>, usize), BlurError>
where
    T: Default + Copy + Send + Sync + 'static + Debug,
    f64: AsPrimitive<T>,
{
    image.check_layout()?;
    let pad_w = kernel_size.width / 2;

    let image_size = image.size();

    let arena_width = image_size.width * CN + pad_w * 2 * CN;
    let mut row = vec![T::default(); arena_width];
    write_arena_row::<T, CN>(&mut row, image, source_y, kernel_size, border_mode, scalar)?;
    Ok((row, image_size.width + pad_w * 2))
}

pub(crate) fn write_arena_row<T, const CN: usize>(
    row: &mut [T],
    image: &BlurImage<T>,
    source_y: usize,
    kernel_size: KernelShape,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<(), BlurError>
where
    T: Default + Copy + Send + Sync + 'static + Debug,
    f64: AsPrimitive<T>,
{
    image.check_layout()?;
    let pad_w = kernel_size.width / 2;

    let image_size = image.size();

    let arena_width = image_size.width * CN + pad_w * 2 * CN;
    if row.len() < arena_width {
        return Err(BlurError::ImagesMustMatch);
    }

    let source_offset = source_y * image.row_stride() as usize;

    let source_row = &image.data.as_ref()[source_offset..(source_offset + image_size.width * CN)];

    let row_dst = &mut row[pad_w * CN..(pad_w * CN + image_size.width * CN)];

    for (dst, src) in row_dst.iter_mut().zip(source_row.iter()) {
        *dst = *src;
    }

    for (x, dst) in (0..pad_w).zip(row.chunks_exact_mut(CN)) {
        match border_mode {
            EdgeMode::Clamp => {
                let old_x = x.saturating_sub(pad_w).min(image_size.width - 1);
                let old_px = old_x * CN;
                let src_iter = &source_row[old_px..(old_px + CN)];
                for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                    *dst = *src;
                }
            }
            EdgeMode::Wrap => {
                let old_x = (x as i64 - pad_w as i64).rem_euclid(image_size.width as i64) as usize;
                let old_px = old_x * CN;
                let src_iter = &source_row[old_px..(old_px + CN)];
                for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                    *dst = *src;
                }
            }
            EdgeMode::Reflect => {
                let old_x = reflect_index(x as isize - pad_w as isize, image_size.width as isize);
                let old_px = old_x * CN;
                let src_iter = &source_row[old_px..(old_px + CN)];
                for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                    *dst = *src;
                }
            }
            EdgeMode::Reflect101 => {
                let old_x =
                    reflect_index_101(x as isize - pad_w as isize, image_size.width as isize);
                let old_px = old_x * CN;
                let src_iter = &source_row[old_px..(old_px + CN)];
                for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                    *dst = *src;
                }
            }
            EdgeMode::Constant => {
                for (i, dst) in dst.iter_mut().enumerate() {
                    *dst = scalar[i].as_();
                }
            }
        }
    }

    for (x, dst) in
        (image_size.width..(image_size.width + pad_w)).zip(row.chunks_exact_mut(CN).rev())
    {
        match border_mode {
            EdgeMode::Clamp => {
                let old_x = x.max(0).min(image_size.width - 1);
                let old_px = old_x * CN;
                let src_iter = &source_row[old_px..(old_px + CN)];
                for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                    *dst = *src;
                }
            }
            EdgeMode::Wrap => {
                let old_x = (x as i64).rem_euclid(image_size.width as i64) as usize;
                let old_px = old_x * CN;
                let src_iter = &source_row[old_px..(old_px + CN)];
                for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                    *dst = *src;
                }
            }
            EdgeMode::Reflect => {
                let old_x = reflect_index(x as isize, image_size.width as isize);
                let old_px = old_x * CN;
                let src_iter = &source_row[old_px..(old_px + CN)];
                for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                    *dst = *src;
                }
            }
            EdgeMode::Reflect101 => {
                let old_x = reflect_index_101(x as isize, image_size.width as isize);
                let old_px = old_x * CN;
                let src_iter = &source_row[old_px..(old_px + CN)];
                for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                    *dst = *src;
                }
            }
            EdgeMode::Constant => {
                for (i, dst) in dst.iter_mut().enumerate() {
                    *dst = scalar[i].as_();
                }
            }
        }
    }

    Ok(())
}

#[derive(Clone)]
pub struct ArenaColumns<T>
where
    T: Copy,
{
    pub top_pad: Vec<T>,
    pub bottom_pad: Vec<T>,
}

impl<T> ArenaColumns<T>
where
    T: Copy,
{
    pub fn new(top_pad: Vec<T>, bottom_pad: Vec<T>) -> ArenaColumns<T> {
        ArenaColumns {
            top_pad,
            bottom_pad,
        }
    }
}

/// Pads a column image with chosen border strategy
pub fn make_arena_columns<T, const CN: usize>(
    image: &[T],
    image_size: ImageSize,
    kernel_size: KernelShape,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<ArenaColumns<T>, BlurError>
where
    T: Default + Copy + Send + Sync + 'static + FromPrimitive,
{
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                mac_avx2::<T, CN>(image, image_size, kernel_size, border_mode, scalar)
            };
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return unsafe {
                mac_sse_4_1::<T, CN>(image, image_size, kernel_size, border_mode, scalar)
            };
        }
    }
    make_arena_columns_exec::<T, CN>(image, image_size, kernel_size, border_mode, scalar)
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
#[target_feature(enable = "avx2")]
unsafe fn mac_avx2<T, const CN: usize>(
    image: &[T],
    image_size: ImageSize,
    kernel_size: KernelShape,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<ArenaColumns<T>, BlurError>
where
    T: Default + Copy + Send + Sync + 'static + FromPrimitive,
{
    make_arena_columns_exec::<T, CN>(image, image_size, kernel_size, border_mode, scalar)
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
#[target_feature(enable = "sse4.1")]
unsafe fn mac_sse_4_1<T, const CN: usize>(
    image: &[T],
    image_size: ImageSize,
    kernel_size: KernelShape,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<ArenaColumns<T>, BlurError>
where
    T: Default + Copy + Send + Sync + 'static + FromPrimitive,
{
    make_arena_columns_exec::<T, CN>(image, image_size, kernel_size, border_mode, scalar)
}

/// Pads a column image with chosen border strategy
#[inline(always)]
fn make_arena_columns_exec<T, const CN: usize>(
    image: &[T],
    image_size: ImageSize,
    kernel_size: KernelShape,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<ArenaColumns<T>, BlurError>
where
    T: Default + Copy + Send + Sync + 'static + FromPrimitive,
{
    check_slice_size(
        image,
        image_size.width * CN,
        image_size.width,
        image_size.height,
        CN,
    )?;
    let pad_h = kernel_size.height / 2;

    let mut top_pad = vec![T::default(); pad_h * image_size.width * CN];
    let mut bottom_pad = vec![T::default(); pad_h * image_size.width * CN];

    let top_pad_stride = image_size.width * CN;

    for (ky, dst) in (0..pad_h).zip(top_pad.chunks_exact_mut(top_pad_stride)) {
        for (kx, dst) in (0..image_size.width).zip(dst.chunks_exact_mut(CN)) {
            match border_mode {
                EdgeMode::Clamp => {
                    let y = ky.saturating_sub(pad_h).min(image_size.height - 1);
                    let v_src = y * top_pad_stride + kx * CN;

                    let src_iter = &image[v_src..(v_src + CN)];
                    for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                        *dst = *src;
                    }
                }
                EdgeMode::Wrap => {
                    let y =
                        (ky as i64 - pad_h as i64).rem_euclid(image_size.height as i64) as usize;
                    let v_src = y * top_pad_stride + kx * CN;
                    let src_iter = &image[v_src..(v_src + CN)];
                    for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                        *dst = *src;
                    }
                }
                EdgeMode::Reflect => {
                    let y = reflect_index(ky as isize - pad_h as isize, image_size.height as isize);
                    let v_src = y * top_pad_stride + kx * CN;
                    let src_iter = &image[v_src..(v_src + CN)];
                    for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                        *dst = *src;
                    }
                }
                EdgeMode::Reflect101 => {
                    let y =
                        reflect_index_101(ky as isize - pad_h as isize, image_size.height as isize);
                    let v_src = y * top_pad_stride + kx * CN;
                    let src_iter = &image[v_src..(v_src + CN)];
                    for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                        *dst = *src;
                    }
                }
                EdgeMode::Constant => {
                    for (i, dst) in dst.iter_mut().enumerate() {
                        *dst = T::from_f64(scalar[i]).unwrap_or_default();
                    }
                }
            }
        }
    }

    let bottom_iter_dst = bottom_pad.chunks_exact_mut(top_pad_stride);

    for (ky, dst) in (0..pad_h).zip(bottom_iter_dst) {
        for (kx, dst) in (0..image_size.width).zip(dst.chunks_exact_mut(CN)) {
            match border_mode {
                EdgeMode::Clamp => {
                    let y = (ky + image_size.height).min(image_size.height - 1);
                    let v_src = y * top_pad_stride + kx * CN;
                    let src_iter = &image[v_src..(v_src + CN)];
                    for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                        *dst = *src;
                    }
                }
                EdgeMode::Wrap => {
                    let y = (ky as i64 + image_size.height as i64)
                        .rem_euclid(image_size.height as i64) as usize;
                    let v_src = y * top_pad_stride + kx * CN;
                    let src_iter = &image[v_src..(v_src + CN)];
                    for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                        *dst = *src;
                    }
                }
                EdgeMode::Reflect => {
                    let y = reflect_index(
                        ky as isize + image_size.height as isize,
                        image_size.height as isize,
                    );
                    let v_src = y * top_pad_stride + kx * CN;
                    let src_iter = &image[v_src..(v_src + CN)];
                    for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                        *dst = *src;
                    }
                }
                EdgeMode::Reflect101 => {
                    let y = reflect_index_101(
                        ky as isize + image_size.height as isize,
                        image_size.height as isize,
                    );
                    let v_src = y * top_pad_stride + kx * CN;
                    let src_iter = &image[v_src..(v_src + CN)];
                    for (dst, src) in dst.iter_mut().zip(src_iter.iter()) {
                        *dst = *src;
                    }
                }
                EdgeMode::Constant => {
                    for (i, dst) in dst.iter_mut().enumerate() {
                        *dst = T::from_f64(scalar[i]).unwrap_or_default();
                    }
                }
            }
        }
    }

    Ok(ArenaColumns::new(top_pad, bottom_pad))
}
