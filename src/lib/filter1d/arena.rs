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
use crate::{EdgeMode, Scalar};
use num_traits::AsPrimitive;

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
pub fn make_arena<T, const COMPONENTS: usize>(
    image: &[T],
    image_size: ImageSize,
    pads: ArenaPads,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<(Vec<T>, Arena), String>
where
    T: Default + Copy + Send + Sync + 'static,
    f64: AsPrimitive<T>,
{
    if image.len() != COMPONENTS * image_size.width * image_size.height {
        return Err(format!(
            "Can't create arena, expected image with size {} but got {}",
            COMPONENTS * image_size.width * image_size.height,
            image.len()
        ));
    }

    let new_height = image_size.height + pads.pad_top + pads.pad_bottom;
    let new_width = image_size.width + pads.pad_left + pads.pad_right;

    let height = image_size.height;
    let width = image_size.width;

    let mut padded_image = vec![T::default(); new_height * new_width * COMPONENTS];

    let old_stride = image_size.width * COMPONENTS;
    let new_stride = new_width * COMPONENTS;

    unsafe {
        let offset = pads.pad_top * new_stride + pads.pad_left * COMPONENTS;
        copy_roi(
            padded_image.get_unchecked_mut(offset..),
            image,
            new_stride,
            old_stride,
            height,
        );
    }

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
                for i in ranges.0.clone() {
                    for j in ranges.1.clone() {
                        let y = i.saturating_sub(pad_h).min(height - 1);
                        let x = j.saturating_sub(pad_w).min(width - 1);
                        unsafe {
                            let v_dst = i * new_stride + j * COMPONENTS;
                            let v_src = y * old_stride + x * COMPONENTS;
                            for i in 0..COMPONENTS {
                                *padded_image.get_unchecked_mut(v_dst + i) =
                                    *image.get_unchecked(v_src + i);
                            }
                        }
                    }
                }
            }
        }
        EdgeMode::Wrap => {
            for ranges in filling_ranges.iter() {
                for i in ranges.0.clone() {
                    for j in ranges.1.clone() {
                        let y = (i as i64 - pad_h as i64).rem_euclid(height as i64 - 1) as usize;
                        let x = (j as i64 - pad_w as i64).rem_euclid(width as i64 - 1) as usize;
                        unsafe {
                            let v_dst = i * new_stride + j * COMPONENTS;
                            let v_src = y * old_stride + x * COMPONENTS;
                            for i in 0..COMPONENTS {
                                *padded_image.get_unchecked_mut(v_dst + i) =
                                    *image.get_unchecked(v_src + i);
                            }
                        }
                    }
                }
            }
        }
        EdgeMode::Reflect => {
            for ranges in filling_ranges.iter() {
                for i in ranges.0.clone() {
                    for j in ranges.1.clone() {
                        let y = reflect_index(i as i64 - pad_h as i64, height as i64 - 1);
                        let x = reflect_index(j as i64 - pad_w as i64, width as i64 - 1);
                        unsafe {
                            let v_dst = i * new_stride + j * COMPONENTS;
                            let v_src = y * old_stride + x * COMPONENTS;
                            for i in 0..COMPONENTS {
                                *padded_image.get_unchecked_mut(v_dst + i) =
                                    *image.get_unchecked(v_src + i);
                            }
                        }
                    }
                }
            }
        }
        EdgeMode::Reflect101 => {
            for ranges in filling_ranges.iter() {
                for i in ranges.0.clone() {
                    for j in ranges.1.clone() {
                        let y = reflect_index_101(i as i64 - pad_h as i64, height as i64 - 1);
                        let x = reflect_index_101(j as i64 - pad_w as i64, width as i64 - 1);
                        unsafe {
                            let v_dst = i * new_stride + j * COMPONENTS;
                            let v_src = y * old_stride + x * COMPONENTS;
                            for i in 0..COMPONENTS {
                                *padded_image.get_unchecked_mut(v_dst + i) =
                                    *image.get_unchecked(v_src + i);
                            }
                        }
                    }
                }
            }
        }
        EdgeMode::Constant => {
            for ranges in filling_ranges.iter() {
                for i in ranges.0.clone() {
                    for j in ranges.1.clone() {
                        unsafe {
                            let v_dst = i * new_stride + j * COMPONENTS;
                            for i in 0..COMPONENTS {
                                *padded_image.get_unchecked_mut(v_dst + i) = scalar[i].as_();
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((
        padded_image,
        Arena::new(new_width, new_height, pad_w, pad_h, COMPONENTS),
    ))
}

/// Pads an image with chosen border strategy
pub fn make_arena_row<T, const COMPONENTS: usize>(
    image: &[T],
    source_y: usize,
    image_size: ImageSize,
    kernel_size: KernelShape,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<(Vec<T>, usize), String>
where
    T: Default + Copy + Send + Sync + 'static,
    f64: AsPrimitive<T>,
{
    if image.len() != COMPONENTS * image_size.width * image_size.height {
        return Err(format!(
            "Can't create arena, expected image with size {} but got {}",
            COMPONENTS * image_size.width * image_size.height,
            image.len()
        ));
    }

    let pad_w = kernel_size.width / 2;

    let arena_width = image_size.width * COMPONENTS + pad_w * 2 * COMPONENTS;
    let mut row = vec![T::default(); arena_width];

    let source_offset = source_y * image_size.width * COMPONENTS;

    let source_row = unsafe { image.get_unchecked(source_offset..) };

    unsafe {
        std::ptr::copy_nonoverlapping(
            source_row.as_ptr(),
            row.get_unchecked_mut((pad_w * COMPONENTS)..).as_mut_ptr(),
            image_size.width * COMPONENTS,
        );
    }

    for x in 0..pad_w {
        match border_mode {
            EdgeMode::Clamp => {
                let old_x = x.saturating_sub(pad_w).min(image_size.width - 1);
                let old_px = old_x * COMPONENTS;
                let px = x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = *source_row.get_unchecked(old_px + i);
                    }
                }
            }
            EdgeMode::Wrap => {
                let old_x =
                    (x as i64 - pad_w as i64).rem_euclid(image_size.width as i64 - 1) as usize;
                let old_px = old_x * COMPONENTS;
                let px = x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = *source_row.get_unchecked(old_px + i);
                    }
                }
            }
            EdgeMode::Reflect => {
                let old_x = reflect_index(x as i64 - pad_w as i64, image_size.width as i64 - 1);
                let old_px = old_x * COMPONENTS;
                let px = x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = *source_row.get_unchecked(old_px + i);
                    }
                }
            }
            EdgeMode::Reflect101 => {
                let old_x = reflect_index_101(x as i64 - pad_w as i64, image_size.width as i64 - 1);
                let old_px = old_x * COMPONENTS;
                let px = x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = *source_row.get_unchecked(old_px + i);
                    }
                }
            }
            EdgeMode::Constant => {
                let px = x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = scalar[i].as_();
                    }
                }
            }
        }
    }

    for x in image_size.width..(image_size.width + pad_w) {
        match border_mode {
            EdgeMode::Clamp => {
                let old_x = x.max(0).min(image_size.width - 1);
                let old_px = old_x * COMPONENTS;
                let px = pad_w * COMPONENTS + x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = *source_row.get_unchecked(old_px + i);
                    }
                }
            }
            EdgeMode::Wrap => {
                let old_x = (x as i64).rem_euclid(image_size.width as i64 - 1) as usize;
                let old_px = old_x * COMPONENTS;
                let px = pad_w * COMPONENTS + x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = *source_row.get_unchecked(old_px + i);
                    }
                }
            }
            EdgeMode::Reflect => {
                let old_x = reflect_index(x as i64, image_size.width as i64 - 1);
                let old_px = old_x * COMPONENTS;
                let px = pad_w * COMPONENTS + x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = *source_row.get_unchecked(old_px + i);
                    }
                }
            }
            EdgeMode::Reflect101 => {
                let old_x = reflect_index_101(x as i64, image_size.width as i64 - 1);
                let old_px = old_x * COMPONENTS;
                let px = pad_w * COMPONENTS + x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = *source_row.get_unchecked(old_px + i);
                    }
                }
            }
            EdgeMode::Constant => {
                let px = pad_w * COMPONENTS + x * COMPONENTS;
                for i in 0..COMPONENTS {
                    unsafe {
                        *row.get_unchecked_mut(px + i) = scalar[i].as_();
                    }
                }
            }
        }
    }

    Ok((row, image_size.width + pad_w * 2))
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
pub fn make_arena_columns<T, const COMPONENTS: usize>(
    image: &[T],
    image_size: ImageSize,
    kernel_size: KernelShape,
    border_mode: EdgeMode,
    scalar: Scalar,
) -> Result<ArenaColumns<T>, String>
where
    T: Default + Copy + Send + Sync + 'static,
    f64: AsPrimitive<T>,
{
    if image.len() != COMPONENTS * image_size.width * image_size.height {
        return Err(format!(
            "Can't create arena, expected image with size {} but got {}",
            COMPONENTS * image_size.width * image_size.height,
            image.len()
        ));
    }
    let pad_h = kernel_size.height / 2;

    let mut top_pad = vec![T::default(); pad_h * image_size.width * COMPONENTS];
    let mut bottom_pad = vec![T::default(); pad_h * image_size.width * COMPONENTS];

    let top_pad_stride = image_size.width * COMPONENTS;

    for ky in 0..pad_h {
        for kx in 0..image_size.width {
            match border_mode {
                EdgeMode::Clamp => {
                    let y = ky.saturating_sub(pad_h).min(image_size.height - 1);
                    unsafe {
                        let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                        let v_src = y * top_pad_stride + kx * COMPONENTS;
                        for i in 0..COMPONENTS {
                            *top_pad.get_unchecked_mut(v_dst + i) = *image.get_unchecked(v_src + i);
                        }
                    }
                }
                EdgeMode::Wrap => {
                    let y = (ky as i64 - pad_h as i64).rem_euclid(image_size.height as i64 - 1)
                        as usize;
                    unsafe {
                        let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                        let v_src = y * top_pad_stride + kx * COMPONENTS;
                        for i in 0..COMPONENTS {
                            *top_pad.get_unchecked_mut(v_dst + i) = *image.get_unchecked(v_src + i);
                        }
                    }
                }
                EdgeMode::Reflect => {
                    let y = reflect_index(ky as i64 - pad_h as i64, image_size.height as i64 - 1);
                    unsafe {
                        let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                        let v_src = y * top_pad_stride + kx * COMPONENTS;
                        for i in 0..COMPONENTS {
                            *top_pad.get_unchecked_mut(v_dst + i) = *image.get_unchecked(v_src + i);
                        }
                    }
                }
                EdgeMode::Reflect101 => {
                    let y =
                        reflect_index_101(ky as i64 - pad_h as i64, image_size.height as i64 - 1);
                    unsafe {
                        let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                        let v_src = y * top_pad_stride + kx * COMPONENTS;
                        for i in 0..COMPONENTS {
                            *top_pad.get_unchecked_mut(v_dst + i) = *image.get_unchecked(v_src + i);
                        }
                    }
                }
                EdgeMode::Constant => unsafe {
                    let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                    for i in 0..COMPONENTS {
                        *top_pad.get_unchecked_mut(v_dst + i) = scalar[i].as_();
                    }
                },
            }
        }
    }

    for ky in 0..pad_h {
        for kx in 0..image_size.width {
            match border_mode {
                EdgeMode::Clamp => {
                    let y = (ky + image_size.height).min(image_size.height - 1);
                    unsafe {
                        let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                        let v_src = y * top_pad_stride + kx * COMPONENTS;
                        for i in 0..COMPONENTS {
                            *bottom_pad.get_unchecked_mut(v_dst + i) =
                                *image.get_unchecked(v_src + i);
                        }
                    }
                }
                EdgeMode::Wrap => {
                    let y = (ky as i64 + image_size.height as i64)
                        .rem_euclid(image_size.height as i64 - 1)
                        as usize;
                    unsafe {
                        let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                        let v_src = y * top_pad_stride + kx * COMPONENTS;
                        for i in 0..COMPONENTS {
                            *bottom_pad.get_unchecked_mut(v_dst + i) =
                                *image.get_unchecked(v_src + i);
                        }
                    }
                }
                EdgeMode::Reflect => {
                    let y = reflect_index(
                        ky as i64 + image_size.height as i64,
                        image_size.height as i64 - 1,
                    );
                    unsafe {
                        let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                        let v_src = y * top_pad_stride + kx * COMPONENTS;
                        for i in 0..COMPONENTS {
                            *bottom_pad.get_unchecked_mut(v_dst + i) =
                                *image.get_unchecked(v_src + i);
                        }
                    }
                }
                EdgeMode::Reflect101 => {
                    let y = reflect_index_101(
                        ky as i64 + image_size.height as i64,
                        image_size.height as i64 - 1,
                    );
                    unsafe {
                        let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                        let v_src = y * top_pad_stride + kx * COMPONENTS;
                        for i in 0..COMPONENTS {
                            *top_pad.get_unchecked_mut(v_dst + i) = *image.get_unchecked(v_src + i);
                        }
                    }
                }
                EdgeMode::Constant => unsafe {
                    let v_dst = ky * top_pad_stride + kx * COMPONENTS;
                    for i in 0..COMPONENTS {
                        *bottom_pad.get_unchecked_mut(v_dst + i) = scalar[i].as_();
                    }
                },
            }
        }
    }

    Ok(ArenaColumns::new(top_pad, bottom_pad))
}
