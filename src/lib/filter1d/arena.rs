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
use crate::EdgeMode;

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

/// Pads an image with chosen border strategy
pub fn make_arena<T, const COMPONENTS: usize>(
    image: &[T],
    image_size: ImageSize,
    kernel_size: KernelShape,
    border_mode: EdgeMode,
) -> Result<(Vec<T>, Arena), String>
where
    T: Default + Copy + Send + Sync,
{
    if image.len() != COMPONENTS * image_size.width * image_size.height {
        return Err(format!(
            "Can't create arena, expected image with size {} but got {}",
            COMPONENTS * image_size.width * image_size.height,
            image.len()
        ));
    }

    let (kw, kh) = (kernel_size.width, kernel_size.height);

    let pad_w = kw / 2;
    let pad_h = kh / 2;

    let new_height = image_size.height + 2 * pad_h;
    let new_width = image_size.width + 2 * pad_w;

    let height = image_size.height;
    let width = image_size.width;

    let mut padded_image = vec![T::default(); new_height * new_width * COMPONENTS];

    let old_stride = image_size.width * COMPONENTS;
    let new_stride = new_width * COMPONENTS;

    unsafe {
        let offset = pad_h * new_stride + pad_w * COMPONENTS;
        copy_roi(
            padded_image.get_unchecked_mut(offset..),
            image,
            new_stride,
            old_stride,
            height,
        );
    }

    let filling_ranges = [
        (0..pad_h, 0..new_width),                                  // Top outer
        (pad_h..(new_height - pad_h), 0..pad_w),                   // Left outer
        ((height + pad_h)..new_height, 0..new_width),              // Bottom outer
        (pad_h..(new_height - pad_h), (width + pad_w)..new_width), // Bottom outer,
    ];

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
        EdgeMode::KernelClip => {
            return Err("KernelClip is not supported in Filter 1D".parse().unwrap())
        }
    }

    Ok((
        padded_image,
        Arena::new(new_width, new_height, pad_w, pad_h, COMPONENTS),
    ))
}
