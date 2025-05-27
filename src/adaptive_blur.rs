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
#![allow(clippy::manual_clamp)]

use crate::{
    filter_2d, filter_2d_fft, gaussian_blur, stack_blur, AnisotropicRadius, BlurError, BlurImage,
    BlurImageMut, ConvolutionMode, EdgeMode, FastBlurChannels, GaussianBlurParams, KernelShape,
    Scalar, ThreadingPolicy, TransferFunction,
};
use num_traits::AsPrimitive;
use std::fmt::Debug;

trait Blend<T> {
    fn blend(
        source: &[T],
        destination: &mut [T],
        mask: &[T],
        width: usize,
        height: usize,
        fast_blur_channels: FastBlurChannels,
    );
}

impl Blend<u8> for u8 {
    fn blend(
        source: &[u8],
        destination: &mut [u8],
        mask: &[u8],
        _: usize,
        _: usize,
        fast_blur_channels: FastBlurChannels,
    ) {
        let channels = fast_blur_channels.channels();
        match fast_blur_channels {
            FastBlurChannels::Plane => {
                for ((dst, src), mask) in destination.iter_mut().zip(source.iter()).zip(mask.iter())
                {
                    let m = *mask;
                    *dst =
                        ((m as u32 * (*src) as u32 + (255 - m as u32) * (*dst) as u32) / 255) as u8;
                }
            }
            FastBlurChannels::Channels3 | FastBlurChannels::Channels4 => {
                for ((dst, src), mask) in destination
                    .chunks_exact_mut(channels)
                    .zip(source.chunks_exact(channels))
                    .zip(mask.iter())
                {
                    let m = *mask;
                    dst[0] =
                        ((m as u32 * src[0] as u32 + (255 - m as u32) * dst[0] as u32) / 255) as u8;
                    dst[1] =
                        ((m as u32 * src[1] as u32 + (255 - m as u32) * dst[1] as u32) / 255) as u8;
                    dst[2] =
                        ((m as u32 * src[2] as u32 + (255 - m as u32) * dst[2] as u32) / 255) as u8;
                }
            }
        }
    }
}

trait BlurEdges<T> {
    fn blur_edges(image: &mut [T], width: usize, height: usize, radius: u32);
}

impl BlurEdges<u8> for u8 {
    fn blur_edges(image: &mut [u8], width: usize, height: usize, radius: u32) {
        let mut blur_img =
            BlurImageMut::borrow(image, width as u32, height as u32, FastBlurChannels::Plane);
        stack_blur(
            &mut blur_img,
            AnisotropicRadius::new(radius),
            ThreadingPolicy::Adaptive,
        )
        .unwrap();
    }
}

trait Blur<T> {
    fn blur(
        image: &[T],
        dst_image: &mut [T],
        width: usize,
        height: usize,
        radius: u32,
        channels: FastBlurChannels,
    );
}

impl Blur<u8> for u8 {
    fn blur(
        image: &[u8],
        dst_image: &mut [u8],
        width: usize,
        height: usize,
        radius: u32,
        channels: FastBlurChannels,
    ) {
        let mut rad = (radius / 2).saturating_sub(1).max(1);
        if rad % 2 == 0 {
            rad += 1;
        }
        let src = BlurImage::borrow(image, width as u32, height as u32, channels);
        let mut dst = BlurImageMut::borrow(dst_image, width as u32, height as u32, channels);
        gaussian_blur(
            &src,
            &mut dst,
            GaussianBlurParams::new_from_kernel(rad as f64),
            EdgeMode::Clamp,
            ThreadingPolicy::Adaptive,
            ConvolutionMode::FixedPoint,
        )
        .unwrap();
    }
}

trait Grayscale<T> {
    fn grayscale(source: &[T], width: usize, height: usize, channels: FastBlurChannels) -> Vec<T>;
}

trait Level<T> {
    fn auto_level(in_plane: &mut [T], width: usize, height: usize);
}

impl Level<u8> for u8 {
    fn auto_level(in_plane: &mut [u8], _: usize, _: usize) {
        let mut min_value = u8::MAX;
        let mut max_value = u8::MIN;

        for &element in in_plane.iter() {
            min_value = min_value.min(element);
            max_value = max_value.max(element);
        }

        if min_value > max_value {
            std::mem::swap(&mut min_value, &mut max_value);
        }

        let new_min = 0;
        let old_range = max_value - min_value;
        let new_range = u8::MAX;

        if old_range != 0 {
            for element in in_plane.iter_mut() {
                *element = (new_min
                    + (((*element as i32 - min_value as i32) * new_range as i32)
                        / old_range as i32))
                    .max(0)
                    .min(255) as u8;
            }
        }
    }
}

trait Gamma<T> {
    fn gamma(
        gamma: &mut [T],
        width: usize,
        height: usize,
        channels: FastBlurChannels,
        transfer_function: TransferFunction,
    );
}

impl Gamma<u8> for u8 {
    fn gamma(
        gamma: &mut [u8],
        _: usize,
        _: usize,
        channels: FastBlurChannels,
        transfer_function: TransferFunction,
    ) {
        let mut lut_table = vec![0u8; 256];
        for (i, item) in lut_table.iter_mut().enumerate().take(256) {
            *item = (transfer_function.gamma(i as f32 * (1. / 255.0)) * 255.).min(255.) as u8;
        }

        match channels {
            FastBlurChannels::Plane => gamma.iter_mut().for_each(|dst| {
                *dst = lut_table[*dst as usize];
            }),
            FastBlurChannels::Channels3 | FastBlurChannels::Channels4 => {
                gamma.chunks_exact_mut(channels.channels()).for_each(|dst| {
                    dst[0] = lut_table[dst[0] as usize];
                    dst[1] = lut_table[dst[1] as usize];
                    dst[2] = lut_table[dst[2] as usize];
                })
            }
        }
    }
}

trait Linearize<T> {
    fn linearize(
        source: &[T],
        width: usize,
        height: usize,
        channels: FastBlurChannels,
        transfer_function: TransferFunction,
    ) -> Vec<T>;
}

trait Edges<T> {
    fn edges(
        source: &[T],
        width: usize,
        height: usize,
        radius: usize,
        border_mode: EdgeMode,
        border_constant: Scalar,
    ) -> Vec<T>;
}

impl Edges<u8> for u8 {
    fn edges(
        source: &[u8],
        width: usize,
        height: usize,
        radius: usize,
        border_mode: EdgeMode,
        border_constant: Scalar,
    ) -> Vec<u8> {
        let mut dst = vec![0u8; width * height];

        let mut radius = radius;
        if radius % 2 == 0 {
            radius += 1;
        }

        let full_size = radius * radius;
        let mut edge_filter = vec![-1f32; full_size];
        edge_filter[radius * (radius / 2) + radius / 2] = radius as f32 * radius as f32 - 1f32;
        if radius > 15 {
            let src_image =
                BlurImage::borrow(source, width as u32, height as u32, FastBlurChannels::Plane);
            let mut dst_image = BlurImageMut::borrow(
                &mut dst,
                width as u32,
                height as u32,
                FastBlurChannels::Plane,
            );
            filter_2d_fft::<u8, f32, f32>(
                &src_image,
                &mut dst_image,
                &edge_filter,
                KernelShape::new(radius, radius),
                border_mode,
                border_constant,
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        } else {
            let src_image =
                BlurImage::borrow(source, width as u32, height as u32, FastBlurChannels::Plane);
            let mut dst_image = BlurImageMut::borrow(
                &mut dst,
                width as u32,
                height as u32,
                FastBlurChannels::Plane,
            );
            filter_2d::<u8, f32>(
                &src_image,
                &mut dst_image,
                &edge_filter,
                KernelShape::new(radius, radius),
                border_mode,
                border_constant,
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        }
        dst
    }
}

impl Linearize<u8> for u8 {
    fn linearize(
        source: &[u8],
        width: usize,
        height: usize,
        channels: FastBlurChannels,
        transfer_function: TransferFunction,
    ) -> Vec<u8> {
        let mut lut_table = vec![0u8; 256];
        for (i, item) in lut_table.iter_mut().enumerate().take(256) {
            *item = (transfer_function.linearize(i as f32 * (1. / 255.0)) * 255.).min(255.) as u8;
        }

        let mut dst = vec![0u8; width * height * channels.channels()];

        match channels {
            FastBlurChannels::Plane => dst.iter_mut().zip(source).for_each(|(dst, src)| {
                *dst = lut_table[*src as usize];
            }),
            FastBlurChannels::Channels3 => dst
                .chunks_exact_mut(3)
                .zip(source.chunks_exact(3))
                .for_each(|(dst, src)| {
                    dst[0] = lut_table[src[0] as usize];
                    dst[1] = lut_table[src[1] as usize];
                    dst[2] = lut_table[src[2] as usize];
                }),
            FastBlurChannels::Channels4 => dst
                .chunks_exact_mut(4)
                .zip(source.chunks_exact(4))
                .for_each(|(dst, src)| {
                    dst[0] = lut_table[src[0] as usize];
                    dst[1] = lut_table[src[1] as usize];
                    dst[2] = lut_table[src[2] as usize];
                    dst[3] = src[3];
                }),
        }
        dst
    }
}

impl Grayscale<u8> for u8 {
    fn grayscale(
        source: &[u8],
        width: usize,
        height: usize,
        channels: FastBlurChannels,
    ) -> Vec<u8> {
        let kr = 0.2126f32;
        let kb = 0.0722f32;
        let kg = 1. - kr - kb;

        const SCALE: f32 = (1i32 << 10i32) as f32;
        let v_kr = (kr * SCALE).round() as i32;
        let v_kb = (kb * SCALE).round() as i32;
        let v_kg = (kg * SCALE).round() as i32;

        match channels {
            FastBlurChannels::Plane => source.to_vec(),
            FastBlurChannels::Channels3 | FastBlurChannels::Channels4 => {
                let mut dest = vec![0u8; width * height];

                for (src, dst) in source
                    .chunks_exact(channels.channels())
                    .zip(dest.iter_mut())
                {
                    *dst =
                        (((src[0] as i32 * v_kr) + (src[1] as i32 * v_kg) + (src[2] as i32 * v_kb))
                            >> 10)
                            .max(0)
                            .min(255) as u8;
                }

                dest
            }
        }
    }
}

fn adaptive_blur_impl<
    T: Copy
        + Grayscale<T>
        + Linearize<T>
        + AsPrimitive<i16>
        + Default
        + Send
        + Sync
        + Edges<T>
        + Level<T>
        + Blur<T>
        + BlurEdges<T>
        + Blend<T>
        + Gamma<T>
        + Debug,
>(
    src_image: &BlurImage<T>,
    dst_image: &mut BlurImageMut<T>,
    radius: u32,
    transfer_function: TransferFunction,
    border_mode: EdgeMode,
    border_constant: Scalar,
) {
    let image = src_image.data.as_ref();
    let width = src_image.width as usize;
    let height = src_image.height as usize;
    let channels = src_image.channels;
    let linear_source = T::linearize(image, width, height, channels, transfer_function);
    let grayscale = T::grayscale(&linear_source, width, height, channels);
    let mut edges = T::edges(
        &grayscale,
        width,
        height,
        radius as usize,
        border_mode,
        border_constant,
    );
    T::auto_level(&mut edges, width, height);
    T::blur_edges(&mut edges, width, height, radius);
    T::auto_level(&mut edges, width, height);

    T::blur(
        &linear_source,
        dst_image.data.borrow_mut(),
        width,
        height,
        radius,
        channels,
    );

    T::blend(
        &linear_source,
        dst_image.data.borrow_mut(),
        &edges,
        width,
        height,
        channels,
    );

    T::gamma(
        dst_image.data.borrow_mut(),
        width,
        height,
        channels,
        transfer_function,
    );
}

/// Performs an adaptive blur on the image
pub fn adaptive_blur(
    src_image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    radius: u32,
    transfer_function: TransferFunction,
    border_mode: EdgeMode,
    border_constant: Scalar,
) -> Result<(), BlurError> {
    src_image.check_layout()?;
    dst_image.check_layout(Some(src_image))?;
    src_image.size_matches_mut(dst_image)?;
    if src_image.row_stride() != src_image.width * src_image.channels.channels() as u32 {
        return Err(BlurError::StrideIsNotSupported);
    }
    if dst_image.row_stride() != dst_image.width * dst_image.channels.channels() as u32 {
        return Err(BlurError::StrideIsNotSupported);
    }
    adaptive_blur_impl(
        src_image,
        dst_image,
        radius,
        transfer_function,
        border_mode,
        border_constant,
    );
    Ok(())
}
