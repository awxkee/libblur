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
use crate::fast_bilateral_filter::fast_bilateral_filter_gray_alpha_impl;
use crate::{
    fast_bilateral_filter, fast_bilateral_filter_f32, fast_bilateral_filter_u16, FastBlurChannels,
};
use image::{
    DynamicImage, GrayAlphaImage, GrayImage, ImageBuffer, Luma, LumaA, Rgb, Rgb32FImage, RgbImage,
    Rgba, Rgba32FImage, RgbaImage,
};

/// Performs fast bilateral filter on the image
///
/// This is fast bilateral approximation, note this behaviour significantly differs from OpenCV.
/// This method has high convergence and will completely blur an image very fast with increasing spatial sigma
///
/// # Arguments
///
/// * `img`: Source image
/// * `dst`: Destination image
/// * `spatial_sigma`: Spatial sigma
/// * `range_sigma`: Range sigma
///
#[must_use]
pub fn fast_bilateral_filter_image(
    image: DynamicImage,
    spatial_sigma: f32,
    range_sigma: f32,
) -> Option<DynamicImage> {
    match image {
        DynamicImage::ImageLuma8(gray) => {
            let mut new_image = gray.as_raw().to_vec();
            fast_bilateral_filter(
                gray.as_raw(),
                &mut new_image,
                gray.width(),
                gray.height(),
                spatial_sigma,
                range_sigma,
                FastBlurChannels::Plane,
            );
            let new_gray_image = GrayImage::from_raw(gray.width(), gray.height(), new_image)?;
            Some(DynamicImage::ImageLuma8(new_gray_image))
        }
        DynamicImage::ImageLumaA8(luma_alpha_image) => {
            let mut new_image =
                vec![
                    0u8;
                    luma_alpha_image.width() as usize * luma_alpha_image.height() as usize * 2
                ];

            fast_bilateral_filter_gray_alpha_impl(
                luma_alpha_image.as_raw(),
                &mut new_image,
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                spatial_sigma,
                range_sigma,
            );

            let new_gray_image = GrayAlphaImage::from_raw(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                new_image,
            )?;
            Some(DynamicImage::ImageLumaA8(new_gray_image))
        }
        DynamicImage::ImageRgb8(rgb_image) => {
            let mut new_image = rgb_image.as_raw().to_vec();

            fast_bilateral_filter(
                rgb_image.as_raw(),
                &mut new_image,
                rgb_image.width(),
                rgb_image.height(),
                spatial_sigma,
                range_sigma,
                FastBlurChannels::Channels3,
            );

            let new_rgb_image =
                RgbImage::from_raw(rgb_image.width(), rgb_image.height(), new_image)?;
            Some(DynamicImage::ImageRgb8(new_rgb_image))
        }
        DynamicImage::ImageRgba8(rgba_image) => {
            let mut new_image = rgba_image.as_raw().to_vec();
            fast_bilateral_filter(
                rgba_image.as_raw(),
                &mut new_image,
                rgba_image.width(),
                rgba_image.height(),
                spatial_sigma,
                range_sigma,
                FastBlurChannels::Channels4,
            );
            let new_rgba_image =
                RgbaImage::from_raw(rgba_image.width(), rgba_image.height(), new_image)?;
            Some(DynamicImage::ImageRgba8(new_rgba_image))
        }
        DynamicImage::ImageLuma16(luma_16) => {
            let mut new_image = luma_16.as_raw().to_vec();

            fast_bilateral_filter_u16(
                luma_16.as_raw(),
                &mut new_image,
                luma_16.width(),
                luma_16.height(),
                spatial_sigma,
                range_sigma,
                FastBlurChannels::Plane,
            );

            let new_rgb_image = ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(
                luma_16.width(),
                luma_16.height(),
                new_image,
            )?;
            Some(DynamicImage::ImageLuma16(new_rgb_image))
        }
        DynamicImage::ImageLumaA16(gray_alpha_16) => {
            let mut new_raw_buffer =
                vec![0u16; gray_alpha_16.width() as usize * gray_alpha_16.height() as usize * 2];

            fast_bilateral_filter_gray_alpha_impl(
                gray_alpha_16.as_raw(),
                &mut new_raw_buffer,
                gray_alpha_16.width(),
                gray_alpha_16.height(),
                spatial_sigma,
                range_sigma,
            );

            let new_gray_image = ImageBuffer::<LumaA<u16>, Vec<u16>>::from_raw(
                gray_alpha_16.width(),
                gray_alpha_16.height(),
                new_raw_buffer,
            )?;
            Some(DynamicImage::ImageLumaA16(new_gray_image))
        }
        DynamicImage::ImageRgb16(rgb_16_image) => {
            let mut new_image = rgb_16_image.as_raw().to_vec();

            fast_bilateral_filter_u16(
                rgb_16_image.as_raw(),
                &mut new_image,
                rgb_16_image.width(),
                rgb_16_image.height(),
                spatial_sigma,
                range_sigma,
                FastBlurChannels::Channels3,
            );

            let new_rgb_image = ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(
                rgb_16_image.width(),
                rgb_16_image.height(),
                new_image,
            )?;
            Some(DynamicImage::ImageRgb16(new_rgb_image))
        }
        DynamicImage::ImageRgba16(rgba_16_image) => {
            let mut new_image = rgba_16_image.as_raw().to_vec();

            fast_bilateral_filter_u16(
                rgba_16_image.as_raw(),
                &mut new_image,
                rgba_16_image.width(),
                rgba_16_image.height(),
                spatial_sigma,
                range_sigma,
                FastBlurChannels::Channels4,
            );

            let new_rgb_image = ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(
                rgba_16_image.width(),
                rgba_16_image.height(),
                new_image,
            )?;
            Some(DynamicImage::ImageRgba16(new_rgb_image))
        }
        DynamicImage::ImageRgb32F(rgb_image_f32) => {
            let mut new_image = rgb_image_f32.as_raw().to_vec();
            fast_bilateral_filter_f32(
                rgb_image_f32.as_raw(),
                &mut new_image,
                rgb_image_f32.width(),
                rgb_image_f32.height(),
                spatial_sigma,
                range_sigma,
                FastBlurChannels::Channels3,
            );
            let new_rgb_image =
                Rgb32FImage::from_raw(rgb_image_f32.width(), rgb_image_f32.height(), new_image)?;
            Some(DynamicImage::ImageRgb32F(new_rgb_image))
        }
        DynamicImage::ImageRgba32F(rgba_image_f32) => {
            let mut new_image = rgba_image_f32.as_raw().to_vec();
            fast_bilateral_filter_f32(
                rgba_image_f32.as_raw(),
                &mut new_image,
                rgba_image_f32.width(),
                rgba_image_f32.height(),
                spatial_sigma,
                range_sigma,
                FastBlurChannels::Channels4,
            );
            let new_rgb_image =
                Rgba32FImage::from_raw(rgba_image_f32.width(), rgba_image_f32.height(), new_image)?;
            Some(DynamicImage::ImageRgba32F(new_rgb_image))
        }
        _ => None,
    }
}
