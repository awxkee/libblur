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
use crate::{
    gaussian_blur, gaussian_blur_f32, gaussian_blur_u16, BlurImage, BlurImageMut, ConvolutionMode,
    EdgeMode2D, FastBlurChannels, GaussianBlurParams, IeeeBinaryConvolutionMode, ThreadingPolicy,
};
use image::{
    DynamicImage, GrayAlphaImage, GrayImage, ImageBuffer, Luma, LumaA, Rgb, Rgb32FImage, RgbImage,
    Rgba, Rgba32FImage, RgbaImage,
};

/// Performs clear gaussian blur on the image
///
/// NOTE: Alpha must be associated if this image with alpha
///
/// # Arguments
///
/// * `image`: Dynamic image provided by image crate.
/// * `params`: See [GaussianBlurParams] for more info.
/// * `edge_mode` - Rule to handle edge mode.
/// * `precise_level` - Gaussian precise level, precise level works only on u8/u16.
/// * `threading_policy` - Threads usage policy.
///
#[must_use]
pub fn gaussian_blur_image(
    image: DynamicImage,
    params: GaussianBlurParams,
    edge_modes: EdgeMode2D,
    precise_level: ConvolutionMode,
    threading_policy: ThreadingPolicy,
) -> Option<DynamicImage> {
    match image {
        DynamicImage::ImageLuma8(gray) => {
            let gray_image =
                BlurImage::borrow(&gray, gray.width(), gray.height(), FastBlurChannels::Plane);
            let mut new_image =
                BlurImageMut::alloc(gray.width(), gray.height(), FastBlurChannels::Plane);

            gaussian_blur(
                &gray_image,
                &mut new_image,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();
            let new_gray_image = GrayImage::from_raw(
                gray.width(),
                gray.height(),
                new_image.data.borrow().to_vec(),
            )?;
            Some(DynamicImage::ImageLuma8(new_gray_image))
        }
        DynamicImage::ImageLumaA8(luma_alpha_image) => {
            let mut intensity_plane = BlurImageMut::alloc(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                FastBlurChannels::Plane,
            );
            let mut alpha_plane = BlurImageMut::alloc(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                FastBlurChannels::Plane,
            );
            let raw_buffer = luma_alpha_image.as_raw();

            for ((intensity, alpha), raw_buffer) in intensity_plane
                .data
                .borrow_mut()
                .iter_mut()
                .zip(alpha_plane.data.borrow_mut().iter_mut())
                .zip(raw_buffer.chunks_exact(2))
            {
                *intensity = raw_buffer[0];
                *alpha = raw_buffer[1];
            }

            let int = intensity_plane.to_immutable_ref();
            let alp = alpha_plane.to_immutable_ref();

            let mut new_intensity_plane = BlurImageMut::alloc(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                FastBlurChannels::Plane,
            );
            let mut new_alpha_plane = BlurImageMut::alloc(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                FastBlurChannels::Plane,
            );

            gaussian_blur(
                &int,
                &mut new_intensity_plane,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();

            gaussian_blur(
                &alp,
                &mut new_alpha_plane,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();

            let mut new_raw_buffer =
                vec![
                    0u8;
                    luma_alpha_image.width() as usize * luma_alpha_image.height() as usize * 2
                ];

            for ((intensity, alpha), raw_buffer) in new_intensity_plane
                .data
                .borrow()
                .iter()
                .zip(new_alpha_plane.data.borrow().iter())
                .zip(new_raw_buffer.chunks_exact_mut(2))
            {
                raw_buffer[0] = *intensity;
                raw_buffer[1] = *alpha;
            }

            let new_gray_image = GrayAlphaImage::from_raw(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                new_raw_buffer,
            )?;
            Some(DynamicImage::ImageLumaA8(new_gray_image))
        }
        DynamicImage::ImageRgb8(img) => {
            let gray_image =
                BlurImage::borrow(&img, img.width(), img.height(), FastBlurChannels::Channels3);
            let mut new_image =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels3);

            gaussian_blur(
                &gray_image,
                &mut new_image,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();

            let new_rgb_image =
                RgbImage::from_raw(img.width(), img.height(), new_image.data.borrow().to_vec())?;
            Some(DynamicImage::ImageRgb8(new_rgb_image))
        }
        DynamicImage::ImageRgba8(img) => {
            let gray_image =
                BlurImage::borrow(&img, img.width(), img.height(), FastBlurChannels::Channels4);
            let mut new_image =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
            gaussian_blur(
                &gray_image,
                &mut new_image,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();
            let new_rgba_image =
                RgbaImage::from_raw(img.width(), img.height(), new_image.data.borrow().to_vec())?;
            Some(DynamicImage::ImageRgba8(new_rgba_image))
        }
        DynamicImage::ImageLuma16(img) => {
            let gray_image =
                BlurImage::borrow(&img, img.width(), img.height(), FastBlurChannels::Plane);
            let mut new_image =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Plane);

            gaussian_blur_u16(
                &gray_image,
                &mut new_image,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();

            let new_rgb_image = ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(
                img.width(),
                img.height(),
                new_image.data.borrow().to_vec(),
            )?;
            Some(DynamicImage::ImageLuma16(new_rgb_image))
        }
        DynamicImage::ImageLumaA16(luma_alpha_image) => {
            let mut intensity_plane = BlurImageMut::alloc(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                FastBlurChannels::Plane,
            );
            let mut alpha_plane = BlurImageMut::alloc(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                FastBlurChannels::Plane,
            );
            let raw_buffer = luma_alpha_image.as_raw();

            for ((intensity, alpha), raw_buffer) in intensity_plane
                .data
                .borrow_mut()
                .iter_mut()
                .zip(alpha_plane.data.borrow_mut().iter_mut())
                .zip(raw_buffer.chunks_exact(2))
            {
                *intensity = raw_buffer[0];
                *alpha = raw_buffer[1];
            }

            let int = intensity_plane.to_immutable_ref();
            let alp = alpha_plane.to_immutable_ref();

            let mut new_intensity_plane = BlurImageMut::alloc(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                FastBlurChannels::Plane,
            );
            let mut new_alpha_plane = BlurImageMut::alloc(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                FastBlurChannels::Plane,
            );

            gaussian_blur_u16(
                &int,
                &mut new_intensity_plane,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();

            gaussian_blur_u16(
                &alp,
                &mut new_alpha_plane,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();

            let mut new_raw_buffer =
                vec![
                    0u16;
                    luma_alpha_image.width() as usize * luma_alpha_image.height() as usize * 2
                ];

            for ((intensity, alpha), raw_buffer) in new_intensity_plane
                .data
                .borrow()
                .iter()
                .zip(new_alpha_plane.data.borrow().iter())
                .zip(new_raw_buffer.chunks_exact_mut(2))
            {
                raw_buffer[0] = *intensity;
                raw_buffer[1] = *alpha;
            }

            let new_gray_image = ImageBuffer::<LumaA<u16>, Vec<u16>>::from_raw(
                luma_alpha_image.width(),
                luma_alpha_image.height(),
                new_raw_buffer,
            )?;
            Some(DynamicImage::ImageLumaA16(new_gray_image))
        }
        DynamicImage::ImageRgb16(img) => {
            let gray_image =
                BlurImage::borrow(&img, img.width(), img.height(), FastBlurChannels::Channels3);
            let mut new_image =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels3);

            gaussian_blur_u16(
                &gray_image,
                &mut new_image,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();

            let new_rgb_image = ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(
                img.width(),
                img.height(),
                new_image.data.borrow().to_vec(),
            )?;
            Some(DynamicImage::ImageRgb16(new_rgb_image))
        }
        DynamicImage::ImageRgba16(img) => {
            let gray_image =
                BlurImage::borrow(&img, img.width(), img.height(), FastBlurChannels::Channels4);
            let mut new_image =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
            gaussian_blur_u16(
                &gray_image,
                &mut new_image,
                params,
                edge_modes,
                threading_policy,
                precise_level,
            )
            .unwrap();

            let new_rgb_image = ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(
                img.width(),
                img.height(),
                new_image.data.borrow().to_vec(),
            )?;
            Some(DynamicImage::ImageRgba16(new_rgb_image))
        }
        DynamicImage::ImageRgb32F(img) => {
            let gray_image =
                BlurImage::borrow(&img, img.width(), img.height(), FastBlurChannels::Channels3);
            let mut new_image =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels3);
            gaussian_blur_f32(
                &gray_image,
                &mut new_image,
                params,
                edge_modes,
                threading_policy,
                IeeeBinaryConvolutionMode::Normal,
            )
            .unwrap();
            let new_rgb_image =
                Rgb32FImage::from_raw(img.width(), img.height(), new_image.data.borrow().to_vec())?;
            Some(DynamicImage::ImageRgb32F(new_rgb_image))
        }
        DynamicImage::ImageRgba32F(img) => {
            let gray_image =
                BlurImage::borrow(&img, img.width(), img.height(), FastBlurChannels::Channels4);
            let mut new_image =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
            gaussian_blur_f32(
                &gray_image,
                &mut new_image,
                params,
                edge_modes,
                threading_policy,
                IeeeBinaryConvolutionMode::Normal,
            )
            .unwrap();
            let new_rgb_image = Rgba32FImage::from_raw(
                img.width(),
                img.height(),
                new_image.data.borrow().to_vec(),
            )?;
            Some(DynamicImage::ImageRgba32F(new_rgb_image))
        }
        _ => None,
    }
}
