/*
 * // Copyright (c) Radzivon Bartoshyk 3/2025. All rights reserved.
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

mod merge;
mod split;

use crate::merge::merge_channels_3;
use crate::split::split_channels_3;
use image::{EncodableLayout, GenericImageView, ImageReader};
use libblur::{
    filter_1d_exact, gaussian_kernel_1d, generate_motion_kernel, motion_blur, sigma_size,
    BlurImage, BlurImageMut, BufferStore, ConvolutionMode, EdgeMode, FastBlurChannels, ImageSize,
    Scalar, ThreadingPolicy,
};
use std::time::Instant;

#[allow(dead_code)]
fn f32_to_f16(bytes: Vec<f32>) -> Vec<u16> {
    bytes
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

#[allow(dead_code)]
fn f16_to_f32(bytes: Vec<u16>) -> Vec<f32> {
    bytes
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect()
}

fn main() {
    let dyn_image = ImageReader::open("./assets/test_image_2.png")
        .unwrap()
        .decode()
        .unwrap();

    let dimensions = dyn_image.dimensions();
    println!("dimensions {:?}", dyn_image.dimensions());
    println!("type {:?}", dyn_image.color());

    // let vldg = dyn_image.to_rgb8();
    // let new_rgb = image::imageops::blur(&vldg, 66.);
    // let dyn_image = DynamicImage::ImageRgb8(dyn_image.to_rgb8());
    // new_dyn.save("output.jpg").unwrap();

    println!("{:?}", dyn_image.color());

    let img = dyn_image.to_rgb8();
    let mut src_bytes = img.as_bytes();
    let components = 3;
    let stride = dimensions.0 as usize * components;
    let mut bytes: Vec<u8> = src_bytes.to_vec();
    let mut dst_bytes: Vec<u8> = src_bytes.to_vec();

    let start_time = Instant::now();

    println!("libblur::stack_blur: {:?}", start_time.elapsed());

    let start = Instant::now();

    // let mut dst_image = BlurImageMut::borrow(&mut src_bytes, dyn_image.width(), dyn_image.height(), FastBlurChannels::Channels3)

    let remapped = src_bytes
        .iter()
        .map(|&x| x as f32 * (1. / 255.))
        .collect::<Vec<f32>>();

    let image = BlurImage::borrow(
        &remapped,
        dyn_image.width(),
        dyn_image.height(),
        FastBlurChannels::Channels3,
    );
    let mut dst_image = BlurImageMut::default();

    libblur::gaussian_blur_f32(
        &image,
        &mut dst_image,
        0,
        55.,
        EdgeMode::Wrap,
        ThreadingPolicy::Single,
    )
    .unwrap();

    dst_bytes = dst_image
        .data
        .borrow()
        .iter()
        .map(|&x| (x * 255f32).round() as u8)
        .collect::<Vec<u8>>();

    // dst_bytes = dst_image.data.borrow().to_vec();
    //
    // filter_2d_rgba_approx::<u8, f32, i32>(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     &kernel,
    //     &kernel,
    //     EdgeMode::Clamp,
    //     ThreadingPolicy::Adaptive,
    // )
    // .unwrap();

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Elapsed time: {:.2?}", elapsed_time);

    // let start_time = Instant::now();
    // let blurred = dyn_image.blur(125f32);
    // println!("Gauss image: {:.2?}", start_time.elapsed());
    // blurred.save("dyn.jpg").unwrap();

    bytes = dst_bytes;

    if components == 3 {
        image::save_buffer(
            "blurred_stack_next.jpg",
            bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "blurred_stack_next.png",
            bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            if components == 3 {
                image::ExtendedColorType::Rgb8
            } else {
                image::ExtendedColorType::Rgba8
            },
        )
        .unwrap();
    }
}
