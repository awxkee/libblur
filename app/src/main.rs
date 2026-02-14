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

use image::imageops::FilterType;
use image::{DynamicImage, EncodableLayout, GenericImageView, ImageReader};
use libblur::{bilateral_filter, complex_gaussian_kernel, fast_bilateral_filter, fast_bilateral_filter_u16, filter_1d_complex, filter_1d_complex_fixed_point, filter_2d_rgb_fft, filter_2d_rgb_fft_complex, filter_2d_rgba_fft, gaussian_blur, gaussian_blur_image, gaussian_kernel_1d, lens_kernel, sigma_size, AnisotropicRadius, BilateralBlurParams, BlurImage, BlurImageMut, BoxBlurParameters, CLTParameters, ConvolutionMode, EdgeMode, EdgeMode2D, FastBlurChannels, GaussianBlurParams, ImageSize, KernelShape, Scalar, ThreadingPolicy, TransferFunction};
use num_complex::Complex;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::any::Any;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Read};
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
    let mut dyn_image = ImageReader::open("./assets/test_image_1_small.jpg")
        .unwrap()
        .decode()
        .unwrap();

    let dimensions = dyn_image.dimensions();
    println!("dimensions {:?}", dyn_image.dimensions());
    println!("type {:?}", dyn_image.color());

    // let first_gauss = gaussian_blur_image(
    //     DynamicImage::from(dyn_image.clone()),
    //     GaussianBlurParams::new_asymmetric_from_sigma(15., 15.),
    //     EdgeMode2D::new(EdgeMode::Clamp),
    //     ConvolutionMode::FixedPoint,
    //     ThreadingPolicy::Adaptive,
    // ).unwrap();
    // first_gauss.save("blurred_image.jpg").unwrap();

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

    let start = Instant::now();

    let mut v_vec = src_bytes
        .to_vec()
        .iter()
        .map(|&x| x)
        // .map(|&x| (x as f32 / 255.))
        // .map(|&x| u16::from_ne_bytes([x, x]))
        .collect::<Vec<u8>>();

    // let mut dst_image = BlurImageMut::borrow(
    //     &mut v_vec,
    //     dyn_image.width(),
    //     dyn_image.height(),
    //     FastBlurChannels::Channels4,
    // );

    // let z0 = v_vec.iter().map(|&x| (x as f32 * (1. / 255.))).collect::<Vec<_>>();
    let cvt = BlurImage::borrow(
        &v_vec,
        dyn_image.width(),
        dyn_image.height(),
        FastBlurChannels::Channels3,
    );
    // let vcvt = cvt.linearize(TransferFunction::Srgb, true).unwrap();

    // let kernel = gaussian_kernel_1d_f64(5, sigma_size_d(2.5));
    let start_time = Instant::now();

    // for i in 0..10 {
    let ks = KernelShape::new(75, 75);
    let motion = lens_kernel(ks, 10., 3., 0.3, 0.5).unwrap();
    // let motion = lens_kernel(KernelShape::new(35, 35), 15., 6., 0.5, 0.2).unwrap();
    // let bokeh = generate_complex_bokeh_kernel(35, 30.);
    let start_time = Instant::now();
    // let gaussian_kernel = gaussian_kernel_1d(31, sigma_size(31.)).iter().map(|&x| Complex::new(x, 0.0)).collect::<Vec<Complex<f32>>>();
    let gaussian_kernel = complex_gaussian_kernel(51., 0.75, 5.);

    let mut dst_image = BlurImageMut::default(); //cvt.clone_as_mut();

    gaussian_blur(
        &cvt,
        &mut dst_image,
        GaussianBlurParams::new_asymmetric_from_sigma(25., 25.),
        EdgeMode::Clamp.as_2d(),
        ThreadingPolicy::Single,
        ConvolutionMode::FixedPoint,
    )
        .unwrap();

    // filter_2d_rgba_fft::<u8, f32, f32>(
    //     &cvt,
    //     &mut dst_image,
    //     &motion,
    //     KernelShape::new(151, 151),
    //     EdgeMode::Clamp,
    //     Scalar::default(),
    //     ThreadingPolicy::Adaptive,
    // )
    // .unwrap();
    // for dst in dst_image.data.borrow_mut().chunks_exact_mut(4) {
    //     dst[3] = 255;
    // }

    println!(
        "libblur::filter_2d_rgba_fft MultiThreading: {:?}",
        start_time.elapsed()
    );

    // }

    // libblur::sobel(
    //     &cvt,
    //     &mut dst_image,
    //     EdgeMode2D::default(),
    //     Scalar::default(),
    //     ThreadingPolicy::Single,
    // )
    // .unwrap();

    // libblur::stack_blur(
    //     &mut dst_image,
    //     AnisotropicRadius::create(35, 35),
    //     ThreadingPolicy::Single,
    // )
    // .unwrap();

    // libblur::motion_blur(
    //     &cvt,
    //     &mut dst_image,
    //     35.,
    //     21,
    //     EdgeMode::Clamp,
    //     Scalar::new(0.0, 0.0, 0.0, 0.0),
    //     ThreadingPolicy::Adaptive,
    // )
    // .unwrap();

    // let j_dag = dst_image.to_immutable_ref();

    // let gamma = cvt.linearize(TransferFunction::Srgb, false).unwrap();
    // let mut vzd = BlurImageMut::default();
    //
    // filter_2d_rgb_fft::<u8, f32>(
    //     &cvt,
    //     &mut dst_image,
    //     &motion,
    //     ks,
    //     EdgeMode2D::new(EdgeMode::Clamp),
    //     Scalar::default(),
    //     ThreadingPolicy::Adaptive,
    // )
    // .unwrap();

    // dst_image = vzd.gamma8(TransferFunction::Srgb, false).unwrap();

    dst_bytes = dst_image
        .data
        .borrow_mut()
        .iter()
        .map(|&x| x)
        // .map(|&x| (x * 255f32).round() as u8)
        // .map(|&x| (x >> 8) as u8)
        .collect::<Vec<u8>>();

    // dst_bytes = dst_image.data.borrow().to_vec();
    //

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
            "blurred_stack_next1.jpg",
            bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "blurred_stack_next_f1.png",
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
