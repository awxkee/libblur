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

use image::{EncodableLayout, GenericImageView, ImageReader};
use libblur::{
    fast_bilateral_filter, fast_bilateral_filter_u16, filter_1d_complex,
    filter_1d_complex_fixed_point, filter_2d_rgba_fft, gaussian_kernel_1d, lens_kernel, sigma_size,
    BlurImage, BlurImageMut, EdgeMode, FastBlurChannels, KernelShape, Scalar, ThreadingPolicy,
    TransferFunction,
};
use num_complex::Complex;
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

fn complex_gaussian_kernel(radius: f64, scale: f64, distortion: f64) -> Vec<Complex<f32>> {
    let kernel_radius = radius.ceil() as usize;
    let mut kernel: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 1 + 2 * (kernel_radius)];

    for (x, dst) in kernel.iter_mut().enumerate() {
        let ax = (x as f64 - radius) * scale / radius;
        let ax2 = ax * ax;
        let exp_a = (-distortion * ax2).exp();
        let val = Complex::new(
            exp_a * (distortion * ax2).cos(),
            exp_a * (distortion * ax2).sin(),
        );
        *dst = val;
    }

    let sum: f64 = kernel.iter().map(|z| z.norm_sqr()).sum::<f64>();
    if sum != 0.0 {
        kernel
            .iter()
            .map(|z| Complex {
                re: (z.re / sum) as f32,
                im: (z.im / sum) as f32,
            })
            .collect::<Vec<_>>()
    } else {
        kernel
            .iter()
            .map(|x| Complex {
                re: x.re as f32,
                im: x.im as f32,
            })
            .collect()
    }
}

fn main() {
    let mut dyn_image = ImageReader::open("./assets/test_image_2.png")
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

    let cvt = BlurImage::borrow(
        &v_vec,
        dyn_image.width(),
        dyn_image.height(),
        FastBlurChannels::Channels3,
    );
    let vcvt = cvt.linearize(TransferFunction::Srgb, true).unwrap();

    let mut dst_image = BlurImageMut::default();
    //
    // libblur::fast_gaussian_next_u16(&mut dst_image, 37, ThreadingPolicy::Single, EdgeMode::Clamp)
    //     .unwrap();
    // let kernel = gaussian_kernel_1d_f64(5, sigma_size_d(2.5));
    let start_time = Instant::now();

    // for i in 0..10 {
    let motion = lens_kernel(KernelShape::new(151, 151), 10., 3., 0.3, 0.5).unwrap();
    // let motion = lens_kernel(KernelShape::new(35, 35), 15., 6., 0.5,0.2).unwrap();
    // let bokeh = generate_complex_bokeh_kernel(35, 30.);
    let start_time = Instant::now();
    // let gaussian_kernel = gaussian_kernel_1d(31, sigma_size(31.)).iter().map(|&x| Complex::new(x, 0.0)).collect::<Vec<Complex<f32>>>();
    let gaussian_kernel = complex_gaussian_kernel(51., 0.75, 25.);

    for i in 0..15 {
        let start_time = Instant::now();
        fast_bilateral_filter_u16(&vcvt, &mut dst_image, 15, 1., 5., ThreadingPolicy::Adaptive)
            .unwrap();
        println!(
            "libblur::fast_bilateral_filter_u16 MultiThreading: {:?}",
            start_time.elapsed()
        );
    }

    // filter_2d_rgba_fft::<u16, f32, f32>(
    //     &image,
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

    // libblur::gaussian_blur(
    //     &image,
    //     &mut dst_image,
    //     GaussianBlurParams {
    //         x_kernel: 7,
    //         x_sigma: 0.,
    //         y_kernel: 9,
    //         y_sigma: 0.,
    //     },
    //     EdgeMode::Clamp,
    //     ThreadingPolicy::Single,
    //     ConvolutionMode::FixedPoint,
    // )
    // .unwrap();

    // libblur::fast_gaussian_next_f32(
    //     &mut dst_image,
    //     AnisotropicRadius::create(8, 35),
    //     ThreadingPolicy::Single,
    //     EdgeMode::Clamp,
    // )
    // .unwrap();

    // libblur::motion_blur(
    //     &image,
    //     &mut dst_image,
    //     35.,
    //     15,
    //     EdgeMode::Clamp,
    //     Scalar::new(0.0, 0.0, 0.0, 0.0),
    //     ThreadingPolicy::Single,
    // )
    // .unwrap();

    let j_dag = dst_image.to_immutable_ref();
    let gamma = j_dag.gamma8(TransferFunction::Srgb, true).unwrap();

    dst_bytes = gamma
        .data
        .as_ref()
        .iter()
        .map(|&x| x)
        // .map(|&x| (x * 255f32).round() as u8)
        // .map(|&x| (x >> 8) as u8)
        .collect::<Vec<u8>>();

    // dst_bytes = dst_image.data.borrow().to_vec();
    //
    // filter_2d_rgba_fft::<u8, f32, i32>(
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
            "blurred_stack_next_f.png",
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
