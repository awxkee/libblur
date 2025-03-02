mod merge;
mod split;

use crate::merge::merge_channels_3;
use crate::split::split_channels_3;
use image::{EncodableLayout, GenericImageView, ImageReader};
use libblur::{
    filter_1d_exact, gaussian_kernel_1d, generate_motion_kernel, motion_blur, sigma_size,
    BlurImage, BlurImageMut, ConvolutionMode, EdgeMode, FastBlurChannels, ImageSize, Scalar,
    ThreadingPolicy,
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
    let dyn_image = ImageReader::open("assets/test_image_2.png")
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
    let src_bytes = img.as_bytes();
    let components = 3;
    let stride = dimensions.0 as usize * components;
    let mut bytes: Vec<u8> = src_bytes.to_vec();
    let mut dst_bytes: Vec<u8> = src_bytes.to_vec();

    let start_time = Instant::now();

    println!("libblur::stack_blur: {:?}", start_time.elapsed());

    let start = Instant::now();

    let rc = bytes
        .iter()
        .map(|&x| x as f32 * (1. / 255.))
        .collect::<Vec<_>>();
    let f32_image = BlurImage::borrow(
        &rc,
        dyn_image.width(),
        dyn_image.height(),
        FastBlurChannels::Channels3,
    );
    let mut target_f32 = BlurImageMut::alloc(
        dyn_image.width(),
        dyn_image.height(),
        FastBlurChannels::Channels3,
    );

    // libblur::fast_bilateral_filter(&f32_image, &mut target_f32, 25, 7f32, 7f32).unwrap();

    libblur::gaussian_blur_f32(
        &f32_image,
        &mut target_f32,
        0,
        25f32,
        EdgeMode::Clamp,
        ThreadingPolicy::Single,
    )
    .unwrap();

    // dst_bytes = target_f32.data.borrow().to_vec();

    dst_bytes = target_f32
        .data
        .borrow()
        .iter()
        .map(|&x| (x * 255f32).round() as u8)
        .collect();

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
