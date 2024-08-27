mod merge;
mod split;

use crate::merge::merge_channels_3;
use crate::split::split_channels_3;
use image::io::Reader as ImageReader;
use image::{EncodableLayout, GenericImageView};
use libblur::{EdgeMode, FastBlurChannels, GaussianPreciseLevel, ThreadingPolicy};
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

fn perform_planar_pass_3(img: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut plane_1 = vec![0u8; width * height];
    let mut plane_2 = vec![0u8; width * height];
    let mut plane_3 = vec![0u8; width * height];
    let mut merged_planes: Vec<u8> = vec![0u8; width * height * 3];

    split_channels_3(img, width, height, &mut plane_1, &mut plane_2, &mut plane_3);

    let mut dst_plane_1 = vec![0u8; width * height];
    let mut dst_plane_2 = vec![0u8; width * height];
    let mut dst_plane_3 = vec![0u8; width * height];

    // libblur::fast_gaussian_next(
    //     &mut dst_plane_1,
    //     width as u32,
    //     width as u32,
    //     height as u32,
    //     35,
    //     FastBlurChannels::Plane,
    //     ThreadingPolicy::Single,
    //     EdgeMode::Clamp,
    // );
    //
    // libblur::fast_gaussian_next(
    //     &mut dst_plane_2,
    //     width as u32,
    //     width as u32,
    //     height as u32,
    //     35,
    //     FastBlurChannels::Plane,
    //     ThreadingPolicy::Single,
    //     EdgeMode::Clamp,
    // );
    //
    // libblur::fast_gaussian_next(
    //     &mut dst_plane_3,
    //     width as u32,
    //     width as u32,
    //     height as u32,
    //     35,
    //     FastBlurChannels::Plane,
    //     ThreadingPolicy::Single,
    //     EdgeMode::Clamp,
    // );

    libblur::gaussian_blur(
        &plane_1,
        width as u32,
        &mut dst_plane_1,
        width as u32,
        width as u32,
        height as u32,
        25 * 2 + 1,
        0.,
        FastBlurChannels::Plane,
        EdgeMode::Clamp,
        ThreadingPolicy::Single,
        GaussianPreciseLevel::INTEGRAL,
    );

    libblur::gaussian_blur(
        &plane_2,
        width as u32,
        &mut dst_plane_2,
        width as u32,
        width as u32,
        height as u32,
        25 * 2 + 1,
        0.,
        FastBlurChannels::Plane,
        EdgeMode::Clamp,
        ThreadingPolicy::Single,
        GaussianPreciseLevel::INTEGRAL,
    );

    libblur::gaussian_blur(
        &plane_3,
        width as u32,
        &mut dst_plane_3,
        width as u32,
        width as u32,
        height as u32,
        25 * 2 + 1,
        0.,
        FastBlurChannels::Plane,
        EdgeMode::Clamp,
        ThreadingPolicy::Single,
        GaussianPreciseLevel::INTEGRAL,
    );

    merge_channels_3(
        &mut merged_planes,
        width,
        height,
        &dst_plane_1,
        &dst_plane_2,
        &dst_plane_3,
    );
    merged_planes
}

fn main() {
    // unsafe {
    //     let v = vdupq_n_s64(2);
    //     let v2 = vdupq_n_s64(3);
    //     let mul = vmulq_s64(v, v2);
    //     let mut t: [i64; 2] = [0i64; 2];
    //     vst1q_s64(t.as_mut_ptr(), mul);
    //     println!("{:?}", t);
    // }
    let img = ImageReader::open("assets/test_image_1_small.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    println!("dimensions {:?}", img.dimensions());
    println!("type {:?}", img.color());

    println!("{:?}", img.color());
    let src_bytes = img.as_bytes();
    let components = 3;
    let stride = dimensions.0 as usize * components;
    let mut bytes: Vec<u8> = Vec::from(src_bytes);
    let mut dst_bytes: Vec<u8> = Vec::with_capacity(dimensions.1 as usize * stride);
    dst_bytes.resize(dimensions.1 as usize * stride, 0);
    unsafe {
        std::ptr::copy_nonoverlapping(
            src_bytes.as_ptr(),
            dst_bytes.as_mut_ptr(),
            dimensions.1 as usize * stride,
        );
    }

    // libblur::stack_blur(
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     255,
    //     FastBlurChannels::Channels4,
    //     ThreadingPolicy::Single,
    // );

    //
    // libblur::tent_blur(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     77,
    //     FastBlurChannels::Channels4,
    //     ThreadingPolicy::Single,
    // );
    // bytes = dst_bytes;

    let start_time = Instant::now();
    // libblur::fast_gaussian_superior(
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     75,
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Single,
    // );
    // libblur::fast_gaussian_next(
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     25,
    //     FastBlurChannels::Channels4,
    //     ThreadingPolicy::Single,
    //     EdgeMode::Clamp,
    // );

    // libblur::gaussian_blur_in_linear(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     67 * 2 + 1,
    //     67. * 2f32 / 6f32,
    //     FastBlurChannels::Channels4,
    //     EdgeMode::KernelClip,
    //     ThreadingPolicy::Single,
    //     TransferFunction::Srgb,
    // );

    // let mut f16_bytes: Vec<f16> = dst_bytes
    //     .iter()
    //     .map(|&x| f16::from_f32(x as f32 * (1. / 255.)))
    //     .collect();

    libblur::gaussian_blur(
        &bytes,
        stride as u32,
        &mut dst_bytes,
        stride as u32,
        dimensions.0,
        dimensions.1,
        37,
        0.,
        FastBlurChannels::Channels3,
        EdgeMode::KernelClip,
        ThreadingPolicy::Single,
        GaussianPreciseLevel::INTEGRAL,
    );

    // stack_blur_f16(
    //     &mut f16_bytes,
    //     dimensions.0,
    //     dimensions.1,
    //     50,
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Single,
    // );

    // fast_gaussian_in_linear(
    //     &mut dst_bytes,
    //     dimensions.0 * components as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     125,
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Single,
    //     TransferFunction::Rec709,
    //     EdgeMode::Clamp,
    // );
    //
    // dst_bytes = f16_bytes
    //     .iter()
    //     .map(|&x| (x.to_f32() * 255f32) as u8)
    //     .collect();

    // dst_bytes = perform_planar_pass_3(&bytes, dimensions.0 as usize, dimensions.1 as usize);

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Elapsed time: {:.2?}", elapsed_time);

    // libblur::gaussian_blur(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     75 * 2 + 1,
    //     75f32*2f32 / 6f32,
    //     FastBlurChannels::Channels4,
    //     EdgeMode::Wrap,
    //     ThreadingPolicy::Adaptive,
    // );
    bytes = dst_bytes;
    // libblur::median_blur(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     35,
    //     FastBlurChannels::Channels4,
    //     ThreadingPolicy::Adaptive,
    // );
    // bytes = dst_bytes;
    // libblur::gaussian_box_blur(&bytes, stride as u32, &mut dst_bytes, stride as u32, dimensions.0, dimensions.1, 77,
    //                            FastBlurChannels::Channels3, ThreadingPolicy::Single);
    // bytes = dst_bytes;

    if components == 3 {
        image::save_buffer(
            "blurred_stack_oklab.jpg",
            bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "blurred_stack_cpu.png",
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
