mod merge;
mod split;

use crate::merge::merge_channels_3;
use crate::split::split_channels_3;
use colorutils_rs::TransferFunction;
use image::{DynamicImage, EncodableLayout, GenericImageView, ImageReader};
use libblur::{
    adaptive_blur, fast_bilateral_filter, filter_1d_approx, filter_1d_exact, filter_1d_rgb_approx,
    filter_1d_rgb_exact, filter_2d_rgb_fft, filter_2d_rgba_fft, generate_motion_kernel,
    get_gaussian_kernel_1d, get_sigma_size, motion_blur, EdgeMode, FastBlurChannels,
    GaussianPreciseLevel, ImageSize, KernelShape, Scalar, ThreadingPolicy,
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

    let start = Instant::now();

    let kernel_size = 75;

    libblur::gaussian_blur(
        &plane_1,
        &mut dst_plane_1,
        width as u32,
        height as u32,
        kernel_size,
        0.,
        FastBlurChannels::Plane,
        EdgeMode::Reflect,
        ThreadingPolicy::Adaptive,
        GaussianPreciseLevel::EXACT,
    );

    println!("libblur::gaussian_blur: {:?}", start.elapsed());

    let kernel = get_gaussian_kernel_1d(kernel_size, get_sigma_size(kernel_size as usize));

    let start = Instant::now();

    filter_1d_exact(
        &plane_2,
        &mut dst_plane_2,
        ImageSize::new(width, height),
        &kernel,
        &kernel,
        EdgeMode::Constant,
        Scalar::new(255.0, 0.0, 0.0, 255.0),
        ThreadingPolicy::Adaptive,
    )
    .unwrap();

    println!("filter_1d: {:?}", start.elapsed());

    // libblur::gaussian_blur(
    //     &plane_2,
    //     width as u32,
    //     &mut dst_plane_2,
    //     width as u32,
    //     width as u32,
    //     height as u32,
    //     25 * 2 + 1,
    //     0.,
    //     FastBlurChannels::Plane,
    //     EdgeMode::Reflect,
    //     ThreadingPolicy::Single,
    //     GaussianPreciseLevel::EXACT,
    // );

    let start = Instant::now();

    libblur::gaussian_blur(
        &plane_3,
        &mut dst_plane_3,
        width as u32,
        height as u32,
        kernel_size,
        0.,
        FastBlurChannels::Plane,
        EdgeMode::Reflect,
        ThreadingPolicy::Adaptive,
        GaussianPreciseLevel::EXACT,
    );

    println!("libblur::gaussian_blur: {:?}", start.elapsed());

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
    let dyn_image = ImageReader::open("assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = dyn_image.dimensions();
    println!("dimensions {:?}", dyn_image.dimensions());
    println!("type {:?}", dyn_image.color());

    // let vldg = dyn_image.to_rgb8();
    // let new_rgb = image::imageops::blur(&vldg, 66.);
    // let new_dyn = DynamicImage::ImageRgb8(new_rgb);
    // new_dyn.save("output.jpg").unwrap();

    println!("{:?}", dyn_image.color());

    let img = dyn_image.to_rgb8();
    let src_bytes = img.as_bytes();
    let components = 3;
    let stride = dimensions.0 as usize * components;
    let mut bytes: Vec<u8> = src_bytes.to_vec();
    let mut dst_bytes: Vec<u8> = src_bytes.to_vec();

    // let mut spawned_bytes = dst_bytes.iter().map(|&x| ((x as u16) << 8) | (x as u16)).collect::<Vec<_>>();

    // let start = Instant::now();
    //
    // libblur::fast_gaussian_next(
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     75,
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Adaptive,
    //     EdgeMode::Clamp,
    // );
    //
    // println!("stackblur {:?}", start.elapsed());

    // dst_bytes = spawned_bytes.iter().map(|&x| (x >> 8) as u8).collect::<Vec<_>>();

    // //
    // libblur::gaussian_box_blur(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     151,
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Single,
    // );
    // // bytes = dst_bytes;

    let start_time = Instant::now();
    libblur::fast_gaussian(
        &mut dst_bytes,
        stride as u32,
        dimensions.0,
        dimensions.1,
        125,
        FastBlurChannels::Channels3,
        ThreadingPolicy::Adaptive,
        EdgeMode::Clamp,
    );
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
    // //

    println!("libblur::stack_blur: {:?}", start_time.elapsed());

    let start = Instant::now();

    // libblur::gaussian_blur(
    //     &bytes,
    //     &mut dst_bytes,
    //     dimensions.0,
    //     dimensions.1,
    //     0,
    //     16f32,
    //     FastBlurChannels::Channels3,
    //     EdgeMode::Clamp,
    //     ThreadingPolicy::Single,
    //     GaussianPreciseLevel::EXACT,
    // );

    // libblur::gaussian_box_blur(
    //     &bytes,
    //     dimensions.0 * 3,
    //     &mut dst_bytes,
    //     dimensions.0 * 3,
    //     dimensions.0,
    //     dimensions.1,
    //     5f32,
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Single,
    // );

    println!("gaussian_blur {:?}", start.elapsed());

    //
    // //
    // println!(
    //     "pure gaussian_blur: {:?}",
    //     start_time.elapsed()
    // );

    // stack_blur_f16(
    //     &mut f16_bytes,
    //     dimensions.0,
    //     dimensions.1,
    //     50,
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Single,
    // );

    // fast_gaussian(
    //     &mut dst_bytes,
    //     dimensions.0 * components as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     125,
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Single,
    //     EdgeMode::Clamp,
    // );

    // fast_bilateral_filter(
    //     src_bytes,
    //     &mut dst_bytes,
    //     dimensions.0,
    //     dimensions.1,
    //     75,
    //     75f32,
    //     755f32,
    //     FastBlurChannels::Channels3,
    // );

    // dst_bytes = f16_bytes
    //     .iter()
    //     .map(|&x| (x.to_f32() * 255f32) as u8)
    //     .collect();

    // let adaptive_blurred = adaptive_blur(
    //     &bytes,
    //     dimensions.0 as usize,
    //     dimensions.1 as usize,
    //     55,
    //     FastBlurChannels::Channels3,
    //     TransferFunction::Srgb,
    //     EdgeMode::Reflect101,
    //     Scalar::default(),
    // );
    //
    // image::save_buffer(
    //     "edges.jpg",
    //     &adaptive_blurred,
    //     dimensions.0,
    //     dimensions.1,
    //     image::ExtendedColorType::Rgb8,
    // )
    // .unwrap();

    // dst_bytes = perform_planar_pass_3(&bytes, dimensions.0 as usize, dimensions.1 as usize);

    // let kernel = get_gaussian_kernel_1d(51, get_sigma_size(51));
    // // // dst_bytes.fill(0);
    // //
    // // let sobel_horizontal: [i16; 3] = [-1, 0, 1];
    // // let sobel_vertical: [i16; 3] = [1, 2, 1];
    // //
    // filter_1d_rgb_exact::<u8, f32>(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     &kernel,
    //     &kernel,
    //     EdgeMode::Clamp,
    //     Scalar::new(255.0, 0., 0., 255.0),
    //     ThreadingPolicy::default(),
    // )
    // .unwrap();

    // motion_blur(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     90f32,
    //     35,
    //     EdgeMode::Wrap,
    //     Scalar::new(255.0, 0.0, 0.0, 255.0),
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Adaptive,
    // );

    // laplacian(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     EdgeMode::Clamp,
    //     Scalar::default(),
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Adaptive,
    // );
    //
    // let motion_kernel = generate_motion_kernel(501, 15.);

    // motion_blur(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     15.,
    //     225,
    //     EdgeMode::Clamp,
    //     Scalar::new(255.0, 0.0, 0.0, 255.0),
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Adaptive,
    // );

    // filter_2d_rgb_fft::<u8, f32, f32>(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     &motion_kernel,
    //     KernelShape::new(501, 501),
    //     EdgeMode::Clamp,
    //     Scalar::new(255.0, 0., 0., 255.0),
    // )
    // .unwrap();

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

    // libblur::gaussian_blur(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     25,
    //     0.,
    //     FastBlurChannels::Channels4,
    //     EdgeMode::Reflect101,
    //     ThreadingPolicy::Adaptive,
    //     GaussianPreciseLevel::INTEGRAL,
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
            "blurred_stack_linear.jpg",
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
