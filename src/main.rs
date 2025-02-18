mod merge;
mod split;

use crate::merge::merge_channels_3;
use crate::split::split_channels_3;
use colorutils_rs::TransferFunction;
use fast_transpose::{transpose_rgba, FlipMode, FlopMode};
use image::imageops::FilterType;
use image::{DynamicImage, EncodableLayout, GenericImageView, ImageReader};
use libblur::{
    fast_gaussian, filter_1d_exact, filter_2d_rgb_fft, generate_motion_kernel,
    get_gaussian_kernel_1d, get_sigma_size, laplacian, motion_blur, sobel, ConvolutionMode,
    EdgeMode, FastBlurChannels, ImageSize, KernelShape, Scalar, ThreadingPolicy,
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
        ConvolutionMode::Exact,
    )
    .unwrap();

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
        ConvolutionMode::Exact,
    )
    .unwrap();

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
    let mut dyn_image = ImageReader::open("assets/test_image_2.png")
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

    // let mut spawned_bytes = dst_bytes.iter().map(|&x| ((x as u16) << 8) | (x as u16)).collect::<Vec<_>>();

    // let start = Instant::now();
    //
    /*libblur::fast_gaussian(
        &mut dst_bytes,
        stride as u32,
        dimensions.0,
        dimensions.1,
        75,
        FastBlurChannels::Channels3,
        ThreadingPolicy::Single,
        EdgeMode::Clamp,
    );*/
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
    //     48f32,
    //     FastBlurChannels::Channels4,
    //     ThreadingPolicy::Single,
    // );
    // // bytes = dst_bytes;

    let start_time = Instant::now();

    // let bytes_16 = bytes.iter().map(|&x| x as u16).collect::<Vec<u16>>();
    // let mut dst_16 = vec![0u16; bytes_16.len()];
    //
    // libblur::gaussian_blur_u16(
    //     &bytes_16,
    //     &mut dst_16,
    //     dimensions.0,
    //     dimensions.1,
    //     0,
    //     12.,
    //     FastBlurChannels::Channels3,
    //     EdgeMode::Clamp,
    //     ThreadingPolicy::Single,
    // );
    //
    // dst_bytes = dst_16.iter().map(|&x| x as u8).collect();

    // sobel(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     EdgeMode::Clamp,
    //     Scalar::new(255.0, 0.0, 0.0, 255.0),
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Single,
    // );

    // let mut f16_bytes: Vec<f16> = dst_bytes
    //     .iter()
    //     .map(|&x| f16::from_f32(x as f32 * (1. / 255.)))
    //     .collect();
    // //

    println!("libblur::stack_blur: {:?}", start_time.elapsed());

    let start = Instant::now();

    // libblur::stack_blur_f32(
    //     &mut j_vet,
    //     dimensions.0,
    //     dimensions.1,
    //     25,
    //     FastBlurChannels::Channels4,
    //     ThreadingPolicy::Adaptive,
    // );

    // libblur::gaussian_blur_f32(
    //     &img_f32,
    //     &mut j_vet,
    //     dimensions.0,
    //     dimensions.1,
    //     21,
    //     3.5f32,
    //     FastBlurChannels::Channels4,
    //     EdgeMode::Clamp,
    //     ThreadingPolicy::Single,
    // );

    // bytes = j_vet.iter().map(|&x| (x * 255f32).round() as u8).collect();
    //
    // libblur::gaussian_box_blur_in_linear(
    //     &bytes,
    //     dimensions.0 * 4,
    //     &mut dst_bytes,
    //     dimensions.0 * 4,
    //     dimensions.0,
    //     dimensions.1,
    //     10f32,
    //     FastBlurChannels::Channels4,
    //     ThreadingPolicy::Single,
    //     TransferFunction::Rec709,
    // );

    // accelerate::acc_convenience::box_convolve(
    //     &bytes,
    //     dimensions.0 as usize * 4,
    //     &mut dst_bytes,
    //     dimensions.0 as usize * 4,
    //   11,
    //     dimensions.0 as usize,
    //     dimensions.1 as usize,
    // );

    println!("gaussian_blur {:?}", start.elapsed());

    // fast_gaussian(
    //     &mut dst_bytes,
    //     dimensions.0 * components as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     15,
    //     FastBlurChannels::Channels4,
    //     ThreadingPolicy::Single,
    //     EdgeMode::Clamp,
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

    motion_blur(
        &bytes,
        dimensions.0 as usize * 3,
        &mut dst_bytes,
        dimensions.0 as usize * 3,
        ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
        90f32,
        35,
        EdgeMode::Clamp,
        Scalar::new(255.0, 0.0, 0.0, 255.0),
        FastBlurChannels::Channels3,
        ThreadingPolicy::Adaptive,
    )
    .unwrap();

    let motion_kernel = generate_motion_kernel(51, 18.);

    // laplacian(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     EdgeMode::Clamp,
    //     Scalar::new(255.0, 0.0, 0.0, 255.0),
    //     FastBlurChannels::Channels3,
    //     ThreadingPolicy::Adaptive,
    // )
    // .unwrap();

    // filter_2d_rgb_fft::<u8, f32, f32>(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     &motion_kernel,
    //     KernelShape::new(51, 51),
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
