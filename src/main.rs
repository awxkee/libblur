use colorutils_rs::{
    linear_to_rgba, luv_with_alpha_to_bgra, luv_with_alpha_to_rgba, rgba_to_linear,
    rgba_to_luv_with_alpha, rgba_to_sigmoidal, sigmoidal_to_rgba, TransferFunction,
};
use image::io::Reader as ImageReader;
use image::{EncodableLayout, GenericImageView};
use libblur::{EdgeMode, FastBlurChannels, ThreadingPolicy};
use std::time::Instant;

#[allow(dead_code)]
fn f32_to_f16(bytes: Vec<f32>) -> Vec<u16> {
    return bytes
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
}

#[allow(dead_code)]
fn f16_to_f32(bytes: Vec<u16>) -> Vec<f32> {
    return bytes
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();
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
    let img = ImageReader::open("assets/beach_horizon.jpg")
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
    let mut bytes: Vec<u8> = Vec::with_capacity(dimensions.1 as usize * stride);
    for i in 0..dimensions.1 as usize * stride {
        bytes.push(src_bytes[i]);
    }
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

    // libblur::stack_blur(
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     75,
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

    libblur::fast_gaussian_in_linear(
        &mut dst_bytes,
        stride as u32,
        dimensions.0,
        dimensions.1,
        168,
        FastBlurChannels::Channels3,
        ThreadingPolicy::Single,
        TransferFunction::Srgb,
        EdgeMode::Reflect,
    );

    // libblur::gaussian_blur_in_linear(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     75 * 2 + 1,
    //     75f32 * 2f32 / 6f32,
    //     FastBlurChannels::Channels3,
    //     EdgeMode::Clamp,
    //     ThreadingPolicy::Adaptive,
    //     TransferFunction::Srgb,
    // );

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
            "blurred_stack_cpu.jpg",
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
