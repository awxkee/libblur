use image::io::Reader as ImageReader;
use image::{EncodableLayout, GenericImageView};
use libblur::FastBlurChannels;
use std::time::Instant;

#[allow(dead_code)]
fn f32_to_f16(bytes: Vec<f32>) -> Vec<u16> {
    return bytes.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
}

#[allow(dead_code)]
fn f16_to_f32(bytes: Vec<u16>) -> Vec<f32> {
    return bytes.iter().map(|&x| half::f16::from_bits(x).to_f32()).collect();
}


fn main() {
    let img = ImageReader::open("assets/filirovska.jpeg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    println!("dimensions {:?}", img.dimensions());

    println!("{:?}", img.color());
    let src_bytes = img.as_bytes();
    let stride = dimensions.0 as usize * 3;
    let mut bytes: Vec<u8> = Vec::with_capacity(dimensions.1 as usize * stride);
    for i in 0..dimensions.1 as usize * stride {
        bytes.push(src_bytes[i]);
    }
    let mut dst_bytes: Vec<u16> = Vec::with_capacity(dimensions.1 as usize * stride);
    dst_bytes.resize(dimensions.1 as usize * stride, 0);
    let start_time = Instant::now();

    libblur::fast_gaussian(
        &mut bytes,
        stride as u32,
        dimensions.0,
        dimensions.1,
        151,
        FastBlurChannels::Channels3,
    );
    // libblur::gaussian_blur_f16(
    //     &f16_bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     151,
    //     151f32 / 3f32,
    //     FastBlurChannels::Channels3,
    // );
    // libblur::median_blur(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     36,
    //     FastBlurChannels::Channels3,
    // );
    // libblur::gaussian_box_blur(&bytes, stride as u32, &mut dst_bytes, stride as u32, dimensions.0, dimensions.1, 128, FastBlurChannels::Channels4);

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Elapsed time: {:.2?}", elapsed_time);

    image::save_buffer(
        "blurred.png",
        bytes.as_bytes(),
        dimensions.0,
        dimensions.1,
        image::ExtendedColorType::Rgb8,
    )
        .unwrap();
}
