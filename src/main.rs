use image::io::Reader as ImageReader;
use image::{DynamicImage, EncodableLayout, GenericImageView, ImageBuffer, Rgb};
use libblur::FastBlurChannels;
use std::time::Instant;

fn main() {
    let img = ImageReader::open("assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    println!("dimensions {:?}", img.dimensions());

    let f32_image = img.to_rgb32f();

    println!("{:?}", img.color());
    let src_bytes = f32_image.as_raw();
    let stride = dimensions.0 as usize * 3;
    let mut bytes: Vec<f32> = src_bytes.clone();
    let mut dst_bytes: Vec<u8> = Vec::with_capacity(dimensions.1 as usize * stride);
    dst_bytes.resize(dimensions.1 as usize * stride, 0);
    let start_time = Instant::now();

    libblur::fast_gaussian_next_f32(
        &mut bytes,
        stride as u32,
        dimensions.0,
        dimensions.1,
        51,
        FastBlurChannels::Channels3,
    );
    // libblur::gaussian_blur(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     171,
    //     171f32 / 3f32,
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

    let img = ImageBuffer::<Rgb<f32>, _>::from_raw(dimensions.0, dimensions.1, bytes);

    let dynamic_img: DynamicImage = DynamicImage::ImageRgb32F(img.unwrap());

    let u8_img = dynamic_img.to_rgb8();

    image::save_buffer(
        "blurred.png",
        u8_img.as_bytes(),
        dimensions.0,
        dimensions.1,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
}
