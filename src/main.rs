use fastblur::FastBlurChannels;
use image::io::Reader as ImageReader;
use image::{EncodableLayout, GenericImageView};
use std::time::Instant;

fn main() {
    let img = ImageReader::open("assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    println!("dimensions {:?}", img.dimensions());

    println!("{:?}", img.color());
    let src_bytes = img.as_bytes();
    let mut bytes: Vec<u8> = Vec::with_capacity((dimensions.0 * dimensions.1 * 3) as usize);
    for i in 0..(dimensions.0 * dimensions.1 * 3) as usize {
        bytes.push(src_bytes[i]);
    }
    let mut dst_bytes: Vec<u8> = Vec::with_capacity((dimensions.0 * dimensions.1 * 3) as usize);
    dst_bytes.resize((dimensions.0 * dimensions.1 * 3) as usize, 0);
    let start_time = Instant::now();
    // fastblur::fast_gaussian_next(&mut bytes, dimensions.0 * 3, dimensions.0, dimensions.1, 125, FastBlurChannels::Channels3);
    fastblur::gaussian_blur(
        &bytes,
        dimensions.0 * 3,
        &mut dst_bytes,
        dimensions.0 * 3,
        dimensions.0,
        dimensions.1,
        151,
        151f32 / 3f32,
        FastBlurChannels::Channels3,
    );
    // fastblur::median_blur(
    //     &bytes,
    //     dimensions.0 * 3,
    //     &mut dst_bytes,
    //     dimensions.0 * 3,
    //     dimensions.0,
    //     dimensions.1,
    //     175,
    //     FastBlurChannels::Channels3,
    // );
    // fastblur::tent_blur(&bytes, dimensions.0 * 3, &mut dst_bytes, dimensions.0 * 3, dimensions.0, dimensions.1, 71, FastBlurChannels::Channels3);

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Elapsed time: {:.2?}", elapsed_time);

    image::save_buffer(
        "blurred.jpg",
        dst_bytes.as_bytes(),
        dimensions.0,
        dimensions.1,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
}
