use criterion::{criterion_group, criterion_main, Criterion};
use image::{EncodableLayout, GenericImageView, ImageReader};
use libblur::{BlurImage, BlurImageMut, FastBlurChannels, ThreadingPolicy};
use opencv::core::{find_file, Mat};
use opencv::imgcodecs::{imread, IMREAD_COLOR};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut c = c.benchmark_group("Median");
    c.sample_size(10);
    let img = ImageReader::open("../assets/test_image_4.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let src_bytes = img.as_bytes();
    let src_image = BlurImage::borrow(
        src_bytes,
        dimensions.0,
        dimensions.1,
        FastBlurChannels::Channels4,
    );
    c.bench_function("libblur: RGBA median blur", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::median_blur(&src_image, &mut dst_image, 7, ThreadingPolicy::Adaptive).unwrap();
        })
    });

    let src = imread(
        &find_file("../assets/test_image_4.png", false, false).unwrap(),
        IMREAD_COLOR,
    )
    .unwrap();
    c.bench_function("OpenCV: RGBA median blur", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::median_blur(&src, &mut dst, 7).unwrap();
        })
    });

    let img = ImageReader::open("../assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let rgb_img = img.to_rgb8();
    let rgb_image = rgb_img.as_bytes();

    let rgb_image = BlurImage::borrow(
        rgb_image,
        dimensions.0,
        dimensions.1,
        FastBlurChannels::Channels4,
    );

    c.bench_function("libblur: RGB median blur", |b| {
        let mut dst_bytes: Vec<u8> = rgb_image.data.as_ref().to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels3,
        );
        b.iter(|| {
            libblur::median_blur(&rgb_image, &mut dst_image, 7, ThreadingPolicy::Adaptive).unwrap();
        })
    });

    let src_rgb = imread(
        &find_file("../assets/test_image_1.jpg", false, false).unwrap(),
        IMREAD_COLOR,
    )
    .unwrap();

    c.bench_function("OpenCV: RGB median blur", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::median_blur(&src_rgb, &mut dst, 7).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
