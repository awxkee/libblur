use criterion::{criterion_group, criterion_main, Criterion};
use image::{EncodableLayout, GenericImageView, ImageReader};
use libblur::{BlurImageMut, FastBlurChannels, ThreadingPolicy};
use opencv::core::{find_file, Mat, Size};
use opencv::imgcodecs::{imread, IMREAD_COLOR};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/test_image_4.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let src_bytes = img.as_bytes();
    c.bench_function("libblur: RGBA stack blur", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::stack_blur(&mut dst_image, 77, ThreadingPolicy::Adaptive).unwrap();
        })
    });

    let src = imread(
        &find_file("../assets/test_image_4.png", false, false).unwrap(),
        IMREAD_COLOR,
    )
    .unwrap();
    c.bench_function("OpenCV: RGBA stack blur", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::stack_blur(&src, &mut dst, Size::new(77, 77)).unwrap();
        })
    });

    let img = ImageReader::open("../assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let rgb_img = img.to_rgb8();
    let rgb_image = rgb_img.as_bytes();

    c.bench_function("libblur: RGB stack blur", |b| {
        let mut dst = rgb_image.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels3,
        );
        b.iter(|| {
            libblur::stack_blur(&mut dst_image, 77, ThreadingPolicy::Adaptive).unwrap();
        });
    });

    let src_rgb = imread(
        &find_file("../assets/test_image_1.jpg", false, false).unwrap(),
        IMREAD_COLOR,
    )
    .unwrap();

    c.bench_function("OpenCV: RGB stack blur", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::stack_blur(&src_rgb, &mut dst, Size::new(77, 77)).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
