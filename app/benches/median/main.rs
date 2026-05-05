use criterion::{criterion_group, criterion_main, Criterion};
use image::{GenericImageView, ImageReader};
use libblur::{BlurImage, BlurImageMut, FastBlurChannels, ThreadingPolicy};
use opencv::core::{find_file, Mat};
use opencv::imgcodecs::{imread, IMREAD_COLOR};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut c = c.benchmark_group("Median");
    c.sample_size(10);
    let img = ImageReader::open("../assets/test_image_2.png")
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

    let src = imread(
        &find_file("../assets/test_image_4.png", false, false).unwrap(),
        IMREAD_COLOR,
    )
    .unwrap();

    opencv::core::set_num_threads(1).unwrap();

    c.bench_function("libblur: RGBA median blur (9)", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::median_blur(&src_image, &mut dst_image, 4, ThreadingPolicy::Single).unwrap();
        })
    });

    c.bench_function("OpenCV: RGBA median blur (9)", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::median_blur(&src, &mut dst, 9).unwrap();
        })
    });

    c.bench_function("libblur: RGBA median blur (7)", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::median_blur(&src_image, &mut dst_image, 3, ThreadingPolicy::Single).unwrap();
        })
    });

    c.bench_function("OpenCV: RGBA median blur (7)", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::median_blur(&src, &mut dst, 7).unwrap();
        })
    });

    c.bench_function("libblur: RGBA median blur (5)", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::median_blur(&src_image, &mut dst_image, 2, ThreadingPolicy::Single).unwrap();
        })
    });

    c.bench_function("OpenCV: RGBA median blur (5)", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::median_blur(&src, &mut dst, 5).unwrap();
        })
    });

    c.bench_function("libblur: RGBA median blur (3)", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::median_blur(&src_image, &mut dst_image, 1, ThreadingPolicy::Single).unwrap();
        })
    });

    c.bench_function("OpenCV: RGBA median blur (3)", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::median_blur(&src, &mut dst, 3).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
