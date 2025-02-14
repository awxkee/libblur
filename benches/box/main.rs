use criterion::{criterion_group, criterion_main, Criterion};
use image::{EncodableLayout, GenericImageView, ImageReader};
use libblur::{FastBlurChannels, ThreadingPolicy};
use opencv::core::{find_file, Mat, Point, Size, BORDER_DEFAULT};
use opencv::imgcodecs::{imread, IMREAD_COLOR};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("assets/test_image_4.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let components = 4;
    let stride = dimensions.0 as usize * components;
    let src_bytes = img.as_bytes();

    c.bench_function("libblur: RGBA box blur", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        b.iter(|| {
            libblur::box_blur(
                src_bytes,
                stride as u32,
                &mut dst_bytes,
                stride as u32,
                dimensions.0,
                dimensions.1,
                77 / 2,
                FastBlurChannels::Channels4,
                ThreadingPolicy::Adaptive,
            );
        })
    });

    c.bench_function("libblur: RGBA box blur Single Thread", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        b.iter(|| {
            libblur::box_blur(
                src_bytes,
                stride as u32,
                &mut dst_bytes,
                stride as u32,
                dimensions.0,
                dimensions.1,
                15,
                FastBlurChannels::Channels4,
                ThreadingPolicy::Single,
            );
        })
    });

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    c.bench_function("Apple Accelerate: RGBA box blur Single Thread", |b| {
        use accelerate::acc_convenience::box_convolve;
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        b.iter(|| {
            box_convolve(
                src_bytes,
                stride,
                &mut dst_bytes,
                stride,
                15,
                dimensions.0 as usize,
                dimensions.1 as usize,
            );
        })
    });

    let src = imread(
        &find_file("assets/test_image_4.png", false, false).unwrap(),
        IMREAD_COLOR,
    )
    .unwrap();
    c.bench_function("OpenCV: RGBA box blur", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::box_filter(
                &src,
                &mut dst,
                -1,
                Size::new(77, 77),
                Point::new(-1, -1),
                false,
                BORDER_DEFAULT,
            )
            .unwrap();
        })
    });

    let img = ImageReader::open("assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let rgb_img = img.to_rgb8();
    let rgb_image = rgb_img.as_bytes();

    c.bench_function("libblur: RGB box blur", |b| {
        let mut dst_bytes: Vec<u8> = rgb_image.to_vec();
        b.iter(|| {
            libblur::box_blur(
                src_bytes,
                rgb_img.dimensions().0 * 3,
                &mut dst_bytes,
                rgb_img.dimensions().0 * 3,
                rgb_img.dimensions().0,
                rgb_img.dimensions().1,
                77 / 2,
                FastBlurChannels::Channels3,
                ThreadingPolicy::Adaptive,
            );
        })
    });

    let src_rgb = imread(
        &find_file("assets/test_image_1.jpg", false, false).unwrap(),
        IMREAD_COLOR,
    )
    .unwrap();

    c.bench_function("OpenCV: RGB box blur", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::box_filter(
                &src_rgb,
                &mut dst,
                -1,
                Size::new(77, 77),
                Point::new(-1, -1),
                false,
                BORDER_DEFAULT,
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
