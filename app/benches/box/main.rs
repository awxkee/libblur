use criterion::{criterion_group, criterion_main, Criterion};
use image::{EncodableLayout, GenericImageView, ImageReader};
use libblur::{BlurImage, BlurImageMut, FastBlurChannels, ThreadingPolicy};
use opencv::core::{find_file, Mat, Point, Size, BORDER_DEFAULT};
use opencv::imgcodecs::{imread, IMREAD_COLOR};

pub fn criterion_benchmark(c: &mut Criterion) {
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

    c.bench_function("libblur: RGBA box blur", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();

        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        b.iter(|| {
            libblur::box_blur(
                &src_image,
                &mut dst_image,
                77 / 2,
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        })
    });

    c.bench_function("libblur: RGBA16 box blur", |b| {
        let src_bytes: Vec<u16> = src_bytes.to_vec().iter().map(|v| *v as u16).collect();
        let mut dst_bytes: Vec<u16> = src_bytes.to_vec().iter().map(|v| *v).collect();

        let src_image = BlurImage::borrow(
            &src_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        b.iter(|| {
            libblur::box_blur_u16(
                &src_image,
                &mut dst_image,
                77 / 2,
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        })
    });

    c.bench_function("libblur: RGBA box blur f32", |b| {
        let mut dst_bytes: Vec<f32> = src_bytes.to_vec().iter().map(|v| *v as f32).collect();
        let src_bytes = src_bytes
            .to_vec()
            .iter()
            .map(|v| *v as f32)
            .collect::<Vec<f32>>();

        let src_image = BlurImage::borrow(
            &src_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        b.iter(|| {
            libblur::box_blur_f32(
                &src_image,
                &mut dst_image,
                77 / 2,
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        })
    });

    c.bench_function("libblur: RGBA box blur f32, Single", |b| {
        let mut dst_bytes: Vec<f32> = src_bytes.to_vec().iter().map(|v| *v as f32).collect();
        let src_bytes = src_bytes
            .to_vec()
            .iter()
            .map(|v| *v as f32)
            .collect::<Vec<f32>>();

        let src_image = BlurImage::borrow(
            &src_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        b.iter(|| {
            libblur::box_blur_f32(&src_image, &mut dst_image, 77 / 2, ThreadingPolicy::Single)
                .unwrap();
        })
    });

    c.bench_function("libblur: RGBA box blur (15) Single Thread", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();

        let src_image = BlurImage::borrow(
            &src_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        b.iter(|| {
            libblur::box_blur(&src_image, &mut dst_image, 15, ThreadingPolicy::Single).unwrap();
        })
    });

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    c.bench_function("Apple Accelerate: RGBA box blur (15) Single Thread", |b| {
        use accelerate::acc_convenience::box_convolve;
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let stride = dimensions.0 as usize * 4;
        b.iter(|| {
            box_convolve(
                src_bytes,
                stride,
                &mut dst_bytes,
                stride,
                31,
                dimensions.0 as usize,
                dimensions.1 as usize,
                false,
            );
        })
    });

    c.bench_function("libblur: RGBA box blur (15) MultiThreaded", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();

        let src_image = BlurImage::borrow(
            &src_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );

        b.iter(|| {
            libblur::box_blur(&src_image, &mut dst_image, 15, ThreadingPolicy::Adaptive).unwrap();
        })
    });

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    c.bench_function("Apple Accelerate: RGBA box blur (15) MultiThreaded", |b| {
        use accelerate::acc_convenience::box_convolve;
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();

        let stride = dimensions.0 as usize * 4;

        b.iter(|| {
            box_convolve(
                src_bytes,
                stride,
                &mut dst_bytes,
                stride,
                31,
                dimensions.0 as usize,
                dimensions.1 as usize,
                true,
            );
        })
    });

    let src = imread(
        &find_file("../assets/test_image_4.png", false, false).unwrap(),
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

    let img = ImageReader::open("../assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let rgb_img = img.to_rgb8();
    let rgb_image = rgb_img.as_bytes();

    c.bench_function("libblur: RGB box blur", |b| {
        let mut dst_bytes: Vec<u8> = rgb_image.to_vec();

        let src_image = BlurImage::borrow(
            &src_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels3,
        );

        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels3,
        );

        b.iter(|| {
            libblur::box_blur(
                &src_image,
                &mut dst_image,
                77 / 2,
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        })
    });

    let src_rgb = imread(
        &find_file("../assets/test_image_1.jpg", false, false).unwrap(),
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
