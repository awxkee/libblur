use criterion::{criterion_group, criterion_main, Criterion};
use image::{EncodableLayout, GenericImageView, ImageReader};

use libblur::{BlurImageMut, EdgeMode, FastBlurChannels, ThreadingPolicy};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/test_image_4.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let src_bytes = img.as_bytes();

    c.bench_function("RGBA fast gaussian", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::fast_gaussian(
                &mut dst_image,
                77,
                ThreadingPolicy::Adaptive,
                EdgeMode::Clamp,
            )
            .unwrap();
        })
    });

    c.bench_function("RGBA fast gaussian ( Single Thread )", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::fast_gaussian(&mut dst_image, 77, ThreadingPolicy::Single, EdgeMode::Clamp)
                .unwrap();
        })
    });

    let rgba_f32 = img.to_rgba32f();

    c.bench_function("RGBA f32 fast gaussian (Single Thread)", |b| {
        let mut dst_bytes: Vec<f32> = rgba_f32.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::fast_gaussian_f32(
                &mut dst_image,
                77,
                ThreadingPolicy::Single,
                EdgeMode::Clamp,
            )
            .unwrap();
        })
    });

    let rgba_u16 = img.to_rgba16();

    c.bench_function("RGBA16 fast gaussian (Single Thread)", |b| {
        let mut dst_bytes: Vec<u16> = rgba_u16.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::fast_gaussian_u16(
                &mut dst_image,
                77,
                ThreadingPolicy::Single,
                EdgeMode::Clamp,
            )
            .unwrap();
        })
    });

    let img = ImageReader::open("../assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();

    let rgb_image = img.to_rgb8();
    let rgb_src_bytes = rgb_image.as_bytes();

    c.bench_function("RGB fast gaussian", |b| {
        let mut dst_bytes: Vec<u8> = rgb_src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels3,
        );
        b.iter(|| {
            libblur::fast_gaussian(
                &mut dst_image,
                77,
                ThreadingPolicy::Adaptive,
                EdgeMode::Clamp,
            )
            .unwrap();
        })
    });

    c.bench_function("RGB fast gaussian ( Single Thread )", |b| {
        let mut dst_bytes: Vec<u8> = rgb_src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels3,
        );
        b.iter(|| {
            libblur::fast_gaussian(&mut dst_image, 77, ThreadingPolicy::Single, EdgeMode::Clamp)
                .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
