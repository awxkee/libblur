use criterion::{criterion_group, criterion_main, Criterion};
use image::{EncodableLayout, GenericImageView, ImageReader};

use libblur::{EdgeMode, FastBlurChannels, ThreadingPolicy};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("assets/test_image_4.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let components = 4;
    let stride = dimensions.0 as usize * components;
    let src_bytes = img.as_bytes();
    c.bench_function("RGBA fast gaussian", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        b.iter(|| {
            libblur::fast_gaussian(
                &mut dst_bytes,
                stride as u32,
                dimensions.0,
                dimensions.1,
                77,
                FastBlurChannels::Channels4,
                ThreadingPolicy::Adaptive,
                EdgeMode::Clamp,
            );
        })
    });

    let rgba_f32 = img.to_rgba32f();

    c.bench_function("RGBA f32 fast gaussian (Single Thread)", |b| {
        let mut dst_bytes: Vec<f32> = rgba_f32.to_vec();
        b.iter(|| {
            libblur::fast_gaussian_f32(
                &mut dst_bytes,
                dimensions.0,
                dimensions.1,
                77,
                FastBlurChannels::Channels4,
                ThreadingPolicy::Single,
                EdgeMode::Clamp,
            );
        })
    });

    let img = ImageReader::open("assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();

    let rgb_image = img.to_rgb8();
    let rgb_src_bytes = rgb_image.as_bytes();

    c.bench_function("RGB fast gaussian", |b| {
        let mut dst_bytes: Vec<u8> = rgb_src_bytes.to_vec();
        b.iter(|| {
            libblur::fast_gaussian(
                &mut dst_bytes,
                img.dimensions().0 * 3,
                img.dimensions().0,
                img.dimensions().1,
                77,
                FastBlurChannels::Channels3,
                ThreadingPolicy::Adaptive,
                EdgeMode::Clamp,
            );
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
