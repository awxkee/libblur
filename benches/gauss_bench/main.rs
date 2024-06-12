use criterion::{criterion_group, criterion_main, Criterion};
use image::io::Reader as ImageReader;
use image::GenericImageView;
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
    c.bench_function("RGBA gauss blur kernel clip", |b| {
        b.iter(|| {
            let mut dst_bytes: Vec<u8> = Vec::with_capacity(dimensions.1 as usize * stride);
            dst_bytes.resize(dimensions.1 as usize * stride, 0);
            libblur::gaussian_blur(
                &src_bytes,
                stride as u32,
                &mut dst_bytes,
                stride as u32,
                dimensions.0,
                dimensions.1,
                55 * 2 + 1,
                (55f32 * 2f32 + 1f32) / 6f32,
                FastBlurChannels::Channels4,
                EdgeMode::KernelClip,
                ThreadingPolicy::Single,
            );
        })
    });
    c.bench_function("RGBA gauss blur edge clamp", |b| {
        b.iter(|| {
            let mut dst_bytes: Vec<u8> = Vec::with_capacity(dimensions.1 as usize * stride);
            dst_bytes.resize(dimensions.1 as usize * stride, 0);
            libblur::gaussian_blur(
                &src_bytes,
                stride as u32,
                &mut dst_bytes,
                stride as u32,
                dimensions.0,
                dimensions.1,
                55 * 2 + 1,
                (55f32 * 2f32 + 1f32) / 6f32,
                FastBlurChannels::Channels4,
                EdgeMode::Clamp,
                ThreadingPolicy::Single,
            );
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
