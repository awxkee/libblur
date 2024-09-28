use criterion::{criterion_group, criterion_main, Criterion};
use image::{GenericImageView, ImageReader};

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
        b.iter(|| {
            let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
