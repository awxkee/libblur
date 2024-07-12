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
    c.bench_function("RGBA fast gaussian", |b| {
        b.iter(|| {
            let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_bytes.as_ptr(),
                    dst_bytes.as_mut_ptr(),
                    dimensions.1 as usize * stride,
                );
            }
            libblur::fast_gaussian(
                &mut dst_bytes,
                stride as u32,
                dimensions.0,
                dimensions.1,
                77,
                FastBlurChannels::Channels4,
                ThreadingPolicy::Single,
                EdgeMode::Clamp
            );
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
