use criterion::{criterion_group, criterion_main, Criterion};
use image::{EncodableLayout, GenericImageView, ImageReader};
use libblur::{
    filter_2d_rgb_fft, generate_motion_kernel, BlurImage, BlurImageMut, EdgeMode, FastBlurChannels,
    KernelShape, Scalar, ThreadingPolicy,
};

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
        FastBlurChannels::Channels3,
    );
    c.bench_function("libblur: RGB motion blur", |b| {
        let mut dst_bytes: Vec<u8> = src_bytes.to_vec();
        let mut dst_image = BlurImageMut::borrow(
            &mut dst_bytes,
            dimensions.0,
            dimensions.1,
            FastBlurChannels::Channels3,
        );
        let motion = generate_motion_kernel(35, 24.);
        b.iter(|| {
            filter_2d_rgb_fft::<u8, f32, f32>(
                &src_image,
                &mut dst_image,
                &motion,
                KernelShape::new(35, 35),
                EdgeMode::Clamp.as_2d(),
                Scalar::default(),
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
