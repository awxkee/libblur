use criterion::{criterion_group, criterion_main, Criterion};
use image::ImageReader;
use libblur::{
    filter_1d_complex, filter_1d_complex_fixed_point, BlurImage, BlurImageMut, EdgeMode,
    FastBlurChannels, Scalar, ThreadingPolicy,
};
use num_complex::Complex;

fn complex_gaussian_kernel(radius: f64, scale: f64, distortion: f64) -> Vec<Complex<f32>> {
    let kernel_radius = radius.ceil() as usize;
    let mut kernel: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 1 + 2 * (kernel_radius)];

    for (x, dst) in kernel.iter_mut().enumerate() {
        let ax = (x as f64 - radius) * scale / radius;
        let ax2 = ax * ax;
        let exp_a = (-distortion * ax2).exp();
        let val = Complex::new(
            exp_a * (distortion * ax2).cos(),
            exp_a * (distortion * ax2).sin(),
        );
        *dst = val;
    }

    let sum: f64 = kernel.iter().map(|z| z.norm_sqr()).sum::<f64>();
    if sum != 0.0 {
        kernel
            .iter()
            .map(|z| Complex {
                re: (z.re / sum) as f32,
                im: (z.im / sum) as f32,
            })
            .collect::<Vec<_>>()
    } else {
        kernel
            .iter()
            .map(|x| Complex {
                re: x.re as f32,
                im: x.im as f32,
            })
            .collect()
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut c = c.benchmark_group("Complex");
    c.sample_size(10);

    let img = ImageReader::open("../assets/test_image_4.png")
        .unwrap()
        .decode()
        .unwrap();
    let src_image = BlurImage::borrow(
        img.as_bytes(),
        img.width(),
        img.height(),
        FastBlurChannels::Channels4,
    );

    c.bench_function("RGBA gauss complex blur [Fixed Point]: 25", |b| {
        let mut dst_bytes =
            BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
        let gaussian_kernel = complex_gaussian_kernel(25., 0.5, 15.);
        b.iter(|| {
            filter_1d_complex_fixed_point::<u8, i16, f32, 4>(
                &src_image,
                &mut dst_bytes,
                &gaussian_kernel,
                &gaussian_kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        })
    });

    c.bench_function("RGBA gauss complex blur: 25", |b| {
        let mut dst_bytes =
            BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
        let gaussian_kernel = complex_gaussian_kernel(25., 0.5, 15.);
        b.iter(|| {
            filter_1d_complex::<u8, f32, 4>(
                &src_image,
                &mut dst_bytes,
                &gaussian_kernel,
                &gaussian_kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        })
    });

    let tmp16 = img
        .as_bytes()
        .iter()
        .map(|&x| u16::from_ne_bytes([x, x]))
        .collect::<Vec<u16>>();
    let src_image16 = BlurImage::borrow(
        &tmp16,
        img.width(),
        img.height(),
        FastBlurChannels::Channels4,
    );

    c.bench_function("RGBA16 gauss complex blur [Fixed Point]: 25", |b| {
        let mut dst_bytes =
            BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
        let gaussian_kernel = complex_gaussian_kernel(25., 0.5, 15.);
        b.iter(|| {
            filter_1d_complex_fixed_point::<u16, i32, f32, 4>(
                &src_image16,
                &mut dst_bytes,
                &gaussian_kernel,
                &gaussian_kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        })
    });

    c.bench_function("RGBA16 gauss complex blur: 25", |b| {
        let mut dst_bytes =
            BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
        let gaussian_kernel = complex_gaussian_kernel(25., 0.5, 15.);
        b.iter(|| {
            filter_1d_complex::<u16, f32, 4>(
                &src_image16,
                &mut dst_bytes,
                &gaussian_kernel,
                &gaussian_kernel,
                EdgeMode::Clamp,
                Scalar::default(),
                ThreadingPolicy::Single,
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
