use criterion::{criterion_group, criterion_main, Criterion};
use image::{GenericImageView, ImageReader};
use libblur::{
    filter_1d_rgb_exact, gaussian_kernel_1d, sigma_size, BlurImage, BlurImageMut, ConvolutionMode,
    EdgeMode, FastBlurChannels, ImageSize, Scalar, ThreadingPolicy,
};
use opencv::core::{find_file, split, AlgorithmHint, Mat, Size, Vector, BORDER_DEFAULT};
use opencv::imgcodecs::{imread, IMREAD_COLOR};

pub(crate) fn split_channels_3<T: Copy>(
    image: &[T],
    width: usize,
    height: usize,
    first: &mut [T],
    second: &mut [T],
    third: &mut [T],
) {
    let mut shift = 0usize;
    let mut shift_plane = 0usize;
    for _ in 0..height {
        let shifted_image = &image[shift..];
        let shifted_first_plane = &mut first[shift_plane..];
        let shifted_second_plane = &mut second[shift_plane..];
        let shifted_third_plane = &mut third[shift_plane..];
        for x in 0..width {
            let px = x * 3;
            shifted_first_plane[x] = shifted_image[px];
            shifted_second_plane[x] = shifted_image[px + 1];
            shifted_third_plane[x] = shifted_image[px + 2];
        }
        shift += width * 3;
        shift_plane += width;
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut c = c.benchmark_group("Gauss");
    c.sample_size(10);
    let img = ImageReader::open("assets/test_image_4.png")
        .unwrap()
        .decode()
        .unwrap();
    let src_image = BlurImage::borrow(
        img.as_bytes(),
        img.width(),
        img.height(),
        FastBlurChannels::Channels4,
    );

    c.bench_function("RGBA gauss blur kernel clip exact: 13", |b| {
        let mut dst_bytes =
            BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
        b.iter(|| {
            libblur::gaussian_blur(
                &src_image,
                &mut dst_bytes,
                13,
                0.,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
                ConvolutionMode::Exact,
            )
            .unwrap();
        })
    });

    c.bench_function("RGBA gauss blur clamp approx: 13", |b| {
        let mut dst_bytes =
            BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
        b.iter(|| {
            libblur::gaussian_blur(
                &src_image,
                &mut dst_bytes,
                13,
                0.,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
                ConvolutionMode::FixedPoint,
            )
            .unwrap();
        })
    });

    c.bench_function("RGBA16 gauss blur edge clamp: rad 51", |b| {
        let mut dst_bytes =
            BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
        let src_bytes = img
            .as_bytes()
            .iter()
            .map(|&v| v as u16)
            .collect::<Vec<u16>>();
        let src_image = BlurImage::borrow(
            &src_bytes,
            img.width(),
            img.height(),
            FastBlurChannels::Channels4,
        );
        b.iter(|| {
            libblur::gaussian_blur_u16(
                &src_image,
                &mut dst_bytes,
                51 * 2 + 1,
                0.,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        })
    });

    let src = imread(
        &find_file(&"assets/test_image_4.png", false, false).unwrap(),
        IMREAD_COLOR,
    )
    .unwrap();

    c.bench_function("OpenCV RGBA Gaussian: 13", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::gaussian_blur(
                &src,
                &mut dst,
                Size::new(13, 13),
                0.,
                0.,
                BORDER_DEFAULT,
                AlgorithmHint::ALGO_HINT_ACCURATE,
            )
            .unwrap();
        })
    });

    c.bench_function("RGBA gauss blur edge clamp: rad 151", |b| {
        let mut dst_bytes =
            BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
        b.iter(|| {
            libblur::gaussian_blur(
                &src_image,
                &mut dst_bytes,
                77 * 2 + 1,
                (77f32 * 2f32 + 1f32) / 6f32,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
                ConvolutionMode::Exact,
            )
            .unwrap();
        })
    });

    c.bench_function("RGBA gauss blur edge clamp approx: rad 151", |b| {
        let mut dst_bytes =
            BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels4);
        b.iter(|| {
            libblur::gaussian_blur(
                &src_image,
                &mut dst_bytes,
                77 * 2 + 1,
                (77f32 * 2f32 + 1f32) / 6f32,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
                ConvolutionMode::FixedPoint,
            )
            .unwrap();
        })
    });

    c.bench_function("OpenCV RGBA Gaussian: rad 151", |b| {
        b.iter(|| {
            let mut dst = Mat::default();
            opencv::imgproc::gaussian_blur(
                &src,
                &mut dst,
                Size::new(77 * 2 + 1, 77 * 2 + 1),
                (77f64 * 2f64 + 1f64) / 6f64,
                (77f64 * 2f64 + 1f64) / 6f64,
                BORDER_DEFAULT,
                AlgorithmHint::ALGO_HINT_ACCURATE,
            )
            .unwrap();
        })
    });

    {
        let img = ImageReader::open("assets/test_image_1.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let src_image = BlurImage::borrow(
            img.as_bytes(),
            img.width(),
            img.height(),
            FastBlurChannels::Channels3,
        );
        c.bench_function("RGB gauss blur edge clamp: 151", |b| {
            let mut dst_bytes =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels3);
            b.iter(|| {
                libblur::gaussian_blur(
                    &src_image,
                    &mut dst_bytes,
                    77 * 2 + 1,
                    (77f32 * 2f32 + 1f32) / 6f32,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    ConvolutionMode::Exact,
                )
                .unwrap();
            })
        });

        c.bench_function("RGB gauss blur edge clamp: 21", |b| {
            let mut dst_bytes =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels3);
            b.iter(|| {
                libblur::gaussian_blur(
                    &src_image,
                    &mut dst_bytes,
                    21,
                    0.,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    ConvolutionMode::Exact,
                )
                .unwrap();
            })
        });

        c.bench_function("Filter 2D Rgb Blur Clamp: 25", |b| {
            let mut dst_bytes =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels3);
            let kernel = gaussian_kernel_1d(25, sigma_size(25f32));
            b.iter(|| {
                filter_1d_rgb_exact(
                    &src_image,
                    &mut dst_bytes,
                    &kernel,
                    &kernel,
                    EdgeMode::Clamp,
                    Scalar::default(),
                    ThreadingPolicy::Adaptive,
                )
                .unwrap();
            })
        });

        c.bench_function("Filter 2D Rgb Blur Clamp: 151", |b| {
            let mut dst_bytes =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels3);
            let kernel = gaussian_kernel_1d(151, sigma_size(151f32));
            b.iter(|| {
                filter_1d_rgb_exact(
                    &src_image,
                    &mut dst_bytes,
                    &kernel,
                    &kernel,
                    EdgeMode::Clamp,
                    Scalar::default(),
                    ThreadingPolicy::Adaptive,
                )
                .unwrap();
            })
        });

        c.bench_function("RGB gauss blur edge clamp approx", |b| {
            let mut dst_bytes =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Channels3);
            b.iter(|| {
                libblur::gaussian_blur(
                    &src_image,
                    &mut dst_bytes,
                    77 * 2 + 1,
                    (77f32 * 2f32 + 1f32) / 6f32,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    ConvolutionMode::FixedPoint,
                )
                .unwrap();
            })
        });

        let src = imread(
            &find_file(&"assets/test_image_1.jpg", false, false).unwrap(),
            IMREAD_COLOR,
        )
        .unwrap();

        c.bench_function("OpenCV RGB Gaussian", |b| {
            b.iter(|| {
                let mut dst = Mat::default();
                opencv::imgproc::gaussian_blur(
                    &src,
                    &mut dst,
                    Size::new(77 * 2 + 1, 77 * 2 + 1),
                    (77f64 * 2f64 + 1f64) / 6f64,
                    (77f64 * 2f64 + 1f64) / 6f64,
                    BORDER_DEFAULT,
                    AlgorithmHint::ALGO_HINT_ACCURATE,
                )
                .unwrap();
            })
        });
    }
    {
        let img = ImageReader::open("assets/test_image_1.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let dimensions = img.dimensions();
        let width = dimensions.0 as usize;
        let height = dimensions.1 as usize;
        let src_bytes = img.as_bytes();
        let mut plane_1 = vec![0u8; width * height];
        let mut plane_2 = vec![0u8; width * height];
        let mut plane_3 = vec![0u8; width * height];

        split_channels_3(
            src_bytes,
            width,
            height,
            &mut plane_1,
            &mut plane_2,
            &mut plane_3,
        );

        let src_image =
            BlurImage::borrow(&plane_1, img.width(), img.height(), FastBlurChannels::Plane);

        c.bench_function("Plane Gauss Blur Clamp: 151", |b| {
            let mut dst_bytes =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Plane);
            b.iter(|| {
                libblur::gaussian_blur(
                    &src_image,
                    &mut dst_bytes,
                    151,
                    0.,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    ConvolutionMode::Exact,
                )
                .unwrap();
            })
        });

        c.bench_function("Plane Gauss Blur Clamp Approx: Rad 151", |b| {
            let mut dst_bytes =
                BlurImageMut::alloc(img.width(), img.height(), FastBlurChannels::Plane);
            b.iter(|| {
                libblur::gaussian_blur(
                    &src_image,
                    &mut dst_bytes,
                    77 * 2 + 1,
                    (77f32 * 2f32 + 1f32) / 6f32,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    ConvolutionMode::FixedPoint,
                )
                .unwrap();
            })
        });

        let src = imread(
            &find_file("assets/test_image_1.jpg", false, false).unwrap(),
            IMREAD_COLOR,
        )
        .unwrap();
        let mut planes = Vector::<Mat>::new();
        split(&src, &mut planes).unwrap();
        let source_plane = planes.get(0).unwrap();

        c.bench_function("OpenCV Plane Gaussian: Rad 151", |b| {
            b.iter(|| {
                let mut dst = Mat::default();
                opencv::imgproc::gaussian_blur(
                    &source_plane,
                    &mut dst,
                    Size::new(77 * 2 + 1, 77 * 2 + 1),
                    (77f64 * 2f64 + 1f64) / 6f64,
                    (77f64 * 2f64 + 1f64) / 6f64,
                    BORDER_DEFAULT,
                    AlgorithmHint::ALGO_HINT_ACCURATE,
                )
                .unwrap();
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
