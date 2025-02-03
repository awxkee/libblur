use criterion::{criterion_group, criterion_main, Criterion};
use image::{GenericImageView, ImageReader};
use libblur::{
    filter_1d_rgb_exact, get_gaussian_kernel_1d, get_sigma_size, EdgeMode, FastBlurChannels,
    GaussianPreciseLevel, ImageSize, Scalar, ThreadingPolicy,
};
use opencv::core::{find_file, split, Mat, Size, Vector, BORDER_DEFAULT};
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
    let img = ImageReader::open("assets/test_image_4.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let components = 4;
    let stride = dimensions.0 as usize * components;
    let src_bytes = img.as_bytes();

    c.bench_function("RGBA gauss blur kernel clip exact: 13", |b| {
        let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
        b.iter(|| {
            libblur::gaussian_blur(
                &src_bytes,
                &mut dst_bytes,
                dimensions.0,
                dimensions.1,
                13,
                0.,
                FastBlurChannels::Channels4,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
                GaussianPreciseLevel::EXACT,
            );
        })
    });

    c.bench_function("RGBA gauss blur clamp approx: 13", |b| {
        let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
        b.iter(|| {
            libblur::gaussian_blur(
                &src_bytes,
                &mut dst_bytes,
                dimensions.0,
                dimensions.1,
                13,
                0.,
                FastBlurChannels::Channels4,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
                GaussianPreciseLevel::INTEGRAL,
            );
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
            )
            .unwrap();
        })
    });

    c.bench_function("RGBA gauss blur edge clamp: rad 151", |b| {
        let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
        b.iter(|| {
            libblur::gaussian_blur(
                &src_bytes,
                &mut dst_bytes,
                dimensions.0,
                dimensions.1,
                77 * 2 + 1,
                (77f32 * 2f32 + 1f32) / 6f32,
                FastBlurChannels::Channels4,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
                GaussianPreciseLevel::EXACT,
            );
        })
    });

    c.bench_function("RGBA gauss blur edge clamp approx: rad 151", |b| {
        let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
        b.iter(|| {
            libblur::gaussian_blur(
                &src_bytes,
                &mut dst_bytes,
                dimensions.0,
                dimensions.1,
                77 * 2 + 1,
                (77f32 * 2f32 + 1f32) / 6f32,
                FastBlurChannels::Channels4,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
                GaussianPreciseLevel::INTEGRAL,
            );
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
            )
            .unwrap();
        })
    });

    {
        let img = ImageReader::open("assets/test_image_1.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let dimensions = img.dimensions();
        let components = 3;
        let stride = dimensions.0 as usize * components;
        let src_bytes = img.as_bytes();
        c.bench_function("RGB gauss blur edge clamp: 151", |b| {
            let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
            b.iter(|| {
                libblur::gaussian_blur(
                    src_bytes,
                    &mut dst_bytes,
                    dimensions.0,
                    dimensions.1,
                    77 * 2 + 1,
                    (77f32 * 2f32 + 1f32) / 6f32,
                    FastBlurChannels::Channels3,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    GaussianPreciseLevel::EXACT,
                );
            })
        });

        c.bench_function("RGB gauss blur edge clamp: 21", |b| {
            let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
            b.iter(|| {
                libblur::gaussian_blur(
                    src_bytes,
                    &mut dst_bytes,
                    dimensions.0,
                    dimensions.1,
                    21,
                    0.,
                    FastBlurChannels::Channels3,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    GaussianPreciseLevel::EXACT,
                );
            })
        });

        c.bench_function("Filter 2D Rgb Blur Clamp: 25", |b| {
            let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
            let kernel = get_gaussian_kernel_1d(25, get_sigma_size(25));
            b.iter(|| {
                filter_1d_rgb_exact(
                    src_bytes,
                    &mut dst_bytes,
                    ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
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
            let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
            let kernel = get_gaussian_kernel_1d(151, get_sigma_size(151));
            b.iter(|| {
                filter_1d_rgb_exact(
                    src_bytes,
                    &mut dst_bytes,
                    ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
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
            let mut dst_bytes: Vec<u8> = vec![0u8; dimensions.1 as usize * stride];
            b.iter(|| {
                libblur::gaussian_blur(
                    &src_bytes,
                    &mut dst_bytes,
                    dimensions.0,
                    dimensions.1,
                    77 * 2 + 1,
                    (77f32 * 2f32 + 1f32) / 6f32,
                    FastBlurChannels::Channels3,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    GaussianPreciseLevel::INTEGRAL,
                );
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

        c.bench_function("Plane Gauss Blur Clamp: 151", |b| {
            let mut dst_plane_1 = vec![0u8; width * height];
            b.iter(|| {
                libblur::gaussian_blur(
                    &plane_1,
                    &mut dst_plane_1,
                    dimensions.0,
                    dimensions.1,
                    151,
                    0.,
                    FastBlurChannels::Plane,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    GaussianPreciseLevel::EXACT,
                );
            })
        });

        c.bench_function("Plane Gauss Blur Clamp Approx: Rad 151", |b| {
            let mut dst_plane_1 = vec![0u8; width * height];
            b.iter(|| {
                libblur::gaussian_blur(
                    &plane_1,
                    &mut dst_plane_1,
                    dimensions.0,
                    dimensions.1,
                    77 * 2 + 1,
                    (77f32 * 2f32 + 1f32) / 6f32,
                    FastBlurChannels::Plane,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                    GaussianPreciseLevel::INTEGRAL,
                );
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
                )
                .unwrap();
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
