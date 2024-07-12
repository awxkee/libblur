use criterion::{criterion_group, criterion_main, Criterion};
use image::io::Reader as ImageReader;
use image::GenericImageView;
use libblur::{EdgeMode, FastBlurChannels, ThreadingPolicy};
use opencv::core::{
    find_file, mean, split, Mat, MatTraitConst, MatTraitConstManual, Size, Vector, BORDER_DEFAULT,
};
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
                77 * 2 + 1,
                (77f32 * 2f32 + 1f32) / 6f32,
                FastBlurChannels::Channels4,
                EdgeMode::KernelClip,
                ThreadingPolicy::Adaptive,
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
                77 * 2 + 1,
                (77f32 * 2f32 + 1f32) / 6f32,
                FastBlurChannels::Channels4,
                EdgeMode::Clamp,
                ThreadingPolicy::Adaptive,
            );
        })
    });

    let src = imread(
        &find_file(&"assets/test_image_4.png", false, false).unwrap(),
        IMREAD_COLOR,
    )
    .unwrap();
    let mut planes = Vector::<Mat>::new();
    // split(&src, &mut planes).unwrap();

    c.bench_function("OpenCV RGBA Gaussian", |b| {
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
        c.bench_function("RGB gauss blur edge clamp", |b| {
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
                    77 * 2 + 1,
                    (77f32 * 2f32 + 1f32) / 6f32,
                    FastBlurChannels::Channels3,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
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
        let components = 3;
        let width = dimensions.0 as usize;
        let height = dimensions.1 as usize;
        let src_bytes = img.as_bytes();
        let mut plane_1 = vec![0u8; width * height];
        let mut plane_2 = vec![0u8; width * height];
        let mut plane_3 = vec![0u8; width * height];

        split_channels_3(&src_bytes, width, height, &mut plane_1, &mut plane_2, &mut plane_3);

        c.bench_function("Plane Gauss Blur Clamp", |b| {
            b.iter(|| {
                let mut dst_plane_1 = vec![0u8; width * height];
                let stride = width;
                libblur::gaussian_blur(
                    &plane_1,
                    stride as u32,
                    &mut dst_plane_1,
                    stride as u32,
                    dimensions.0,
                    dimensions.1,
                    77 * 2 + 1,
                    (77f32 * 2f32 + 1f32) / 6f32,
                    FastBlurChannels::Plane,
                    EdgeMode::Clamp,
                    ThreadingPolicy::Adaptive,
                );
            })
        });

        let src = imread(
            &find_file(&"assets/test_image_1.jpg", false, false).unwrap(),
            IMREAD_COLOR,
        )
            .unwrap();
        let mut planes = Vector::<Mat>::new();
        split(&src, &mut planes).unwrap();
        let source_plane = planes.get(0).unwrap();

        c.bench_function("OpenCV Plane Gaussian", |b| {
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
