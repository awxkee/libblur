use crate::clahe::ClaheGridSize;
use yuvutils_rs::{rgb_to_yuv444, rgb_to_yuv_nv24, rgba_to_yuv444, rgba_to_yuv_nv24, yuv444_to_rgb, yuv444_to_rgba, yuv_nv24_to_rgb, yuv_nv24_to_rgba, YuvRange, YuvStandardMatrix};

use crate::equalize_hist::EqualizeHistogramChannels::{Channels3, Channels4};
use crate::histogram::{
    make_histogram_1_channel, make_histogram_channel_1_region, make_histogram_channel_3_region,
};
use crate::hsl::Rgb;
use crate::hsv::Hsv;
use crate::luv::Luv;

#[repr(u8)]
#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub enum EqualizeHistogramChannels {
    Channels3 = 0,
    Channels4 = 1,
}

impl From<u8> for EqualizeHistogramChannels {
    fn from(value: u8) -> Self {
        return match value {
            0 => Channels3,
            1 => Channels4,
            _ => panic!("Not implemented"),
        };
    }
}

pub fn minmax(slice: &[u64]) -> (u64, u64) {
    let mut min_value = u64::MAX;
    let mut max_value = u64::MIN;
    let mut i = 0usize;
    let count = slice.len();

    while i < count {
        let value = slice[i];
        if value < min_value {
            min_value = value;
        }
        if value > max_value {
            max_value = value;
        }
        i += 1;
    }
    return (min_value, max_value);
}

pub fn minmax_index(slice: &[u64]) -> (usize, usize) {
    let mut min_value = u64::MAX;
    let mut max_value = u64::MIN;
    let mut min_index = 0usize;
    let mut max_index = 0usize;
    let mut i = 0usize;
    let count = slice.len();

    while i < count {
        let value = slice[i];
        if value < min_value {
            min_value = value;
            min_index = i;
        }
        if value > max_value {
            max_value = value;
            max_index = i;
        }
        i += 1;
    }
    return (min_index, max_index);
}

pub fn cdf(arr: &mut [u64]) {
    let mut sum: u64 = 0u64;
    for i in 0..arr.len() {
        sum += arr[i];
        arr[i] = sum;
    }
}

#[allow(dead_code)]
pub fn equalize_histogram_squares<const CHANNELS: u8>(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    clahe_grid_size: ClaheGridSize,
) {
    if clahe_grid_size.w == 0 || clahe_grid_size.h == 0 {
        panic!("zero sized grid is not accepted");
    }
    let horizontal_tile_size = width / clahe_grid_size.w;
    let vertical_tile_size = height / clahe_grid_size.h;
    let tiles_horizontal = width / horizontal_tile_size;
    let tiles_vertical = height / vertical_tile_size;
    for w in 0..tiles_horizontal {
        for h in 0..tiles_vertical {
            let start_x = w * horizontal_tile_size;
            let start_y = h * vertical_tile_size;
            let mut end_x = (w + 1) * horizontal_tile_size;
            if w + 1 == tiles_horizontal {
                end_x = width;
            }
            let mut end_y = (h + 1) * vertical_tile_size;
            if h + 1 == tiles_vertical {
                end_y = height;
            }
            equalize_histogram_region_hsv::<CHANNELS>(
                in_place, stride, start_x, end_x, start_y, end_y, width, height,
            );
        }
    }
}

#[allow(dead_code)]
pub fn equalize_histogram<const CHANNELS: u8>(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
) {
    equalize_histogram_region_yuv::<CHANNELS>(in_place, stride, width, height);
    // equalize_histogram_region_hsv::<CHANNELS>(in_place, stride, 0, width, 0, height, width, height);
    // equalize_histogram_region_luv::<CHANNELS>(in_place, stride, 0, width, 0, height, width, height);
}

// #[allow(dead_code)]
// fn equalize_histogram_region_yuv<const CHANNELS: u8>(
//     in_place: &mut [u8],
//     stride: u32,
//     width: u32,
//     height: u32,
// ) {
//     let h_channels: EqualizeHistogramChannels = CHANNELS.into();
//     let channels: usize;
//     match h_channels {
//         Channels3 => {
//             channels = 3;
//         }
//         Channels4 => {
//             channels = 4;
//         }
//     }
//
//     let mut y_plane: Vec<u8> = Vec::new();
//     y_plane.resize(width as usize * height as usize, 0u8);
//
//     let mut uv_plane: Vec<u8> = Vec::new();
//     let uv_stride = width as usize * 2usize;
//     uv_plane.resize(uv_stride * height as usize, 0u8);
//
//     match h_channels {
//         Channels3 => {
//             rgb_to_yuv_nv24(
//                 &mut y_plane,
//                 width,
//                 &mut uv_plane,
//                 uv_stride as u32,
//                 in_place,
//                 stride,
//                 width,
//                 height,
//                 YuvRange::Full,
//                 YuvStandardMatrix::Bt709,
//             );
//         }
//         Channels4 => {
//             rgba_to_yuv_nv24(
//                 &mut y_plane,
//                 width,
//                 &mut uv_plane,
//                 uv_stride as u32,
//                 in_place,
//                 stride,
//                 width,
//                 height,
//                 YuvRange::Full,
//                 YuvStandardMatrix::Bt709,
//             );
//         }
//     }
//
//     let histogram = make_histogram_1_channel(&y_plane, width, width, height);
//     let mut bins = histogram.bins;
//
//     cdf(&mut bins);
//
//     let pixels_count = width * height;
//
//     let (min_bin, _) = minmax(&bins);
//
//     let distance_r = 1f64 / (pixels_count as f64 - min_bin as f64);
//
//     for i in 0..256usize {
//         if distance_r != 0f64 {
//             bins[i] = (255f64 * (bins[i] as f64 - min_bin as f64) * distance_r)
//                 .round()
//                 .min(255f64) as u64;
//         }
//     }
//
//     // let mut y_shift = 0usize;
//     // for _ in 0usize..height as usize {
//     //     for x in 0usize..width as usize {
//     //         y_plane[y_shift + x] = bins[y_plane[y_shift + x] as usize] as u8;
//     //     }
//     //     y_shift += width as usize;
//     // }
//
//     match h_channels {
//         Channels3 => {
//             yuv_nv24_to_rgb(
//                 &y_plane,
//                 width,
//                 &uv_plane,
//                 uv_stride as u32,
//                 in_place,
//                 stride,
//                 width,
//                 height,
//                 YuvRange::Full,
//                 YuvStandardMatrix::Bt709,
//             );
//         }
//         Channels4 => {
//             yuv_nv24_to_rgba(
//                 &y_plane,
//                 width,
//                 &uv_plane,
//                 uv_stride as u32,
//                 in_place,
//                 stride,
//                 width,
//                 height,
//                 YuvRange::Full,
//                 YuvStandardMatrix::Bt709,
//             );
//         }
//     }
// }

#[allow(dead_code)]
fn equalize_histogram_region_yuv<const CHANNELS: u8>(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
) {
    let h_channels: EqualizeHistogramChannels = CHANNELS.into();
    let channels: usize;
    match h_channels {
        Channels3 => {
            channels = 3;
        }
        Channels4 => {
            channels = 4;
        }
    }

    let mut y_plane: Vec<u8> = Vec::new();
    y_plane.resize(width as usize * height as usize, 0u8);

    let mut u_plane: Vec<u8> = Vec::new();
    u_plane.resize(width as usize * height as usize, 0u8);

    let mut v_plane: Vec<u8> = Vec::new();
    v_plane.resize(width as usize * height as usize, 0u8);

    match h_channels {
        Channels3 => {
            rgb_to_yuv444(
                &mut y_plane,
                width,
                &mut u_plane,
                width,
                &mut v_plane,
                width,
                in_place,
                stride,
                width,
                height,
                YuvRange::Full,
                YuvStandardMatrix::Bt709,
            );
        }
        Channels4 => {
            rgba_to_yuv444(
                &mut y_plane,
                width,
                &mut u_plane,
                width,
                &mut v_plane,
                width,
                in_place,
                stride,
                width,
                height,
                YuvRange::Full,
                YuvStandardMatrix::Bt709,
            );
        }
    }

    let histogram = make_histogram_1_channel(&y_plane, width, width, height);
    let mut bins = histogram.bins;

    cdf(&mut bins);

    let pixels_count = width * height;

    let (min_bin, _) = minmax(&bins);

    let distance_r = 1f64 / (pixels_count as f64 - min_bin as f64);

    for i in 0..256usize {
        if distance_r != 0f64 {
            bins[i] = (255f64 * (bins[i] as f64 - min_bin as f64) * distance_r)
                .round()
                .min(255f64) as u64;
        }
    }

    let mut y_shift = 0usize;
    for _ in 0usize..height as usize {
        for x in 0usize..width as usize {
            y_plane[y_shift + x] = bins[y_plane[y_shift + x] as usize] as u8;
        }
        y_shift += width as usize;
    }

    match h_channels {
        Channels3 => {
            yuv444_to_rgb(
                &y_plane,
                width,
                &u_plane,
                width,
                &v_plane,
                width,
                in_place,
                stride,
                width,
                height,
                YuvRange::Full,
                YuvStandardMatrix::Bt709,
            );
        }
        Channels4 => {
            yuv444_to_rgba(
                &y_plane,
                width,
                &u_plane,
                width,
                &v_plane,
                width,
                in_place,
                stride,
                width,
                height,
                YuvRange::Full,
                YuvStandardMatrix::Bt709,
            );
        }
    }
}

#[allow(dead_code)]
fn equalize_histogram_region_hsv<const CHANNELS: u8>(
    in_place: &mut [u8],
    stride: u32,
    start_x: u32,
    end_x: u32,
    start_y: u32,
    end_y: u32,
    width: u32,
    height: u32,
) {
    let h_channels: EqualizeHistogramChannels = CHANNELS.into();
    let channels: usize;
    match h_channels {
        Channels3 => {
            channels = 3;
        }
        Channels4 => {
            channels = 4;
        }
    }

    let hist_width = end_x as usize - start_x as usize;
    let hist_height = end_y as usize - start_y as usize;

    let mut hsl_image: Vec<f32> = Vec::new();
    hsl_image.resize(hist_width * 3usize * hist_height, 0f32);

    let mut y_shift = stride as usize * start_y as usize;
    let mut hsl_shift = 0usize;
    for _ in start_y as usize..(end_y as usize) {
        for (x, j) in (start_x as usize..width as usize).zip(0..hist_width) {
            let px = x * channels;
            let h_px = j * 3usize;

            let rgb = Rgb::<u8>::new(
                in_place[y_shift + px],
                in_place[y_shift + px + 1],
                in_place[y_shift + px + 2],
            );
            let hsv = rgb.to_hsv();
            hsl_image[hsl_shift + h_px] = hsv.get_hue();
            hsl_image[hsl_shift + h_px + 1] = hsv.get_saturation();
            hsl_image[hsl_shift + h_px + 2] = hsv.get_value();
        }
        y_shift += stride as usize;
        hsl_shift += hist_width * 3usize;
    }

    let histogram = make_histogram_channel_3_region(
        &hsl_image,
        hist_width as u32 * 3u32,
        start_x,
        end_x,
        start_y,
        end_y,
    );
    let mut bins = histogram.bins;

    // clip_hist(&mut bins);

    cdf(&mut bins);

    let pixels_count = hist_width * hist_height;

    let (min_bin, _) = minmax(&bins);

    let distance_r = 1f64 / (pixels_count as f64 - min_bin as f64);

    let mut f_bins: [f32; 101] = [0f32; 101];

    for i in 0..101usize {
        if distance_r != 0f64 {
            f_bins[i] = (100f64 * (bins[i] as f64 - min_bin as f64) * distance_r)
                .round()
                .min(100f64) as f32
                / 100f32;
        }
    }

    let mut hsl_shift = 0usize;
    let mut y_shift = stride as usize * start_y as usize;
    for _ in start_y as usize..end_y as usize {
        for (x, j) in (start_x as usize..width as usize).zip(0..hist_width) {
            let px = x * channels;
            let h_px = j * 3;

            let vl = (hsl_image[hsl_shift + h_px + 2] * 100f32)
                .round()
                .min(100f32)
                .max(0f32) as usize;
            hsl_image[hsl_shift + h_px + 2] = f_bins[vl];

            let hsv = Hsv::from_components(
                hsl_image[hsl_shift + h_px],
                hsl_image[hsl_shift + h_px + 1],
                hsl_image[hsl_shift + h_px + 2],
            );
            let rgb = hsv.to_rgb8();

            in_place[y_shift + px] = rgb.r;
            in_place[y_shift + px + 1] = rgb.g;
            in_place[y_shift + px + 2] = rgb.b;
        }
        y_shift += stride as usize;
        hsl_shift += hist_width * 3usize;
    }
}

#[allow(dead_code)]
fn equalize_histogram_region_luv<const CHANNELS: u8>(
    in_place: &mut [u8],
    stride: u32,
    start_x: u32,
    end_x: u32,
    start_y: u32,
    end_y: u32,
    width: u32,
    _: u32,
) {
    let h_channels: EqualizeHistogramChannels = CHANNELS.into();
    let channels: usize;
    match h_channels {
        Channels3 => {
            channels = 3;
        }
        Channels4 => {
            channels = 4;
        }
    }

    let hist_width = end_x as usize - start_x as usize;
    let hist_height = end_y as usize - start_y as usize;

    let mut hsl_image: Vec<f32> = Vec::new();
    hsl_image.resize(hist_width * 3usize * hist_height, 0f32);

    let mut y_shift = stride as usize * start_y as usize;
    let mut hsl_shift = 0usize;
    for _ in start_y as usize..(end_y as usize) {
        for (x, j) in (start_x as usize..width as usize).zip(0..hist_width) {
            let px = x * channels;
            let h_px = j * 3usize;

            let rgb = Rgb::<u8>::new(
                in_place[y_shift + px],
                in_place[y_shift + px + 1],
                in_place[y_shift + px + 2],
            );
            let luv = rgb.to_luv();
            hsl_image[hsl_shift + h_px] = luv.l;
            hsl_image[hsl_shift + h_px + 1] = luv.u;
            hsl_image[hsl_shift + h_px + 2] = luv.v;
        }
        y_shift += stride as usize;
        hsl_shift += hist_width * 3usize;
    }

    let histogram = make_histogram_channel_1_region(
        &hsl_image,
        hist_width as u32 * 3u32,
        start_x,
        end_x,
        start_y,
        end_y,
    );
    let mut bins = histogram.bins;

    // clip_hist(&mut bins);

    cdf(&mut bins);

    let pixels_count = hist_width * hist_height;

    let (min_bin, _) = minmax(&bins);

    let distance_r = 1f64 / (pixels_count as f64 - min_bin as f64);

    let mut f_bins: [f32; 101] = [0f32; 101];

    for i in 0..101usize {
        if distance_r != 0f64 {
            f_bins[i] = (100f64 * (bins[i] as f64 - min_bin as f64) * distance_r)
                .round()
                .min(100f64)
                .max(0f64) as f32;
        }
    }

    let mut hsl_shift = 0usize;
    let mut y_shift = stride as usize * start_y as usize;
    for _ in start_y as usize..end_y as usize {
        for (x, j) in (start_x as usize..width as usize).zip(0..hist_width) {
            let px = x * channels;
            let h_px = j * 3;

            let vl = hsl_image[hsl_shift + h_px] as usize;
            hsl_image[hsl_shift + h_px] = f_bins[vl];

            let hsv = Luv::new(
                hsl_image[hsl_shift + h_px],
                hsl_image[hsl_shift + h_px + 1],
                hsl_image[hsl_shift + h_px + 2],
            );
            let rgb = hsv.to_rgb();

            in_place[y_shift + px] = rgb.r;
            in_place[y_shift + px + 1] = rgb.g;
            in_place[y_shift + px + 2] = rgb.b;
        }
        y_shift += stride as usize;
        hsl_shift += hist_width * 3usize;
    }
}

fn clip_hist_2(bins: &mut [u64]) {
    let sums: u64 = bins.iter().sum();
    let mean: u64 = (sums as f64 / bins.len() as f64).round() as u64;
    let threshold_value: f64 = mean as f64 * 10f64;
    let clip_limit = mean + threshold_value as u64;
    let mut excess = 0u64;

    for i in 0..bins.len() {
        if bins[i] > clip_limit {
            excess += bins[i];
        }
    }

    let mean_excess = (excess as f64 / bins.len() as f64) as u64;

    for i in 0..bins.len() {
        if bins[i] > clip_limit {
            bins[i] = clip_limit + mean_excess;
        } else {
            bins[i] = bins[i] + mean_excess;
        }
    }
}

fn clip_hist(bins: &mut [u64]) {
    let sums: u64 = bins.iter().sum();
    let mean: u64 = (sums as f64 / bins.len() as f64).round() as u64;
    let threshold_value: f64 = mean as f64 * 1.5f64;
    let clip_limit = mean + threshold_value as u64;
    for i in 0..bins.len() {
        if bins[i] > clip_limit {
            bins[i] = clip_limit;
        }
    }
}

#[allow(dead_code)]
pub fn adjust_ppg_red_channel<const CHANNELS: u8>(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
) {
    let h_channels: EqualizeHistogramChannels = CHANNELS.into();
    let channels: usize;
    match h_channels {
        Channels3 => {
            channels = 3;
        }
        Channels4 => {
            channels = 4;
        }
    }

    let mut y_shift = 0usize;

    y_shift = 0usize;
    for _ in 0..height as usize {
        for x in 0..width as usize {
            let px = x * channels;

            in_place[y_shift + px] = ((in_place[y_shift + px] as u16 * 5) / 4).min(255) as u8;
            in_place[y_shift + px + 1] =
                ((in_place[y_shift + px + 1] as u16 * 4) / 5).min(255) as u8;
            in_place[y_shift + px + 2] =
                ((in_place[y_shift + px + 2] as u16 * 4) / 5).min(255) as u8;
        }
        y_shift += stride as usize;
    }
}
