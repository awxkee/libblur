use yuvutils_rs::{rgb_to_yuv444, rgba_to_yuv444, yuv444_to_rgb, yuv444_with_alpha_to_rgba, YuvRange, YuvStandardMatrix};
use crate::clahe::AheImplementation::{Ahe, Clahe};
use crate::equalize_hist::EqualizeHistogramChannels::{Channels3, Channels4};
use crate::equalize_hist::{cdf, minmax, EqualizeHistogramChannels};
use crate::histogram::{make_histogram_region, ImageSingleHistogram, make_histogram_region_f32, ImageSingle100Histogram};
use crate::hsl::Rgb;
use crate::lab::Lab;
use crate::luv::Luv;

enum AheImplementation {
    Ahe = 1,
    Clahe = 2,
}

impl From<u8> for AheImplementation {
    fn from(value: u8) -> Self {
        match value {
            1 => Ahe,
            2 => Clahe,
            _ => {
                panic!("{} not implemented in AHE", value)
            }
        }
    }
}

fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a * (1f32 - f) + (b * f)
}

fn blerp(c00: f32, c10: f32, c01: f32, c11: f32, tx: f32, ty: f32) -> f32 {
    lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty)
}

pub fn clahe_yuv_rgba(in_place: &mut [u8], stride: u32, width: u32, height: u32, threshold: f32, grid_size: ClaheGridSize) {
    clahe_yuv_impl::<{ Channels4 as u8 }, { Clahe as u8 }>(
        in_place, stride, width, height, threshold, grid_size,
    );
}

pub fn clahe_yuv_rgb(in_place: &mut [u8], stride: u32, width: u32, height: u32, threshold: f32, grid_size: ClaheGridSize) {
    clahe_yuv_impl::<{ Channels3 as u8 }, { Clahe as u8 }>(
        in_place, stride, width, height, threshold, grid_size,
    );
}

pub fn ahe_yuv_rgb(in_place: &mut [u8], stride: u32, width: u32, height: u32, threshold: f32, grid_size: ClaheGridSize) {
    clahe_yuv_impl::<{ Channels3 as u8 }, { Ahe as u8 }>(
        in_place, stride, width, height, threshold, grid_size,
    );
}

pub struct ClaheGridSize {
    pub w: u32,
    pub h: u32,
}

impl ClaheGridSize {
    pub fn new(w: u32, h: u32) -> ClaheGridSize {
        ClaheGridSize { w, h }
    }
}

#[allow(dead_code)]
fn clahe_yuv_impl<const CHANNELS: u8, const IMPLEMENTATION: u8>(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    threshold: f32,
    clahe_grid_size: ClaheGridSize,
) {
    if clahe_grid_size.w == 0 || clahe_grid_size.h == 0 {
        panic!("zero sized grid is not accepted");
    }
    let h_channels: EqualizeHistogramChannels = CHANNELS.into();
    let implementation: AheImplementation = IMPLEMENTATION.into();
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

    let mut a_plane: Vec<u8> = Vec::new();
    if h_channels == Channels4 {
        a_plane.resize(width as usize * height as usize, 0u8);
    }

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
            let mut a_shift = 0usize;
            let mut y_shift = 0usize;
            for _ in 0usize..height as usize {
                for x in 0usize..width as usize {
                    a_plane[a_shift + x] = in_place[y_shift + x * 4 + 3];
                }
                y_shift += stride as usize;
                a_shift += width as usize;
            }
        }
    }

    let mut histograms: Vec<Vec<ImageSingleHistogram>> = vec![];

    let horizontal_tile_size = width / clahe_grid_size.w;
    let vertical_tile_size = height / clahe_grid_size.h;
    let tiles_horizontal = width / horizontal_tile_size;
    let tiles_vertical = height / vertical_tile_size;
    for h in 0..tiles_vertical {
        let mut regions_hist: Vec<ImageSingleHistogram> = vec![];
        for w in 0..tiles_horizontal {
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

            let mut region_hist =
                make_histogram_region(&y_plane, width, start_x, end_x, start_y, end_y);

            let mut bins = region_hist.bins;
            match implementation {
                Clahe => {
                    clip_hist_clahe(&mut bins, threshold);
                }
                _ => {}
            }
            cdf(&mut bins);

            let (min_bin, _) = minmax(&bins);

            let distance_r =
                1f64 / ((end_y - start_y) as f64 * (end_x - start_x) as f64 - min_bin as f64);

            for i in 0..256usize {
                if distance_r != 0f64 {
                    bins[i] = (255f64 * (bins[i] as f64 - min_bin as f64) * distance_r)
                        .round()
                        .min(255f64) as u64;
                }
            }

            region_hist.bins = bins;

            regions_hist.push(region_hist);
        }
        histograms.push(regions_hist);
    }

    for y in 0usize..height as usize {
        for x in 0usize..width as usize {
            let c_x_f =
                (x as f32 - horizontal_tile_size as f32 / 2f32) / horizontal_tile_size as f32;
            let r_y_f = (y as f32 - vertical_tile_size as f32 / 2f32) / vertical_tile_size as f32;

            let x1 = (x as f32 - ((c_x_f as i64) as f32 + 0.5f32) * horizontal_tile_size as f32)
                / horizontal_tile_size as f32;
            let y1 = (y as f32 - ((r_y_f as i64) as f32 + 0.5f32) * vertical_tile_size as f32)
                / vertical_tile_size as f32;

            let value = y_plane[width as usize * y + x] as usize;

            let r_y = r_y_f.max(0f32) as i64;
            let c_x = c_x_f.max(0f32) as i64;

            let r = (r_y as usize).min(tiles_vertical as usize - 1usize);
            let c = (c_x as usize).min(tiles_horizontal as usize - 1usize);
            let bin1 = histograms[r][c].bins[value] as f32;
            let bin2 =
                histograms[r][(c + 1).min(tiles_horizontal as usize - 1usize)].bins[value] as f32;
            let bin3 =
                histograms[(r + 1).min(tiles_vertical as usize - 1usize)][c].bins[value] as f32;
            let bin4 = histograms[(r + 1).min(tiles_vertical as usize - 1usize)]
                [(c + 1).min(tiles_horizontal as usize - 1usize)]
                .bins[value] as f32;
            let interpolated = blerp(bin1, bin2, bin3, bin4, x1, y1);
            y_plane[width as usize * y + x] = interpolated as u8;
        }
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
            yuv444_with_alpha_to_rgba(
                &y_plane,
                width,
                &u_plane,
                width,
                &v_plane,
                width,
                &a_plane,
                width,
                in_place,
                stride,
                width,
                height,
                YuvRange::Full,
                YuvStandardMatrix::Bt709,
                false,
            );
        }
    }
}

pub fn clahe_luv_rgb(in_place: &mut [u8], stride: u32, width: u32, height: u32, threshold: f32, grid_size: ClaheGridSize) {
    clahe_luv_impl::<{ Channels3 as u8 }, { Clahe as u8 }>(
        in_place, stride, width, height, threshold, grid_size,
    );
}

pub fn ahe_luv_rgb(in_place: &mut [u8], stride: u32, width: u32, height: u32, threshold: f32, grid_size: ClaheGridSize) {
    clahe_luv_impl::<{ Channels3 as u8 }, { Ahe as u8 }>(
        in_place, stride, width, height, threshold, grid_size,
    );
}

#[allow(dead_code)]
fn clahe_luv_impl<const CHANNELS: u8, const IMPLEMENTATION: u8>(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    threshold: f32,
    clahe_grid_size: ClaheGridSize,
) {
    if clahe_grid_size.w == 0 || clahe_grid_size.h == 0 {
        panic!("zero sized grid is not accepted");
    }
    let h_channels: EqualizeHistogramChannels = CHANNELS.into();
    let implementation: AheImplementation = IMPLEMENTATION.into();
    let channels: usize;
    match h_channels {
        Channels3 => {
            channels = 3;
        }
        Channels4 => {
            channels = 4;
        }
    }

    let mut luv_image: Vec<f32> = Vec::new();
    let luv_stride = width as usize * 3usize;
    luv_image.resize(luv_stride * height as usize, 0f32);

    let mut y_shift = 0usize;
    let mut luv_shift = 0usize;
    for _ in 0usize..(height as usize) {
        for (x, j) in (0usize..width as usize).zip(0usize..width as usize) {
            let px = x * channels;
            let h_px = j * 3usize;

            let rgb = Rgb::<u8>::new(
                in_place[y_shift + px],
                in_place[y_shift + px + 1],
                in_place[y_shift + px + 2],
            );
            let luv = rgb.to_luv();
            luv_image[luv_shift + h_px] = luv.l;
            luv_image[luv_shift + h_px + 1] = luv.u;
            luv_image[luv_shift + h_px + 2] = luv.v;
        }
        y_shift += stride as usize;
        luv_shift += luv_stride;
    }

    let mut histograms: Vec<Vec<ImageSingle100Histogram>> = vec![];

    let horizontal_tile_size = width / clahe_grid_size.w;
    let vertical_tile_size = height / clahe_grid_size.h;
    let tiles_horizontal = width / horizontal_tile_size;
    let tiles_vertical = height / vertical_tile_size;
    for h in 0..tiles_vertical {
        let mut regions_hist: Vec<ImageSingle100Histogram> = vec![];
        for w in 0..tiles_horizontal {
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

            let mut region_hist =
                make_histogram_region_f32(&luv_image, luv_stride as u32, start_x, end_x, start_y, end_y);

            let mut bins = region_hist.bins;
            match implementation {
                Clahe => {
                    clip_hist_clahe(&mut bins, threshold);
                }
                _ => {}
            }
            cdf(&mut bins);

            let (min_bin, _) = minmax(&bins);

            let distance_r =
                1f64 / ((end_y - start_y) as f64 * (end_x - start_x) as f64 - min_bin as f64);

            for i in 0..100usize {
                if distance_r != 0f64 {
                    bins[i] = (100f64 * (bins[i] as f64 - min_bin as f64) * distance_r)
                        .round()
                        .min(100f64)
                        .max(0f64) as u64;
                }
            }

            region_hist.bins = bins;

            regions_hist.push(region_hist);
        }
        histograms.push(regions_hist);
    }

    for y in 0usize..height as usize {
        for x in 0usize..width as usize {
            let c_x_f =
                (x as f32 - horizontal_tile_size as f32 / 2f32) / horizontal_tile_size as f32;
            let r_y_f = (y as f32 - vertical_tile_size as f32 / 2f32) / vertical_tile_size as f32;

            let x1 = (x as f32 - ((c_x_f as i64) as f32 + 0.5f32) * horizontal_tile_size as f32)
                / horizontal_tile_size as f32;
            let y1 = (y as f32 - ((r_y_f as i64) as f32 + 0.5f32) * vertical_tile_size as f32)
                / vertical_tile_size as f32;

            let px = x * 3;

            let value = luv_image[luv_stride * y + px].min(100f32).max(0f32) as usize;

            let r_y = r_y_f.max(0f32) as i64;
            let c_x = c_x_f.max(0f32) as i64;

            let r = (r_y as usize).min(tiles_vertical as usize - 1usize);
            let c = (c_x as usize).min(tiles_horizontal as usize - 1usize);
            let bin1 = histograms[r][c].bins[value] as f32;
            let bin2 =
                histograms[r][(c + 1).min(tiles_horizontal as usize - 1usize)].bins[value] as f32;
            let bin3 =
                histograms[(r + 1).min(tiles_vertical as usize - 1usize)][c].bins[value] as f32;
            let bin4 = histograms[(r + 1).min(tiles_vertical as usize - 1usize)]
                [(c + 1).min(tiles_horizontal as usize - 1usize)]
                .bins[value] as f32;
            let interpolated = blerp(bin1, bin2, bin3, bin4, x1, y1);
            luv_image[luv_stride * y + px] = interpolated.min(100f32).max(0f32);
        }
    }

    let mut luv_shift = 0usize;
    let mut y_shift = 0usize;
    for _ in 0usize..height as usize {
        for (x, j) in (0usize..width as usize).zip(0usize..width as usize) {
            let px = x * channels;
            let h_px = j * 3;

            let luv = Luv::new(
                luv_image[luv_shift + h_px],
                luv_image[luv_shift + h_px + 1],
                luv_image[luv_shift + h_px + 2],
            );

            let rgb = luv.to_rgb();

            in_place[y_shift + px] = rgb.r;
            in_place[y_shift + px + 1] = rgb.g;
            in_place[y_shift + px + 2] = rgb.b;
        }
        y_shift += stride as usize;
        luv_shift += luv_stride;
    }
}

pub fn clahe_lab_rgb(in_place: &mut [u8], stride: u32, width: u32, height: u32, threshold: f32, grid_size: ClaheGridSize) {
    clahe_lab_impl::<{ Channels3 as u8 }, { Clahe as u8 }>(
        in_place, stride, width, height, threshold, grid_size,
    );
}

pub fn ahe_lab_rgb(in_place: &mut [u8], stride: u32, width: u32, height: u32, threshold: f32, grid_size: ClaheGridSize) {
    clahe_lab_impl::<{ Channels3 as u8 }, { Ahe as u8 }>(
        in_place, stride, width, height, threshold, grid_size,
    );
}

#[allow(dead_code)]
fn clahe_lab_impl<const CHANNELS: u8, const IMPLEMENTATION: u8>(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    threshold: f32,
    clahe_grid_size: ClaheGridSize,
) {
    if clahe_grid_size.w == 0 || clahe_grid_size.h == 0 {
        panic!("zero sized grid is not accepted");
    }
    let h_channels: EqualizeHistogramChannels = CHANNELS.into();
    let implementation: AheImplementation = IMPLEMENTATION.into();
    let channels: usize;
    match h_channels {
        Channels3 => {
            channels = 3;
        }
        Channels4 => {
            channels = 4;
        }
    }

    let mut luv_image: Vec<f32> = Vec::new();
    let luv_stride = width as usize * 3usize;
    luv_image.resize(luv_stride * height as usize, 0f32);

    let mut y_shift = 0usize;
    let mut luv_shift = 0usize;
    for _ in 0usize..(height as usize) {
        for (x, j) in (0usize..width as usize).zip(0usize..width as usize) {
            let px = x * channels;
            let h_px = j * 3usize;

            let rgb = Rgb::<u8>::new(
                in_place[y_shift + px],
                in_place[y_shift + px + 1],
                in_place[y_shift + px + 2],
            );
            let luv = rgb.to_lab();
            luv_image[luv_shift + h_px] = luv.l;
            luv_image[luv_shift + h_px + 1] = luv.a;
            luv_image[luv_shift + h_px + 2] = luv.b;
        }
        y_shift += stride as usize;
        luv_shift += luv_stride;
    }

    let mut histograms: Vec<Vec<ImageSingle100Histogram>> = vec![];

    let horizontal_tile_size = width / clahe_grid_size.w;
    let vertical_tile_size = height / clahe_grid_size.h;
    let tiles_horizontal = width / horizontal_tile_size;
    let tiles_vertical = height / vertical_tile_size;
    for h in 0..tiles_vertical {
        let mut regions_hist: Vec<ImageSingle100Histogram> = vec![];
        for w in 0..tiles_horizontal {
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

            let mut region_hist =
                make_histogram_region_f32(&luv_image, luv_stride as u32, start_x, end_x, start_y, end_y);

            let mut bins = region_hist.bins;
            match implementation {
                Clahe => {
                    clip_hist_clahe(&mut bins, threshold);
                }
                _ => {}
            }
            cdf(&mut bins);

            let (min_bin, _) = minmax(&bins);

            let distance_r =
                1f64 / ((end_y - start_y) as f64 * (end_x - start_x) as f64 - min_bin as f64);

            for i in 0..100usize {
                if distance_r != 0f64 {
                    bins[i] = (100f64 * (bins[i] as f64 - min_bin as f64) * distance_r)
                        .round()
                        .min(100f64)
                        .max(0f64) as u64;
                }
            }

            region_hist.bins = bins;

            regions_hist.push(region_hist);
        }
        histograms.push(regions_hist);
    }

    for y in 0usize..height as usize {
        for x in 0usize..width as usize {
            let c_x_f =
                (x as f32 - horizontal_tile_size as f32 / 2f32) / horizontal_tile_size as f32;
            let r_y_f = (y as f32 - vertical_tile_size as f32 / 2f32) / vertical_tile_size as f32;

            let x1 = (x as f32 - ((c_x_f as i64) as f32 + 0.5f32) * horizontal_tile_size as f32)
                / horizontal_tile_size as f32;
            let y1 = (y as f32 - ((r_y_f as i64) as f32 + 0.5f32) * vertical_tile_size as f32)
                / vertical_tile_size as f32;

            let px = x * 3;

            let value = luv_image[luv_stride * y + px].min(100f32).max(0f32) as usize;

            let r_y = r_y_f.max(0f32) as i64;
            let c_x = c_x_f.max(0f32) as i64;

            let r = (r_y as usize).min(tiles_vertical as usize - 1usize);
            let c = (c_x as usize).min(tiles_horizontal as usize - 1usize);
            let bin1 = histograms[r][c].bins[value] as f32;
            let bin2 =
                histograms[r][(c + 1).min(tiles_horizontal as usize - 1usize)].bins[value] as f32;
            let bin3 =
                histograms[(r + 1).min(tiles_vertical as usize - 1usize)][c].bins[value] as f32;
            let bin4 = histograms[(r + 1).min(tiles_vertical as usize - 1usize)]
                [(c + 1).min(tiles_horizontal as usize - 1usize)]
                .bins[value] as f32;
            let interpolated = blerp(bin1, bin2, bin3, bin4, x1, y1);
            luv_image[luv_stride * y + px] = interpolated.min(100f32).max(0f32);
        }
    }

    let mut luv_shift = 0usize;
    let mut y_shift = 0usize;
    for _ in 0usize..height as usize {
        for (x, j) in (0usize..width as usize).zip(0usize..width as usize) {
            let px = x * channels;
            let h_px = j * 3;

            let hsv = Lab::new(
                luv_image[luv_shift + h_px],
                luv_image[luv_shift + h_px + 1],
                luv_image[luv_shift + h_px + 2],
            );
            let rgb = hsv.to_rgb();

            in_place[y_shift + px] = rgb.r;
            in_place[y_shift + px + 1] = rgb.g;
            in_place[y_shift + px + 2] = rgb.b;
        }
        y_shift += stride as usize;
        luv_shift += luv_stride;
    }
}

fn clip_hist_clahe(bins: &mut [u64], level: f32) {
    let sums: u64 = bins.iter().sum();
    let mean: u64 = (sums as f64 / bins.len() as f64).round() as u64;
    let threshold_value: f64 = mean as f64 * level as f64;
    let clip_limit = mean + threshold_value as u64;
    let mut excess = 0u64;

    for i in 0..bins.len() {
        if bins[i] > clip_limit {
            excess += bins[i];
        }
    }

    let mean_excess = (excess as f64 / bins.len() as f64) as u64;

    for i in 0..bins.len() {
        if bins[i] >= clip_limit {
            bins[i] = clip_limit + mean_excess;
        } else {
            bins[i] = bins[i] + mean_excess;
        }
    }
}
