extern crate core;

use std::{fs, slice};
use std::time::Instant;

use image::{EncodableLayout, GenericImageView};
use image::io::Reader as ImageReader;
use yuvutils_rs::{yuv_nv12_p10_msb_to_bgra, YuvRange, YuvStandardMatrix};
use crate::clahe::{ahe_lab_rgb, ahe_luv_rgb, ahe_yuv_rgb, clahe_luv_rgb, clahe_yuv_rgb, clahe_yuv_rgba, ClaheGridSize};

use crate::equalize_hist::{equalize_histogram, equalize_histogram_squares, EqualizeHistogramChannels};
use crate::surface_type::reformat_surface_u8_to_linear;

mod equalize_hist;
mod histogram;
mod hsl;
mod hsv;
mod lab;
mod xyz;
mod surface_type;
mod gamma_curves;
mod android_bitmap;
mod android_rgba;
mod clahe;
mod luv;

#[allow(dead_code)]
fn f32_to_f16(bytes: Vec<f32>) -> Vec<u16> {
    return bytes
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
}

#[allow(dead_code)]
fn f16_to_f32(bytes: Vec<u16>) -> Vec<f32> {
    return bytes
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();
}

fn rgb_to_rgba(vec: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.resize(height as usize * width as usize * 4usize, 0u8);
    let src_stride = width as usize * 3;
    let dst_stride = width as usize * 4;
    for y in 0..height as usize {
        for x in 0..width as usize {
            bytes[dst_stride * y + x*4] = vec[src_stride * y + x * 3];
            bytes[dst_stride * y + x*4 + 1] = vec[src_stride * y + x * 3 + 1];
            bytes[dst_stride * y + x*4 + 2] = vec[src_stride * y + x * 3 + 2];
            bytes[dst_stride * y + x*4 + 3] = 255;
        }
    }
    bytes
}

fn main() {
    let img = ImageReader::open("assets/white_noise.jpeg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    println!("dimensions {:?}", img.dimensions());

    println!("{:?}", img.color());
    let mut src_bytes = img.as_bytes();
    let mv = rgb_to_rgba(src_bytes, dimensions.0, dimensions.1);
    src_bytes = &mv;
    let channels = 4;
    let stride = dimensions.0 as usize * channels;
    let mut bytes: Vec<u8> = Vec::with_capacity(dimensions.1 as usize * stride);
    for i in 0..dimensions.1 as usize * stride {
        bytes.push(src_bytes[i]);
    }
    let mut dst_bytes: Vec<u8> = Vec::with_capacity(dimensions.1 as usize * stride);
    dst_bytes.resize(dimensions.1 as usize * stride, 0);

    let mut start_time_std = Instant::now();

    start_time_std = Instant::now();

    let elapsed_time_std = start_time_std.elapsed();
    println!("Double pass time: {:.2?}", elapsed_time_std);

    let start_time = Instant::now();

    // let mut y_plane: Vec<u8> = Vec::new();
    // y_plane.resize(dimensions.0 as usize * dimensions.1 as usize, 0u8);
    //
    // let mut u_plane: Vec<u8> = Vec::new();
    // u_plane.resize((dimensions.0 as usize / 2) * dimensions.1 as usize, 0u8);
    //
    // let mut v_plane: Vec<u8> = Vec::new();
    // v_plane.resize((dimensions.0 as usize / 2) * dimensions.1 as usize, 0u8);
    //
    // rgb_to_yuv422(
    //     &mut y_plane,
    //     dimensions.0,
    //     &mut u_plane,
    //     dimensions.0 / 2,
    //     &mut v_plane,
    //     dimensions.0 / 2,
    //     &bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     YuvRange::TV,
    //     YuvStandardMatrix::Bt709,
    // );
    //
    // yuv422_to_rgb(
    //     &y_plane,
    //     dimensions.0,
    //     &u_plane,
    //     dimensions.0 / 2,
    //     &v_plane,
    //     dimensions.0 / 2,
    //     &mut bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     YuvRange::TV,
    //     Bt709,
    // );

    // ahe_luv_rgb(&mut bytes, stride as u32, dimensions.0, dimensions.1, 3f32, ClaheGridSize::new(8, 8));
    equalize_histogram::<{EqualizeHistogramChannels::Channels4 as u8}>(&mut bytes, stride as u32, dimensions.0, dimensions.1);
    // equalize_histogram_squares::<{EqualizeHistogramChannels::Channels3 as u8}>(&mut bytes, stride as u32, dimensions.0, dimensions.1, ClaheGridSize::new(8, 8));
    //
    // libblur::fast_gaussian(
    //     &mut bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     512,
    //     FastBlurChannels::Channels3,
    // );
    // libblur::gaussian_blur(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     127*2+1,
    //     256f32 / 6f32,
    //     FastBlurChannels::Channels3,
    // );
    // libblur::median_blur(
    //     &bytes,
    //     stride as u32,
    //     &mut dst_bytes,
    //     stride as u32,
    //     dimensions.0,
    //     dimensions.1,
    //     36,
    //     FastBlurChannels::Channels3,
    // );
    // libblur::gaussian_box_blur(&bytes, stride as u32, &mut dst_bytes, stride as u32, dimensions.0, dimensions.1, 128, FastBlurChannels::Channels3);

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Elapsed time: {:.2?}", elapsed_time);

    if channels == 4 {
        image::save_buffer(
            "blurred_f_a.png",
            bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgba8,
        )
            .unwrap();
    } else {
        image::save_buffer(
            "blurred_f_a.jpg",
            bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgb8,
        )
            .unwrap();
    }

    // let read_file_8 = fs::read("file.yuv").unwrap();
    // println!("Read file yuv size {}", read_file_8.len());
    // let slice = read_file_8.as_ptr() as *const u16;
    // let read_file_slice = unsafe { slice::from_raw_parts(slice, read_file_8.len() / 2) } ;
    // let mut read_file: Vec<u16> = Vec::with_capacity(read_file_8.len() / 2);
    //
    // // Iterate over the u8 vector in chunks of 2 bytes and convert to u16
    // for chunk in read_file_8.chunks_exact(2) {
    //     let u16_value = u16::from_be_bytes([chunk[1], chunk[0]]);
    //     read_file.push(u16_value);
    // }
    //
    // let nv12_width = 720usize;
    // let nv12_height = 1280usize;
    // let uv_stride = ((nv12_width + 1) / 2) * 2;
    // let uv_height = (nv12_height + 1) / 2;
    // let y_stride = nv12_width * std::mem::size_of::<u16>();
    //
    // let mut y_plane: Vec<u16> = Vec::new();
    // y_plane.resize(nv12_width * nv12_height, 0u16);
    // let mut uv_plane: Vec<u16> = Vec::new();
    // uv_plane.resize(uv_stride * uv_height, 0u16);
    //
    // for i in 0..nv12_width * nv12_height {
    //     y_plane[i] = read_file[i];
    // }
    //
    // let uv_data = &read_file[nv12_width * nv12_height..];
    // for i in 0..uv_data.len() {
    //     uv_plane[i] = uv_data[i];
    // }
    //
    // let u_data = &read_file[nv12_width * nv12_height..];
    // let v_data = &u_data[u_data.len() / 2..];
    // let mut i = 0usize;
    // // for k in 0..v_data.len() {
    // //     uv_plane[i] = u_data[k];
    // //     uv_plane[i + 1] = v_data[k];
    // //     i += 2;
    // // }
    //
    // let mut bgra_buffer: Vec<u8> = Vec::new();
    // bgra_buffer.resize(nv12_width * 4 * nv12_height, 0u8);
    //
    // let start_time = Instant::now();
    //
    // yuv_nv12_p10_msb_to_bgra(
    //     y_plane.as_slice(),
    //     y_stride as u32,
    //     uv_plane.as_slice(),
    //     uv_stride as u32 * std::mem::size_of::<u16>() as u32,
    //     bgra_buffer.as_mut_slice(),
    //     nv12_width as u32 * 4u32,
    //     nv12_width as u32,
    //     nv12_height as u32,
    //     YuvRange::TV,
    //     YuvStandardMatrix::Bt709,
    // );
    // let elapsed_time = start_time.elapsed();
    // // Print the elapsed time in milliseconds
    // println!("NV12 Execution time: {:.2?}", elapsed_time);
    //
    // let mut rgba_buffer: Vec<u8> = Vec::new();
    // rgba_buffer.resize(nv12_width * 4 * nv12_height, 0u8);
    // let nv_stride = nv12_width * 4;
    //
    // for y in 0..nv12_height {
    //     for x in 0..nv12_width {
    //         rgba_buffer[y * nv_stride + x * 4] = bgra_buffer[y * nv_stride + x * 4 + 2];
    //         rgba_buffer[y * nv_stride + x * 4 + 1] = bgra_buffer[y * nv_stride + x * 4 + 1];
    //         rgba_buffer[y * nv_stride + x * 4 + 2] = bgra_buffer[y * nv_stride + x * 4];
    //         rgba_buffer[y * nv_stride + x * 4 + 3] = bgra_buffer[y * nv_stride + x * 4 + 3];
    //     }
    // }
    //
    // image::save_buffer(
    //     "converted.png",
    //     rgba_buffer.as_bytes(),
    //     nv12_width as u32,
    //     nv12_height as u32,
    //     image::ExtendedColorType::Rgba8,
    // )
    //     .unwrap();
}
