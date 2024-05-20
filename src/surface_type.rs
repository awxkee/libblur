use std::slice;
use crate::android_rgba::{Rgba, ToRgba8, ToRgbaF32};

use crate::gamma_curves::gamma_curve;
use crate::hsl::Rgb;
use crate::lab::Lab;

pub fn reformat_surface_u8_to_linear(
    original: &[u8],
    src_stride: u32,
    destination: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut src_shift = 0usize;
    let mut dst_shift = 0usize;
    let dst_o_ptr = destination.as_mut_ptr();
    for _ in 0..height as usize {
        let src_ptr = unsafe { original.as_ptr().add(src_shift) };
        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * 4) };
        let dst_ptr = unsafe { dst_o_ptr.add(dst_shift) as *mut f32 };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * 4) };

        for x in 0..width as usize {
            let px = x * 4;
            let src_pixel = Rgba::<u8>::new(
                src_slice[px],
                src_slice[px + 1],
                src_slice[px + 2],
                src_slice[px + 3],
            );
            let gamma_pixel = src_pixel.to_rgba_f32();
            let linear_pixel = Rgba::<f32>::new(
                gamma_curve::srgb_to_linear(gamma_pixel.r),
                gamma_curve::srgb_to_linear(gamma_pixel.g),
                gamma_curve::srgb_to_linear(gamma_pixel.b),
                gamma_curve::srgb_to_linear(gamma_pixel.a),
            );
            dst_slice[px] = linear_pixel.r;
            dst_slice[px + 1] = linear_pixel.g;
            dst_slice[px + 2] = linear_pixel.b;
            dst_slice[px + 3] = linear_pixel.a;
        }
        dst_shift += dst_stride as usize;
        src_shift += src_stride as usize;
    }
}

pub fn reformat_surface_linear_to_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut src_shift = 0usize;
    let mut dst_shift = 0usize;
    for _ in 0..height as usize {
        let src_ptr = unsafe { (src.as_ptr() as *mut u8).add(src_shift) as *mut f32 };
        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * 4) };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_shift) };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * 4) };

        for x in 0..width as usize {
            let px = x * 4;
            let linear_pixel = Rgba::<f32>::new(
                src_slice[px],
                src_slice[px + 1],
                src_slice[px + 2],
                src_slice[px + 3],
            );
            let gamma_pixel = Rgba::<f32>::new(
                gamma_curve::srgb_from_linear(linear_pixel.r),
                gamma_curve::srgb_from_linear(linear_pixel.g),
                gamma_curve::srgb_from_linear(linear_pixel.b),
                gamma_curve::srgb_from_linear(linear_pixel.a),
            );
            let final_pixel = gamma_pixel.to_rgba8();
            dst_slice[px] = final_pixel.r;
            dst_slice[px + 1] = final_pixel.g;
            dst_slice[px + 2] = final_pixel.b;
            dst_slice[px + 3] = final_pixel.a;
        }
        dst_shift += dst_stride as usize;
        src_shift += src_stride as usize;
    }
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C-unwind" fn aire_reformat_surface_u8_to_linear(
    src: *const u8,
    src_stride: u32,
    dst: *const f32,
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let src_slice = unsafe { slice::from_raw_parts(src, src_stride as usize * height as usize) };
    let dst_slice =
        unsafe { slice::from_raw_parts_mut(dst as *mut u8, dst_stride as usize * height as usize) };
    reformat_surface_u8_to_linear(src_slice, src_stride, dst_slice, dst_stride, width, height);
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C-unwind" fn aire_reformat_surface_linear_to_u8(
    src: *const f32,
    src_stride: u32,
    dst: *const u8,
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let src_slice = unsafe { slice::from_raw_parts(src as *const u8, src_stride as usize * height as usize) };
    let dst_slice =
        unsafe { slice::from_raw_parts_mut(dst as *mut u8, dst_stride as usize * height as usize) };
    reformat_surface_linear_to_u8(src_slice, src_stride, dst_slice, dst_stride, width, height);
}

pub fn reformat_surface_u8_to_laba(
    original: &[u8],
    src_stride: u32,
    destination: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut src_shift = 0usize;
    let mut dst_shift = 0usize;
    for _ in 0..height as usize {
        let src_ptr = unsafe { original.as_ptr().add(src_shift) };
        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * 4) };
        let dst_ptr = unsafe { destination.as_mut_ptr().add(dst_shift) as *mut f32 };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * 4) };

        for x in 0..width as usize {
            let px = x * 4;
            let src_pixel = Rgb::new(
                src_slice[px],
                src_slice[px + 1],
                src_slice[px + 2],
            );
            let lab = src_pixel.to_lab();
            let a_f32 = src_slice[px + 3] as f32 / 255f32;
            let laba_pixel = Rgba::<f32>::new(
                lab.l,
                lab.a,
                lab.b,
                a_f32,
            );
            dst_slice[px] = laba_pixel.r;
            dst_slice[px + 1] = laba_pixel.g;
            dst_slice[px + 2] = laba_pixel.b;
            dst_slice[px + 3] = laba_pixel.a;
        }
        dst_shift += dst_stride as usize;
        src_shift += src_stride as usize;
    }
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C-unwind" fn aire_reformat_surface_u8_to_laba(
    src: *const u8,
    src_stride: u32,
    dst: *const f32,
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let src_slice = unsafe { slice::from_raw_parts(src, src_stride as usize * height as usize) };
    let dst_slice =
        unsafe { slice::from_raw_parts_mut(dst as *mut u8, dst_stride as usize * height as usize) };
    reformat_surface_u8_to_laba(src_slice, src_stride, dst_slice, dst_stride, width, height);
}

pub fn reformat_surface_laba_to_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut src_shift = 0usize;
    let mut dst_shift = 0usize;
    for _ in 0..height as usize {
        let src_ptr = unsafe { (src.as_ptr() as *mut u8).add(src_shift) as *mut f32 };
        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * 4) };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_shift) };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * 4) };

        for x in 0..width as usize {
            let px = x * 4;
            let lab = Lab::new(src_slice[px], src_slice[px + 1], src_slice[px + 2]);
            let rgb_pixel = lab.to_rgb8();
            let alpha = (src_slice[px + 3] * 255f32).min(255f32).max(0f32) as u8;
            dst_slice[px] = rgb_pixel.r;
            dst_slice[px + 1] = rgb_pixel.g;
            dst_slice[px + 2] = rgb_pixel.b;
            dst_slice[px + 3] = alpha;
        }
        dst_shift += dst_stride as usize;
        src_shift += src_stride as usize;
    }
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C-unwind" fn aire_reformat_surface_laba_to_u8(
    src: *const f32,
    src_stride: u32,
    dst: *const u8,
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let src_slice = unsafe { slice::from_raw_parts(src as *const u8, src_stride as usize * height as usize) };
    let dst_slice =
        unsafe { slice::from_raw_parts_mut(dst as *mut u8, dst_stride as usize * height as usize) };
    reformat_surface_laba_to_u8(src_slice, src_stride, dst_slice, dst_stride, width, height);
}