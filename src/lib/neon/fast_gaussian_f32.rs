use crate::neon::load_f32;
use crate::reflect_index;
use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, EdgeMode, FastBlurChannels};
use std::arch::aarch64::*;

pub fn fast_gaussian_vertical_pass_neon_f32<
    T,
    const CHANNELS_COUNT: usize,
    const EDGE_MODE: usize,
>(
    undef_bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
) {
    let edge_mode: EdgeMode = EDGE_MODE.into();
    let bytes: &UnsafeSlice<'_, f32> = unsafe { std::mem::transmute(undef_bytes) };
    let mut buffer: [[f32; 4]; 1024] = [[0f32; 4]; 1024];
    let channels: FastBlurChannels = CHANNELS_COUNT.into();

    let safe_pixel_count_x = match channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 2,
    };

    let height_wide = height as i64;

    let radius_64 = radius as i64;
    let weight = 1.0f32 / ((radius as f32) * (radius as f32));
    let f_weight = unsafe { vdupq_n_f32(weight) };
    let channels_count = match channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    };
    for x in start..std::cmp::min(width, end) {
        let mut diffs: float32x4_t = unsafe { vdupq_n_f32(0f32) };
        let mut summs: float32x4_t = unsafe { vdupq_n_f32(0f32) };

        let start_y = 0 - 2 * radius as i64;
        for y in start_y..height_wide {
            let current_y = (y * (stride as i64)) as usize;

            if y >= 0 {
                let current_px = ((std::cmp::max(x, 0)) * channels_count) as usize;

                let prepared_px = unsafe { vmulq_f32(summs, f_weight) };

                if CHANNELS_COUNT == 4 {
                    unsafe {
                        let dst_ptr = bytes.slice.as_ptr().add(current_y + current_px) as *mut f32;
                        vst1q_f32(dst_ptr, prepared_px)
                    }
                } else {
                    let new_r = unsafe { vgetq_lane_f32::<0>(prepared_px) };
                    let new_g = unsafe { vgetq_lane_f32::<1>(prepared_px) };
                    let new_b = unsafe { vgetq_lane_f32::<2>(prepared_px) };

                    let offset = current_y + current_px;

                    unsafe {
                        bytes.write(offset, new_r);
                        bytes.write(offset + 1, new_g);
                        bytes.write(offset + 2, new_b);
                    }
                }

                let arr_index = ((y - radius_64) & 1023) as usize;
                let d_arr_index = (y & 1023) as usize;

                let d_buf_ptr = unsafe { buffer.as_mut_ptr().add(d_arr_index) as *mut f32 };
                let mut d_stored = unsafe { vld1q_f32(d_buf_ptr) };
                d_stored = unsafe { vmulq_n_f32(d_stored, 2f32) };

                let buf_ptr = unsafe { buffer.as_mut_ptr().add(arr_index) as *mut f32 };
                let a_stored = unsafe { vld1q_f32(buf_ptr) };

                diffs = unsafe { vaddq_f32(diffs, vsubq_f32(a_stored, d_stored)) };
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;
                let buf_ptr = unsafe { buffer.as_mut_ptr().add(arr_index) as *mut f32 };
                let mut stored = unsafe { vld1q_f32(buf_ptr) };
                stored = unsafe { vmulq_n_f32(stored, 2f32) };
                diffs = unsafe { vsubq_f32(diffs, stored) };
            }

            let next_row_y =
                clamp_edge!(edge_mode, y + radius_64, 0, height_wide - 1) * (stride as usize);
            let next_row_x = (x * channels_count) as usize;

            let s_ptr = unsafe { bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut f32 };
            let pixel_color = load_f32(
                s_ptr,
                x as i64 + safe_pixel_count_x < width as i64,
                channels_count as usize,
            );

            let arr_index = ((y + radius_64) & 1023) as usize;
            let buf_ptr = unsafe { buffer.as_mut_ptr().add(arr_index) as *mut f32 };

            diffs = unsafe { vaddq_f32(diffs, pixel_color) };
            summs = unsafe { vaddq_f32(summs, diffs) };
            unsafe {
                vst1q_f32(buf_ptr, pixel_color);
            }
        }
    }
}

pub fn fast_gaussian_horizontal_pass_neon_f32<
    T,
    const CHANNELS_COUNT: usize,
    const EDGE_MODE: usize,
>(
    undef_bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
) {
    let edge_mode: EdgeMode = EDGE_MODE.into();
    let bytes: &UnsafeSlice<'_, f32> = unsafe { std::mem::transmute(undef_bytes) };
    let mut buffer: [[f32; 4]; 1024] = [[0f32; 4]; 1024];
    let channels: FastBlurChannels = CHANNELS_COUNT.into();
    let safe_pixel_count_x = match channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 2,
    };

    let radius_64 = radius as i64;
    let width_wide = width as i64;
    let weight = 1.0f32 / ((radius as f32) * (radius as f32));
    let f_weight = unsafe { vdupq_n_f32(weight) };
    let channels_count = match channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    };
    for y in start..std::cmp::min(height, end) {
        let mut diffs: float32x4_t = unsafe { vdupq_n_f32(0f32) };
        let mut summs: float32x4_t = unsafe { vdupq_n_f32(0f32) };

        let current_y = ((y as i64) * (stride as i64)) as usize;

        let start_x = 0 - 2 * radius_64;
        for x in start_x..(width as i64) {
            if x >= 0 {
                let current_px = ((std::cmp::max(x, 0) as u32) * channels_count) as usize;

                let prepared_px = unsafe { vmulq_f32(summs, f_weight) };

                if CHANNELS_COUNT == 4 {
                    unsafe {
                        let dst_ptr = bytes.slice.as_ptr().add(current_y + current_px) as *mut f32;
                        vst1q_f32(dst_ptr, prepared_px)
                    }
                } else {
                    let new_r = unsafe { vgetq_lane_f32::<0>(prepared_px) };
                    let new_g = unsafe { vgetq_lane_f32::<1>(prepared_px) };
                    let new_b = unsafe { vgetq_lane_f32::<2>(prepared_px) };

                    let offset = current_y + current_px;

                    unsafe {
                        bytes.write(offset, new_r);
                        bytes.write(offset + 1, new_g);
                        bytes.write(offset + 2, new_b);
                    }
                }

                let arr_index = ((x - radius_64) & 1023) as usize;
                let d_arr_index = (x & 1023) as usize;

                let d_buf_ptr = unsafe { buffer.as_mut_ptr().add(d_arr_index) as *mut f32 };
                let mut d_stored = unsafe { vld1q_f32(d_buf_ptr) };
                d_stored = unsafe { vmulq_n_f32(d_stored, 2f32) };

                let buf_ptr = unsafe { buffer.as_mut_ptr().add(arr_index) as *mut f32 };
                let a_stored = unsafe { vld1q_f32(buf_ptr) };

                diffs = unsafe { vaddq_f32(diffs, vsubq_f32(a_stored, d_stored)) };
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                let buf_ptr = unsafe { buffer.as_mut_ptr().add(arr_index) as *mut f32 };
                let mut stored = unsafe { vld1q_f32(buf_ptr) };
                stored = unsafe { vmulq_n_f32(stored, 2f32) };
                diffs = unsafe { vsubq_f32(diffs, stored) };
            }

            let next_row_y = (y as usize) * (stride as usize);
            let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide - 1);
            let next_row_px = next_row_x * channels_count as usize;

            let s_ptr = unsafe { bytes.slice.as_ptr().add(next_row_y + next_row_px) as *mut f32 };
            let pixel_color = load_f32(
                s_ptr,
                (next_row_x as i64) + (safe_pixel_count_x as i64) < width as i64,
                channels_count as usize,
            );

            let arr_index = ((x + radius_64) & 1023) as usize;
            let buf_ptr = unsafe { buffer.as_mut_ptr().add(arr_index) as *mut f32 };

            diffs = unsafe { vaddq_f32(diffs, pixel_color) };
            summs = unsafe { vaddq_f32(summs, diffs) };
            unsafe {
                vst1q_f32(buf_ptr, pixel_color);
            }
        }
    }
}
