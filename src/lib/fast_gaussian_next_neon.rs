// Copyright (c) Radzivon Bartoshyk. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub mod neon_support {
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;
    use std::arch::aarch64::{
        int32x4_t, vaddq_s32, vcombine_u16, vcvtq_f32_s32, vcvtq_s32_f32, vdupq_n_f32, vdupq_n_s32,
        vget_lane_u8, vget_low_u16, vld1_u8, vld1q_s32, vmovl_u16, vmovl_u8, vmulq_f32,
        vmulq_n_s32, vqmovn_u16, vqmovun_s32, vreinterpretq_s32_u32, vrndq_f32, vst1q_s32,
        vsubq_s32,
    };
    use std::ptr;

    pub(crate) fn fast_gaussian_next_vertical_pass_neon_u8(
        bytes: &UnsafeSlice<u8>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        channels: FastBlurChannels,
    ) {
        let mut buffer: [[i32; 4]; 1024] = [[0; 4]; 1024];
        let mut safe_transient_store: [u8; 8] = [0; 8];

        let safe_pixel_count_x = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 2,
        };

        let height_wide = height as i64;

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let f_weight = unsafe { vdupq_n_f32(weight) };
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for x in start..std::cmp::min(width, end) {
            let mut diffs: int32x4_t = unsafe { vdupq_n_s32(0) };
            let mut ders: int32x4_t = unsafe { vdupq_n_s32(0) };
            let mut summs: int32x4_t = unsafe { vdupq_n_s32(0) };

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;

                if y >= 0 {
                    let current_px = ((std::cmp::max(x, 0)) * channels_count) as usize;

                    let prepared_px_s32 = unsafe {
                        vcvtq_s32_f32(vrndq_f32(vmulq_f32(vcvtq_f32_s32(summs), f_weight)))
                    };
                    let prepared_u16 = unsafe { vqmovun_s32(prepared_px_s32) };
                    let prepared_u8 =
                        unsafe { vqmovn_u16(vcombine_u16(prepared_u16, prepared_u16)) };

                    let new_r = unsafe { vget_lane_u8::<0>(prepared_u8) };
                    let new_g = unsafe { vget_lane_u8::<1>(prepared_u8) };
                    let new_b = unsafe { vget_lane_u8::<2>(prepared_u8) };

                    unsafe {
                        bytes.write(current_y + current_px, new_r);
                        bytes.write(current_y + current_px + 1, new_g);
                        bytes.write(current_y + current_px + 2, new_b);
                    }

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;

                    let buf_ptr = buffer[d_arr_index].as_mut_ptr();
                    let stored = unsafe { vld1q_s32(buf_ptr) };

                    let buf_ptr_1 = buffer[d_arr_index_1].as_mut_ptr();
                    let stored_1 = unsafe { vld1q_s32(buf_ptr_1) };

                    let buf_ptr_2 = buffer[d_arr_index_2].as_mut_ptr();
                    let stored_2 = unsafe { vld1q_s32(buf_ptr_2) };

                    let new_diff =
                        unsafe { vsubq_s32(vmulq_n_s32(vsubq_s32(stored, stored_1), 3), stored_2) };
                    diffs = unsafe { vaddq_s32(diffs, new_diff) };
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let stored = unsafe { vld1q_s32(buf_ptr) };

                    let buf_ptr_1 = buffer[arr_index_1].as_mut_ptr();
                    let stored_1 = unsafe { vld1q_s32(buf_ptr_1) };

                    let new_diff = unsafe { vmulq_n_s32(vsubq_s32(stored, stored_1), 3) };

                    diffs = unsafe { vaddq_s32(diffs, new_diff) };
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let stored = unsafe { vld1q_s32(buf_ptr) };
                    diffs = unsafe { vsubq_s32(diffs, vmulq_n_s32(stored, 3)) };
                }

                let next_row_y = (std::cmp::min(
                    std::cmp::max(y + ((3 * radius_64) >> 1), 0),
                    height_wide - 1,
                ) as usize)
                    * (stride as usize);
                let next_row_x = (x * channels_count) as usize;

                let edge_wh_ptr: *const u8;
                let s_ptr = unsafe { bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut u8 };
                if x as i64 + safe_pixel_count_x < width as i64 {
                    edge_wh_ptr = s_ptr;
                } else {
                    unsafe {
                        ptr::copy_nonoverlapping(
                            s_ptr,
                            safe_transient_store.as_mut_ptr(),
                            channels_count as usize,
                        );
                    }
                    edge_wh_ptr = safe_transient_store.as_ptr();
                }
                let pixel_color = unsafe {
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))))
                };

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer[arr_index].as_mut_ptr();

                diffs = unsafe { vaddq_s32(diffs, pixel_color) };
                ders = unsafe { vaddq_s32(ders, diffs) };
                summs = unsafe { vaddq_s32(summs, ders) };
                unsafe {
                    vst1q_s32(buf_ptr, pixel_color);
                }
            }
        }
    }

    pub(crate) fn fast_gaussian_next_horizontal_pass_neon_u8(
        bytes: &UnsafeSlice<u8>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        channels: FastBlurChannels,
    ) {
        let mut buffer: [[i32; 4]; 1024] = [[0; 4]; 1024];
        let mut safe_transient_store: [u8; 8] = [0; 8];

        let safe_pixel_count_x = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 2,
        };

        let width_wide = width as i64;

        let radius_64 = radius as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let f_weight = unsafe { vdupq_n_f32(weight) };
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for y in start..std::cmp::min(height, end) {
            let mut diffs: int32x4_t = unsafe { vdupq_n_s32(0) };
            let mut ders: int32x4_t = unsafe { vdupq_n_s32(0) };
            let mut summs: int32x4_t = unsafe { vdupq_n_s32(0) };

            let current_y = ((y as i64) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = x as usize * channels_count as usize;

                    let prepared_px_s32 = unsafe {
                        vcvtq_s32_f32(vrndq_f32(vmulq_f32(vcvtq_f32_s32(summs), f_weight)))
                    };
                    let prepared_u16 = unsafe { vqmovun_s32(prepared_px_s32) };
                    let prepared_u8 =
                        unsafe { vqmovn_u16(vcombine_u16(prepared_u16, prepared_u16)) };

                    let new_r = unsafe { vget_lane_u8::<0>(prepared_u8) };
                    let new_g = unsafe { vget_lane_u8::<1>(prepared_u8) };
                    let new_b = unsafe { vget_lane_u8::<2>(prepared_u8) };

                    unsafe {
                        bytes.write(current_y + current_px, new_r);
                        bytes.write(current_y + current_px + 1, new_g);
                        bytes.write(current_y + current_px + 2, new_b);
                    }

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;

                    let buf_ptr = buffer[d_arr_index].as_mut_ptr();
                    let stored = unsafe { vld1q_s32(buf_ptr) };

                    let buf_ptr_1 = buffer[d_arr_index_1].as_mut_ptr();
                    let stored_1 = unsafe { vld1q_s32(buf_ptr_1) };

                    let buf_ptr_2 = buffer[d_arr_index_2].as_mut_ptr();
                    let stored_2 = unsafe { vld1q_s32(buf_ptr_2) };

                    let new_diff =
                        unsafe { vsubq_s32(vmulq_n_s32(vsubq_s32(stored, stored_1), 3), stored_2) };
                    diffs = unsafe { vaddq_s32(diffs, new_diff) };
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let stored = unsafe { vld1q_s32(buf_ptr) };

                    let buf_ptr_1 = buffer[arr_index_1].as_mut_ptr();
                    let stored_1 = unsafe { vld1q_s32(buf_ptr_1) };

                    let new_diff = unsafe { vmulq_n_s32(vsubq_s32(stored, stored_1), 3) };

                    diffs = unsafe { vaddq_s32(diffs, new_diff) };
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    let buf_ptr = buffer[arr_index].as_mut_ptr();
                    let stored = unsafe { vld1q_s32(buf_ptr) };
                    diffs = unsafe { vsubq_s32(diffs, vmulq_n_s32(stored, 3)) };
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x =
                    ((std::cmp::min(std::cmp::max(x + 3 * radius_64 / 2, 0), width_wide - 1)
                        as u32)
                        * channels_count) as usize;

                let edge_wh_ptr: *const u8;
                let s_ptr = unsafe { bytes.slice.as_ptr().add(next_row_y + next_row_x) as *mut u8 };
                if x as i64 + safe_pixel_count_x < width as i64 {
                    edge_wh_ptr = s_ptr;
                } else {
                    unsafe {
                        ptr::copy_nonoverlapping(
                            s_ptr,
                            safe_transient_store.as_mut_ptr(),
                            channels_count as usize,
                        );
                    }
                    edge_wh_ptr = safe_transient_store.as_ptr();
                }
                let pixel_color = unsafe {
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))))
                };

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;
                let buf_ptr = buffer[arr_index].as_mut_ptr();

                diffs = unsafe { vaddq_s32(diffs, pixel_color) };
                ders = unsafe { vaddq_s32(ders, diffs) };
                summs = unsafe { vaddq_s32(summs, ders) };
                unsafe {
                    vst1q_s32(buf_ptr, pixel_color);
                }
            }
        }
    }
}

#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
pub mod neon_support {
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    #[allow(dead_code)]
    pub(crate) fn fast_gaussian_next_vertical_pass_neon_u8(
        _bytes: &UnsafeSlice<u8>,
        _stride: u32,
        _width: u32,
        _height: u32,
        _radius: u32,
        _start: u32,
        _end: u32,
        _channels: FastBlurChannels,
    ) {
    }

    #[allow(dead_code)]
    pub(crate) fn fast_gaussian_next_horizontal_pass_neon_u8(
        _bytes: &UnsafeSlice<u8>,
        _stride: u32,
        _width: u32,
        _height: u32,
        _radius: u32,
        _start: u32,
        _end: u32,
        _channels: FastBlurChannels,
    ) {
    }
}
