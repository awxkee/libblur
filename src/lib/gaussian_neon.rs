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
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32, vdupq_n_f32,
        vget_low_u16, vld1q_f32, vmovl_u16, vmovl_u8, vmulq_f32, vqmovn_u16, vqmovn_u32, vst1_u8,
    };
    use std::arch::aarch64::{
        uint8x16_t, vget_high_u8, vget_low_u8, vld1q_u8, vmovl_high_u16, vrndq_f32,
    };

    use crate::neon_utils::neon_utils::load_u8_u32;
    #[allow(unused_imports)]
    use crate::unsafe_slice::UnsafeSlice;

    #[allow(dead_code)]
    #[cfg(target_arch = "aarch64")]
    pub fn gaussian_blur_horizontal_pass_impl_neon_3channels_u8(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        kernel_size: usize,
        kernel: &Vec<f32>,
        start_y: u32,
        end_y: u32,
    ) {
        let half_kernel = (kernel_size / 2) as i32;

        let mut safe_transient_store: [u8; 8] = [0; 8];

        let eraser_store: [f32; 4] = [1f32, 1f32, 1f32, 0f32];

        let channels_count = 3;

        let eraser: float32x4_t = unsafe { vld1q_f32(eraser_store.as_ptr()) };

        for y in start_y..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store: float32x4_t = unsafe { vdupq_n_f32(0f32) };
                for r in -half_kernel..=half_kernel {
                    let px =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize
                            * 3;
                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                    let pixel_colors_u32 = load_u8_u32(s_ptr, x + 3 < width, channels_count);
                    let mut pixel_colors_f32 =
                        unsafe { vmulq_f32(vcvtq_f32_u32(pixel_colors_u32), eraser) };
                    let weight = kernel[(r + half_kernel) as usize];
                    let f_weight: float32x4_t = unsafe { vdupq_n_f32(weight) };
                    pixel_colors_f32 = unsafe { vmulq_f32(pixel_colors_f32, f_weight) };
                    store = unsafe { vaddq_f32(store, pixel_colors_f32) };
                }

                let px = x as usize * channels_count;

                let dst_ptr = unsafe { unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8 };
                let px_16 = unsafe { vqmovn_u32(vcvtq_u32_f32(vrndq_f32(store))) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                if x + 3 < width {
                    unsafe {
                        vst1_u8(dst_ptr, px_8);
                    };
                } else {
                    unsafe {
                        vst1_u8(safe_transient_store.as_mut_ptr(), px_8);
                    }
                    unsafe {
                        unsafe_dst.write(y_dst_shift + px, safe_transient_store[0]);
                        unsafe_dst.write(y_dst_shift + px + 1, safe_transient_store[1]);
                        unsafe_dst.write(y_dst_shift + px + 2, safe_transient_store[2]);
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    #[cfg(target_arch = "aarch64")]
    pub fn gaussian_blur_vertical_pass_impl_neon_3channels_u8(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: usize,
        kernel: &Vec<f32>,
        start_y: u32,
        end_y: u32,
    ) {
        let channels_count = 3;
        let half_kernel = (kernel_size / 2) as i32;

        let mut safe_transient_store: [u8; 8] = [0; 8];

        let eraser_store: [f32; 4] = [1f32, 1f32, 1f32, 0f32];
        let eraser: float32x4_t = unsafe { vld1q_f32(eraser_store.as_ptr()) };

        for y in start_y..end_y {
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store: float32x4_t = unsafe { vdupq_n_f32(0f32) };
                let px = x as usize * channels_count;
                for r in -half_kernel..=half_kernel {
                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;

                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                    let pixel_colors_u32 = load_u8_u32(s_ptr, x + 3 < width, channels_count);
                    let mut pixel_colors_f32 =
                        unsafe { vmulq_f32(vcvtq_f32_u32(pixel_colors_u32), eraser) };
                    let weight = kernel[(r + half_kernel) as usize];
                    let f_weight: float32x4_t = unsafe { vdupq_n_f32(weight) };
                    pixel_colors_f32 = unsafe { vmulq_f32(pixel_colors_f32, f_weight) };
                    store = unsafe { vaddq_f32(store, pixel_colors_f32) };
                }

                let dst_ptr = unsafe { unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8 };
                let px_16 = unsafe { vqmovn_u32(vcvtq_u32_f32(vrndq_f32(store))) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                if x + 3 < width {
                    unsafe {
                        vst1_u8(dst_ptr, px_8);
                    };
                } else {
                    unsafe {
                        vst1_u8(safe_transient_store.as_mut_ptr(), px_8);
                    }
                    unsafe {
                        unsafe_dst.write(y_dst_shift + px, safe_transient_store[0]);
                        unsafe_dst.write(y_dst_shift + px + 1, safe_transient_store[1]);
                        unsafe_dst.write(y_dst_shift + px + 2, safe_transient_store[2]);
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    #[cfg(target_arch = "aarch64")]
    pub fn gaussian_blur_horizontal_pass_impl_neon_4channels_u8(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        kernel_size: usize,
        kernel: &Vec<f32>,
        start_y: u32,
        end_y: u32,
    ) {
        let half_kernel = (kernel_size / 2) as i32;

        let mut safe_transient_store: [u8; 8] = [0; 8];

        let channels_count: u32 = 4;

        for y in start_y..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store: float32x4_t = unsafe { vdupq_n_f32(0f32) };

                let mut r = -half_kernel;

                while r + 4 <= half_kernel && x as i64 + r as i64 + 4 < width as i64 {
                    let px =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize
                            * channels_count as usize;
                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                    let pixel_colors: uint8x16_t = unsafe { vld1q_u8(s_ptr) };
                    let mut pixel_colors_u16 = unsafe { vmovl_u8(vget_low_u8(pixel_colors)) };
                    let mut pixel_colors_u32 = unsafe { vmovl_u16(vget_low_u16(pixel_colors_u16)) };
                    let mut pixel_colors_f32 = unsafe { vcvtq_f32_u32(pixel_colors_u32) };
                    let mut weight = kernel[(r + half_kernel) as usize];
                    let mut f_weight: float32x4_t = unsafe { vdupq_n_f32(weight) };
                    pixel_colors_f32 = unsafe { vmulq_f32(pixel_colors_f32, f_weight) };
                    store = unsafe { vaddq_f32(store, pixel_colors_f32) };

                    pixel_colors_u32 = unsafe { vmovl_high_u16(pixel_colors_u16) };
                    pixel_colors_f32 = unsafe { vcvtq_f32_u32(pixel_colors_u32) };

                    weight = kernel[(r + half_kernel + 1) as usize];
                    f_weight = unsafe { vdupq_n_f32(weight) };
                    pixel_colors_f32 = unsafe { vmulq_f32(pixel_colors_f32, f_weight) };
                    store = unsafe { vaddq_f32(store, pixel_colors_f32) };

                    pixel_colors_u16 = unsafe { vmovl_u8(vget_high_u8(pixel_colors)) };

                    pixel_colors_u32 = unsafe { vmovl_u16(vget_low_u16(pixel_colors_u16)) };
                    let mut pixel_colors_f32 = unsafe { vcvtq_f32_u32(pixel_colors_u32) };
                    let mut weight = kernel[(r + half_kernel + 2) as usize];
                    let mut f_weight: float32x4_t = unsafe { vdupq_n_f32(weight) };
                    pixel_colors_f32 = unsafe { vmulq_f32(pixel_colors_f32, f_weight) };
                    store = unsafe { vaddq_f32(store, pixel_colors_f32) };

                    pixel_colors_u32 = unsafe { vmovl_high_u16(pixel_colors_u16) };
                    pixel_colors_f32 = unsafe { vcvtq_f32_u32(pixel_colors_u32) };

                    weight = kernel[(r + half_kernel + 3) as usize];
                    f_weight = unsafe { vdupq_n_f32(weight) };
                    pixel_colors_f32 = unsafe { vmulq_f32(pixel_colors_f32, f_weight) };
                    store = unsafe { vaddq_f32(store, pixel_colors_f32) };

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x = std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                    let px = current_x * channels_count as usize;
                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                    let pixel_colors_u32 =
                        load_u8_u32(s_ptr, current_x as i64 + 2 < width as i64, channels_count as usize);
                    let mut pixel_colors_f32 = unsafe { vcvtq_f32_u32(pixel_colors_u32) };
                    let weight = kernel[(r + half_kernel) as usize];
                    let f_weight: float32x4_t = unsafe { vdupq_n_f32(weight) };
                    pixel_colors_f32 = unsafe { vmulq_f32(pixel_colors_f32, f_weight) };
                    store = unsafe { vaddq_f32(store, pixel_colors_f32) };

                    r += 1;
                }

                let px = x as usize * channels_count as usize;

                let dst_ptr = unsafe { unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8 };
                let px_16 = unsafe { vqmovn_u32(vcvtq_u32_f32(vrndq_f32(store))) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                if x + 2 < width {
                    unsafe {
                        vst1_u8(dst_ptr, px_8);
                    };
                } else {
                    unsafe {
                        vst1_u8(safe_transient_store.as_mut_ptr(), px_8);
                    }
                    unsafe {
                        unsafe_dst.write(y_dst_shift + px, safe_transient_store[0]);
                        unsafe_dst.write(y_dst_shift + px + 1, safe_transient_store[1]);
                        unsafe_dst.write(y_dst_shift + px + 2, safe_transient_store[2]);
                        unsafe_dst.write(y_dst_shift + px + 3, safe_transient_store[3]);
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    #[cfg(target_arch = "aarch64")]
    pub fn gaussian_blur_vertical_pass_impl_neon_4channels_u8(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: usize,
        kernel: &Vec<f32>,
        start_y: u32,
        end_y: u32,
    ) {
        let half_kernel = (kernel_size / 2) as i32;

        let mut safe_transient_store: [u8; 8] = [0; 8];

        let channels_count: u32 = 4;

        for y in start_y..end_y {
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store: float32x4_t = unsafe { vdupq_n_f32(0f32) };

                let mut r = -half_kernel;

                let px = x as usize * channels_count as usize;

                while r <= half_kernel {
                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                    let pixel_colors_u32 =
                        load_u8_u32(s_ptr, x + 2 < width, channels_count as usize);
                    let mut pixel_colors_f32 = unsafe { vcvtq_f32_u32(pixel_colors_u32) };
                    let weight = kernel[(r + half_kernel) as usize];
                    let f_weight: float32x4_t = unsafe { vdupq_n_f32(weight) };
                    pixel_colors_f32 = unsafe { vmulq_f32(pixel_colors_f32, f_weight) };
                    store = unsafe { vaddq_f32(store, pixel_colors_f32) };

                    r += 1;
                }

                let dst_ptr = unsafe { unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8 };
                let px_16 = unsafe { vqmovn_u32(vcvtq_u32_f32(vrndq_f32(store))) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                if x + 2 < width {
                    unsafe {
                        vst1_u8(dst_ptr, px_8);
                    };
                } else {
                    unsafe {
                        vst1_u8(safe_transient_store.as_mut_ptr(), px_8);
                    }
                    unsafe {
                        unsafe_dst.write(y_dst_shift + px, safe_transient_store[0]);
                        unsafe_dst.write(y_dst_shift + px + 1, safe_transient_store[1]);
                        unsafe_dst.write(y_dst_shift + px + 2, safe_transient_store[2]);
                        unsafe_dst.write(y_dst_shift + px + 3, safe_transient_store[3]);
                    }
                }
            }
        }
    }
}

#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
pub mod neon_support {
    use crate::unsafe_slice::UnsafeSlice;

    #[allow(dead_code)]
    pub fn gaussian_blur_horizontal_pass_impl_neon_3channels_u8(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _dst_stride: u32,
        _width: u32,
        _kernel_size: usize,
        _kernel: &Vec<f32>,
        _start_y: u32,
        _end_y: u32,
    ) {
    }

    #[allow(dead_code)]
    pub fn gaussian_blur_vertical_pass_impl_neon_3channels_u8(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _st_stride: u32,
        _width: u32,
        _height: u32,
        _kernel_size: usize,
        _kernel: &Vec<f32>,
        _start_y: u32,
        _end_y: u32,
    ) {
    }

    #[allow(dead_code)]
    pub fn gaussian_blur_vertical_pass_impl_neon_4channels_u8(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _dst_stride: u32,
        _width: u32,
        _height: u32,
        _kernel_size: usize,
        _kernel: &Vec<f32>,
        _start_y: u32,
        _end_y: u32,
    ) {
    }

    #[allow(dead_code)]
    pub fn gaussian_blur_horizontal_pass_impl_neon_4channels_u8(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _dst_stride: u32,
        _width: u32,
        _kernel_size: usize,
        _kernel: &Vec<f32>,
        _start_y: u32,
        _end_y: u32,
    ) {
    }
}
