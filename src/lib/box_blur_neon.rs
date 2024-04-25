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
    use std::arch::aarch64::{uint32x4_t, vaddq_u32, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32, vdupq_n_f32, vdupq_n_u32, vget_low_u16, vld1_u8, vld1q_u32, vmovl_u16, vmovl_u8, vmulq_f32, vmulq_u32, vqmovn_u16, vqmovn_u32, vst1_u8, vsubq_u32};
    use std::ptr;
    use crate::FastBlurChannels;

    #[allow(dead_code)]
    pub(crate) fn box_blur_horizontal_pass_4channels_u8_impl(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        radius: u32,
        start_y: u32,
        end_y: u32,
        channels: FastBlurChannels,
    ) {
        let safe_pixel_count_x = match channels {
            FastBlurChannels::Channels3 => { 3 }
            FastBlurChannels::Channels4 => { 2 }
        };

        let eraser_store: [u32; 4] = [1u32, 1u32, 1u32, 0u32];
        let eraser: uint32x4_t = unsafe { vld1q_u32(eraser_store.as_ptr()) };

        let kernel_scale = 1f32 / (radius * 2) as f32;
        let v_kernel_scale = unsafe { vdupq_n_f32(kernel_scale) };
        let kernel_size = radius * 2 + 1;
        let edge_count = (kernel_size / 2) + 1;
        let v_edge_count = unsafe { vdupq_n_u32(edge_count) };

        let mut safe_transient_store: [u8; 8] = [0; 8];

        let channels_count: u32 = match channels {
            FastBlurChannels::Channels3 => { 3 }
            FastBlurChannels::Channels4 => { 4 }
        };
        let half_kernel = kernel_size / 2;

        for y in start_y..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store: uint32x4_t;
            {
                let edge_wh_ptr: *const u8;
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift) };
                if safe_pixel_count_x < width {
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
                let edge_colors =
                    unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))) };
                store = unsafe { vmulq_u32(edge_colors, v_edge_count) };
            }

            for x in 1..std::cmp::min(half_kernel, width) {
                let px = x as usize * channels_count as usize;
                let edge_wh_ptr: *const u8;
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                if safe_pixel_count_x < width {
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
                let edge_colors =
                    unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))) };
                store = unsafe { vaddq_u32(store, edge_colors) };
            }

            for x in 0..width {
                // preload edge pixels

                let next =
                    std::cmp::min(x + half_kernel, width - 1) as usize * channels_count as usize;

                let previous = std::cmp::max(x as i64 - half_kernel as i64, 0) as usize
                    * channels_count as usize;

                // subtract previous
                {
                    let edge_wh_ptr: *const u8;
                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + previous) };
                    if x + safe_pixel_count_x < width {
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
                    let edge_colors =
                        unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))) };
                    store = unsafe { vsubq_u32(store, edge_colors) };
                }

                // add next
                {
                    let edge_wh_ptr: *const u8;
                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + next) };
                    if x + safe_pixel_count_x < width {
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
                    let edge_colors =
                        unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))) };
                    store = unsafe { vaddq_u32(store, edge_colors) };
                }

                let px = x as usize * channels_count as usize;

                let dst_ptr = unsafe { unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8 };

                match channels {
                    FastBlurChannels::Channels3 => {
                        store = unsafe { vmulq_u32(store, eraser) };
                    }
                    FastBlurChannels::Channels4 => {}
                }

                let scale_store =
                    unsafe { vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(store), v_kernel_scale)) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                if x + safe_pixel_count_x < width {
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
                        match channels {
                            FastBlurChannels::Channels3 => {}
                            FastBlurChannels::Channels4 => {
                                unsafe_dst.write(y_dst_shift + px + 3, safe_transient_store[3]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn box_blur_vertical_pass_4channels_u8_impl(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start_x: u32,
        end_x: u32,
        channels: FastBlurChannels,
    ) {
        let safe_pixel_count_x = match channels {
            FastBlurChannels::Channels3 => { 3 }
            FastBlurChannels::Channels4 => { 2 }
        };

        let eraser_store: [u32; 4] = [1u32, 1u32, 1u32, 0u32];
        let eraser: uint32x4_t = unsafe { vld1q_u32(eraser_store.as_ptr()) };

        let kernel_scale = 1f32 / (radius * 2) as f32;
        let v_kernel_scale = unsafe { vdupq_n_f32(kernel_scale) };
        let kernel_size = radius * 2 + 1;
        let edge_count = (kernel_size / 2) + 1;
        let v_edge_count = unsafe { vdupq_n_u32(edge_count) };

        let mut safe_transient_store: [u8; 8] = [0; 8];

        let channels_count: u32 = match channels {
            FastBlurChannels::Channels3 => { 3 }
            FastBlurChannels::Channels4 => { 4 }
        };
        let half_kernel = kernel_size / 2;

        for x in start_x..end_x {
            let px = x as usize * channels_count as usize;

            let mut store: uint32x4_t;
            {
                let edge_wh_ptr: *const u8;
                let s_ptr = unsafe { src.as_ptr().add(px) };
                if safe_pixel_count_x < width {
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
                let edge_colors =
                    unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))) };
                store = unsafe { vmulq_u32(edge_colors, v_edge_count) };
            }

            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let edge_wh_ptr: *const u8;
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                if safe_pixel_count_x < width {
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
                let edge_colors =
                    unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))) };
                store = unsafe { vaddq_u32(store, edge_colors) };
            }

            for y in 0..height {
                // preload edge pixels
                let next =
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                // subtract previous
                {
                    let edge_wh_ptr: *const u8;
                    let s_ptr = unsafe { src.as_ptr().add(previous + px) };
                    if x + safe_pixel_count_x < width {
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
                    let edge_colors =
                        unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))) };
                    store = unsafe { vsubq_u32(store, edge_colors) };
                }

                // add next
                {
                    let edge_wh_ptr: *const u8;
                    let s_ptr = unsafe { src.as_ptr().add(next + px) };
                    if x + safe_pixel_count_x < width {
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
                    let edge_colors =
                        unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_wh_ptr)))) };
                    store = unsafe { vaddq_u32(store, edge_colors) };
                }

                let px = x as usize * channels_count as usize;

                let dst_ptr = unsafe { unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8 };

                match channels {
                    FastBlurChannels::Channels3 => {
                        store = unsafe { vmulq_u32(store, eraser) };
                    }
                    FastBlurChannels::Channels4 => {}
                }

                let scale_store =
                    unsafe { vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(store), v_kernel_scale)) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };

                // Since the main axis X it is important not overwrite or data race the next tile
                if x + safe_pixel_count_x < std::cmp::min(end_x, width) {
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
                        match channels {
                            FastBlurChannels::Channels3 => {}
                            FastBlurChannels::Channels4 => {
                                unsafe_dst.write(y_dst_shift + px + 3, safe_transient_store[3]);
                            }
                        }
                    }
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
    pub(crate) fn box_blur_horizontal_pass_4channels_u8_impl(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _dst_stride: u32,
        _width: u32,
        _radius: u32,
        _box_channels: FastBlurChannels,
        _start_y: u32,
        _end_y: u32,
        _channels: FastBlurChannels,
    ) {}

    #[allow(dead_code)]
    pub(crate) fn box_blur_vertical_pass_4channels_u8_impl(
        _src: &Vec<u8>,
        _src_stride: u32,
        _unsafe_dst: &UnsafeSlice<u8>,
        _dst_stride: u32,
        _width: u32,
        _height: u32,
        _radius: u32,
        _start_y: u32,
        _end_y: u32,
        _channels: FastBlurChannels,
    ) {}
}
