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

#[cfg(all(
    any(target_arch = "arm", target_arch = "aarch64"),
    target_feature = "neon"
))]
pub mod neon_support {
    use std::arch::aarch64::*;

    use crate::neon_utils::neon_utils::{load_u8_u16, load_u8_u32_fast};
    use crate::unsafe_slice::UnsafeSlice;

    #[allow(dead_code)]
    pub(crate) fn box_blur_horizontal_pass_neon<const CHANNEL_CONFIGURATION: usize>(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        radius: u32,
        start_y: u32,
        end_y: u32,
    ) {
        let eraser_store: [u32; 4] = [1u32, 1u32, 1u32, 0u32];
        let eraser: uint32x4_t = unsafe { vld1q_u32(eraser_store.as_ptr()) };

        let kernel_scale = 1f32 / (radius * 2) as f32;
        let v_kernel_scale = unsafe { vdupq_n_f32(kernel_scale) };
        let kernel_size = radius * 2 + 1;
        let edge_count = (kernel_size / 2) + 1;
        let v_edge_count = unsafe { vdupq_n_u32(edge_count) };

        let half_kernel = kernel_size / 2;

        for y in start_y..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store: uint32x4_t;
            {
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift) };
                let edge_colors = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr) };
                store = unsafe { vmulq_u32(edge_colors, v_edge_count) };
            }

            for x in 1..std::cmp::min(half_kernel, width) {
                let px = x as usize * CHANNEL_CONFIGURATION;
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                let edge_colors = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                store = unsafe { vaddw_u16(store, edge_colors) };
            }

            for x in 0..width {
                // preload edge pixels

                // subtract previous
                {
                    let previous_x = std::cmp::max(x as i64 - half_kernel as i64, 0) as usize;
                    let previous = previous_x * CHANNEL_CONFIGURATION;
                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + previous) };
                    let edge_colors = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                    store = unsafe { vsubw_u16(store, edge_colors) };
                }

                // add next
                {
                    let next_x = std::cmp::min(x + half_kernel, width - 1) as usize;

                    let next = next_x * CHANNEL_CONFIGURATION;

                    let s_ptr = unsafe { src.as_ptr().add(y_src_shift + next) };
                    let edge_colors = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                    store = unsafe { vaddw_u16(store, edge_colors) };
                }

                let px = x as usize * CHANNEL_CONFIGURATION;

                if CHANNEL_CONFIGURATION == 3 {
                    store = unsafe { vmulq_u32(store, eraser) };
                }

                let scale_store =
                    unsafe { vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(store), v_kernel_scale)) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                let pixel = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };
                let bits = pixel.to_le_bytes();
                unsafe {
                    unsafe_dst.write(y_dst_shift + px, bits[0]);
                    unsafe_dst.write(y_dst_shift + px + 1, bits[1]);
                    unsafe_dst.write(y_dst_shift + px + 2, bits[2]);
                    if CHANNEL_CONFIGURATION == 4 {
                        unsafe_dst.write(y_dst_shift + px + 3, bits[3]);
                    }
                }
            }
        }
    }

    pub(crate) fn box_blur_vertical_pass_neon<const CHANNEL_CONFIGURATION: usize>(
        src: &Vec<u8>,
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        _: u32,
        height: u32,
        radius: u32,
        start_x: u32,
        end_x: u32,
    ) {
        let eraser_store: [u32; 4] = [1u32, 1u32, 1u32, 0u32];
        let eraser: uint32x4_t = unsafe { vld1q_u32(eraser_store.as_ptr()) };

        let kernel_scale = 1f32 / (radius * 2) as f32;
        let v_kernel_scale = unsafe { vdupq_n_f32(kernel_scale) };
        let kernel_size = radius * 2 + 1;
        let edge_count = (kernel_size / 2) + 1;
        let v_edge_count = unsafe { vdupq_n_u32(edge_count) };

        let half_kernel = kernel_size / 2;

        for x in start_x..end_x {
            let px = x as usize * CHANNEL_CONFIGURATION;

            let mut store: uint32x4_t;
            {
                let s_ptr = unsafe { src.as_ptr().add(px) };
                let edge_colors = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr) };
                store = unsafe { vmulq_u32(edge_colors, v_edge_count) };
            }

            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                let edge_colors = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                store = unsafe { vaddw_u16(store, edge_colors) };
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
                    let s_ptr = unsafe { src.as_ptr().add(previous + px) };
                    let edge_colors = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                    store = unsafe { vsubw_u16(store, edge_colors) };
                }

                // add next
                {
                    let s_ptr = unsafe { src.as_ptr().add(next + px) };
                    let edge_colors = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                    store = unsafe { vaddw_u16(store, edge_colors) };
                }

                let px = x as usize * CHANNEL_CONFIGURATION;

                if CHANNEL_CONFIGURATION == 3 {
                    store = unsafe { vmulq_u32(store, eraser) };
                }

                let scale_store =
                    unsafe { vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(store), v_kernel_scale)) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                let pixel = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };
                let bits = pixel.to_le_bytes();
                unsafe {
                    unsafe_dst.write(y_dst_shift + px, bits[0]);
                    unsafe_dst.write(y_dst_shift + px + 1, bits[1]);
                    unsafe_dst.write(y_dst_shift + px + 2, bits[2]);
                    if CHANNEL_CONFIGURATION == 4 {
                        unsafe_dst.write(y_dst_shift + px + 3, bits[3]);
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
    pub(crate) fn box_blur_horizontal_pass_neon(
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
    ) {
    }

    #[allow(dead_code)]
    pub(crate) fn box_blur_vertical_pass_neon(
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
    ) {
    }
}
