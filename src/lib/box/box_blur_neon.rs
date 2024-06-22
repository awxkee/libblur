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
    use crate::neon::{load_u8_u16, load_u8_u32_fast, vmulq_u32_f32};
    use std::arch::aarch64::*;

    use crate::unsafe_slice::UnsafeSlice;

    #[allow(dead_code)]
    pub(crate) fn box_blur_horizontal_pass_neon<T, const CHANNEL_CONFIGURATION: usize>(
        undefined_src: &[T],
        src_stride: u32,
        undefined_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        width: u32,
        radius: u32,
        start_y: u32,
        end_y: u32,
    ) {
        let src: &[u8] = unsafe { std::mem::transmute(undefined_src) };
        let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undefined_dst) };
        let eraser_store: [u32; 4] = [1u32, 1u32, 1u32, 0u32];
        let eraser: uint32x4_t = unsafe { vld1q_u32(eraser_store.as_ptr()) };

        let kernel_size = radius * 2 + 1;
        let edge_count = (kernel_size / 2) + 1;
        let v_edge_count = unsafe { vdupq_n_u32(edge_count) };

        let v_weight = unsafe { vdupq_n_f32(1f32 / (radius * 2) as f32) };

        let half_kernel = kernel_size / 2;

        let mut yy = start_y;

        for y in (yy..end_y.saturating_sub(4)).step_by(4) {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store_0: uint32x4_t;
            let mut store_1: uint32x4_t;
            let mut store_2: uint32x4_t;
            let mut store_3: uint32x4_t;
            {
                let s_ptr_0 = unsafe { src.as_ptr().add(y_src_shift) };
                let edge_colors_0 = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr_0) };
                store_0 = unsafe { vmulq_u32(edge_colors_0, v_edge_count) };

                let s_ptr_1 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize) };
                let edge_colors_1 = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr_1) };
                store_1 = unsafe { vmulq_u32(edge_colors_1, v_edge_count) };

                let s_ptr_2 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize * 2) };
                let edge_colors_2 = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr_2) };
                store_2 = unsafe { vmulq_u32(edge_colors_2, v_edge_count) };

                let s_ptr_3 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize * 3) };
                let edge_colors_3 = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr_3) };
                store_3 = unsafe { vmulq_u32(edge_colors_3, v_edge_count) };
            }

            for x in 1..std::cmp::min(half_kernel, width) {
                let px = x as usize * CHANNEL_CONFIGURATION;

                let s_ptr_0 = unsafe { src.as_ptr().add(y_src_shift + px) };
                let edge_colors_0 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_0) };
                store_0 = unsafe { vaddw_u16(store_0, edge_colors_0) };

                let s_ptr_1 = unsafe { src.as_ptr().add(y_src_shift + src_stride as usize + px) };
                let edge_colors_1 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_1) };
                store_1 = unsafe { vaddw_u16(store_1, edge_colors_1) };

                let s_ptr_2 =
                    unsafe { src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px) };
                let edge_colors_2 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_2) };
                store_2 = unsafe { vaddw_u16(store_2, edge_colors_2) };

                let s_ptr_3 =
                    unsafe { src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px) };
                let edge_colors_3 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_3) };
                store_3 = unsafe { vaddw_u16(store_3, edge_colors_3) };
            }

            for x in 0..width {
                // preload edge pixels

                // subtract previous
                {
                    let previous_x = std::cmp::max(x as i64 - half_kernel as i64, 0) as usize;
                    let previous = previous_x * CHANNEL_CONFIGURATION;

                    let s_ptr_0 = unsafe { src.as_ptr().add(y_src_shift + previous) };
                    let edge_colors_0 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_0) };
                    store_0 = unsafe { vsubw_u16(store_0, edge_colors_0) };

                    let s_ptr_1 = unsafe {
                        src.as_ptr()
                            .add(y_src_shift + src_stride as usize + previous)
                    };
                    let edge_colors_1 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_1) };
                    store_1 = unsafe { vsubw_u16(store_1, edge_colors_1) };

                    let s_ptr_2 = unsafe {
                        src.as_ptr()
                            .add(y_src_shift + src_stride as usize * 2 + previous)
                    };
                    let edge_colors_2 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_2) };
                    store_2 = unsafe { vsubw_u16(store_2, edge_colors_2) };

                    let s_ptr_3 = unsafe {
                        src.as_ptr()
                            .add(y_src_shift + src_stride as usize * 3 + previous)
                    };
                    let edge_colors_3 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_3) };
                    store_3 = unsafe { vsubw_u16(store_3, edge_colors_3) };
                }

                // add next
                {
                    let next_x = std::cmp::min(x + half_kernel, width - 1) as usize;

                    let next = next_x * CHANNEL_CONFIGURATION;

                    let s_ptr_0 = unsafe { src.as_ptr().add(y_src_shift + next) };
                    let edge_colors_0 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_0) };
                    store_0 = unsafe { vaddw_u16(store_0, edge_colors_0) };

                    let s_ptr_1 =
                        unsafe { src.as_ptr().add(y_src_shift + src_stride as usize + next) };
                    let edge_colors_1 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_1) };
                    store_1 = unsafe { vaddw_u16(store_1, edge_colors_1) };

                    let s_ptr_2 = unsafe {
                        src.as_ptr()
                            .add(y_src_shift + src_stride as usize * 2 + next)
                    };
                    let edge_colors_2 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_2) };
                    store_2 = unsafe { vaddw_u16(store_2, edge_colors_2) };

                    let s_ptr_3 = unsafe {
                        src.as_ptr()
                            .add(y_src_shift + src_stride as usize * 3 + next)
                    };
                    let edge_colors_3 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr_3) };
                    store_3 = unsafe { vaddw_u16(store_3, edge_colors_3) };
                }

                let px = x as usize * CHANNEL_CONFIGURATION;

                if CHANNEL_CONFIGURATION == 3 {
                    store_0 = unsafe { vmulq_u32(store_0, eraser) };
                    store_1 = unsafe { vmulq_u32(store_1, eraser) };
                    store_2 = unsafe { vmulq_u32(store_2, eraser) };
                    store_3 = unsafe { vmulq_u32(store_3, eraser) };
                }

                let scale_store = unsafe { vmulq_u32_f32(store_0, v_weight) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                let pixel_0 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

                let scale_store = unsafe { vmulq_u32_f32(store_1, v_weight) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                let pixel_1 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

                let scale_store = unsafe { vmulq_u32_f32(store_2, v_weight) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                let pixel_2 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

                let scale_store = unsafe { vmulq_u32_f32(store_3, v_weight) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                let pixel_3 = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

                let bytes_offset_0 = y_dst_shift + px;
                let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
                let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
                let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;
                if CHANNEL_CONFIGURATION == 4 {
                    unsafe {
                        let dst_ptr_0 = unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut u32;
                        dst_ptr_0.write_unaligned(pixel_0);

                        let dst_ptr_1 = unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut u32;
                        dst_ptr_1.write_unaligned(pixel_1);

                        let dst_ptr_2 = unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut u32;
                        dst_ptr_2.write_unaligned(pixel_2);

                        let dst_ptr_3 = unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut u32;
                        dst_ptr_3.write_unaligned(pixel_3);
                    }
                } else {
                    let bits_0 = pixel_0.to_le_bytes();
                    unsafe {
                        unsafe_dst.write(bytes_offset_0, bits_0[0]);
                        unsafe_dst.write(bytes_offset_0 + 1, bits_0[1]);
                        unsafe_dst.write(bytes_offset_0 + 2, bits_0[2]);
                    }

                    let bits_1 = pixel_1.to_le_bytes();
                    unsafe {
                        unsafe_dst.write(bytes_offset_1, bits_1[0]);
                        unsafe_dst.write(bytes_offset_1 + 1, bits_1[1]);
                        unsafe_dst.write(bytes_offset_1 + 2, bits_1[2]);
                    }

                    let bits_2 = pixel_2.to_le_bytes();
                    unsafe {
                        unsafe_dst.write(bytes_offset_2, bits_2[0]);
                        unsafe_dst.write(bytes_offset_2 + 1, bits_2[1]);
                        unsafe_dst.write(bytes_offset_2 + 2, bits_2[2]);
                    }

                    let bits_3 = pixel_3.to_le_bytes();
                    unsafe {
                        unsafe_dst.write(bytes_offset_3, bits_3[0]);
                        unsafe_dst.write(bytes_offset_3 + 1, bits_3[1]);
                        unsafe_dst.write(bytes_offset_3 + 2, bits_3[2]);
                    }
                }
            }

            yy = y;
        }

        for y in yy..end_y {
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

                let scale_store = unsafe { vmulq_u32_f32(store, v_weight) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                let pixel = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };

                let bytes_offset = y_dst_shift + px;
                if CHANNEL_CONFIGURATION == 4 {
                    unsafe {
                        let dst_ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u32;
                        dst_ptr.write_unaligned(pixel);
                    }
                } else {
                    let bits = pixel.to_le_bytes();
                    unsafe {
                        unsafe_dst.write(bytes_offset, bits[0]);
                        unsafe_dst.write(bytes_offset + 1, bits[1]);
                        unsafe_dst.write(bytes_offset + 2, bits[2]);
                    }
                }
            }
        }
    }

    pub(crate) fn box_blur_vertical_pass_neon<T, const CHANNEL_CONFIGURATION: usize>(
        undefined_src: &[T],
        src_stride: u32,
        undefined_unsafe_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        _: u32,
        height: u32,
        radius: u32,
        start_x: u32,
        end_x: u32,
    ) {
        let src: &[u8] = unsafe { std::mem::transmute(undefined_src) };
        let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undefined_unsafe_dst) };
        let eraser_store: [u32; 4] = [1u32, 1u32, 1u32, 0u32];
        let eraser: uint32x4_t = unsafe { vld1q_u32(eraser_store.as_ptr()) };

        let kernel_size = radius * 2 + 1;
        let edge_count = (kernel_size / 2) + 1;
        let v_edge_count = unsafe { vdupq_n_u32(edge_count) };

        let half_kernel = kernel_size / 2;

        let v_weight = unsafe { vdupq_n_f32(1f32 / (radius * 2) as f32) };

        let mut cx = start_x;

        for x in (start_x..end_x.saturating_sub(2)).step_by(2) {
            let px = x as usize * CHANNEL_CONFIGURATION;

            let mut store_0: uint32x4_t;
            let mut store_1: uint32x4_t;
            {
                let s_ptr = unsafe { src.as_ptr().add(px) };
                let edge_colors_0 = unsafe { load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr) };
                let edge_colors_1 = unsafe {
                    load_u8_u32_fast::<CHANNEL_CONFIGURATION>(s_ptr.add(CHANNEL_CONFIGURATION))
                };
                store_0 = unsafe { vmulq_u32(edge_colors_0, v_edge_count) };
                store_1 = unsafe { vmulq_u32(edge_colors_1, v_edge_count) };
            }

            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
                let s_ptr = unsafe { src.as_ptr().add(y_src_shift + px) };
                let edge_colors_0 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                let edge_colors_1 = unsafe {
                    load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr.add(CHANNEL_CONFIGURATION))
                };
                store_0 = unsafe { vaddw_u16(store_0, edge_colors_0) };
                store_1 = unsafe { vaddw_u16(store_1, edge_colors_1) };
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
                    let edge_colors_0 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                    let edge_colors_1 = unsafe {
                        load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr.add(CHANNEL_CONFIGURATION))
                    };
                    store_0 = unsafe { vsubw_u16(store_0, edge_colors_0) };
                    store_1 = unsafe { vsubw_u16(store_1, edge_colors_1) };
                }

                // add next
                {
                    let s_ptr = unsafe { src.as_ptr().add(next + px) };
                    let edge_colors_0 = unsafe { load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr) };
                    let edge_colors_1 = unsafe {
                        load_u8_u16::<CHANNEL_CONFIGURATION>(s_ptr.add(CHANNEL_CONFIGURATION))
                    };
                    store_0 = unsafe { vaddw_u16(store_0, edge_colors_0) };
                    store_1 = unsafe { vaddw_u16(store_1, edge_colors_1) };
                }

                let px = x as usize * CHANNEL_CONFIGURATION;

                if CHANNEL_CONFIGURATION == 3 {
                    store_0 = unsafe { vmulq_u32(store_0, eraser) };
                    store_1 = unsafe { vmulq_u32(store_1, eraser) };
                }

                let scale_store_0 = unsafe { vmulq_u32_f32(store_0, v_weight) };
                let scale_store_1 = unsafe { vmulq_u32_f32(store_1, v_weight) };
                if CHANNEL_CONFIGURATION == 3 {
                    let px_16 = unsafe { vqmovn_u32(scale_store_0) };
                    let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                    let pixel = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };
                    let pixel_bits_0 = pixel.to_le_bytes();

                    let px_16 = unsafe { vqmovn_u32(scale_store_1) };
                    let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                    let pixel = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };
                    let pixel_bits_1 = pixel.to_le_bytes();

                    unsafe {
                        let offset = y_dst_shift + px;
                        unsafe_dst.write(offset, pixel_bits_0[0]);
                        unsafe_dst.write(offset + 1, pixel_bits_0[1]);
                        unsafe_dst.write(offset + 2, pixel_bits_0[2]);
                        if CHANNEL_CONFIGURATION == 4 {
                            unsafe_dst.write(offset + 3, pixel_bits_0[3]);
                        }
                    }

                    unsafe {
                        let offset = y_dst_shift + px + CHANNEL_CONFIGURATION;
                        unsafe_dst.write(offset, pixel_bits_1[0]);
                        unsafe_dst.write(offset + 1, pixel_bits_1[1]);
                        unsafe_dst.write(offset + 2, pixel_bits_1[2]);
                        if CHANNEL_CONFIGURATION == 4 {
                            unsafe_dst.write(offset + 3, pixel_bits_1[3]);
                        }
                    }
                } else {
                    unsafe {
                        let offset = y_dst_shift + px;
                        let ptr = unsafe_dst.slice.get_unchecked(offset).get();
                        let px_16_lo = vqmovn_u32(scale_store_0);
                        let px_16_hi = vqmovn_u32(scale_store_1);
                        let px = vqmovn_u16(vcombine_u16(px_16_lo, px_16_hi));
                        vst1_u8(ptr, px);
                    }
                }
            }

            cx = x;
        }

        for x in cx..end_x {
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

                let scale_store = unsafe { vmulq_u32_f32(store, v_weight) };
                let px_16 = unsafe { vqmovn_u32(scale_store) };
                let px_8 = unsafe { vqmovn_u16(vcombine_u16(px_16, px_16)) };
                let pixel = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(px_8)) };
                let bytes_offset = y_dst_shift + px;
                if CHANNEL_CONFIGURATION == 4 {
                    unsafe {
                        let dst_ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u32;
                        dst_ptr.write_unaligned(pixel);
                    }
                } else {
                    let bits = pixel.to_le_bytes();
                    unsafe {
                        unsafe_dst.write(bytes_offset, bits[0]);
                        unsafe_dst.write(bytes_offset + 1, bits[1]);
                        unsafe_dst.write(bytes_offset + 2, bits[2]);
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
        _src: &[u8],
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
        _src: &[u8],
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
