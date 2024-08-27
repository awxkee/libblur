/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::gaussian::gaussian_approx::{PRECISION, ROUNDING_APPROX};
use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::neon::load_u8_u16_fast;
use crate::unsafe_slice::UnsafeSlice;
use crate::{
    accumulate_2_forward_approx, accumulate_4_by_4_forward_approx, accumulate_4_forward_approx
    , write_u8_by_channels_approx,
};
use std::arch::aarch64::*;

pub fn gaussian_blur_horizontal_pass_filter_approx_neon<const CHANNEL_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    filter: &Vec<GaussianFilter<i16>>,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
        let shuffle_1 = vld1_u8(shuf_table_1.as_ptr());
        let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
        let shuffle_2 = vld1_u8(shuf_table_2.as_ptr());
        let shuffle = vcombine_u8(shuffle_1, shuffle_2);

        let mut cy = start_y;

        for y in (cy..end_y.saturating_sub(4)).step_by(4) {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store_0 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store_1 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store_2 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store_3 = vdupq_n_s32(ROUNDING_APPROX);

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut j = 0usize;

                if CHANNEL_CONFIGURATION == 4 {
                    while j + 16 < current_filter.size && x as i64 + j as i64 + 16i64 < width as i64
                    {
                        let px = std::cmp::min(
                            std::cmp::max(filter_start as i64 + j as i64, 0),
                            (width - 1) as i64,
                        ) as usize
                            * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_u8_x4(s_ptr);
                        let pixel_colors_1 = vld1q_u8_x4(s_ptr.add(src_stride as usize));
                        let pixel_colors_2 = vld1q_u8_x4(s_ptr.add(src_stride as usize * 2));
                        let pixel_colors_3 = vld1q_u8_x4(s_ptr.add(src_stride as usize * 3));

                        let weight_ptr = filter_weights.as_ptr().add(j);
                        let weights = vld1_s16_x4(weight_ptr);

                        accumulate_4_by_4_forward_approx!(store_0, pixel_colors_0, weights);
                        accumulate_4_by_4_forward_approx!(store_1, pixel_colors_1, weights);
                        accumulate_4_by_4_forward_approx!(store_2, pixel_colors_2, weights);
                        accumulate_4_by_4_forward_approx!(store_3, pixel_colors_3, weights);

                        j += 16;
                    }
                }

                while j + 4 < current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                        < width as i64
                {
                    let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors_0: uint8x16_t = vld1q_u8(s_ptr);
                    let mut pixel_colors_1: uint8x16_t = vld1q_u8(s_ptr.add(src_stride as usize));
                    let mut pixel_colors_2: uint8x16_t =
                        vld1q_u8(s_ptr.add(src_stride as usize * 2));
                    let mut pixel_colors_3: uint8x16_t =
                        vld1q_u8(s_ptr.add(src_stride as usize * 3));

                    let weight_ptr = filter_weights.as_ptr().add(j);
                    let weights = vld1_s16(weight_ptr);

                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors_0 = vqtbl1q_u8(pixel_colors_0, shuffle);
                        pixel_colors_1 = vqtbl1q_u8(pixel_colors_1, shuffle);
                        pixel_colors_2 = vqtbl1q_u8(pixel_colors_2, shuffle);
                        pixel_colors_3 = vqtbl1q_u8(pixel_colors_3, shuffle);
                    }

                    accumulate_4_forward_approx!(store_0, pixel_colors_0, weights);
                    accumulate_4_forward_approx!(store_1, pixel_colors_1, weights);
                    accumulate_4_forward_approx!(store_2, pixel_colors_2, weights);
                    accumulate_4_forward_approx!(store_3, pixel_colors_3, weights);

                    j += 4;
                }

                while j + 2 <= current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 2 } else { 3 })
                        < width as i64
                {
                    let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors_0: uint8x8_t = vld1_u8(s_ptr);
                    let mut pixel_colors_1: uint8x8_t = vld1_u8(s_ptr.add(src_stride as usize));
                    let mut pixel_colors_2: uint8x8_t = vld1_u8(s_ptr.add(src_stride as usize * 2));
                    let mut pixel_colors_3: uint8x8_t = vld1_u8(s_ptr.add(src_stride as usize * 3));

                    let weights_ptr = filter_weights.as_ptr().add(j);
                    let weights = vreinterpretq_s16_s32(vsetq_lane_s32::<0>(
                        (weights_ptr as *const i32).read_unaligned(),
                        vdupq_n_s32(0),
                    ));

                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors_0 = vtbl1_u8(pixel_colors_0, shuffle_1);
                        pixel_colors_1 = vtbl1_u8(pixel_colors_1, shuffle_1);
                        pixel_colors_2 = vtbl1_u8(pixel_colors_2, shuffle_1);
                        pixel_colors_3 = vtbl1_u8(pixel_colors_3, shuffle_1);
                    }

                    accumulate_2_forward_approx!(store_0, pixel_colors_0, weights);
                    accumulate_2_forward_approx!(store_1, pixel_colors_1, weights);
                    accumulate_2_forward_approx!(store_2, pixel_colors_2, weights);
                    accumulate_2_forward_approx!(store_3, pixel_colors_3, weights);

                    j += 2;
                }

                while j < current_filter.size {
                    let current_x = filter_start + j;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = load_u8_u16_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let pixel_colors_1 =
                        load_u8_u16_fast::<CHANNEL_CONFIGURATION>(s_ptr.add(src_stride as usize));
                    let pixel_colors_2 = load_u8_u16_fast::<CHANNEL_CONFIGURATION>(
                        s_ptr.add(src_stride as usize * 2),
                    );
                    let pixel_colors_3 = load_u8_u16_fast::<CHANNEL_CONFIGURATION>(
                        s_ptr.add(src_stride as usize * 3),
                    );
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = vld1_dup_s16(weight);
                    store_0 = vmlal_s16(store_0, vreinterpret_s16_u16(pixel_colors_0), f_weight);
                    store_1 = vmlal_s16(store_1, vreinterpret_s16_u16(pixel_colors_1), f_weight);
                    store_2 = vmlal_s16(store_2, vreinterpret_s16_u16(pixel_colors_2), f_weight);
                    store_3 = vmlal_s16(store_3, vreinterpret_s16_u16(pixel_colors_3), f_weight);

                    j += 1;
                }

                write_u8_by_channels_approx!(
                    store_0,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    y_dst_shift,
                    x
                );
                let off1 = y_dst_shift + src_stride as usize;
                write_u8_by_channels_approx!(store_1, CHANNEL_CONFIGURATION, unsafe_dst, off1, x);
                let off2 = y_dst_shift + src_stride as usize * 2;
                write_u8_by_channels_approx!(store_2, CHANNEL_CONFIGURATION, unsafe_dst, off2, x);
                let off3 = y_dst_shift + src_stride as usize * 3;
                write_u8_by_channels_approx!(store_3, CHANNEL_CONFIGURATION, unsafe_dst, off3, x);
            }
            cy = y;
        }

        for y in (cy..end_y.saturating_sub(2)).step_by(2) {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store_0 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store_1 = vdupq_n_s32(ROUNDING_APPROX);

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut j = 0usize;

                if CHANNEL_CONFIGURATION == 4 {
                    while j + 16 < current_filter.size && x as i64 + j as i64 + 16i64 < width as i64
                    {
                        let px = std::cmp::min(
                            std::cmp::max(filter_start as i64 + j as i64, 0),
                            (width - 1) as i64,
                        ) as usize
                            * CHANNEL_CONFIGURATION;
                        let s_ptr = src.as_ptr().add(y_src_shift + px);
                        let pixel_colors_0 = vld1q_u8_x4(s_ptr);
                        let pixel_colors_1 = vld1q_u8_x4(s_ptr.add(src_stride as usize));

                        let weight_ptr = filter_weights.as_ptr().add(j);
                        let weights = vld1_s16_x4(weight_ptr);

                        accumulate_4_by_4_forward_approx!(store_0, pixel_colors_0, weights);
                        accumulate_4_by_4_forward_approx!(store_1, pixel_colors_1, weights);

                        j += 16;
                    }
                }

                while j + 4 < current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                        < width as i64
                {
                    let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors_0: uint8x16_t = vld1q_u8(s_ptr);
                    let mut pixel_colors_1: uint8x16_t = vld1q_u8(s_ptr.add(src_stride as usize));
                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors_0 = vqtbl1q_u8(pixel_colors_0, shuffle);
                        pixel_colors_1 = vqtbl1q_u8(pixel_colors_1, shuffle);
                    }
                    let weight_ptr = filter_weights.as_ptr().add(j);

                    let weights = vld1_s16(weight_ptr);

                    accumulate_4_forward_approx!(store_0, pixel_colors_0, weights);
                    accumulate_4_forward_approx!(store_1, pixel_colors_1, weights);

                    j += 4;
                }

                while j + 2 <= current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 2 } else { 3 })
                        < width as i64
                {
                    let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors_0: uint8x8_t = vld1_u8(s_ptr);
                    let mut pixel_colors_1: uint8x8_t = vld1_u8(s_ptr.add(src_stride as usize));

                    let weights_ptr = filter_weights.as_ptr().add(j);
                    let weights = vreinterpretq_s16_s32(vsetq_lane_s32::<0>(
                        (weights_ptr as *const i32).read_unaligned(),
                        vdupq_n_s32(0),
                    ));

                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors_0 = vtbl1_u8(pixel_colors_0, shuffle_1);
                        pixel_colors_1 = vtbl1_u8(pixel_colors_1, shuffle_1);
                    }

                    accumulate_2_forward_approx!(store_0, pixel_colors_0, weights);
                    accumulate_2_forward_approx!(store_1, pixel_colors_1, weights);

                    j += 2;
                }

                while j < current_filter.size {
                    let current_x = filter_start + j;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = load_u8_u16_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let pixel_colors_1 =
                        load_u8_u16_fast::<CHANNEL_CONFIGURATION>(s_ptr.add(src_stride as usize));
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = vld1_dup_s16(weight);
                    store_0 = vmlal_s16(store_0, vreinterpret_s16_u16(pixel_colors_0), f_weight);
                    store_1 = vmlal_s16(store_1, vreinterpret_s16_u16(pixel_colors_1), f_weight);

                    j += 1;
                }

                write_u8_by_channels_approx!(
                    store_0,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    y_dst_shift,
                    x
                );
                write_u8_by_channels_approx!(
                    store_1,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    y_dst_shift + src_stride as usize,
                    x
                );
            }
            cy = y;
        }

        for y in cy..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut store = vdupq_n_s32(ROUNDING_APPROX);

                let mut j = 0;

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                while j + 4 < current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 4 } else { 6 })
                        < width as i64
                {
                    let px = (filter_start + j) * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let mut pixel_colors: uint8x16_t = vld1q_u8(s_ptr);
                    let weights_ptr = filter_weights.as_ptr().add(j);

                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_colors = vqtbl1q_u8(pixel_colors, shuffle);
                    }

                    let weights = vld1_s16(weights_ptr);

                    accumulate_4_forward_approx!(store, pixel_colors, weights);

                    j += 4;
                }

                while j + 2 < current_filter.size
                    && filter_start as i64
                        + j as i64
                        + (if CHANNEL_CONFIGURATION == 4 { 2 } else { 3 })
                        < width as i64
                {
                    let current_x = filter_start + j;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let weights_ptr = filter_weights.as_ptr().add(j);
                    let weights = vreinterpretq_s16_s32(vsetq_lane_s32::<0>(
                        (weights_ptr as *const i32).read_unaligned(),
                        vdupq_n_s32(0),
                    ));
                    let mut pixel_color = vld1_u8(s_ptr);
                    if CHANNEL_CONFIGURATION == 3 {
                        pixel_color = vtbl1_u8(pixel_color, shuffle_1);
                    }

                    accumulate_2_forward_approx!(store, pixel_color, weights);

                    j += 2;
                }

                while j < current_filter.size {
                    let current_x = filter_start + j;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors = load_u8_u16_fast::<CHANNEL_CONFIGURATION>(s_ptr);
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = vld1_dup_s16(weight);
                    store = vmlal_s16(store, vreinterpret_s16_u16(pixel_colors), f_weight);

                    j += 1;
                }

                write_u8_by_channels_approx!(
                    store,
                    CHANNEL_CONFIGURATION,
                    unsafe_dst,
                    y_dst_shift,
                    x
                );
            }
        }
    }
}
