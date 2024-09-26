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
use crate::neon::{load_u8_u16_fast, load_u8_u16_one};
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;

pub fn gaussian_blur_vertical_pass_filter_approx_neon<const CHANNEL_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    _: u32,
    filter: &[GaussianFilter<i16>],
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let zeros = vdupq_n_s32(0);

        let total_size = CHANNEL_CONFIGURATION * width as usize;

        for y in start_y..end_y {
            let y_dst_shift = y as usize * dst_stride as usize;

            let current_filter = filter.get_unchecked(y as usize);
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            let mut cx = 0usize;

            while cx + 32 < total_size {
                let mut store0 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store1 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store2 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store3 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store4 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store5 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store6 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store7 = vdupq_n_s32(ROUNDING_APPROX);

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = vld1q_dup_s16(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8x2 = vld1q_u8_x2(s_ptr);
                    let hi_16 = vreinterpretq_s16_u16(vmovl_high_u8(pixels_u8x2.0));
                    let lo_16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pixels_u8x2.0)));
                    store0 = vmlal_s16(store0, vget_low_s16(lo_16), vget_low_s16(f_weight));
                    store1 = vmlal_high_s16(store1, lo_16, f_weight);
                    store2 = vmlal_s16(store2, vget_low_s16(hi_16), vget_low_s16(f_weight));
                    store3 = vmlal_high_s16(store3, hi_16, f_weight);

                    let hi_16 = vreinterpretq_s16_u16(vmovl_high_u8(pixels_u8x2.1));
                    let lo_16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pixels_u8x2.1)));
                    store4 = vmlal_s16(store4, vget_low_s16(lo_16), vget_low_s16(f_weight));
                    store5 = vmlal_high_s16(store5, lo_16, f_weight);
                    store6 = vmlal_s16(store6, vget_low_s16(hi_16), vget_low_s16(f_weight));
                    store7 = vmlal_high_s16(store7, hi_16, f_weight);

                    j += 1;
                }

                let store_0 = vshrn_n_s32::<PRECISION>(store0);
                let store_1 = vshrn_n_s32::<PRECISION>(store1);
                let store_2 = vshrn_n_s32::<PRECISION>(store2);
                let store_3 = vshrn_n_s32::<PRECISION>(store3);
                let store_4 = vshrn_n_s32::<PRECISION>(store4);
                let store_5 = vshrn_n_s32::<PRECISION>(store5);
                let store_6 = vshrn_n_s32::<PRECISION>(store6);
                let store_7 = vshrn_n_s32::<PRECISION>(store7);

                let zeros = vdupq_n_s16(0);
                let store_lo =
                    vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(store_0, store_1), zeros));
                let store_hi =
                    vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(store_2, store_3), zeros));
                let store_x = vcombine_u8(vqmovn_u16(store_lo), vqmovn_u16(store_hi));

                let store_lo =
                    vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(store_4, store_5), zeros));
                let store_hi =
                    vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(store_6, store_7), zeros));
                let store_k = vcombine_u8(vqmovn_u16(store_lo), vqmovn_u16(store_hi));

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                let store = uint8x16x2_t(store_x, store_k);
                vst1q_u8_x2(dst_ptr, store);

                cx += 32;
            }

            while cx + 16 < total_size {
                let mut store0 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store1 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store2 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store3 = vdupq_n_s32(ROUNDING_APPROX);

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = vld1q_dup_s16(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = vld1q_u8(s_ptr);
                    let hi_16 = vreinterpretq_s16_u16(vmovl_high_u8(pixels_u8));
                    let lo_16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pixels_u8)));
                    store0 = vmlal_s16(store0, vget_low_s16(lo_16), vget_low_s16(f_weight));
                    store1 = vmlal_high_s16(store1, lo_16, f_weight);
                    store2 = vmlal_s16(store2, vget_low_s16(hi_16), vget_low_s16(f_weight));
                    store3 = vmlal_high_s16(store3, hi_16, f_weight);

                    j += 1;
                }

                let store_0 = vshrn_n_s32::<PRECISION>(store0);
                let store_1 = vshrn_n_s32::<PRECISION>(store1);
                let store_2 = vshrn_n_s32::<PRECISION>(store2);
                let store_3 = vshrn_n_s32::<PRECISION>(store3);

                let zeros = vdupq_n_s16(0);
                let store_lo = vmaxq_s16(vcombine_s16(store_0, store_1), zeros);
                let store_hi = vmaxq_s16(vcombine_s16(store_2, store_3), zeros);
                let store = vcombine_u8(
                    vqmovn_u16(vreinterpretq_u16_s16(store_lo)),
                    vqmovn_u16(vreinterpretq_u16_s16(store_hi)),
                );

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                vst1q_u8(dst_ptr, store);

                cx += 16;
            }

            while cx + 8 < total_size {
                let mut store0 = vdupq_n_s32(ROUNDING_APPROX);
                let mut store1 = vdupq_n_s32(ROUNDING_APPROX);

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = vld1q_dup_s16(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = vld1_u8(s_ptr);
                    let pixels_u16 = vreinterpretq_s16_u16(vmovl_u8(pixels_u8));
                    store0 = vmlal_s16(store0, vget_low_s16(pixels_u16), vget_low_s16(f_weight));
                    store1 = vmlal_high_s16(store1, pixels_u16, f_weight);

                    j += 1;
                }

                let store_0 = vshrn_n_s32::<PRECISION>(store0);
                let store_1 = vshrn_n_s32::<PRECISION>(store1);

                let i16_stat = vcombine_s16(store_0, store_1);

                let store_lo = vreinterpretq_u16_s16(vmaxq_s16(i16_stat, vdupq_n_s16(0)));
                let store = vqmovn_u16(store_lo);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                vst1_u8(dst_ptr, store);
                cx += 8;
            }

            while cx + 4 < total_size {
                let mut store0 = vdupq_n_s32(ROUNDING_APPROX);

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = vld1_dup_s16(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u32 = vreinterpret_s16_u16(load_u8_u16_fast::<4>(s_ptr));
                    store0 = vmlal_s16(store0, pixels_u32, f_weight);

                    j += 1;
                }

                store0 = vmaxq_s32(store0, zeros);
                let saturated_store = vshrn_n_s32::<PRECISION>(store0);
                let store = vreinterpretq_u16_s16(vcombine_s16(saturated_store, saturated_store));
                let store_u8 = vqmovn_u16(store);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u32;

                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_u8));
                dst_ptr.write_unaligned(pixel);

                cx += 4;
            }

            while cx < total_size {
                let mut store0 = vdupq_n_s32(ROUNDING_APPROX);

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = filter_weights.as_ptr().add(j);
                    let f_weight = vld1_dup_s16(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixel_s16 = load_u8_u16_one(s_ptr);
                    store0 = vmlal_s16(store0, vreinterpret_s16_u16(pixel_s16), f_weight);

                    j += 1;
                }

                store0 = vmaxq_s32(store0, zeros);
                let saturated_store = vshrn_n_s32::<PRECISION>(store0);
                let store = vreinterpretq_u16_s16(vcombine_s16(saturated_store, saturated_store));
                let store_u8 = vmovn_u16(store);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                let pixel = vget_lane_u8::<0>(store_u8);
                dst_ptr.write_unaligned(pixel);

                cx += 1;
            }
        }
    }
}
