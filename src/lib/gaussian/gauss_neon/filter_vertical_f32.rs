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
use std::arch::aarch64::*;

use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::neon::{prefer_vfma_f32, prefer_vfmaq_f32};
use crate::unsafe_slice::UnsafeSlice;

pub fn gaussian_blur_vertical_pass_filter_f32_neon<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    _: u32,
    filter: &Vec<GaussianFilter>,
    start_y: u32,
    end_y: u32,
) {
    let src: &[f32] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, f32> = unsafe { std::mem::transmute(undef_unsafe_dst) };
    let zeros = unsafe { vdupq_n_f32(0f32) };

    let total_size = CHANNEL_CONFIGURATION * width as usize;

    for y in start_y..end_y {
        let y_dst_shift = y as usize * dst_stride as usize;

        let current_filter = unsafe { filter.get_unchecked(y as usize) };
        let filter_start = current_filter.start;
        let filter_weights = &current_filter.filter;

        let mut cx = 0usize;

        unsafe {
            while cx + 32 < total_size {
                let mut store0 = zeros;
                let mut store1 = zeros;
                let mut store2 = zeros;
                let mut store3 = zeros;
                let mut store4 = zeros;
                let mut store5 = zeros;
                let mut store6 = zeros;
                let mut store7 = zeros;

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let quadruple_0 = vld1q_f32_x4(s_ptr);
                    let quadruple_1 = vld1q_f32_x4(s_ptr.add(16));
                    store0 = prefer_vfmaq_f32(store0, quadruple_0.0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, quadruple_0.1, f_weight);
                    store2 = prefer_vfmaq_f32(store2, quadruple_0.2, f_weight);
                    store3 = prefer_vfmaq_f32(store3, quadruple_0.3, f_weight);
                    store4 = prefer_vfmaq_f32(store4, quadruple_1.0, f_weight);
                    store5 = prefer_vfmaq_f32(store5, quadruple_1.1, f_weight);
                    store6 = prefer_vfmaq_f32(store6, quadruple_1.2, f_weight);
                    store7 = prefer_vfmaq_f32(store7, quadruple_1.3, f_weight);

                    j += 1;
                }

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut f32;

                vst1q_f32_x4(dst_ptr, float32x4x4_t(store0, store1, store2, store3));
                vst1q_f32_x4(
                    dst_ptr.add(16),
                    float32x4x4_t(store4, store5, store6, store7),
                );

                cx += 32;
            }

            while cx + 16 < total_size {
                let mut store0: float32x4_t = zeros;
                let mut store1: float32x4_t = zeros;
                let mut store2: float32x4_t = zeros;
                let mut store3: float32x4_t = zeros;

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let quadruple = vld1q_f32_x4(s_ptr);
                    store0 = prefer_vfmaq_f32(store0, quadruple.0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, quadruple.1, f_weight);
                    store2 = prefer_vfmaq_f32(store2, quadruple.2, f_weight);
                    store3 = prefer_vfmaq_f32(store3, quadruple.3, f_weight);

                    j += 1;
                }

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut f32;
                vst1q_f32_x4(dst_ptr, float32x4x4_t(store0, store1, store2, store3));

                cx += 16;
            }

            while cx + 8 < total_size {
                let mut store0: float32x4_t = zeros;
                let mut store1: float32x4_t = zeros;

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pairs = vld1q_f32_x2(s_ptr);
                    store0 = prefer_vfmaq_f32(store0, pairs.0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pairs.1, f_weight);

                    j += 1;
                }

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut f32;
                vst1q_f32_x2(dst_ptr, float32x4x2_t(store0, store1));

                cx += 8;
            }

            while cx + 4 < total_size {
                let mut store0: float32x4_t = zeros;

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixel = vld1q_f32(s_ptr);
                    store0 = prefer_vfmaq_f32(store0, pixel, f_weight);

                    j += 1;
                }

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut f32;
                vst1q_f32(dst_ptr, store0);

                cx += 4;
            }

            while cx < total_size {
                let mut store0 = vdup_n_f32(0f32);

                let mut j = 0usize;
                while j < current_filter.size {
                    let weight = *filter_weights.get_unchecked(j);
                    let f_weight = vdup_n_f32(weight);

                    let py = filter_start + j;
                    let y_src_shift = py * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let px = s_ptr.read_unaligned().to_bits() as u64;
                    let pixel = vcreate_f32(px);
                    store0 = prefer_vfma_f32(store0, pixel, f_weight);

                    j += 1;
                }

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut f32;

                let pixel = vget_lane_f32::<0>(store0);
                dst_ptr.write_unaligned(pixel);

                cx += 1;
            }
        }
    }
}
