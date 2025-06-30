/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
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

use crate::box_filter::avx::vrgba8_low_radius::box_blur_vertical_pass_avx2_def_lr;
use crate::sse::{load_u8_s32_fast, store_u8_u32};
use crate::unsafe_slice::UnsafeSlice;
use std::arch::x86_64::*;

pub(crate) fn box_blur_vertical_pass_avx2(
    src: &[u8],
    src_stride: u32,
    dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    w: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    unsafe {
        if radius < 119 {
            box_blur_vertical_pass_avx2_def_lr(
                src, src_stride, dst, dst_stride, w, height, radius, start_x, end_x,
            )
        } else {
            box_blur_vertical_pass_avx2_def(
                src, src_stride, dst, dst_stride, w, height, radius, start_x, end_x,
            )
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn box_blur_vertical_pass_avx2_def(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    _: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let half_kernel = kernel_size / 2;

    let mut cx = start_x;

    let v_edge_count = _mm256_set1_epi32(edge_count as i32);
    let v_weight = _mm256_set1_ps(1f32 / (radius * 2 + 1) as f32);

    while cx + 64 < end_x {
        let px = cx as usize;

        let mut store_0: __m256i;
        let mut store_1: __m256i;
        let mut store_2: __m256i;
        let mut store_3: __m256i;
        let mut store_4: __m256i;
        let mut store_5: __m256i;
        let mut store_6: __m256i;
        let mut store_7: __m256i;

        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge0 = _mm256_loadu_si256(s_ptr as *const _);
            let edge1 = _mm256_loadu_si256(s_ptr.add(32) as *const _);
            let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
            let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

            let lo1 = _mm256_unpacklo_epi8(edge1, _mm256_setzero_si256());
            let hi1 = _mm256_unpackhi_epi8(edge1, _mm256_setzero_si256());

            let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
            let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
            let i16_l1 = _mm256_unpacklo_epi16(hi0, _mm256_setzero_si256());
            let i16_h1 = _mm256_unpackhi_epi16(hi0, _mm256_setzero_si256());

            let i16_l01 = _mm256_unpacklo_epi16(lo1, _mm256_setzero_si256());
            let i16_h01 = _mm256_unpackhi_epi16(lo1, _mm256_setzero_si256());
            let i16_l11 = _mm256_unpacklo_epi16(hi1, _mm256_setzero_si256());
            let i16_h11 = _mm256_unpackhi_epi16(hi1, _mm256_setzero_si256());

            store_0 = _mm256_madd_epi16(i16_l0, v_edge_count);
            store_1 = _mm256_madd_epi16(i16_h0, v_edge_count);

            store_2 = _mm256_madd_epi16(i16_l1, v_edge_count);
            store_3 = _mm256_madd_epi16(i16_h1, v_edge_count);

            store_4 = _mm256_madd_epi16(i16_l01, v_edge_count);
            store_5 = _mm256_madd_epi16(i16_h01, v_edge_count);

            store_6 = _mm256_madd_epi16(i16_l11, v_edge_count);
            store_7 = _mm256_madd_epi16(i16_h11, v_edge_count);
        }

        unsafe {
            for y in 1..=half_kernel as usize {
                let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge0 = _mm256_loadu_si256(s_ptr as *const _);
                let edge1 = _mm256_loadu_si256(s_ptr.add(32) as *const _);
                let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
                let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

                let lo1 = _mm256_unpacklo_epi8(edge1, _mm256_setzero_si256());
                let hi1 = _mm256_unpackhi_epi8(edge1, _mm256_setzero_si256());

                let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
                let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
                let i16_l1 = _mm256_unpacklo_epi16(hi0, _mm256_setzero_si256());
                let i16_h1 = _mm256_unpackhi_epi16(hi0, _mm256_setzero_si256());

                let i16_l01 = _mm256_unpacklo_epi16(lo1, _mm256_setzero_si256());
                let i16_h01 = _mm256_unpackhi_epi16(lo1, _mm256_setzero_si256());
                let i16_l11 = _mm256_unpacklo_epi16(hi1, _mm256_setzero_si256());
                let i16_h11 = _mm256_unpackhi_epi16(hi1, _mm256_setzero_si256());

                store_0 = _mm256_add_epi32(store_0, i16_l0);
                store_1 = _mm256_add_epi32(store_1, i16_h0);

                store_2 = _mm256_add_epi32(store_2, i16_l1);
                store_3 = _mm256_add_epi32(store_3, i16_h1);

                store_4 = _mm256_add_epi32(store_4, i16_l01);
                store_5 = _mm256_add_epi32(store_5, i16_h01);

                store_6 = _mm256_add_epi32(store_6, i16_l11);
                store_7 = _mm256_add_epi32(store_7, i16_h11);
            }
        }

        unsafe {
            for y in 0..height {
                // preload edge pixels
                let next =
                    std::cmp::min(y + half_kernel + 1, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                {
                    let px = cx as usize;

                    let store0_ps = _mm256_cvtepi32_ps(store_0);
                    let store1_ps = _mm256_cvtepi32_ps(store_1);
                    let store2_ps = _mm256_cvtepi32_ps(store_2);
                    let store3_ps = _mm256_cvtepi32_ps(store_3);
                    let store4_ps = _mm256_cvtepi32_ps(store_4);
                    let store5_ps = _mm256_cvtepi32_ps(store_5);
                    let store6_ps = _mm256_cvtepi32_ps(store_6);
                    let store7_ps = _mm256_cvtepi32_ps(store_7);

                    let r0 = _mm256_mul_ps(store0_ps, v_weight);
                    let r1 = _mm256_mul_ps(store1_ps, v_weight);
                    let r2 = _mm256_mul_ps(store2_ps, v_weight);
                    let r3 = _mm256_mul_ps(store3_ps, v_weight);
                    let r4 = _mm256_mul_ps(store4_ps, v_weight);
                    let r5 = _mm256_mul_ps(store5_ps, v_weight);
                    let r6 = _mm256_mul_ps(store6_ps, v_weight);
                    let r7 = _mm256_mul_ps(store7_ps, v_weight);

                    let scale_store_0 = _mm256_cvtps_epi32(r0);
                    let scale_store_1 = _mm256_cvtps_epi32(r1);
                    let scale_store_2 = _mm256_cvtps_epi32(r2);
                    let scale_store_3 = _mm256_cvtps_epi32(r3);
                    let scale_store_4 = _mm256_cvtps_epi32(r4);
                    let scale_store_5 = _mm256_cvtps_epi32(r5);
                    let scale_store_6 = _mm256_cvtps_epi32(r6);
                    let scale_store_7 = _mm256_cvtps_epi32(r7);

                    let offset = y_dst_shift + px;
                    let ptr = unsafe_dst.slice.get_unchecked(offset).get();

                    let set0 = _mm256_packus_epi32(scale_store_0, scale_store_1);
                    let set1 = _mm256_packus_epi32(scale_store_2, scale_store_3);
                    let set2 = _mm256_packus_epi32(scale_store_4, scale_store_5);
                    let set3 = _mm256_packus_epi32(scale_store_6, scale_store_7);

                    let full_set0 = _mm256_packus_epi16(set0, set1);
                    let full_set1 = _mm256_packus_epi16(set2, set3);
                    _mm256_storeu_si256(ptr as *mut _, full_set0);
                    _mm256_storeu_si256(ptr.add(32) as *mut _, full_set1);
                }

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge0 = _mm256_loadu_si256(s_ptr as *const _);
                    let edge1 = _mm256_loadu_si256(s_ptr.add(32) as *const _);
                    let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
                    let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

                    let lo1 = _mm256_unpacklo_epi8(edge1, _mm256_setzero_si256());
                    let hi1 = _mm256_unpackhi_epi8(edge1, _mm256_setzero_si256());

                    let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
                    let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
                    let i16_l1 = _mm256_unpacklo_epi16(hi0, _mm256_setzero_si256());
                    let i16_h1 = _mm256_unpackhi_epi16(hi0, _mm256_setzero_si256());

                    let i16_l01 = _mm256_unpacklo_epi16(lo1, _mm256_setzero_si256());
                    let i16_h01 = _mm256_unpackhi_epi16(lo1, _mm256_setzero_si256());
                    let i16_l11 = _mm256_unpacklo_epi16(hi1, _mm256_setzero_si256());
                    let i16_h11 = _mm256_unpackhi_epi16(hi1, _mm256_setzero_si256());

                    store_0 = _mm256_sub_epi32(store_0, i16_l0);
                    store_1 = _mm256_sub_epi32(store_1, i16_h0);

                    store_2 = _mm256_sub_epi32(store_2, i16_l1);
                    store_3 = _mm256_sub_epi32(store_3, i16_h1);

                    store_4 = _mm256_sub_epi32(store_4, i16_l01);
                    store_5 = _mm256_sub_epi32(store_5, i16_h01);

                    store_6 = _mm256_sub_epi32(store_6, i16_l11);
                    store_7 = _mm256_sub_epi32(store_7, i16_h11);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge0 = _mm256_loadu_si256(s_ptr as *const _);
                    let edge1 = _mm256_loadu_si256(s_ptr.add(32) as *const _);
                    let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
                    let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

                    let lo1 = _mm256_unpacklo_epi8(edge1, _mm256_setzero_si256());
                    let hi1 = _mm256_unpackhi_epi8(edge1, _mm256_setzero_si256());

                    let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
                    let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
                    let i16_l1 = _mm256_unpacklo_epi16(hi0, _mm256_setzero_si256());
                    let i16_h1 = _mm256_unpackhi_epi16(hi0, _mm256_setzero_si256());

                    let i16_l01 = _mm256_unpacklo_epi16(lo1, _mm256_setzero_si256());
                    let i16_h01 = _mm256_unpackhi_epi16(lo1, _mm256_setzero_si256());
                    let i16_l11 = _mm256_unpacklo_epi16(hi1, _mm256_setzero_si256());
                    let i16_h11 = _mm256_unpackhi_epi16(hi1, _mm256_setzero_si256());

                    store_0 = _mm256_add_epi32(store_0, i16_l0);
                    store_1 = _mm256_add_epi32(store_1, i16_h0);

                    store_2 = _mm256_add_epi32(store_2, i16_l1);
                    store_3 = _mm256_add_epi32(store_3, i16_h1);

                    store_4 = _mm256_add_epi32(store_4, i16_l01);
                    store_5 = _mm256_add_epi32(store_5, i16_h01);

                    store_6 = _mm256_add_epi32(store_6, i16_l11);
                    store_7 = _mm256_add_epi32(store_7, i16_h11);
                }
            }
        }

        cx += 64;
    }

    while cx + 32 < end_x {
        let px = cx as usize;

        let mut store_0: __m256i;
        let mut store_1: __m256i;
        let mut store_2: __m256i;
        let mut store_3: __m256i;

        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge = _mm256_loadu_si256(s_ptr as *const _);
            let lo0 = _mm256_unpacklo_epi8(edge, _mm256_setzero_si256());
            let hi0 = _mm256_unpackhi_epi8(edge, _mm256_setzero_si256());

            let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
            let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
            let i16_l1 = _mm256_unpacklo_epi16(hi0, _mm256_setzero_si256());
            let i16_h1 = _mm256_unpackhi_epi16(hi0, _mm256_setzero_si256());

            store_0 = _mm256_madd_epi16(i16_l0, v_edge_count);
            store_1 = _mm256_madd_epi16(i16_h0, v_edge_count);

            store_2 = _mm256_madd_epi16(i16_l1, v_edge_count);
            store_3 = _mm256_madd_epi16(i16_h1, v_edge_count);
        }

        unsafe {
            for y in 1..=half_kernel as usize {
                let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge = _mm256_loadu_si256(s_ptr as *const _);
                let lo0 = _mm256_unpacklo_epi8(edge, _mm256_setzero_si256());
                let hi0 = _mm256_unpackhi_epi8(edge, _mm256_setzero_si256());

                let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
                let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
                let i16_l1 = _mm256_unpacklo_epi16(hi0, _mm256_setzero_si256());
                let i16_h1 = _mm256_unpackhi_epi16(hi0, _mm256_setzero_si256());

                store_0 = _mm256_add_epi32(store_0, i16_l0);
                store_1 = _mm256_add_epi32(store_1, i16_h0);

                store_2 = _mm256_add_epi32(store_2, i16_l1);
                store_3 = _mm256_add_epi32(store_3, i16_h1);
            }
        }

        unsafe {
            for y in 0..height {
                // preload edge pixels
                let next = (y + half_kernel + 1).min(height - 1) as usize * src_stride as usize;
                let previous =
                    (y as i64 - half_kernel as i64).max(0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                {
                    let px = cx as usize;

                    let store0_ps = _mm256_cvtepi32_ps(store_0);
                    let store1_ps = _mm256_cvtepi32_ps(store_1);
                    let store2_ps = _mm256_cvtepi32_ps(store_2);
                    let store3_ps = _mm256_cvtepi32_ps(store_3);

                    let r0 = _mm256_mul_ps(store0_ps, v_weight);
                    let r1 = _mm256_mul_ps(store1_ps, v_weight);
                    let r2 = _mm256_mul_ps(store2_ps, v_weight);
                    let r3 = _mm256_mul_ps(store3_ps, v_weight);

                    let scale_store_0 = _mm256_cvtps_epi32(r0);
                    let scale_store_1 = _mm256_cvtps_epi32(r1);
                    let scale_store_2 = _mm256_cvtps_epi32(r2);
                    let scale_store_3 = _mm256_cvtps_epi32(r3);

                    let offset = y_dst_shift + px;
                    let ptr = unsafe_dst.slice.get_unchecked(offset).get();

                    let set0 = _mm256_packus_epi32(scale_store_0, scale_store_1);
                    let set1 = _mm256_packus_epi32(scale_store_2, scale_store_3);

                    let full_set = _mm256_packus_epi16(set0, set1);
                    _mm256_storeu_si256(ptr as *mut _, full_set);
                }

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge = _mm256_loadu_si256(s_ptr as *const _);
                    let lo0 = _mm256_unpacklo_epi8(edge, _mm256_setzero_si256());
                    let hi0 = _mm256_unpackhi_epi8(edge, _mm256_setzero_si256());

                    let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
                    let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
                    let i16_l1 = _mm256_unpacklo_epi16(hi0, _mm256_setzero_si256());
                    let i16_h1 = _mm256_unpackhi_epi16(hi0, _mm256_setzero_si256());

                    store_0 = _mm256_sub_epi32(store_0, i16_l0);
                    store_1 = _mm256_sub_epi32(store_1, i16_h0);

                    store_2 = _mm256_sub_epi32(store_2, i16_l1);
                    store_3 = _mm256_sub_epi32(store_3, i16_h1);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge = _mm256_loadu_si256(s_ptr as *const _);
                    let lo0 = _mm256_unpacklo_epi8(edge, _mm256_setzero_si256());
                    let hi0 = _mm256_unpackhi_epi8(edge, _mm256_setzero_si256());

                    let i16_l0 = _mm256_unpacklo_epi16(lo0, _mm256_setzero_si256());
                    let i16_h0 = _mm256_unpackhi_epi16(lo0, _mm256_setzero_si256());
                    let i16_l1 = _mm256_unpacklo_epi16(hi0, _mm256_setzero_si256());
                    let i16_h1 = _mm256_unpackhi_epi16(hi0, _mm256_setzero_si256());

                    store_0 = _mm256_add_epi32(store_0, i16_l0);
                    store_1 = _mm256_add_epi32(store_1, i16_h0);

                    store_2 = _mm256_add_epi32(store_2, i16_l1);
                    store_3 = _mm256_add_epi32(store_3, i16_h1);
                }
            }
        }

        cx += 32;
    }

    let v_edge_count = _mm_set1_epi32(edge_count as i32);
    let v_weight = _mm_set1_ps(1f32 / (radius * 2 + 1) as f32);

    while cx + 16 < end_x {
        let px = cx as usize;

        let mut store_0: __m128i;
        let mut store_1: __m128i;
        let mut store_2: __m128i;
        let mut store_3: __m128i;

        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge = _mm_loadu_si128(s_ptr as *const _);
            let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());
            let hi0 = _mm_unpackhi_epi8(edge, _mm_setzero_si128());

            let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
            let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());
            let i16_l1 = _mm_unpacklo_epi16(hi0, _mm_setzero_si128());
            let i16_h1 = _mm_unpackhi_epi16(hi0, _mm_setzero_si128());

            store_0 = _mm_madd_epi16(i16_l0, v_edge_count);
            store_1 = _mm_madd_epi16(i16_h0, v_edge_count);

            store_2 = _mm_madd_epi16(i16_l1, v_edge_count);
            store_3 = _mm_madd_epi16(i16_h1, v_edge_count);
        }

        unsafe {
            for y in 1..=half_kernel as usize {
                let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge = _mm_loadu_si128(s_ptr as *const _);
                let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());
                let hi0 = _mm_unpackhi_epi8(edge, _mm_setzero_si128());

                let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());
                let i16_l1 = _mm_unpacklo_epi16(hi0, _mm_setzero_si128());
                let i16_h1 = _mm_unpackhi_epi16(hi0, _mm_setzero_si128());

                store_0 = _mm_add_epi32(store_0, i16_l0);
                store_1 = _mm_add_epi32(store_1, i16_h0);

                store_2 = _mm_add_epi32(store_2, i16_l1);
                store_3 = _mm_add_epi32(store_3, i16_h1);
            }
        }

        unsafe {
            for y in 0..height {
                // preload edge pixels
                let next =
                    std::cmp::min(y + half_kernel + 1, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                {
                    let px = cx as usize;

                    let store0_ps = _mm_cvtepi32_ps(store_0);
                    let store1_ps = _mm_cvtepi32_ps(store_1);
                    let store2_ps = _mm_cvtepi32_ps(store_2);
                    let store3_ps = _mm_cvtepi32_ps(store_3);

                    let r0 = _mm_mul_ps(store0_ps, v_weight);
                    let r1 = _mm_mul_ps(store1_ps, v_weight);
                    let r2 = _mm_mul_ps(store2_ps, v_weight);
                    let r3 = _mm_mul_ps(store3_ps, v_weight);

                    let scale_store_0 = _mm_cvtps_epi32(r0);
                    let scale_store_1 = _mm_cvtps_epi32(r1);
                    let scale_store_2 = _mm_cvtps_epi32(r2);
                    let scale_store_3 = _mm_cvtps_epi32(r3);

                    let offset = y_dst_shift + px;
                    let ptr = unsafe_dst.slice.get_unchecked(offset).get();

                    let set0 = _mm_packus_epi32(scale_store_0, scale_store_1);
                    let set1 = _mm_packus_epi32(scale_store_2, scale_store_3);

                    let full_set = _mm_packus_epi16(set0, set1);
                    _mm_storeu_si128(ptr as *mut _, full_set);
                }

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge = _mm_loadu_si128(s_ptr as *const _);
                    let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());
                    let hi0 = _mm_unpackhi_epi8(edge, _mm_setzero_si128());

                    let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());
                    let i16_l1 = _mm_unpacklo_epi16(hi0, _mm_setzero_si128());
                    let i16_h1 = _mm_unpackhi_epi16(hi0, _mm_setzero_si128());

                    store_0 = _mm_sub_epi32(store_0, i16_l0);
                    store_1 = _mm_sub_epi32(store_1, i16_h0);

                    store_2 = _mm_sub_epi32(store_2, i16_l1);
                    store_3 = _mm_sub_epi32(store_3, i16_h1);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge = _mm_loadu_si128(s_ptr as *const _);
                    let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());
                    let hi0 = _mm_unpackhi_epi8(edge, _mm_setzero_si128());

                    let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());
                    let i16_l1 = _mm_unpacklo_epi16(hi0, _mm_setzero_si128());
                    let i16_h1 = _mm_unpackhi_epi16(hi0, _mm_setzero_si128());

                    store_0 = _mm_add_epi32(store_0, i16_l0);
                    store_1 = _mm_add_epi32(store_1, i16_h0);

                    store_2 = _mm_add_epi32(store_2, i16_l1);
                    store_3 = _mm_add_epi32(store_3, i16_h1);
                }
            }
        }

        cx += 16;
    }

    while cx + 8 < end_x {
        let px = cx as usize;

        let mut store_0: __m128i;
        let mut store_1: __m128i;

        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge = _mm_loadu_si64(s_ptr as *const _);
            let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

            let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
            let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

            store_0 = _mm_madd_epi16(i16_l0, v_edge_count);
            store_1 = _mm_madd_epi16(i16_h0, v_edge_count);
        }

        unsafe {
            for y in 1..=half_kernel {
                let y_src_shift = y.min(height - 1) as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);

                let edge = _mm_loadu_si64(s_ptr as *const _);
                let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

                let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

                store_0 = _mm_add_epi32(store_0, i16_l0);
                store_1 = _mm_add_epi32(store_1, i16_h0);
            }
        }

        unsafe {
            for y in 0..height {
                // preload edge pixels
                let next =
                    std::cmp::min(y + half_kernel + 1, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

                {
                    let px = cx as usize;

                    let store0_ps = _mm_cvtepi32_ps(store_0);
                    let store1_ps = _mm_cvtepi32_ps(store_1);

                    let r0 = _mm_mul_ps(store0_ps, v_weight);
                    let r1 = _mm_mul_ps(store1_ps, v_weight);

                    let scale_store_0 = _mm_cvtps_epi32(r0);
                    let scale_store_1 = _mm_cvtps_epi32(r1);

                    let offset = y_dst_shift + px;
                    let ptr = unsafe_dst.slice.get_unchecked(offset).get();

                    let set0 = _mm_packus_epi32(scale_store_0, scale_store_1);

                    let full_set = _mm_packus_epi16(set0, _mm_setzero_si128());
                    _mm_storeu_si64(ptr as *mut _, full_set);
                }

                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge = _mm_loadu_si64(s_ptr as *const _);
                    let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

                    let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

                    store_0 = _mm_sub_epi32(store_0, i16_l0);
                    store_1 = _mm_sub_epi32(store_1, i16_h0);
                }

                // add next
                {
                    let s_ptr = src.as_ptr().add(next + px);
                    let edge = _mm_loadu_si64(s_ptr as *const _);
                    let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

                    let i16_l0 = _mm_unpacklo_epi16(lo0, _mm_setzero_si128());
                    let i16_h0 = _mm_unpackhi_epi16(lo0, _mm_setzero_si128());

                    store_0 = _mm_add_epi32(store_0, i16_l0);
                    store_1 = _mm_add_epi32(store_1, i16_h0);
                }
            }
        }

        cx += 8;
    }

    const TAIL_CN: usize = 1;

    for x in cx..end_x {
        let px = x as usize * TAIL_CN;

        let mut store;
        unsafe {
            let s_ptr = src.as_ptr().add(px);
            let edge_colors = load_u8_s32_fast::<TAIL_CN>(s_ptr);
            store = _mm_madd_epi16(edge_colors, v_edge_count);
        }

        unsafe {
            for y in 1..=half_kernel as usize {
                let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let edge_colors = load_u8_s32_fast::<TAIL_CN>(s_ptr);
                store = _mm_add_epi32(store, edge_colors);
            }
        }

        for y in 0..height {
            // preload edge pixels
            let next =
                std::cmp::min(y + half_kernel + 1, height - 1) as usize * src_stride as usize;
            let previous =
                std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
            let y_dst_shift = dst_stride as usize * y as usize;

            let px = x as usize;

            unsafe {
                let r0 = _mm_mul_ps(_mm_cvtepi32_ps(store), v_weight);
                let ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8;
                store_u8_u32::<TAIL_CN>(ptr, _mm_cvtps_epi32(r0));
            }

            // subtract previous
            unsafe {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge_colors = load_u8_s32_fast::<TAIL_CN>(s_ptr);
                store = _mm_sub_epi32(store, edge_colors);
            }

            // add next
            unsafe {
                let s_ptr = src.as_ptr().add(next + px);
                let edge_colors = load_u8_s32_fast::<TAIL_CN>(s_ptr);
                store = _mm_add_epi32(store, edge_colors);
            }
        }
    }
}
