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

use crate::avx::shuffle;
use crate::sse::{load_u8_s16_fast, load_u8_s32_fast, store_u8_s16, store_u8_u32};
use crate::unsafe_slice::UnsafeSlice;
use std::arch::x86_64::*;

pub(crate) fn box_blur_vertical_pass_avx2<T>(
    undefined_src: &[T],
    src_stride: u32,
    undefined_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    w: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undefined_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undefined_unsafe_dst) };

    unsafe {
        if radius < 240 {
            box_blur_vertical_pass_avx2_def_lr(
                src, src_stride, unsafe_dst, dst_stride, w, height, radius, start_x, end_x,
            )
        } else {
            box_blur_vertical_pass_avx2_def(
                src, src_stride, unsafe_dst, dst_stride, w, height, radius, start_x, end_x,
            )
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn box_blur_vertical_pass_avx2_def(
    undefined_src: &[u8],
    src_stride: u32,
    undefined_unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    w: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    box_blur_vertical_pass_avx2_impl::<false>(
        undefined_src,
        src_stride,
        undefined_unsafe_dst,
        dst_stride,
        w,
        height,
        radius,
        start_x,
        end_x,
    )
}

#[inline(always)]
unsafe fn box_blur_vertical_pass_avx2_impl<const FMA: bool>(
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

    let v_edge_count = unsafe { _mm256_set1_epi32(edge_count as i32) };
    let v_weight = unsafe { _mm256_set1_ps(1f32 / (radius * 2) as f32) };

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
            for y in 1..half_kernel as usize {
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
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

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
            for y in 1..half_kernel as usize {
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
                let next = (y + half_kernel).min(height - 1) as usize * src_stride as usize;
                let previous =
                    (y as i64 - half_kernel as i64).max(0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

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
        }

        cx += 32;
    }

    let v_edge_count = unsafe { _mm_set1_epi32(edge_count as i32) };
    let v_weight = unsafe { _mm_set1_ps(1f32 / (radius * 2) as f32) };

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
            for y in 1..half_kernel as usize {
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
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

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
            for y in 1..std::cmp::min(half_kernel, height) {
                let y_src_shift = y as usize * src_stride as usize;
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
                    std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
                let previous =
                    std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
                let y_dst_shift = dst_stride as usize * y as usize;

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
            for y in 1..half_kernel as usize {
                let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let edge_colors = load_u8_s32_fast::<TAIL_CN>(s_ptr);
                store = _mm_add_epi32(store, edge_colors);
            }
        }

        for y in 0..height {
            // preload edge pixels
            let next = std::cmp::min(y + half_kernel, height - 1) as usize * src_stride as usize;
            let previous =
                std::cmp::max(y as i64 - half_kernel as i64, 0) as usize * src_stride as usize;
            let y_dst_shift = dst_stride as usize * y as usize;

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

            let px = x as usize;

            unsafe {
                let r0 = _mm_mul_ps(_mm_cvtepi32_ps(store), v_weight);
                let ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8;
                store_u8_u32::<TAIL_CN>(ptr, _mm_cvtps_epi32(r0));
            }
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn box_blur_vertical_pass_avx2_def_lr(
    undefined_src: &[u8],
    src_stride: u32,
    undefined_unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    w: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) {
    box_blur_vertical_pass_avx2_impl_lr::<false>(
        undefined_src,
        src_stride,
        undefined_unsafe_dst,
        dst_stride,
        w,
        height,
        radius,
        start_x,
        end_x,
    )
}

#[inline(always)]
unsafe fn box_blur_vertical_pass_avx2_impl_lr<const FMA: bool>(
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

    let v_edge_count = unsafe { _mm256_set1_epi8(edge_count as i8) };
    let v_weight = unsafe { _mm256_set1_ps(1f32 / (radius * 2) as f32) };

    assert!(end_x >= start_x);

    let buf_size = end_x as usize - start_x as usize;

    let buf_cap = buf_size.div_ceil(16) * 16 + 16;

    let mut buffer = vec![0u16; buf_cap];

    // Configure initial accumulator

    let mut buf_cx = 0usize;

    while cx + 64 < end_x {
        let px = cx as usize;

        let mut store_0: __m256i;
        let mut store_1: __m256i;
        let mut store_2: __m256i;
        let mut store_3: __m256i;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = _mm256_loadu_si256(s_ptr as *const _);
        let edge1 = _mm256_loadu_si256(s_ptr.add(32) as *const _);
        let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
        let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

        let lo1 = _mm256_unpacklo_epi8(edge1, _mm256_setzero_si256());
        let hi1 = _mm256_unpackhi_epi8(edge1, _mm256_setzero_si256());

        store_0 = _mm256_maddubs_epi16(lo0, v_edge_count);
        store_1 = _mm256_maddubs_epi16(hi0, v_edge_count);

        store_2 = _mm256_maddubs_epi16(lo1, v_edge_count);
        store_3 = _mm256_maddubs_epi16(hi1, v_edge_count);

        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = _mm256_loadu_si256(s_ptr as *const _);
            let edge1 = _mm256_loadu_si256(s_ptr.add(32) as *const _);
            let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
            let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

            let lo1 = _mm256_unpacklo_epi8(edge1, _mm256_setzero_si256());
            let hi1 = _mm256_unpackhi_epi8(edge1, _mm256_setzero_si256());

            store_0 = _mm256_adds_epu16(store_0, lo0);
            store_1 = _mm256_adds_epu16(store_1, hi0);

            store_2 = _mm256_adds_epu16(store_2, lo1);
            store_3 = _mm256_adds_epu16(store_3, hi1);
        }

        _mm256_storeu_si256(
            buffer.get_unchecked_mut(buf_cx..).as_mut_ptr() as *mut _,
            store_0,
        );
        _mm256_storeu_si256(
            buffer.get_unchecked_mut(buf_cx + 16..).as_mut_ptr() as *mut _,
            store_1,
        );
        _mm256_storeu_si256(
            buffer.get_unchecked_mut(buf_cx + 32..).as_mut_ptr() as *mut _,
            store_2,
        );
        _mm256_storeu_si256(
            buffer.get_unchecked_mut(buf_cx + 48..).as_mut_ptr() as *mut _,
            store_3,
        );

        buf_cx += 64;
        cx += 64;
    }

    while cx + 32 < end_x {
        let px = cx as usize;

        let mut store_0: __m256i;
        let mut store_1: __m256i;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = _mm256_loadu_si256(s_ptr as *const _);
        let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
        let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

        store_0 = _mm256_maddubs_epi16(lo0, v_edge_count);
        store_1 = _mm256_maddubs_epi16(hi0, v_edge_count);

        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = _mm256_loadu_si256(s_ptr as *const _);
            let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
            let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

            store_0 = _mm256_adds_epu16(store_0, lo0);
            store_1 = _mm256_adds_epu16(store_1, hi0);
        }

        _mm256_storeu_si256(
            buffer.get_unchecked_mut(buf_cx..).as_mut_ptr() as *mut _,
            store_0,
        );
        _mm256_storeu_si256(
            buffer.get_unchecked_mut(buf_cx + 16..).as_mut_ptr() as *mut _,
            store_1,
        );

        buf_cx += 32;
        cx += 32;
    }

    while cx + 16 < end_x {
        let px = cx as usize;

        let mut store_0: __m256i;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = _mm256_castsi128_si256(_mm_loadu_si128(s_ptr as *const _));

        const MASK: i32 = shuffle(3, 1, 2, 0);

        let lo0 = _mm256_unpacklo_epi8(
            _mm256_permute4x64_epi64::<MASK>(edge0),
            _mm256_setzero_si256(),
        );

        store_0 = _mm256_maddubs_epi16(lo0, v_edge_count);

        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = _mm256_castsi128_si256(_mm_loadu_si128(s_ptr as *const _));
            let lo0 = _mm256_unpacklo_epi8(
                _mm256_permute4x64_epi64::<MASK>(edge0),
                _mm256_setzero_si256(),
            );

            store_0 = _mm256_adds_epu16(store_0, lo0);
        }

        _mm256_storeu_si256(
            buffer.get_unchecked_mut(buf_cx..).as_mut_ptr() as *mut _,
            store_0,
        );

        buf_cx += 16;
        cx += 16;
    }

    let v_edge_count = unsafe { _mm_set1_epi8(edge_count as i8) };

    while cx + 8 < end_x {
        let px = cx as usize;

        let s_ptr = src.as_ptr().add(px);
        let edge0 = _mm_loadu_si64(s_ptr as *const _);

        let lo0 = _mm_unpacklo_epi8(edge0, _mm_setzero_si128());

        let mut store_0 = _mm_maddubs_epi16(lo0, v_edge_count);

        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            let s_ptr = src.as_ptr().add(y_src_shift + px);

            let edge0 = _mm_loadu_si64(s_ptr as *const _);
            let lo0 = _mm_unpacklo_epi8(edge0, _mm_setzero_si128());

            store_0 = _mm_adds_epu16(store_0, lo0);
        }

        _mm_storeu_si128(
            buffer.get_unchecked_mut(buf_cx..).as_mut_ptr() as *mut _,
            store_0,
        );

        buf_cx += 8;
        cx += 8;
    }

    while cx < end_x {
        let px = cx as usize;

        let s_ptr = src.get_unchecked(px);
        let edge0 = _mm_setr_epi16(*s_ptr as i16, 0, 0, 0, 0, 0, 0, 0);

        let lo0 = _mm_unpacklo_epi8(edge0, _mm_setzero_si128());

        let mut store_0 = _mm_maddubs_epi16(lo0, v_edge_count);

        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            let s_ptr = src.get_unchecked(y_src_shift + px);

            let edge0 = _mm_setr_epi16(*s_ptr as i16, 0, 0, 0, 0, 0, 0, 0);
            let lo0 = _mm_unpacklo_epi8(edge0, _mm_setzero_si128());

            store_0 = _mm_adds_epu16(store_0, lo0);
        }

        _mm_storeu_si16(
            buffer.get_unchecked_mut(buf_cx..).as_mut_ptr() as *mut _,
            store_0,
        );

        buf_cx += 1;
        cx += 1;
    }

    for y in 0..height as usize {
        let mut buf_cx = 0usize;
        let mut cx = start_x as usize;

        // preload edge pixels
        let next = (y + half_kernel as usize).min(height as usize - 1) * src_stride as usize;
        let previous = (y as i64 - half_kernel as i64).max(0) as usize * src_stride as usize;
        let y_dst_shift = dst_stride as usize * y;

        while cx + 64 < end_x as usize {
            let px = cx;

            let mut store_0 =
                _mm256_loadu_si256(buffer.get_unchecked(buf_cx..).as_ptr() as *const _);
            let mut store_1 =
                _mm256_loadu_si256(buffer.get_unchecked(buf_cx + 16..).as_ptr() as *const _);
            let mut store_2 =
                _mm256_loadu_si256(buffer.get_unchecked(buf_cx + 32..).as_ptr() as *const _);
            let mut store_3 =
                _mm256_loadu_si256(buffer.get_unchecked(buf_cx + 48..).as_ptr() as *const _);

            unsafe {
                // subtract previous
                {
                    let s_ptr = src.as_ptr().add(previous + px);
                    let edge0 = _mm256_loadu_si256(s_ptr as *const _);
                    let edge1 = _mm256_loadu_si256(s_ptr.add(32) as *const _);
                    let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
                    let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

                    let lo1 = _mm256_unpacklo_epi8(edge1, _mm256_setzero_si256());
                    let hi1 = _mm256_unpackhi_epi8(edge1, _mm256_setzero_si256());

                    store_0 = _mm256_subs_epu16(store_0, lo0);
                    store_1 = _mm256_subs_epu16(store_1, hi0);

                    store_2 = _mm256_subs_epu16(store_2, lo1);
                    store_3 = _mm256_subs_epu16(store_3, hi1);
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

                    store_0 = _mm256_adds_epu16(store_0, lo0);
                    store_1 = _mm256_adds_epu16(store_1, hi0);

                    store_2 = _mm256_adds_epu16(store_2, lo1);
                    store_3 = _mm256_adds_epu16(store_3, hi1);
                }

                let px = cx;

                _mm256_storeu_si256(buffer.get_unchecked(buf_cx..).as_ptr() as *mut _, store_0);
                _mm256_storeu_si256(
                    buffer.get_unchecked(buf_cx + 16..).as_ptr() as *mut _,
                    store_1,
                );
                _mm256_storeu_si256(
                    buffer.get_unchecked(buf_cx + 32..).as_ptr() as *mut _,
                    store_2,
                );
                _mm256_storeu_si256(
                    buffer.get_unchecked(buf_cx + 48..).as_ptr() as *mut _,
                    store_3,
                );

                let jw0 = _mm256_unpacklo_epi16(store_0, _mm256_setzero_si256());
                let jw1 = _mm256_unpackhi_epi16(store_0, _mm256_setzero_si256());
                let jw2 = _mm256_unpacklo_epi16(store_1, _mm256_setzero_si256());
                let jw3 = _mm256_unpackhi_epi16(store_1, _mm256_setzero_si256());
                let jw4 = _mm256_unpacklo_epi16(store_2, _mm256_setzero_si256());
                let jw5 = _mm256_unpackhi_epi16(store_2, _mm256_setzero_si256());
                let jw6 = _mm256_unpacklo_epi16(store_3, _mm256_setzero_si256());
                let jw7 = _mm256_unpackhi_epi16(store_3, _mm256_setzero_si256());

                let store0_ps = _mm256_cvtepi32_ps(jw0);
                let store1_ps = _mm256_cvtepi32_ps(jw1);
                let store2_ps = _mm256_cvtepi32_ps(jw2);
                let store3_ps = _mm256_cvtepi32_ps(jw3);
                let store4_ps = _mm256_cvtepi32_ps(jw4);
                let store5_ps = _mm256_cvtepi32_ps(jw5);
                let store6_ps = _mm256_cvtepi32_ps(jw6);
                let store7_ps = _mm256_cvtepi32_ps(jw7);

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

            cx += 64;
            buf_cx += 64;
        }

        while cx + 32 < end_x as usize {
            let px = cx;

            let mut store_0 =
                _mm256_loadu_si256(buffer.get_unchecked(buf_cx..).as_ptr() as *const _);
            let mut store_1 =
                _mm256_loadu_si256(buffer.get_unchecked(buf_cx + 16..).as_ptr() as *const _);

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge0 = _mm256_loadu_si256(s_ptr as *const _);
                let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
                let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

                store_0 = _mm256_subs_epu16(store_0, lo0);
                store_1 = _mm256_subs_epu16(store_1, hi0);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge0 = _mm256_loadu_si256(s_ptr as *const _);
                let lo0 = _mm256_unpacklo_epi8(edge0, _mm256_setzero_si256());
                let hi0 = _mm256_unpackhi_epi8(edge0, _mm256_setzero_si256());

                store_0 = _mm256_adds_epu16(store_0, lo0);
                store_1 = _mm256_adds_epu16(store_1, hi0);
            }

            let px = cx;

            _mm256_storeu_si256(buffer.get_unchecked(buf_cx..).as_ptr() as *mut _, store_0);
            _mm256_storeu_si256(
                buffer.get_unchecked(buf_cx + 16..).as_ptr() as *mut _,
                store_1,
            );

            let jw0 = _mm256_unpacklo_epi16(store_0, _mm256_setzero_si256());
            let jw1 = _mm256_unpackhi_epi16(store_0, _mm256_setzero_si256());
            let jw2 = _mm256_unpacklo_epi16(store_1, _mm256_setzero_si256());
            let jw3 = _mm256_unpackhi_epi16(store_1, _mm256_setzero_si256());

            let store0_ps = _mm256_cvtepi32_ps(jw0);
            let store1_ps = _mm256_cvtepi32_ps(jw1);
            let store2_ps = _mm256_cvtepi32_ps(jw2);
            let store3_ps = _mm256_cvtepi32_ps(jw3);

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

            let full_set0 = _mm256_packus_epi16(set0, set1);
            _mm256_storeu_si256(ptr as *mut _, full_set0);

            cx += 32;
            buf_cx += 32;
        }

        while cx + 16 < end_x as usize {
            let px = cx;

            let mut store_0 =
                _mm256_loadu_si256(buffer.get_unchecked(buf_cx..).as_ptr() as *const _);

            const MASK: i32 = shuffle(3, 1, 2, 0);

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge0 = _mm256_castsi128_si256(_mm_loadu_si128(s_ptr as *const _));
                let lo0 = _mm256_unpacklo_epi8(
                    _mm256_permute4x64_epi64::<MASK>(edge0),
                    _mm256_setzero_si256(),
                );

                store_0 = _mm256_subs_epu16(store_0, lo0);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge0 = _mm256_castsi128_si256(_mm_loadu_si128(s_ptr as *const _));
                let lo0 = _mm256_unpacklo_epi8(
                    _mm256_permute4x64_epi64::<MASK>(edge0),
                    _mm256_setzero_si256(),
                );

                store_0 = _mm256_adds_epu16(store_0, lo0);
            }

            let px = cx;

            _mm256_storeu_si256(buffer.get_unchecked(buf_cx..).as_ptr() as *mut _, store_0);

            let jw0 = _mm256_unpacklo_epi16(store_0, _mm256_setzero_si256());
            let jw1 = _mm256_unpackhi_epi16(store_0, _mm256_setzero_si256());

            let store0_ps = _mm256_cvtepi32_ps(jw0);
            let store1_ps = _mm256_cvtepi32_ps(jw1);

            let r0 = _mm256_mul_ps(store0_ps, v_weight);
            let r1 = _mm256_mul_ps(store1_ps, v_weight);

            let scale_store_0 = _mm256_cvtps_epi32(r0);
            let scale_store_1 = _mm256_cvtps_epi32(r1);

            let offset = y_dst_shift + px;
            let ptr = unsafe_dst.slice.get_unchecked(offset).get();

            let set0 = _mm256_packus_epi32(scale_store_0, scale_store_1);

            let full_set0 = _mm256_packus_epi16(set0, set0);
            _mm_storeu_si128(ptr as *mut _, _mm256_castsi256_si128(full_set0));

            cx += 16;
            buf_cx += 16;
        }

        let v_weight = unsafe { _mm_set1_ps(1f32 / (radius * 2) as f32) };

        while cx + 8 < end_x as usize {
            let px = cx;

            let mut store_0 = _mm_loadu_si128(buffer.get_unchecked(buf_cx..).as_ptr() as *const _);

            // subtract previous
            {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge = _mm_loadu_si64(s_ptr as *const _);
                let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

                store_0 = _mm_subs_epu16(store_0, lo0);
            }

            // add next
            {
                let s_ptr = src.as_ptr().add(next + px);
                let edge = _mm_loadu_si64(s_ptr as *const _);
                let lo0 = _mm_unpacklo_epi8(edge, _mm_setzero_si128());

                store_0 = _mm_adds_epu16(store_0, lo0);
            }

            let px = cx;

            let jw = store_0;

            _mm_storeu_si128(buffer.get_unchecked(buf_cx..).as_ptr() as *mut _, store_0);

            let store_0 = _mm_unpacklo_epi16(jw, _mm_setzero_si128());
            let store_1 = _mm_unpackhi_epi16(jw, _mm_setzero_si128());

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

            cx += 8;
            buf_cx += 8;
        }

        const TAIL_CN: usize = 1;

        for x in cx..end_x as usize {
            let px = x * TAIL_CN;

            let mut store = _mm_loadu_si16(buffer.get_unchecked(buf_cx..).as_ptr() as *const _);

            // subtract previous
            unsafe {
                let s_ptr = src.as_ptr().add(previous + px);
                let edge_colors = load_u8_s16_fast::<TAIL_CN>(s_ptr);
                store = _mm_subs_epu16(store, edge_colors);
            }

            // add next
            unsafe {
                let s_ptr = src.as_ptr().add(next + px);
                let edge_colors = load_u8_s16_fast::<TAIL_CN>(s_ptr);
                store = _mm_adds_epu16(store, edge_colors);
            }

            let px = x;

            _mm_storeu_si16(buffer.get_unchecked(buf_cx..).as_ptr() as *mut _, store);

            let store = _mm_unpacklo_epi16(store, _mm_setzero_si128());
            let r0 = _mm_mul_ps(_mm_cvtepi32_ps(store), v_weight);
            let ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + px) as *mut u8;
            store_u8_s16::<TAIL_CN>(ptr, _mm_cvtps_epi32(r0));
            buf_cx += 1;
        }
    }
}
