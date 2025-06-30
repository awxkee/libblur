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

use crate::sse::{load_u8_s32_fast, store_u8_u32, write_u8};
use crate::unsafe_slice::UnsafeSlice;
use std::arch::x86_64::*;

pub(crate) fn box_blur_horizontal_pass_avx<const CN: usize>(
    src: &[u8],
    src_stride: u32,
    dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        box_blur_horizontal_pass_impl::<CN>(
            src, src_stride, dst, dst_stride, width, radius, start_y, end_y,
        );
    }
}

#[inline(always)]
pub(crate) unsafe fn load_u8_s32_fast_x2<const CN: usize>(
    ptr0: *const u8,
    ptr1: *const u8,
) -> __m256i {
    let sh1 = _mm256_setr_epi8(
        0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1,
        2, -1, -1, -1, 3, -1, -1, -1,
    );
    if CN == 4 {
        let v0 = _mm_loadu_si32(ptr0 as *const _);
        let v1 = _mm_loadu_si32(ptr1 as *const _);

        _mm256_shuffle_epi8(
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(v0), v1),
            sh1,
        )
    } else if CN == 3 {
        let mut v0 = _mm_loadu_si16(ptr0);
        v0 = _mm_insert_epi8::<2>(v0, ptr0.add(2).read_unaligned() as i32);
        let mut v1 = _mm_loadu_si16(ptr1);
        v1 = _mm_insert_epi8::<2>(v1, ptr1.add(2).read_unaligned() as i32);
        _mm256_shuffle_epi8(
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(v0), v1),
            sh1,
        )
    } else if CN == 2 {
        let v0 = _mm_loadu_si16(ptr0);
        let v1 = _mm_loadu_si16(ptr1);
        _mm256_shuffle_epi8(
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(v0), v1),
            sh1,
        )
    } else {
        _mm256_shuffle_epi8(
            _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(_mm_cvtsi32_si128(ptr0.read_unaligned() as i32)),
                _mm_cvtsi32_si128(ptr1.read_unaligned() as i32),
            ),
            sh1,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn box_blur_horizontal_pass_impl<const CN: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    let kernel_size = radius * 2 + 1;
    let edge_count = (kernel_size / 2) + 1;
    let v_edge_count = _mm256_set1_epi32(edge_count as i32);

    let v_weight = _mm256_set1_ps(1f32 / (radius * 2 + 1) as f32);

    let half_kernel = kernel_size / 2;

    let mut yy = start_y;

    while yy + 6 < end_y {
        let y = yy;
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store_0: __m256i;
        let mut store_1: __m256i;
        let mut store_2: __m256i;

        unsafe {
            let s_ptr_0 = src.as_ptr().add(y_src_shift);
            let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize);
            let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2);
            let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3);
            let s_ptr_4 = src.as_ptr().add(y_src_shift + src_stride as usize * 4);
            let s_ptr_5 = src.as_ptr().add(y_src_shift + src_stride as usize * 5);

            let edge_colors_0 = load_u8_s32_fast_x2::<CN>(s_ptr_0, s_ptr_1);
            let edge_colors_1 = load_u8_s32_fast_x2::<CN>(s_ptr_2, s_ptr_3);
            let edge_colors_2 = load_u8_s32_fast_x2::<CN>(s_ptr_4, s_ptr_5);

            store_0 = _mm256_madd_epi16(edge_colors_0, v_edge_count);
            store_1 = _mm256_madd_epi16(edge_colors_1, v_edge_count);
            store_2 = _mm256_madd_epi16(edge_colors_2, v_edge_count);
        }

        unsafe {
            for x in 1..=half_kernel as usize {
                let px = x.min(width as usize - 1) * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + px);
                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + px);
                let s_ptr_2 = src.as_ptr().add(y_src_shift + src_stride as usize * 2 + px);
                let s_ptr_3 = src.as_ptr().add(y_src_shift + src_stride as usize * 3 + px);
                let s_ptr_4 = src.as_ptr().add(y_src_shift + src_stride as usize * 4 + px);
                let s_ptr_5 = src.as_ptr().add(y_src_shift + src_stride as usize * 5 + px);

                let edge_colors_0 = load_u8_s32_fast_x2::<CN>(s_ptr_0, s_ptr_1);
                let edge_colors_1 = load_u8_s32_fast_x2::<CN>(s_ptr_2, s_ptr_3);
                let edge_colors_2 = load_u8_s32_fast_x2::<CN>(s_ptr_4, s_ptr_5);

                store_0 = _mm256_add_epi32(store_0, edge_colors_0);
                store_1 = _mm256_add_epi32(store_1, edge_colors_1);
                store_2 = _mm256_add_epi32(store_2, edge_colors_2);
            }
        }

        for x in 0..width {
            let px = x as usize * CN;

            unsafe {
                let scale_store_ps0 = _mm256_cvtepi32_ps(store_0);
                let scale_store_ps1 = _mm256_cvtepi32_ps(store_1);
                let scale_store_ps2 = _mm256_cvtepi32_ps(store_2);

                let r0 = _mm256_mul_ps(scale_store_ps0, v_weight);
                let r1 = _mm256_mul_ps(scale_store_ps1, v_weight);
                let r2 = _mm256_mul_ps(scale_store_ps2, v_weight);

                let scale_store0 = _mm256_cvtps_epi32(r0);
                let scale_store1 = _mm256_cvtps_epi32(r1);
                let scale_store2 = _mm256_cvtps_epi32(r2);

                let px_160 = _mm256_packus_epi32(scale_store0, _mm256_setzero_si256());
                let px_161 = _mm256_packus_epi32(scale_store1, _mm256_setzero_si256());
                let px_162 = _mm256_packus_epi32(scale_store2, _mm256_setzero_si256());

                let px_80 = _mm256_packus_epi16(px_160, _mm256_setzero_si256());
                let px_81 = _mm256_packus_epi16(px_161, _mm256_setzero_si256());
                let px_82 = _mm256_packus_epi16(px_162, _mm256_setzero_si256());

                let bytes_offset_0 = y_dst_shift + px;
                let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
                let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
                let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;
                let bytes_offset_4 = y_dst_shift + dst_stride as usize * 4 + px;
                let bytes_offset_5 = y_dst_shift + dst_stride as usize * 5 + px;

                let dst_ptr_0 = unsafe_dst.slice.as_ptr().add(bytes_offset_0) as *mut u8;
                let dst_ptr_1 = unsafe_dst.slice.as_ptr().add(bytes_offset_1) as *mut u8;
                let dst_ptr_2 = unsafe_dst.slice.as_ptr().add(bytes_offset_2) as *mut u8;
                let dst_ptr_3 = unsafe_dst.slice.as_ptr().add(bytes_offset_3) as *mut u8;
                let dst_ptr_4 = unsafe_dst.slice.as_ptr().add(bytes_offset_4) as *mut u8;
                let dst_ptr_5 = unsafe_dst.slice.as_ptr().add(bytes_offset_5) as *mut u8;

                write_u8::<CN>(dst_ptr_0, _mm256_castsi256_si128(px_80));
                write_u8::<CN>(dst_ptr_1, _mm256_extracti128_si256::<1>(px_80));
                write_u8::<CN>(dst_ptr_2, _mm256_castsi256_si128(px_81));
                write_u8::<CN>(dst_ptr_3, _mm256_extracti128_si256::<1>(px_81));
                write_u8::<CN>(dst_ptr_4, _mm256_castsi256_si128(px_82));
                write_u8::<CN>(dst_ptr_5, _mm256_extracti128_si256::<1>(px_82));
            }

            // subtract previous
            unsafe {
                let previous_x = (x as i64 - half_kernel as i64).max(0) as usize;
                let previous = previous_x * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + previous);
                let s_ptr_1 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize + previous);
                let s_ptr_2 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 2 + previous);
                let s_ptr_3 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 3 + previous);
                let s_ptr_4 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 4 + previous);
                let s_ptr_5 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 5 + previous);

                let edge_colors_0 = load_u8_s32_fast_x2::<CN>(s_ptr_0, s_ptr_1);
                let edge_colors_1 = load_u8_s32_fast_x2::<CN>(s_ptr_2, s_ptr_3);
                let edge_colors_2 = load_u8_s32_fast_x2::<CN>(s_ptr_4, s_ptr_5);

                store_0 = _mm256_sub_epi32(store_0, edge_colors_0);
                store_1 = _mm256_sub_epi32(store_1, edge_colors_1);
                store_2 = _mm256_sub_epi32(store_2, edge_colors_2);
            }

            // add next
            unsafe {
                let next_x = (x + half_kernel + 1).min(width - 1) as usize;

                let next = next_x * CN;

                let s_ptr_0 = src.as_ptr().add(y_src_shift + next);
                let s_ptr_1 = src.as_ptr().add(y_src_shift + src_stride as usize + next);
                let s_ptr_2 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 2 + next);
                let s_ptr_3 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 3 + next);
                let s_ptr_4 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 4 + next);
                let s_ptr_5 = src
                    .as_ptr()
                    .add(y_src_shift + src_stride as usize * 5 + next);

                let edge_colors_0 = load_u8_s32_fast_x2::<CN>(s_ptr_0, s_ptr_1);
                let edge_colors_1 = load_u8_s32_fast_x2::<CN>(s_ptr_2, s_ptr_3);
                let edge_colors_2 = load_u8_s32_fast_x2::<CN>(s_ptr_4, s_ptr_5);

                store_0 = _mm256_add_epi32(store_0, edge_colors_0);
                store_1 = _mm256_add_epi32(store_1, edge_colors_1);
                store_2 = _mm256_add_epi32(store_2, edge_colors_2);
            }
        }

        yy += 6;
    }

    for y in yy..end_y {
        let y_src_shift = y as usize * src_stride as usize;
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut store;

        unsafe {
            let s_ptr = src.as_ptr().add(y_src_shift);
            let edge_colors = load_u8_s32_fast::<CN>(s_ptr);
            store = _mm_madd_epi16(edge_colors, _mm256_castsi256_si128(v_edge_count));
        }

        unsafe {
            for x in 1usize..=half_kernel as usize {
                let px = x.min(width as usize - 1) * CN;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let edge_colors = load_u8_s32_fast::<CN>(s_ptr);
                store = _mm_add_epi32(store, edge_colors);
            }
        }

        for x in 0..width {
            let px = x as usize * CN;

            unsafe {
                let r0 = _mm_mul_ps(_mm_cvtepi32_ps(store), _mm256_castps256_ps128(v_weight));
                let bytes_offset = y_dst_shift + px;
                let ptr = unsafe_dst.slice.as_ptr().add(bytes_offset) as *mut u8;
                store_u8_u32::<CN>(ptr, _mm_cvtps_epi32(r0));
            }

            // subtract previous
            unsafe {
                let previous_x = (x as i64 - half_kernel as i64).max(0) as usize;
                let previous = previous_x * CN;
                let s_ptr = src.as_ptr().add(y_src_shift + previous);
                let edge_colors = load_u8_s32_fast::<CN>(s_ptr);
                store = _mm_sub_epi32(store, edge_colors);
            }

            // add next
            unsafe {
                let next_x = (x + half_kernel + 1).min(width - 1) as usize;

                let next = next_x * CN;

                let s_ptr = src.as_ptr().add(y_src_shift + next);
                let edge_colors = load_u8_s32_fast::<CN>(s_ptr);
                store = _mm_add_epi32(store, edge_colors);
            }
        }
    }
}
