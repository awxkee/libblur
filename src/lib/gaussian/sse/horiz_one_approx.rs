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

use crate::gaussian::gaussian_approx::PRECISION;
use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::sse::{_mm_loadu_si128_x2, load_u8_s16_fast, shuffle};
use crate::unsafe_slice::UnsafeSlice;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_loadu_si16x(mem_addr: *const u8) -> __m128i {
    let item = (mem_addr as *const i16).read_unaligned();
    _mm_set1_epi16(item)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_sum_clamp(v: __m128i) -> i32 {
    const SHUFFLE_1: i32 = shuffle(1, 0, 3, 2);
    let hi64 = _mm_shuffle_epi32::<SHUFFLE_1>(v);
    let sum64 = _mm_add_epi32(hi64, v);
    let hi32 = _mm_shufflelo_epi16::<SHUFFLE_1>(sum64); // Swap the low two elements
    let sum32 = _mm_add_epi32(sum64, hi32);
    let cutoff = _mm_set1_epi32(255);
    let lowest = _mm_setzero_si128();
    _mm_cvtsi128_si32(_mm_min_epi32(
        _mm_max_epi32(_mm_srai_epi32::<PRECISION>(sum32), lowest),
        cutoff,
    ))
}

pub fn gaussian_sse_horiz_one_chan_u8_approx(
    undef_src: &[u8],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[i16],
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        gaussian_sse_horiz_one_chan_impl(
            undef_src,
            src_stride,
            undef_unsafe_dst,
            dst_stride,
            width,
            kernel_size,
            kernel,
            start_y,
            end_y,
        );
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn gaussian_sse_horiz_one_chan_impl(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[i16],
    start_y: u32,
    end_y: u32,
) {
    let half_kernel = (kernel_size / 2) as i32;

    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(4)).step_by(4) {
        for x in 0..width {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let y_src_shift_next = y_src_shift + src_stride as usize;

            let mut store0 = _mm_setzero_si128();
            let mut store1 = _mm_setzero_si128();
            let mut store2 = _mm_setzero_si128();
            let mut store3 = _mm_setzero_si128();

            let zeros = _mm_setzero_si128();

            let mut r = -half_kernel;

            let edge_value_check = x as i64 + r as i64;
            if edge_value_check < 0 {
                let diff = edge_value_check.abs();
                let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                let value0 = s_ptr.read_unaligned() as i32;
                let pixel_colors_0 = _mm_setr_epi32(value0, 0, 0, 0);

                let s_ptr_next = src.as_ptr().add(y_src_shift_next); // Here we're always at zero
                let value1 = s_ptr_next.read_unaligned() as i32;
                let pixel_colors_1 = _mm_setr_epi32(value1, 0, 0, 0);

                let s_ptr_next_2 = s_ptr_next.add(src_stride as usize); // Here we're always at zero
                let value2 = s_ptr_next_2.read_unaligned() as i32;
                let pixel_colors_2 = _mm_setr_epi32(value2, 0, 0, 0);

                let s_ptr_next_3 = s_ptr_next_2.add(src_stride as usize); // Here we're always at zero
                let value3 = s_ptr_next_3.read_unaligned() as i32;
                let pixel_colors_3 = _mm_setr_epi32(value3, 0, 0, 0);
                for i in 0..diff as usize {
                    let weights = kernel.as_ptr().add(i);
                    let f_weight = _mm_loadu_si16x(weights as *const u8);
                    store0 = _mm_add_epi32(store0, _mm_madd_epi16(pixel_colors_0, f_weight));
                    store1 = _mm_add_epi32(store1, _mm_madd_epi16(pixel_colors_1, f_weight));
                    store2 = _mm_add_epi32(store2, _mm_madd_epi16(pixel_colors_2, f_weight));
                    store3 = _mm_add_epi32(store3, _mm_madd_epi16(pixel_colors_3, f_weight));
                }
                r += diff as i32;
            }

            while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                let pixel_colors_u8_0 = _mm_loadu_si128(s_ptr as *const __m128i);
                let pixel_colors_u8_1 = _mm_loadu_si128(s_ptr_next as *const __m128i);
                let pixel_colors_u8_2 =
                    _mm_loadu_si128(s_ptr_next.add(src_stride as usize) as *const __m128i);
                let pixel_colors_u8_3 =
                    _mm_loadu_si128(s_ptr_next.add(src_stride as usize * 2) as *const __m128i);
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let weights = _mm_loadu_si128_x2(weight as *const u8);

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_0, zeros), weights.0),
                );
                store1 = _mm_add_epi32(
                    store1,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_1, zeros), weights.0),
                );
                store2 = _mm_add_epi32(
                    store2,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_2, zeros), weights.0),
                );
                store3 = _mm_add_epi32(
                    store3,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_3, zeros), weights.0),
                );

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8_0, zeros), weights.1),
                );
                store1 = _mm_add_epi32(
                    store1,
                    _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8_1, zeros), weights.1),
                );
                store2 = _mm_add_epi32(
                    store2,
                    _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8_2, zeros), weights.1),
                );
                store3 = _mm_add_epi32(
                    store3,
                    _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8_3, zeros), weights.1),
                );

                r += 16;
            }

            while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                let pixel_colors_u8_0 = _mm_loadu_si64(s_ptr);
                let pixel_colors_u8_1 = _mm_loadu_si64(s_ptr_next);
                let pixel_colors_u8_2 = _mm_loadu_si64(s_ptr_next.add(src_stride as usize));
                let pixel_colors_u8_3 = _mm_loadu_si64(s_ptr_next.add(src_stride as usize * 2));
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let weights = _mm_loadu_si128(weight as *const __m128i);

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_0, zeros), weights),
                );
                store1 = _mm_add_epi32(
                    store1,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_1, zeros), weights),
                );
                store2 = _mm_add_epi32(
                    store2,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_2, zeros), weights),
                );
                store3 = _mm_add_epi32(
                    store3,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_3, zeros), weights),
                );

                r += 8;
            }

            while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_i16_0 = load_u8_s16_fast::<4>(s_ptr);

                let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                let pixel_colors_i16_1 = load_u8_s16_fast::<4>(s_ptr_next);

                let s_ptr_next_2 = s_ptr_next.add(src_stride as usize);
                let pixel_colors_i16_2 = load_u8_s16_fast::<4>(s_ptr_next_2);

                let s_ptr_next_3 = s_ptr_next_2.add(src_stride as usize);
                let pixel_colors_i16_3 = load_u8_s16_fast::<4>(s_ptr_next_3);

                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_loadu_si64(weight as *const u8);
                store0 = _mm_add_epi32(store0, _mm_madd_epi16(pixel_colors_i16_0, f_weight));
                store1 = _mm_add_epi32(store1, _mm_madd_epi16(pixel_colors_i16_1, f_weight));
                store2 = _mm_add_epi32(store2, _mm_madd_epi16(pixel_colors_i16_2, f_weight));
                store3 = _mm_add_epi32(store3, _mm_madd_epi16(pixel_colors_i16_3, f_weight));

                r += 4;
            }

            while r <= half_kernel {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                let pixel_colors_0 = _mm_setr_epi32(s_ptr.read_unaligned() as i32, 0, 0, 0);
                let pixel_colors_1 = _mm_setr_epi32(s_ptr_next.read_unaligned() as i32, 0, 0, 0);
                let pixel_colors_2 = _mm_setr_epi32(
                    s_ptr_next.add(src_stride as usize).read_unaligned() as i32,
                    0,
                    0,
                    0,
                );
                let pixel_colors_3 = _mm_setr_epi32(
                    s_ptr_next.add(src_stride as usize * 2).read_unaligned() as i32,
                    0,
                    0,
                    0,
                );
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_loadu_si16x(weight as *const u8);
                store0 = _mm_add_epi32(store0, _mm_madd_epi16(pixel_colors_0, f_weight));
                store1 = _mm_add_epi32(store1, _mm_madd_epi16(pixel_colors_1, f_weight));
                store2 = _mm_add_epi32(store2, _mm_madd_epi16(pixel_colors_2, f_weight));
                store3 = _mm_add_epi32(store3, _mm_madd_epi16(pixel_colors_3, f_weight));

                r += 1;
            }

            let agg0 = _mm_sum_clamp(store0);
            let offset0 = y_dst_shift + x as usize;
            let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset0) as *mut u8;
            dst_ptr0.write_unaligned(agg0 as u8);

            let agg1 = _mm_sum_clamp(store1);
            let offset1 = offset0 + dst_stride as usize;
            let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset1) as *mut u8;
            dst_ptr1.write_unaligned(agg1 as u8);

            let agg2 = _mm_sum_clamp(store2);
            let offset2 = offset1 + dst_stride as usize;
            let dst_ptr2 = unsafe_dst.slice.as_ptr().add(offset2) as *mut u8;
            dst_ptr2.write_unaligned(agg2 as u8);

            let agg3 = _mm_sum_clamp(store3);
            let offset3 = offset2 + dst_stride as usize;
            let dst_ptr3 = unsafe_dst.slice.as_ptr().add(offset3) as *mut u8;
            dst_ptr3.write_unaligned(agg3 as u8);
        }
        _cy = y;
    }

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        unsafe {
            let zeros = _mm_setzero_si128();
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let y_src_shift_next = y_src_shift + src_stride as usize;

                let mut store0 = _mm_setzero_si128();
                let mut store1 = _mm_setzero_si128();

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let value0 = s_ptr.read_unaligned() as i32;
                    let pixel_colors_0 = _mm_setr_epi32(value0, 0, 0, 0);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next); // Here we're always at zero
                    let value1 = s_ptr_next.read_unaligned() as i32;
                    let pixel_colors_1 = _mm_setr_epi32(value1, 0, 0, 0);
                    for i in 0..diff as usize {
                        let weights = kernel.as_ptr().add(i);
                        let f_weight = _mm_loadu_si16x(weights as *const u8);
                        store0 = _mm_add_epi32(store0, _mm_madd_epi16(pixel_colors_0, f_weight));
                        store1 = _mm_add_epi32(store1, _mm_madd_epi16(pixel_colors_1, f_weight));
                    }
                    r += diff as i32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8_0 = _mm_loadu_si128(s_ptr as *const __m128i);
                    let pixel_colors_u8_1 = _mm_loadu_si128(s_ptr_next as *const __m128i);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = _mm_loadu_si128_x2(weight as *const u8);

                    store0 = _mm_add_epi32(
                        store0,
                        _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_0, zeros), weights.0),
                    );
                    store1 = _mm_add_epi32(
                        store1,
                        _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_1, zeros), weights.0),
                    );

                    store0 = _mm_add_epi32(
                        store0,
                        _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8_0, zeros), weights.1),
                    );
                    store1 = _mm_add_epi32(
                        store1,
                        _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8_1, zeros), weights.1),
                    );

                    r += 16;
                }

                while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_u8_0 = _mm_loadu_si64(s_ptr);
                    let pixel_colors_u8_1 = _mm_loadu_si64(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights = _mm_loadu_si128(weight as *const __m128i);

                    store0 = _mm_add_epi32(
                        store0,
                        _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_0, zeros), weights),
                    );
                    store1 = _mm_add_epi32(
                        store1,
                        _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_1, zeros), weights),
                    );

                    r += 8;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = load_u8_s16_fast::<4>(s_ptr);

                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_1 = load_u8_s16_fast::<4>(s_ptr_next);

                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = _mm_loadu_si64(weight as *const u8);
                    store0 = _mm_add_epi32(store0, _mm_madd_epi16(pixel_colors_0, f_weight));
                    store1 = _mm_add_epi32(store1, _mm_madd_epi16(pixel_colors_1, f_weight));

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_0 = _mm_setr_epi32(s_ptr.read_unaligned() as i32, 0, 0, 0);
                    let pixel_colors_1 =
                        _mm_setr_epi32(s_ptr_next.read_unaligned() as i32, 0, 0, 0);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = _mm_loadu_si16x(weight as *const u8);
                    store0 = _mm_add_epi32(store0, _mm_madd_epi16(pixel_colors_0, f_weight));
                    store1 = _mm_add_epi32(store1, _mm_madd_epi16(pixel_colors_1, f_weight));

                    r += 1;
                }

                let agg0 = _mm_sum_clamp(store0);
                let offset0 = y_dst_shift + x as usize;
                let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset0) as *mut u8;
                dst_ptr0.write_unaligned(agg0 as u8);

                let agg1 = _mm_sum_clamp(store1);
                let offset1 = offset0 + dst_stride as usize;
                let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset1) as *mut u8;
                dst_ptr1.write_unaligned(agg1 as u8);
            }
        }
        _cy = y;
    }

    for y in _cy..end_y {
        let zeros = _mm_setzero_si128();
        for x in 0..width {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store = _mm_setzero_si128();

            let mut r = -half_kernel;

            let edge_value_check = x as i64 + r as i64;
            if edge_value_check < 0 {
                let diff = edge_value_check.abs();
                let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                let value = s_ptr.read_unaligned() as i32;
                let pixel_colors = _mm_setr_epi32(value, 0, 0, 0);
                for i in 0..diff as usize {
                    let weights = kernel.as_ptr().add(i);
                    let f_weight = _mm_loadu_si16x(weights as *const u8);
                    store = _mm_add_epi32(store, _mm_madd_epi16(pixel_colors, f_weight));
                }
                r += diff as i32;
            }

            while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_u8 = _mm_loadu_si128(s_ptr as *const __m128i);
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let weights = _mm_loadu_si128_x2(weight as *const u8);

                store = _mm_add_epi32(
                    store,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8, zeros), weights.0),
                );

                store = _mm_add_epi32(
                    store,
                    _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8, zeros), weights.1),
                );

                r += 16;
            }

            while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors = _mm_loadu_si64(s_ptr);
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let weights = _mm_loadu_si128(weight as *const __m128i);

                store = _mm_add_epi32(
                    store,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors, zeros), weights),
                );

                r += 8;
            }

            while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_i32 = load_u8_s16_fast::<4>(s_ptr);
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_loadu_si64(weight as *const u8);
                store = _mm_add_epi32(store, _mm_madd_epi16(pixel_colors_i32, f_weight));

                r += 4;
            }

            while r <= half_kernel {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let value = s_ptr.read_unaligned() as i32;
                let pixel_colors = _mm_setr_epi32(value, 0, 0, 0);
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_loadu_si16x(weight as *const u8);
                store = _mm_add_epi32(store, _mm_madd_epi16(pixel_colors, f_weight));

                r += 1;
            }

            let agg = _mm_sum_clamp(store);
            let offset = y_dst_shift + x as usize;
            let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut u8;
            dst_ptr.write_unaligned(agg as u8);
        }
    }
}

pub fn gaussian_sse_horiz_one_chan_filter_approx_u8(
    undef_src: &[u8],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    filter: &[GaussianFilter<i16>],
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        gaussian_sse_horiz_one_chan_filter_impl(
            undef_src,
            src_stride,
            undef_unsafe_dst,
            dst_stride,
            width,
            filter,
            start_y,
            end_y,
        );
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn gaussian_sse_horiz_one_chan_filter_impl(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    filter: &[GaussianFilter<i16>],
    start_y: u32,
    end_y: u32,
) {
    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        for x in 0..width {
            let current_filter = filter.get_unchecked(x as usize);
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let y_src_shift_next = y_src_shift + src_stride as usize;

            let mut store0 = _mm_setzero_si128();
            let mut store1 = _mm_setzero_si128();

            let zeros = _mm_setzero_si128();

            let mut r = 0;

            while r + 16 < current_filter.size
                && ((filter_start as i64 + r as i64 + 16i64) < width as i64)
            {
                let current_x = std::cmp::min(
                    std::cmp::max(filter_start as i64 + r as i64, 0),
                    (width - 1) as i64,
                ) as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                let pixel_colors_u8_0 = _mm_loadu_si128(s_ptr as *const __m128i);
                let pixel_colors_u8_1 = _mm_loadu_si128(s_ptr_next as *const __m128i);
                let weight = filter_weights.as_ptr().add(r);
                let weights = _mm_loadu_si128_x2(weight as *const u8);

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_0, zeros), weights.0),
                );
                store1 = _mm_add_epi32(
                    store1,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_1, zeros), weights.0),
                );

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8_0, zeros), weights.1),
                );
                store1 = _mm_add_epi32(
                    store1,
                    _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8_1, zeros), weights.1),
                );

                r += 16;
            }

            while r + 8 < current_filter.size
                && ((filter_start as i64 + r as i64 + 8i64) < width as i64)
            {
                let current_x = std::cmp::min(
                    std::cmp::max(filter_start as i64 + r as i64, 0),
                    (width - 1) as i64,
                ) as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                let pixel_colors_u8_0 = _mm_loadu_si64(s_ptr);
                let pixel_colors_u8_1 = _mm_loadu_si64(s_ptr_next);
                let weight = filter_weights.as_ptr().add(r);
                let weights = _mm_loadu_si128(weight as *const __m128i);

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_0, zeros), weights),
                );
                store1 = _mm_add_epi32(
                    store1,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8_1, zeros), weights),
                );

                r += 8;
            }

            while r + 4 < current_filter.size
                && ((filter_start as i64 + r as i64 + 4i64) < width as i64)
            {
                let current_x = std::cmp::min(
                    std::cmp::max(filter_start as i64 + r as i64, 0),
                    (width - 1) as i64,
                ) as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_0 = load_u8_s16_fast::<4>(s_ptr);

                let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                let pixel_colors_1 = load_u8_s16_fast::<4>(s_ptr_next);

                let weight = filter_weights.as_ptr().add(r);
                let f_weight = _mm_loadu_si64(weight as *const u8);
                store0 = _mm_add_epi32(store0, _mm_madd_epi16(pixel_colors_0, f_weight));
                store1 = _mm_add_epi32(store1, _mm_madd_epi16(pixel_colors_1, f_weight));

                r += 4;
            }

            while r < current_filter.size {
                let current_x = std::cmp::min(
                    std::cmp::max(filter_start as i64 + r as i64, 0),
                    (width - 1) as i64,
                ) as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                let pixel_colors_0 = _mm_setr_epi32(s_ptr.read_unaligned() as i32, 0, 0, 0);
                let pixel_colors_1 = _mm_setr_epi32(s_ptr_next.read_unaligned() as i32, 0, 0, 0);
                let weight = filter_weights.as_ptr().add(r);
                let f_weight = _mm_loadu_si16x(weight as *const u8);
                store0 = _mm_add_epi32(store0, _mm_madd_epi16(pixel_colors_0, f_weight));
                store1 = _mm_add_epi32(store1, _mm_madd_epi16(pixel_colors_1, f_weight));

                r += 1;
            }

            let agg0 = _mm_sum_clamp(store0);
            let offset0 = y_dst_shift + x as usize;
            let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset0) as *mut u8;
            dst_ptr0.write_unaligned(agg0 as u8);

            let agg1 = _mm_sum_clamp(store1);
            let offset1 = offset0 + dst_stride as usize;
            let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset1) as *mut u8;
            dst_ptr1.write_unaligned(agg1 as u8);
        }
        _cy = y;
    }

    for y in _cy..end_y {
        let zeros = _mm_setzero_si128();
        for x in 0..width {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let current_filter = filter.get_unchecked(x as usize);
            let filter_start = current_filter.start;
            let filter_weights = &current_filter.filter;

            let mut store = _mm_setzero_si128();

            let mut r = 0;

            while r + 16 < current_filter.size
                && ((filter_start as i64 + r as i64 + 16i64) < width as i64)
            {
                let current_x = std::cmp::min(
                    std::cmp::max(filter_start as i64 + r as i64, 0),
                    (width - 1) as i64,
                ) as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_u8 = _mm_loadu_si128(s_ptr as *const __m128i);
                let weight = filter_weights.as_ptr().add(r);
                let weights = _mm_loadu_si128_x2(weight as *const u8);

                store = _mm_add_epi32(
                    store,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors_u8, zeros), weights.0),
                );

                store = _mm_add_epi32(
                    store,
                    _mm_madd_epi16(_mm_unpackhi_epi8(pixel_colors_u8, zeros), weights.1),
                );

                r += 16;
            }

            while r + 8 < current_filter.size
                && ((filter_start as i64 + r as i64 + 8i64) < width as i64)
            {
                let current_x = std::cmp::min(
                    std::cmp::max(filter_start as i64 + r as i64, 0),
                    (width - 1) as i64,
                ) as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors = _mm_loadu_si64(s_ptr);
                let weight = filter_weights.as_ptr().add(r);
                let weights = _mm_loadu_si128(weight as *const __m128i);

                store = _mm_add_epi32(
                    store,
                    _mm_madd_epi16(_mm_unpacklo_epi8(pixel_colors, zeros), weights),
                );

                r += 8;
            }

            while r + 4 < current_filter.size
                && ((filter_start as i64 + r as i64 + 4i64) < width as i64)
            {
                let current_x = std::cmp::min(
                    std::cmp::max(filter_start as i64 + r as i64, 0),
                    (width - 1) as i64,
                ) as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_i16 = load_u8_s16_fast::<4>(s_ptr);
                let weight = filter_weights.as_ptr().add(r);
                let f_weight = _mm_loadu_si64(weight as *const u8);
                store = _mm_add_epi32(store, _mm_madd_epi16(pixel_colors_i16, f_weight));

                r += 4;
            }

            while r < current_filter.size {
                let current_x = std::cmp::min(
                    std::cmp::max(filter_start as i64 + r as i64, 0),
                    (width - 1) as i64,
                ) as usize;
                let px = current_x;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let value = s_ptr.read_unaligned() as i32;
                let pixel_colors = _mm_setr_epi32(value, 0, 0, 0);
                let weight = filter_weights.as_ptr().add(r).read_unaligned();
                let f_weight = _mm_loadu_si16x(weight as *const u8);
                store = _mm_add_epi32(store, _mm_madd_epi16(pixel_colors, f_weight));

                r += 1;
            }

            let agg = _mm_sum_clamp(store);
            let offset = y_dst_shift + x as usize;
            let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut u8;
            dst_ptr.write_unaligned(agg as u8);
        }
    }
}
