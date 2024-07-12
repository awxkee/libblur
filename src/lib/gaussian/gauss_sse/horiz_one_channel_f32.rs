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

use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::sse::{_mm_hsum_ps, _mm_loadu_ps_x4};
use crate::unsafe_slice::UnsafeSlice;
use erydanos::_mm_prefer_fma_ps;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn gaussian_horiz_one_chan_f32<T>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) {
    let src: &[f32] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, f32> = unsafe { std::mem::transmute(undef_unsafe_dst) };
    let half_kernel = (kernel_size / 2) as i32;

    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let y_src_shift_next = y_src_shift + src_stride as usize;
                let y_dst_shift_next = y_dst_shift + dst_stride as usize;

                let mut store0 = _mm_setzero_ps();
                let mut store1 = _mm_setzero_ps();

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let pixel_colors_f32_0 = _mm_setr_ps(s_ptr.read_unaligned(), 0., 0., 0.);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next); // Here we're always at zero
                    let pixel_colors_f32_1 = _mm_setr_ps(s_ptr_next.read_unaligned(), 0., 0., 0.);
                    for i in 0..diff as usize {
                        let weight = *kernel.get_unchecked(i);
                        let f_weight = _mm_setr_ps(weight, 0., 0., 0.);
                        store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_0, f_weight);
                        store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_1, f_weight);
                    }
                    r += diff as i32;
                }

                while r + 32 <= half_kernel && ((x as i64 + r as i64 + 32i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set0_0 = _mm_loadu_ps_x4(s_ptr);
                    let pixel_colors_f32_set0_1 = _mm_loadu_ps_x4(s_ptr.add(16));

                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);

                    let pixel_colors_f32_set_next_0 = _mm_loadu_ps_x4(s_ptr_next);
                    let pixel_colors_f32_set_next_1 = _mm_loadu_ps_x4(s_ptr_next.add(16));

                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set_0 = _mm_loadu_ps_x4(weight);
                    let weights_set_1 = _mm_loadu_ps_x4(weight.add(16));

                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_0.0, weights_set_0.0);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_0.1, weights_set_0.1);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_0.2, weights_set_0.2);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_0.3, weights_set_0.3);

                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_1.0, weights_set_1.0);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_1.1, weights_set_1.1);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_1.2, weights_set_1.2);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_1.3, weights_set_1.3);

                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_0.0, weights_set_0.0);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_0.1, weights_set_0.1);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_0.2, weights_set_0.2);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_0.3, weights_set_0.3);

                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_1.0, weights_set_1.0);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_1.1, weights_set_1.1);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_1.2, weights_set_1.2);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_1.3, weights_set_1.3);

                    r += 32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = _mm_loadu_ps_x4(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_set_1 = _mm_loadu_ps_x4(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set = _mm_loadu_ps_x4(weight);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set_0.0, weights_set.0);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set_0.1, weights_set.1);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set_0.2, weights_set.2);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set_0.3, weights_set.3);

                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_set_1.0, weights_set.0);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_set_1.1, weights_set.1);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_set_1.2, weights_set.2);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_set_1.3, weights_set.3);

                    r += 16;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_0 = _mm_loadu_ps(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = _mm_loadu_ps(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = _mm_loadu_ps(weight);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_0, f_weight);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_1, f_weight);

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_0 = _mm_setr_ps(s_ptr.read_unaligned(), 0., 0., 0.);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = _mm_setr_ps(s_ptr_next.read_unaligned(), 0., 0., 0.);
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_set1_ps(weight);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_0, f_weight);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_1, f_weight);

                    r += 1;
                }

                let agg0 = _mm_hsum_ps(store0);
                let offset = y_dst_shift + x as usize;
                let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr0.write_unaligned(agg0);

                let agg1 = _mm_hsum_ps(store1);
                let offset = y_dst_shift_next + x as usize;
                let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr1.write_unaligned(agg1);
            }
        }
        _cy = y;
    }

    for y in _cy..end_y {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store = _mm_setzero_ps();

                let mut r = -half_kernel;

                let edge_value_check = x as i64 + r as i64;
                if edge_value_check < 0 {
                    let diff = edge_value_check.abs();
                    let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                    let pixel_colors_f32 = _mm_setr_ps(s_ptr.read_unaligned(), 0., 0., 0.);
                    for i in 0..diff as usize {
                        let weight = *kernel.get_unchecked(i);
                        let f_weight = _mm_setr_ps(weight, 0., 0., 0.);
                        store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);
                    }
                    r += diff as i32;
                }

                while r + 32 <= half_kernel && ((x as i64 + r as i64 + 32i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = _mm_loadu_ps_x4(s_ptr);
                    let pixel_colors_f32_set_1 = _mm_loadu_ps_x4(s_ptr.add(16));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set_0 = _mm_loadu_ps_x4(weight);
                    let weights_set_1 = _mm_loadu_ps_x4(weight.add(16));

                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_0.0, weights_set_0.0);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_0.1, weights_set_0.1);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_0.2, weights_set_0.2);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_0.3, weights_set_0.3);

                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_1.0, weights_set_1.0);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_1.1, weights_set_1.1);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_1.2, weights_set_1.2);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_1.3, weights_set_1.3);

                    r += 32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set = _mm_loadu_ps_x4(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set = _mm_loadu_ps_x4(weight);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set.0, weights_set.0);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set.1, weights_set.1);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set.2, weights_set.2);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set.3, weights_set.3);

                    r += 16;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = _mm_loadu_ps(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight = _mm_loadu_ps(weight);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = _mm_setr_ps(s_ptr.read_unaligned(), 0., 0., 0.);
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = _mm_setr_ps(weight, 0., 0., 0.);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                    r += 1;
                }

                let agg = _mm_hsum_ps(store);
                let offset = y_dst_shift + x as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr.write_unaligned(agg);
            }
        }
    }
}

pub fn gaussian_horiz_one_chan_filter_f32<T>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    filter: &Vec<GaussianFilter>,
    start_y: u32,
    end_y: u32,
) {
    let src: &[f32] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, f32> = unsafe { std::mem::transmute(undef_unsafe_dst) };

    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let y_src_shift_next = y_src_shift + src_stride as usize;
                let y_dst_shift_next = y_dst_shift + dst_stride as usize;

                let mut store0 = _mm_setzero_ps();
                let mut store1 = _mm_setzero_ps();

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut r = 0;

                while r + 32 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 32i64) < width as i64)
                {
                    let current_x = std::cmp::min(
                        std::cmp::max(filter_start as i64 + r as i64, 0),
                        (width - 1) as i64,
                    ) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set0_0 = _mm_loadu_ps_x4(s_ptr);
                    let pixel_colors_f32_set0_1 = _mm_loadu_ps_x4(s_ptr.add(16));

                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);

                    let pixel_colors_f32_set_next_0 = _mm_loadu_ps_x4(s_ptr_next);
                    let pixel_colors_f32_set_next_1 = _mm_loadu_ps_x4(s_ptr_next.add(16));

                    let weight = filter_weights.as_ptr().add(r);
                    let weights_set_0 = _mm_loadu_ps_x4(weight);
                    let weights_set_1 = _mm_loadu_ps_x4(weight.add(16));

                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_0.0, weights_set_0.0);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_0.1, weights_set_0.1);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_0.2, weights_set_0.2);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_0.3, weights_set_0.3);

                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_1.0, weights_set_1.0);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_1.1, weights_set_1.1);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_1.2, weights_set_1.2);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set0_1.3, weights_set_1.3);

                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_0.0, weights_set_0.0);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_0.1, weights_set_0.1);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_0.2, weights_set_0.2);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_0.3, weights_set_0.3);

                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_1.0, weights_set_1.0);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_1.1, weights_set_1.1);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_1.2, weights_set_1.2);
                    store1 =
                        _mm_prefer_fma_ps(store1, pixel_colors_f32_set_next_1.3, weights_set_1.3);

                    r += 32;
                }

                while r + 16 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 16i64) < width as i64)
                {
                    let current_x = std::cmp::min(
                        std::cmp::max(filter_start as i64 + r as i64, 0),
                        (width - 1) as i64,
                    ) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = _mm_loadu_ps_x4(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_set_1 = _mm_loadu_ps_x4(s_ptr_next);
                    let weight = filter_weights.as_ptr().add(r);
                    let weights_set = _mm_loadu_ps_x4(weight);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set_0.0, weights_set.0);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set_0.1, weights_set.1);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set_0.2, weights_set.2);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_set_0.3, weights_set.3);

                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_set_1.0, weights_set.0);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_set_1.1, weights_set.1);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_set_1.2, weights_set.2);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_set_1.3, weights_set.3);

                    r += 16;
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
                    let pixel_colors_f32_0 = _mm_loadu_ps(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = _mm_loadu_ps(s_ptr_next);
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight = _mm_loadu_ps(weight);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_0, f_weight);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_1, f_weight);

                    r += 4;
                }

                while r < current_filter.size {
                    let current_x = std::cmp::min(
                        std::cmp::max(filter_start as i64 + r as i64, 0),
                        (width - 1) as i64,
                    ) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_0 = _mm_setr_ps(s_ptr.read_unaligned(), 0., 0., 0.);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = _mm_setr_ps(s_ptr_next.read_unaligned(), 0., 0., 0.);
                    let weight = filter_weights.as_ptr().add(r).read_unaligned();
                    let f_weight = _mm_set1_ps(weight);
                    store0 = _mm_prefer_fma_ps(store0, pixel_colors_f32_0, f_weight);
                    store1 = _mm_prefer_fma_ps(store1, pixel_colors_f32_1, f_weight);

                    r += 1;
                }

                let agg0 = _mm_hsum_ps(store0);
                let offset = y_dst_shift + x as usize;
                let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr0.write_unaligned(agg0);

                let agg1 = _mm_hsum_ps(store1);
                let offset = y_dst_shift_next + x as usize;
                let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr1.write_unaligned(agg1);
            }
        }
        _cy = y;
    }

    for y in _cy..end_y {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store = _mm_setzero_ps();

                let current_filter = filter.get_unchecked(x as usize);
                let filter_start = current_filter.start;
                let filter_weights = &current_filter.filter;

                let mut r = 0usize;

                while r + 32 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 32i64) < width as i64)
                {
                    let current_x = std::cmp::min(
                        std::cmp::max(filter_start as i64 + r as i64, 0),
                        (width - 1) as i64,
                    ) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = _mm_loadu_ps_x4(s_ptr);
                    let pixel_colors_f32_set_1 = _mm_loadu_ps_x4(s_ptr.add(16));
                    let weight = filter_weights.as_ptr().add(r);
                    let weights_set_0 = _mm_loadu_ps_x4(weight);
                    let weights_set_1 = _mm_loadu_ps_x4(weight.add(16));

                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_0.0, weights_set_0.0);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_0.1, weights_set_0.1);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_0.2, weights_set_0.2);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_0.3, weights_set_0.3);

                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_1.0, weights_set_1.0);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_1.1, weights_set_1.1);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_1.2, weights_set_1.2);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set_1.3, weights_set_1.3);

                    r += 32;
                }

                while r + 16 < current_filter.size
                    && ((filter_start as i64 + r as i64 + 16i64) < width as i64)
                {
                    let current_x = std::cmp::min(
                        std::cmp::max(filter_start as i64 + r as i64, 0),
                        (width - 1) as i64,
                    ) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set = _mm_loadu_ps_x4(s_ptr);
                    let weight = filter_weights.as_ptr().add(r);
                    let weights_set = _mm_loadu_ps_x4(weight);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set.0, weights_set.0);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set.1, weights_set.1);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set.2, weights_set.2);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32_set.3, weights_set.3);

                    r += 16;
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
                    let pixel_colors_f32 = _mm_loadu_ps(s_ptr);
                    let weight = filter_weights.as_ptr().add(r);
                    let f_weight = _mm_loadu_ps(weight);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                    r += 4;
                }

                while r < current_filter.size {
                    let current_x = std::cmp::min(
                        std::cmp::max(filter_start as i64 + r as i64, 0),
                        (width - 1) as i64,
                    ) as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = _mm_setr_ps(s_ptr.read_unaligned(), 0., 0., 0.);
                    let weight = filter_weights.as_ptr().add(r).read_unaligned();
                    let f_weight = _mm_setr_ps(weight, 0., 0., 0.);
                    store = _mm_prefer_fma_ps(store, pixel_colors_f32, f_weight);

                    r += 1;
                }

                let agg = _mm_hsum_ps(store);
                let offset = y_dst_shift + x as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr.write_unaligned(agg);
            }
        }
    }
}
