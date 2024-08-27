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

use crate::gaussian::sse::gauss_utils::_mm_opt_fma_ps;
use crate::sse::{
    _mm_broadcast_first, _mm_broadcast_fourth, _mm_broadcast_second, _mm_broadcast_third,
    _mm_loadu_ps_x4, _mm_split_rgb_5_ps, load_f32, store_f32,
};
use crate::unsafe_slice::UnsafeSlice;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

macro_rules! accumulate_5_items {
    ($store0:expr, $pixel_colors_0:expr, $f_weights:expr, $last_weight:expr, $fma: expr) => {{
        $store0 =
            _mm_opt_fma_ps::<$fma>($store0, $pixel_colors_0.0, _mm_broadcast_first($f_weights));
        $store0 =
            _mm_opt_fma_ps::<$fma>($store0, $pixel_colors_0.1, _mm_broadcast_second($f_weights));
        $store0 =
            _mm_opt_fma_ps::<$fma>($store0, $pixel_colors_0.2, _mm_broadcast_third($f_weights));
        $store0 =
            _mm_opt_fma_ps::<$fma>($store0, $pixel_colors_0.3, _mm_broadcast_fourth($f_weights));
        $store0 = _mm_opt_fma_ps::<$fma>($store0, $pixel_colors_0.4, _mm_set1_ps($last_weight));
    }};
}

macro_rules! accumulate_4_items {
    ($store0:expr, $pixel_colors_0:expr, $f_weights:expr, $fma: expr) => {{
        $store0 =
            _mm_opt_fma_ps::<$fma>($store0, $pixel_colors_0.0, _mm_broadcast_first($f_weights));
        $store0 =
            _mm_opt_fma_ps::<$fma>($store0, $pixel_colors_0.1, _mm_broadcast_second($f_weights));
        $store0 =
            _mm_opt_fma_ps::<$fma>($store0, $pixel_colors_0.2, _mm_broadcast_third($f_weights));
        $store0 =
            _mm_opt_fma_ps::<$fma>($store0, $pixel_colors_0.3, _mm_broadcast_fourth($f_weights));
    }};
}

pub fn gaussian_horiz_sse_t_f_chan_f32<T, const CHANNEL_CONFIGURATION: usize, const FMA: bool>(
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
    unsafe {
        if FMA {
            gaussian_horiz_sse_t_f_chan_fma::<T, CHANNEL_CONFIGURATION>(
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
        } else {
            gaussian_horiz_sse_t_f_chan_gen::<T, CHANNEL_CONFIGURATION>(
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
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn gaussian_horiz_sse_t_f_chan_gen<T, const CHANNEL_CONFIGURATION: usize>(
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
    gaussian_horiz_sse_t_f_chan_f32_impl::<T, CHANNEL_CONFIGURATION, false>(
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

#[inline]
#[target_feature(enable = "sse4.1,fma")]
unsafe fn gaussian_horiz_sse_t_f_chan_fma<T, const CHANNEL_CONFIGURATION: usize>(
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
    gaussian_horiz_sse_t_f_chan_f32_impl::<T, CHANNEL_CONFIGURATION, true>(
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

unsafe fn gaussian_horiz_sse_t_f_chan_f32_impl<
    T,
    const CHANNEL_CONFIGURATION: usize,
    const FMA: bool,
>(
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
    let src: &[f32] = std::mem::transmute(undef_src);
    let unsafe_dst: &UnsafeSlice<'_, f32> = std::mem::transmute(undef_unsafe_dst);
    let half_kernel = (kernel_size / 2) as i32;

    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        for x in 0..width {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store0 = _mm_setzero_ps();
            let mut store1 = _mm_setzero_ps();

            let mut r = -half_kernel;

            let edge_value_check = x as i64 + r as i64;
            if edge_value_check < 0 {
                let diff = edge_value_check.abs();
                let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                let pixel_colors_0 = load_f32::<CHANNEL_CONFIGURATION>(s_ptr);
                let s_ptr_next = src.as_ptr().add(y_src_shift + src_stride as usize); // Here we're always at zero
                let pixel_colors_1 = load_f32::<CHANNEL_CONFIGURATION>(s_ptr_next);
                for i in 0..diff as usize {
                    let weights = kernel.as_ptr().add(i);
                    let f_weight = _mm_load1_ps(weights);
                    store0 = _mm_opt_fma_ps::<FMA>(store0, pixel_colors_0, f_weight);
                    store1 = _mm_opt_fma_ps::<FMA>(store1, pixel_colors_1, f_weight);
                }
                r += diff as i32;
            }

            if CHANNEL_CONFIGURATION == 4 {
                while r + 8 <= half_kernel && ((x as i64 + r as i64 + 8i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = _mm_loadu_ps_x4(s_ptr);
                    let pixel_colors_1 = _mm_loadu_ps_x4(s_ptr.add(src_stride as usize));
                    let pixel_colors_n_0 = _mm_loadu_ps_x4(s_ptr.add(16));
                    let pixel_colors_n_1 = _mm_loadu_ps_x4(s_ptr.add(src_stride as usize).add(16));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);

                    let f_weights = _mm_loadu_ps(weight);
                    let f_weights_n = _mm_loadu_ps(weight.add(4));

                    accumulate_4_items!(store0, pixel_colors_0, f_weights, FMA);
                    accumulate_4_items!(store1, pixel_colors_1, f_weights, FMA);

                    accumulate_4_items!(store0, pixel_colors_n_0, f_weights_n, FMA);
                    accumulate_4_items!(store1, pixel_colors_n_1, f_weights_n, FMA);

                    r += 8;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_0 = _mm_loadu_ps_x4(s_ptr);
                    let pixel_colors_1 = _mm_loadu_ps_x4(s_ptr.add(src_stride as usize));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);

                    let f_weights = _mm_loadu_ps(weight);

                    accumulate_4_items!(store0, pixel_colors_0, f_weights, FMA);
                    accumulate_4_items!(store1, pixel_colors_1, f_weights, FMA);

                    r += 4;
                }
            } else if CHANNEL_CONFIGURATION == 3 {
                while r + 5 <= half_kernel && ((x as i64 + r as i64 + 6i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    // (r0 g0 b0 r1) (g1 b1 r2 g2) (b2 r3 g3 b3) (r4 g4 b5 undef)
                    let pixel_colors_0 = _mm_loadu_ps_x4(s_ptr);
                    let pixel_colors_1 = _mm_loadu_ps_x4(s_ptr.add(src_stride as usize));
                    // r0 g0 b0 0
                    let set0 = _mm_split_rgb_5_ps(pixel_colors_0);
                    let set1 = _mm_split_rgb_5_ps(pixel_colors_1);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weights = _mm_loadu_ps(weight);
                    let last_weight = weight.add(4).read_unaligned();

                    accumulate_5_items!(store0, set0, f_weights, last_weight, FMA);
                    accumulate_5_items!(store1, set1, f_weights, last_weight, FMA);

                    r += 5;
                }
            }

            while r <= half_kernel {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x * CHANNEL_CONFIGURATION;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_0 = load_f32::<CHANNEL_CONFIGURATION>(s_ptr);
                let s_ptr_next = s_ptr.add(src_stride as usize);
                let pixel_colors_1 = load_f32::<CHANNEL_CONFIGURATION>(s_ptr_next);
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_load1_ps(weight);
                store0 = _mm_opt_fma_ps::<FMA>(store0, pixel_colors_0, f_weight);
                store1 = _mm_opt_fma_ps::<FMA>(store1, pixel_colors_1, f_weight);

                r += 1;
            }

            let offset = y_dst_shift + x as usize * CHANNEL_CONFIGURATION;
            let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
            store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store0);

            let offset = offset + dst_stride as usize;
            let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
            store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store1);
        }
        _cy = y;
    }

    for y in _cy..end_y {
        for x in 0..width {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store = _mm_setzero_ps();

            let mut r = -half_kernel;

            let edge_value_check = x as i64 + r as i64;
            if edge_value_check < 0 {
                let diff = edge_value_check.abs();
                let s_ptr = src.as_ptr().add(y_src_shift); // Here we're always at zero
                let pixel_colors_f32 = load_f32::<CHANNEL_CONFIGURATION>(s_ptr);
                for i in 0..diff as usize {
                    let weights = kernel.as_ptr().add(i);
                    let f_weight = _mm_load1_ps(weights);
                    store = _mm_opt_fma_ps::<FMA>(store, pixel_colors_f32, f_weight);
                }
                r += diff as i32;
            }

            if CHANNEL_CONFIGURATION == 4 {
                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let px = _mm_loadu_ps_x4(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);

                    let f_weights = _mm_loadu_ps(weight);

                    accumulate_4_items!(store, px, f_weights, FMA);

                    r += 4;
                }
            } else if CHANNEL_CONFIGURATION == 3 {
                while r + 5 <= half_kernel && ((x as i64 + r as i64 + 6i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x * CHANNEL_CONFIGURATION;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    // (r0 g0 b0 r1) (g1 b1 r2 g2) (b2 r3 g3 b3) (r4 g4 b5 undef)
                    let pixel_colors_o = _mm_loadu_ps_x4(s_ptr);
                    // r0 g0 b0 0
                    let pixel_set = _mm_split_rgb_5_ps(pixel_colors_o);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weights = _mm_loadu_ps(weight);
                    let last_weight = weight.add(4).read_unaligned();

                    accumulate_5_items!(store, pixel_set, f_weights, last_weight, FMA);

                    r += 5;
                }
            }

            while r <= half_kernel {
                let current_x =
                    std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                        as usize;
                let px = current_x * CHANNEL_CONFIGURATION;
                let s_ptr = src.as_ptr().add(y_src_shift + px);
                let pixel_colors_f32 = load_f32::<CHANNEL_CONFIGURATION>(s_ptr);
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_load1_ps(weight);
                store = _mm_opt_fma_ps::<FMA>(store, pixel_colors_f32, f_weight);

                r += 1;
            }

            let offset = y_dst_shift + x as usize * CHANNEL_CONFIGURATION;
            let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
            store_f32::<CHANNEL_CONFIGURATION>(dst_ptr, store);
        }
    }
}
