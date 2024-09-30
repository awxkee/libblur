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

use crate::unsafe_slice::UnsafeSlice;
use crate::{clamp_edge, reflect_101, reflect_index, EdgeMode};
use erydanos::_mm_prefer_fma_ps;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn gaussian_blur_vertical_pass_impl_f32_sse<
    T,
    const CHANNEL_CONFIGURATION: usize,
    const FMA: bool,
>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
    edge_mode: EdgeMode,
) {
    unsafe {
        if FMA {
            gaussian_blur_vertical_pass_impl_f32_sse_fma::<T, CHANNEL_CONFIGURATION>(
                undef_src,
                src_stride,
                undef_unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                start_y,
                end_y,
                edge_mode,
            );
        } else {
            gaussian_blur_vertical_pass_impl_f32_sse_gen::<T, CHANNEL_CONFIGURATION>(
                undef_src,
                src_stride,
                undef_unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                start_y,
                end_y,
                edge_mode,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn gaussian_blur_vertical_pass_impl_f32_sse_gen<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
    edge_mode: EdgeMode,
) {
    gaussian_blur_vertical_pass_impl_f32_sse_impl::<T, CHANNEL_CONFIGURATION, false>(
        undef_src,
        src_stride,
        undef_unsafe_dst,
        dst_stride,
        width,
        height,
        kernel_size,
        kernel,
        start_y,
        end_y,
        edge_mode,
    );
}

#[inline]
#[target_feature(enable = "sse4.1,fma")]
unsafe fn gaussian_blur_vertical_pass_impl_f32_sse_fma<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
    edge_mode: EdgeMode,
) {
    gaussian_blur_vertical_pass_impl_f32_sse_impl::<T, CHANNEL_CONFIGURATION, true>(
        undef_src,
        src_stride,
        undef_unsafe_dst,
        dst_stride,
        width,
        height,
        kernel_size,
        kernel,
        start_y,
        end_y,
        edge_mode,
    );
}

unsafe fn gaussian_blur_vertical_pass_impl_f32_sse_impl<
    T,
    const CHANNEL_CONFIGURATION: usize,
    const FMA: bool,
>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
    edge_mode: EdgeMode,
) {
    let src: &[f32] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undef_unsafe_dst) };
    let half_kernel = (kernel_size / 2) as i32;

    let zeros = unsafe { _mm_setzero_ps() };
    let total_length = width as usize * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut cx = 0usize;

        while cx + 24 < total_length {
            let mut store0 = zeros;
            let mut store1 = zeros;
            let mut store2 = zeros;
            let mut store3 = zeros;
            let mut store4 = zeros;
            let mut store5 = zeros;

            let mut r = -half_kernel;
            while r <= half_kernel {
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_load1_ps(weight);

                let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let px_0 = _mm_loadu_ps(s_ptr);
                let px_1 = _mm_loadu_ps(s_ptr.add(4));
                let px_2 = _mm_loadu_ps(s_ptr.add(8));
                let px_3 = _mm_loadu_ps(s_ptr.add(12));
                let px_4 = _mm_loadu_ps(s_ptr.add(16));
                let px_5 = _mm_loadu_ps(s_ptr.add(20));
                store0 = _mm_prefer_fma_ps(store0, px_0, f_weight);
                store1 = _mm_prefer_fma_ps(store1, px_1, f_weight);
                store2 = _mm_prefer_fma_ps(store2, px_2, f_weight);
                store3 = _mm_prefer_fma_ps(store3, px_3, f_weight);
                store4 = _mm_prefer_fma_ps(store4, px_4, f_weight);
                store5 = _mm_prefer_fma_ps(store5, px_5, f_weight);

                r += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            _mm_storeu_ps(dst_ptr, store0);
            _mm_storeu_ps(dst_ptr.add(4), store1);
            _mm_storeu_ps(dst_ptr.add(8), store2);
            _mm_storeu_ps(dst_ptr.add(12), store3);
            _mm_storeu_ps(dst_ptr.add(16), store4);
            _mm_storeu_ps(dst_ptr.add(20), store5);

            cx += 24;
        }

        while cx + 16 < total_length {
            let mut store0 = zeros;
            let mut store1 = zeros;
            let mut store2 = zeros;
            let mut store3 = zeros;

            let mut r = -half_kernel;
            while r <= half_kernel {
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_load1_ps(weight);

                let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let px_0 = _mm_loadu_ps(s_ptr);
                let px_1 = _mm_loadu_ps(s_ptr.add(4));
                let px_2 = _mm_loadu_ps(s_ptr.add(8));
                let px_3 = _mm_loadu_ps(s_ptr.add(12));
                store0 = _mm_prefer_fma_ps(store0, px_0, f_weight);
                store1 = _mm_prefer_fma_ps(store1, px_1, f_weight);
                store2 = _mm_prefer_fma_ps(store2, px_2, f_weight);
                store3 = _mm_prefer_fma_ps(store3, px_3, f_weight);

                r += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            _mm_storeu_ps(dst_ptr, store0);
            _mm_storeu_ps(dst_ptr.add(4), store1);
            _mm_storeu_ps(dst_ptr.add(8), store2);
            _mm_storeu_ps(dst_ptr.add(12), store3);

            cx += 16;
        }

        while cx + 8 < total_length {
            let mut store0 = zeros;
            let mut store1 = zeros;

            let mut r = -half_kernel;
            while r <= half_kernel {
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_load1_ps(weight);

                let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                let y_src_shift = py as usize * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let px_0 = _mm_loadu_ps(s_ptr);
                let px_1 = _mm_loadu_ps(s_ptr.add(4));
                store0 = _mm_prefer_fma_ps(store0, px_0, f_weight);
                store1 = _mm_prefer_fma_ps(store1, px_1, f_weight);
                r += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            _mm_storeu_ps(dst_ptr, store0);
            _mm_storeu_ps(dst_ptr.add(4), store1);

            cx += 8;
        }

        while cx + 4 < total_length {
            let mut store0 = zeros;

            let mut r = -half_kernel;
            while r <= half_kernel {
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_load1_ps(weight);

                let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let lo_lo = _mm_loadu_ps(s_ptr);
                store0 = _mm_prefer_fma_ps(store0, lo_lo, f_weight);

                r += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            _mm_storeu_ps(dst_ptr, store0);

            cx += 4;
        }

        while cx < total_length {
            let mut store0 = zeros;

            let mut r = -half_kernel;
            while r <= half_kernel {
                let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                let f_weight = _mm_load1_ps(weight);

                let py = clamp_edge!(edge_mode, y as i64 + r as i64, 0i64, height as i64 - 1);
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let f_pixel = _mm_setr_ps(s_ptr.read_unaligned(), 0., 0., 0.);
                store0 = _mm_prefer_fma_ps(store0, f_pixel, f_weight);

                r += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);

            let pixel = _mm_extract_ps::<0>(store0);
            (dst_ptr as *mut i32).write_unaligned(pixel);

            cx += 1;
        }
    }
}
