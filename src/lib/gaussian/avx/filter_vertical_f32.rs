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
use crate::unsafe_slice::UnsafeSlice;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::gaussian::avx::utils::_mm256_opt_fma_ps;
use crate::gaussian::gauss_sse::_mm_opt_fma_ps;

pub fn gaussian_blur_vertical_pass_filter_f32_avx<
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
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        if FMA {
            gaussian_blur_vertical_pass_filter_fma::<T, CHANNEL_CONFIGURATION>(
                undef_src,
                src_stride,
                undef_unsafe_dst,
                dst_stride,
                width,
                height,
                filter,
                start_y,
                end_y,
            );
        } else {
            gaussian_blur_vertical_pass_filter_gen::<T, CHANNEL_CONFIGURATION>(
                undef_src,
                src_stride,
                undef_unsafe_dst,
                dst_stride,
                width,
                height,
                filter,
                start_y,
                end_y,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn gaussian_blur_vertical_pass_filter_gen<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    gaussian_blur_vertical_pass_filter_impl::<T, CHANNEL_CONFIGURATION, false>(
        undef_src,
        src_stride,
        undef_unsafe_dst,
        dst_stride,
        width,
        height,
        filter,
        start_y,
        end_y,
    );
}

#[target_feature(enable = "avx2,fma")]
unsafe fn gaussian_blur_vertical_pass_filter_fma<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    gaussian_blur_vertical_pass_filter_impl::<T, CHANNEL_CONFIGURATION, true>(
        undef_src,
        src_stride,
        undef_unsafe_dst,
        dst_stride,
        width,
        height,
        filter,
        start_y,
        end_y,
    );
}

unsafe fn gaussian_blur_vertical_pass_filter_impl<
    T,
    const CHANNEL_CONFIGURATION: usize,
    const FMA: bool,
>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    _: u32,
    filter: &Vec<GaussianFilter<f32>>,
    start_y: u32,
    end_y: u32,
) {
    let src: &[f32] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, f32> = unsafe { std::mem::transmute(undef_unsafe_dst) };

    let zeros = unsafe { _mm256_setzero_ps() };

    let total_length = width as usize * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut cx = 0usize;

        let current_filter = unsafe { filter.get_unchecked(y as usize) };
        let filter_start = current_filter.start;
        let filter_weights = &current_filter.filter;

        while cx + 64 < total_length {
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
                let f_weight = _mm256_set1_ps(weight);

                let py = filter_start + j;
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let px_0 = _mm256_loadu_ps(s_ptr);
                let px_1 = _mm256_loadu_ps(s_ptr.add(8));
                let px_2 = _mm256_loadu_ps(s_ptr.add(16));
                let px_3 = _mm256_loadu_ps(s_ptr.add(24));
                let px_4 = _mm256_loadu_ps(s_ptr.add(32));
                let px_5 = _mm256_loadu_ps(s_ptr.add(40));
                let px_6 = _mm256_loadu_ps(s_ptr.add(48));
                let px_7 = _mm256_loadu_ps(s_ptr.add(56));
                store0 = _mm256_opt_fma_ps::<FMA>(store0, px_0, f_weight);
                store1 = _mm256_opt_fma_ps::<FMA>(store1, px_1, f_weight);
                store2 = _mm256_opt_fma_ps::<FMA>(store2, px_2, f_weight);
                store3 = _mm256_opt_fma_ps::<FMA>(store3, px_3, f_weight);
                store4 = _mm256_opt_fma_ps::<FMA>(store4, px_4, f_weight);
                store5 = _mm256_opt_fma_ps::<FMA>(store5, px_5, f_weight);
                store6 = _mm256_opt_fma_ps::<FMA>(store6, px_6, f_weight);
                store7 = _mm256_opt_fma_ps::<FMA>(store7, px_7, f_weight);

                j += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            _mm256_storeu_ps(dst_ptr, store0);
            _mm256_storeu_ps(dst_ptr.add(8), store1);
            _mm256_storeu_ps(dst_ptr.add(16), store2);
            _mm256_storeu_ps(dst_ptr.add(24), store3);
            _mm256_storeu_ps(dst_ptr.add(32), store4);
            _mm256_storeu_ps(dst_ptr.add(40), store5);
            _mm256_storeu_ps(dst_ptr.add(48), store6);
            _mm256_storeu_ps(dst_ptr.add(56), store7);

            cx += 64;
        }

        while cx + 32 < total_length {
            let mut store0 = zeros;
            let mut store1 = zeros;
            let mut store2 = zeros;
            let mut store3 = zeros;

            let mut j = 0usize;
            while j < current_filter.size {
                let weight = *filter_weights.get_unchecked(j);
                let f_weight = _mm256_set1_ps(weight);

                let py = filter_start + j;
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let px_0 = _mm256_loadu_ps(s_ptr);
                let px_1 = _mm256_loadu_ps(s_ptr.add(8));
                let px_2 = _mm256_loadu_ps(s_ptr.add(16));
                let px_3 = _mm256_loadu_ps(s_ptr.add(24));
                store0 = _mm256_opt_fma_ps::<FMA>(store0, px_0, f_weight);
                store1 = _mm256_opt_fma_ps::<FMA>(store1, px_1, f_weight);
                store2 = _mm256_opt_fma_ps::<FMA>(store2, px_2, f_weight);
                store3 = _mm256_opt_fma_ps::<FMA>(store3, px_3, f_weight);

                j += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            _mm256_storeu_ps(dst_ptr, store0);
            _mm256_storeu_ps(dst_ptr.add(8), store1);
            _mm256_storeu_ps(dst_ptr.add(16), store2);
            _mm256_storeu_ps(dst_ptr.add(24), store3);

            cx += 32;
        }

        while cx + 16 < total_length {
            let mut store0 = zeros;
            let mut store1 = zeros;

            let mut j = 0usize;
            while j < current_filter.size {
                let weight = *filter_weights.get_unchecked(j);
                let f_weight = _mm256_set1_ps(weight);

                let py = filter_start + j;
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let px_0 = _mm256_loadu_ps(s_ptr);
                let px_1 = _mm256_loadu_ps(s_ptr.add(8));
                store0 = _mm256_opt_fma_ps::<FMA>(store0, px_0, f_weight);
                store1 = _mm256_opt_fma_ps::<FMA>(store1, px_1, f_weight);

                j += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            _mm256_storeu_ps(dst_ptr, store0);
            _mm256_storeu_ps(dst_ptr.add(8), store1);

            cx += 16;
        }

        while cx + 8 < total_length {
            let mut store0 = zeros;

            let mut j = 0usize;
            while j < current_filter.size {
                let weight = *filter_weights.get_unchecked(j);
                let f_weight = _mm256_set1_ps(weight);

                let py = filter_start + j;
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let px = _mm256_loadu_ps(s_ptr);
                store0 = _mm256_opt_fma_ps::<FMA>(store0, px, f_weight);

                j += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            _mm256_storeu_ps(dst_ptr, store0);

            cx += 8;
        }

        while cx + 4 < total_length {
            let mut store0 = _mm_setzero_ps();

            let mut j = 0usize;
            while j < current_filter.size {
                let weight = *filter_weights.get_unchecked(j);
                let f_weight = _mm_set1_ps(weight);

                let py = filter_start + j;
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let lo_lo = _mm_loadu_ps(s_ptr);
                store0 = _mm_opt_fma_ps::<FMA>(store0, lo_lo, f_weight);

                j += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            _mm_storeu_ps(dst_ptr, store0);

            cx += 4;
        }

        while cx < total_length {
            let mut store0 = _mm_setzero_ps();

            let mut j = 0usize;
            while j < current_filter.size {
                let weight = *filter_weights.get_unchecked(j);
                let f_weight = _mm_set1_ps(weight);

                let py = filter_start + j;
                let y_src_shift = py * src_stride as usize;
                let s_ptr = src.as_ptr().add(y_src_shift + cx);
                let f_pixel = _mm_setr_ps(s_ptr.read_unaligned(), 0., 0., 0.);
                store0 = _mm_opt_fma_ps::<FMA>(store0, f_pixel, f_weight);

                j += 1;
            }

            let dst_ptr = (unsafe_dst.slice.as_ptr() as *mut f32).add(y_dst_shift + cx);
            let pixel = _mm_extract_ps::<0>(store0);
            (dst_ptr as *mut i32).write_unaligned(pixel);

            cx += 1;
        }
    }
}
