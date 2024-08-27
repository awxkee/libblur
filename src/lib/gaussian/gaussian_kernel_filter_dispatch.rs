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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::gaussian::avx::gaussian_blur_vertical_pass_filter_f32_avx;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::gaussian::neon::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::gaussian::neon::{
    gaussian_blur_horizontal_pass_filter_neon, gaussian_blur_vertical_pass_filter_neon,
};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::gaussian::sse::*;
use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::gaussian::gaussian_horizontal::gaussian_blur_horizontal_pass_impl_clip_edge;
use crate::gaussian::gaussian_vertical::gaussian_blur_vertical_pass_clip_edge_impl;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::{AsPrimitive, FromPrimitive};
use rayon::ThreadPool;

pub(crate) fn gaussian_blur_vertical_pass_edge_clip_dispatch<
    T: FromPrimitive + Default + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter<f32>>,
    thread_pool: &Option<ThreadPool>,
    thread_count: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + 'static + AsPrimitive<f32>,
    f32: ToStorage<T>,
{
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_fma_available = std::arch::is_x86_feature_detected!("fma");
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_avx_available = std::arch::is_x86_feature_detected!("avx2");
    let mut _dispatcher: fn(
        src: &[T],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        width: u32,
        height: u32,
        filter: &Vec<GaussianFilter<f32>>,
        start_y: u32,
        end_y: u32,
    ) = gaussian_blur_vertical_pass_clip_edge_impl::<T, CHANNEL_CONFIGURATION>;
    if std::any::type_name::<T>() == "u8" {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channels
            if _is_sse_available {
                if _is_fma_available {
                    _dispatcher =
                        gaussian_blur_vertical_pass_filter_sse::<T, CHANNEL_CONFIGURATION, true>;
                } else {
                    _dispatcher =
                        gaussian_blur_vertical_pass_filter_sse::<T, CHANNEL_CONFIGURATION, false>;
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channels
            _dispatcher = gaussian_blur_vertical_pass_filter_neon::<T, CHANNEL_CONFIGURATION>;
        }
    }
    if std::any::type_name::<T>() == "f32" {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channels
            _dispatcher = gaussian_blur_vertical_pass_filter_f32_neon::<T, CHANNEL_CONFIGURATION>;
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channels
            if _is_sse_available {
                if _is_fma_available {
                    _dispatcher = gaussian_blur_vertical_pass_filter_f32_sse::<
                        T,
                        CHANNEL_CONFIGURATION,
                        true,
                    >;
                } else {
                    _dispatcher = gaussian_blur_vertical_pass_filter_f32_sse::<
                        T,
                        CHANNEL_CONFIGURATION,
                        false,
                    >;
                }
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channels
            if _is_avx_available {
                if _is_fma_available {
                    _dispatcher = gaussian_blur_vertical_pass_filter_f32_avx::<
                        T,
                        CHANNEL_CONFIGURATION,
                        true,
                    >;
                } else {
                    _dispatcher = gaussian_blur_vertical_pass_filter_f32_avx::<
                        T,
                        CHANNEL_CONFIGURATION,
                        false,
                    >;
                }
            }
        }
    }
    let unsafe_dst = UnsafeSlice::new(dst);
    if let Some(thread_pool) = thread_pool {
        thread_pool.scope(|scope| {
            let segment_size = height / thread_count;

            for i in 0..thread_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_y = height;
                }

                scope.spawn(move |_| {
                    _dispatcher(
                        src,
                        src_stride,
                        &unsafe_dst,
                        dst_stride,
                        width,
                        height,
                        filter,
                        start_y,
                        end_y,
                    );
                });
            }
        });
    } else {
        _dispatcher(
            src,
            src_stride,
            &unsafe_dst,
            dst_stride,
            width,
            height,
            filter,
            0,
            height,
        );
    }
}

pub(crate) fn gaussian_blur_horizontal_pass_edge_clip_dispatch<
    T: FromPrimitive + Default + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter<f32>>,
    thread_pool: &Option<ThreadPool>,
    thread_count: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + 'static + AsPrimitive<f32>,
    f32: AsPrimitive<T> + ToStorage<T>,
{
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_fma_available = std::arch::is_x86_feature_detected!("fma");

    let mut _dispatcher: fn(
        src: &[T],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        width: u32,
        filter: &Vec<GaussianFilter<f32>>,
        start_y: u32,
        end_y: u32,
    ) = gaussian_blur_horizontal_pass_impl_clip_edge::<T, CHANNEL_CONFIGURATION>;
    if CHANNEL_CONFIGURATION >= 3 {
        if std::any::type_name::<T>() == "u8" {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if _is_sse_available {
                    if _is_fma_available {
                        _dispatcher = gaussian_blur_horizontal_pass_filter_sse::<
                            T,
                            CHANNEL_CONFIGURATION,
                            true,
                        >;
                    } else {
                        _dispatcher = gaussian_blur_horizontal_pass_filter_sse::<
                            T,
                            CHANNEL_CONFIGURATION,
                            false,
                        >;
                    }
                }
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_blur_horizontal_pass_filter_neon::<T, CHANNEL_CONFIGURATION>;
            }
        }
    }
    if std::any::type_name::<T>() == "f32" {
        if CHANNEL_CONFIGURATION == 1 {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_horiz_one_chan_filter_f32::<T>;
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if _is_sse_available {
                    _dispatcher = gaussian_horiz_one_chan_filter_f32::<T>;
                }
            }
        } else if CHANNEL_CONFIGURATION == 3 || CHANNEL_CONFIGURATION == 4 {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_horiz_t_f_chan_filter_f32::<T, CHANNEL_CONFIGURATION>;
            }
        }
    }
    if std::any::type_name::<T>() == "u8" {
        if CHANNEL_CONFIGURATION == 1 {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_horiz_one_chan_filter_u8::<T>;
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if _is_sse_available {
                    _dispatcher = gaussian_sse_horiz_one_chan_filter_u8::<T>;
                }
            }
        }
    }
    let unsafe_dst = UnsafeSlice::new(dst);
    if let Some(thread_pool) = thread_pool {
        thread_pool.scope(|scope| {
            let segment_size = height / thread_count;
            for i in 0..thread_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_y = height;
                }

                scope.spawn(move |_| {
                    _dispatcher(
                        src,
                        src_stride,
                        &unsafe_dst,
                        dst_stride,
                        width,
                        filter,
                        start_y,
                        end_y,
                    );
                });
            }
        });
    } else {
        _dispatcher(
            src,
            src_stride,
            &unsafe_dst,
            dst_stride,
            width,
            filter,
            0,
            height,
        );
    }
}
