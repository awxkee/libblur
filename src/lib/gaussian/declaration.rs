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

use half::f16;
use num_traits::cast::FromPrimitive;
use num_traits::AsPrimitive;
use rayon::ThreadPool;

use crate::channels_configuration::FastBlurChannels;
use crate::edge_mode::EdgeMode;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::gaussian::avx::{
    gaussian_blur_vertical_pass_impl_avx, gaussian_blur_vertical_pass_impl_f32_avx,
};
use crate::gaussian::gaussian_approx_dispatch::gaussian_blur_impl_approx;
use crate::gaussian::gaussian_filter::create_filter;
use crate::gaussian::gaussian_horizontal::gaussian_blur_horizontal_pass_impl;
use crate::gaussian::gaussian_kernel::get_gaussian_kernel_1d;
use crate::gaussian::gaussian_kernel_filter_dispatch::{
    gaussian_blur_horizontal_pass_edge_clip_dispatch,
    gaussian_blur_vertical_pass_edge_clip_dispatch, GaussianClipHorizontalPass,
    GaussianClipVerticalPass,
};
use crate::gaussian::gaussian_precise_level::GaussianPreciseLevel;
use crate::gaussian::gaussian_vertical::gaussian_blur_vertical_pass_c_impl;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::gaussian::neon::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::gaussian::sse::{
    gaussian_blur_horizontal_pass_impl_sse, gaussian_blur_vertical_pass_impl_f32_sse,
    gaussian_blur_vertical_pass_impl_sse, gaussian_horiz_one_chan_f32,
    gaussian_horiz_sse_t_f_chan_f32, gaussian_sse_horiz_one_chan_u8,
};
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::{get_sigma_size, ThreadingPolicy};

trait GaussianHorizontalDispatch<T> {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>() -> fn(
        src: &[T],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        width: u32,
        kernel_size: usize,
        kernel: &[f32],
        start_y: u32,
        end_y: u32,
    );
}

impl GaussianHorizontalDispatch<u16> for u16 {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    ) -> fn(&[u16], u32, &UnsafeSlice<u16>, u32, u32, usize, &[f32], u32, u32) {
        gaussian_blur_horizontal_pass_impl::<u16, CHANNEL_CONFIGURATION, EDGE_MODE>
    }
}

impl GaussianHorizontalDispatch<f16> for f16 {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    ) -> fn(&[f16], u32, &UnsafeSlice<f16>, u32, u32, usize, &[f32], u32, u32) {
        gaussian_blur_horizontal_pass_impl::<f16, CHANNEL_CONFIGURATION, EDGE_MODE>
    }
}

impl GaussianHorizontalDispatch<u8> for u8 {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    ) -> fn(&[u8], u32, &UnsafeSlice<u8>, u32, u32, usize, &[f32], u32, u32) {
        let mut _dispatcher: fn(
            src: &[u8],
            src_stride: u32,
            unsafe_dst: &UnsafeSlice<u8>,
            dst_stride: u32,
            width: u32,
            kernel_size: usize,
            kernel: &[f32],
            start_y: u32,
            end_y: u32,
        ) = gaussian_blur_horizontal_pass_impl::<u8, CHANNEL_CONFIGURATION, EDGE_MODE>;
        let edge_mode: EdgeMode = EDGE_MODE.into();
        if CHANNEL_CONFIGURATION >= 3 && edge_mode == EdgeMode::Clamp {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                let _is_fma_available = std::arch::is_x86_feature_detected!("fma");
                if _is_sse_available {
                    if !_is_fma_available {
                        _dispatcher = gaussian_blur_horizontal_pass_impl_sse::<
                            u8,
                            CHANNEL_CONFIGURATION,
                            true,
                        >;
                    } else {
                        _dispatcher = gaussian_blur_horizontal_pass_impl_sse::<
                            u8,
                            CHANNEL_CONFIGURATION,
                            false,
                        >;
                    }
                }
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_blur_horizontal_pass_neon::<u8, CHANNEL_CONFIGURATION>;
            }
        }
        if edge_mode == EdgeMode::Clamp && CHANNEL_CONFIGURATION == 1 {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_horiz_one_chan_u8::<u8>;
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                if _is_sse_available {
                    _dispatcher = gaussian_sse_horiz_one_chan_u8::<u8>;
                }
            }
        }
        _dispatcher
    }
}

impl GaussianHorizontalDispatch<f32> for f32 {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    ) -> fn(&[f32], u32, &UnsafeSlice<f32>, u32, u32, usize, &[f32], u32, u32) {
        let mut _dispatcher: fn(
            src: &[f32],
            src_stride: u32,
            unsafe_dst: &UnsafeSlice<f32>,
            dst_stride: u32,
            width: u32,
            kernel_size: usize,
            kernel: &[f32],
            start_y: u32,
            end_y: u32,
        ) = gaussian_blur_horizontal_pass_impl::<f32, CHANNEL_CONFIGURATION, EDGE_MODE>;
        let edge_mode: EdgeMode = EDGE_MODE.into();
        if edge_mode == EdgeMode::Clamp && CHANNEL_CONFIGURATION == 1 {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_horiz_one_chan_f32::<f32>;
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                let _is_fma_available = std::arch::is_x86_feature_detected!("fma");
                if _is_sse_available {
                    _dispatcher = gaussian_horiz_one_chan_f32::<f32>;
                }
            }
        } else if edge_mode == EdgeMode::Clamp && CHANNEL_CONFIGURATION >= 3 {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_horiz_t_f_chan_f32::<f32, CHANNEL_CONFIGURATION>;
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                let _is_fma_available = std::arch::is_x86_feature_detected!("fma");
                if _is_sse_available {
                    if _is_fma_available {
                        _dispatcher =
                            gaussian_horiz_sse_t_f_chan_f32::<f32, CHANNEL_CONFIGURATION, true>;
                    } else {
                        _dispatcher =
                            gaussian_horiz_sse_t_f_chan_f32::<f32, CHANNEL_CONFIGURATION, false>;
                    }
                }
            }
        }
        _dispatcher
    }
}

#[allow(clippy::type_complexity)]
fn gaussian_blur_horizontal_pass<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static
        + AsPrimitive<f32>
        + GaussianHorizontalDispatch<T>,
    const CHANNEL_CONFIGURATION: usize,
    const EDGE_MODE: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    thread_pool: &Option<ThreadPool>,
    thread_count: u32,
) where
    f32: AsPrimitive<T> + ToStorage<T>,
{
    let mut _dispatcher: fn(
        src: &[T],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        width: u32,
        kernel_size: usize,
        kernel: &[f32],
        start_y: u32,
        end_y: u32,
    ) = T::get_pass::<CHANNEL_CONFIGURATION, EDGE_MODE>();
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
                        kernel_size,
                        kernel,
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
            kernel_size,
            kernel,
            0,
            height,
        );
    }
}

fn gaussian_blur_vertical_pass_impl<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static
        + AsPrimitive<f32>,
    const CHANNEL_CONFIGURATION: usize,
    const EDGE_MODE: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) where
    f32: AsPrimitive<T> + ToStorage<T>,
{
    gaussian_blur_vertical_pass_c_impl::<T, CHANNEL_CONFIGURATION, EDGE_MODE>(
        src,
        src_stride,
        unsafe_dst,
        dst_stride,
        width,
        height,
        kernel_size,
        kernel,
        start_y,
        end_y,
    );
}

trait GaussianVerticalPassDispatch<T> {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>() -> fn(
        src: &[T],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: usize,
        kernel: &[f32],
        start_y: u32,
        end_y: u32,
    );
}

impl GaussianVerticalPassDispatch<f16> for f16 {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    ) -> fn(&[f16], u32, &UnsafeSlice<f16>, u32, u32, u32, usize, &[f32], u32, u32) {
        gaussian_blur_vertical_pass_impl::<f16, CHANNEL_CONFIGURATION, EDGE_MODE>
    }
}

impl GaussianVerticalPassDispatch<u16> for u16 {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    ) -> fn(&[u16], u32, &UnsafeSlice<u16>, u32, u32, u32, usize, &[f32], u32, u32) {
        gaussian_blur_vertical_pass_impl::<u16, CHANNEL_CONFIGURATION, EDGE_MODE>
    }
}

impl GaussianVerticalPassDispatch<u8> for u8 {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    ) -> fn(&[u8], u32, &UnsafeSlice<u8>, u32, u32, u32, usize, &[f32], u32, u32) {
        let mut _dispatcher: fn(
            src: &[u8],
            src_stride: u32,
            unsafe_dst: &UnsafeSlice<u8>,
            dst_stride: u32,
            width: u32,
            height: u32,
            kernel_size: usize,
            kernel: &[f32],
            start_y: u32,
            end_y: u32,
        ) = gaussian_blur_vertical_pass_impl::<u8, CHANNEL_CONFIGURATION, EDGE_MODE>;
        let edge_mode: EdgeMode = EDGE_MODE.into();
        if edge_mode == EdgeMode::Clamp {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                let _is_fma_available = std::arch::is_x86_feature_detected!("fma");
                let _is_avx_available = std::arch::is_x86_feature_detected!("avx2");
                // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channel
                if _is_sse_available {
                    if _is_fma_available {
                        _dispatcher =
                            gaussian_blur_vertical_pass_impl_sse::<u8, CHANNEL_CONFIGURATION, true>;
                    } else {
                        _dispatcher = gaussian_blur_vertical_pass_impl_sse::<
                            u8,
                            CHANNEL_CONFIGURATION,
                            false,
                        >;
                    }
                }
                if _is_avx_available {
                    if _is_fma_available {
                        _dispatcher =
                            gaussian_blur_vertical_pass_impl_avx::<u8, CHANNEL_CONFIGURATION, true>;
                    } else {
                        _dispatcher = gaussian_blur_vertical_pass_impl_avx::<
                            u8,
                            CHANNEL_CONFIGURATION,
                            false,
                        >;
                    }
                }
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channel
                _dispatcher = gaussian_blur_vertical_pass_neon::<u8, CHANNEL_CONFIGURATION>;
            }
        }
        _dispatcher
    }
}

impl GaussianVerticalPassDispatch<f32> for f32 {
    fn get_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    ) -> fn(&[f32], u32, &UnsafeSlice<f32>, u32, u32, u32, usize, &[f32], u32, u32) {
        let mut _dispatcher: fn(
            src: &[f32],
            src_stride: u32,
            unsafe_dst: &UnsafeSlice<f32>,
            dst_stride: u32,
            width: u32,
            height: u32,
            kernel_size: usize,
            kernel: &[f32],
            start_y: u32,
            end_y: u32,
        ) = gaussian_blur_vertical_pass_impl::<f32, CHANNEL_CONFIGURATION, EDGE_MODE>;
        let edge_mode: EdgeMode = EDGE_MODE.into();
        if edge_mode == EdgeMode::Clamp {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                let _is_fma_available = std::arch::is_x86_feature_detected!("fma");
                let _is_avx_available = std::arch::is_x86_feature_detected!("avx2");
                // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channel
                if _is_sse_available {
                    if _is_fma_available {
                        _dispatcher = gaussian_blur_vertical_pass_impl_f32_sse::<
                            f32,
                            CHANNEL_CONFIGURATION,
                            true,
                        >;
                    } else {
                        _dispatcher = gaussian_blur_vertical_pass_impl_f32_sse::<
                            f32,
                            CHANNEL_CONFIGURATION,
                            false,
                        >;
                    }
                }
                // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channel
                if _is_avx_available {
                    if _is_fma_available {
                        _dispatcher = gaussian_blur_vertical_pass_impl_f32_avx::<
                            f32,
                            CHANNEL_CONFIGURATION,
                            true,
                        >;
                    } else {
                        _dispatcher = gaussian_blur_vertical_pass_impl_f32_avx::<
                            f32,
                            CHANNEL_CONFIGURATION,
                            false,
                        >;
                    }
                }
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channel
                _dispatcher = gaussian_blur_vertical_pass_f32_neon::<f32, CHANNEL_CONFIGURATION>;
            }
        }
        _dispatcher
    }
}

#[allow(clippy::type_complexity)]
fn gaussian_blur_vertical_pass<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static
        + AsPrimitive<f32>
        + GaussianVerticalPassDispatch<T>,
    const CHANNEL_CONFIGURATION: usize,
    const EDGE_MODE: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    thread_pool: &Option<ThreadPool>,
    thread_count: u32,
) where
    f32: AsPrimitive<T> + ToStorage<T>,
{
    let mut _dispatcher: fn(
        src: &[T],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: usize,
        kernel: &[f32],
        start_y: u32,
        end_y: u32,
    ) = T::get_pass::<CHANNEL_CONFIGURATION, EDGE_MODE>();
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
                        kernel_size,
                        kernel,
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
            kernel_size,
            kernel,
            0,
            height,
        );
    }
}

fn gaussian_blur_impl<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + 'static
        + AsPrimitive<f32>
        + GaussianVerticalPassDispatch<T>
        + GaussianHorizontalDispatch<T>
        + GaussianClipVerticalPass<T>
        + GaussianClipHorizontalPass<T>,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) where
    f32: AsPrimitive<T> + ToStorage<T>,
{
    if kernel_size % 2 == 0 {
        panic!("kernel size must be odd");
    }
    let mut transient: Vec<T> =
        vec![T::from_u32(0).unwrap_or_default(); dst_stride as usize * height as usize];

    let thread_count = threading_policy.get_threads_count(width, height) as u32;
    let pool = if thread_count == 1 {
        None
    } else {
        let hold = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count as usize)
            .build()
            .unwrap();
        Some(hold)
    };

    match edge_mode {
        EdgeMode::Reflect => {
            let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
            gaussian_blur_horizontal_pass::<T, CHANNEL_CONFIGURATION, { EdgeMode::Reflect as usize }>(
                src,
                src_stride,
                &mut transient,
                dst_stride,
                width,
                height,
                kernel.len(),
                &kernel,
                &pool,
                thread_count,
            );
            gaussian_blur_vertical_pass::<T, CHANNEL_CONFIGURATION, { EdgeMode::Reflect as usize }>(
                &transient,
                dst_stride,
                dst,
                dst_stride,
                width,
                height,
                kernel.len(),
                &kernel,
                &pool,
                thread_count,
            );
        }
        EdgeMode::Wrap => {
            let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
            gaussian_blur_horizontal_pass::<T, CHANNEL_CONFIGURATION, { EdgeMode::Wrap as usize }>(
                src,
                src_stride,
                &mut transient,
                dst_stride,
                width,
                height,
                kernel.len(),
                &kernel,
                &pool,
                thread_count,
            );
            gaussian_blur_vertical_pass::<T, CHANNEL_CONFIGURATION, { EdgeMode::Wrap as usize }>(
                &transient,
                dst_stride,
                dst,
                dst_stride,
                width,
                height,
                kernel.len(),
                &kernel,
                &pool,
                thread_count,
            );
        }
        EdgeMode::Clamp => {
            let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
            gaussian_blur_horizontal_pass::<T, CHANNEL_CONFIGURATION, { EdgeMode::Clamp as usize }>(
                src,
                src_stride,
                &mut transient,
                dst_stride,
                width,
                height,
                kernel.len(),
                &kernel,
                &pool,
                thread_count,
            );
            gaussian_blur_vertical_pass::<T, CHANNEL_CONFIGURATION, { EdgeMode::Clamp as usize }>(
                &transient,
                dst_stride,
                dst,
                dst_stride,
                width,
                height,
                kernel.len(),
                &kernel,
                &pool,
                thread_count,
            );
        }
        EdgeMode::Reflect101 => {
            let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
            gaussian_blur_horizontal_pass::<
                T,
                CHANNEL_CONFIGURATION,
                { EdgeMode::Reflect101 as usize },
            >(
                src,
                src_stride,
                &mut transient,
                dst_stride,
                width,
                height,
                kernel.len(),
                &kernel,
                &pool,
                thread_count,
            );
            gaussian_blur_vertical_pass::<
                T,
                CHANNEL_CONFIGURATION,
                { EdgeMode::Reflect101 as usize },
            >(
                &transient,
                dst_stride,
                dst,
                dst_stride,
                width,
                height,
                kernel.len(),
                &kernel,
                &pool,
                thread_count,
            );
        }
        EdgeMode::KernelClip => {
            let horizontal_filter = create_filter(width as usize, kernel_size, sigma);
            let vertical_filter = create_filter(height as usize, kernel_size, sigma);
            gaussian_blur_horizontal_pass_edge_clip_dispatch::<T, CHANNEL_CONFIGURATION>(
                src,
                dst_stride,
                &mut transient,
                dst_stride,
                width,
                height,
                &horizontal_filter,
                &pool,
                thread_count,
            );
            gaussian_blur_vertical_pass_edge_clip_dispatch::<T, CHANNEL_CONFIGURATION>(
                &transient,
                dst_stride,
                dst,
                dst_stride,
                width,
                height,
                &vertical_filter,
                &pool,
                thread_count,
            );
        }
    }
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// O(R) complexity.
///
/// # Arguments
///
/// * `stride` - Lane length, default is width * channels_count * size_of(PixelType) if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
/// * `precise_level` - Gaussian precise level
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
#[allow(clippy::too_many_arguments)]
pub fn gaussian_blur(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
    precise_level: GaussianPreciseLevel,
) {
    let sigma = if sigma <= 0. {
        get_sigma_size(kernel_size as usize)
    } else {
        sigma
    };
    match precise_level {
        GaussianPreciseLevel::EXACT => {
            let _dispatcher = match channels {
                FastBlurChannels::Plane => gaussian_blur_impl::<u8, 1>,
                FastBlurChannels::Channels3 => gaussian_blur_impl::<u8, 3>,
                FastBlurChannels::Channels4 => gaussian_blur_impl::<u8, 4>,
            };
            _dispatcher(
                src,
                src_stride,
                dst,
                dst_stride,
                width,
                height,
                kernel_size,
                sigma,
                threading_policy,
                edge_mode,
            );
        }
        GaussianPreciseLevel::INTEGRAL => {
            let _dispatcher = match channels {
                FastBlurChannels::Plane => gaussian_blur_impl_approx::<1>,
                FastBlurChannels::Channels3 => gaussian_blur_impl_approx::<3>,
                FastBlurChannels::Channels4 => gaussian_blur_impl_approx::<4>,
            };
            _dispatcher(
                src,
                src_stride,
                dst,
                dst_stride,
                width,
                height,
                kernel_size,
                sigma,
                threading_policy,
                edge_mode,
            );
        }
    }
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// O(R) complexity.
///
/// # Arguments
///
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_blur_u16(
    src: &[u16],
    dst: &mut [u16],
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) {
    let _dispatcher = match channels {
        FastBlurChannels::Plane => gaussian_blur_impl::<u16, 1>,
        FastBlurChannels::Channels3 => gaussian_blur_impl::<u16, 3>,
        FastBlurChannels::Channels4 => gaussian_blur_impl::<u16, 4>,
    };
    let sigma = if sigma <= 0. {
        get_sigma_size(kernel_size as usize)
    } else {
        sigma
    };
    _dispatcher(
        src,
        width * channels.get_channels() as u32,
        dst,
        width * channels.get_channels() as u32,
        width,
        height,
        kernel_size,
        sigma,
        threading_policy,
        edge_mode,
    );
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// O(R) complexity.
///
/// # Arguments
///
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
#[allow(clippy::too_many_arguments)]
pub fn gaussian_blur_f32(
    src: &[f32],
    dst: &mut [f32],
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) {
    let _dispatcher = match channels {
        FastBlurChannels::Plane => gaussian_blur_impl::<f32, 1>,
        FastBlurChannels::Channels3 => gaussian_blur_impl::<f32, 3>,
        FastBlurChannels::Channels4 => gaussian_blur_impl::<f32, 4>,
    };
    let sigma = if sigma <= 0. {
        get_sigma_size(kernel_size as usize)
    } else {
        sigma
    };
    _dispatcher(
        src,
        width * channels.get_channels() as u32,
        dst,
        width * channels.get_channels() as u32,
        width,
        height,
        kernel_size,
        sigma,
        threading_policy,
        edge_mode,
    );
}

/// Performs gaussian blur on the image.
///
/// This performs a gaussian kernel filter on the image producing beautiful looking result.
/// Preferred if you need to perform an advanced signal analysis after.
/// O(R) complexity.
///
/// # Arguments
///
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `kernel_size` - Length of gaussian kernel. Panic if kernel size is not odd, even kernels with unbalanced center is not accepted.
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. If zero of negative then *get_sigma_size* will be used
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_blur_f16(
    src: &[u16],
    dst: &mut [u16],
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    channels: FastBlurChannels,
    edge_mode: EdgeMode,
    threading_policy: ThreadingPolicy,
) {
    let _dispatcher = match channels {
        FastBlurChannels::Plane => gaussian_blur_impl::<half::f16, 1>,
        FastBlurChannels::Channels3 => gaussian_blur_impl::<half::f16, 3>,
        FastBlurChannels::Channels4 => gaussian_blur_impl::<half::f16, 4>,
    };
    let sigma = if sigma <= 0. {
        get_sigma_size(kernel_size as usize)
    } else {
        sigma
    };
    _dispatcher(
        unsafe { std::mem::transmute::<&[u16], &[half::f16]>(src) },
        width * channels.get_channels() as u32,
        unsafe { std::mem::transmute::<&mut [u16], &mut [half::f16]>(dst) },
        width * channels.get_channels() as u32,
        width,
        height,
        kernel_size,
        sigma,
        threading_policy,
        edge_mode,
    );
}
