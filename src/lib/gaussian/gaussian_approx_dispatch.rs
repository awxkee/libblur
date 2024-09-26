/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
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
use crate::gaussian::gaussian_approx::PRECISION;
use crate::gaussian::gaussian_approx_horizontal::{
    gaussian_blur_horizontal_pass_impl_approx, gaussian_blur_horizontal_pass_impl_clip_edge_approx,
};
use crate::gaussian::gaussian_approx_vertical::{
    gaussian_blur_vertical_pass_c_approx, gaussian_blur_vertical_pass_clip_edge_approx,
};
use crate::gaussian::gaussian_filter::{create_integral_filter, GaussianFilter};
use crate::gaussian::gaussian_kernel::get_gaussian_kernel_1d_integral;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::gaussian::neon::{
    gaussian_blur_horizontal_pass_approx_neon, gaussian_blur_vertical_approx_neon,
};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::gaussian::neon::{
    gaussian_blur_horizontal_pass_filter_approx_neon,
    gaussian_blur_vertical_pass_filter_approx_neon, gaussian_horiz_one_approx_u8,
    gaussian_horiz_one_chan_filter_approx,
};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::gaussian::sse::{
    gaussian_blur_horizontal_pass_approx_sse, gaussian_blur_horizontal_pass_filter_approx_sse,
    gaussian_blur_vertical_pass_approx_sse, gaussian_blur_vertical_pass_filter_approx_sse,
    gaussian_sse_horiz_one_chan_filter_approx_u8, gaussian_sse_horiz_one_chan_u8_approx,
};
use crate::unsafe_slice::UnsafeSlice;
use crate::{EdgeMode, ThreadingPolicy};
use rayon::ThreadPool;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn gaussian_blur_horizontal_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[i16],
    thread_pool: &Option<ThreadPool>,
    thread_count: u32,
) {
    let mut _dispatcher: fn(
        src: &[u8],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        kernel_size: usize,
        kernel: &[i16],
        start_y: u32,
        end_y: u32,
    ) = gaussian_blur_horizontal_pass_impl_approx::<CHANNEL_CONFIGURATION, EDGE_MODE>;
    let _edge_mode: EdgeMode = EDGE_MODE.into();
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if (CHANNEL_CONFIGURATION == 3 || CHANNEL_CONFIGURATION == 4)
            && _edge_mode == EdgeMode::Clamp
        {
            _dispatcher = gaussian_blur_horizontal_pass_approx_neon::<CHANNEL_CONFIGURATION>;
        } else if CHANNEL_CONFIGURATION == 1 && _edge_mode == EdgeMode::Clamp {
            _dispatcher = gaussian_horiz_one_approx_u8;
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if _is_sse_available
            && _edge_mode == EdgeMode::Clamp
            && (CHANNEL_CONFIGURATION == 3 || CHANNEL_CONFIGURATION == 4)
        {
            _dispatcher = gaussian_blur_horizontal_pass_approx_sse::<CHANNEL_CONFIGURATION>;
        } else if _is_sse_available && _edge_mode == EdgeMode::Clamp && CHANNEL_CONFIGURATION == 1 {
            _dispatcher = gaussian_sse_horiz_one_chan_u8_approx;
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

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn gaussian_blur_vertical_pass<const CHANNEL_CONFIGURATION: usize, const EDGE_MODE: usize>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[i16],
    thread_pool: &Option<ThreadPool>,
    thread_count: u32,
) {
    let mut _dispatcher: fn(
        src: &[u8],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: usize,
        kernel: &[i16],
        start_y: u32,
        end_y: u32,
    ) = gaussian_blur_vertical_pass_c_approx::<CHANNEL_CONFIGURATION, EDGE_MODE>;
    let _edge_mode: EdgeMode = EDGE_MODE.into();
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if _edge_mode == EdgeMode::Clamp {
            _dispatcher = gaussian_blur_vertical_approx_neon::<CHANNEL_CONFIGURATION>;
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if _is_sse_available && _edge_mode == EdgeMode::Clamp {
            _dispatcher = gaussian_blur_vertical_pass_approx_sse::<CHANNEL_CONFIGURATION>;
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

#[allow(clippy::type_complexity)]
pub(crate) fn gaussian_blur_vertical_pass_approx_clip_dispatch<
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &[GaussianFilter<i16>],
    thread_pool: &Option<ThreadPool>,
    thread_count: u32,
) {
    let mut _dispatcher: fn(
        src: &[u8],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        height: u32,
        filter: &[GaussianFilter<i16>],
        start_y: u32,
        end_y: u32,
    ) = gaussian_blur_vertical_pass_clip_edge_approx::<CHANNEL_CONFIGURATION>;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = gaussian_blur_vertical_pass_filter_approx_neon::<CHANNEL_CONFIGURATION>;
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if _is_sse_available {
            _dispatcher = gaussian_blur_vertical_pass_filter_approx_sse::<CHANNEL_CONFIGURATION>;
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

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn gaussian_blur_horizontal_pass_clip_approx_dispatch<const CHANNEL_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &[GaussianFilter<i16>],
    thread_pool: &Option<ThreadPool>,
    thread_count: u32,
) {
    let mut _dispatcher: fn(
        src: &[u8],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u8>,
        dst_stride: u32,
        width: u32,
        filter: &[GaussianFilter<i16>],
        start_y: u32,
        end_y: u32,
    ) = gaussian_blur_horizontal_pass_impl_clip_edge_approx::<CHANNEL_CONFIGURATION>;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if CHANNEL_CONFIGURATION == 3 || CHANNEL_CONFIGURATION == 4 {
            _dispatcher = gaussian_blur_horizontal_pass_filter_approx_neon::<CHANNEL_CONFIGURATION>;
        } else if CHANNEL_CONFIGURATION == 1 {
            _dispatcher = gaussian_horiz_one_chan_filter_approx;
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if _is_sse_available && (CHANNEL_CONFIGURATION == 3 || CHANNEL_CONFIGURATION == 4) {
            _dispatcher = gaussian_blur_horizontal_pass_filter_approx_sse::<CHANNEL_CONFIGURATION>;
        } else if _is_sse_available && CHANNEL_CONFIGURATION == 1 {
            _dispatcher = gaussian_sse_horiz_one_chan_filter_approx_u8;
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn gaussian_blur_impl_approx<const CHANNEL_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) {
    if kernel_size % 2 == 0 {
        panic!("kernel size must be odd");
    }
    let mut transient: Vec<u8> = vec![0; dst_stride as usize * height as usize];

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
            let kernel = get_gaussian_kernel_1d_integral(kernel_size, sigma);
            gaussian_blur_horizontal_pass::<CHANNEL_CONFIGURATION, { EdgeMode::Reflect as usize }>(
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
            gaussian_blur_vertical_pass::<CHANNEL_CONFIGURATION, { EdgeMode::Reflect as usize }>(
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
            let kernel = get_gaussian_kernel_1d_integral(kernel_size, sigma);
            gaussian_blur_horizontal_pass::<CHANNEL_CONFIGURATION, { EdgeMode::Wrap as usize }>(
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
            gaussian_blur_vertical_pass::<CHANNEL_CONFIGURATION, { EdgeMode::Wrap as usize }>(
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
            let kernel = get_gaussian_kernel_1d_integral(kernel_size, sigma);
            gaussian_blur_horizontal_pass::<CHANNEL_CONFIGURATION, { EdgeMode::Clamp as usize }>(
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
            gaussian_blur_vertical_pass::<CHANNEL_CONFIGURATION, { EdgeMode::Clamp as usize }>(
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
            let kernel = get_gaussian_kernel_1d_integral(kernel_size, sigma);
            gaussian_blur_horizontal_pass::<CHANNEL_CONFIGURATION, { EdgeMode::Reflect101 as usize }>(
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
            gaussian_blur_vertical_pass::<CHANNEL_CONFIGURATION, { EdgeMode::Reflect101 as usize }>(
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
            let horizontal_filter =
                create_integral_filter::<PRECISION>(width as usize, kernel_size, sigma);
            let vertical_filter =
                create_integral_filter::<PRECISION>(height as usize, kernel_size, sigma);
            gaussian_blur_horizontal_pass_clip_approx_dispatch::<CHANNEL_CONFIGURATION>(
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
            gaussian_blur_vertical_pass_approx_clip_dispatch::<CHANNEL_CONFIGURATION>(
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
