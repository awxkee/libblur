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

use num_traits::cast::FromPrimitive;
use num_traits::AsPrimitive;
use rayon::ThreadPool;

use crate::channels_configuration::FastBlurChannels;
use crate::edge_mode::EdgeMode;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::gaussian::gauss_neon::*;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::gaussian::gauss_sse::gaussian_horiz_one_chan_f32;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::gaussian::gauss_sse::gaussian_horiz_sse_t_f_chan_f32;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::gaussian::gauss_sse::gaussian_sse_horiz_one_chan_u8;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::gaussian::gauss_sse::{
    gaussian_blur_horizontal_pass_impl_sse, gaussian_blur_vertical_pass_impl_sse,
};
use crate::gaussian::gaussian_filter::create_filter;
use crate::gaussian::gaussian_horizontal::gaussian_blur_horizontal_pass_impl;
use crate::gaussian::gaussian_kernel::get_gaussian_kernel_1d;
use crate::gaussian::gaussian_kernel_filter_dispatch::{
    gaussian_blur_horizontal_pass_edge_clip_dispatch,
    gaussian_blur_vertical_pass_edge_clip_dispatch,
};
use crate::gaussian::gaussian_vertical::gaussian_blur_vertical_pass_c_impl;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::ThreadingPolicy;

fn gaussian_blur_horizontal_pass<
    T: FromPrimitive + Default + Send + Sync,
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
    kernel: &Vec<f32>,
    thread_pool: &ThreadPool,
    thread_count: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + 'static + AsPrimitive<f32>,
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
    ) = gaussian_blur_horizontal_pass_impl::<T, CHANNEL_CONFIGURATION, EDGE_MODE>;
    let edge_mode: EdgeMode = EDGE_MODE.into();
    if CHANNEL_CONFIGURATION >= 3 {
        if std::any::type_name::<T>() == "u8" && edge_mode == EdgeMode::Clamp {
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                target_feature = "sse4.1"
            ))]
            {
                _dispatcher = gaussian_blur_horizontal_pass_impl_sse::<T, CHANNEL_CONFIGURATION>;
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_blur_horizontal_pass_neon::<T, CHANNEL_CONFIGURATION>;
            }
        }
    }
    if std::any::type_name::<T>() == "f32" {
        if edge_mode == EdgeMode::Clamp && CHANNEL_CONFIGURATION == 1 {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_horiz_one_chan_f32::<T>;
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                target_feature = "sse4.1"
            ))]
            {
                _dispatcher = gaussian_horiz_one_chan_f32::<T>;
            }
        } else if edge_mode == EdgeMode::Clamp && CHANNEL_CONFIGURATION >= 3 {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = gaussian_horiz_t_f_chan_f32::<T, CHANNEL_CONFIGURATION>;
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                target_feature = "sse4.1"
            ))]
            {
                _dispatcher = gaussian_horiz_sse_t_f_chan_f32::<T, CHANNEL_CONFIGURATION>;
            }
        }
    }
    if edge_mode == EdgeMode::Clamp && CHANNEL_CONFIGURATION == 1 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if std::any::type_name::<T>() == "u8" {
            _dispatcher = gaussian_horiz_one_chan_u8::<T>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        if std::any::type_name::<T>() == "u8" {
            _dispatcher = gaussian_sse_horiz_one_chan_u8::<T>;
        }
    }
    let unsafe_dst = UnsafeSlice::new(dst);
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
}

fn gaussian_blur_vertical_pass_impl<
    T: FromPrimitive + Default + Send + Sync,
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
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + 'static + AsPrimitive<f32>,
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

fn gaussian_blur_vertical_pass<
    T: FromPrimitive + Default + Send + Sync,
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
    kernel: &Vec<f32>,
    thread_pool: &ThreadPool,
    thread_count: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + 'static + AsPrimitive<f32>,
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
    ) = gaussian_blur_vertical_pass_impl::<T, CHANNEL_CONFIGURATION, EDGE_MODE>;
    let edge_mode: EdgeMode = EDGE_MODE.into();
    if std::any::type_name::<T>() == "u8" && edge_mode == EdgeMode::Clamp {
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channel
            _dispatcher = gaussian_blur_vertical_pass_impl_sse::<T, CHANNEL_CONFIGURATION>;
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            // Generally vertical pass do not depends on any specific channel configuration so it is allowed to make a vectorized calls for any channel
            _dispatcher = gaussian_blur_vertical_pass_neon::<T, CHANNEL_CONFIGURATION>;
        }
    }
    let unsafe_dst = UnsafeSlice::new(dst);
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
}

fn gaussian_blur_impl<
    T: FromPrimitive + Default + Send + Sync,
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
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + 'static + AsPrimitive<f32>,
    f32: AsPrimitive<T> + ToStorage<T>,
{
    if kernel_size % 2 == 0 {
        panic!("kernel size must be odd");
    }
    let mut transient: Vec<T> =
        vec![T::from_u32(0).unwrap_or_default(); dst_stride as usize * height as usize];

    let thread_count = threading_policy.get_threads_count(width, height) as u32;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();

    match edge_mode {
        EdgeMode::Reflect => {
            let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
            gaussian_blur_horizontal_pass::<T, CHANNEL_CONFIGURATION, { EdgeMode::Reflect as usize }>(
                &src,
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
                &src,
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
                &src,
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
                &src,
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
                &src,
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
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. Default - kernel_size / 6
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
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
) {
    match channels {
        FastBlurChannels::Plane => {
            gaussian_blur_impl::<u8, 1>(
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
        FastBlurChannels::Channels3 => {
            gaussian_blur_impl::<u8, 3>(
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
        FastBlurChannels::Channels4 => {
            gaussian_blur_impl::<u8, 4>(
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
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. Default - kernel_size / 6
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
    match channels {
        FastBlurChannels::Plane => {
            gaussian_blur_impl::<u16, 1>(
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
        FastBlurChannels::Channels3 => {
            gaussian_blur_impl::<u16, 3>(
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
        FastBlurChannels::Channels4 => {
            gaussian_blur_impl::<u16, 4>(
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
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. Default - kernel_size / 6
/// * `channels` - Count of channels in the image
/// * `edge_mode` - Rule to handle edge mode
/// * `threading_policy` - Threading policy according to *ThreadingPolicy*
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
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
    match channels {
        FastBlurChannels::Plane => {
            gaussian_blur_impl::<f32, 1>(
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
        FastBlurChannels::Channels3 => {
            gaussian_blur_impl::<f32, 3>(
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
        FastBlurChannels::Channels4 => {
            gaussian_blur_impl::<f32, 4>(
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
/// * `sigma` - Sigma for a gaussian kernel, corresponds to kernel flattening level. Default - kernel_size / 6
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
    match channels {
        FastBlurChannels::Plane => {
            gaussian_blur_impl::<half::f16, 1>(
                unsafe { std::mem::transmute(src) },
                width * channels.get_channels() as u32,
                unsafe { std::mem::transmute(dst) },
                width * channels.get_channels() as u32,
                width,
                height,
                kernel_size,
                sigma,
                threading_policy,
                edge_mode,
            );
        }
        FastBlurChannels::Channels3 => {
            gaussian_blur_impl::<half::f16, 3>(
                unsafe { std::mem::transmute(src) },
                width * channels.get_channels() as u32,
                unsafe { std::mem::transmute(dst) },
                width * channels.get_channels() as u32,
                width,
                height,
                kernel_size,
                sigma,
                threading_policy,
                edge_mode,
            );
        }
        FastBlurChannels::Channels4 => {
            gaussian_blur_impl::<half::f16, 4>(
                unsafe { std::mem::transmute(src) },
                width * channels.get_channels() as u32,
                unsafe { std::mem::transmute(dst) },
                width * channels.get_channels() as u32,
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
