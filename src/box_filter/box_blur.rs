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

use colorutils_rs::linear_to_planar::linear_to_plane;
use colorutils_rs::planar_to_linear::plane_to_linear;
use colorutils_rs::{
    linear_to_rgb, linear_to_rgba, rgb_to_linear, rgba_to_linear, TransferFunction,
};
use half::f16;
use num_traits::cast::FromPrimitive;
use num_traits::AsPrimitive;
use rayon::ThreadPool;

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use crate::box_filter::box_blur_neon::*;
use crate::channels_configuration::FastBlurChannels;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::util::check_slice_size;
use crate::{BlurError, BlurImage, BlurImageMut, ThreadingPolicy};

fn box_blur_horizontal_pass_impl<T, J, const CN: usize>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) where
    T: std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + FromPrimitive
        + Default
        + Send
        + Sync
        + AsPrimitive<J>,
    J: FromPrimitive
        + Copy
        + std::ops::Mul<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + AsPrimitive<f32>,
    f32: ToStorage<T>,
{
    let kernel_size = radius * 2 + 1;
    let edge_count = J::from_u32((kernel_size / 2) + 1).unwrap();
    let half_kernel = kernel_size / 2;

    let weight = 1f32 / (radius * 2) as f32;

    for y in start_y..end_y {
        let mut weight0;
        let mut weight1 = J::from_u32(0u32).unwrap();
        let mut weight2 = J::from_u32(0u32).unwrap();
        let mut weight3 = J::from_u32(0u32).unwrap();
        let y_src_shift = (y * src_stride) as usize;
        let y_dst_shift = (y * dst_stride) as usize;
        // replicate edge
        weight0 = (unsafe { *src.get_unchecked(y_src_shift) }.as_()) * edge_count;
        if CN > 1 {
            weight1 = (unsafe { *src.get_unchecked(y_src_shift + 1) }.as_()) * edge_count;
        }
        if CN > 2 {
            weight2 = (unsafe { *src.get_unchecked(y_src_shift + 2) }.as_()) * edge_count;
        }
        if CN == 4 {
            weight3 = (unsafe { *src.get_unchecked(y_src_shift + 3) }.as_()) * edge_count;
        }

        for x in 1..half_kernel as usize {
            let px = x.min(width as usize - 1) * CN;
            weight0 += unsafe { *src.get_unchecked(y_src_shift + px) }.as_();
            if CN > 1 {
                weight1 += unsafe { *src.get_unchecked(y_src_shift + px + 1) }.as_();
            }
            if CN > 2 {
                weight2 += unsafe { *src.get_unchecked(y_src_shift + px + 2) }.as_();
            }
            if CN == 4 {
                weight3 += unsafe { *src.get_unchecked(y_src_shift + px + 3) }.as_();
            }
        }

        for x in 0..width as usize {
            let next = (x + half_kernel as usize).min(width as usize - 1) * CN;
            let previous = (x as i64 - half_kernel as i64).max(0) as usize * CN;
            let px = x * CN;
            // Prune previous and add next and compute mean

            weight0 += unsafe { *src.get_unchecked(y_src_shift + next) }.as_();
            if CN > 1 {
                weight1 += unsafe { *src.get_unchecked(y_src_shift + next + 1) }.as_();
            }
            if CN > 2 {
                weight2 += unsafe { *src.get_unchecked(y_src_shift + next + 2) }.as_();
            }

            weight0 -= unsafe { *src.get_unchecked(y_src_shift + previous) }.as_();
            if CN > 1 {
                weight1 -= unsafe { *src.get_unchecked(y_src_shift + previous + 1) }.as_();
            }
            if CN > 2 {
                weight2 -= unsafe { *src.get_unchecked(y_src_shift + previous + 2) }.as_();
            }

            if CN == 4 {
                weight3 += unsafe { *src.get_unchecked(y_src_shift + next + 3) }.as_();
                weight3 -= unsafe { *src.get_unchecked(y_src_shift + previous + 3) }.as_();
            }

            let write_offset = y_dst_shift + px;
            unsafe {
                unsafe_dst.write(write_offset, (weight0.as_() * weight).to_());
                if CN > 1 {
                    unsafe_dst.write(write_offset + 1, (weight1.as_() * weight).to_());
                }
                if CN > 2 {
                    unsafe_dst.write(write_offset + 2, (weight2.as_() * weight).to_());
                }
                if CN == 4 {
                    unsafe_dst.write(write_offset + 3, (weight3.as_() * weight).to_());
                }
            }
        }
    }
}

trait BoxBlurHorizontalPass<T> {
    #[allow(clippy::type_complexity)]
    fn get_horizontal_pass<const CHANNEL_CONFIGURATION: usize>() -> fn(
        src: &[T],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        width: u32,
        radius: u32,
        start_y: u32,
        end_y: u32,
    );
}

impl BoxBlurHorizontalPass<f32> for f32 {
    #[allow(clippy::type_complexity)]
    fn get_horizontal_pass<const CHANNEL_CONFIGURATION: usize>(
    ) -> fn(&[f32], u32, &UnsafeSlice<f32>, u32, u32, u32, u32, u32) {
        box_blur_horizontal_pass_impl::<f32, f32, CHANNEL_CONFIGURATION>
    }
}

impl BoxBlurHorizontalPass<f16> for f16 {
    #[allow(clippy::type_complexity)]
    fn get_horizontal_pass<const CHANNEL_CONFIGURATION: usize>(
    ) -> fn(&[f16], u32, &UnsafeSlice<f16>, u32, u32, u32, u32, u32) {
        box_blur_horizontal_pass_impl::<f16, f32, CHANNEL_CONFIGURATION>
    }
}

impl BoxBlurHorizontalPass<u16> for u16 {
    #[allow(clippy::type_complexity)]
    fn get_horizontal_pass<const CHANNEL_CONFIGURATION: usize>(
    ) -> fn(&[u16], u32, &UnsafeSlice<u16>, u32, u32, u32, u32, u32) {
        box_blur_horizontal_pass_impl::<u16, u32, CHANNEL_CONFIGURATION>
    }
}

impl BoxBlurHorizontalPass<u8> for u8 {
    #[allow(clippy::type_complexity)]
    fn get_horizontal_pass<const CHANNEL_CONFIGURATION: usize>(
    ) -> fn(&[u8], u32, &UnsafeSlice<u8>, u32, u32, u32, u32, u32) {
        let mut _dispatcher_horizontal: fn(
            src: &[u8],
            src_stride: u32,
            unsafe_dst: &UnsafeSlice<u8>,
            dst_stride: u32,
            width: u32,
            radius: u32,
            start_y: u32,
            end_y: u32,
        ) = box_blur_horizontal_pass_impl::<u8, u32, CHANNEL_CONFIGURATION>;
        if CHANNEL_CONFIGURATION >= 3 {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                _dispatcher_horizontal = box_blur_horizontal_pass_neon::<u8, CHANNEL_CONFIGURATION>;
                #[cfg(feature = "rdm")]
                {
                    if std::arch::is_aarch64_feature_detected!("rdm") {
                        use crate::box_filter::box_blur_neon_q0_31::box_blur_horizontal_pass_neon_rdm;
                        _dispatcher_horizontal =
                            box_blur_horizontal_pass_neon_rdm::<u8, CHANNEL_CONFIGURATION>;
                    }
                }
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(feature = "sse")]
                {
                    let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                    if is_sse_available {
                        use crate::box_filter::box_blur_sse::box_blur_horizontal_pass_sse;
                        _dispatcher_horizontal =
                            box_blur_horizontal_pass_sse::<u8, { CHANNEL_CONFIGURATION }>;
                    }
                }
            }
        }
        _dispatcher_horizontal
    }
}

#[allow(clippy::type_complexity)]
fn box_blur_horizontal_pass<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + AsPrimitive<u32>
        + AsPrimitive<u64>
        + AsPrimitive<f32>
        + AsPrimitive<f64>
        + BoxBlurHorizontalPass<T>,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    pool: &Option<ThreadPool>,
    thread_count: u32,
) where
    f32: ToStorage<T>,
{
    let _dispatcher_horizontal = T::get_horizontal_pass::<CHANNEL_CONFIGURATION>();
    let unsafe_dst = UnsafeSlice::new(dst);
    if let Some(pool) = pool {
        pool.scope(|scope| {
            let segment_size = height / thread_count;
            for i in 0..thread_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_y = height;
                }

                scope.spawn(move |_| {
                    _dispatcher_horizontal(
                        src,
                        src_stride,
                        &unsafe_dst,
                        dst_stride,
                        width,
                        radius,
                        start_y,
                        end_y,
                    );
                });
            }
        });
    } else {
        _dispatcher_horizontal(
            src,
            src_stride,
            &unsafe_dst,
            dst_stride,
            width,
            radius,
            0,
            height,
        );
    }
}

fn box_blur_vertical_pass_impl<T, J>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    _: u32,
    height: u32,
    radius: u32,
    start_x: u32,
    end_x: u32,
) where
    T: std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + FromPrimitive
        + Default
        + Send
        + Sync
        + AsPrimitive<J>,
    J: FromPrimitive
        + Copy
        + std::ops::Mul<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + AsPrimitive<f32>
        + Default,
    f32: ToStorage<T>,
{
    let kernel_size = radius * 2 + 1;

    let edge_count = J::from_u32((kernel_size / 2) + 1).unwrap();
    let half_kernel = kernel_size / 2;

    let weight = 1f32 / (radius * 2) as f32;

    let buf_size = end_x - start_x;

    let buf_cap = buf_size as usize;
    let mut buffer = vec![J::default(); buf_cap];

    let src_lane = &src[start_x as usize..end_x as usize];

    for (x, (v, bf)) in src_lane.iter().zip(buffer.iter_mut()).enumerate() {
        let mut w = v.as_() * edge_count;
        for y in 1..half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            unsafe {
                w += src.get_unchecked(y_src_shift + x + start_x as usize).as_();
            }
        }
        *bf = w;
    }

    for y in 0..height {
        let next = (y + half_kernel).min(height - 1) as usize * src_stride as usize;
        let previous = (y as i64 - half_kernel as i64).max(0) as usize * src_stride as usize;
        let y_dst_shift = dst_stride as usize * y as usize;

        let next_row = unsafe { src.get_unchecked(next..next + end_x as usize) };

        let previous_row = unsafe { src.get_unchecked(previous..previous + end_x as usize) };

        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                unsafe_dst
                    .slice
                    .as_ptr()
                    .add(y_dst_shift + start_x as usize) as *mut T,
                end_x as usize - start_x as usize,
            )
        };

        let previous_row = &previous_row[start_x as usize..];
        let next_row = &next_row[start_x as usize..];

        for (((src_next, src_previous), buffer), dst) in next_row
            .iter()
            .zip(previous_row.iter())
            .zip(buffer.iter_mut())
            .zip(dst.iter_mut())
        {
            let mut weight0 = *buffer;

            weight0 += src_next.as_();
            weight0 -= src_previous.as_();

            *buffer = weight0;

            *dst = (weight0.as_() * weight).to_();
        }
    }
}

trait BoxBlurVerticalPass<T> {
    #[allow(clippy::type_complexity)]
    fn get_box_vertical_pass<const CHANNELS_CONFIGURATION: usize>() -> fn(
        src: &[T],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<T>,
        dst_stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start_x: u32,
        end_x: u32,
    );
}

impl BoxBlurVerticalPass<f16> for f16 {
    #[allow(clippy::type_complexity)]
    fn get_box_vertical_pass<const CHANNELS_CONFIGURATION: usize>(
    ) -> fn(&[f16], u32, &UnsafeSlice<f16>, u32, u32, u32, u32, u32, u32) {
        box_blur_vertical_pass_impl::<f16, f32>
    }
}

impl BoxBlurVerticalPass<f32> for f32 {
    #[allow(clippy::type_complexity)]
    fn get_box_vertical_pass<const CHANNELS_CONFIGURATION: usize>(
    ) -> fn(&[f32], u32, &UnsafeSlice<f32>, u32, u32, u32, u32, u32, u32) {
        box_blur_vertical_pass_impl::<f32, f32>
    }
}

impl BoxBlurVerticalPass<u16> for u16 {
    #[allow(clippy::type_complexity)]
    fn get_box_vertical_pass<const CHANNELS_CONFIGURATION: usize>(
    ) -> fn(&[u16], u32, &UnsafeSlice<u16>, u32, u32, u32, u32, u32, u32) {
        box_blur_vertical_pass_impl::<u16, u32>
    }
}

impl BoxBlurVerticalPass<u8> for u8 {
    #[allow(clippy::type_complexity)]
    fn get_box_vertical_pass<const CHANNELS_CONFIGURATION: usize>(
    ) -> fn(&[u8], u32, &UnsafeSlice<u8>, u32, u32, u32, u32, u32, u32) {
        let mut _dispatcher_vertical: fn(
            src: &[u8],
            src_stride: u32,
            unsafe_dst: &UnsafeSlice<u8>,
            dst_stride: u32,
            width: u32,
            height: u32,
            radius: u32,
            start_x: u32,
            end_x: u32,
        ) = box_blur_vertical_pass_impl::<u8, u32>;
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            #[cfg(feature = "sse")]
            {
                let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                if is_sse_available {
                    use crate::box_filter::box_blur_sse::box_blur_vertical_pass_sse;
                    _dispatcher_vertical = box_blur_vertical_pass_sse::<u8>;
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let is_avx_available = std::arch::is_x86_feature_detected!("avx2");
            if is_avx_available {
                use crate::box_filter::box_blur_avx::box_blur_vertical_pass_avx2;
                _dispatcher_vertical = box_blur_vertical_pass_avx2::<u8>;
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _dispatcher_vertical = box_blur_vertical_pass_neon::<u8>;
            #[cfg(feature = "rdm")]
            {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    use crate::box_filter::box_blur_neon_q0_31::box_blur_vertical_pass_neon_rdm;
                    _dispatcher_vertical = box_blur_vertical_pass_neon_rdm::<u8>;
                }
            }
        }
        _dispatcher_vertical
    }
}

#[allow(clippy::type_complexity)]
fn box_blur_vertical_pass<
    T: FromPrimitive
        + Default
        + Sync
        + Send
        + Copy
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + AsPrimitive<u32>
        + AsPrimitive<u64>
        + AsPrimitive<f32>
        + AsPrimitive<f64>
        + BoxBlurVerticalPass<T>,
    const CN: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    pool: &Option<ThreadPool>,
    thread_count: u32,
) where
    f32: ToStorage<T>,
{
    let _dispatcher_vertical = T::get_box_vertical_pass::<CN>();
    let unsafe_dst = UnsafeSlice::new(dst);

    if let Some(pool) = pool {
        pool.scope(|scope| {
            let total_width = width as usize * CN;
            let segment_size = total_width / thread_count as usize;
            for i in 0..thread_count as usize {
                let start_x = i * segment_size;
                let mut end_x = (i + 1) * segment_size;
                if i == thread_count as usize - 1 {
                    end_x = total_width;
                }

                scope.spawn(move |_| {
                    _dispatcher_vertical(
                        src,
                        src_stride,
                        &unsafe_dst,
                        dst_stride,
                        width,
                        height,
                        radius,
                        start_x as u32,
                        end_x as u32,
                    );
                });
            }
        });
    } else {
        let total_width = width as usize * CN;
        _dispatcher_vertical(
            src,
            src_stride,
            &unsafe_dst,
            dst_stride,
            width,
            height,
            radius,
            0,
            total_width as u32,
        );
    }
}

fn box_blur_impl<
    T: FromPrimitive
        + Default
        + Sync
        + Send
        + Copy
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + AsPrimitive<u32>
        + AsPrimitive<u64>
        + AsPrimitive<f32>
        + AsPrimitive<f64>
        + BoxBlurHorizontalPass<T>
        + BoxBlurVerticalPass<T>,
    const CN: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    pool: &Option<ThreadPool>,
    thread_count: u32,
) -> Result<(), BlurError>
where
    f32: ToStorage<T>,
{
    check_slice_size(
        src,
        src_stride as usize,
        width as usize,
        height as usize,
        CN,
    )?;
    check_slice_size(
        dst,
        dst_stride as usize,
        width as usize,
        height as usize,
        CN,
    )?;
    let mut transient: Vec<T> = vec![T::default(); dst_stride as usize * height as usize];
    box_blur_horizontal_pass::<T, CN>(
        src,
        src_stride,
        &mut transient,
        dst_stride,
        width,
        height,
        radius,
        pool,
        thread_count,
    );

    box_blur_vertical_pass::<T, CN>(
        &transient,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        radius,
        pool,
        thread_count,
    );

    Ok(())
}

/// Performs box blur on the image.
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn box_blur(
    image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    radius: u32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let width = image.width;
    let height = image.height;
    let thread_count = threading_policy.thread_count(width, height) as u32;
    let _dispatcher = match image.channels {
        FastBlurChannels::Plane => box_blur_impl::<u8, 1>,
        FastBlurChannels::Channels3 => box_blur_impl::<u8, 3>,
        FastBlurChannels::Channels4 => box_blur_impl::<u8, 4>,
    };
    let pool = if thread_count == 1 {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count as usize)
                .build()
                .unwrap(),
        )
    };
    let dst_stride = dst_image.row_stride();
    let dst = dst_image.data.borrow_mut();
    _dispatcher(
        image.data.as_ref(),
        image.row_stride(),
        dst,
        dst_stride,
        width,
        height,
        radius,
        &pool,
        thread_count,
    )?;
    Ok(())
}

/// Performs box blur on the image.
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn box_blur_u16(
    image: &BlurImage<u16>,
    dst_image: &mut BlurImageMut<u16>,
    radius: u32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let width = image.width;
    let height = image.height;
    let thread_count = threading_policy.thread_count(width, height) as u32;
    let pool = if thread_count == 1 {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count as usize)
                .build()
                .unwrap(),
        )
    };
    let dispatcher = match image.channels {
        FastBlurChannels::Plane => box_blur_impl::<u16, 1>,
        FastBlurChannels::Channels3 => box_blur_impl::<u16, 3>,
        FastBlurChannels::Channels4 => box_blur_impl::<u16, 4>,
    };
    let dst_stride = dst_image.row_stride();
    let dst = dst_image.data.borrow_mut();
    dispatcher(
        image.data.as_ref(),
        image.row_stride(),
        dst,
        dst_stride,
        width,
        height,
        radius,
        &pool,
        thread_count,
    )?;
    Ok(())
}

/// Performs box blur on the image.
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn box_blur_f32(
    image: &BlurImage<f32>,
    dst_image: &mut BlurImageMut<f32>,
    radius: u32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let width = image.width;
    let height = image.height;
    let thread_count = threading_policy.thread_count(width, height) as u32;
    let pool = if thread_count == 1 {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count as usize)
                .build()
                .unwrap(),
        )
    };
    let dispatcher = match image.channels {
        FastBlurChannels::Plane => box_blur_impl::<f32, 1>,
        FastBlurChannels::Channels3 => box_blur_impl::<f32, 3>,
        FastBlurChannels::Channels4 => box_blur_impl::<f32, 4>,
    };
    let dst_stride = dst_image.row_stride();
    let dst = dst_image.data.borrow_mut();
    dispatcher(
        image.data.as_ref(),
        image.row_stride(),
        dst,
        dst_stride,
        width,
        height,
        radius,
        &pool,
        thread_count,
    )
}

/// Performs box blur on the image in linear colorspace
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
/// * `threading_policy` - Threads usage policy.
/// * `transfer_function` - Transfer function in linear colorspace.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn box_blur_in_linear(
    image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    radius: u32,
    threading_policy: ThreadingPolicy,
    transfer_function: TransferFunction,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;

    let mut linear_data = BlurImageMut::alloc(image.width, image.height, image.channels);
    let mut linear_data_2 = BlurImageMut::alloc(image.width, image.height, image.channels);

    let forward_transformer = match image.channels {
        FastBlurChannels::Plane => plane_to_linear,
        FastBlurChannels::Channels3 => rgb_to_linear,
        FastBlurChannels::Channels4 => rgba_to_linear,
    };

    let inverse_transformer = match image.channels {
        FastBlurChannels::Plane => linear_to_plane,
        FastBlurChannels::Channels3 => linear_to_rgb,
        FastBlurChannels::Channels4 => linear_to_rgba,
    };

    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let dst_stride = dst_image.row_stride();

    forward_transformer(
        image.data.as_ref(),
        image.row_stride(),
        linear_data.data.borrow_mut(),
        width * size_of::<f32>() as u32 * channels.channels() as u32,
        width,
        height,
        transfer_function,
    );

    let linear_close = linear_data.to_immutable_ref();

    box_blur_f32(&linear_close, &mut linear_data_2, radius, threading_policy)?;

    inverse_transformer(
        linear_data_2.data.borrow_mut(),
        width * size_of::<f32>() as u32 * channels.channels() as u32,
        dst_image.data.borrow_mut(),
        dst_stride,
        width,
        height,
        transfer_function,
    );
    Ok(())
}

#[inline]
fn create_box_gauss(sigma: f32, n: usize) -> Vec<u32> {
    let n_float = n as f32;

    // Ideal averaging filter width
    let w_ideal = (12.0 * sigma * sigma / n_float).sqrt() + 1.0;
    let mut wl: u32 = w_ideal.floor() as u32;

    if wl % 2 == 0 {
        wl -= 1;
    };

    let wu = wl + 2;

    let wl_float = wl as f32;
    let m_ideal = (12.0 * sigma * sigma
        - n_float * wl_float * wl_float
        - 4.0 * n_float * wl_float
        - 3.0 * n_float)
        / (-4.0 * wl_float - 4.0);
    let m: usize = m_ideal.round() as usize;

    let mut sizes = Vec::<u32>::new();

    for i in 0..n {
        if i < m {
            let mut new_val = wl / 2;
            if new_val % 2 == 0 {
                new_val += 1;
            }
            sizes.push(new_val);
        } else {
            let mut new_val = wu / 2;
            if new_val % 2 == 0 {
                new_val += 1;
            }
            sizes.push(new_val);
        }
    }

    sizes
}

fn tent_blur_impl<
    T: FromPrimitive
        + Default
        + Sync
        + Send
        + Copy
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + AsPrimitive<u32>
        + AsPrimitive<u64>
        + AsPrimitive<f32>
        + AsPrimitive<f64>
        + BoxBlurHorizontalPass<T>
        + BoxBlurVerticalPass<T>,
    const CN: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    f32: ToStorage<T>,
{
    assert!(sigma > 0.0, "Sigma can't be 0");
    let thread_count = threading_policy.thread_count(width, height) as u32;
    let pool = if thread_count == 1 {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count as usize)
                .build()
                .unwrap(),
        )
    };
    let mut transient: Vec<T> =
        vec![T::from_u32(0).unwrap_or_default(); width as usize * height as usize * CN];
    let boxes = create_box_gauss(sigma, 2);
    box_blur_impl::<T, CN>(
        src,
        src_stride,
        &mut transient,
        width * CN as u32,
        width,
        height,
        boxes[0],
        &pool,
        thread_count,
    )?;
    box_blur_impl::<T, CN>(
        &transient,
        width * CN as u32,
        dst,
        dst_stride,
        width,
        height,
        boxes[1],
        &pool,
        thread_count,
    )?;
    Ok(())
}

/// Performs tent blur on the image.
///
/// Tent blur just makes a two passes box blur on the image since two times box it is almost equal to tent filter.
/// <https://en.wikipedia.org/wiki/Central_limit_theorem>
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn tent_blur(
    image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let dispatcher = match image.channels {
        FastBlurChannels::Plane => tent_blur_impl::<u8, 1>,
        FastBlurChannels::Channels3 => tent_blur_impl::<u8, 3>,
        FastBlurChannels::Channels4 => tent_blur_impl::<u8, 4>,
    };
    let src = image.data.as_ref();
    let src_stride = image.row_stride();
    let dst_stride = dst_image.row_stride();
    let dst = dst_image.data.borrow_mut();
    let width = image.width;
    let height = image.height;
    dispatcher(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        sigma,
        threading_policy,
    )
}

/// Performs tent blur on the image.
///
/// Tent blur just makes a two passes box blur on the image since two times box it is almost equal to tent filter.
/// <https://en.wikipedia.org/wiki/Central_limit_theorem>
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn tent_blur_u16(
    image: &BlurImage<u16>,
    dst_image: &mut BlurImageMut<u16>,
    sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let dispatcher = match image.channels {
        FastBlurChannels::Plane => tent_blur_impl::<u16, 1>,
        FastBlurChannels::Channels3 => tent_blur_impl::<u16, 3>,
        FastBlurChannels::Channels4 => tent_blur_impl::<u16, 4>,
    };
    let src = image.data.as_ref();
    let src_stride = image.row_stride();
    let dst_stride = dst_image.row_stride();
    let dst = dst_image.data.borrow_mut();
    let width = image.width;
    let height = image.height;
    dispatcher(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        sigma,
        threading_policy,
    )
}

/// Performs tent blur on the image.
///
/// Tent blur just makes a two passes box blur on the image since two times box it is almost equal to tent filter.
/// <https://en.wikipedia.org/wiki/Central_limit_theorem>
///
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn tent_blur_f32(
    image: &BlurImage<f32>,
    dst_image: &mut BlurImageMut<f32>,
    sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let dispatcher = match image.channels {
        FastBlurChannels::Plane => tent_blur_impl::<f32, 1>,
        FastBlurChannels::Channels3 => tent_blur_impl::<f32, 3>,
        FastBlurChannels::Channels4 => tent_blur_impl::<f32, 4>,
    };
    let src = image.data.as_ref();
    let src_stride = image.row_stride();
    let dst_stride = dst_image.row_stride();
    let dst = dst_image.data.borrow_mut();
    let width = image.width;
    let height = image.height;
    dispatcher(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        sigma,
        threading_policy,
    )
}

/// Performs tent blur on the image in linear colorspace
///
/// Tent blur just makes a two passes box blur on the image since two times box it is almost equal to tent filter.
/// <https://en.wikipedia.org/wiki/Central_limit_theorem>
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn tent_blur_in_linear(
    image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    sigma: f32,
    threading_policy: ThreadingPolicy,
    transfer_function: TransferFunction,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;

    let mut linear_data = BlurImageMut::alloc(image.width, image.height, image.channels);
    let mut linear_data_2 = BlurImageMut::alloc(image.width, image.height, image.channels);

    let forward_transformer = match image.channels {
        FastBlurChannels::Plane => plane_to_linear,
        FastBlurChannels::Channels3 => rgb_to_linear,
        FastBlurChannels::Channels4 => rgba_to_linear,
    };

    let inverse_transformer = match image.channels {
        FastBlurChannels::Plane => linear_to_plane,
        FastBlurChannels::Channels3 => linear_to_rgb,
        FastBlurChannels::Channels4 => linear_to_rgba,
    };

    let channels = image.channels;
    let width = image.width;

    forward_transformer(
        image.data.as_ref(),
        image.row_stride(),
        linear_data.data.borrow_mut(),
        width * size_of::<f32>() as u32 * channels.channels() as u32,
        width,
        image.height,
        transfer_function,
    );

    let immutable_linear_ref = linear_data.to_immutable_ref();

    tent_blur_f32(
        &immutable_linear_ref,
        &mut linear_data_2,
        sigma,
        threading_policy,
    )?;

    let height = image.height;
    let dst_stride = dst_image.row_stride();

    inverse_transformer(
        linear_data_2.data.borrow_mut(),
        width * size_of::<f32>() as u32 * channels.channels() as u32,
        dst_image.data.borrow_mut(),
        dst_stride,
        width,
        height,
        transfer_function,
    );
    Ok(())
}

fn gaussian_box_blur_impl<
    T: FromPrimitive
        + Default
        + Sync
        + Send
        + Copy
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + AsPrimitive<u32>
        + AsPrimitive<u64>
        + AsPrimitive<f32>
        + AsPrimitive<f64>
        + BoxBlurHorizontalPass<T>
        + BoxBlurVerticalPass<T>,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    f32: ToStorage<T>,
{
    assert!(sigma > 0.0, "Sigma can't be 0");
    let thread_count = threading_policy.thread_count(width, height) as u32;
    let pool = if thread_count == 1 {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count as usize)
                .build()
                .unwrap(),
        )
    };
    let mut transient: Vec<T> = vec![T::default(); dst_stride as usize * height as usize];
    let boxes = create_box_gauss(sigma, 3);
    box_blur_impl::<T, CHANNEL_CONFIGURATION>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        boxes[0],
        &pool,
        thread_count,
    )?;
    box_blur_impl::<T, CHANNEL_CONFIGURATION>(
        dst,
        dst_stride,
        &mut transient,
        dst_stride,
        width,
        height,
        boxes[1],
        &pool,
        thread_count,
    )?;
    box_blur_impl::<T, CHANNEL_CONFIGURATION>(
        &transient,
        dst_stride,
        dst,
        dst_stride,
        width,
        height,
        boxes[2],
        &pool,
        thread_count,
    )?;
    Ok(())
}

/// Performs gaussian box blur approximation on the image.
///
/// This method launches three times box blur on the image since 2 passes box filter it is a tent filter and 3 passes of box blur it is almost gaussian filter.
/// <https://en.wikipedia.org/wiki/Central_limit_theorem>
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// Even it is having low complexity it is slow filter.
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `sigma` - Flattening level.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_box_blur(
    image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let dispatcher = match image.channels {
        FastBlurChannels::Plane => gaussian_box_blur_impl::<u8, 1>,
        FastBlurChannels::Channels3 => gaussian_box_blur_impl::<u8, 3>,
        FastBlurChannels::Channels4 => gaussian_box_blur_impl::<u8, 4>,
    };
    let src = image.data.as_ref();
    let src_stride = image.row_stride();
    let dst_stride = dst_image.row_stride();
    let dst = dst_image.data.borrow_mut();
    let width = image.width;
    let height = image.height;
    dispatcher(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        sigma,
        threading_policy,
    )
}

/// Performs gaussian box blur approximation on the image.
///
/// This method launches three times box blur on the image since 2 passes box filter it is a tent filter and 3 passes of box blur it is almost gaussian filter.
/// <https://en.wikipedia.org/wiki/Central_limit_theorem>
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// Even it is having low complexity it is slow filter.
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `sigma` - Flattening level.
/// * `threading_policy` - Threading policy, see [ThreadingPolicy] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_box_blur_u16(
    image: &BlurImage<u16>,
    dst_image: &mut BlurImageMut<u16>,
    sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let channels = image.channels;
    let executor = match channels {
        FastBlurChannels::Plane => gaussian_box_blur_impl::<u16, 1>,
        FastBlurChannels::Channels3 => gaussian_box_blur_impl::<u16, 3>,
        FastBlurChannels::Channels4 => gaussian_box_blur_impl::<u16, 4>,
    };
    let src = image.data.as_ref();
    let src_stride = image.row_stride();
    let dst_stride = dst_image.row_stride();
    let dst = dst_image.data.borrow_mut();
    let width = image.width;
    let height = image.height;
    executor(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        sigma,
        threading_policy,
    )
}

/// Performs gaussian box blur approximation on the image.
///
/// This method launches three times box blur on the image since 2 passes box filter it is a tent filter and 3 passes of box blur it is almost gaussian filter.
/// <https://en.wikipedia.org/wiki/Central_limit_theorem>
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// Even it is having low complexity it is slow filter.
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `sigma` - Flattening level.
/// * `threading_policy` - Threading policy, see [ThreadingPolicy] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_box_blur_f32(
    image: &BlurImage<f32>,
    dst_image: &mut BlurImageMut<f32>,
    sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let channels = image.channels;
    let dispatcher = match channels {
        FastBlurChannels::Plane => gaussian_box_blur_impl::<f32, 1>,
        FastBlurChannels::Channels3 => gaussian_box_blur_impl::<f32, 3>,
        FastBlurChannels::Channels4 => gaussian_box_blur_impl::<f32, 4>,
    };
    let src = image.data.as_ref();
    let src_stride = image.row_stride();
    let dst_stride = dst_image.row_stride();
    let dst = dst_image.data.borrow_mut();
    let width = image.width;
    let height = image.height;
    dispatcher(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        sigma,
        threading_policy,
    )
}

/// Performs gaussian box blur approximation on the image.
///
/// This method launches three times box blur on the image since 2 passes box filter it is a tent filter and 3 passes of box blur it is almost gaussian filter.
/// <https://en.wikipedia.org/wiki/Central_limit_theorem>
///
/// Convergence of this function is very high so strong effect applies very fast
///
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Source immutable image, see [BlurImage] for more info.
/// * `dst_image` - Destination mutable image, see [BlurImageMut] for more info.
/// * `sigma` - Flattening level.
/// * `threading_policy` - Threading policy, see [ThreadingPolicy] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_box_blur_in_linear(
    image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    sigma: f32,
    threading_policy: ThreadingPolicy,
    transfer_function: TransferFunction,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    let mut linear_data = BlurImageMut::alloc(image.width, image.height, image.channels);
    let mut linear_data_2 = BlurImageMut::alloc(image.width, image.height, image.channels);

    let forward_transformer = match image.channels {
        FastBlurChannels::Plane => plane_to_linear,
        FastBlurChannels::Channels3 => rgb_to_linear,
        FastBlurChannels::Channels4 => rgba_to_linear,
    };

    let inverse_transformer = match image.channels {
        FastBlurChannels::Plane => linear_to_plane,
        FastBlurChannels::Channels3 => linear_to_rgb,
        FastBlurChannels::Channels4 => linear_to_rgba,
    };

    let channels = image.channels;
    let width = image.width;
    let height = image.height;

    forward_transformer(
        image.data.as_ref(),
        image.row_stride(),
        linear_data.data.borrow_mut(),
        width * size_of::<f32>() as u32 * channels.channels() as u32,
        width,
        height,
        transfer_function,
    );

    let immutable_ref = linear_data.to_immutable_ref();

    gaussian_box_blur_f32(&immutable_ref, &mut linear_data_2, sigma, threading_policy)?;

    let dst_stride = dst_image.row_stride();

    inverse_transformer(
        linear_data_2.data.borrow(),
        width * std::mem::size_of::<f32>() as u32 * channels.channels() as u32,
        dst_image.data.borrow_mut(),
        dst_stride,
        width,
        height,
        transfer_function,
    );
    Ok(())
}
