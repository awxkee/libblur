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

use crate::channels_configuration::FastBlurChannels;
use crate::primitives::PrimitiveCast;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::util::check_slice_size;
use crate::{BlurError, BlurImage, BlurImageMut, ThreadingPolicy};
#[cfg(feature = "nightly_f16")]
use core::f16;
use novtb::{ParallelZonedIterator, TbSliceMut};
use num_traits::cast::FromPrimitive;
use num_traits::AsPrimitive;
use std::fmt::Debug;

/// Both kernels are expected to be odd.
#[derive(Copy, Clone, Debug)]
pub struct BoxBlurParameters {
    /// X-axis kernel size
    pub x_axis_kernel: u32,
    /// Y-axis kernel size
    pub y_axis_kernel: u32,
}

/// Central limit theorem based blurs parameters.
pub struct CLTParameters {
    /// X-axis sigma
    pub x_sigma: f32,
    /// Y-axis sigma
    pub y_sigma: f32,
}

impl CLTParameters {
    pub fn new(sigma: f32) -> CLTParameters {
        CLTParameters {
            x_sigma: sigma,
            y_sigma: sigma,
        }
    }

    fn validate(&self) -> Result<(), BlurError> {
        if self.x_sigma <= 0. {
            return Err(BlurError::NegativeOrZeroSigma);
        }
        if self.y_sigma <= 0. {
            return Err(BlurError::NegativeOrZeroSigma);
        }
        Ok(())
    }
}

impl BoxBlurParameters {
    /// Kernel is expected to be odd.
    pub fn new(kernel: u32) -> BoxBlurParameters {
        BoxBlurParameters {
            x_axis_kernel: kernel,
            y_axis_kernel: kernel,
        }
    }

    fn x_radius(&self) -> u32 {
        (self.x_axis_kernel / 2).max(1)
    }

    fn y_radius(&self) -> u32 {
        (self.y_axis_kernel / 2).max(1)
    }

    fn validate(&self) -> Result<(), BlurError> {
        if self.x_axis_kernel % 2 == 0 {
            return Err(BlurError::OddKernel(self.x_axis_kernel as usize));
        }
        if self.y_axis_kernel % 2 == 0 {
            return Err(BlurError::OddKernel(self.y_axis_kernel as usize));
        }
        Ok(())
    }
}

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
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + Default + Send + Sync + PrimitiveCast<J>,
    J: Copy
        + std::ops::Mul<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + PrimitiveCast<f32>,
    f32: ToStorage<T>,
    u32: PrimitiveCast<J>,
{
    let kernel_size = radius * 2 + 1;
    let edge_count: J = ((kernel_size / 2) + 1).cast_();
    let half_kernel = kernel_size / 2;

    let weight = 1f32 / (radius * 2 + 1) as f32;

    for y in start_y..end_y {
        let mut weight1: J = 0u32.cast_();
        let mut weight2: J = 0u32.cast_();
        let mut weight3: J = 0u32.cast_();
        let y_src_shift = (y * src_stride) as usize;
        let y_dst_shift = (y * dst_stride) as usize;

        let dst_row = unsafe {
            std::slice::from_raw_parts_mut(
                unsafe_dst.slice.get_unchecked(y_dst_shift).get(),
                width as usize * CN,
            )
        };

        // replicate edge
        let mut weight0 = (unsafe { *src.get_unchecked(y_src_shift) }.cast_()) * edge_count;
        if CN > 1 {
            weight1 = (unsafe { *src.get_unchecked(y_src_shift + 1) }.cast_()) * edge_count;
        }
        if CN > 2 {
            weight2 = (unsafe { *src.get_unchecked(y_src_shift + 2) }.cast_()) * edge_count;
        }
        if CN == 4 {
            weight3 = (unsafe { *src.get_unchecked(y_src_shift + 3) }.cast_()) * edge_count;
        }

        for x in 1..=half_kernel as usize {
            let px = x.min(width as usize - 1) * CN;
            weight0 += unsafe { *src.get_unchecked(y_src_shift + px) }.cast_();
            if CN > 1 {
                weight1 += unsafe { *src.get_unchecked(y_src_shift + px + 1) }.cast_();
            }
            if CN > 2 {
                weight2 += unsafe { *src.get_unchecked(y_src_shift + px + 2) }.cast_();
            }
            if CN == 4 {
                weight3 += unsafe { *src.get_unchecked(y_src_shift + px + 3) }.cast_();
            }
        }

        let current_row = &src[y_src_shift..(y_src_shift + width as usize * CN)];

        for x in 0..(half_kernel as usize).min(width as usize) {
            let next = (x + half_kernel as usize + 1).min(width as usize - 1) * CN;
            let previous = (x as i64 - half_kernel as i64).max(0) as usize * CN;
            let px = x * CN;
            // Prune previous and add next and compute mean

            unsafe {
                let write_offset = px;
                *dst_row.get_unchecked_mut(write_offset) = (weight0.cast_() * weight).to_();
                if CN > 1 {
                    *dst_row.get_unchecked_mut(write_offset + 1) = (weight1.cast_() * weight).to_();
                }
                if CN > 2 {
                    *dst_row.get_unchecked_mut(write_offset + 2) = (weight2.cast_() * weight).to_();
                }
                if CN == 4 {
                    *dst_row.get_unchecked_mut(write_offset + 3) = (weight3.cast_() * weight).to_();
                }
            }

            weight0 += unsafe { *current_row.get_unchecked(next) }.cast_();
            if CN > 1 {
                weight1 += unsafe { *current_row.get_unchecked(next + 1) }.cast_();
            }
            if CN > 2 {
                weight2 += unsafe { *current_row.get_unchecked(next + 2) }.cast_();
            }

            weight0 -= unsafe { *current_row.get_unchecked(previous) }.cast_();
            if CN > 1 {
                weight1 -= unsafe { *current_row.get_unchecked(previous + 1) }.cast_();
            }
            if CN > 2 {
                weight2 -= unsafe { *current_row.get_unchecked(previous + 2) }.cast_();
            }

            if CN == 4 {
                weight3 += unsafe { *current_row.get_unchecked(next + 3) }.cast_();
                weight3 -= unsafe { *current_row.get_unchecked(previous + 3) }.cast_();
            }
        }

        let max_x_before_clamping = (width - 1).saturating_sub(half_kernel + 1);
        let row_length = current_row.len();

        let mut last_processed_item = half_kernel;

        if ((half_kernel as usize * 2 + 1) * CN < row_length)
            && ((max_x_before_clamping as usize * CN) < row_length)
        {
            let data_section = current_row;
            let advanced_kernel_part = &data_section[(half_kernel as usize * 2 + 1) * CN..];
            let section_length = max_x_before_clamping as usize - half_kernel as usize;
            let dst = &mut dst_row
                [half_kernel as usize * CN..(half_kernel as usize * CN + section_length * CN)];

            for ((dst, src_previous), src_next) in dst
                .chunks_exact_mut(CN)
                .zip(data_section.chunks_exact(CN))
                .zip(advanced_kernel_part.chunks_exact(CN))
            {
                dst[0] = (weight0.cast_() * weight).to_();
                if CN > 1 {
                    dst[1] = (weight1.cast_() * weight).to_();
                }
                if CN > 2 {
                    dst[2] = (weight2.cast_() * weight).to_();
                }
                if CN == 4 {
                    dst[3] = (weight3.cast_() * weight).to_();
                }

                weight0 += src_next[0].cast_();
                if CN > 1 {
                    weight1 += src_next[1].cast_();
                }
                if CN > 2 {
                    weight2 += src_next[2].cast_();
                }
                if CN == 4 {
                    weight3 += src_next[3].cast_();
                }

                weight0 -= src_previous[0].cast_();
                if CN > 1 {
                    weight1 -= src_previous[1].cast_();
                }
                if CN > 2 {
                    weight2 -= src_previous[2].cast_();
                }
                if CN == 4 {
                    weight3 -= src_previous[3].cast_();
                }
            }

            last_processed_item = max_x_before_clamping;
        }

        for x in last_processed_item as usize..width as usize {
            let next = (x + half_kernel as usize + 1).min(width as usize - 1) * CN;
            let previous = (x as i64 - half_kernel as i64).max(0) as usize * CN;
            let px = x * CN;
            // Prune previous and add next and compute mean

            unsafe {
                let write_offset = px;
                *dst_row.get_unchecked_mut(write_offset) = (weight0.cast_() * weight).to_();
                if CN > 1 {
                    *dst_row.get_unchecked_mut(write_offset + 1) = (weight1.cast_() * weight).to_();
                }
                if CN > 2 {
                    *dst_row.get_unchecked_mut(write_offset + 2) = (weight2.cast_() * weight).to_();
                }
                if CN == 4 {
                    *dst_row.get_unchecked_mut(write_offset + 3) = (weight3.cast_() * weight).to_();
                }
            }

            weight0 += unsafe { *current_row.get_unchecked(next) }.cast_();
            if CN > 1 {
                weight1 += unsafe { *current_row.get_unchecked(next + 1) }.cast_();
            }
            if CN > 2 {
                weight2 += unsafe { *current_row.get_unchecked(next + 2) }.cast_();
            }

            weight0 -= unsafe { *current_row.get_unchecked(previous) }.cast_();
            if CN > 1 {
                weight1 -= unsafe { *current_row.get_unchecked(previous + 1) }.cast_();
            }
            if CN > 2 {
                weight2 -= unsafe { *current_row.get_unchecked(previous + 2) }.cast_();
            }

            if CN == 4 {
                weight3 += unsafe { *current_row.get_unchecked(next + 3) }.cast_();
                weight3 -= unsafe { *current_row.get_unchecked(previous + 3) }.cast_();
            }
        }
    }
}

trait BoxBlurHorizontalPass<T> {
    #[allow(clippy::type_complexity)]
    fn get_horizontal_pass<const CN: usize>() -> fn(
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
    fn get_horizontal_pass<const CN: usize>(
    ) -> fn(&[f32], u32, &UnsafeSlice<f32>, u32, u32, u32, u32, u32) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::box_filter::neon::box_blur_horizontal_pass_neon_rgba_f32;
            box_blur_horizontal_pass_neon_rgba_f32::<CN>
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::box_filter::avx::box_blur_horizontal_pass_avx_f32;
                return box_blur_horizontal_pass_avx_f32::<CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::box_filter::sse::box_blur_horizontal_pass_sse_f32;
                return box_blur_horizontal_pass_sse_f32::<CN>;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            box_blur_horizontal_pass_impl::<f32, f32, CN>
        }
    }
}

#[cfg(feature = "nightly_f16")]
impl BoxBlurHorizontalPass<f16> for f16 {
    #[allow(clippy::type_complexity)]
    fn get_horizontal_pass<const CN: usize>(
    ) -> fn(&[f16], u32, &UnsafeSlice<f16>, u32, u32, u32, u32, u32) {
        box_blur_horizontal_pass_impl::<f16, f32, CN>
    }
}

impl BoxBlurHorizontalPass<u16> for u16 {
    #[allow(clippy::type_complexity)]
    fn get_horizontal_pass<const CN: usize>(
    ) -> fn(&[u16], u32, &UnsafeSlice<u16>, u32, u32, u32, u32, u32) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::box_filter::neon::box_blur_horizontal_pass_neon_rgba16;
            box_blur_horizontal_pass_neon_rgba16::<CN>
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::box_filter::sse::box_blur_horizontal_pass_sse16;
                return box_blur_horizontal_pass_sse16::<CN>;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            box_blur_horizontal_pass_impl::<u16, u32, CN>
        }
    }
}

impl BoxBlurHorizontalPass<u8> for u8 {
    #[allow(clippy::type_complexity)]
    fn get_horizontal_pass<const CN: usize>(
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
        ) = box_blur_horizontal_pass_impl::<u8, u32, CN>;
        if CN >= 3 {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::box_filter::neon::box_blur_horizontal_pass_neon;
                _dispatcher_horizontal = box_blur_horizontal_pass_neon::<CN>;
                #[cfg(feature = "rdm")]
                {
                    use crate::box_filter::neon::box_blur_horizontal_pass_neon_rdm;
                    if std::arch::is_aarch64_feature_detected!("rdm") {
                        _dispatcher_horizontal = box_blur_horizontal_pass_neon_rdm::<CN>;
                    }
                }
            }
            #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
            {
                if std::arch::is_x86_feature_detected!("sse4.1") {
                    use crate::box_filter::sse::box_blur_horizontal_pass_sse;
                    _dispatcher_horizontal = box_blur_horizontal_pass_sse::<{ CN }>;
                }
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    use crate::box_filter::avx::box_blur_horizontal_pass_avx;
                    _dispatcher_horizontal = box_blur_horizontal_pass_avx::<{ CN }>;
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
    const CN: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    thread_count: u32,
) where
    f32: ToStorage<T>,
{
    let _dispatcher_horizontal = T::get_horizontal_pass::<CN>();
    let unsafe_dst = UnsafeSlice::new(dst);
    let pool = novtb::ThreadPool::new(thread_count as usize);
    pool.parallel_for(|thread_index| {
        let segment_size = height / thread_count;
        let start_y = thread_index as u32 * segment_size;
        let mut end_y = (thread_index as u32 + 1) * segment_size;
        if thread_index as u32 == thread_count - 1 {
            end_y = height;
        }
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
    T: std::ops::AddAssign + std::ops::SubAssign + Copy + Default + Send + Sync + PrimitiveCast<J>,
    J: Copy
        + std::ops::Mul<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + PrimitiveCast<f32>
        + Default,
    f32: ToStorage<T>,
    u32: PrimitiveCast<J>,
{
    let kernel_size = radius * 2 + 1;

    let edge_count: J = ((kernel_size / 2) + 1).cast_();
    let half_kernel = kernel_size / 2;

    let weight = 1f32 / (radius * 2 + 1) as f32;

    let buf_size = end_x - start_x;

    let buf_cap = buf_size as usize;
    let mut buffer = vec![J::default(); buf_cap];

    let src_lane = &src[start_x as usize..end_x as usize];

    for (x, (v, bf)) in src_lane.iter().zip(buffer.iter_mut()).enumerate() {
        let mut w = v.cast_() * edge_count;
        for y in 1..=half_kernel as usize {
            let y_src_shift = y.min(height as usize - 1) * src_stride as usize;
            unsafe {
                w += src
                    .get_unchecked(y_src_shift + x + start_x as usize)
                    .cast_();
            }
        }
        *bf = w;
    }

    for y in 0..height {
        let next = (y + half_kernel + 1).min(height - 1) as usize * src_stride as usize;
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

            *dst = (weight0.cast_() * weight).to_();

            weight0 += src_next.cast_();
            weight0 -= src_previous.cast_();

            *buffer = weight0;
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

#[cfg(feature = "nightly_f16")]
impl BoxBlurVerticalPass<f16> for f16 {
    #[allow(clippy::type_complexity)]
    fn get_box_vertical_pass<const CHANNELS_CONFIGURATION: usize>(
    ) -> fn(&[f16], u32, &UnsafeSlice<f16>, u32, u32, u32, u32, u32, u32) {
        box_blur_vertical_pass_impl::<f16, f32>
    }
}

impl BoxBlurVerticalPass<f32> for f32 {
    #[allow(clippy::type_complexity)]
    fn get_box_vertical_pass<const CN: usize>(
    ) -> fn(&[f32], u32, &UnsafeSlice<f32>, u32, u32, u32, u32, u32, u32) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::box_filter::neon::box_blur_vertical_pass_neon_rgba_f32;
            box_blur_vertical_pass_neon_rgba_f32
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            box_blur_vertical_pass_impl::<f32, f32>
        }
    }
}

impl BoxBlurVerticalPass<u16> for u16 {
    #[allow(clippy::type_complexity)]
    fn get_box_vertical_pass<const CN: usize>(
    ) -> fn(&[u16], u32, &UnsafeSlice<u16>, u32, u32, u32, u32, u32, u32) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::box_filter::neon::box_blur_vertical_pass_neon_rgba16;
            box_blur_vertical_pass_neon_rgba16
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::box_filter::sse::box_blur_vertical_pass_sse16;
                return box_blur_vertical_pass_sse16;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            box_blur_vertical_pass_impl::<u16, u32>
        }
    }
}

impl BoxBlurVerticalPass<u8> for u8 {
    #[allow(clippy::type_complexity)]
    fn get_box_vertical_pass<const CN: usize>(
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
                    use crate::box_filter::sse::box_blur_vertical_pass_sse;
                    _dispatcher_vertical = box_blur_vertical_pass_sse;
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let is_avx_available = std::arch::is_x86_feature_detected!("avx2");
            if is_avx_available {
                use crate::box_filter::avx::box_blur_vertical_pass_avx2;
                _dispatcher_vertical = box_blur_vertical_pass_avx2;
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::box_filter::neon::box_blur_vertical_pass_neon;
            _dispatcher_vertical = box_blur_vertical_pass_neon;
            #[cfg(feature = "rdm")]
            {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    use crate::box_filter::neon::box_blur_vertical_pass_neon_rdm;
                    _dispatcher_vertical = box_blur_vertical_pass_neon_rdm;
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
    thread_count: u32,
) where
    f32: ToStorage<T>,
{
    let _dispatcher_vertical = T::get_box_vertical_pass::<CN>();
    let unsafe_dst = UnsafeSlice::new(dst);

    let pool = novtb::ThreadPool::new(thread_count as usize);
    pool.parallel_for(|thread_index| {
        let total_width = width as usize * CN;
        let segment_size = total_width / thread_count as usize;
        let start_x = thread_index * segment_size;
        let mut end_x = (thread_index + 1) * segment_size;
        if thread_index == thread_count as usize - 1 {
            end_x = total_width;
        }
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

trait RingBufferHandler<T> {
    fn box_filter_ring_buffer<const CN: usize>(
        src: &[T],
        src_stride: u32,
        dst: &mut [T],
        dst_stride: u32,
        width: u32,
        height: u32,
        parameters: BoxBlurParameters,
        thread_count: u32,
    ) -> Result<(), BlurError>;
    const BOX_RING_IN_SINGLE_THREAD: bool;
}

impl RingBufferHandler<u8> for u8 {
    fn box_filter_ring_buffer<const CN: usize>(
        src: &[u8],
        src_stride: u32,
        dst: &mut [u8],
        dst_stride: u32,
        width: u32,
        height: u32,
        parameters: BoxBlurParameters,
        thread_count: u32,
    ) -> Result<(), BlurError>
    where
        (): VRowSum<u8, u32>,
    {
        ring_box_filter::<u8, u32, CN>(
            src,
            src_stride,
            dst,
            dst_stride,
            width,
            height,
            parameters,
            thread_count,
        )
    }
    const BOX_RING_IN_SINGLE_THREAD: bool = false;
}

impl RingBufferHandler<u16> for u16 {
    fn box_filter_ring_buffer<const CN: usize>(
        src: &[u16],
        src_stride: u32,
        dst: &mut [u16],
        dst_stride: u32,
        width: u32,
        height: u32,
        parameters: BoxBlurParameters,
        thread_count: u32,
    ) -> Result<(), BlurError>
    where
        (): VRowSum<u16, u32>,
    {
        ring_box_filter::<u16, u32, CN>(
            src,
            src_stride,
            dst,
            dst_stride,
            width,
            height,
            parameters,
            thread_count,
        )
    }
    #[cfg(target_arch = "aarch64")]
    const BOX_RING_IN_SINGLE_THREAD: bool = false;
    #[cfg(not(target_arch = "aarch64"))]
    const BOX_RING_IN_SINGLE_THREAD: bool = true;
}

impl RingBufferHandler<f32> for f32 {
    fn box_filter_ring_buffer<const CN: usize>(
        src: &[f32],
        src_stride: u32,
        dst: &mut [f32],
        dst_stride: u32,
        width: u32,
        height: u32,
        parameters: BoxBlurParameters,
        thread_count: u32,
    ) -> Result<(), BlurError>
    where
        (): VRowSum<f32, f32>,
    {
        ring_box_filter::<f32, f32, CN>(
            src,
            src_stride,
            dst,
            dst_stride,
            width,
            height,
            parameters,
            thread_count,
        )
    }
    const BOX_RING_IN_SINGLE_THREAD: bool = true;
}

#[cfg(feature = "nightly_f16")]
impl RingBufferHandler<f16> for f16 {
    fn box_filter_ring_buffer<const CN: usize>(
        src: &[f16],
        src_stride: u32,
        dst: &mut [f16],
        dst_stride: u32,
        width: u32,
        height: u32,
        parameters: BoxBlurParameters,
        thread_count: u32,
    ) -> Result<(), BlurError>
    where
        (): VRowSum<f16, f32>,
    {
        ring_box_filter::<f16, f32, CN>(
            src,
            src_stride,
            dst,
            dst_stride,
            width,
            height,
            parameters,
            thread_count,
        )
    }
    const BOX_RING_IN_SINGLE_THREAD: bool = true;
}

trait VRowSum<T, J> {
    #[allow(clippy::type_complexity)]
    fn ring_vertical_row_summ(
    ) -> fn(src: &[&[T]; 2], dst: &mut [T], working_row: &mut [J], radius: u32);
}

impl VRowSum<u8, u32> for () {
    fn ring_vertical_row_summ() -> fn(&[&[u8]; 2], &mut [u8], &mut [u32], u32) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "rdm")]
            if std::arch::is_aarch64_feature_detected!("rdm") {
                use crate::box_filter::neon::neon_ring_vertical_row_summ_rdm;
                return neon_ring_vertical_row_summ_rdm;
            }
            use crate::box_filter::neon::neon_ring_vertical_row_summ;
            neon_ring_vertical_row_summ
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::box_filter::avx::avx_ring_vertical_row_summ;
                return avx_ring_vertical_row_summ;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::box_filter::sse::sse_ring_vertical_row_summ;
                return sse_ring_vertical_row_summ;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        ring_vertical_row_summ
    }
}

impl VRowSum<u16, u32> for () {
    fn ring_vertical_row_summ() -> fn(&[&[u16]; 2], &mut [u16], &mut [u32], u32) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::box_filter::neon::neon_ring_vertical_row_summ16;
            neon_ring_vertical_row_summ16
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::box_filter::sse::sse_ring_vertical_row_summ16;
                return sse_ring_vertical_row_summ16;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        ring_vertical_row_summ
    }
}

impl VRowSum<f32, f32> for () {
    fn ring_vertical_row_summ() -> fn(&[&[f32]; 2], &mut [f32], &mut [f32], u32) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::box_filter::neon::neon_ring_vertical_row_summ_f32;
            neon_ring_vertical_row_summ_f32
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::box_filter::sse::sse_ring_vertical_row_summ_f32;
                return sse_ring_vertical_row_summ_f32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        ring_vertical_row_summ
    }
}

#[cfg(feature = "nightly_f16")]
impl VRowSum<f16, f32> for () {
    fn ring_vertical_row_summ() -> fn(&[&[f16]; 2], &mut [f16], &mut [f32], u32) {
        ring_vertical_row_summ
    }
}

#[allow(unused)]
fn ring_vertical_row_summ<
    T: Sync + Send + Copy + PrimitiveCast<J>,
    J: Default
        + Sync
        + Send
        + Copy
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + PrimitiveCast<f32>,
>(
    src: &[&[T]; 2],
    dst: &mut [T],
    working_row: &mut [J],
    radius: u32,
) where
    f32: ToStorage<T>,
{
    let next_row = src[1];
    let previous_row = src[0];
    let weight = 1. / (radius as f32 * 2. + 1.);
    for (((src_next, src_previous), buffer), dst) in next_row
        .iter()
        .zip(previous_row.iter())
        .zip(working_row.iter_mut())
        .zip(dst.iter_mut())
    {
        let mut weight0 = *buffer;

        *dst = (weight0.cast_() * weight).to_();

        weight0 += src_next.cast_();
        weight0 -= src_previous.cast_();

        *buffer = weight0;
    }
}

fn ring_box_filter<
    T: Default
        + Sync
        + Send
        + Debug
        + Copy
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + PrimitiveCast<u32>
        + PrimitiveCast<u64>
        + PrimitiveCast<f32>
        + PrimitiveCast<f64>
        + BoxBlurHorizontalPass<T>
        + BoxBlurVerticalPass<T>
        + PrimitiveCast<J>,
    J: Default
        + Sync
        + Send
        + Copy
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + PrimitiveCast<f32>,
    const CN: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    parameters: BoxBlurParameters,
    thread_count: u32,
) -> Result<(), BlurError>
where
    f32: ToStorage<T>,
    (): VRowSum<T, J>,
{
    let y_kernel_size = parameters.y_axis_kernel as usize;
    let x_radius = parameters.x_radius();
    let y_radius = parameters.y_radius();
    let working_stride = width as usize * CN;

    let horizontal_handler = T::get_horizontal_pass::<CN>();
    let ring_vsum = <() as VRowSum<T, J>>::ring_vertical_row_summ();

    if thread_count > 1 {
        let tile_size = height as usize / thread_count as usize;
        let pool = novtb::ThreadPool::new(thread_count as usize);
        dst.tb_par_chunks_mut(dst_stride as usize * tile_size)
            .for_each_enumerated(&pool, |cy, dst_rows| {
                let source_y = cy * tile_size;

                let mut working_row = vec![J::default(); working_stride];
                let ring_size = y_kernel_size + 1;
                let mut buffer = vec![T::default(); working_stride * ring_size];

                let half_kernel = y_kernel_size / 2;

                if source_y == 0 {
                    let dst0 = UnsafeSlice::new(&mut buffer[..working_stride]);
                    let src0 =
                        &src[source_y * src_stride as usize..(source_y + 1) * src_stride as usize];

                    horizontal_handler(
                        &src0[..width as usize * CN],
                        src_stride,
                        &dst0,
                        working_stride as u32,
                        width,
                        x_radius,
                        0,
                        1,
                    );

                    let (src_row, rest) = buffer.split_at_mut(working_stride);
                    for dst in rest.chunks_exact_mut(working_stride).take(half_kernel) {
                        for (dst, src) in dst.iter_mut().zip(src_row.iter()) {
                            *dst = *src;
                        }
                    }
                } else {
                    for src_y in 0..=half_kernel {
                        let s_y = (src_y as i64 + source_y as i64 - half_kernel as i64 - 1)
                            .clamp(0, height as i64 - 1) as usize;

                        let dst0 = UnsafeSlice::new(
                            &mut buffer[src_y * working_stride..(src_y + 1) * working_stride],
                        );
                        let src0 = &src[s_y * src_stride as usize..(s_y + 1) * src_stride as usize];

                        horizontal_handler(
                            &src0[..width as usize * CN],
                            src_stride,
                            &dst0,
                            working_stride as u32,
                            width,
                            x_radius,
                            0,
                            1,
                        );
                    }
                }

                let mut start_ky = y_kernel_size / 2;

                start_ky %= ring_size;

                let mut has_warmed_up = false;

                let rows_count = dst_rows.len() / dst_stride as usize;

                for (y, dy) in (source_y..source_y + rows_count + half_kernel + 1)
                    .zip(0..rows_count + half_kernel + 1)
                {
                    let new_y = y.min(height as usize - 1);

                    let src0 = &src[new_y * src_stride as usize..(new_y + 1) * src_stride as usize];

                    let dst0 = UnsafeSlice::new(
                        &mut buffer[start_ky * working_stride..(start_ky + 1) * working_stride],
                    );

                    horizontal_handler(
                        &src0[..width as usize * CN],
                        src_stride,
                        &dst0,
                        working_stride as u32,
                        width,
                        x_radius,
                        0,
                        1,
                    );

                    if dy > half_kernel {
                        if !has_warmed_up {
                            for row in buffer.chunks_exact(working_stride).take(ring_size - 1) {
                                for (dst, src) in working_row.iter_mut().zip(row.iter()) {
                                    *dst += src.cast_();
                                }
                            }
                            has_warmed_up = true;
                        }

                        let ky0 = (start_ky + 1) % ring_size;
                        let ky1 = (start_ky) % ring_size;

                        let brow0 = &buffer[ky0 * working_stride..(ky0 + 1) * working_stride];
                        let brow1 = &buffer[ky1 * working_stride..(ky1 + 1) * working_stride];

                        let capture = [brow0, brow1];

                        let dy = dy - half_kernel - 1;

                        let dst0 =
                            &mut dst_rows[dy * dst_stride as usize..(dy + 1) * dst_stride as usize];

                        ring_vsum(
                            &capture,
                            &mut dst0[..width as usize * CN],
                            &mut working_row,
                            y_radius,
                        );
                    }

                    start_ky += 1;
                    start_ky %= ring_size;
                }
            });
    } else {
        let mut working_row = vec![J::default(); working_stride];
        let ring_size = y_kernel_size + 1;
        let mut buffer = vec![T::default(); working_stride * ring_size];

        let dst0 = UnsafeSlice::new(&mut buffer[..working_stride]);

        horizontal_handler(
            &src[..width as usize * CN],
            src_stride,
            &dst0,
            working_stride as u32,
            width,
            x_radius,
            0,
            1,
        );

        let half_kernel = y_kernel_size / 2;

        let (src_row, rest) = buffer.split_at_mut(working_stride);
        for dst in rest.chunks_exact_mut(working_stride).take(half_kernel) {
            for (dst, src) in dst.iter_mut().zip(src_row.iter()) {
                *dst = *src;
            }
        }

        let mut start_ky = y_kernel_size / 2 + 1;

        start_ky %= ring_size;

        let mut has_warmed_up = false;

        for y in 1..height as usize + half_kernel + 1 {
            let new_y = y.min(height as usize - 1);

            let src0 = &src[new_y * src_stride as usize..(new_y + 1) * src_stride as usize];

            let dst0 = UnsafeSlice::new(
                &mut buffer[start_ky * working_stride..(start_ky + 1) * working_stride],
            );

            horizontal_handler(
                &src0[..width as usize * CN],
                src_stride,
                &dst0,
                working_stride as u32,
                width,
                x_radius,
                0,
                1,
            );

            if y > half_kernel {
                if !has_warmed_up {
                    for row in buffer.chunks_exact(working_stride).take(ring_size - 1) {
                        for (dst, src) in working_row.iter_mut().zip(row.iter()) {
                            *dst += src.cast_();
                        }
                    }
                    has_warmed_up = true;
                }

                let ky0 = (start_ky + 1) % ring_size;
                let ky1 = (start_ky) % ring_size;

                let brow0 = &buffer[ky0 * working_stride..(ky0 + 1) * working_stride];
                let brow1 = &buffer[ky1 * working_stride..(ky1 + 1) * working_stride];

                let capture = [brow0, brow1];

                let dy = y - half_kernel - 1;

                let dst0 = &mut dst[dy * dst_stride as usize..(dy + 1) * dst_stride as usize];

                ring_vsum(
                    &capture,
                    &mut dst0[..width as usize * CN],
                    &mut working_row,
                    y_radius,
                );
            }

            start_ky += 1;
            start_ky %= ring_size;
        }
    }

    Ok(())
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
        + BoxBlurVerticalPass<T>
        + RingBufferHandler<T>,
    const CN: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    parameters: BoxBlurParameters,
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
    // Ring buffer is less effective in Single Threaded mode.
    if parameters.y_radius() < 55 && (thread_count > 1 || T::BOX_RING_IN_SINGLE_THREAD) {
        return T::box_filter_ring_buffer::<CN>(
            src,
            src_stride,
            dst,
            dst_stride,
            width,
            height,
            parameters,
            thread_count,
        );
    }
    let mut transient: Vec<T> = vec![T::default(); dst_stride as usize * height as usize];
    box_blur_horizontal_pass::<T, CN>(
        src,
        src_stride,
        &mut transient,
        dst_stride,
        width,
        height,
        parameters.x_radius(),
        thread_count,
    );

    box_blur_vertical_pass::<T, CN>(
        &transient,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        parameters.y_radius(),
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
/// * `parameters` - see [BoxBlurParameters] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn box_blur(
    image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    parameters: BoxBlurParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    parameters.validate()?;
    if parameters.x_axis_kernel == 1 && parameters.y_axis_kernel == 1 {
        return image.copy_to_mut(dst_image);
    }
    let width = image.width;
    let height = image.height;
    let thread_count = threading_policy.thread_count(width, height) as u32;
    let _dispatcher = match image.channels {
        FastBlurChannels::Plane => box_blur_impl::<u8, 1>,
        FastBlurChannels::Channels3 => box_blur_impl::<u8, 3>,
        FastBlurChannels::Channels4 => box_blur_impl::<u8, 4>,
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
        parameters,
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
/// * `parameters` - see [BoxBlurParameters] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn box_blur_u16(
    image: &BlurImage<u16>,
    dst_image: &mut BlurImageMut<u16>,
    parameters: BoxBlurParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    if parameters.x_axis_kernel == 1 && parameters.y_axis_kernel == 1 {
        return image.copy_to_mut(dst_image);
    }
    let width = image.width;
    let height = image.height;
    let thread_count = threading_policy.thread_count(width, height) as u32;
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
        parameters,
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
/// * `parameters` - see [BoxBlurParameters] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn box_blur_f32(
    image: &BlurImage<f32>,
    dst_image: &mut BlurImageMut<f32>,
    parameters: BoxBlurParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    parameters.validate()?;
    if parameters.x_axis_kernel == 1 && parameters.y_axis_kernel == 1 {
        return image.copy_to_mut(dst_image);
    }
    let width = image.width;
    let height = image.height;
    let thread_count = threading_policy.thread_count(width, height) as u32;
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
        parameters,
        thread_count,
    )
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
        + BoxBlurVerticalPass<T>
        + RingBufferHandler<T>,
    const CN: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    parameters: CLTParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    f32: ToStorage<T>,
{
    parameters.validate()?;
    let thread_count = threading_policy.thread_count(width, height) as u32;
    let mut transient: Vec<T> =
        vec![T::from_u32(0).unwrap_or_default(); width as usize * height as usize * CN];
    let boxes_horizontal = create_box_gauss(parameters.x_sigma, 2);
    let boxes_vertical = create_box_gauss(parameters.y_sigma, 2);
    box_blur_impl::<T, CN>(
        src,
        src_stride,
        &mut transient,
        width * CN as u32,
        width,
        height,
        BoxBlurParameters {
            x_axis_kernel: boxes_horizontal[0] * 2 + 1,
            y_axis_kernel: boxes_vertical[0] * 2 + 1,
        },
        thread_count,
    )?;
    box_blur_impl::<T, CN>(
        &transient,
        width * CN as u32,
        dst,
        dst_stride,
        width,
        height,
        BoxBlurParameters {
            x_axis_kernel: boxes_horizontal[1] * 2 + 1,
            y_axis_kernel: boxes_vertical[1] * 2 + 1,
        },
        thread_count,
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
/// * `parameters` - See [CLTParameters] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn tent_blur(
    image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    parameters: CLTParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    parameters.validate()?;
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
        parameters,
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
/// * `parameters` - See [CLTParameters] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn tent_blur_u16(
    image: &BlurImage<u16>,
    dst_image: &mut BlurImageMut<u16>,
    parameters: CLTParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    parameters.validate()?;
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
        parameters,
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
/// * `parameters` - See [CLTParameters] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn tent_blur_f32(
    image: &BlurImage<f32>,
    dst_image: &mut BlurImageMut<f32>,
    parameters: CLTParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    parameters.validate()?;
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
        parameters,
        threading_policy,
    )
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
        + BoxBlurVerticalPass<T>
        + RingBufferHandler<T>,
    const CN: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    parameters: CLTParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError>
where
    f32: ToStorage<T>,
{
    parameters.validate()?;
    let thread_count = threading_policy.thread_count(width, height) as u32;
    let mut transient: Vec<T> = vec![T::default(); dst_stride as usize * height as usize];
    let boxes_horizontal = create_box_gauss(parameters.x_sigma, 3);
    let boxes_vertical = create_box_gauss(parameters.y_sigma, 3);
    box_blur_impl::<T, CN>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        BoxBlurParameters {
            x_axis_kernel: boxes_horizontal[0] * 2 + 1,
            y_axis_kernel: boxes_vertical[0] * 2 + 1,
        },
        thread_count,
    )?;
    box_blur_impl::<T, CN>(
        dst,
        dst_stride,
        &mut transient,
        dst_stride,
        width,
        height,
        BoxBlurParameters {
            x_axis_kernel: boxes_horizontal[1] * 2 + 1,
            y_axis_kernel: boxes_vertical[1] * 2 + 1,
        },
        thread_count,
    )?;
    box_blur_impl::<T, CN>(
        &transient,
        dst_stride,
        dst,
        dst_stride,
        width,
        height,
        BoxBlurParameters {
            x_axis_kernel: boxes_horizontal[2] * 2 + 1,
            y_axis_kernel: boxes_vertical[2] * 2 + 1,
        },
        thread_count,
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
/// * `parameters` - See [CLTParameters] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_box_blur(
    image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    parameters: CLTParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    parameters.validate()?;
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
        parameters,
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
/// * `parameters` - See [CLTParameters] for more info.
/// * `threading_policy` - Threading policy, see [ThreadingPolicy] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_box_blur_u16(
    image: &BlurImage<u16>,
    dst_image: &mut BlurImageMut<u16>,
    parameters: CLTParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    parameters.validate()?;
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
        parameters,
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
/// * `parameters` - See [CLTParameters] for more info.
/// * `threading_policy` - Threading policy, see [ThreadingPolicy] for more info.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn gaussian_box_blur_f32(
    image: &BlurImage<f32>,
    dst_image: &mut BlurImageMut<f32>,
    parameters: CLTParameters,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    dst_image.check_layout(Some(image))?;
    image.size_matches_mut(dst_image)?;
    parameters.validate()?;
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
        parameters,
        threading_policy,
    )
}
