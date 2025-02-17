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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::stackblur::neon::{
    HorizontalNeonStackBlurPassFloat32, VerticalNeonStackBlurPassFloat32,
};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::stackblur::sse::{HorizontalSseStackBlurPassFloat32, VerticalSseStackBlurPassFloat32};
use crate::stackblur::*;
use crate::unsafe_slice::UnsafeSlice;
use crate::util::check_slice_size;
use crate::{BlurError, FastBlurChannels, ThreadingPolicy};

fn stack_blur_worker_horizontal(
    slice: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    fn pass<const N: usize>(
        slice: &UnsafeSlice<f32>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        thread_count: usize,
    ) {
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            any(target_arch = "x86_64", target_arch = "x86"),
        )))]
        fn select_blur_pass<const N: usize>() -> impl StackBlurWorkingPass<f32, N> {
            HorizontalStackBlurPass::<f32, f32, f32, N>::default()
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        fn select_blur_pass<const N: usize>() -> impl StackBlurWorkingPass<f32, N> {
            HorizontalNeonStackBlurPassFloat32::<f32, f32, N>::default()
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        fn select_blur_pass<const N: usize>() -> Box<dyn StackBlurWorkingPass<f32, N>> {
            #[cfg(feature = "sse")]
            if std::arch::is_x86_feature_detected!("sse4.1") {
                Box::new(HorizontalSseStackBlurPassFloat32::<f32, f32, N>::default())
            } else {
                Box::new(HorizontalStackBlurPass::<f32, f32, f32, N>::default())
            }
            #[cfg(not(feature = "sse"))]
            Box::new(HorizontalStackBlurPass::<f32, f32, f32, N>::default())
        }
        let executor = select_blur_pass::<N>();
        executor.pass(slice, stride, width, height, radius, thread, thread_count);
    }
    match channels {
        FastBlurChannels::Plane => {
            pass::<1>(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels3 => {
            pass::<3>(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels4 => {
            pass::<4>(slice, stride, width, height, radius, thread, thread_count);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stack_blur_worker_vertical(
    slice: &UnsafeSlice<f32>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    fn pass<const N: usize>(
        slice: &UnsafeSlice<f32>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        thread_count: usize,
    ) {
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            any(target_arch = "x86_64", target_arch = "x86"),
        )))]
        fn select_blur_pass<const N: usize>() -> impl StackBlurWorkingPass<f32, N> {
            VerticalStackBlurPass::<f32, f32, f32, N>::default()
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        fn select_blur_pass<const N: usize>() -> impl StackBlurWorkingPass<f32, N> {
            VerticalNeonStackBlurPassFloat32::<f32, f32, N>::default()
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        fn select_blur_pass<const N: usize>() -> Box<dyn StackBlurWorkingPass<f32, N>> {
            #[cfg(feature = "sse")]
            if std::arch::is_x86_feature_detected!("sse4.1") {
                Box::new(VerticalSseStackBlurPassFloat32::<f32, f32, N>::default())
            } else {
                Box::new(VerticalStackBlurPass::<f32, f32, f32, N>::default())
            }
            #[cfg(not(feature = "sse"))]
            Box::new(VerticalStackBlurPass::<f32, f32, f32, N>::default())
        }
        let executor = select_blur_pass::<N>();
        executor.pass(slice, stride, width, height, radius, thread, thread_count);
    }
    match channels {
        FastBlurChannels::Plane => {
            pass::<1>(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels3 => {
            pass::<3>(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels4 => {
            pass::<4>(slice, stride, width, height, radius, thread, thread_count);
        }
    }
}

/// Fastest available blur option in f32, values may be denormalized, or normalized
///
/// Fast gaussian approximation using stack blur.
///
/// # Arguments
/// * `in_place` - mutable buffer contains image data that will be used as a source and destination.
/// * `stride` - Elements per row, lane length.
/// * `width` - image width
/// * `height` - image height
/// * `radius` - radius almost is not limited, minimum is one
/// * `channels` - Count of channels of the image
/// * `threading_policy` - Threads usage policy
///
/// # Complexity
/// O(1) complexity.
pub fn stack_blur_f32(
    in_place: &mut [f32],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    check_slice_size(
        in_place,
        stride as usize,
        width as usize,
        height as usize,
        channels.get_channels(),
    )?;
    let radius = radius.max(1);
    let thread_count = threading_policy.thread_count(width, height) as u32;
    if thread_count == 1 {
        let slice = UnsafeSlice::new(in_place);
        stack_blur_worker_horizontal(&slice, stride, width, height, radius, channels, 0, 1);
        stack_blur_worker_vertical(&slice, stride, width, height, radius, channels, 0, 1);
        return Ok(());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();
    pool.scope(|scope| {
        let slice = UnsafeSlice::new(in_place);
        for i in 0..thread_count {
            scope.spawn(move |_| {
                stack_blur_worker_horizontal(
                    &slice,
                    stride,
                    width,
                    height,
                    radius,
                    channels,
                    i as usize,
                    thread_count as usize,
                );
            });
        }
    });
    pool.scope(|scope| {
        let slice = UnsafeSlice::new(in_place);
        for i in 0..thread_count {
            scope.spawn(move |_| {
                stack_blur_worker_vertical(
                    &slice,
                    stride,
                    width,
                    height,
                    radius,
                    channels,
                    i as usize,
                    thread_count as usize,
                );
            });
        }
    });
    Ok(())
}
