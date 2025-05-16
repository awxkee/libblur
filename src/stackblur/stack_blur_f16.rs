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

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use crate::stackblur::neon::{
    HorizontalNeonStackBlurPassFloat16, VerticalNeonStackBlurPassFloat16,
};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::stackblur::sse::{HorizontalSseStackBlurPassFloat16, VerticalSseStackBlurPassFloat16};
use crate::stackblur::{HorizontalStackBlurPass, StackBlurWorkingPass, VerticalStackBlurPass};
use crate::unsafe_slice::UnsafeSlice;
use crate::{AnisotropicRadius, BlurError, BlurImageMut, FastBlurChannels, ThreadingPolicy};
use half::f16;

fn stack_blur_worker_horizontal(
    slice: &UnsafeSlice<f16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
    match channels {
        FastBlurChannels::Plane => {
            let mut _executor: Box<dyn StackBlurWorkingPass<f16, 1>> =
                Box::new(HorizontalStackBlurPass::<f16, f32, f32, 1>::default());
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(feature = "sse")]
                if std::arch::is_x86_feature_detected!("sse4.1")
                    && std::arch::is_x86_feature_detected!("f16c")
                {
                    _executor =
                        Box::new(HorizontalSseStackBlurPassFloat16::<f16, f32, 1>::default());
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                _executor = Box::new(HorizontalNeonStackBlurPassFloat16::<f16, f32, 1>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels3 => {
            let mut _executor: Box<dyn StackBlurWorkingPass<f16, 3>> =
                Box::new(HorizontalStackBlurPass::<f16, f32, f32, 3>::default());
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(feature = "sse")]
                if std::arch::is_x86_feature_detected!("sse4.1")
                    && std::arch::is_x86_feature_detected!("f16c")
                {
                    _executor =
                        Box::new(HorizontalSseStackBlurPassFloat16::<f16, f32, 3>::default());
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                _executor = Box::new(HorizontalNeonStackBlurPassFloat16::<f16, f32, 3>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels4 => {
            let mut _executor: Box<dyn StackBlurWorkingPass<f16, 4>> =
                Box::new(HorizontalStackBlurPass::<f16, f32, f32, 4>::default());
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(feature = "sse")]
                if std::arch::is_x86_feature_detected!("sse4.1")
                    && std::arch::is_x86_feature_detected!("f16c")
                {
                    _executor =
                        Box::new(HorizontalSseStackBlurPassFloat16::<f16, f32, 4>::default());
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                _executor = Box::new(HorizontalNeonStackBlurPassFloat16::<f16, f32, 4>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
    }
}

fn stack_blur_worker_vertical(
    slice: &UnsafeSlice<f16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    match channels {
        FastBlurChannels::Plane => {
            let mut _executor: Box<dyn StackBlurWorkingPass<f16, 1>> =
                Box::new(VerticalStackBlurPass::<f16, f32, f32, 1>::default());
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(feature = "sse")]
                if std::arch::is_x86_feature_detected!("sse4.1")
                    && std::arch::is_x86_feature_detected!("f16c")
                {
                    _executor = Box::new(VerticalSseStackBlurPassFloat16::<f16, f32, 1>::default());
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                _executor = Box::new(VerticalNeonStackBlurPassFloat16::<f16, f32, 1>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels3 => {
            let mut _executor: Box<dyn StackBlurWorkingPass<f16, 3>> =
                Box::new(VerticalStackBlurPass::<f16, f32, f32, 3>::default());
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(feature = "sse")]
                if std::arch::is_x86_feature_detected!("sse4.1")
                    && std::arch::is_x86_feature_detected!("f16c")
                {
                    _executor = Box::new(VerticalSseStackBlurPassFloat16::<f16, f32, 3>::default());
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                _executor = Box::new(VerticalNeonStackBlurPassFloat16::<f16, f32, 3>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels4 => {
            let mut _executor: Box<dyn StackBlurWorkingPass<f16, 4>> =
                Box::new(VerticalStackBlurPass::<f16, f32, f32, 4>::default());
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(feature = "sse")]
                if std::arch::is_x86_feature_detected!("sse4.1")
                    && std::arch::is_x86_feature_detected!("f16c")
                {
                    _executor = Box::new(VerticalSseStackBlurPassFloat16::<f16, f32, 4>::default());
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                _executor = Box::new(VerticalNeonStackBlurPassFloat16::<f16, f32, 4>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
    }
}

/// Fastest available blur option in f16, values may be denormalized, or normalized
///
/// Fast gaussian approximation using stack blur.
///
/// # Arguments
/// * `image` - mutable buffer contains image data that will be used as a source and destination.
/// * `radius` - radius almost is not limited, minimum is one
/// * `threading_policy` - Threads usage policy
///
/// # Complexity
/// O(1) complexity.
pub fn stack_blur_f16(
    image: &mut BlurImageMut<f16>,
    radius: AnisotropicRadius,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let radius = radius.clamp(1, 1449);
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let thread_count = threading_policy.thread_count(width, height) as u32;
    if thread_count == 1 {
        let slice = UnsafeSlice::new(image.data.borrow_mut());
        stack_blur_worker_horizontal(&slice, stride, width, height, radius.x_axis, channels, 0, 1);
        stack_blur_worker_vertical(&slice, stride, width, height, radius.y_axis, channels, 0, 1);
        return Ok(());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();
    pool.scope(|scope| {
        let slice = UnsafeSlice::new(image.data.borrow_mut());
        for i in 0..thread_count {
            scope.spawn(move |_| {
                stack_blur_worker_horizontal(
                    &slice,
                    stride,
                    width,
                    height,
                    radius.x_axis,
                    channels,
                    i as usize,
                    thread_count as usize,
                );
            });
        }
    });
    pool.scope(|scope| {
        let slice = UnsafeSlice::new(image.data.borrow_mut());
        for i in 0..thread_count {
            scope.spawn(move |_| {
                stack_blur_worker_vertical(
                    &slice,
                    stride,
                    width,
                    height,
                    radius.y_axis,
                    channels,
                    i as usize,
                    thread_count as usize,
                );
            });
        }
    });
    Ok(())
}
