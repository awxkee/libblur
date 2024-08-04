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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::stack_blur_pass_neon_f16;
use crate::stack_blur::StackBlurPass;
use crate::stack_blur_f32::stack_blur_pass_f;
use crate::unsafe_slice::UnsafeSlice;
use crate::{FastBlurChannels, ThreadingPolicy};
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    all(target_feature = "sse4.1", target_feature = "f16c")
))]
use crate::sse::stack_blur_pass_sse_f16;

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
    match channels {
        FastBlurChannels::Plane => {
            let mut _dispatcher: fn(
                &UnsafeSlice<f16>,
                u32,
                u32,
                u32,
                u32,
                StackBlurPass,
                usize,
                usize,
            ) = stack_blur_pass_f::<f16, f32, 1>;
            _dispatcher(
                &slice,
                stride,
                width,
                height,
                radius,
                StackBlurPass::HORIZONTAL,
                thread,
                thread_count,
            );
        }
        FastBlurChannels::Channels3 => {
            let mut _dispatcher: fn(
                &UnsafeSlice<f16>,
                u32,
                u32,
                u32,
                u32,
                StackBlurPass,
                usize,
                usize,
            ) = stack_blur_pass_f::<f16, f32, 3>;
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = stack_blur_pass_neon_f16::<3>;
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                all(target_feature = "sse4.1", target_feature = "f16c")
            ))]
            {
                _dispatcher = stack_blur_pass_sse_f16::<3>;
            }
            _dispatcher(
                &slice,
                stride,
                width,
                height,
                radius,
                StackBlurPass::HORIZONTAL,
                thread,
                thread_count,
            );
        }
        FastBlurChannels::Channels4 => {
            let mut _dispatcher: fn(
                &UnsafeSlice<f16>,
                u32,
                u32,
                u32,
                u32,
                StackBlurPass,
                usize,
                usize,
            ) = stack_blur_pass_f::<f16, f32, 4>;
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = stack_blur_pass_neon_f16::<4>;
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                all(target_feature = "sse4.1", target_feature = "f16c")
            ))]
            {
                _dispatcher = stack_blur_pass_sse_f16::<4>;
            }
            _dispatcher(
                &slice,
                stride,
                width,
                height,
                radius,
                StackBlurPass::HORIZONTAL,
                thread,
                thread_count,
            );
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
            let mut _dispatcher: fn(
                &UnsafeSlice<f16>,
                u32,
                u32,
                u32,
                u32,
                StackBlurPass,
                usize,
                usize,
            ) = stack_blur_pass_f::<f16, f32, 1>;
            _dispatcher(
                &slice,
                stride,
                width,
                height,
                radius,
                StackBlurPass::VERTICAL,
                thread,
                thread_count,
            );
        }
        FastBlurChannels::Channels3 => {
            let mut _dispatcher: fn(
                &UnsafeSlice<f16>,
                u32,
                u32,
                u32,
                u32,
                StackBlurPass,
                usize,
                usize,
            ) = stack_blur_pass_f::<f16, f32, 3>;
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = stack_blur_pass_neon_f16::<3>;
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                all(target_feature = "sse4.1", target_feature = "f16c")
            ))]
            {
                _dispatcher = stack_blur_pass_sse_f16::<3>;
            }
            _dispatcher(
                &slice,
                stride,
                width,
                height,
                radius,
                StackBlurPass::VERTICAL,
                thread,
                thread_count,
            );
        }
        FastBlurChannels::Channels4 => {
            let mut _dispatcher: fn(
                &UnsafeSlice<f16>,
                u32,
                u32,
                u32,
                u32,
                StackBlurPass,
                usize,
                usize,
            ) = stack_blur_pass_f::<f16, f32, 4>;
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher = stack_blur_pass_neon_f16::<4>;
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                all(target_feature = "sse4.1", target_feature = "f16c")
            ))]
            {
                _dispatcher = stack_blur_pass_sse_f16::<4>;
            }
            _dispatcher(
                &slice,
                stride,
                width,
                height,
                radius,
                StackBlurPass::VERTICAL,
                thread,
                thread_count,
            );
        }
    }
}

/// Fastest available blur option in f16, values may be denormalized, or normalized
///
/// Fast gaussian approximation using stack blur
/// This is a very fast approximation using f32 accumulator size
///
/// # Arguments
/// * `in_place` - mutable buffer contains image data that will be used as a source and destination
/// * `width` - image width
/// * `height` - image height
/// * `radius` - radius almost is not limited for f32 implementation
/// * `channels` - Count of channels of the image, only 3 and 4 is supported, alpha position, and channels order does not matter
/// * `threading_policy` - Threads usage policy
///
/// # Complexity
/// O(1) complexity.
pub fn stack_blur_f16(
    in_place: &mut [f16],
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    let stride = width * channels.get_channels() as u32;
    let radius = std::cmp::max(radius, 2);
    let thread_count = threading_policy.get_threads_count(width, height) as u32;
    if thread_count == 1 {
        let slice = UnsafeSlice::new(in_place);
        stack_blur_worker_horizontal(&slice, stride, width, height, radius, channels, 0, 1);
        stack_blur_worker_vertical(&slice, stride, width, height, radius, channels, 0, 1);
        return;
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
    })
}
