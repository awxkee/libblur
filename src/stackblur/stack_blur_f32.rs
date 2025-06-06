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
    HorizontalNeonStackBlurPassFloat32, VerticalNeonStackBlurPassFloat32,
};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::stackblur::sse::{HorizontalSseStackBlurPassFloat32, VerticalSseStackBlurPassFloat32};
use crate::stackblur::*;
use crate::unsafe_slice::UnsafeSlice;
use crate::{AnisotropicRadius, BlurError, BlurImageMut, FastBlurChannels, ThreadingPolicy};

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
            all(target_arch = "aarch64", feature = "neon"),
            any(target_arch = "x86_64", target_arch = "x86"),
        )))]
        fn select_blur_pass<const N: usize>() -> impl StackBlurWorkingPass<f32, N> {
            HorizontalStackBlurPass::<f32, f32, f32, N>::default()
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
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
            all(target_arch = "aarch64", feature = "neon"),
            any(target_arch = "x86_64", target_arch = "x86"),
        )))]
        fn select_blur_pass<const N: usize>() -> impl StackBlurWorkingPass<f32, N> {
            VerticalStackBlurPass::<f32, f32, f32, N>::default()
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
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
/// * `image` - mutable buffer contains image data that will be used as a source and destination.
/// * `radius` - radius almost is not limited, minimum is one.
/// * `threading_policy` - Threads usage policy
///
/// # Complexity
/// O(1) complexity.
pub fn stack_blur_f32(
    image: &mut BlurImageMut<f32>,
    radius: AnisotropicRadius,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let radius = radius.max(1);
    let thread_count = threading_policy.thread_count(image.width, image.height) as u32;
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    if thread_count == 1 {
        let slice = UnsafeSlice::new(image.data.borrow_mut());
        stack_blur_worker_horizontal(&slice, stride, width, height, radius.x_axis, channels, 0, 1);
        stack_blur_worker_vertical(&slice, stride, width, height, radius.y_axis, channels, 0, 1);
        return Ok(());
    }
    let pool = novtb::ThreadPool::new(thread_count as usize);
    let slice = UnsafeSlice::new(image.data.borrow_mut());
    pool.parallel_for(|thread_index| {
        stack_blur_worker_horizontal(
            &slice,
            stride,
            width,
            height,
            radius.x_axis,
            channels,
            thread_index,
            thread_count as usize,
        );
    });
    pool.parallel_for(|thread_index| {
        stack_blur_worker_vertical(
            &slice,
            stride,
            width,
            height,
            radius.y_axis,
            channels,
            thread_index,
            thread_count as usize,
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_blur_f32_q_k5() {
        let width: usize = 148;
        let height: usize = 148;
        let mut dst = vec![0.32423f32; width * height * 3];
        let mut dst_image = BlurImageMut::borrow(
            &mut dst,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        stack_blur_f32(
            &mut dst_image,
            AnisotropicRadius::new(5),
            ThreadingPolicy::Single,
        )
        .unwrap();
        for (i, &cn) in dst.iter().enumerate() {
            let diff = (cn - 0.32423f32).abs();
            assert!(
                diff <= 1e-4,
                "Diff expected to be less than 1e-4 but it was {diff} at {i}"
            );
        }
    }
}
