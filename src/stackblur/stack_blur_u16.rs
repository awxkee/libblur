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
use crate::stackblur::{HorizontalStackBlurPass, StackBlurWorkingPass, VerticalStackBlurPass};
use crate::unsafe_slice::UnsafeSlice;
use crate::{BlurError, BlurImageMut, FastBlurChannels, ThreadingPolicy};

const LARGE_RADIUS_CUTOFF: u32 = 135;

fn stack_blur_worker_horizontal(
    slice: &UnsafeSlice<u16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    fn pass<const N: usize>(
        slice: &UnsafeSlice<u16>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        thread_count: usize,
    ) {
        if LARGE_RADIUS_CUTOFF > radius {
            let executor = HorizontalStackBlurPass::<u16, i32, f32, N>::default();
            executor.pass(slice, stride, width, height, radius, thread, thread_count);
        } else {
            let executor = HorizontalStackBlurPass::<u16, i64, f64, N>::default();
            executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
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
    slice: &UnsafeSlice<u16>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    fn pass<const N: usize>(
        slice: &UnsafeSlice<u16>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        thread_count: usize,
    ) {
        if LARGE_RADIUS_CUTOFF > radius {
            let executor = VerticalStackBlurPass::<u16, i32, f32, N>::default();
            executor.pass(slice, stride, width, height, radius, thread, thread_count);
        } else {
            let executor = VerticalStackBlurPass::<u16, i64, f64, N>::default();
            executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
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

/// Fastest available blur option
///
/// Fast gaussian approximation using stack blur for u16 image
///
/// # Arguments
/// * `image` - mutable buffer contains image data that will be used as a source and destination.
/// * `channels` - Count of channels of the image
/// * `threading_policy` - Threads usage policy
///
/// # Complexity
/// O(1) complexity.
pub fn stack_blur_u16(
    image: &mut BlurImageMut<u16>,
    radius: u32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    let radius = radius.max(1);
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    #[allow(clippy::manual_clamp)]
    let radius = radius.max(1).min(2000);
    let thread_count = threading_policy.thread_count(width, height) as u32;
    if thread_count == 1 {
        let slice = UnsafeSlice::new(image.data.borrow_mut());
        stack_blur_worker_horizontal(&slice, stride, width, height, radius, channels, 0, 1);
        stack_blur_worker_vertical(&slice, stride, width, height, radius, channels, 0, 1);
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
                    radius,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_blur_u16_q_k5() {
        let width: usize = 148;
        let height: usize = 148;
        let mut dst = vec![43231u16; width * height * 3];
        let mut dst_image = BlurImageMut::borrow(
            &mut dst,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        stack_blur_u16(&mut dst_image, 5, ThreadingPolicy::Single).unwrap();
        for (i, &cn) in dst.iter().enumerate() {
            let diff = (cn as i32 - 43231i32).abs();
            assert!(
                diff <= 20,
                "Diff expected to be less than 20 but it was {diff} at {i}"
            );
        }
    }
}
