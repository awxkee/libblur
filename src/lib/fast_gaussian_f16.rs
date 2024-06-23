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

pub(crate) mod fast_gaussian_f16 {
    use crate::fast_gaussian_f32::{
        fast_gaussian_horizontal_pass_f32, fast_gaussian_vertical_pass_f32,
    };
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    pub(crate) fn fast_gaussian_impl_f16(
        bytes: &mut [u16],
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        channels: FastBlurChannels,
    ) {
        let _dispatcher_vertical: fn(&UnsafeSlice<half::f16>, u32, u32, u32, u32, u32, u32) =
            match channels {
                FastBlurChannels::Channels3 => fast_gaussian_vertical_pass_f32::<half::f16, 3>,
                FastBlurChannels::Channels4 => fast_gaussian_vertical_pass_f32::<half::f16, 4>,
            };
        let _dispatcher_horizontal: fn(&UnsafeSlice<half::f16>, u32, u32, u32, u32, u32, u32) =
            match channels {
                FastBlurChannels::Channels3 => fast_gaussian_horizontal_pass_f32::<half::f16, 3>,
                FastBlurChannels::Channels4 => fast_gaussian_horizontal_pass_f32::<half::f16, 4>,
            };
        let unsafe_image: UnsafeSlice<half::f16> =
            UnsafeSlice::new(unsafe { std::mem::transmute(bytes) });
        let thread_count = std::cmp::max(std::cmp::min(width * height / (256 * 256), 12), 1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count as usize)
            .build()
            .unwrap();
        pool.scope(|scope| {
            let segment_size = width / thread_count;

            for i in 0..thread_count {
                let start_x = i * segment_size;
                let mut end_x = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_x = width;
                }
                scope.spawn(move |_| {
                    _dispatcher_vertical(
                        &unsafe_image,
                        stride,
                        width,
                        height,
                        radius,
                        start_x,
                        end_x,
                    );
                });
            }
        });
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
                        &unsafe_image,
                        stride,
                        width,
                        height,
                        radius,
                        start_y,
                        end_y,
                    );
                });
            }
        });
    }
}
