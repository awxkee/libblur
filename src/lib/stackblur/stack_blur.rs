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

use crate::stackblur::{HorizontalStackBlurPass, StackBlurWorkingPass, VerticalStackBlurPass};
use crate::unsafe_slice::UnsafeSlice;
use crate::{FastBlurChannels, ThreadingPolicy};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::stackblur::neon::{HorizontalNeonStackBlurPass, VerticalNeonStackBlurPass};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::stackblur::sse::{HorizontalSseStackBlurPass, VerticalSseStackBlurPass};

fn stack_blur_worker_horizontal(
    slice: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
    match channels {
        FastBlurChannels::Plane => {
            let executor = Box::new(HorizontalStackBlurPass::<u8, i32, 1>::default());
            executor.pass(slice, stride, width, height, radius, thread, thread_count);
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i64, i64, 1>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 1>
            // };
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Horizontal,
            //     thread,
            //     thread_count,
            // );
        }
        FastBlurChannels::Channels3 => {
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i32, i64, 3>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 3>
            // };

            // if radius < BASE_RADIUS_I64_CUTOFF {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i32::<3>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse::<3, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse::<3, false>;
            //             }
            //         }
            //     }
            //     #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            //     {
            //         _dispatcher = stack_blur_pass_wasm_i32::<3>;
            //     }
            // } else {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i64::<3>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse_i64::<3, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse_i64::<3, false>;
            //             }
            //         }
            //     }
            // }
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Horizontal,
            //     thread,
            //     thread_count,
            // );
            let mut _executor: Box<dyn StackBlurWorkingPass<u8, i32, 3>> = Box::new(HorizontalStackBlurPass::<u8, i32, 3>::default());
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _executor = Box::new(HorizontalNeonStackBlurPass::<u8, i32, 3>::default());
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                if _is_sse_available {
                    _executor = Box::new(HorizontalSseStackBlurPass::<u8, i32, 3>::default());
                }
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels4 => {
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i64, i64, 4>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 4>
            // };
            // if radius < BASE_RADIUS_I64_CUTOFF {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i32::<4>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse::<4, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse::<4, false>;
            //             }
            //         }
            //     }
            //     #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            //     {
            //         _dispatcher = stack_blur_pass_wasm_i32::<4>;
            //     }
            // } else {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i64::<4>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse_i64::<4, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse_i64::<4, false>;
            //             }
            //         }
            //     }
            // }
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Horizontal,
            //     thread,
            //     thread_count,
            // );
            let mut _executor: Box<dyn StackBlurWorkingPass<u8, i32, 4>> = Box::new(HorizontalStackBlurPass::<u8, i32, 4>::default());
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _executor = Box::new(HorizontalNeonStackBlurPass::<u8, i32, 4>::default());
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                if _is_sse_available {
                    _executor = Box::new(HorizontalSseStackBlurPass::<u8, i32, 4>::default());
                }
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stack_blur_worker_vertical(
    slice: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
    match channels {
        FastBlurChannels::Plane => {
            let mut _executor: Box<dyn StackBlurWorkingPass<u8, i32, 1>> = Box::new(VerticalStackBlurPass::<u8, i32, 1>::default());
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _executor = Box::new(VerticalNeonStackBlurPass::<u8, i32, 1>::default());
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                if _is_sse_available {
                    _executor = Box::new(VerticalSseStackBlurPass::<u8, i32, 1>::default());
                }
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels3 => {
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i64, i64, 3>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 3>
            // };
            // if radius < BASE_RADIUS_I64_CUTOFF {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i32::<3>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse::<3, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse::<3, false>;
            //             }
            //         }
            //     }
            //     #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            //     {
            //         _dispatcher = stack_blur_pass_wasm_i32::<3>;
            //     }
            // } else {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i64::<3>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse_i64::<3, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse_i64::<3, false>;
            //             }
            //         }
            //     }
            // }
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Vertical,
            //     thread,
            //     thread_count,
            // );
            let mut _executor: Box<dyn StackBlurWorkingPass<u8, i32, 3>> = Box::new(VerticalStackBlurPass::<u8, i32, 3>::default());
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _executor = Box::new(VerticalNeonStackBlurPass::<u8, i32, 3>::default());
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                if _is_sse_available {
                    _executor = Box::new(VerticalSseStackBlurPass::<u8, i32, 3>::default());
                }
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels4 => {
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i64, i64, 4>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 4>
            // };
            // if radius < BASE_RADIUS_I64_CUTOFF {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i32::<4>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse::<4, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse::<4, false>;
            //             }
            //         }
            //     }
            //     #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            //     {
            //         _dispatcher = stack_blur_pass_wasm_i32::<3>;
            //     }
            // } else {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i64::<4>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse_i64::<4, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse_i64::<4, false>;
            //             }
            //         }
            //     }
            // }
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Vertical,
            //     thread,
            //     thread_count,
            // );
            let mut _executor: Box<dyn StackBlurWorkingPass<u8, i32, 4>> = Box::new(VerticalStackBlurPass::<u8, i32, 4>::default());
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _executor = Box::new(VerticalNeonStackBlurPass::<u8, i32, 4>::default());
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
                if _is_sse_available {
                    _executor = Box::new(VerticalSseStackBlurPass::<u8, i32, 4>::default());
                }
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
    }
}

/// Fastest available blur option
///
/// Fast gaussian approximation using stack blur.
///
/// # Arguments
/// * `in_place` - mutable buffer contains image data that will be used as a source and destination
/// * `stride` - Bytes per lane, default is width * channels_count if not aligned
/// * `width` - image width
/// * `height` - image height
/// * `radius` - radius almost is not limited, minimum is one
/// * `channels` - Count of channels of the image, only 3 and 4 is supported, alpha position, and channels order does not matter
/// * `threading_policy` - Threads usage policy
///
/// # Complexity
/// O(1) complexity.
pub fn stack_blur(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    let radius = radius.max(1);
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
