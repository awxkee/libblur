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

use crate::unsafe_slice::UnsafeSlice;
use std::thread;
use crate::channels_configuration::FastBlurChannels;
use crate::median_blur_gen::median_blur_impl;

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn median_blur(
    src: &Vec<u8>,
    src_stride: u32,
    dst: &mut Vec<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    median_channels: FastBlurChannels,
) {
    let unsafe_dst = UnsafeSlice::new(dst);
    thread::scope(|scope| {
        let thread_count = std::cmp::max(std::cmp::min(width * height / (256 * 256), 12), 1);
        let segment_size = height / thread_count;
        let mut handles = vec![];
        for i in 0..thread_count {
            let start_y = i * segment_size;
            let mut end_y = (i + 1) * segment_size;
            if i == thread_count - 1 {
                end_y = height;
            }
            let handle = scope.spawn(move || {
                median_blur_impl(
                    src,
                    src_stride,
                    &unsafe_dst,
                    dst_stride,
                    width,
                    height,
                    radius,
                    median_channels,
                    start_y,
                    end_y,
                );
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
    });
}
