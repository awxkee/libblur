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

use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::gaussian::gaussian_horizontal::gaussian_blur_horizontal_pass_impl_clip_edge;
use crate::gaussian::gaussian_vertical::gaussian_blur_vertical_pass_clip_edge_impl;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::FromPrimitive;
use rayon::ThreadPool;

pub(crate) fn gaussian_blur_vertical_pass_edge_clip_dispatch<
    T: FromPrimitive + Default + Into<f32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter>,
    thread_pool: &ThreadPool,
    thread_count: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let unsafe_dst = UnsafeSlice::new(dst);
    thread_pool.scope(|scope| {
        let segment_size = height / thread_count;

        for i in 0..thread_count {
            let start_y = i * segment_size;
            let mut end_y = (i + 1) * segment_size;
            if i == thread_count - 1 {
                end_y = height;
            }

            scope.spawn(move |_| {
                gaussian_blur_vertical_pass_clip_edge_impl::<T, CHANNEL_CONFIGURATION>(
                    src,
                    src_stride,
                    &unsafe_dst,
                    dst_stride,
                    width,
                    height,
                    filter,
                    start_y,
                    end_y,
                );
            });
        }
    });
}

pub(crate) fn gaussian_blur_horizontal_pass_edge_clip_dispatch<
    T: FromPrimitive + Default + Into<f32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter>,
    thread_pool: &ThreadPool,
    thread_count: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let unsafe_dst = UnsafeSlice::new(dst);
    thread_pool.scope(|scope| {
        let segment_size = height / thread_count;
        for i in 0..thread_count {
            let start_y = i * segment_size;
            let mut end_y = (i + 1) * segment_size;
            if i == thread_count - 1 {
                end_y = height;
            }

            scope.spawn(move |_| {
                gaussian_blur_horizontal_pass_impl_clip_edge::<T, CHANNEL_CONFIGURATION>(
                    src,
                    src_stride,
                    &unsafe_dst,
                    dst_stride,
                    width,
                    filter,
                    start_y,
                    end_y,
                );
            });
        }
    });
}