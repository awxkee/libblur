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

mod gaussian_f16_impl {
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    pub(crate) fn gaussian_blur_horizontal_pass_impl_f16(
        src: &[u16],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u16>,
        dst_stride: u32,
        width: u32,
        kernel_size: usize,
        gaussian_channels: FastBlurChannels,
        kernel: &Vec<f32>,
        start_y: u32,
        end_y: u32,
    ) {
        let half_kernel = (kernel_size / 2) as i32;
        let channels_count = match gaussian_channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for y in start_y..end_y {
            let y_src_shift = y as usize * src_stride as usize;
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let mut weights: [f32; 4] = [0f32; 4];
                for r in -half_kernel..=half_kernel {
                    let px =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize
                            * channels_count;
                    let weight = kernel[(r + half_kernel) as usize];
                    weights[0] += half::f16::from_bits(src[y_src_shift + px]).to_f32() * weight;
                    weights[1] += half::f16::from_bits(src[y_src_shift + px + 1]).to_f32() * weight;
                    weights[2] += half::f16::from_bits(src[y_src_shift + px + 2]).to_f32() * weight;
                    match gaussian_channels {
                        FastBlurChannels::Channels3 => {}
                        FastBlurChannels::Channels4 => {
                            weights[3] +=
                                half::f16::from_bits(src[y_src_shift + px + 3]).to_f32() * weight;
                        }
                    }
                }

                let px = x as usize * channels_count;

                unsafe {
                    unsafe_dst.write(y_dst_shift + px, half::f16::from_f32(weights[0]).to_bits());
                    unsafe_dst.write(
                        y_dst_shift + px + 1,
                        half::f16::from_f32(weights[1]).to_bits(),
                    );
                    unsafe_dst.write(
                        y_dst_shift + px + 2,
                        half::f16::from_f32(weights[2]).to_bits(),
                    );
                    match gaussian_channels {
                        FastBlurChannels::Channels3 => {}
                        FastBlurChannels::Channels4 => {
                            unsafe_dst.write(
                                y_dst_shift + px + 3,
                                half::f16::from_f32(weights[3]).to_bits(),
                            );
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn gaussian_blur_vertical_pass_impl_f16(
        src: &[u16],
        src_stride: u32,
        unsafe_dst: &UnsafeSlice<u16>,
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: usize,
        gaussian_channels: FastBlurChannels,
        kernel: &Vec<f32>,
        start_y: u32,
        end_y: u32,
    ) {
        let half_kernel = (kernel_size / 2) as i32;
        let channels_count = match gaussian_channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for y in start_y..end_y {
            let y_dst_shift = y as usize * dst_stride as usize;
            for x in 0..width {
                let px = x as usize * channels_count;
                let mut weights: [f32; 4] = [0f32; 4];
                for r in -half_kernel..=half_kernel {
                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let weight = kernel[(r + half_kernel) as usize];
                    weights[0] += half::f16::from_bits(src[y_src_shift + px]).to_f32() * weight;
                    weights[1] += half::f16::from_bits(src[y_src_shift + px + 1]).to_f32() * weight;
                    weights[2] += half::f16::from_bits(src[y_src_shift + px + 2]).to_f32() * weight;
                    match gaussian_channels {
                        FastBlurChannels::Channels3 => {}
                        FastBlurChannels::Channels4 => {
                            weights[3] +=
                                half::f16::from_bits(src[y_src_shift + px + 3]).to_f32() * weight;
                        }
                    }
                }

                unsafe {
                    unsafe_dst.write(y_dst_shift + px, half::f16::from_f32(weights[0]).to_bits());
                    unsafe_dst.write(
                        y_dst_shift + px + 1,
                        half::f16::from_f32(weights[1]).to_bits(),
                    );
                    unsafe_dst.write(
                        y_dst_shift + px + 2,
                        half::f16::from_f32(weights[2]).to_bits(),
                    );
                    match gaussian_channels {
                        FastBlurChannels::Channels3 => {}
                        FastBlurChannels::Channels4 => {
                            unsafe_dst.write(
                                y_dst_shift + px + 3,
                                half::f16::from_f32(weights[3]).to_bits(),
                            );
                        }
                    }
                }
            }
        }
    }
}

pub(crate) mod gaussian_f16 {
    use rayon::ThreadPool;

    use crate::gaussian_f16::gaussian_f16_impl;
    use crate::gaussian_helper::get_gaussian_kernel_1d;
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    fn gaussian_blur_horizontal_pass_f16(
        src: &[u16],
        src_stride: u32,
        dst: &mut [u16],
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: usize,
        gaussian_channels: FastBlurChannels,
        kernel: &Vec<f32>,
        thread_pool: &ThreadPool,
        thread_count: u32,
    ) {
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
                    gaussian_f16_impl::gaussian_blur_horizontal_pass_impl_f16(
                        src,
                        src_stride,
                        &unsafe_dst,
                        dst_stride,
                        width,
                        kernel_size,
                        gaussian_channels,
                        kernel,
                        start_y,
                        end_y,
                    );
                });
            }
        });
    }

    fn gaussian_blur_vertical_pass_f16(
        src: &[u16],
        src_stride: u32,
        dst: &mut [u16],
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: usize,
        gaussian_channels: FastBlurChannels,
        kernel: &Vec<f32>,
        thread_pool: &ThreadPool,
        thread_count: u32,
    ) {
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
                    gaussian_f16_impl::gaussian_blur_vertical_pass_impl_f16(
                        src,
                        src_stride,
                        &unsafe_dst,
                        dst_stride,
                        width,
                        height,
                        kernel_size,
                        gaussian_channels,
                        kernel,
                        start_y,
                        end_y,
                    );
                });
            }
        });
    }

    pub(crate) fn gaussian_blur_impl_f16(
        src: &[u16],
        src_stride: u32,
        dst: &mut [u16],
        dst_stride: u32,
        width: u32,
        height: u32,
        kernel_size: u32,
        sigma: f32,
        box_channels: FastBlurChannels,
    ) {
        let kernel = get_gaussian_kernel_1d(kernel_size, sigma);
        if kernel_size % 2 == 0 {
            panic!("kernel size must be odd");
        }
        let mut transient: Vec<u16> = Vec::with_capacity(dst_stride as usize * height as usize);
        transient.resize(dst_stride as usize * height as usize, 0);

        let thread_count = std::cmp::max(std::cmp::min(width * height / (256 * 256), 12), 1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count as usize)
            .build()
            .unwrap();

        gaussian_blur_horizontal_pass_f16(
            &src,
            src_stride,
            &mut transient,
            dst_stride,
            width,
            height,
            kernel.len(),
            box_channels,
            &kernel,
            &pool,
            thread_count,
        );
        gaussian_blur_vertical_pass_f16(
            &transient,
            dst_stride,
            dst,
            dst_stride,
            width,
            height,
            kernel.len(),
            box_channels,
            &kernel,
            &pool,
            thread_count,
        );
    }
}
