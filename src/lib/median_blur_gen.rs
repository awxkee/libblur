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

use crate::FastBlurChannels;
use crate::unsafe_slice::UnsafeSlice;

struct MedianHistogram {
    r: [i32; 256],
    g: [i32; 256],
    b: [i32; 256],
    a: [i32; 256],
    n: i32,
}

fn add_rgb_pixels(
    src: &Vec<u8>,
    src_stride: u32,
    y: i64,
    x: i64,
    width: u32,
    height: u32,
    size: u32,
    median_channels: FastBlurChannels,
    histogram: &mut MedianHistogram,
) {
    if x < 0 || x >= width as i64 {
        return;
    }
    let channels_count = match median_channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    } as usize;
    let px = x as usize * channels_count;
    let cap = std::cmp::min(y + size as i64, height as i64 - 1);
    let start = y - size as i64;
    for i in start..=cap {
        if i < 0 {
            continue;
        }
        let y_shift = i as usize * src_stride as usize;
        histogram.r[usize::from(src[y_shift + px])] += 1;
        histogram.g[usize::from(src[y_shift + px + 1])] += 1;
        histogram.b[usize::from(src[y_shift + px + 2])] += 1;
        match median_channels {
            FastBlurChannels::Channels3 => {}
            FastBlurChannels::Channels4 => {
                histogram.a[usize::from(src[y_shift + px + 3])] += 1;
            }
        }
        histogram.n += 1;
    }
}

fn remove_rgb_pixels(
    src: &Vec<u8>,
    src_stride: u32,
    y: i64,
    x: i64,
    width: u32,
    height: u32,
    size: u32,
    median_channels: FastBlurChannels,
    histogram: &mut MedianHistogram,
) {
    if x < 0 || x >= width as i64 {
        return;
    }
    let channels_count = match median_channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    } as usize;
    let px = x as usize * channels_count;
    let cap = std::cmp::min(y + size as i64, height as i64 - 1);
    let start = y - size as i64;
    for i in start..=cap {
        if i < 0 {
            continue;
        }
        let y_shift = i as usize * src_stride as usize;
        histogram.r[usize::from(src[y_shift + px])] -= 1;
        histogram.g[usize::from(src[y_shift + px + 1])] -= 1;
        histogram.b[usize::from(src[y_shift + px + 2])] -= 1;
        match median_channels {
            FastBlurChannels::Channels3 => {}
            FastBlurChannels::Channels4 => {
                histogram.a[usize::from(src[y_shift + px + 3])] -= 1;
            }
        }
        histogram.n -= 1;
    }
}

fn init_histogram(
    src: &Vec<u8>,
    src_stride: u32,
    y: u32,
    width: u32,
    height: u32,
    radius: u32,
    median_channels: FastBlurChannels,
    histogram: &mut MedianHistogram,
) {
    histogram.r = [0; 256];
    histogram.g = [0; 256];
    histogram.b = [0; 256];
    histogram.a = [0; 256];
    histogram.n = 0;

    for j in 0..std::cmp::min(radius, width) {
        add_rgb_pixels(
            src,
            src_stride,
            y as i64,
            j as i64,
            width,
            height,
            radius,
            median_channels,
            histogram,
        );
    }
}

fn median_filter(x: [i32; 256], n: i32) -> i32 {
    let mut n = n / 2;
    let mut i = 0i64;
    while i < 256 && i >= 0 {
        n -= x[i as usize];
        if n > 0 {
            i += 1;
        } else {
            break;
        }
    }
    i as i32
}

pub fn median_blur_impl(
    src: &Vec<u8>,
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    median_channels: FastBlurChannels,
    start_y: u32,
    end_y: u32,
) {
    let channels_count = match median_channels {
        FastBlurChannels::Channels3 => 3,
        FastBlurChannels::Channels4 => 4,
    } as usize;
    for y in start_y..end_y {
        let y_src_offset = y as usize * src_stride as usize;
        let y_dst_offset = y as usize * dst_stride as usize;
        let mut histogram = MedianHistogram {
            r: [0; 256],
            g: [0; 256],
            b: [0; 256],
            a: [0; 256],
            n: 0,
        };
        for x in 0..width {
            let px = x as usize * channels_count;
            if x == 0 {
                init_histogram(
                    src,
                    src_stride,
                    y,
                    width,
                    height,
                    radius,
                    median_channels,
                    &mut histogram,
                );
            } else {
                remove_rgb_pixels(
                    src,
                    src_stride,
                    y as i64,
                    x as i64 - radius as i64,
                    width,
                    height,
                    radius as u32,
                    median_channels,
                    &mut histogram,
                );
                add_rgb_pixels(
                    src,
                    src_stride,
                    y as i64,
                    x as i64 + radius as i64,
                    width,
                    height,
                    radius as u32,
                    median_channels,
                    &mut histogram,
                );
            }

            if histogram.n > 0 {
                unsafe {
                    unsafe_dst.write(
                        y_dst_offset + px,
                        median_filter(histogram.r, histogram.n) as u8,
                    );
                    unsafe_dst.write(
                        y_dst_offset + px + 1,
                        median_filter(histogram.g, histogram.n) as u8,
                    );
                    unsafe_dst.write(
                        y_dst_offset + px + 2,
                        median_filter(histogram.b, histogram.n) as u8,
                    );
                    match median_channels {
                        FastBlurChannels::Channels3 => {}
                        FastBlurChannels::Channels4 => {
                            unsafe_dst.write(
                                y_dst_offset + px + 3,
                                median_filter(histogram.a, histogram.n) as u8,
                            );
                        }
                    };
                }
            } else {
                unsafe {
                    unsafe_dst.write(y_dst_offset + px, src[y_src_offset + px]);
                    unsafe_dst.write(y_dst_offset + px + 1, src[y_src_offset + px + 1]);
                    unsafe_dst.write(y_dst_offset + px + 2, src[y_src_offset + px + 2]);
                    match median_channels {
                        FastBlurChannels::Channels3 => {}
                        FastBlurChannels::Channels4 => {
                            unsafe_dst.write(y_dst_offset + px + 3, src[y_src_offset + px + 3]);
                        }
                    };
                }
            }
        }
    }
}
