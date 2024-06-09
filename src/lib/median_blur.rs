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

use crate::channels_configuration::FastBlurChannels;
use crate::ThreadingPolicy;
use crate::unsafe_slice::UnsafeSlice;

struct MedianHistogram {
    r: [i32; 256],
    g: [i32; 256],
    b: [i32; 256],
    a: [i32; 256],
    n: i32,
}

fn add_rgb_pixels<const CHANNEL_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    y: i64,
    x: i64,
    width: u32,
    height: u32,
    size: u32,
    histogram: &mut MedianHistogram,
) {
    if x < 0 || x >= width as i64 {
        return;
    }
    let px = x as usize * CHANNEL_CONFIGURATION;
    let cap = std::cmp::min(y + size as i64, height as i64 - 1);
    let start = y - size as i64;
    for i in start..=cap {
        if i < 0 {
            continue;
        }
        let y_shift = i as usize * src_stride as usize;
        let v0 = unsafe { *src.get_unchecked(y_shift + px) };
        unsafe {
            *histogram.r.get_unchecked_mut(usize::from(v0)) += 1;
        }
        let v1 = unsafe { *src.get_unchecked(y_shift + px + 1) };
        unsafe {
            *histogram.g.get_unchecked_mut(usize::from(v1)) += 1;
        }
        let v2 = unsafe { *src.get_unchecked(y_shift + px + 2) };
        unsafe {
            *histogram.b.get_unchecked_mut(usize::from(v2)) += 1;
        }
        if CHANNEL_CONFIGURATION == 4 {
            let v3 = unsafe { *src.get_unchecked(y_shift + px + 3) };
            unsafe {
                *histogram.a.get_unchecked_mut(usize::from(v3)) += 1;
            }
            histogram.n += 1;
        }
    }
}

fn remove_rgb_pixels<const CHANNELS_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    y: i64,
    x: i64,
    width: u32,
    height: u32,
    size: u32,
    histogram: &mut MedianHistogram,
) {
    if x < 0 || x >= width as i64 {
        return;
    }
    let px = x as usize * CHANNELS_CONFIGURATION;
    let cap = std::cmp::min(y + size as i64, height as i64 - 1);
    let start = y - size as i64;
    for i in start..=cap {
        if i < 0 {
            continue;
        }
        let y_shift = i as usize * src_stride as usize;
        let v0 = unsafe { *src.get_unchecked(y_shift + px) };
        unsafe {
            *histogram.r.get_unchecked_mut(usize::from(v0)) -= 1;
        }
        let v1 = unsafe { *src.get_unchecked(y_shift + px + 1) };
        unsafe {
            *histogram.g.get_unchecked_mut(usize::from(v1)) -= 1;
        }
        let v2 = unsafe { *src.get_unchecked(y_shift + px + 2) };
        unsafe {
            *histogram.b.get_unchecked_mut(usize::from(v2)) -= 1;
        }
        if CHANNELS_CONFIGURATION == 4 {
            let v3 = unsafe { *src.get_unchecked(y_shift + px + 3) };
            histogram.a[usize::from(v3)] -= 1;
        }
        histogram.n -= 1;
    }
}

fn init_histogram<const CHANNELS_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    y: u32,
    width: u32,
    height: u32,
    radius: u32,
    histogram: &mut MedianHistogram,
) {
    histogram.r = [0; 256];
    histogram.g = [0; 256];
    histogram.b = [0; 256];
    histogram.a = [0; 256];
    histogram.n = 0;

    for j in 0..std::cmp::min(radius, width) {
        add_rgb_pixels::<CHANNELS_CONFIGURATION>(
            src,
            src_stride,
            y as i64,
            j as i64,
            width,
            height,
            radius,
            histogram,
        );
    }
}

fn median_filter(x: [i32; 256], n: i32) -> i32 {
    let mut n = n / 2;
    let mut i = 0i64;
    while i < 256 && i >= 0 {
        n -= unsafe { *x.get_unchecked(i as usize) };
        if n > 0 {
            i += 1;
        } else {
            break;
        }
    }
    i as i32
}

fn median_blur_impl<const CHANNELS_CONFIGURATION: usize>(
    src: &[u8],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
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
            let px = x as usize * CHANNELS_CONFIGURATION;
            if x == 0 {
                init_histogram::<CHANNELS_CONFIGURATION>(
                    src,
                    src_stride,
                    y,
                    width,
                    height,
                    radius,
                    &mut histogram,
                );
            } else {
                remove_rgb_pixels::<CHANNELS_CONFIGURATION>(
                    src,
                    src_stride,
                    y as i64,
                    x as i64 - radius as i64,
                    width,
                    height,
                    radius,
                    &mut histogram,
                );
                add_rgb_pixels::<CHANNELS_CONFIGURATION>(
                    src,
                    src_stride,
                    y as i64,
                    x as i64 + radius as i64,
                    width,
                    height,
                    radius,
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
                    if CHANNELS_CONFIGURATION == 4 {
                        unsafe_dst.write(
                            y_dst_offset + px + 3,
                            median_filter(histogram.a, histogram.n) as u8,
                        );
                    }
                }
            } else {
                unsafe {
                    unsafe_dst.write(y_dst_offset + px, *src.get_unchecked(y_src_offset + px));
                    unsafe_dst.write(y_dst_offset + px + 1, *src.get_unchecked(y_src_offset + px + 1));
                    unsafe_dst.write(y_dst_offset + px + 2, *src.get_unchecked(y_src_offset + px + 2));
                    if CHANNELS_CONFIGURATION == 4 {
                        unsafe_dst.write(y_dst_offset + px + 3, *src.get_unchecked(y_src_offset + px + 3));
                    }
                }
            }
        }
    }
}

#[no_mangle]
#[allow(dead_code)]
pub fn median_blur(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    median_channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    let unsafe_dst = UnsafeSlice::new(dst);
    let thread_count = threading_policy.get_threads_count(width, height) as u32;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();
    pool.scope(|scope| {
        let segment_size = height / thread_count;
        for i in 0..thread_count {
            let start_y = i * segment_size;
            let mut end_y = (i + 1) * segment_size;
            if i == thread_count - 1 {
                end_y = height;
            }
            scope.spawn(move |_| {
                match median_channels {
                    FastBlurChannels::Channels3 => {
                        median_blur_impl::<3>(
                            src,
                            src_stride,
                            &unsafe_dst,
                            dst_stride,
                            width,
                            height,
                            radius,
                            start_y,
                            end_y,
                        );
                    }
                    FastBlurChannels::Channels4 => {
                        median_blur_impl::<4>(
                            src,
                            src_stride,
                            &unsafe_dst,
                            dst_stride,
                            width,
                            height,
                            radius,
                            start_y,
                            end_y,
                        );
                    }
                }
            });
        }
    });
}
