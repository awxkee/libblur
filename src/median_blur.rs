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
use crate::unsafe_slice::UnsafeSlice;
use crate::{BlurError, BlurImage, BlurImageMut, ThreadingPolicy};

struct MedianHistogram {
    r: [i32; 256],
    g: [i32; 256],
    b: [i32; 256],
    a: [i32; 256],
    n: i32,
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn add_rgb_pixels<const CN: usize>(
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
    let px = x as usize * CN;
    let cap = (y + size as i64).min(height as i64 - 1);
    let start = y - size as i64;
    for i in start..=cap {
        if i < 0 {
            continue;
        }
        let y_shift = i as usize * src_stride as usize;
        let bytes_offset = y_shift + px;
        let v0 = unsafe { *src.get_unchecked(bytes_offset) };
        unsafe {
            *histogram.r.get_unchecked_mut(v0 as usize) += 1;
        }
        if CN > 1 {
            let v1 = unsafe { *src.get_unchecked(bytes_offset + 1) };
            unsafe {
                *histogram.g.get_unchecked_mut(v1 as usize) += 1;
            }
        }
        if CN > 2 {
            let v2 = unsafe { *src.get_unchecked(bytes_offset + 2) };
            unsafe {
                *histogram.b.get_unchecked_mut(v2 as usize) += 1;
            }
        }
        if CN == 4 {
            unsafe {
                let v3 = *src.get_unchecked(bytes_offset + 3);
                *histogram.a.get_unchecked_mut(v3 as usize) += 1;
            }
        }
        histogram.n += 1;
    }
}

#[inline(always)]
fn remove_rgb_pixels<const CN: usize>(
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
    let px = x as usize * CN;
    let cap = (y + size as i64).min(height as i64 - 1);
    let start = y - size as i64;
    for i in start..=cap {
        if i < 0 {
            continue;
        }
        let y_shift = i as usize * src_stride as usize;
        let bytes_offset = y_shift + px;
        let v0 = unsafe { *src.get_unchecked(bytes_offset) };
        unsafe {
            let k = histogram.r.get_unchecked_mut(v0 as usize);
            let x = *k - 1;
            *k = x;
        }
        if CN > 1 {
            let v1 = unsafe { *src.get_unchecked(bytes_offset + 1) };
            unsafe {
                *histogram.g.get_unchecked_mut(v1 as usize) -= 1;
            }
        }
        if CN > 2 {
            let v2 = unsafe { *src.get_unchecked(bytes_offset + 2) };
            unsafe {
                *histogram.b.get_unchecked_mut(v2 as usize) -= 1;
            }
        }
        if CN == 4 {
            let v3 = unsafe { *src.get_unchecked(bytes_offset + 3) };
            *unsafe { histogram.a.get_unchecked_mut(v3 as usize) } -= 1;
        }
        histogram.n -= 1;
    }
}

#[inline(always)]
fn init_histogram<const CN: usize>(
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

    for j in 0..radius.min(width) {
        add_rgb_pixels::<CN>(
            src, src_stride, y as i64, j as i64, width, height, radius, histogram,
        );
    }
}

#[inline(always)]
fn median_filter(x: [i32; 256], n: i32) -> i32 {
    let mut n = n / 2;
    let mut i = 0i64;
    #[allow(clippy::manual_range_contains)]
    while i < 256 && i >= 0 {
        unsafe {
            n -= *x.get_unchecked(i as usize);
        }
        if n > 0 {
            i += 1;
        } else {
            break;
        }
    }
    i as i32
}

fn median_blur_impl<const CN: usize>(
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
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                median_blur_impl_avx2::<CN>(
                    src, src_stride, unsafe_dst, dst_stride, width, height, radius, start_y, end_y,
                )
            };
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return unsafe {
                median_blur_impl_sse_4_1::<CN>(
                    src, src_stride, unsafe_dst, dst_stride, width, height, radius, start_y, end_y,
                )
            };
        }
    }
    median_blur_impl_exec::<CN>(
        src, src_stride, unsafe_dst, dst_stride, width, height, radius, start_y, end_y,
    );
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
#[target_feature(enable = "avx2")]
unsafe fn median_blur_impl_avx2<const CN: usize>(
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
    median_blur_impl_exec::<CN>(
        src, src_stride, unsafe_dst, dst_stride, width, height, radius, start_y, end_y,
    );
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
#[target_feature(enable = "sse4.1")]
unsafe fn median_blur_impl_sse_4_1<const CN: usize>(
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
    median_blur_impl_exec::<CN>(
        src, src_stride, unsafe_dst, dst_stride, width, height, radius, start_y, end_y,
    );
}

#[inline(always)]
fn median_blur_impl_exec<const CN: usize>(
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
        let mut histogram = MedianHistogram {
            r: [0; 256],
            g: [0; 256],
            b: [0; 256],
            a: [0; 256],
            n: 0,
        };
        for x in 0..width {
            let px = x as usize * CN;
            if x == 0 {
                init_histogram::<CN>(src, src_stride, y, width, height, radius, &mut histogram);
            } else {
                remove_rgb_pixels::<CN>(
                    src,
                    src_stride,
                    y as i64,
                    x as i64 - radius as i64,
                    width,
                    height,
                    radius,
                    &mut histogram,
                );
                add_rgb_pixels::<CN>(
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

            let y_src_offset = y as usize * src_stride as usize;
            let y_dst_offset = y as usize * dst_stride as usize;

            push_hist::<CN>(src, unsafe_dst, y_src_offset, y_dst_offset, &histogram, px);
        }
    }
}

#[inline(always)]
fn push_hist<const CN: usize>(
    src: &[u8],
    unsafe_dst: &UnsafeSlice<u8>,
    y_src_offset: usize,
    y_dst_offset: usize,
    histogram: &MedianHistogram,
    px: usize,
) {
    if histogram.n > 0 {
        unsafe {
            let bytes_offset = y_dst_offset + px;
            unsafe_dst.write(bytes_offset, median_filter(histogram.r, histogram.n) as u8);
            if CN > 1 {
                unsafe_dst.write(
                    bytes_offset + 1,
                    median_filter(histogram.g, histogram.n) as u8,
                );
            }
            if CN > 2 {
                unsafe_dst.write(
                    bytes_offset + 2,
                    median_filter(histogram.b, histogram.n) as u8,
                );
            }
            if CN == 4 {
                unsafe_dst.write(
                    bytes_offset + 3,
                    median_filter(histogram.a, histogram.n) as u8,
                );
            }
        }
    } else {
        unsafe {
            let bytes_offset = y_dst_offset + px;
            let src_offset = y_src_offset + px;
            unsafe_dst.write(bytes_offset, *src.get_unchecked(src_offset));
            if CN > 1 {
                unsafe_dst.write(bytes_offset + 1, *src.get_unchecked(src_offset + 1));
            }
            if CN > 2 {
                unsafe_dst.write(bytes_offset + 2, *src.get_unchecked(src_offset + 2));
            }
            if CN == 4 {
                unsafe_dst.write(bytes_offset + 3, *src.get_unchecked(src_offset + 3));
            }
        }
    }
}

/// Performs median blur on the image.
///
/// This performs a median kernel filter on the image producing edge preserving blur result.
/// Preferred if you need to save edges.
/// O(R) complexity.
///
/// # Arguments
///
/// * `src_image` - Src image, see [BlurImage] for more info
/// * `dst_image` - Destination image, see [BlurImageMut] for more info
/// * `radius` - Radius of kernel
/// * `channels` - Count of channels in the image, see [FastBlurChannels] for more info
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
#[allow(clippy::too_many_arguments)]
pub fn median_blur(
    src_image: &BlurImage<u8>,
    dst_image: &mut BlurImageMut<u8>,
    radius: u32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    src_image.check_layout()?;
    dst_image.check_layout(Some(src_image))?;
    src_image.size_matches_mut(dst_image)?;
    let _dispatcher = match src_image.channels {
        FastBlurChannels::Plane => median_blur_impl::<1>,
        FastBlurChannels::Channels3 => median_blur_impl::<3>,
        FastBlurChannels::Channels4 => median_blur_impl::<4>,
    };
    let width = src_image.width;
    let height = src_image.height;
    let src_stride = src_image.row_stride();
    let dst_stride = dst_image.row_stride();

    let thread_count = threading_policy.thread_count(width, height) as u32;

    let pool = novtb::ThreadPool::new(thread_count as usize);

    let unsafe_dst = UnsafeSlice::new(dst_image.data.borrow_mut());

    let src = src_image.data.as_ref();
    pool.parallel_for(|thread_index| {
        let segment_size = height / thread_count;
        let start_y = thread_index as u32 * segment_size;
        let mut end_y = (thread_index as u32 + 1) * segment_size;
        if thread_index as u32 == thread_count - 1 {
            end_y = height;
        }
        _dispatcher(
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
    });
    Ok(())
}
