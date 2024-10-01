/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn copy_row_neon(dst: &mut [u8], src: &[u8], start: usize, stride: usize) -> usize {
    let mut _cx = start;
    while _cx + 64 < stride {
        let rows = vld1q_u8_x4(src.as_ptr().add(_cx));
        vst1q_u8_x4(dst.as_mut_ptr().add(_cx), rows);
        _cx += 64;
    }

    while _cx + 32 < stride {
        let rows = vld1q_u8_x2(src.as_ptr().add(_cx));
        vst1q_u8_x2(dst.as_mut_ptr().add(_cx), rows);
        _cx += 32;
    }

    while _cx + 16 < stride {
        let rows = vld1q_u8(src.as_ptr().add(_cx));
        vst1q_u8(dst.as_mut_ptr().add(_cx), rows);
        _cx += 16;
    }

    while _cx + 8 < stride {
        let rows = vld1_u8(src.as_ptr().add(_cx));
        vst1_u8(dst.as_mut_ptr().add(_cx), rows);
        _cx += 8;
    }

    _cx
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn copy_row_sse(dst: &mut [u8], src: &[u8], start: usize, stride: usize) -> usize {
    let mut _cx = start;
    while _cx + 64 < stride {
        let offset_ptr = src.as_ptr().add(_cx);
        let row0 = _mm_loadu_si128(offset_ptr as *const __m128i);
        let row1 = _mm_loadu_si128(offset_ptr.add(16) as *const __m128i);
        let row2 = _mm_loadu_si128(offset_ptr.add(32) as *const __m128i);
        let row3 = _mm_loadu_si128(offset_ptr.add(48) as *const __m128i);
        let dst_offset_ptr = dst.as_mut_ptr().add(_cx);
        _mm_storeu_si128(dst_offset_ptr as *mut __m128i, row0);
        _mm_storeu_si128(dst_offset_ptr.add(16) as *mut __m128i, row1);
        _mm_storeu_si128(dst_offset_ptr.add(32) as *mut __m128i, row2);
        _mm_storeu_si128(dst_offset_ptr.add(48) as *mut __m128i, row3);
        _cx += 64;
    }

    while _cx + 32 < stride {
        let offset_ptr = src.as_ptr().add(_cx);
        let row0 = _mm_loadu_si128(offset_ptr as *const __m128i);
        let row1 = _mm_loadu_si128(offset_ptr.add(16) as *const __m128i);
        let dst_offset_ptr = dst.as_mut_ptr().add(_cx);
        _mm_storeu_si128(dst_offset_ptr as *mut __m128i, row0);
        _mm_storeu_si128(dst_offset_ptr.add(16) as *mut __m128i, row1);
        _cx += 32;
    }

    while _cx + 16 < stride {
        let offset_ptr = src.as_ptr().add(_cx);
        let row0 = _mm_loadu_si128(offset_ptr as *const __m128i);
        let dst_offset_ptr = dst.as_mut_ptr().add(_cx);
        _mm_storeu_si128(dst_offset_ptr as *mut __m128i, row0);
        _cx += 16;
    }

    while _cx + 8 < stride {
        let offset_ptr = src.as_ptr().add(_cx);
        let row0 = _mm_loadu_si64(offset_ptr);
        let dst_offset_ptr = dst.as_mut_ptr().add(_cx);
        std::ptr::copy_nonoverlapping(&row0 as *const _ as *const u8, dst_offset_ptr, 8);
        _cx += 8;
    }

    _cx
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn copy_row_avx(dst: &mut [u8], src: &[u8], start: usize, stride: usize) -> usize {
    let mut _cx = start;

    while _cx + 128 < stride {
        let offset_ptr = src.as_ptr().add(_cx);
        let row0 = _mm256_loadu_si256(offset_ptr as *const __m256i);
        let row1 = _mm256_loadu_si256(offset_ptr.add(32) as *const __m256i);
        let row2 = _mm256_loadu_si256(offset_ptr.add(64) as *const __m256i);
        let row3 = _mm256_loadu_si256(offset_ptr.add(96) as *const __m256i);
        let dst_offset_ptr = dst.as_mut_ptr().add(_cx);
        _mm256_storeu_si256(dst_offset_ptr as *mut __m256i, row0);
        _mm256_storeu_si256(dst_offset_ptr.add(32) as *mut __m256i, row1);
        _mm256_storeu_si256(dst_offset_ptr.add(64) as *mut __m256i, row2);
        _mm256_storeu_si256(dst_offset_ptr.add(96) as *mut __m256i, row3);
        _cx += 128;
    }

    while _cx + 64 < stride {
        let offset_ptr = src.as_ptr().add(_cx);
        let row0 = _mm256_loadu_si256(offset_ptr as *const __m256i);
        let row1 = _mm256_loadu_si256(offset_ptr.add(32) as *const __m256i);
        let dst_offset_ptr = dst.as_mut_ptr().add(_cx);
        _mm256_storeu_si256(dst_offset_ptr as *mut __m256i, row0);
        _mm256_storeu_si256(dst_offset_ptr.add(32) as *mut __m256i, row1);
        _cx += 64;
    }

    while _cx + 32 < stride {
        let offset_ptr = src.as_ptr().add(_cx);
        let row0 = _mm256_loadu_si256(offset_ptr as *const __m256i);
        let dst_offset_ptr = dst.as_mut_ptr().add(_cx);
        _mm256_storeu_si256(dst_offset_ptr as *mut __m256i, row0);
        _cx += 32;
    }

    while _cx + 16 < stride {
        let offset_ptr = src.as_ptr().add(_cx);
        let row0 = _mm_loadu_si128(offset_ptr as *const __m128i);
        let dst_offset_ptr = dst.as_mut_ptr().add(_cx);
        _mm_storeu_si128(dst_offset_ptr as *mut __m128i, row0);
        _cx += 16;
    }

    while _cx + 8 < stride {
        let offset_ptr = src.as_ptr().add(_cx);
        let row0 = _mm_loadu_si64(offset_ptr);
        let dst_offset_ptr = dst.as_mut_ptr().add(_cx);
        std::ptr::copy_nonoverlapping(&row0 as *const _ as *const u8, dst_offset_ptr, 8);
        _cx += 8;
    }

    _cx
}

/// Copies ROI from one image to another
#[allow(clippy::type_complexity)]
pub fn copy_roi<T>(arena: &mut [T], roi: &[T], arena_stride: usize, stride: usize, height: usize)
where
    T: Copy,
{
    if std::any::type_name::<T>() == "u8" {
        let mut dst: &mut [u8] = unsafe { std::mem::transmute(arena) };
        let mut src = unsafe { std::mem::transmute::<&[T], &[u8]>(roi) };
        let mut _row_handle: Option<unsafe fn(&mut [u8], &[u8], usize, usize) -> usize> = None;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _row_handle = Some(copy_row_neon);
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _row_handle = Some(copy_row_sse);
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                _row_handle = Some(copy_row_avx);
            }
        }
        unsafe {
            for y in 0..height {
                let mut _cx = 0usize;

                if let Some(row_handle) = _row_handle {
                    _cx = row_handle(dst, src, _cx, stride);
                }

                while _cx < stride {
                    *dst.get_unchecked_mut(_cx) = *src.get_unchecked(_cx);
                    _cx += 1;
                }

                if y + 1 < height {
                    dst = dst.get_unchecked_mut(arena_stride..);
                    src = src.get_unchecked(stride..);
                }
            }
        }
    } else {
        let mut dst = arena;
        let mut src = roi;
        unsafe {
            for y in 0..height {
                let mut _cx = 0usize;

                while _cx < stride {
                    *dst.get_unchecked_mut(_cx) = *src.get_unchecked(_cx);
                    _cx += 1;
                }

                if y + 1 < height {
                    dst = dst.get_unchecked_mut(arena_stride..);
                    src = src.get_unchecked(stride..);
                }
            }
        }
    }
}
