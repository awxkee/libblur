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
use crate::filter1d::Arena;
use crate::filter2d::convolve_op::convolve_segment_2d;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::filter2d::neon::{convolve_segment_neon_2d_u8_f32, convolve_segment_neon_2d_u8_i16};
use crate::filter2d::scan_point_2d::ScanPoint2d;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::filter2d::sse::{convolve_segment_sse_2d_u8_f32, convolve_segment_sse_2d_u8_i16};
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageSize;

pub trait Filter2dHandler<T, F> {
    fn get_executor() -> fn(
        arena: Arena,
        arena_source: &[T],
        dst: &UnsafeSlice<T>,
        image_size: ImageSize,
        prepared_kernel: &[ScanPoint2d<F>],
        y: usize,
    );
}

macro_rules! default_2d_column_handler {
    ($store:ty, $intermediate:ty) => {
        impl Filter2dHandler<$store, $intermediate> for $store {
            fn get_executor() -> fn(
                arena: Arena,
                arena_source: &[$store],
                dst: &UnsafeSlice<$store>,
                image_size: ImageSize,
                prepared_kernel: &[ScanPoint2d<$intermediate>],
                y: usize,
            ) {
                convolve_segment_2d
            }
        }
    };
}

impl Filter2dHandler<u8, f32> for u8 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_executor() -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, &[ScanPoint2d<f32>], usize) {
        convolve_segment_2d
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_executor() -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, &[ScanPoint2d<f32>], usize) {
        convolve_segment_neon_2d_u8_f32
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_executor() -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, &[ScanPoint2d<f32>], usize) {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return convolve_segment_sse_2d_u8_f32;
        }
        convolve_segment_2d
    }
}

impl Filter2dHandler<u8, i16> for i16 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_executor() -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, &[ScanPoint2d<i16>], usize) {
        convolve_segment_2d
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_executor() -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, &[ScanPoint2d<i16>], usize) {
        convolve_segment_neon_2d_u8_i16
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_executor() -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, &[ScanPoint2d<i16>], usize) {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return convolve_segment_sse_2d_u8_i16;
        }
        convolve_segment_2d
    }
}

default_2d_column_handler!(u8, i16);
default_2d_column_handler!(u8, u16);
default_2d_column_handler!(u8, i32);
default_2d_column_handler!(u8, i64);
default_2d_column_handler!(u8, f64);
default_2d_column_handler!(i8, i16);
default_2d_column_handler!(i8, u16);
default_2d_column_handler!(i8, i32);
default_2d_column_handler!(i8, i64);
default_2d_column_handler!(i8, f32);
default_2d_column_handler!(i8, f64);
default_2d_column_handler!(i16, f32);
default_2d_column_handler!(i16, f64);
default_2d_column_handler!(i16, i32);
default_2d_column_handler!(i16, i64);
default_2d_column_handler!(u16, f32);
default_2d_column_handler!(u16, f64);
default_2d_column_handler!(u16, i32);
default_2d_column_handler!(u16, i64);
default_2d_column_handler!(i32, f32);
default_2d_column_handler!(i32, f64);
default_2d_column_handler!(i64, f32);
default_2d_column_handler!(i64, f64);
default_2d_column_handler!(u32, f32);
default_2d_column_handler!(u32, f64);
default_2d_column_handler!(u64, f32);
default_2d_column_handler!(u64, f64);
