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
use crate::filter1d::arena::Arena;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::filter1d::avx::filter_row_avx_f32_f32;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::filter1d::avx::filter_row_avx_u8_f32;
use crate::filter1d::filter_row::filter_row;
use crate::filter1d::filter_row_cg_symmetric::filter_color_group_symmetrical_row;
use crate::filter1d::filter_scan::ScanPoint1d;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::filter1d::neon::{filter_row_neon_f32_f32, filter_row_neon_u8_f32};
use crate::filter1d::region::FilterRegion;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::filter1d::sse::filter_row_sse_f32_f32;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::filter1d::sse::filter_row_sse_u8_f32;
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageSize;
use half::f16;

pub trait Filter1DRowHandler<T, F> {
    fn get_row_handler(
        is_symmetric_kernel: bool,
    ) -> fn(
        arena: Arena,
        arena_src: &[T],
        dst: &UnsafeSlice<T>,
        image_size: ImageSize,
        filter_region: FilterRegion,
        scanned_kernel: &[ScanPoint1d<F>],
    );
}

impl Filter1DRowHandler<u8, f32> for u8 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_row_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if is_symmetric_kernel {
            filter_color_group_symmetrical_row::<u8, f32, 1>
        } else {
            filter_row
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_row_handler(
        _: bool,
    ) -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        filter_row_neon_u8_f32
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_row_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            return filter_row_avx_u8_f32;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return filter_row_sse_u8_f32;
        }
        if is_symmetric_kernel {
            filter_color_group_symmetrical_row::<u8, f32, 1>
        } else {
            filter_row
        }
    }
}

impl Filter1DRowHandler<f32, f32> for f32 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_row_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[f32], &UnsafeSlice<f32>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if is_symmetric_kernel {
            filter_color_group_symmetrical_row::<f32, f32, 1>
        } else {
            filter_row
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_row_handler(
        _: bool,
    ) -> fn(Arena, &[f32], &UnsafeSlice<f32>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        filter_row_neon_f32_f32
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_row_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[f32], &UnsafeSlice<f32>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            return filter_row_avx_f32_f32;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return filter_row_sse_f32_f32;
        }
        if is_symmetric_kernel {
            filter_color_group_symmetrical_row::<f32, f32, 1>
        } else {
            filter_row
        }
    }
}

macro_rules! default_1d_row_handler {
    ($store:ty, $intermediate:ty) => {
        impl Filter1DRowHandler<$store, $intermediate> for $store {
            fn get_row_handler(
                is_symmetric_kernel: bool,
            ) -> fn(
                Arena,
                &[$store],
                &UnsafeSlice<$store>,
                ImageSize,
                FilterRegion,
                &[ScanPoint1d<$intermediate>],
            ) {
                if is_symmetric_kernel {
                    filter_color_group_symmetrical_row::<$store, $intermediate, 1>
                } else {
                    filter_row
                }
            }
        }
    };
}

default_1d_row_handler!(i8, f32);
default_1d_row_handler!(i8, f64);
default_1d_row_handler!(u8, f64);
default_1d_row_handler!(u8, i16);
default_1d_row_handler!(u8, u16);
default_1d_row_handler!(u8, i32);
default_1d_row_handler!(u8, u32);
default_1d_row_handler!(u16, f32);
default_1d_row_handler!(u16, f64);
default_1d_row_handler!(i16, f32);
default_1d_row_handler!(i16, f64);
default_1d_row_handler!(u32, f32);
default_1d_row_handler!(u32, f64);
default_1d_row_handler!(i32, f32);
default_1d_row_handler!(i32, f64);
default_1d_row_handler!(f16, f32);
default_1d_row_handler!(f16, f64);
default_1d_row_handler!(f32, f64);
default_1d_row_handler!(f64, f64);
