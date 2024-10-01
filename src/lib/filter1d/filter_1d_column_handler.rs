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
use crate::filter1d::filter_column::filter_column;
use crate::filter1d::filter_scan::ScanPoint1d;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::filter1d::neon::{filter_column_neon_f32_f32, filter_column_neon_u8_f32};
use crate::filter1d::region::FilterRegion;
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageSize;
use half::f16;

pub trait Filter1DColumnHandler<T, F> {
    fn get_column_handler() -> fn(
        arena: Arena,
        &[T],
        dst: &UnsafeSlice<T>,
        image_size: ImageSize,
        FilterRegion,
        scanned_kernel: &[ScanPoint1d<F>],
    );
}

impl Filter1DColumnHandler<u8, f32> for u8 {
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    fn get_column_handler(
    ) -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        filter_column
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_column_handler(
    ) -> fn(Arena, &[u8], &UnsafeSlice<u8>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        filter_column_neon_u8_f32
    }
}

impl Filter1DColumnHandler<f32, f32> for f32 {
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    fn get_column_handler(
    ) -> fn(Arena, &[f32], &UnsafeSlice<f32>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        filter_column
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_column_handler(
    ) -> fn(Arena, &[f32], &UnsafeSlice<f32>, ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        filter_column_neon_f32_f32
    }
}

macro_rules! default_1d_column_handler {
    ($store:ty, $intermediate:ty) => {
        impl Filter1DColumnHandler<$store, $intermediate> for $store {
            fn get_column_handler() -> fn(
                Arena,
                &[$store],
                &UnsafeSlice<$store>,
                ImageSize,
                FilterRegion,
                &[ScanPoint1d<$intermediate>],
            ) {
                filter_column
            }
        }
    };
}

default_1d_column_handler!(u8, f64);
default_1d_column_handler!(u16, f32);
default_1d_column_handler!(u16, f64);
default_1d_column_handler!(i16, f32);
default_1d_column_handler!(i16, f64);
default_1d_column_handler!(u32, f32);
default_1d_column_handler!(u32, f64);
default_1d_column_handler!(i32, f32);
default_1d_column_handler!(i32, f64);
default_1d_column_handler!(f16, f32);
default_1d_column_handler!(f16, f64);
default_1d_column_handler!(f32, f64);
default_1d_column_handler!(f64, f64);