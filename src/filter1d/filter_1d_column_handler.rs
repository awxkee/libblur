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
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
use crate::filter1d::avx::{
    filter_column_avx_f32_f32, filter_column_avx_symm_u8_f32, filter_column_avx_u8_f32,
};
use crate::filter1d::filter_column::filter_column;
use crate::filter1d::filter_column_symmetric::filter_symmetric_column;
use crate::filter1d::filter_scan::ScanPoint1d;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::filter1d::neon::{
    filter_column_neon_f32_f32, filter_column_neon_symm_f32_f32, filter_column_neon_u8_f32,
    filter_column_neon_u8_i16, filter_column_symm_neon_u8_i16, filter_symm_column_neon_u8_f32,
};
use crate::filter1d::region::FilterRegion;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::filter1d::sse::{
    filter_column_sse_f32_f32, filter_column_sse_u8_f32, filter_column_sse_u8_i16,
    filter_column_symm_sse_u8_f32, filter_column_symm_sse_u8_i16,
};
use crate::ImageSize;
use half::f16;

#[allow(dead_code, unused)]
#[derive(Clone, Debug)]
pub struct FilterBrows<'a, T> {
    pub(crate) brows: Vec<&'a [&'a [T]]>,
}

pub trait Filter1DColumnHandlerMultipleRows<T, F> {
    fn get_column_handler_multiple_rows(
        is_symmetric_kernel: bool,
    ) -> Option<
        fn(
            arena: Arena,
            FilterBrows<T>,
            dst: &mut [T],
            image_size: ImageSize,
            dst_stride: usize,
            scanned_kernel: &[ScanPoint1d<F>],
        ),
    >;
}

pub trait Filter1DColumnHandler<T, F> {
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(
        arena: Arena,
        &[&[T]],
        dst: &mut [T],
        image_size: ImageSize,
        FilterRegion,
        scanned_kernel: &[ScanPoint1d<F>],
    );
}

impl Filter1DColumnHandler<u8, f32> for u8 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_handler(
        _: bool,
    ) -> fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        filter_column
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if is_symmetric_kernel {
            filter_symm_column_neon_u8_f32
        } else {
            filter_column_neon_u8_f32
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        #[cfg(feature = "avx")]
        if std::arch::is_x86_feature_detected!("avx2") {
            if is_symmetric_kernel {
                return filter_column_avx_symm_u8_f32;
            }
            return filter_column_avx_u8_f32;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            if is_symmetric_kernel {
                return filter_column_symm_sse_u8_f32;
            }
            return filter_column_sse_u8_f32;
        }
        filter_column
    }
}

impl Filter1DColumnHandler<u8, i16> for u8 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<i16>]) {
        if is_symmetric_kernel {
            filter_symmetric_column
        } else {
            filter_column
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<i16>]) {
        if is_symmetric_kernel {
            filter_column_symm_neon_u8_i16
        } else {
            filter_column_neon_u8_i16
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<i16>]) {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            if is_symmetric_kernel {
                return filter_column_symm_sse_u8_i16;
            }
            return filter_column_sse_u8_i16;
        }
        if is_symmetric_kernel {
            filter_symmetric_column
        } else {
            filter_column
        }
    }
}

impl Filter1DColumnHandler<f32, f32> for f32 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[f32]], &mut [f32], ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if is_symmetric_kernel {
            filter_symmetric_column
        } else {
            filter_column
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[f32]], &mut [f32], ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if is_symmetric_kernel {
            filter_column_neon_symm_f32_f32
        } else {
            filter_column_neon_f32_f32
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[f32]], &mut [f32], ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        #[cfg(feature = "avx")]
        if std::arch::is_x86_feature_detected!("avx2") {
            if std::arch::is_x86_feature_detected!("fma") && is_symmetric_kernel {
                use crate::filter1d::avx::filter_column_avx_f32_f32_symm;
                return filter_column_avx_f32_f32_symm;
            }
            return filter_column_avx_f32_f32;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return filter_column_sse_f32_f32;
        }
        if is_symmetric_kernel {
            filter_symmetric_column
        } else {
            filter_column
        }
    }
}

impl Filter1DColumnHandler<u16, f32> for u16 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[u16]], &mut [u16], ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if is_symmetric_kernel {
            use crate::filter1d::neon::filter_symm_column_neon_u16_f32;
            filter_symm_column_neon_u16_f32
        } else {
            filter_column
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[u16]], &mut [u16], ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if is_symmetric_kernel {
            #[cfg(feature = "avx")]
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_column_avx_symm_u16_f32;
                return filter_column_avx_symm_u16_f32;
            }
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_column_symm_sse_u16_f32;
                return filter_column_symm_sse_u16_f32;
            }
            filter_symmetric_column
        } else {
            filter_column
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_handler(
        is_symmetric_kernel: bool,
    ) -> fn(Arena, &[&[u16]], &mut [u16], ImageSize, FilterRegion, &[ScanPoint1d<f32>]) {
        if is_symmetric_kernel {
            filter_symmetric_column
        } else {
            filter_column
        }
    }
}

impl Filter1DColumnHandlerMultipleRows<u8, f32> for u8 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_column_handler_multiple_rows(
        is_symmetric_kernel: bool,
    ) -> Option<fn(Arena, FilterBrows<u8>, &mut [u8], ImageSize, usize, &[ScanPoint1d<f32>])> {
        if is_symmetric_kernel {
            use crate::filter1d::neon::filter_symm_column_neon_u8_f32_x3;
            return Some(filter_symm_column_neon_u8_f32_x3);
        }
        None
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_handler_multiple_rows(
        _: bool,
    ) -> Option<fn(Arena, FilterBrows<u8>, &mut [u8], ImageSize, usize, &[ScanPoint1d<f32>])> {
        None
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_handler_multiple_rows(
        is_symmetric_kernel: bool,
    ) -> Option<fn(Arena, FilterBrows<u8>, &mut [u8], ImageSize, usize, &[ScanPoint1d<f32>])> {
        if is_symmetric_kernel {
            #[cfg(feature = "avx")]
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_column_avx_symm_u8_f32_x2;
                return Some(filter_column_avx_symm_u8_f32_x2);
            }
        }
        None
    }
}

impl Filter1DColumnHandlerMultipleRows<u16, f32> for u16 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_column_handler_multiple_rows(
        is_symmetric_kernel: bool,
    ) -> Option<fn(Arena, FilterBrows<u16>, &mut [u16], ImageSize, usize, &[ScanPoint1d<f32>])>
    {
        if is_symmetric_kernel {
            use crate::filter1d::neon::filter_symm_column_neon_u16_f32_x3;
            return Some(filter_symm_column_neon_u16_f32_x3);
        }
        None
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_handler_multiple_rows(
        _: bool,
    ) -> Option<fn(Arena, FilterBrows<u16>, &mut [u16], ImageSize, usize, &[ScanPoint1d<f32>])>
    {
        None
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_handler_multiple_rows(
        is_symmetric_kernel: bool,
    ) -> Option<fn(Arena, FilterBrows<u16>, &mut [u16], ImageSize, usize, &[ScanPoint1d<f32>])>
    {
        if is_symmetric_kernel {
            #[cfg(feature = "avx")]
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_column_avx_symm_u16_f32_x2;
                return Some(filter_column_avx_symm_u16_f32_x2);
            }
        }
        None
    }
}

macro_rules! default_1d_column_handler {
    ($store:ty, $intermediate:ty) => {
        impl Filter1DColumnHandler<$store, $intermediate> for $store {
            fn get_column_handler(
                is_symmetric_kernel: bool,
            ) -> fn(
                Arena,
                &[&[$store]],
                &mut [$store],
                ImageSize,
                FilterRegion,
                &[ScanPoint1d<$intermediate>],
            ) {
                if is_symmetric_kernel {
                    filter_symmetric_column
                } else {
                    filter_column
                }
            }
        }
    };
}

macro_rules! default_1d_column_multiple_rows {
    ($store:ty, $intermediate:ty) => {
        impl Filter1DColumnHandlerMultipleRows<$store, $intermediate> for $store {
            fn get_column_handler_multiple_rows(
                _: bool,
            ) -> Option<
                fn(
                    Arena,
                    FilterBrows<$store>,
                    &mut [$store],
                    ImageSize,
                    usize,
                    &[ScanPoint1d<$intermediate>],
                ),
            > {
                None
            }
        }
    };
}

default_1d_column_handler!(i8, f32);
default_1d_column_handler!(i8, f64);
default_1d_column_handler!(u8, f64);
default_1d_column_handler!(u8, u16);
default_1d_column_handler!(u8, i32);
default_1d_column_handler!(u8, u32);
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

default_1d_column_multiple_rows!(i8, f32);
default_1d_column_multiple_rows!(i8, f64);
default_1d_column_multiple_rows!(u8, f64);
default_1d_column_multiple_rows!(u8, u16);
default_1d_column_multiple_rows!(u8, i16);
default_1d_column_multiple_rows!(u8, i32);
default_1d_column_multiple_rows!(u8, u32);
default_1d_column_multiple_rows!(u16, f64);
default_1d_column_multiple_rows!(i16, f32);
default_1d_column_multiple_rows!(i16, f64);
default_1d_column_multiple_rows!(u32, f32);
default_1d_column_multiple_rows!(u32, f64);
default_1d_column_multiple_rows!(i32, f32);
default_1d_column_multiple_rows!(i32, f64);
default_1d_column_multiple_rows!(f16, f32);
default_1d_column_multiple_rows!(f16, f64);
default_1d_column_multiple_rows!(f32, f64);
default_1d_column_multiple_rows!(f64, f64);
default_1d_column_multiple_rows!(f32, f32);
