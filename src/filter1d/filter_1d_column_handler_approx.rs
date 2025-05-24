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
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::filter1d::avx::{filter_column_avx_symm_u8_i32_app, filter_column_avx_u8_i32_app};
use crate::filter1d::filter_1d_column_handler::FilterBrows;
use crate::filter1d::filter_column_approx::filter_column_approx;
use crate::filter1d::filter_column_approx_symmetric::filter_column_symmetric_approx;
use crate::filter1d::filter_scan::ScanPoint1d;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use crate::filter1d::neon::{filter_column_neon_u8_i32_app, filter_column_symm_neon_u8_i32_app};
use crate::filter1d::region::FilterRegion;
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::filter1d::sse::{filter_column_sse_u8_i32_app, filter_column_symm_u8_i32_app};
use crate::ImageSize;

pub trait Filter1DColumnHandlerApprox<T, F> {
    fn get_column_handler(
        is_kernel_symmetric: bool,
    ) -> fn(
        arena: Arena,
        &[&[T]],
        dst: &mut [T],
        image_size: ImageSize,
        FilterRegion,
        scanned_kernel: &[ScanPoint1d<F>],
    );
}

pub trait Filter1DColumnMultipleRowsApprox<T, F> {
    fn get_column_multiple_rows(
        is_symmetric_kernel: bool,
        kernel: &[ScanPoint1d<F>],
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

macro_rules! default_1d_column_handler {
    ($store:ty, $intermediate:ty) => {
        impl Filter1DColumnHandlerApprox<$store, $intermediate> for $store {
            fn get_column_handler(
                is_kernel_symmetric: bool,
            ) -> fn(
                Arena,
                &[&[$store]],
                &mut [$store],
                ImageSize,
                FilterRegion,
                &[ScanPoint1d<$intermediate>],
            ) {
                if is_kernel_symmetric {
                    filter_column_symmetric_approx
                } else {
                    filter_column_approx
                }
            }
        }
    };
}

impl Filter1DColumnHandlerApprox<u16, u32> for u16 {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn get_column_handler(
        is_kernel_symmetric: bool,
    ) -> fn(Arena, &[&[u16]], &mut [u16], ImageSize, FilterRegion, &[ScanPoint1d<u32>]) {
        if is_kernel_symmetric {
            use crate::filter1d::neon::filter_column_symm_neon_uq15_u16;
            filter_column_symm_neon_uq15_u16
        } else {
            filter_column_approx
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_handler(
        is_kernel_symmetric: bool,
    ) -> fn(Arena, &[&[u16]], &mut [u16], ImageSize, FilterRegion, &[ScanPoint1d<u32>]) {
        if is_kernel_symmetric {
            filter_column_symmetric_approx
        } else {
            filter_column_approx
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_handler(
        is_kernel_symmetric: bool,
    ) -> fn(Arena, &[&[u16]], &mut [u16], ImageSize, FilterRegion, &[ScanPoint1d<u32>]) {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") && is_kernel_symmetric {
            use crate::filter1d::avx::filter_column_avx_symm_uq15_u16;
            return filter_column_avx_symm_uq15_u16;
        }
        #[cfg(feature = "sse")]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") && is_kernel_symmetric {
                use crate::filter1d::sse::filter_column_sse_symm_uq15_u16;
                return filter_column_sse_symm_uq15_u16;
            }
        }
        if is_kernel_symmetric {
            filter_column_symmetric_approx
        } else {
            filter_column_approx
        }
    }
}

impl Filter1DColumnHandlerApprox<u8, i32> for u8 {
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_handler(
        is_kernel_symmetric: bool,
    ) -> fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<i32>]) {
        if is_kernel_symmetric {
            filter_column_symmetric_approx
        } else {
            filter_column_approx
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn get_column_handler(
        is_kernel_symmetric: bool,
    ) -> fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<i32>]) {
        if is_kernel_symmetric {
            #[cfg(feature = "rdm")]
            {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    use crate::filter1d::neon::filter_column_symm_neon_u8_i32_rdm;
                    return filter_column_symm_neon_u8_i32_rdm;
                }
            }
            filter_column_symm_neon_u8_i32_app
        } else {
            #[cfg(feature = "rdm")]
            {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    use crate::filter1d::neon::filter_column_neon_u8_i32_i16_qrdm_app;
                    return filter_column_neon_u8_i32_i16_qrdm_app;
                }
            }
            filter_column_neon_u8_i32_app
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_handler(
        is_kernel_symmetric: bool,
    ) -> fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<i32>]) {
        #[cfg(all(target_arch = "x86_64", feature = "nightly_avx512"))]
        if is_kernel_symmetric && std::arch::is_x86_feature_detected!("avx512bw") {
            use crate::filter1d::avx512::filter_column_avx512_symm_u8_i32_app;
            return filter_column_avx512_symm_u8_i32_app;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            if is_kernel_symmetric {
                return filter_column_avx_symm_u8_i32_app;
            }
            return filter_column_avx_u8_i32_app;
        }
        #[cfg(feature = "sse")]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            if is_kernel_symmetric {
                return filter_column_symm_u8_i32_app;
            }
            return filter_column_sse_u8_i32_app;
        }
        if is_kernel_symmetric {
            filter_column_symmetric_approx
        } else {
            filter_column_approx
        }
    }
}

impl Filter1DColumnMultipleRowsApprox<u8, i32> for u8 {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn get_column_multiple_rows(
        is_symmetric_kernel: bool,
        _: &[ScanPoint1d<i32>],
    ) -> Option<fn(Arena, FilterBrows<u8>, &mut [u8], ImageSize, usize, &[ScanPoint1d<i32>])> {
        if is_symmetric_kernel {
            #[cfg(feature = "rdm")]
            {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    use crate::filter1d::neon::filter_column_symm_neon_u8_i32_rdm_x3;
                    return Some(filter_column_symm_neon_u8_i32_rdm_x3);
                }
            }
        }
        None
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_multiple_rows(
        _is_symmetric: bool,
        _: &[ScanPoint1d<i32>],
    ) -> Option<fn(Arena, FilterBrows<u8>, &mut [u8], ImageSize, usize, &[ScanPoint1d<i32>])> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if _is_symmetric && std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_column_avx_symm_u8_i32_app_x2;
                return Some(filter_column_avx_symm_u8_i32_app_x2);
            }
        }
        None
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_multiple_rows(
        _: bool,
        _: &[ScanPoint1d<i32>],
    ) -> Option<fn(Arena, FilterBrows<u8>, &mut [u8], ImageSize, usize, &[ScanPoint1d<i32>])> {
        None
    }
}

impl Filter1DColumnMultipleRowsApprox<u16, u32> for u16 {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn get_column_multiple_rows(
        is_symmetric_kernel: bool,
        _: &[ScanPoint1d<u32>],
    ) -> Option<fn(Arena, FilterBrows<u16>, &mut [u16], ImageSize, usize, &[ScanPoint1d<u32>])>
    {
        if is_symmetric_kernel {
            use crate::filter1d::neon::filter_symm_column_neon_uq15_u16_x3;
            return Some(filter_symm_column_neon_uq15_u16_x3);
        }
        None
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_column_multiple_rows(
        _: bool,
        _: &[ScanPoint1d<u32>],
    ) -> Option<fn(Arena, FilterBrows<u16>, &mut [u16], ImageSize, usize, &[ScanPoint1d<u32>])>
    {
        None
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_column_multiple_rows(
        _is_kernel_symmetric: bool,
        _: &[ScanPoint1d<u32>],
    ) -> Option<fn(Arena, FilterBrows<u16>, &mut [u16], ImageSize, usize, &[ScanPoint1d<u32>])>
    {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") && _is_kernel_symmetric {
            use crate::filter1d::avx::filter_column_avx_symm_uq15_u16_x2;
            return Some(filter_column_avx_symm_uq15_u16_x2);
        }
        None
    }
}

default_1d_column_handler!(u8, i64);
default_1d_column_handler!(u8, u16);
default_1d_column_handler!(u8, i16);
default_1d_column_handler!(u8, u32);
default_1d_column_handler!(u8, u64);
default_1d_column_handler!(i8, i32);
default_1d_column_handler!(i8, i64);
default_1d_column_handler!(i8, i16);
default_1d_column_handler!(u16, i32);
default_1d_column_handler!(u16, i64);
default_1d_column_handler!(u16, u64);
default_1d_column_handler!(i16, i32);
default_1d_column_handler!(i16, i64);

macro_rules! default_1d_column_multiple_rows {
    ($store:ty, $intermediate:ty) => {
        impl Filter1DColumnMultipleRowsApprox<$store, $intermediate> for $store {
            fn get_column_multiple_rows(
                _: bool,
                _: &[ScanPoint1d<$intermediate>],
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

default_1d_column_multiple_rows!(u8, i64);
default_1d_column_multiple_rows!(u8, u16);
default_1d_column_multiple_rows!(u8, i16);
default_1d_column_multiple_rows!(u8, u32);
default_1d_column_multiple_rows!(u8, u64);
default_1d_column_multiple_rows!(i8, i32);
default_1d_column_multiple_rows!(i8, i64);
default_1d_column_multiple_rows!(i8, i16);
default_1d_column_multiple_rows!(u16, i32);
default_1d_column_multiple_rows!(u16, i64);
default_1d_column_multiple_rows!(u16, u64);
default_1d_column_multiple_rows!(i16, i32);
default_1d_column_multiple_rows!(i16, i64);
