/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
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
use crate::filter1d::filter_column_complex::filter_column_complex;
use crate::filter1d::filter_row_complex::filter_row_complex;
use crate::{Arena, ImageSize};
use num_complex::Complex;

pub trait ComplexDispatch<T, F> {
    fn column_dispatch(
        is_symmetric_kernel: bool,
    ) -> fn(
        arena: Arena,
        arena_src: &[&[Complex<F>]],
        dst: &mut [T],
        image_size: ImageSize,
        kernel: &[Complex<F>],
    );

    fn row_dispatch(
        is_symmetric_kernel: bool,
    ) -> fn(
        arena: Arena,
        arena_src: &[T],
        dst: &mut [Complex<F>],
        image_size: ImageSize,
        kernel: &[Complex<F>],
    );
}

impl ComplexDispatch<u8, f32> for u8 {
    fn column_dispatch(
        _: bool,
    ) -> fn(Arena, &[&[Complex<f32>]], &mut [u8], ImageSize, &[Complex<f32>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "nightly_fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::filter1d::neon::filter_column_complex_u8_f32_fcma;
                return filter_column_complex_u8_f32_fcma;
            }
            use crate::filter1d::neon::filter_column_complex_u8_f32;
            filter_column_complex_u8_f32
        }
        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_avx_column_complex_u8_f32;
                return filter_avx_column_complex_u8_f32;
            }
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_column_complex_u8_f32;
                return filter_sse_column_complex_u8_f32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            filter_column_complex
        }
    }

    fn row_dispatch(_: bool) -> fn(Arena, &[u8], &mut [Complex<f32>], ImageSize, &[Complex<f32>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter1d::neon::filter_row_complex_u8_f32;
            filter_row_complex_u8_f32
        }
        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::filter1d::avx::filter_avx_row_complex_u8_f32;
                return filter_avx_row_complex_u8_f32;
            }
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_row_complex_u8_f32;
                return filter_sse_row_complex_u8_f32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            filter_row_complex
        }
    }
}

impl ComplexDispatch<u16, f32> for u16 {
    fn column_dispatch(
        _: bool,
    ) -> fn(Arena, &[&[Complex<f32>]], &mut [u16], ImageSize, &[Complex<f32>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "nightly_fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::filter1d::neon::filter_column_complex_u16_f32_fcma;
                return filter_column_complex_u16_f32_fcma;
            }
            use crate::filter1d::neon::filter_column_complex_u16_f32;
            filter_column_complex_u16_f32
        }
        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_avx_column_complex_u16_f32;
                return filter_avx_column_complex_u16_f32;
            }
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_column_complex_u16_f32;
                return filter_sse_column_complex_u16_f32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            filter_column_complex
        }
    }

    fn row_dispatch(_: bool) -> fn(Arena, &[u16], &mut [Complex<f32>], ImageSize, &[Complex<f32>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter1d::neon::filter_row_complex_u16_f32;
            filter_row_complex_u16_f32
        }
        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::filter1d::avx::filter_avx_row_complex_u16_f32;
                return filter_avx_row_complex_u16_f32;
            }
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_row_complex_u16_f32;
                return filter_sse_row_complex_u16_f32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            filter_row_complex
        }
    }
}

impl ComplexDispatch<f32, f32> for f32 {
    fn column_dispatch(
        _: bool,
    ) -> fn(Arena, &[&[Complex<f32>]], &mut [f32], ImageSize, &[Complex<f32>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "nightly_fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::filter1d::neon::filter_column_complex_f32_f32_fcma;
                return filter_column_complex_f32_f32_fcma;
            }
            use crate::filter1d::neon::filter_column_complex_f32_f32;
            filter_column_complex_f32_f32
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_column_complex_f32_f32;
                return filter_sse_column_complex_f32_f32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            filter_column_complex
        }
    }

    fn row_dispatch(_: bool) -> fn(Arena, &[f32], &mut [Complex<f32>], ImageSize, &[Complex<f32>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter1d::neon::filter_row_complex_f32_f32;
            filter_row_complex_f32_f32
        }
        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::filter1d::avx::filter_avx_row_complex_f32_f32;
                return filter_avx_row_complex_f32_f32;
            }
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_row_complex_f32_f32;
                return filter_sse_row_complex_f32_f32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            filter_row_complex
        }
    }
}

macro_rules! define_complex {
    ($s: ty, $d: ty) => {
        impl ComplexDispatch<$s, $d> for $s {
            fn column_dispatch(
                _: bool,
            ) -> fn(Arena, &[&[Complex<$d>]], &mut [$s], ImageSize, &[Complex<$d>]) {
                filter_column_complex
            }

            fn row_dispatch(
                _: bool,
            ) -> fn(Arena, &[$s], &mut [Complex<$d>], ImageSize, &[Complex<$d>]) {
                filter_row_complex
            }
        }
    };
}

define_complex!(i8, f32);
define_complex!(i8, f64);
define_complex!(u8, f64);
define_complex!(u8, u16);
define_complex!(u8, i32);
define_complex!(u8, u32);
define_complex!(u16, f64);
define_complex!(i16, f32);
define_complex!(i16, f64);
define_complex!(u32, f32);
define_complex!(u32, f64);
define_complex!(i32, f32);
define_complex!(i32, f64);
define_complex!(f32, f64);
define_complex!(f64, f64);
