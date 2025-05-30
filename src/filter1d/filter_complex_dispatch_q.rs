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
use crate::{Arena, ImageSize};
use num_complex::Complex;

pub trait ComplexDispatchQ<T, I> {
    fn column_dispatch(
        is_symmetric_kernel: bool,
    ) -> fn(
        arena: Arena,
        arena_src: &[&[Complex<I>]],
        dst: &mut [T],
        image_size: ImageSize,
        kernel: &[Complex<i16>],
    );

    fn row_dispatch(
        is_symmetric_kernel: bool,
    ) -> fn(
        arena: Arena,
        arena_src: &[T],
        dst: &mut [Complex<I>],
        image_size: ImageSize,
        kernel: &[Complex<i16>],
    );
}

impl ComplexDispatchQ<u8, i16> for u8 {
    fn column_dispatch(
        _: bool,
    ) -> fn(Arena, &[&[Complex<i16>]], &mut [u8], ImageSize, &[Complex<i16>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter1d::neon::filter_column_complex_u8_i32;
            filter_column_complex_u8_i32
        }
        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_avx_column_complex_u8_i32;
                return filter_avx_column_complex_u8_i32;
            }
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_column_complex_u8_i32;
                return filter_sse_column_complex_u8_i32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::filter1d::filter_column_complex_q::filter_column_complex_q;
            filter_column_complex_q
        }
    }

    fn row_dispatch(_: bool) -> fn(Arena, &[u8], &mut [Complex<i16>], ImageSize, &[Complex<i16>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter1d::neon::filter_row_complex_u8_i32_q;
            filter_row_complex_u8_i32_q
        }
        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_avx_row_complex_u8_i32;
                return filter_avx_row_complex_u8_i32;
            }
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_row_complex_u8_i32;
                return filter_sse_row_complex_u8_i32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::filter1d::filter_row_complex_q::filter_row_complex_q;
            filter_row_complex_q
        }
    }
}

impl ComplexDispatchQ<u16, i32> for u16 {
    fn column_dispatch(
        _: bool,
    ) -> fn(Arena, &[&[Complex<i32>]], &mut [u16], ImageSize, &[Complex<i16>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter1d::neon::filter_column_complex_u16_i32;
            filter_column_complex_u16_i32
        }
        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_avx_column_complex_u16_i32;
                return filter_avx_column_complex_u16_i32;
            }
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_column_complex_u16_i32;
                return filter_sse_column_complex_u16_i32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::filter1d::filter_column_complex_q::filter_column_complex_q;
            filter_column_complex_q
        }
    }

    fn row_dispatch(_: bool) -> fn(Arena, &[u16], &mut [Complex<i32>], ImageSize, &[Complex<i16>]) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter1d::neon::filter_row_complex_u16_i32_q;
            filter_row_complex_u16_i32_q
        }
        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::filter1d::avx::filter_avx_row_complex_u16_i32;
                return filter_avx_row_complex_u16_i32;
            }
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::filter1d::sse::filter_sse_row_complex_u16_i32;
                return filter_sse_row_complex_u16_i32;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::filter1d::filter_row_complex_q::filter_row_complex_q;
            filter_row_complex_q
        }
    }
}
