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
use crate::filter1d::filter_row_approx::filter_row_approx;
use crate::filter1d::filter_row_symmetric_approx::filter_row_symmetric_approx;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::ImageSize;

pub trait Filter1DRgbRowHandlerApprox<T, F> {
    fn get_rgb_row_handler_apr(
        is_symmetrical_kernel: bool,
    ) -> fn(
        arena: Arena,
        arena_src: &[T],
        dst: &mut [T],
        image_size: ImageSize,
        filter_region: FilterRegion,
        scanned_kernel: &[ScanPoint1d<F>],
    );
}

macro_rules! default_1d_row_handler {
    ($store:ty, $intermediate:ty) => {
        impl Filter1DRgbRowHandlerApprox<$store, $intermediate> for $store {
            fn get_rgb_row_handler_apr(
                is_symmetrical_kernel: bool,
            ) -> fn(
                Arena,
                &[$store],
                &mut [$store],
                ImageSize,
                FilterRegion,
                &[ScanPoint1d<$intermediate>],
            ) {
                if is_symmetrical_kernel {
                    filter_row_symmetric_approx::<$store, $intermediate, 3>
                } else {
                    filter_row_approx::<$store, $intermediate, 3>
                }
            }
        }
    };
}

impl Filter1DRgbRowHandlerApprox<u8, i32> for u8 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn get_rgb_row_handler_apr(
        is_symmetrical_kernel: bool,
    ) -> fn(Arena, &[u8], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<i32>]) {
        if is_symmetrical_kernel {
            filter_row_symmetric_approx::<u8, i32, 3>
        } else {
            filter_row_approx::<u8, i32, 3>
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn get_rgb_row_handler_apr(
        is_symmetrical_kernel: bool,
    ) -> fn(Arena, &[u8], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<i32>]) {
        if is_symmetrical_kernel {
            #[cfg(feature = "rdm")]
            {
                use crate::cpu_features::is_aarch_rdm_supported;
                if is_aarch_rdm_supported() {
                    use crate::filter1d::neon::filter_row_symm_neon_u8_i32_rdm;
                    return filter_row_symm_neon_u8_i32_rdm::<3>;
                }
            }
            use crate::filter1d::neon::filter_row_symm_neon_u8_i32;
            filter_row_symm_neon_u8_i32::<3>
        } else {
            use crate::filter1d::neon::filter_row_neon_u8_i32_app;
            filter_row_neon_u8_i32_app::<3>
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn get_rgb_row_handler_apr(
        is_symmetrical_kernel: bool,
    ) -> fn(Arena, &[u8], &mut [u8], ImageSize, FilterRegion, &[ScanPoint1d<i32>]) {
        #[cfg(feature = "avx")]
        if std::arch::is_x86_feature_detected!("avx2") {
            return if is_symmetrical_kernel {
                use crate::filter1d::avx::filter_row_avx_symm_u8_i32_app;
                filter_row_avx_symm_u8_i32_app::<3>
            } else {
                use crate::filter1d::avx::filter_row_avx_u8_i32_app;
                filter_row_avx_u8_i32_app::<3>
            };
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            if is_symmetrical_kernel {
                use crate::filter1d::sse::filter_row_symm_sse_u8_i32_app;
                return filter_row_symm_sse_u8_i32_app::<3>;
            }
            use crate::filter1d::sse::filter_row_sse_u8_i32;
            return filter_row_sse_u8_i32::<3>;
        }
        if is_symmetrical_kernel {
            filter_row_symmetric_approx::<u8, i32, 3>
        } else {
            filter_row_approx::<u8, i32, 3>
        }
    }
}

default_1d_row_handler!(u8, i64);
default_1d_row_handler!(u8, u16);
default_1d_row_handler!(u8, i16);
default_1d_row_handler!(u8, u32);
default_1d_row_handler!(u8, u64);
default_1d_row_handler!(i8, i32);
default_1d_row_handler!(i8, i64);
default_1d_row_handler!(i8, i16);
default_1d_row_handler!(u16, u32);
default_1d_row_handler!(u16, i32);
default_1d_row_handler!(u16, i64);
default_1d_row_handler!(u16, u64);
default_1d_row_handler!(i16, i32);
default_1d_row_handler!(i16, i64);
