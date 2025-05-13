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
use crate::edge_mode::BorderHandle;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::ImageSize;

#[allow(dead_code)]
pub struct RowsHolder<'a, T> {
    pub(crate) holder: Vec<&'a [T]>,
}

#[allow(dead_code)]
pub struct RowsHolderMut<'a, T> {
    pub(crate) holder: Vec<&'a mut [T]>,
}

type BInterpolateHandler<T, F> = fn(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<T>,
    m_dst: &mut RowsHolderMut<T>,
    image_size: ImageSize,
    scanned_kernel: &[ScanPoint1d<F>],
);

pub trait Filter1DRowHandlerBInterpolateApr<T, F> {
    fn get_row_handler_binter_apr<const N: usize>(
        is_kernel_symmetric: bool,
        kernel: &[ScanPoint1d<F>],
    ) -> Option<BInterpolateHandler<T, F>>;
}

impl Filter1DRowHandlerBInterpolateApr<u8, i32> for u8 {
    fn get_row_handler_binter_apr<const N: usize>(
        is_kernel_symmetric: bool,
        _kernel: &[ScanPoint1d<i32>],
    ) -> Option<BInterpolateHandler<u8, i32>> {
        if is_kernel_symmetric {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    if _kernel.len() == 5 {
                        let all_positive = _kernel.iter().all(|&x| x.weight > 0);
                        if all_positive {
                            use crate::filter1d::avx::filter_row_avx_symm_u8_uq0_7_k5;
                            return Some(filter_row_avx_symm_u8_uq0_7_k5::<N>);
                        }
                    }
                    if _kernel.len() < 9 {
                        let all_positive = _kernel.iter().all(|&x| x.weight > 0);
                        if all_positive {
                            use crate::filter1d::avx::filter_row_avx_symm_u8_uq0_7_any;
                            return Some(filter_row_avx_symm_u8_uq0_7_any::<N>);
                        }
                    }
                    use crate::filter1d::avx::filter_row_avx_symm_u8_i32_app_binter;
                    return Some(filter_row_avx_symm_u8_i32_app_binter::<N>);
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                if _kernel.len() == 5 {
                    let all_positive = _kernel.iter().all(|&x| x.weight > 0);
                    if all_positive {
                        use crate::filter1d::neon::filter_row_symm_neon_binter_u8_uq0_7_x5;
                        return Some(filter_row_symm_neon_binter_u8_uq0_7_x5::<N>);
                    }
                }
                if _kernel.len() < 9 {
                    let all_positive = _kernel.iter().all(|&x| x.weight > 0);
                    if all_positive {
                        use crate::filter1d::neon::filter_row_symm_neon_binter_u8_u0_7;
                        return Some(filter_row_symm_neon_binter_u8_u0_7::<N>);
                    }
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "rdm"))]
            {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    use crate::filter1d::neon::filter_row_symm_neon_binter_u8_i32_rdm;
                    return Some(filter_row_symm_neon_binter_u8_i32_rdm::<N>);
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::filter1d::neon::filter_row_symm_neon_binter_u8_i32;
                return Some(filter_row_symm_neon_binter_u8_i32::<N>);
            }
        }
        None
    }
}

macro_rules! d_binter {
    ($store:ty, $intermediate:ty) => {
        impl Filter1DRowHandlerBInterpolateApr<$store, $intermediate> for $store {
            fn get_row_handler_binter_apr<const N: usize>(
                _: bool,
                _: &[ScanPoint1d<$intermediate>],
            ) -> Option<BInterpolateHandler<$store, $intermediate>> {
                None
            }
        }
    };
}

d_binter!(u8, i64);
d_binter!(u8, u16);
d_binter!(u8, i16);
d_binter!(u8, u32);
d_binter!(u8, u64);
d_binter!(i8, i32);
d_binter!(i8, i64);
d_binter!(i8, i16);
d_binter!(u16, u32);
d_binter!(u16, i32);
d_binter!(u16, i64);
d_binter!(u16, u64);
d_binter!(i16, i32);
d_binter!(i16, i64);
