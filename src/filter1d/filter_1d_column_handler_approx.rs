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
use crate::ImageSize;
use crate::filter1d::arena::Arena;
use crate::filter1d::filter_1d_column_handler::FilterBrows;
use crate::filter1d::filter_column_approx::filter_column_approx;
use crate::filter1d::filter_column_approx_symmetric::filter_column_symmetric_approx;
use crate::filter1d::region::FilterRegion;
use std::sync::Arc;

pub trait ResolveColumnHandlerApprox<T>: Send + Sync {
    fn single_row(
        &self,
        arena: Arena,
        rows: &[&[T]],
        dst: &mut [T],
        image_size: ImageSize,
        region: FilterRegion,
    );

    fn multiple_rows(
        &self,
        arena: Arena,
        rows: FilterBrows<T>,
        dst: &mut [T],
        image_size: ImageSize,
        dst_stride: usize,
    ) -> bool; // returns false if no multi-row impl available
    fn has_multiple_rows(&self) -> bool;
}

#[allow(unused)]
pub(crate) fn make_q07_weights(weights: &[i32]) -> Vec<i8> {
    let mut shifted = weights
        .iter()
        .map(|&x| (x >> 8).min(i8::MAX as i32) as i8)
        .collect::<Vec<_>>();
    let mut sum: u32 = shifted.iter().map(|&x| x as u32).sum();
    if sum > 128 {
        let half = shifted.len() / 2;
        while sum > 128 {
            shifted[half] = shifted[half].saturating_sub(1);
            sum -= 1;
        }
    } else if sum < 128 {
        let half = shifted.len() / 2;
        while sum < 128 {
            shifted[half] = shifted[half].saturating_add(1);
            sum += 1;
        }
    }
    shifted
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[allow(unused)]
pub(crate) fn make_uq07_weights(weights: &[i32]) -> Vec<u8> {
    let mut shifted = weights
        .iter()
        .map(|&x| (x >> 8).min(i8::MAX as i32) as u8)
        .collect::<Vec<_>>();
    let mut sum: u32 = shifted.iter().map(|&x| x as u32).sum();
    if sum > 128 {
        let half = shifted.len() / 2;
        while sum > 128 {
            shifted[half] = shifted[half].saturating_sub(1);
            sum -= 1;
        }
    } else if sum < 128 {
        let half = shifted.len() / 2;
        while sum < 128 {
            shifted[half] = shifted[half].saturating_add(1);
            sum += 1;
        }
    }
    shifted
}

struct ColumnHandlerApprox<T, F> {
    single_row: fn(Arena, &[&[T]], &mut [T], ImageSize, FilterRegion, &[F]),
    multiple_rows: Option<fn(Arena, FilterBrows<T>, &mut [T], ImageSize, usize, &[F])>,
    kernel: Vec<F>,
}

impl<T: Send + Sync, F: Send + Sync> ResolveColumnHandlerApprox<T> for ColumnHandlerApprox<T, F> {
    fn single_row(
        &self,
        arena: Arena,
        rows: &[&[T]],
        dst: &mut [T],
        image_size: ImageSize,
        region: FilterRegion,
    ) {
        (self.single_row)(arena, rows, dst, image_size, region, &self.kernel)
    }

    fn multiple_rows(
        &self,
        arena: Arena,
        rows: FilterBrows<T>,
        dst: &mut [T],
        image_size: ImageSize,
        dst_stride: usize,
    ) -> bool {
        match self.multiple_rows {
            Some(f) => {
                f(arena, rows, dst, image_size, dst_stride, &self.kernel);
                true
            }
            None => false,
        }
    }

    fn has_multiple_rows(&self) -> bool {
        self.multiple_rows.is_some()
    }
}

pub trait BuildColumnHandlerApprox<T, F> {
    fn build_column_handler(
        is_kernel_symmetric: bool,
        kernel: &[F],
    ) -> Arc<dyn ResolveColumnHandlerApprox<T> + Send + Sync>;
}

macro_rules! default_1d_column_handler {
    ($store:ty, $intermediate:ty) => {
        impl BuildColumnHandlerApprox<$store, $intermediate> for $store {
            fn build_column_handler(
                is_kernel_symmetric: bool,
                kernel: &[$intermediate],
            ) -> Arc<dyn ResolveColumnHandlerApprox<$store> + Send + Sync> {
                Arc::new(ColumnHandlerApprox {
                    single_row: if is_kernel_symmetric {
                        filter_column_symmetric_approx
                    } else {
                        filter_column_approx
                    },
                    multiple_rows: None,
                    kernel: kernel.to_vec(),
                })
            }
        }
    };
}

impl BuildColumnHandlerApprox<u8, i32> for u8 {
    fn build_column_handler(
        is_kernel_symmetric: bool,
        kernel: &[i32],
    ) -> Arc<dyn ResolveColumnHandlerApprox<u8> + Send + Sync> {
        let _single_row: fn(Arena, &[&[u8]], &mut [u8], ImageSize, FilterRegion, &[i32]);
        let _multiple_rows: Option<
            fn(Arena, FilterBrows<u8>, &mut [u8], ImageSize, usize, &[i32]),
        >;

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if is_kernel_symmetric && std::arch::is_x86_feature_detected!("avx512bw") {
            use crate::filter1d::avx512::filter_column_avx512_symm_u8_i32_app;
            let single_row = filter_column_avx512_symm_u8_i32_app;
            if is_kernel_symmetric && kernel.len() <= 7 && kernel.iter().all(|&x| x > 0) {
                let q07_weights = make_q07_weights(kernel);
                use crate::filter1d::avx512::filter_column_avx512_symm_u8_uq0_7;
                let v_prepared = q07_weights
                    .iter()
                    .map(|&x| {
                        let z = x.to_ne_bytes();
                        i32::from_ne_bytes([z[0], z[0], z[0], z[0]])
                    })
                    .collect::<Vec<_>>();
                return Arc::new(ColumnHandlerApprox {
                    single_row: filter_column_avx512_symm_u8_uq0_7,
                    multiple_rows: None,
                    kernel: v_prepared,
                });
            }
            let v_packed_i16 = kernel
                .iter()
                .map(|&x| {
                    let z = x.to_ne_bytes();
                    i32::from_ne_bytes([z[0], z[1], z[0], z[1]])
                })
                .collect::<Vec<_>>();
            return Arc::new(ColumnHandlerApprox {
                single_row,
                multiple_rows: None,
                kernel: v_packed_i16,
            });
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::filter1d::avx::{
                filter_column_avx_symm_u8_i32_app, filter_column_avx_symm_u8_i32_app_x2,
                filter_column_avx_u8_i32_app,
            };
            if is_kernel_symmetric && kernel.len() <= 7 && kernel.iter().all(|&x| x > 0) {
                let q07_weights = make_q07_weights(kernel);
                let v_prepared = q07_weights
                    .iter()
                    .map(|&x| {
                        let z = x.to_ne_bytes();
                        i32::from_ne_bytes([z[0], z[0], z[0], z[0]])
                    })
                    .collect::<Vec<_>>();
                use crate::filter1d::avx::filter_column_avx_symm_u8_uq0_7;
                return Arc::new(ColumnHandlerApprox {
                    single_row: filter_column_avx_symm_u8_uq0_7,
                    multiple_rows: None,
                    kernel: v_prepared,
                });
            }
            let v_packed_i16 = kernel
                .iter()
                .map(|&x| {
                    let z = x.to_ne_bytes();
                    i32::from_ne_bytes([z[0], z[1], z[0], z[1]])
                })
                .collect::<Vec<_>>();
            if is_kernel_symmetric {
                _single_row = filter_column_avx_symm_u8_i32_app;
                _multiple_rows = Some(filter_column_avx_symm_u8_i32_app_x2);
            } else {
                _single_row = filter_column_avx_u8_i32_app;
                _multiple_rows = None;
            }
            return Arc::new(ColumnHandlerApprox {
                single_row: _single_row,
                multiple_rows: _multiple_rows,
                kernel: v_packed_i16,
            });
        }

        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            use crate::filter1d::sse::{
                filter_column_sse_symm_u8_uq0_7, filter_column_sse_u8_i32_app,
                filter_column_symm_u8_i32_app,
            };
            if is_kernel_symmetric && kernel.len() <= 7 && kernel.iter().all(|&x| x > 0) {
                let q07_weights = make_q07_weights(kernel);
                let v_prepared = q07_weights
                    .iter()
                    .map(|&x| {
                        let z = x.to_ne_bytes();
                        i32::from_ne_bytes([z[0], z[0], z[0], z[0]])
                    })
                    .collect::<Vec<_>>();
                return Arc::new(ColumnHandlerApprox {
                    single_row: filter_column_sse_symm_u8_uq0_7,
                    multiple_rows: None,
                    kernel: v_prepared,
                });
            }
            let v_packed_i16 = kernel
                .iter()
                .map(|&x| {
                    let z = x.to_ne_bytes();
                    i32::from_ne_bytes([z[0], z[1], z[0], z[1]])
                })
                .collect::<Vec<_>>();
            let single_row = if is_kernel_symmetric {
                filter_column_symm_u8_i32_app
            } else {
                filter_column_sse_u8_i32_app
            };
            return Arc::new(ColumnHandlerApprox {
                single_row,
                multiple_rows: None,
                kernel: v_packed_i16,
            });
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter1d::neon::{
                filter_column_neon_u8_i32_app, filter_column_symm_neon_u8_i32_app,
            };
            if is_kernel_symmetric {
                if kernel.len() <= 7 && kernel.iter().all(|&x| x > 0) {
                    let q07_weights = make_uq07_weights(kernel);
                    use crate::filter1d::neon::filter_column_symm_neon_u8_uq0_7;
                    return Arc::new(ColumnHandlerApprox {
                        single_row: filter_column_symm_neon_u8_uq0_7,
                        multiple_rows: None,
                        kernel: q07_weights,
                    });
                }

                #[cfg(feature = "rdm")]
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    use crate::filter1d::neon::{
                        filter_column_symm_neon_u8_i32_rdm, filter_column_symm_neon_u8_i32_rdm_x3,
                    };
                    return Arc::new(ColumnHandlerApprox {
                        single_row: filter_column_symm_neon_u8_i32_rdm,
                        multiple_rows: Some(filter_column_symm_neon_u8_i32_rdm_x3),
                        kernel: kernel.to_vec(),
                    });
                }
                Arc::new(ColumnHandlerApprox {
                    single_row: filter_column_symm_neon_u8_i32_app,
                    multiple_rows: None,
                    kernel: kernel.to_vec(),
                })
            } else {
                #[cfg(feature = "rdm")]
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    use crate::filter1d::neon::filter_column_neon_u8_i32_i16_qrdm_app;
                    return Arc::new(ColumnHandlerApprox {
                        single_row: filter_column_neon_u8_i32_i16_qrdm_app,
                        multiple_rows: None,
                        kernel: kernel.to_vec(),
                    });
                }
                Arc::new(ColumnHandlerApprox {
                    single_row: filter_column_neon_u8_i32_app,
                    multiple_rows: None,
                    kernel: kernel.to_vec(),
                })
            }
        }

        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        Arc::new(ColumnHandlerApprox {
            single_row: if is_kernel_symmetric {
                filter_column_symmetric_approx
            } else {
                filter_column_approx
            },
            multiple_rows: None,
            kernel: kernel.to_vec(),
        })
    }
}

impl BuildColumnHandlerApprox<u16, u32> for u16 {
    fn build_column_handler(
        is_kernel_symmetric: bool,
        kernel: &[u32],
    ) -> Arc<dyn ResolveColumnHandlerApprox<u16> + Send + Sync> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") && is_kernel_symmetric {
            use crate::filter1d::avx::{
                filter_column_avx_symm_uq15_u16, filter_column_avx_symm_uq15_u16_x2,
            };
            return Arc::new(ColumnHandlerApprox {
                single_row: filter_column_avx_symm_uq15_u16,
                multiple_rows: Some(filter_column_avx_symm_uq15_u16_x2),
                kernel: kernel.to_vec(),
            });
        }

        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") && is_kernel_symmetric {
            use crate::filter1d::sse::filter_column_sse_symm_uq15_u16;
            return Arc::new(ColumnHandlerApprox {
                single_row: filter_column_sse_symm_uq15_u16,
                multiple_rows: None,
                kernel: kernel.to_vec(),
            });
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if is_kernel_symmetric {
            use crate::filter1d::neon::{
                filter_column_symm_neon_uq15_u16, filter_symm_column_neon_uq15_u16_x3,
            };
            return Arc::new(ColumnHandlerApprox {
                single_row: filter_column_symm_neon_uq15_u16,
                multiple_rows: Some(filter_symm_column_neon_uq15_u16_x3),
                kernel: kernel.to_vec(),
            });
        }

        Arc::new(ColumnHandlerApprox {
            single_row: if is_kernel_symmetric {
                filter_column_symmetric_approx
            } else {
                filter_column_approx
            },
            multiple_rows: None,
            kernel: kernel.to_vec(),
        })
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
