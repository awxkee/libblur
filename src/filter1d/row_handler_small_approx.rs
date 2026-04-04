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
use crate::ImageSize;
use crate::edge_mode::BorderHandle;
use std::sync::Arc;

#[allow(dead_code)]
pub struct RowsHolder<'a, T> {
    pub(crate) holder: [&'a [T]; 1],
}

#[allow(dead_code)]
pub struct RowsHolderMut<'a, T> {
    pub(crate) holder: [&'a mut [T]; 1],
}

type BInterpolateHandler<T, F> = fn(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<T>,
    m_dst: &mut RowsHolderMut<T>,
    image_size: ImageSize,
    scanned_kernel: &[F],
);

pub trait ResolveRowHandlerBInter<T>: Send + Sync {
    fn handle_row(
        &self,
        edge_mode: BorderHandle,
        src: &RowsHolder<T>,
        dst: &mut RowsHolderMut<T>,
        image_size: ImageSize,
    );
}

struct RowHandlerBInter<T, F> {
    handler: BInterpolateHandler<T, F>,
    kernel: Vec<F>,
}

impl<T: Send + Sync, F: Send + Sync> ResolveRowHandlerBInter<T> for RowHandlerBInter<T, F> {
    fn handle_row(
        &self,
        edge_mode: BorderHandle,
        src: &RowsHolder<T>,
        dst: &mut RowsHolderMut<T>,
        image_size: ImageSize,
    ) {
        (self.handler)(edge_mode, src, dst, image_size, &self.kernel)
    }
}

pub trait BuildRowHandlerBInter<T, F> {
    fn build_row_handler_binter<const N: usize>(
        is_kernel_symmetric: bool,
        kernel: &[F],
    ) -> Option<Arc<dyn ResolveRowHandlerBInter<T> + Send + Sync>>;
}

// ── helper to wrap a found handler ───────────────────────────────────────────
#[allow(unused)]
fn wrap_binter<T: Send + Sync + 'static, F: Send + Sync + Clone + 'static>(
    handler: BInterpolateHandler<T, F>,
    kernel: Vec<F>,
) -> Option<Arc<dyn ResolveRowHandlerBInter<T> + Send + Sync>> {
    Some(Arc::new(RowHandlerBInter { handler, kernel }))
}

// ── u8 / i32 ─────────────────────────────────────────────────────────────────
impl BuildRowHandlerBInter<u8, i32> for u8 {
    fn build_row_handler_binter<const N: usize>(
        is_kernel_symmetric: bool,
        kernel: &[i32],
    ) -> Option<Arc<dyn ResolveRowHandlerBInter<u8> + Send + Sync>> {
        if !is_kernel_symmetric {
            return None;
        }

        let _all_positive = kernel.iter().all(|&x| x > 0);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if std::arch::is_x86_feature_detected!("avx512bw") {
            if kernel.len() == 3 && _all_positive {
                use crate::filter1d::avx512::filter_row_avx512_symm_u8_uq0_7_k3;
                use crate::filter1d::filter_1d_column_handler_approx::make_q07_weights;
                return wrap_binter(
                    filter_row_avx512_symm_u8_uq0_7_k3::<N>,
                    make_q07_weights(kernel),
                );
            }
            use crate::filter1d::avx512::filter_row_avx512_symm_u8_i32_app_binter;

            let v_prepared = kernel
                .iter()
                .map(|&x| {
                    let z = x.to_ne_bytes();
                    i32::from_ne_bytes([z[0], z[1], z[0], z[1]])
                })
                .collect::<Vec<_>>();
            return wrap_binter(filter_row_avx512_symm_u8_i32_app_binter::<N>, v_prepared);
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            if kernel.len() == 3 && _all_positive {
                use crate::filter1d::avx::filter_row_avx_symm_u8_uq0_7_k3;
                use crate::filter1d::filter_1d_column_handler_approx::make_q07_weights;
                return wrap_binter(
                    filter_row_avx_symm_u8_uq0_7_k3::<N>,
                    make_q07_weights(kernel),
                );
            }
            if kernel.len() == 5 && _all_positive {
                use crate::filter1d::avx::filter_row_avx_symm_u8_uq0_7_k5;
                use crate::filter1d::filter_1d_column_handler_approx::make_q07_weights;
                return wrap_binter(
                    filter_row_avx_symm_u8_uq0_7_k5::<N>,
                    make_q07_weights(kernel),
                );
            }
            if kernel.len() < 9 && _all_positive {
                use crate::filter1d::avx::filter_row_avx_symm_u8_uq0_7_any;
                use crate::filter1d::filter_1d_column_handler_approx::make_q07_weights;
                return wrap_binter(
                    filter_row_avx_symm_u8_uq0_7_any::<N>,
                    make_q07_weights(kernel),
                );
            }
            use crate::filter1d::avx::filter_row_avx_symm_u8_i32_app_binter;
            let v_prepared = kernel
                .iter()
                .map(|&x| {
                    let z = x.to_ne_bytes();
                    i32::from_ne_bytes([z[0], z[1], z[0], z[1]])
                })
                .collect::<Vec<_>>();
            return wrap_binter(filter_row_avx_symm_u8_i32_app_binter::<N>, v_prepared);
        }

        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") && kernel.len() < 9 && _all_positive {
            use crate::filter1d::filter_1d_column_handler_approx::make_q07_weights;
            use crate::filter1d::sse::filter_row_sse_symm_u8_uq0_7_any;
            let v_prepared = make_q07_weights(kernel)
                .iter()
                .map(|&x| {
                    let z = x.to_ne_bytes();
                    i32::from_ne_bytes([z[0], z[0], z[0], z[0]])
                })
                .collect::<Vec<_>>();

            return wrap_binter(filter_row_sse_symm_u8_uq0_7_any::<N>, v_prepared);
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter1d::filter_1d_column_handler_approx::make_uq07_weights;
            if kernel.len() == 3 && _all_positive {
                use crate::filter1d::neon::filter_row_symm_neon_binter_u8_uq0_7_k3;
                return wrap_binter(
                    filter_row_symm_neon_binter_u8_uq0_7_k3::<N>,
                    make_uq07_weights(kernel),
                );
            }
            if kernel.len() == 5 && _all_positive {
                use crate::filter1d::neon::filter_row_symm_neon_binter_u8_uq0_7_x5;
                return wrap_binter(
                    filter_row_symm_neon_binter_u8_uq0_7_x5::<N>,
                    make_uq07_weights(kernel),
                );
            }
            if kernel.len() < 9 && _all_positive {
                use crate::filter1d::neon::filter_row_symm_neon_binter_u8_u0_7;
                return wrap_binter(
                    filter_row_symm_neon_binter_u8_u0_7::<N>,
                    make_uq07_weights(kernel),
                );
            }
            #[cfg(feature = "rdm")]
            if std::arch::is_aarch64_feature_detected!("rdm") {
                use crate::filter1d::neon::filter_row_symm_neon_binter_u8_i32_rdm;
                return wrap_binter(filter_row_symm_neon_binter_u8_i32_rdm::<N>, kernel.to_vec());
            }
            use crate::filter1d::neon::filter_row_symm_neon_binter_u8_i32;
            wrap_binter(filter_row_symm_neon_binter_u8_i32::<N>, kernel.to_vec())
        }

        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        None
    }
}

macro_rules! d_binter {
    ($store:ty, $intermediate:ty) => {
        impl BuildRowHandlerBInter<$store, $intermediate> for $store {
            fn build_row_handler_binter<const N: usize>(
                _: bool,
                _: &[$intermediate],
            ) -> Option<Arc<dyn ResolveRowHandlerBInter<$store> + Send + Sync>> {
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
