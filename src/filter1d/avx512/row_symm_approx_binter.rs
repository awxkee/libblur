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
use crate::edge_mode::border_interpolate;
use crate::filter1d::avx512::sse_utils::*;
use crate::filter1d::avx512::utils::{
    _mm256_mul_add_symm_epi8_by_epi16_x4, _mm256_mul_epi8_by_epi16_x4, _mm256_pack_epi32_x4_epi8,
    _mm512_mul_add_symm_epi8_by_epi16_x4, _mm512_mul_epi8_by_epi16_x4, _mm512_pack_epi32_x4_epi8,
};
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::row_handler_small_approx::{RowsHolder, RowsHolderMut};
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use crate::BorderHandle;
use std::arch::x86_64::*;

pub(crate) fn filter_row_avx512_symm_u8_i32_app_binter<const N: usize>(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<u8>,
    m_dst: &mut RowsHolderMut<u8>,
    image_size: ImageSize,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let unit = ExecutionUnit::<N>::default();
        unit.pass(edge_mode, m_src, m_dst, image_size, scanned_kernel);
    }
}

#[derive(Default, Copy, Clone)]
struct ExecutionUnit<const N: usize> {}

impl<const N: usize> ExecutionUnit<N> {
    #[target_feature(enable = "avx512bw")]
    unsafe fn pass(
        &self,
        edge_mode: BorderHandle,
        m_src: &RowsHolder<u8>,
        m_dst: &mut RowsHolderMut<u8>,
        image_size: ImageSize,
        scanned_kernel: &[ScanPoint1d<i32>],
    ) {
        let width = image_size.width;

        let v_prepared = scanned_kernel
            .iter()
            .map(|&x| {
                let z = x.weight.to_ne_bytes();
                i32::from_ne_bytes([z[0], z[1], z[0], z[1]])
            })
            .collect::<Vec<_>>();

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let min_left = half_len.min(width);
        let s_kernel = half_len as i64;

        for (src, dst) in m_src.holder.iter().zip(m_dst.holder.iter_mut()) {
            let dst = &mut **dst;
            let mut f_cx = 0usize;

            while f_cx < min_left {
                for c in 0..N {
                    let coeff = *scanned_kernel.get_unchecked(half_len);
                    let mx = f_cx as i64 - s_kernel;
                    let mut k0: i32 = *src.get_unchecked(f_cx * N + c) as i32 * coeff.weight;

                    for i in 0..half_len {
                        let coeff = *scanned_kernel.get_unchecked(i);
                        let rollback = length - i - 1;

                        let src0 = border_interpolate!(
                            src,
                            edge_mode,
                            i as i64 + mx,
                            0,
                            width as i64,
                            N,
                            c
                        );
                        let src1 = border_interpolate!(
                            src,
                            edge_mode,
                            rollback as i64 + mx,
                            0,
                            width as i64,
                            N,
                            c
                        );

                        k0 += (src0 as i32 + src1 as i32) * coeff.weight;
                    }

                    *dst.get_unchecked_mut(f_cx * N + c) = k0.to_approx_();
                }
                f_cx += 1;
            }

            let mut m_cx = f_cx * N;

            let s_half = half_len * N;
            let m_right = width.saturating_sub(half_len);
            let max_width = m_right * N;

            let coeff = _mm512_set1_epi32(*v_prepared.get_unchecked(half_len));

            while m_cx + 64 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source = _mm512_loadu_si512(
                    shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _
                );
                let mut k0 = _mm512_mul_epi8_by_epi16_x4(source, coeff);

                for i in 0..half_len {
                    let rollback = length - i - 1;
                    let coeff = _mm512_set1_epi32(*v_prepared.get_unchecked(i));
                    let v_source0 = _mm512_loadu_si512(
                        shifted_src.get_unchecked((i * N)..).as_ptr() as *const _,
                    );
                    let v_source1 = _mm512_loadu_si512(
                        shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                    );
                    k0 = _mm512_mul_add_symm_epi8_by_epi16_x4(k0, v_source0, v_source1, coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm512_storeu_si512(dst_ptr0 as *mut _, _mm512_pack_epi32_x4_epi8(k0));
                m_cx += 64;
            }

            let coeff = _mm512_castsi512_si256(coeff);

            while m_cx + 32 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source = _mm256_loadu_si256(
                    shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _
                );
                let mut k0 = _mm256_mul_epi8_by_epi16_x4(source, coeff);

                for i in 0..half_len {
                    let rollback = length - i - 1;
                    let coeff = _mm256_set1_epi32(*v_prepared.get_unchecked(i));
                    let v_source0 = _mm256_loadu_si256(
                        shifted_src.get_unchecked((i * N)..).as_ptr() as *const _,
                    );
                    let v_source1 = _mm256_loadu_si256(
                        shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                    );
                    k0 = _mm256_mul_add_symm_epi8_by_epi16_x4(k0, v_source0, v_source1, coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm256_storeu_si256(dst_ptr0 as *mut _, _mm256_pack_epi32_x4_epi8(k0));
                m_cx += 32;
            }

            while m_cx + 16 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source =
                    _mm_loadu_si128(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
                let mut k0 = _mm_mul_epi8_by_epi16_x4(source, _mm256_castsi256_si128(coeff));

                for i in 0..half_len {
                    let rollback = length - i - 1;
                    let coeff = _mm_set1_epi32(*v_prepared.get_unchecked(i));
                    let v_source0 =
                        _mm_loadu_si128(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                    let v_source1 = _mm_loadu_si128(
                        shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                    );
                    k0 = _mm_mul_add_symm_epi8_by_epi16_x4(k0, v_source0, v_source1, coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm_storeu_si128(dst_ptr0 as *mut _, _mm_pack_epi32_x2_epi8(k0));
                m_cx += 16;
            }

            while m_cx + 8 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source =
                    _mm_loadu_si64(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
                let mut k0 = _mm_mul_epi8_by_epi16_x2(source, _mm256_castsi256_si128(coeff));

                for i in 0..half_len {
                    let rollback = length - i - 1;
                    let coeff = _mm_set1_epi32(*v_prepared.get_unchecked(i));
                    let v_source0 =
                        _mm_loadu_si64(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                    let v_source1 = _mm_loadu_si64(
                        shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                    );
                    k0 = _mm_mul_add_symm_epi8_by_epi16_x2(k0, v_source0, v_source1, coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm_storeu_si64(dst_ptr0 as *mut _, _mm_pack_epi32_epi8(k0));
                m_cx += 8;
            }

            while m_cx + 4 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source =
                    _mm_loadu_si32(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
                let mut k0 = _mm_mul_epi8_by_epi16_x2(source, _mm256_castsi256_si128(coeff));

                for i in 0..half_len {
                    let rollback = length - i - 1;
                    let coeff = _mm_set1_epi32(*v_prepared.get_unchecked(i));
                    let v_source0 =
                        _mm_loadu_si32(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                    let v_source1 = _mm_loadu_si32(
                        shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                    );
                    k0 = _mm_mul_add_symm_epi8_by_epi16_x2(k0, v_source0, v_source1, coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm_storeu_si32(dst_ptr0 as *mut _, _mm_pack_epi32_epi8(k0));
                m_cx += 4;
            }

            let coeff = *scanned_kernel.get_unchecked(half_len);

            for zx in m_cx..max_width {
                let x = zx - s_half;

                let shifted_src = src.get_unchecked(x..);
                let mut k0 = *shifted_src.get_unchecked(half_len * N) as i32 * coeff.weight;

                for i in 0..half_len {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    let rollback = length - i - 1;

                    k0 += (*shifted_src.get_unchecked(i * N) as i16
                        + *shifted_src.get_unchecked(rollback * N) as i16)
                        as i32
                        * coeff.weight;
                }

                *dst.get_unchecked_mut(zx) = k0.to_approx_();
            }

            f_cx = m_right;

            while f_cx < width {
                for c in 0..N {
                    let mx = f_cx as i64 - s_kernel;
                    let mut k0 = *src.get_unchecked(f_cx * N + c) as i32 * coeff.weight;

                    for i in 0..half_len {
                        let coeff = *scanned_kernel.get_unchecked(i);
                        let rollback = length - i - 1;

                        let src0 = border_interpolate!(
                            src,
                            edge_mode,
                            i as i64 + mx,
                            0,
                            width as i64,
                            N,
                            c
                        );
                        let src1 = border_interpolate!(
                            src,
                            edge_mode,
                            rollback as i64 + mx,
                            0,
                            width as i64,
                            N,
                            c
                        );

                        k0 += (src0 as i32 + src1 as i32) * coeff.weight;
                    }

                    *dst.get_unchecked_mut(f_cx * N + c) = k0.to_approx_();
                }
                f_cx += 1;
            }
        }
    }
}
