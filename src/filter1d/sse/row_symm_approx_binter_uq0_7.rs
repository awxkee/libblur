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
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::row_handler_small_approx::{RowsHolder, RowsHolderMut};
use crate::img_size::ImageSize;
use crate::BorderHandle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn filter_row_sse_symm_u8_uq0_7_any<const N: usize>(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<u8>,
    m_dst: &mut RowsHolderMut<u8>,
    image_size: ImageSize,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let mut shifted = scanned_kernel
            .iter()
            .map(|&x| ((x.weight) >> 8).min(i8::MAX as i32) as i8)
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
        let unit = ExecutionUnit::<N>::default();
        unit.pass(edge_mode, m_src, m_dst, image_size, &shifted);
    }
}

#[derive(Default, Copy, Clone)]
struct ExecutionUnit<const N: usize> {}

impl<const N: usize> ExecutionUnit<N> {
    #[target_feature(enable = "sse4.1")]
    unsafe fn pass(
        &self,
        edge_mode: BorderHandle,
        m_src: &RowsHolder<u8>,
        m_dst: &mut RowsHolderMut<u8>,
        image_size: ImageSize,
        scanned_kernel: &[i8],
    ) {
        let width = image_size.width;

        let v_prepared = scanned_kernel
            .iter()
            .map(|&x| {
                let z = x.to_ne_bytes();
                i32::from_ne_bytes([z[0], z[0], z[0], z[0]])
            })
            .collect::<Vec<_>>();

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let min_left = half_len.min(width);
        let s_kernel = half_len as i64;

        for (src, dst) in m_src.holder.iter().zip(m_dst.holder.iter_mut()) {
            let dst = &mut **dst;
            let mut f_cx = 0usize;

            let coeff = *scanned_kernel.get_unchecked(half_len);

            while f_cx < min_left {
                for c in 0..N {
                    let mx = f_cx as i64 - s_kernel;
                    let mut k0: u16 = *src.get_unchecked(f_cx * N + c) as u16 * coeff as u16;

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

                        k0 += (src0 as u16 + src1 as u16) * coeff as u16;
                    }

                    *dst.get_unchecked_mut(f_cx * N + c) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
                }
                f_cx += 1;
            }

            let mut m_cx = f_cx * N;

            let s_half = half_len * N;
            let m_right = width.saturating_sub(half_len);
            let max_width = m_right * N;

            let coeff = _mm_set1_epi32(*v_prepared.get_unchecked(half_len));
            let rnd = _mm_set1_epi16(1 << 6);

            while m_cx + 16 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source =
                    _mm_loadu_si128(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
                let mut k0 = _mm_add_epi16(
                    rnd,
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(source, _mm_setzero_si128()), coeff),
                );
                let mut k1 = _mm_add_epi16(
                    rnd,
                    _mm_maddubs_epi16(_mm_unpackhi_epi8(source, _mm_setzero_si128()), coeff),
                );

                for i in 0..half_len {
                    let rollback = length - i - 1;
                    let coeff = _mm_set1_epi32(*v_prepared.get_unchecked(i));
                    let v_source0 =
                        _mm_loadu_si128(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                    let v_source1 = _mm_loadu_si128(
                        shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                    );
                    k0 = _mm_add_epi16(
                        k0,
                        _mm_maddubs_epi16(_mm_unpacklo_epi8(v_source0, v_source1), coeff),
                    );
                    k1 = _mm_add_epi16(
                        k1,
                        _mm_maddubs_epi16(_mm_unpackhi_epi8(v_source0, v_source1), coeff),
                    );
                }

                k0 = _mm_srai_epi16::<7>(k0);
                k1 = _mm_srai_epi16::<7>(k1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm_storeu_si128(dst_ptr0 as *mut __m128i, _mm_packus_epi16(k0, k1));
                m_cx += 16;
            }

            while m_cx + 8 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source =
                    _mm_loadu_si64(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
                let mut k0 = _mm_add_epi16(
                    rnd,
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(source, _mm_setzero_si128()), coeff),
                );

                for i in 0..half_len {
                    let rollback = length - i - 1;
                    let coeff = _mm_set1_epi32(*v_prepared.get_unchecked(i));
                    let v_source0 =
                        _mm_loadu_si64(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                    let v_source1 = _mm_loadu_si64(
                        shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                    );
                    k0 = _mm_add_epi16(
                        k0,
                        _mm_maddubs_epi16(_mm_unpacklo_epi8(v_source0, v_source1), coeff),
                    );
                }

                k0 = _mm_srai_epi16::<7>(k0);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm_storeu_si64(
                    dst_ptr0 as *mut _,
                    _mm_packus_epi16(k0, _mm_setzero_si128()),
                );
                m_cx += 8;
            }

            while m_cx + 4 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source =
                    _mm_loadu_si32(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
                let mut k0 = _mm_add_epi16(
                    rnd,
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(source, _mm_setzero_si128()), coeff),
                );

                for i in 0..half_len {
                    let rollback = length - i - 1;
                    let coeff = _mm_set1_epi32(*v_prepared.get_unchecked(i));
                    let v_source0 =
                        _mm_loadu_si32(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
                    let v_source1 = _mm_loadu_si32(
                        shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                    );
                    k0 = _mm_add_epi16(
                        k0,
                        _mm_maddubs_epi16(_mm_unpacklo_epi8(v_source0, v_source1), coeff),
                    );
                }

                k0 = _mm_srai_epi16::<7>(k0);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm_storeu_si32(
                    dst_ptr0 as *mut _,
                    _mm_packus_epi16(k0, _mm_setzero_si128()),
                );
                m_cx += 4;
            }

            let coeff = *scanned_kernel.get_unchecked(half_len);

            for zx in m_cx..max_width {
                let x = zx - s_half;

                let shifted_src = src.get_unchecked(x..);
                let mut k0 = *shifted_src.get_unchecked(half_len * N) as u16 * coeff as u16;

                for i in 0..half_len {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    let rollback = length - i - 1;

                    k0 += (*shifted_src.get_unchecked(i * N) as u16
                        + *shifted_src.get_unchecked(rollback * N) as u16)
                        * coeff as u16;
                }

                *dst.get_unchecked_mut(zx) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
            }

            f_cx = m_right;

            while f_cx < width {
                for c in 0..N {
                    let mx = f_cx as i64 - s_kernel;
                    let mut k0: u16 = *src.get_unchecked(f_cx * N + c) as u16 * coeff as u16;

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

                        k0 += (src0 as u16 + src1 as u16) * coeff as u16;
                    }

                    *dst.get_unchecked_mut(f_cx * N + c) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
                }
                f_cx += 1;
            }
        }
    }
}
