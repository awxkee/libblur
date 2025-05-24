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
use crate::edge_mode::clamp_edge;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::row_handler_small_approx::{RowsHolder, RowsHolderMut};
use crate::img_size::ImageSize;
use crate::BorderHandle;
use std::arch::x86_64::*;

pub(crate) fn filter_row_avx512_symm_u8_uq0_7_k3<const N: usize>(
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
    #[target_feature(enable = "avx512bw")]
    unsafe fn pass(
        &self,
        edge_mode: BorderHandle,
        m_src: &RowsHolder<u8>,
        m_dst: &mut RowsHolderMut<u8>,
        image_size: ImageSize,
        scanned_kernel: &[i8],
    ) {
        let width = image_size.width;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let min_left = half_len.min(width);
        let s_kernel = half_len as i64;

        let c0 = scanned_kernel[0];
        let c1 = scanned_kernel[1];

        let vc0 = _mm512_set1_epi8(scanned_kernel[0]);
        let vc1 = _mm512_set1_epi8(scanned_kernel[1]);

        let rnd = _mm512_set1_epi16(1i16 << 6);

        for (src, dst) in m_src.holder.iter().zip(m_dst.holder.iter_mut()) {
            let dst = &mut **dst;
            let mut f_cx = 0usize;

            while f_cx < min_left {
                let mx = f_cx as i64 - s_kernel;
                let e0 = clamp_edge!(edge_mode.edge_mode, mx, 0, width as i64);
                let e1 = clamp_edge!(edge_mode.edge_mode, 2i64 + mx, 0, width as i64);

                for c in 0..N {
                    let mut k0: u16 = *src.get_unchecked(f_cx * N + c) as u16 * c1 as u16;

                    let src0 = *src.get_unchecked(e0 * N + c);
                    let src1 = *src.get_unchecked(e1 * N + c);

                    k0 += (src0 as u16 + src1 as u16) * c0 as u16;

                    *dst.get_unchecked_mut(f_cx * N + c) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
                }
                f_cx += 1;
            }

            let mut m_cx = f_cx * N;

            let s_half = half_len * N;
            let m_right = width.saturating_sub(half_len);
            let max_width = m_right * N;

            while m_cx + 64 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let v_source0 =
                    _mm512_loadu_si512(shifted_src.get_unchecked(0..).as_ptr() as *const _);
                let source =
                    _mm512_loadu_si512(shifted_src.get_unchecked(N..).as_ptr() as *const _);
                let v_source4 =
                    _mm512_loadu_si512(shifted_src.get_unchecked((2 * N)..).as_ptr() as *const _);

                let mut k0 = _mm512_add_epi16(
                    rnd,
                    _mm512_maddubs_epi16(_mm512_unpacklo_epi8(source, _mm512_setzero_si512()), vc1),
                );
                let mut k1 = _mm512_add_epi16(
                    rnd,
                    _mm512_maddubs_epi16(_mm512_unpackhi_epi8(source, _mm512_setzero_si512()), vc1),
                );

                k0 = _mm512_add_epi16(
                    k0,
                    _mm512_maddubs_epi16(_mm512_unpacklo_epi8(v_source0, v_source4), vc0),
                );
                k1 = _mm512_add_epi16(
                    k1,
                    _mm512_maddubs_epi16(_mm512_unpackhi_epi8(v_source0, v_source4), vc0),
                );

                k0 = _mm512_srai_epi16::<7>(k0);
                k1 = _mm512_srai_epi16::<7>(k1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm512_storeu_si512(dst_ptr0 as *mut _, _mm512_packus_epi16(k0, k1));
                m_cx += 64;
            }

            while m_cx + 32 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let v_source0 =
                    _mm256_loadu_si256(shifted_src.get_unchecked(0..).as_ptr() as *const _);
                let source =
                    _mm256_loadu_si256(shifted_src.get_unchecked(N..).as_ptr() as *const _);
                let v_source4 =
                    _mm256_loadu_si256(shifted_src.get_unchecked((2 * N)..).as_ptr() as *const _);

                let mut k0 = _mm256_add_epi16(
                    _mm512_castsi512_si256(rnd),
                    _mm256_maddubs_epi16(
                        _mm256_unpacklo_epi8(source, _mm256_setzero_si256()),
                        _mm512_castsi512_si256(vc1),
                    ),
                );
                let mut k1 = _mm256_add_epi16(
                    _mm512_castsi512_si256(rnd),
                    _mm256_maddubs_epi16(
                        _mm256_unpackhi_epi8(source, _mm256_setzero_si256()),
                        _mm512_castsi512_si256(vc1),
                    ),
                );

                k0 = _mm256_add_epi16(
                    k0,
                    _mm256_maddubs_epi16(
                        _mm256_unpacklo_epi8(v_source0, v_source4),
                        _mm512_castsi512_si256(vc0),
                    ),
                );
                k1 = _mm256_add_epi16(
                    k1,
                    _mm256_maddubs_epi16(
                        _mm256_unpackhi_epi8(v_source0, v_source4),
                        _mm512_castsi512_si256(vc0),
                    ),
                );

                k0 = _mm256_srai_epi16::<7>(k0);
                k1 = _mm256_srai_epi16::<7>(k1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm256_storeu_si256(dst_ptr0 as *mut _, _mm256_packus_epi16(k0, k1));
                m_cx += 32;
            }

            while m_cx + 16 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let v_source0 =
                    _mm_loadu_si128(shifted_src.get_unchecked(0..).as_ptr() as *const _);
                let source = _mm_loadu_si128(shifted_src.get_unchecked(N..).as_ptr() as *const _);
                let v_source4 =
                    _mm_loadu_si128(shifted_src.get_unchecked((2 * N)..).as_ptr() as *const _);

                let mut k0 = _mm_add_epi16(
                    _mm512_castsi512_si128(rnd),
                    _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(source, _mm_setzero_si128()),
                        _mm512_castsi512_si128(vc1),
                    ),
                );
                let mut k1 = _mm_add_epi16(
                    _mm512_castsi512_si128(rnd),
                    _mm_maddubs_epi16(
                        _mm_unpackhi_epi8(source, _mm_setzero_si128()),
                        _mm512_castsi512_si128(vc1),
                    ),
                );

                k0 = _mm_add_epi16(
                    k0,
                    _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(v_source0, v_source4),
                        _mm512_castsi512_si128(vc0),
                    ),
                );
                k1 = _mm_add_epi16(
                    k1,
                    _mm_maddubs_epi16(
                        _mm_unpackhi_epi8(v_source0, v_source4),
                        _mm512_castsi512_si128(vc0),
                    ),
                );

                k0 = _mm_srai_epi16::<7>(k0);
                k1 = _mm_srai_epi16::<7>(k1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm_storeu_si128(dst_ptr0 as *mut _, _mm_packus_epi16(k0, k1));
                m_cx += 16;
            }

            while m_cx + 8 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source = _mm_loadu_si64(shifted_src.get_unchecked(N..).as_ptr() as *const _);
                let mut k0 = _mm_add_epi16(
                    _mm512_castsi512_si128(rnd),
                    _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(source, _mm_setzero_si128()),
                        _mm512_castsi512_si128(vc1),
                    ),
                );

                let v_source0 = _mm_loadu_si64(shifted_src.get_unchecked(..).as_ptr() as *const _);
                let v_source1 =
                    _mm_loadu_si64(shifted_src.get_unchecked((2 * N)..).as_ptr() as *const _);

                k0 = _mm_add_epi16(
                    k0,
                    _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(v_source0, v_source1),
                        _mm512_castsi512_si128(vc0),
                    ),
                );

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

                let source = _mm_loadu_si32(shifted_src.get_unchecked(N..).as_ptr() as *const _);
                let mut k0 = _mm_add_epi16(
                    _mm512_castsi512_si128(rnd),
                    _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(source, _mm_setzero_si128()),
                        _mm512_castsi512_si128(vc1),
                    ),
                );

                let v_source0 = _mm_loadu_si32(shifted_src.get_unchecked(..).as_ptr() as *const _);
                let v_source1 =
                    _mm_loadu_si32(shifted_src.get_unchecked((2 * N)..).as_ptr() as *const _);

                k0 = _mm_add_epi16(
                    k0,
                    _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(v_source0, v_source1),
                        _mm512_castsi512_si128(vc0),
                    ),
                );

                k0 = _mm_srai_epi16::<7>(k0);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                _mm_storeu_si32(
                    dst_ptr0 as *mut _,
                    _mm_packus_epi16(k0, _mm_setzero_si128()),
                );
                m_cx += 4;
            }

            for zx in m_cx..max_width {
                let x = zx - s_half;

                let shifted_src = src.get_unchecked(x..);
                let mut k0: u16 = *shifted_src.get_unchecked(N) as u16 * c1 as u16;

                k0 += (*shifted_src.get_unchecked(0) as u16
                    + *shifted_src.get_unchecked(2 * N) as u16)
                    * c0 as u16;

                *dst.get_unchecked_mut(zx) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
            }

            f_cx = m_right;

            while f_cx < width {
                let mx = f_cx as i64 - s_kernel;
                let e0 = clamp_edge!(edge_mode.edge_mode, mx, 0, width as i64);
                let e1 = clamp_edge!(edge_mode.edge_mode, 2i64 + mx, 0, width as i64);

                for c in 0..N {
                    let mut k0: u16 = *src.get_unchecked(f_cx * N + c) as u16 * c1 as u16;

                    let src0 = *src.get_unchecked(e0 * N + c);
                    let src1 = *src.get_unchecked(e1 * N + c);

                    k0 += (src0 as u16 + src1 as u16) * c0 as u16;

                    *dst.get_unchecked_mut(f_cx * N + c) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
                }
                f_cx += 1;
            }
        }
    }
}
