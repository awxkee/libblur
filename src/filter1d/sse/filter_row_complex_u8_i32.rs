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
use crate::img_size::ImageSize;
use num_complex::Complex;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn filter_sse_row_complex_u8_i32(
    arena: Arena,
    arena_src: &[u8],
    dst: &mut [Complex<i16>],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        filter_row_complex_u8_i32_impl(arena, arena_src, dst, image_size, kernel);
    }
}

#[inline(always)]
pub(crate) unsafe fn widen_mul(a: __m128i, b: __m128i) -> (__m128i, __m128i) {
    let lo = _mm_mullo_epi16(a, b);
    let hi = _mm_mulhi_epi16(a, b);
    (_mm_unpacklo_epi16(lo, hi), _mm_unpackhi_epi16(lo, hi))
}

#[inline(always)]
pub(crate) unsafe fn widen_mul_lo(a: __m128i, b: __m128i) -> __m128i {
    let lo = _mm_mullo_epi16(a, b);
    let hi = _mm_mulhi_epi16(a, b);
    _mm_unpacklo_epi16(lo, hi)
}

#[target_feature(enable = "sse4.1")]
unsafe fn filter_row_complex_u8_i32_impl(
    arena: Arena,
    arena_src: &[u8],
    dst: &mut [Complex<i16>],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        let width = image_size.width;

        let src = arena_src;

        let local_src = src;

        let length = kernel.len();

        let mut cx = 0usize;

        let max_width = width * arena.components;

        let c_re = _mm_set1_epi16(kernel.get_unchecked(0).re);
        let c_im = _mm_set1_epi16(kernel.get_unchecked(0).im);

        let rnd = _mm_set1_epi32(1 << 14);

        while cx + 8 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = _mm_loadu_si64(shifted_src.as_ptr().cast());

            let lo = _mm_unpacklo_epi8(values, _mm_setzero_si128());

            let (mut r0, mut r1) = widen_mul(lo, c_re);
            let (mut i0, mut i1) = widen_mul(lo, c_im);

            r0 = _mm_add_epi32(rnd, r0);
            r1 = _mm_add_epi32(rnd, r1);
            i0 = _mm_add_epi32(rnd, i0);
            i1 = _mm_add_epi32(rnd, i1);

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm_set1_epi16(k.re);
                let c_im = _mm_set1_epi16(k.im);

                let values = _mm_loadu_si64(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let lo = _mm_unpacklo_epi8(values, _mm_setzero_si128());

                let (zr0, zr1) = widen_mul(lo, c_re);
                let (zi0, zi1) = widen_mul(lo, c_im);

                r0 = _mm_add_epi32(r0, zr0);
                r1 = _mm_add_epi32(r1, zr1);

                i0 = _mm_add_epi32(i0, zi0);
                i1 = _mm_add_epi32(i1, zi1);
            }

            r0 = _mm_srai_epi32::<15>(r0);
            r1 = _mm_srai_epi32::<15>(r1);
            i0 = _mm_srai_epi32::<15>(i0);
            i1 = _mm_srai_epi32::<15>(i1);

            let dst0 = dst.get_unchecked_mut(cx..);

            let packed_r0 = _mm_packs_epi32(r0, r1);
            let packed_i0 = _mm_packs_epi32(i0, i1);

            let (z0, z1) = (
                _mm_unpacklo_epi16(packed_r0, packed_i0),
                _mm_unpackhi_epi16(packed_r0, packed_i0),
            );

            _mm_storeu_si128(dst0.as_mut_ptr().cast(), z0);
            _mm_storeu_si128(dst0.get_unchecked_mut(4..).as_mut_ptr().cast(), z1);
            cx += 8;
        }

        while cx + 4 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = _mm_loadu_si32(shifted_src.as_ptr().cast());

            let lo = _mm_unpacklo_epi8(values, _mm_setzero_si128());

            let mut r0 = _mm_add_epi32(rnd, widen_mul_lo(lo, c_re));
            let mut i0 = _mm_add_epi32(rnd, widen_mul_lo(lo, c_im));

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm_set1_epi16(k.re);
                let c_im = _mm_set1_epi16(k.im);

                let values = _mm_loadu_si32(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let lo = _mm_unpacklo_epi8(values, _mm_setzero_si128());

                r0 = _mm_add_epi32(r0, widen_mul_lo(lo, c_re));
                i0 = _mm_add_epi32(i0, widen_mul_lo(lo, c_im));
            }

            r0 = _mm_srai_epi32::<15>(r0);
            i0 = _mm_srai_epi32::<15>(i0);

            let dst0 = dst.get_unchecked_mut(cx..);

            let packed_r0 = _mm_packs_epi32(r0, _mm_setzero_si128());
            let packed_i0 = _mm_packs_epi32(i0, _mm_setzero_si128());

            let z0 = _mm_unpacklo_epi16(packed_r0, packed_i0);

            _mm_storeu_si128(dst0.as_mut_ptr().cast(), z0);
            cx += 4;
        }

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);
            let a0 = _mm_set1_epi32(*shifted_src.get_unchecked(0) as i32);
            let mut r0 = _mm_add_epi32(rnd, _mm_madd_epi16(a0, c_re));
            let mut i0 = _mm_add_epi32(rnd, _mm_madd_epi16(a0, c_im));

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm_set1_epi32(k.re as i32);
                let c_im = _mm_set1_epi32(k.im as i32);

                let a0 = _mm_set1_epi32(*shifted_src.get_unchecked(i * arena.components) as i32);

                r0 = _mm_add_epi32(r0, _mm_madd_epi16(a0, c_re));
                i0 = _mm_add_epi32(i0, _mm_madd_epi16(a0, c_im));
            }

            r0 = _mm_srai_epi32::<15>(r0);
            i0 = _mm_srai_epi32::<15>(i0);

            let p_r = _mm_packs_epi32(r0, r0);
            let p_i = _mm_packs_epi32(i0, i0);

            _mm_storeu_si16(dst.get_unchecked_mut(x..).as_mut_ptr().cast(), p_r);
            _mm_storeu_si16(
                (dst.get_unchecked_mut(x..).as_mut_ptr() as *mut i16)
                    .add(1)
                    .cast(),
                p_i,
            );
        }
    }
}
