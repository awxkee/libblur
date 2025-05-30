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
use crate::filter1d::avx::filter_column_complex_u8_i32::v_interleave_i32x8;
use crate::img_size::ImageSize;
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) fn filter_avx_row_complex_u16_i32(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [Complex<i32>],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        filter_row_avx_complex_u16_f32_impl(arena, arena_src, dst, image_size, kernel);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn filter_row_avx_complex_u16_f32_impl(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [Complex<i32>],
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

        let c_re = _mm256_set1_epi32(kernel.get_unchecked(0).re as i32);
        let c_im = _mm256_set1_epi32(kernel.get_unchecked(0).im as i32);

        let rnd = _mm256_set1_epi32(1 << 13);

        while cx + 16 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values0 = _mm256_loadu_si256(shifted_src.as_ptr().cast());

            let a0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(values0));
            let a1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(values0));

            let mut r0 = _mm256_add_epi32(rnd, _mm256_mullo_epi32(a0, c_re));
            let mut r1 = _mm256_add_epi32(rnd, _mm256_mullo_epi32(a1, c_re));

            let mut i0 = _mm256_add_epi32(rnd, _mm256_mullo_epi32(a0, c_im));
            let mut i1 = _mm256_add_epi32(rnd, _mm256_mullo_epi32(a1, c_im));

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm256_set1_epi32(k.re as i32);
                let c_im = _mm256_set1_epi32(k.im as i32);

                let values0 = _mm256_loadu_si256(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let a0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(values0));
                let a1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(values0));

                r0 = _mm256_add_epi32(r0, _mm256_mullo_epi32(a0, c_re));
                r1 = _mm256_add_epi32(r1, _mm256_mullo_epi32(a1, c_re));

                i0 = _mm256_add_epi32(i0, _mm256_mullo_epi32(a0, c_im));
                i1 = _mm256_add_epi32(i1, _mm256_mullo_epi32(a1, c_im));
            }

            let dst0 = dst.get_unchecked_mut(cx..);

            r0 = _mm256_srai_epi32::<14>(r0);
            r1 = _mm256_srai_epi32::<14>(r1);

            i0 = _mm256_srai_epi32::<14>(i0);
            i1 = _mm256_srai_epi32::<14>(i1);

            let (z0, z1) = v_interleave_i32x8(r0, i0);
            let (z2, z3) = v_interleave_i32x8(r1, i1);

            _mm256_storeu_si256(dst0.as_mut_ptr().cast(), z0);
            _mm256_storeu_si256(dst0.get_unchecked_mut(4..).as_mut_ptr().cast(), z1);
            _mm256_storeu_si256(dst0.get_unchecked_mut(8..).as_mut_ptr().cast(), z2);
            _mm256_storeu_si256(dst0.get_unchecked_mut(12..).as_mut_ptr().cast(), z3);
            cx += 16;
        }

        let c_re = _mm256_castsi256_si128(c_re);
        let c_im = _mm256_castsi256_si128(c_im);

        let rnd = _mm256_castsi256_si128(rnd);

        while cx + 8 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = _mm_loadu_si128(shifted_src.as_ptr().cast());

            let a0 = _mm_unpacklo_epi16(values, _mm_setzero_si128());
            let a1 = _mm_unpackhi_epi16(values, _mm_setzero_si128());

            let mut r0 = _mm_add_epi32(rnd, _mm_mullo_epi32(a0, c_re));
            let mut r1 = _mm_add_epi32(rnd, _mm_mullo_epi32(a1, c_re));

            let mut i0 = _mm_add_epi32(rnd, _mm_mullo_epi32(a0, c_im));
            let mut i1 = _mm_add_epi32(rnd, _mm_mullo_epi32(a1, c_im));

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm_set1_epi32(k.re as i32);
                let c_im = _mm_set1_epi32(k.im as i32);

                let values = _mm_loadu_si128(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let a0 = _mm_unpacklo_epi16(values, _mm_setzero_si128());
                let a1 = _mm_unpackhi_epi16(values, _mm_setzero_si128());

                r0 = _mm_add_epi32(r0, _mm_mullo_epi32(a0, c_re));
                r1 = _mm_add_epi32(r1, _mm_mullo_epi32(a1, c_re));

                i0 = _mm_add_epi32(i0, _mm_mullo_epi32(a0, c_im));
                i1 = _mm_add_epi32(i1, _mm_mullo_epi32(a1, c_im));
            }

            let dst0 = dst.get_unchecked_mut(cx..);

            r0 = _mm_srai_epi32::<14>(r0);
            r1 = _mm_srai_epi32::<14>(r1);
            i0 = _mm_srai_epi32::<14>(i0);
            i1 = _mm_srai_epi32::<14>(i1);

            let (z0, z1) = (_mm_unpacklo_epi32(r0, i0), _mm_unpackhi_epi32(r0, i0));
            let (z2, z3) = (_mm_unpacklo_epi32(r1, i1), _mm_unpackhi_epi32(r1, i1));

            _mm_storeu_si128(dst0.as_mut_ptr().cast(), z0);
            _mm_storeu_si128(dst0.get_unchecked_mut(2..).as_mut_ptr().cast(), z1);
            _mm_storeu_si128(dst0.get_unchecked_mut(4..).as_mut_ptr().cast(), z2);
            _mm_storeu_si128(dst0.get_unchecked_mut(6..).as_mut_ptr().cast(), z3);
            cx += 8;
        }

        while cx + 4 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = _mm_loadu_si64(shifted_src.as_ptr().cast());

            let a0 = _mm_unpacklo_epi16(values, _mm_setzero_si128());

            let mut r0 = _mm_add_epi32(rnd, _mm_mullo_epi32(a0, c_re));
            let mut i0 = _mm_add_epi32(rnd, _mm_mullo_epi32(a0, c_im));

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm_set1_epi32(k.re as i32);
                let c_im = _mm_set1_epi32(k.im as i32);

                let values = _mm_loadu_si64(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let a0 = _mm_unpacklo_epi16(values, _mm_setzero_si128());

                r0 = _mm_add_epi32(r0, _mm_mullo_epi32(a0, c_re));
                i0 = _mm_add_epi32(i0, _mm_mullo_epi32(a0, c_im));
            }

            r0 = _mm_srai_epi32::<14>(r0);
            i0 = _mm_srai_epi32::<14>(i0);

            let dst0 = dst.get_unchecked_mut(cx..);

            let (z0, z1) = (_mm_unpacklo_epi32(r0, i0), _mm_unpackhi_epi32(r0, i0));

            _mm_storeu_si128(dst0.as_mut_ptr().cast(), z0);
            _mm_storeu_si128(dst0.get_unchecked_mut(2..).as_mut_ptr().cast(), z1);
            cx += 4;
        }

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);
            let a0 = _mm_set1_epi32(*shifted_src.get_unchecked(0) as i32);
            let mut r0 = _mm_add_epi32(rnd, _mm_mullo_epi32(a0, c_re));
            let mut i0 = _mm_add_epi32(rnd, _mm_mullo_epi32(a0, c_im));

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm_set1_epi32(k.re as i32);
                let c_im = _mm_set1_epi32(k.im as i32);

                let a0 = _mm_set1_epi32(*shifted_src.get_unchecked(i * arena.components) as i32);

                r0 = _mm_add_epi32(r0, _mm_mullo_epi32(a0, c_re));
                i0 = _mm_add_epi32(i0, _mm_mullo_epi32(a0, c_im));
            }

            r0 = _mm_srai_epi32::<14>(r0);
            i0 = _mm_srai_epi32::<14>(i0);

            _mm_storeu_si32(dst.get_unchecked_mut(x..).as_mut_ptr().cast(), r0);
            _mm_storeu_si32(
                (dst.get_unchecked_mut(x..).as_mut_ptr() as *mut i32)
                    .add(1)
                    .cast(),
                i0,
            );
        }
    }
}
